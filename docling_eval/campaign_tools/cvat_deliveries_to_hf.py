from __future__ import annotations

import json
import logging
import shutil
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import typer

from docling_eval.datamodels.dataset_record import DatasetRecord, FieldType
from docling_eval.dataset_builders.file_dataset_builder import FileDatasetBuilder

_LOGGER = logging.getLogger(__name__)

app = typer.Typer(
    help=(
        "Aggregate DoclingDocument JSON exports produced by the CVAT delivery "
        "pipeline into a single HuggingFace-ready parquet dataset with subset tags."
    ),
    no_args_is_help=True,
    add_completion=False,
)


class DeliveryExportKind(str, Enum):
    GROUND_TRUTH = "ground_truth"
    PREDICTIONS = "predictions"

    def folder_name(self) -> str:
        """Return the default directory name for this export kind."""
        if self == DeliveryExportKind.GROUND_TRUTH:
            return "ground_truth_json"
        return "predictions_json"


@dataclass(frozen=True)
class ConfigEntry:
    name: str
    split: str
    path_pattern: str


@dataclass(frozen=True)
class CombinedBuildStats:
    record_count: int
    submission_count: int
    subsets: List[str]

    def as_config(self, dataset_dir_name: str, split: str) -> ConfigEntry:
        pattern = f"{dataset_dir_name}/{split}/shard_*.parquet"
        return ConfigEntry(name="default", split=split, path_pattern=pattern)


FEATURE_FIELD_RENDER: Dict[FieldType, tuple[str, str]] = {
    FieldType.STRING: ("dtype", "string"),
    FieldType.BINARY: ("dtype", "binary"),
    FieldType.IMAGE_LIST: ("sequence", "image"),
    FieldType.STRING_LIST: ("sequence", "string"),
}


def iter_dataset_features() -> List[tuple[str, str, str]]:
    rows: List[tuple[str, str, str]] = []
    for field_name, field_type in DatasetRecord._get_field_definitions().items():
        if field_type not in FEATURE_FIELD_RENDER:
            raise ValueError(f"Unhandled field type {field_type} for {field_name}")
        attr, attr_value = FEATURE_FIELD_RENDER[field_type]
        rows.append((field_name, attr, attr_value))
    return rows


# Align with run_cvat_deliveries_pipeline output layout.
SUBMISSION_DIR_GLOB = "submission-*"


def discover_subset_sources(
    deliveries_root: Path,
    export_kind: DeliveryExportKind,
    custom_dirname: str | None = None,
) -> Dict[str, List[Path]]:
    """
    Discover subset source directories containing JSON exports.

    Args:
        deliveries_root: Root directory containing submission folders
        export_kind: Type of export to discover (ground truth or predictions)
        custom_dirname: Optional custom directory name to use instead of the default

    Returns:
        Dictionary mapping subset names to lists of source directories
    """
    subset_dirs: Dict[str, List[Path]] = defaultdict(list)
    dirname = (
        custom_dirname if custom_dirname is not None else export_kind.folder_name()
    )

    for submission_dir in sorted(
        p for p in deliveries_root.glob(SUBMISSION_DIR_GLOB) if p.is_dir()
    ):
        for subset_dir in sorted(p for p in submission_dir.iterdir() if p.is_dir()):
            candidate = subset_dir / dirname
            if candidate.is_dir():
                subset_dirs[subset_dir.name].append(candidate)
            else:
                _LOGGER.debug("Skipping %s (no %s)", candidate, dirname)

    return subset_dirs


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_file(source: Path, destination: Path) -> None:
    try:
        destination.symlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def populate_staging_dir(
    staging_dir: Path,
    subset_sources: Dict[str, List[Path]],
) -> tuple[int, Dict[str, str]]:
    """
    Populate staging directory with files from all subsets.

    Returns:
        Tuple of (total_file_count, mapping from original filename stem to subset_name)
    """
    file_index = 0
    file_to_subset: Dict[str, str] = {}

    for subset_name in sorted(subset_sources.keys()):
        for source_dir in subset_sources[subset_name]:
            json_files = sorted(p for p in source_dir.glob("*.json") if p.is_file())
            if not json_files:
                _LOGGER.warning("No JSON files found under %s", source_dir)
                continue

            for json_file in json_files:
                # Use original filename to preserve doc_id
                destination = staging_dir / json_file.name
                link_file(json_file, destination)
                # Map the original filename stem to subset name
                original_stem = json_file.stem
                file_to_subset[original_stem] = subset_name
                file_index += 1

    return file_index, file_to_subset


def read_num_rows(dataset_root: Path) -> int:
    infos_path = dataset_root / "dataset_infos.json"
    if not infos_path.exists():
        return 0

    with infos_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    test_info = data.get("test")
    if isinstance(test_info, dict):
        num_rows = test_info.get("num_rows")
        if isinstance(num_rows, int):
            return num_rows
    return 0


def iter_records_with_tags(
    builder: FileDatasetBuilder, file_to_subset: Dict[str, str]
) -> Iterable[DatasetRecord]:
    """
    Iterate over records from FileDatasetBuilder and add subset tags.

    The builder creates records with doc_id from filename.stem (original filename),
    so we match it to our file_to_subset mapping to find the subset name.
    """
    for record in builder.iterate():
        # FileDatasetBuilder sets doc_id to filename.stem (original filename)
        # Match it to our file_to_subset mapping
        subset_name = file_to_subset.get(record.doc_id)
        if subset_name:
            record.tags.append(f"subset:{subset_name}")
        yield record


def build_combined_dataset(
    subset_sources: Dict[str, List[Path]],
    staging_dir: Path,
    output_root: Path,
    dataset_dir_name: str,
    split: str,
    chunk_size: int,
    export_kind: DeliveryExportKind,
    force: bool,
) -> CombinedBuildStats | None:
    """
    Build a single combined dataset from all subsets with subset tags.
    """
    target_root = output_root / dataset_dir_name
    if target_root.exists():
        if not force:
            raise RuntimeError(
                f"{target_root} already exists. Use --force to overwrite."
            )
        _LOGGER.warning(
            "Removing existing dataset directory at %s due to --force flag", target_root
        )
        shutil.rmtree(target_root)

    ensure_clean_dir(staging_dir)
    processed, file_to_subset = populate_staging_dir(staging_dir, subset_sources)
    if processed == 0:
        _LOGGER.warning("No JSON payloads found across all subsets. Skipping.")
        shutil.rmtree(staging_dir)
        return None

    builder = FileDatasetBuilder(
        name=f"combined-{export_kind.value}",
        dataset_source=staging_dir,
        target=target_root,
        split=split,
        file_extensions=["json"],
    )

    # Custom save logic that adds tags
    from docling.utils.utils import chunkify

    from docling_eval.utils.utils import save_shard_to_disk, write_datasets_info

    test_dir = target_root / split
    test_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    chunk_count = 0

    for record_chunk in chunkify(
        iter_records_with_tags(builder, file_to_subset), chunk_size
    ):
        record_list = [r.as_record_dict() for r in record_chunk]
        save_shard_to_disk(
            items=record_list,
            dataset_path=test_dir,
            schema=DatasetRecord.pyarrow_schema(),
            shard_id=chunk_count,
        )
        count += len(record_list)
        chunk_count += 1

    write_datasets_info(
        name=builder.name,
        output_dir=target_root,
        num_train_rows=0,
        num_test_rows=count,
        features=DatasetRecord.features(),
    )

    shutil.rmtree(staging_dir)

    record_count = read_num_rows(target_root)
    submission_count = sum(len(dirs) for dirs in subset_sources.values())
    subset_names = sorted(subset_sources.keys())

    return CombinedBuildStats(
        record_count=record_count if record_count else processed,
        submission_count=submission_count,
        subsets=subset_names,
    )


def render_configs_block(config: ConfigEntry) -> List[str]:
    lines: List[str] = ["configs:"]
    lines.append(f"- config_name: {config.name}")
    lines.append("  data_files:")
    lines.append(f"    - split: {config.split}")
    lines.append(f"      path: {config.path_pattern}")
    return lines


def render_dataset_info_block(config: ConfigEntry) -> List[str]:
    lines: List[str] = ["dataset_info:"]
    feature_rows = iter_dataset_features()
    lines.append(f"- config_name: {config.name}")
    lines.append("  features:")
    for feature_name, attr, value in feature_rows:
        lines.append(f"  - name: {feature_name}")
        lines.append(f"    {attr}: {value}")
    lines.append("")
    return lines


def write_readme(
    output_root: Path,
    config: ConfigEntry,
    stats: CombinedBuildStats,
    license_name: str,
    export_kind: DeliveryExportKind,
    custom_dirname: str | None = None,
) -> None:
    lines: List[str] = ["---"]
    lines.extend(render_configs_block(config))
    lines.extend(render_dataset_info_block(config))
    lines.append(f"license: {license_name}")
    lines.append("---")
    lines.append("")
    lines.append("# CVAT Delivery Aggregation")
    lines.append(
        "This repository consolidates DoclingDocument exports produced by the "
        "CVAT delivery pipeline into a single HuggingFace-ready parquet dataset."
    )
    lines.append("")
    dirname = (
        custom_dirname if custom_dirname is not None else export_kind.folder_name()
    )
    lines.append(f"- Source payloads: `{dirname}`")
    if custom_dirname:
        lines.append(
            f"  - Note: Using custom directory name (default would be `{export_kind.folder_name()}`)"
        )
    lines.append("- Builder: `FileDatasetBuilder` with JSON inputs")
    lines.append(
        "- Subset information: Each record includes a `tags` field with "
        "`subset:<name>` entries indicating the source subset"
    )
    lines.append("")
    lines.append("## Dataset Statistics")
    lines.append(f"- Total records: {stats.record_count}")
    lines.append(f"- Total submissions: {stats.submission_count}")
    lines.append(f"- Subsets included: {', '.join(f'`{s}`' for s in stats.subsets)}")
    readme_path = output_root / "README.md"
    with readme_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


@app.command()
def main(
    deliveries_root: Path = typer.Argument(
        ..., help="Root directory containing submission folders."
    ),
    output_dir: Path = typer.Argument(
        ..., help="Output directory where HuggingFace-ready datasets will be written."
    ),
    export_kind: DeliveryExportKind = typer.Option(
        DeliveryExportKind.GROUND_TRUTH,
        "--export-kind",
        case_sensitive=False,
        help="Whether to aggregate ground truth JSON or prediction JSON exports.",
    ),
    split: str = typer.Option(
        "test", "--split", help="Dataset split label used in the output structure."
    ),
    dataset_dir_name: str = typer.Option(
        "gt_dataset",
        "--dataset-dir-name",
        help="Name of the dataset directory created in the output folder.",
    ),
    chunk_size: int = typer.Option(
        80,
        "--chunk-size",
        help="Number of records per parquet shard.",
    ),
    license_name: str = typer.Option(
        "apache-2.0",
        "--license",
        help="License string embedded into the README preamble.",
    ),
    force: bool = typer.Option(
        False,
        "--force/--no-force",
        help="Overwrite any existing dataset directory and staging data.",
    ),
    gt_json_dirname: str | None = typer.Option(
        None,
        "--gt-json-dirname",
        help=(
            "Custom directory name for ground truth JSON exports "
            "(default: ground_truth_json). Only used when --export-kind is ground_truth."
        ),
    ),
    pred_json_dirname: str | None = typer.Option(
        None,
        "--pred-json-dirname",
        help=(
            "Custom directory name for prediction JSON exports "
            "(default: predictions_json). Only used when --export-kind is predictions."
        ),
    ),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    deliveries_root = deliveries_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    staging_root = output_dir / "_staging"

    if not deliveries_root.exists():
        raise typer.BadParameter(f"{deliveries_root} does not exist.")

    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_clean_dir(staging_root)

    # Select the appropriate custom directory name based on export kind
    if export_kind == DeliveryExportKind.GROUND_TRUTH:
        custom_dirname = gt_json_dirname
    else:
        custom_dirname = pred_json_dirname

    dirname = (
        custom_dirname if custom_dirname is not None else export_kind.folder_name()
    )
    subset_sources = discover_subset_sources(
        deliveries_root, export_kind, custom_dirname
    )
    if not subset_sources:
        typer.echo(
            f"No subset directories with {dirname} found under {deliveries_root}",
            err=True,
        )
        raise typer.Exit(code=1)

    # Sort subset sources for deterministic ordering
    sorted_subset_sources = {k: sorted(v) for k, v in sorted(subset_sources.items())}

    staging_dir = staging_root / "combined"
    stats = build_combined_dataset(
        subset_sources=sorted_subset_sources,
        staging_dir=staging_dir,
        output_root=output_dir,
        dataset_dir_name=dataset_dir_name,
        split=split,
        chunk_size=chunk_size,
        export_kind=export_kind,
        force=force,
    )

    shutil.rmtree(staging_root, ignore_errors=True)

    if not stats:
        typer.echo("No datasets were produced.", err=True)
        raise typer.Exit(code=1)

    config = stats.as_config(dataset_dir_name, split)
    write_readme(
        output_root=output_dir,
        config=config,
        stats=stats,
        license_name=license_name,
        export_kind=export_kind,
        custom_dirname=custom_dirname,
    )


if __name__ == "__main__":
    app()
