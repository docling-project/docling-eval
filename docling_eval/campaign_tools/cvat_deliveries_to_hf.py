from __future__ import annotations

import json
import logging
import shutil
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Sequence

import typer

from docling_eval.datamodels.dataset_record import DatasetRecord, FieldType
from docling_eval.dataset_builders.file_dataset_builder import FileDatasetBuilder

_LOGGER = logging.getLogger(__name__)

app = typer.Typer(
    help=(
        "Aggregate DoclingDocument JSON exports produced by the CVAT delivery "
        "pipeline into HuggingFace-ready parquet datasets, grouped by subset name."
    ),
    no_args_is_help=True,
    add_completion=False,
)


class DeliveryExportKind(str, Enum):
    GROUND_TRUTH = "ground_truth_json"
    PREDICTIONS = "predictions_json"

    def folder_name(self) -> str:
        return self.value


@dataclass(frozen=True)
class ConfigEntry:
    name: str
    split: str
    path_pattern: str


@dataclass(frozen=True)
class SubsetBuildStats:
    name: str
    submissions: int
    record_count: int

    def as_config(self, dataset_dir_name: str, split: str) -> ConfigEntry:
        pattern = f"{self.name}/{dataset_dir_name}/{split}/shard_*.parquet"
        return ConfigEntry(name=self.name, split=split, path_pattern=pattern)


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
    deliveries_root: Path, export_kind: DeliveryExportKind
) -> Dict[str, List[Path]]:
    subset_dirs: Dict[str, List[Path]] = defaultdict(list)

    for submission_dir in sorted(
        p for p in deliveries_root.glob(SUBMISSION_DIR_GLOB) if p.is_dir()
    ):
        for subset_dir in sorted(p for p in submission_dir.iterdir() if p.is_dir()):
            candidate = subset_dir / export_kind.folder_name()
            if candidate.is_dir():
                subset_dirs[subset_dir.name].append(candidate)
            else:
                _LOGGER.debug(
                    "Skipping %s (no %s)", candidate, export_kind.folder_name()
                )

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
    source_dirs: Sequence[Path],
) -> int:
    file_index = 0

    for dir_idx, source_dir in enumerate(source_dirs):
        json_files = sorted(p for p in source_dir.glob("*.json") if p.is_file())
        if not json_files:
            _LOGGER.warning("No JSON files found under %s", source_dir)
            continue

        for json_file in json_files:
            link_name = f"{dir_idx:03d}_{file_index:06d}_{json_file.name}"
            destination = staging_dir / link_name
            link_file(json_file, destination)
            file_index += 1

    return file_index


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


def build_subset_dataset(
    subset_name: str,
    subset_dirs: Sequence[Path],
    staging_root: Path,
    output_root: Path,
    dataset_dir_name: str,
    split: str,
    chunk_size: int,
    export_kind: DeliveryExportKind,
    force: bool,
) -> SubsetBuildStats | None:
    target_root = output_root / subset_name / dataset_dir_name
    if target_root.exists():
        if not force:
            raise RuntimeError(
                f"{target_root} already exists. Use --force to overwrite."
            )
        _LOGGER.warning(
            "Removing existing dataset directory at %s due to --force flag", target_root
        )
        shutil.rmtree(target_root)

    ensure_clean_dir(staging_root / subset_name)
    staging_dir = staging_root / subset_name
    processed = populate_staging_dir(staging_dir, subset_dirs)
    if processed == 0:
        _LOGGER.warning("Subset %s had no JSON payloads. Skipping.", subset_name)
        shutil.rmtree(staging_dir)
        return None

    builder = FileDatasetBuilder(
        name=f"{subset_name}-{export_kind.value}",
        dataset_source=staging_dir,
        target=target_root,
        split=split,
        file_extensions=["json"],
    )
    builder.save_to_disk(chunk_size=chunk_size)
    shutil.rmtree(staging_dir)

    record_count = read_num_rows(target_root)
    submission_count = len(subset_dirs)

    return SubsetBuildStats(
        name=subset_name,
        submissions=submission_count,
        record_count=record_count if record_count else processed,
    )


def render_configs_block(configs: Sequence[ConfigEntry]) -> List[str]:
    lines: List[str] = ["configs:"]
    for entry in configs:
        lines.append(f"- config_name: {entry.name}")
        lines.append("  data_files:")
        lines.append(f"    - split: {entry.split}")
        lines.append(f"      path: {entry.path_pattern}")
    return lines


def render_dataset_info_block(configs: Sequence[ConfigEntry]) -> List[str]:
    lines: List[str] = ["dataset_info:"]
    feature_rows = iter_dataset_features()
    for entry in configs:
        lines.append(f"- config_name: {entry.name}")
        lines.append("  features:")
        for feature_name, attr, value in feature_rows:
            lines.append(f"  - name: {feature_name}")
            lines.append(f"    {attr}: {value}")
        lines.append("")
    return lines


def write_readme(
    output_root: Path,
    configs: Sequence[ConfigEntry],
    stats: Sequence[SubsetBuildStats],
    license_name: str,
    export_kind: DeliveryExportKind,
) -> None:
    lines: List[str] = ["---"]
    lines.extend(render_configs_block(configs))
    lines.extend(render_dataset_info_block(configs))
    lines.append(f"license: {license_name}")
    lines.append("---")
    lines.append("")
    lines.append("# CVAT Delivery Aggregation")
    lines.append(
        "This repository consolidates DoclingDocument exports produced by the "
        "CVAT delivery pipeline into HuggingFace-ready parquet datasets."
    )
    lines.append("")
    lines.append(f"- Source payloads: `{export_kind.folder_name()}`")
    lines.append("- Builder: `FileDatasetBuilder` with JSON inputs")
    lines.append("")
    lines.append("## Subsets")
    for subset in stats:
        lines.append(
            f"- `{subset.name}`: {subset.record_count} documents from "
            f"{subset.submissions} submissions"
        )
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
        help="Name of the dataset directory created inside each subset folder.",
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
        help="Overwrite any existing subset dataset directories and staging data.",
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

    subset_sources = discover_subset_sources(deliveries_root, export_kind)
    if not subset_sources:
        typer.echo(
            f"No subset directories with {export_kind.folder_name()} found under "
            f"{deliveries_root}",
            err=True,
        )
        raise typer.Exit(code=1)

    built_stats: List[SubsetBuildStats] = []
    for subset_name in sorted(subset_sources.keys()):
        stats = build_subset_dataset(
            subset_name=subset_name,
            subset_dirs=sorted(subset_sources[subset_name]),
            staging_root=staging_root,
            output_root=output_dir,
            dataset_dir_name=dataset_dir_name,
            split=split,
            chunk_size=chunk_size,
            export_kind=export_kind,
            force=force,
        )
        if stats:
            built_stats.append(stats)

    shutil.rmtree(staging_root, ignore_errors=True)

    if not built_stats:
        typer.echo("No datasets were produced.", err=True)
        raise typer.Exit(code=1)

    configs = [stat.as_config(dataset_dir_name, split) for stat in built_stats]
    write_readme(
        output_root=output_dir,
        configs=configs,
        stats=built_stats,
        license_name=license_name,
        export_kind=export_kind,
    )


if __name__ == "__main__":
    app()
