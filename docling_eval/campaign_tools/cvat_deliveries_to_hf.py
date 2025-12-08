from __future__ import annotations

import json
import logging
import mimetypes
import shutil
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatch
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

import typer
from docling_core.types.io import DocumentStream

from docling_eval.datamodels.cvat_types import AnnotationOverview
from docling_eval.datamodels.dataset_record import DatasetRecord, FieldType
from docling_eval.dataset_builders.file_dataset_builder import FileDatasetBuilder
from docling_eval.utils.utils import get_binhash

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

    def as_config(
        self, dataset_dir_name: str, split: str, config_name: str | None = None
    ) -> ConfigEntry:
        name = config_name if config_name is not None else "default"
        pattern = f"{dataset_dir_name}/{split}/shard_*.parquet"
        return ConfigEntry(name=name, split=split, path_pattern=pattern)


@dataclass(frozen=True)
class DocumentAsset:
    path: Path
    mime_type: str
    doc_hash: str | None


@dataclass(frozen=True)
class SplitRule:
    pattern: str
    split: str


@dataclass(frozen=True)
class BuildResult:
    config: ConfigEntry
    stats: CombinedBuildStats


def _find_overview_path(assets_root: Path) -> Path | None:
    for candidate_name in ("cvat_annotation_overview.json", "cvat_overview.json"):
        candidate_path = assets_root / candidate_name
        if candidate_path.exists():
            return candidate_path
    return None


def _load_subset_assets(
    subset_name: str,
    datasets_root: Path,
    assets_dirname: str,
    *,
    strict: bool,
) -> Dict[str, DocumentAsset]:
    assets_root = datasets_root / subset_name / assets_dirname
    if not assets_root.exists():
        message = (
            f"Assets root {assets_root} for subset {subset_name} does not exist; "
            "provide a correct --datasets-root or omit it."
        )
        if strict:
            raise FileNotFoundError(message)
        _LOGGER.warning(message)
        return {}

    overview_path = _find_overview_path(assets_root)
    if overview_path is None:
        message = (
            "No cvat_annotation_overview.json or cvat_overview.json found "
            f"for subset {subset_name} under {assets_root}"
        )
        if strict:
            raise FileNotFoundError(message)
        _LOGGER.warning(message)
        return {}

    overview = AnnotationOverview.load_from_json(overview_path)
    assets: Dict[str, DocumentAsset] = {}
    for annotated_doc in overview.doc_annotations:
        asset_path = annotated_doc.bin_file
        if not asset_path.is_absolute():
            asset_path = (assets_root / asset_path).resolve()

        if not asset_path.exists():
            message = (
                f"Binary file {asset_path} referenced in {overview_path} "
                f"is missing for subset {subset_name}"
            )
            if strict:
                raise FileNotFoundError(message)
            _LOGGER.warning(message)
            continue

        assets[annotated_doc.doc_name] = DocumentAsset(
            path=asset_path,
            mime_type=annotated_doc.mime_type,
            doc_hash=annotated_doc.doc_hash if annotated_doc.doc_hash else None,
        )
    return assets


def _build_assets_index(
    subset_names: Sequence[str],
    datasets_root: Path | None,
    assets_dirname: str,
    *,
    strict: bool,
) -> Dict[str, Dict[str, DocumentAsset]]:
    if datasets_root is None:
        return {}

    assets_index: Dict[str, Dict[str, DocumentAsset]] = {}
    for subset_name in subset_names:
        subset_assets = _load_subset_assets(
            subset_name,
            datasets_root,
            assets_dirname,
            strict=strict,
        )
        if subset_assets:
            assets_index[subset_name] = subset_assets
        elif strict:
            raise ValueError(
                f"No assets discovered for subset {subset_name} under {datasets_root}"
            )

    return assets_index


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
    allowed_submissions: Set[str] | None = None,
) -> Dict[str, List[Path]]:
    """
    Discover subset source directories containing JSON exports.

    Args:
        deliveries_root: Root directory containing submission folders
        export_kind: Type of export to discover (ground truth or predictions)
        custom_dirname: Optional custom directory name to use instead of the default
        allowed_submissions: Optional set of submission directory names to include

    Returns:
        Dictionary mapping subset names to lists of source directories
    """
    subset_dirs: Dict[str, List[Path]] = defaultdict(list)
    dirname = (
        custom_dirname if custom_dirname is not None else export_kind.folder_name()
    )

    discovered_submissions: Set[str] = set()

    for submission_dir in sorted(
        p
        for p in deliveries_root.glob(SUBMISSION_DIR_GLOB)
        if p.is_dir() and (allowed_submissions is None or p.name in allowed_submissions)
    ):
        discovered_submissions.add(submission_dir.name)
        for subset_dir in sorted(p for p in submission_dir.iterdir() if p.is_dir()):
            candidate = subset_dir / dirname
            if candidate.is_dir():
                subset_dirs[subset_dir.name].append(candidate)
            else:
                _LOGGER.debug("Skipping %s (no %s)", candidate, dirname)

    if allowed_submissions is not None:
        missing = allowed_submissions.difference(discovered_submissions)
        if missing:
            _LOGGER.warning(
                "Requested submissions not found under %s: %s",
                deliveries_root,
                ", ".join(sorted(missing)),
            )

    return subset_dirs


def _parse_subset_split_rules(rules: List[str]) -> List[SplitRule]:
    parsed_rules: List[SplitRule] = []
    for rule in rules:
        if "=" not in rule:
            raise typer.BadParameter(
                f"Invalid --subset-split value '{rule}'. Expected format pattern=split."
            )
        pattern, split = rule.split("=", maxsplit=1)
        if not pattern or not split:
            raise typer.BadParameter(
                f"Invalid --subset-split value '{rule}'. Both pattern and split are required."
            )
        parsed_rules.append(SplitRule(pattern=pattern, split=split))
    return parsed_rules


def _route_subsets_to_splits(
    subset_sources: Dict[str, List[Path]],
    rules: List[SplitRule],
    default_split: str,
    *,
    fail_on_unmatched: bool,
) -> Dict[str, Dict[str, List[Path]]]:
    split_map: Dict[str, Dict[str, List[Path]]] = defaultdict(dict)

    for subset_name, sources in subset_sources.items():
        matched_split = None
        for rule in rules:
            if fnmatch(subset_name, rule.pattern):
                matched_split = rule.split
                break

        if matched_split is None:
            if fail_on_unmatched:
                raise RuntimeError(
                    f"Subset '{subset_name}' did not match any --subset-split rule "
                    "and --fail-on-unmatched was provided."
                )
            matched_split = default_split

        split_map[matched_split][subset_name] = sources

    return split_map


def _dataset_dir_for_split(base_name: str, split: str, multi_split: bool) -> str:
    if not multi_split:
        return base_name
    return f"{base_name}-{split}"


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
    builder: FileDatasetBuilder,
    file_to_subset: Dict[str, str],
    subset_assets: Dict[str, Dict[str, DocumentAsset]] | None = None,
    *,
    assets_required: bool = False,
) -> Iterable[DatasetRecord]:
    """
    Iterate over records from FileDatasetBuilder and add subset tags.

    The builder creates records with doc_id from filename.stem (original filename),
    so we match it to our file_to_subset mapping to find the subset name. When
    subset_assets is provided, the original binary and mime_type are pulled
    from the CVAT overview instead of the JSON payload.
    """
    subset_assets = subset_assets or {}
    for record in builder.iterate():
        # FileDatasetBuilder sets doc_id to filename.stem (original filename)
        # Match it to our file_to_subset mapping
        subset_name = file_to_subset.get(record.doc_id)
        if subset_name:
            record.tags.append(f"subset:{subset_name}")

            assets_for_subset = subset_assets.get(subset_name)
            if assets_required and assets_for_subset is None:
                raise ValueError(
                    f"No assets loaded for subset {subset_name}; "
                    "ensure --datasets-root points to the base datasets."
                )

            if assets_for_subset:
                asset = assets_for_subset.get(record.doc_id)
                if asset is None and assets_required:
                    raise ValueError(
                        f"Document {record.doc_id} missing from overview for subset {subset_name}"
                    )
                if asset:
                    try:
                        file_bytes = asset.path.read_bytes()
                    except OSError as exc:
                        if assets_required:
                            raise
                        _LOGGER.warning(
                            "Unable to read binary for %s (%s): %s",
                            record.doc_id,
                            asset.path,
                            exc,
                        )
                    else:
                        record.doc_path = asset.path
                        record.doc_hash = asset.doc_hash or get_binhash(file_bytes)
                        resolved_mime = (
                            asset.mime_type or mimetypes.guess_type(asset.path.name)[0]
                        )
                        if resolved_mime:
                            record.mime_type = resolved_mime
                        record.original = DocumentStream(
                            name=asset.path.name,
                            stream=BytesIO(file_bytes),
                        )
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
    subset_assets: Dict[str, Dict[str, DocumentAsset]] | None = None,
    *,
    assets_required: bool = False,
) -> CombinedBuildStats | None:
    """
    Build a single combined dataset from all subsets with subset tags. When
    subset_assets is provided, originals are populated from the referenced
    binaries described in the CVAT overviews.
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
        iter_records_with_tags(
            builder=builder,
            file_to_subset=file_to_subset,
            subset_assets=subset_assets,
            assets_required=assets_required,
        ),
        chunk_size,
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


def render_configs_block(configs: Sequence[ConfigEntry]) -> List[str]:
    lines: List[str] = ["configs:"]
    for config in configs:
        lines.append(f"- config_name: {config.name}")
        lines.append("  data_files:")
        lines.append(f"    - split: {config.split}")
        lines.append(f"      path: {config.path_pattern}")
    return lines


def render_dataset_info_block(configs: Sequence[ConfigEntry]) -> List[str]:
    lines: List[str] = ["dataset_info:"]
    for config in configs:
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
    builds: Sequence[BuildResult],
    license_name: str,
    export_kind: DeliveryExportKind,
    custom_dirname: str | None = None,
) -> None:
    lines: List[str] = ["---"]
    configs = [build.config for build in builds]
    lines.extend(render_configs_block(configs))
    lines.extend(render_dataset_info_block(configs))
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
    for build in builds:
        lines.append(
            f"- `{build.config.name}` ({build.config.split}): "
            f"{build.stats.record_count} records from {build.stats.submission_count} submissions; "
            f"subsets: {', '.join(f'`{s}`' for s in build.stats.subsets)}"
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
    subset_split: List[str] | None = typer.Option(
        None,
        "--subset-split",
        help=(
            "Route subsets to splits using pattern=split (fnmatch). "
            "Example: --subset-split pdf_val=validation --subset-split 'pdf_train_*'=train"
        ),
    ),
    fail_on_unmatched: bool = typer.Option(
        False,
        "--fail-on-unmatched/--allow-unmatched",
        help=(
            "Error if a subset does not match any --subset-split rule instead of "
            "falling back to the default --split."
        ),
    ),
    datasets_root: Path | None = typer.Option(
        None,
        "--datasets-root",
        help=(
            "Root containing <subset>/<assets-dirname>/cvat_annotation_overview.json "
            "or cvat_overview.json. When provided, originals are pulled from the "
            "referenced PDF/image binaries instead of the JSON payload."
        ),
    ),
    assets_dirname: str = typer.Option(
        "cvat_dataset_preannotated",
        "--assets-dirname",
        help="Name of the directory under each subset that holds the CVAT assets tree.",
    ),
    include_submissions: List[str] | None = typer.Option(
        None,
        "--include-submission",
        "-s",
        help=(
            "Restrict processing to these submission directories "
            "(by folder name, e.g., submission-01). Can be provided multiple times."
        ),
    ),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    deliveries_root = deliveries_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    datasets_root = datasets_root.expanduser().resolve() if datasets_root else None
    staging_root = output_dir / "_staging"
    subset_split_rules = _parse_subset_split_rules(subset_split) if subset_split else []

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
        deliveries_root,
        export_kind,
        custom_dirname,
        allowed_submissions=set(include_submissions) if include_submissions else None,
    )
    if not subset_sources:
        typer.echo(
            f"No subset directories with {dirname} found under {deliveries_root}",
            err=True,
        )
        raise typer.Exit(code=1)

    # Sort subset sources for deterministic ordering
    sorted_subset_sources = {k: sorted(v) for k, v in sorted(subset_sources.items())}
    subset_assets = _build_assets_index(
        subset_names=list(sorted_subset_sources.keys()),
        datasets_root=datasets_root,
        assets_dirname=assets_dirname,
        strict=datasets_root is not None,
    )
    if datasets_root is not None:
        missing_subsets = [
            subset
            for subset in sorted_subset_sources.keys()
            if subset not in subset_assets
        ]
        if missing_subsets:
            raise RuntimeError(
                "Missing assets for subsets: "
                + ", ".join(missing_subsets)
                + ". Ensure datasets-root contains matching subsets."
            )

    staging_dir = staging_root / "combined"
    split_map = (
        _route_subsets_to_splits(
            subset_sources=sorted_subset_sources,
            rules=subset_split_rules,
            default_split=split,
            fail_on_unmatched=fail_on_unmatched,
        )
        if subset_split_rules
        else {split: sorted_subset_sources}
    )

    build_results: List[BuildResult] = []
    multi_split = len(split_map) > 1

    for split_name in sorted(split_map.keys()):
        staging_dir = staging_root / f"combined-{split_name}"
        dataset_dir = _dataset_dir_for_split(dataset_dir_name, split_name, multi_split)
        stats = build_combined_dataset(
            subset_sources=split_map[split_name],
            staging_dir=staging_dir,
            output_root=output_dir,
            dataset_dir_name=dataset_dir,
            split=split_name,
            chunk_size=chunk_size,
            export_kind=export_kind,
            force=force,
            subset_assets=subset_assets,
            assets_required=datasets_root is not None,
        )
        if stats is not None:
            config_name = split_name if multi_split or subset_split_rules else None
            config = stats.as_config(
                dataset_dir_name=dataset_dir,
                split=split_name,
                config_name=config_name,
            )
            build_results.append(BuildResult(config=config, stats=stats))

    shutil.rmtree(staging_root, ignore_errors=True)

    if not build_results:
        typer.echo("No datasets were produced.", err=True)
        raise typer.Exit(code=1)

    write_readme(
        output_root=output_dir,
        builds=build_results,
        license_name=license_name,
        export_kind=export_kind,
        custom_dirname=custom_dirname,
    )


if __name__ == "__main__":
    app()
