from __future__ import annotations

import logging
import mimetypes
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Set

import typer

from docling_eval.campaign_tools.cvat_deliveries_to_hf import (
    BuildResult,
    ConfigEntry,
    DeliveryExportKind,
    DocumentAsset,
    _dataset_dir_for_split,
    _load_doc_split_assignments,
    _parse_split_file_rules,
    _parse_subset_split_rules,
    _route_documents_to_splits,
    _route_subsets_to_splits,
    build_combined_dataset,
    collect_subset_json_files,
    ensure_clean_dir,
    iter_dataset_features,
    render_configs_block,
    render_dataset_info_block,
)

_LOGGER = logging.getLogger(__name__)

app = typer.Typer(
    help=(
        "Aggregate DoclingDocument JSON exports from a flat delivery_evaluations layout "
        "into a HuggingFace-ready parquet dataset with batch tags."
    ),
    no_args_is_help=True,
    add_completion=False,
)


def discover_batch_sources(
    deliveries_root: Path,
    json_dirname: str,
    allowed_batches: Set[str] | None = None,
) -> Dict[str, List[Path]]:
    """Discover <batch>/<json_dirname> folders under a flat deliveries root."""
    batch_dirs: Dict[str, List[Path]] = {}
    discovered_batches: Set[str] = set()

    for batch_dir in sorted(p for p in deliveries_root.iterdir() if p.is_dir()):
        batch_name = batch_dir.name
        if batch_name.startswith("."):
            continue
        if allowed_batches is not None and batch_name not in allowed_batches:
            continue

        discovered_batches.add(batch_name)
        candidate = batch_dir / json_dirname
        if candidate.is_dir():
            batch_dirs[batch_name] = [candidate]
        else:
            _LOGGER.debug("Skipping %s (no %s)", batch_dir, json_dirname)

    if allowed_batches is not None:
        missing = allowed_batches.difference(discovered_batches)
        if missing:
            _LOGGER.warning(
                "Requested batches not found under %s: %s",
                deliveries_root,
                ", ".join(sorted(missing)),
            )

    return batch_dirs


def find_duplicate_doc_ids(
    subset_sources: Dict[str, List[Path]]
) -> Dict[str, List[str]]:
    """Return doc-id collisions across subsets keyed by document id."""
    doc_id_to_subsets: Dict[str, List[str]] = {}

    for subset_name in sorted(subset_sources.keys()):
        for source_dir in subset_sources[subset_name]:
            for json_file in sorted(source_dir.glob("*.json")):
                doc_id = json_file.stem
                if doc_id not in doc_id_to_subsets:
                    doc_id_to_subsets[doc_id] = [subset_name]
                    continue
                if subset_name not in doc_id_to_subsets[doc_id]:
                    doc_id_to_subsets[doc_id].append(subset_name)

    return {
        doc_id: sorted(subsets)
        for doc_id, subsets in doc_id_to_subsets.items()
        if len(subsets) > 1
    }


def discover_flat_pdf_assets(
    subset_sources: Dict[str, List[Path]],
    assets_root: Path,
) -> Dict[str, Dict[str, DocumentAsset]]:
    """
    Discover per-document PDF assets under a flat assets root.

    Expected layout:
      <assets_root>/<subset_name>/<doc_id>.pdf
    """
    assets_by_subset: Dict[str, Dict[str, DocumentAsset]] = {}

    for subset_name in sorted(subset_sources.keys()):
        subset_dir = assets_root / subset_name
        if not subset_dir.exists():
            raise FileNotFoundError(
                f"Missing assets directory for subset {subset_name}: {subset_dir}"
            )

        doc_ids: Set[str] = set()
        for source_dir in subset_sources[subset_name]:
            for json_file in sorted(source_dir.glob("*.json")):
                doc_ids.add(json_file.stem)

        subset_assets: Dict[str, DocumentAsset] = {}
        for doc_id in sorted(doc_ids):
            pdf_path = subset_dir / f"{doc_id}.pdf"
            if not pdf_path.exists():
                raise FileNotFoundError(
                    f"Missing PDF asset for subset {subset_name}, doc_id {doc_id}: {pdf_path}"
                )

            guessed_mime, _ = mimetypes.guess_type(pdf_path.name)
            subset_assets[doc_id] = DocumentAsset(
                path=pdf_path,
                mime_type=guessed_mime or "application/pdf",
                doc_hash=None,
            )

        assets_by_subset[subset_name] = subset_assets

    return assets_by_subset


def write_flat_readme(
    output_root: Path,
    builds: Sequence[BuildResult],
    license_name: str,
    json_dirname: str,
    assets_root: Path | None = None,
) -> None:
    lines: List[str] = ["---"]
    configs = [build.config for build in builds]
    lines.extend(render_configs_block(configs))
    lines.extend(render_dataset_info_block(configs))
    lines.append(f"license: {license_name}")
    lines.append("---")
    lines.append("")
    lines.append("# Flat CVAT Delivery Aggregation")
    lines.append(
        "This repository consolidates DoclingDocument exports produced from flat "
        "CVAT delivery batches into a single HuggingFace-ready parquet dataset."
    )
    lines.append("")
    lines.append(f"- Source payloads: `{json_dirname}`")
    lines.append("- Builder: `FileDatasetBuilder` with JSON inputs")
    lines.append(
        "- Batch information: Each record includes a `tags` field with "
        "`subset:<batch_name>` entries indicating the source batch"
    )
    if assets_root is not None:
        lines.append(
            f"- Original binaries: loaded from `{assets_root}` using `<subset>/<doc_id>.pdf` mapping"
        )
    lines.append("")
    lines.append("## Dataset Statistics")
    for build in builds:
        batch_count = build.stats.submission_count
        lines.append(
            f"- `{build.config.name}` ({build.config.split}): "
            f"{build.stats.record_count} records from {batch_count} batches; "
            f"batches: {', '.join(f'`{batch}`' for batch in build.stats.subsets)}"
        )

    readme_path = output_root / "README.md"
    readme_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


@app.command()
def main(
    deliveries_root: Path = typer.Argument(
        ...,
        help="Root directory containing flat batch folders (e.g., batch_*/docling_json).",
    ),
    output_dir: Path = typer.Argument(
        ...,
        help="Output directory where HuggingFace-ready datasets will be written.",
    ),
    json_dirname: str = typer.Option(
        "docling_json",
        "--json-dirname",
        help="Directory name under each batch containing JSON payloads.",
    ),
    split: str = typer.Option(
        "test",
        "--split",
        help="Dataset split label used in the output structure.",
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
    subset_split: List[str] | None = typer.Option(
        None,
        "--subset-split",
        help=(
            "Route batches to splits using pattern=split (fnmatch). "
            "Example: --subset-split 'batch_01*'=validation --subset-split 'batch_2*'=test"
        ),
    ),
    split_file: List[str] | None = typer.Option(
        None,
        "--split-file",
        help=(
            "Assign documents to splits using split=path, where each file lists one "
            "document id per line. Example: --split-file train=/path/train.txt "
            "--split-file validation=/path/validation.txt"
        ),
    ),
    fail_on_unmatched: bool = typer.Option(
        False,
        "--fail-on-unmatched/--allow-unmatched",
        help=(
            "With --split-file, error if any exported document is not listed in a "
            "split file; otherwise unmatched documents are omitted. With "
            "--subset-split, error if a batch does not match any rule instead of "
            "falling back to the default --split."
        ),
    ),
    include_batches: List[str] | None = typer.Option(
        None,
        "--include-batch",
        "-b",
        help=(
            "Restrict processing to these batch directories "
            "(by folder name, e.g., batch_01_forms_datasheets_accounting). "
            "Can be provided multiple times."
        ),
    ),
    assets_root: Path | None = typer.Option(
        None,
        "--assets-root",
        help=(
            "Optional root containing original per-document PDFs as "
            "<subset>/<doc_id>.pdf. When provided, each record stores the original PDF "
            "instead of relying only on JSON payload fields."
        ),
    ),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    deliveries_root = deliveries_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    assets_root = assets_root.expanduser().resolve() if assets_root else None
    staging_root = output_dir / "_staging"
    subset_split_rules = _parse_subset_split_rules(subset_split) if subset_split else []
    split_file_rules = _parse_split_file_rules(split_file) if split_file else []

    if not deliveries_root.exists():
        raise typer.BadParameter(f"{deliveries_root} does not exist.")
    if assets_root is not None and not assets_root.exists():
        raise typer.BadParameter(f"{assets_root} does not exist.")
    if split_file_rules and subset_split_rules:
        raise typer.BadParameter("Use either --split-file or --subset-split, not both.")

    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_clean_dir(staging_root)

    subset_sources = discover_batch_sources(
        deliveries_root=deliveries_root,
        json_dirname=json_dirname,
        allowed_batches=set(include_batches) if include_batches else None,
    )
    if not subset_sources:
        typer.echo(
            f"No batch directories with {json_dirname} found under {deliveries_root}",
            err=True,
        )
        raise typer.Exit(code=1)

    sorted_subset_sources = {k: sorted(v) for k, v in sorted(subset_sources.items())}
    subset_assets = (
        discover_flat_pdf_assets(sorted_subset_sources, assets_root)
        if assets_root is not None
        else {}
    )

    duplicate_doc_ids = find_duplicate_doc_ids(sorted_subset_sources)
    if duplicate_doc_ids:
        sample = sorted(duplicate_doc_ids.items())[:10]
        sample_rendered = "; ".join(
            f"{doc_id} -> {', '.join(subsets)}" for doc_id, subsets in sample
        )
        raise RuntimeError(
            "Duplicate document IDs detected across batches. "
            "This would overwrite files in staging. "
            f"Found {len(duplicate_doc_ids)} duplicates. Sample: {sample_rendered}"
        )

    all_subset_json_files = collect_subset_json_files(sorted_subset_sources)
    split_map: Dict[str, Dict[str, List[Path]]]
    if split_file_rules:
        doc_to_split = _load_doc_split_assignments(split_file_rules)
        split_map = _route_documents_to_splits(
            subset_json_files=all_subset_json_files,
            doc_to_split=doc_to_split,
            fail_on_unmatched=fail_on_unmatched,
        )
    elif subset_split_rules:
        routed_subset_sources = _route_subsets_to_splits(
            subset_sources=sorted_subset_sources,
            rules=subset_split_rules,
            default_split=split,
            fail_on_unmatched=fail_on_unmatched,
        )
        split_map = {
            split_name: collect_subset_json_files(subset_sources)
            for split_name, subset_sources in sorted(routed_subset_sources.items())
        }
    else:
        split_map = {split: all_subset_json_files}

    build_results: List[BuildResult] = []
    multi_split = len(split_map) > 1

    for split_name in sorted(split_map.keys()):
        staging_dir = staging_root / f"combined-{split_name}"
        dataset_dir = _dataset_dir_for_split(dataset_dir_name, split_name, multi_split)
        subset_json_files = split_map[split_name]
        source_count = len(
            {
                json_file.parent
                for paths in subset_json_files.values()
                for json_file in paths
            }
        )

        stats = build_combined_dataset(
            subset_json_files=subset_json_files,
            staging_dir=staging_dir,
            output_root=output_dir,
            dataset_dir_name=dataset_dir,
            split=split_name,
            chunk_size=chunk_size,
            export_kind=DeliveryExportKind.GROUND_TRUTH,
            force=force,
            source_count=source_count,
            subset_assets=subset_assets,
            assets_required=assets_root is not None,
        )
        if stats is None:
            continue

        config = stats.as_config(
            dataset_dir_name=dataset_dir,
            split=split_name,
        )
        build_results.append(BuildResult(config=config, stats=stats))

    shutil.rmtree(staging_root, ignore_errors=True)

    if not build_results:
        typer.echo("No datasets were produced.", err=True)
        raise typer.Exit(code=1)

    write_flat_readme(
        output_root=output_dir,
        builds=build_results,
        license_name=license_name,
        json_dirname=json_dirname,
        assets_root=assets_root,
    )


if __name__ == "__main__":
    app()
