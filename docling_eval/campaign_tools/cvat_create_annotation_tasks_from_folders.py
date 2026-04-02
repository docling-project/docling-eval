"""
CLI utility to batch-create datasets for annotation workflows from directories containing plain files.

Given a root input directory containing subdirectories (each with plain files: PDF, image, etc.),
this tool creates, for each subdirectory:
  - gt_dataset: ground truth dataset
  - eval_dataset: weak annotation dataset using Docling predictions
  - cvat_dataset_preannotated: CVAT-ready input structure for annotation

If a subdirectory contains more than max_files_per_chunk files (default: 1000), it will be
automatically split into multiple chunks with separate output directories named:
  - {subdir_name}_chunk_001, {subdir_name}_chunk_002, etc.

This is useful for preparing large-scale annotation tasks for CVAT or similar tools while
avoiding "too many files open" errors.

Usage:
    uv run python docling_eval/campaign_tools/cvat_create_annotation_tasks_from_folders.py batch-prepare --input-directory <input_dir> --output-directory <output_dir> [--sliding-window <int>] [--use-predictions/--no-use-predictions] [--max-files-per-chunk <int>] [--start-index <int>] [--end-index <int>] [--chunk-number-offset <int>]

Arguments:
    input_directory: Root directory containing subdirectories with files to process
    output_directory: Where to store the generated datasets (one subdir per input subdir, with chunk suffixes if needed)
    sliding_window: Number of pages per CVAT task (default: 1)
    use_predictions: Whether to create prediction dataset and use predictions in CVAT (default: True)
    max_files_per_chunk: Maximum number of files to process per chunk (default: 1000)
    start_index: Start processing from this absolute file index within each sorted subdirectory (default: 0)
    end_index: Stop processing at this absolute file index within each sorted subdirectory, exclusive. -1 means process through the end (default: -1)
    chunk_number_offset: Add this offset to generated chunk numbers so resumed runs can continue numbering from previous chunks (default: 0)
"""

from pathlib import Path
from typing import Optional

import typer

from docling_eval.cli.main import create_cvat, create_eval, create_gt
from docling_eval.datamodels.types import BenchMarkNames, PredictionProviderType
from docling_eval.utils.utils import count_pages_in_file

app = typer.Typer(add_completion=False)


def process_subdirectories(
    input_directory: Path,
    output_directory: Path,
    sliding_window: int = 1,
    use_predictions: bool = True,
    max_files_per_chunk: int = 1000,
    start_index: int = 0,
    end_index: int = -1,
    chunk_number_offset: int = 0,
) -> None:
    """
    For each subdirectory in input_directory, create gt_dataset, eval_dataset, and cvat_dataset_preannotated
    in the corresponding output_directory. If a subdirectory contains more than max_files_per_chunk files,
    it will be automatically split into multiple chunks with separate output directories.

    Args:
        input_directory: Root directory with subdirectories to process
        output_directory: Where to store generated datasets
        sliding_window: Number of pages per CVAT task (default: 1)
        use_predictions: Whether to create prediction dataset and use predictions in CVAT
        max_files_per_chunk: Maximum number of files to process per chunk (default: 1000)
        start_index: Start processing from this absolute file index within the sorted file list
        end_index: Stop processing at this absolute file index, exclusive. -1 means process through the end
        chunk_number_offset: Add this offset to output chunk numbering
    """
    input_directory = input_directory.expanduser().resolve()
    output_directory = output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    subdirs = [d for d in input_directory.iterdir() if d.is_dir()]
    if not subdirs:
        typer.echo(f"No subdirectories found in {input_directory}")
        raise typer.Exit(1)

    for subdir in subdirs:
        subdir_name = subdir.name

        # Collect all files and count pages
        all_files: list[Path] = []
        for ext in ["pdf", "tif", "tiff", "jpg", "jpeg", "png", "bmp", "gif", "json"]:
            all_files.extend(subdir.glob(f"*.{ext}"))
            all_files.extend(subdir.glob(f"*.{ext.upper()}"))
        all_files.sort()

        total_files = len(all_files)
        total_pages = sum(count_pages_in_file(f) for f in all_files)
        selected_begin_index = max(start_index, 0)
        selected_end_index = (
            total_files if end_index < 0 else min(end_index, total_files)
        )

        if selected_begin_index > total_files:
            selected_begin_index = total_files
        if selected_end_index < selected_begin_index:
            selected_end_index = selected_begin_index

        selected_files = selected_end_index - selected_begin_index

        typer.echo(f"\nProcessing: {subdir_name}")
        typer.echo(f"  Total files found: {total_files}")
        typer.echo(f"  Total pages found: {total_pages}")
        typer.echo(
            f"  Selected file range: [{selected_begin_index}:{selected_end_index}] ({selected_files} files)"
        )

        if selected_files == 0:
            typer.echo("  No files selected for this subdirectory, skipping.")
            continue

        # Calculate number of chunks needed
        num_chunks = (selected_files + max_files_per_chunk - 1) // max_files_per_chunk
        typer.echo(
            f"  Will create {num_chunks} chunk(s) of max {max_files_per_chunk} files each"
        )

        # Process each chunk
        for chunk_idx in range(num_chunks):
            begin_index = selected_begin_index + chunk_idx * max_files_per_chunk
            chunk_end_index = min(
                selected_begin_index + (chunk_idx + 1) * max_files_per_chunk,
                selected_end_index,
            )
            files_in_chunk = chunk_end_index - begin_index
            output_chunk_number = chunk_number_offset + chunk_idx + 1

            # Create chunk-specific output directory
            if (
                num_chunks == 1
                and selected_begin_index == 0
                and selected_end_index == total_files
                and chunk_number_offset == 0
            ):
                # Single chunk, use original directory name
                chunk_output_dir = output_directory / subdir_name
            else:
                # Multiple chunks, append chunk number
                chunk_output_dir = (
                    output_directory / f"{subdir_name}_chunk_{output_chunk_number:03d}"
                )

            chunk_output_dir.mkdir(parents=True, exist_ok=True)

            typer.echo(
                f"  Processing chunk {chunk_idx + 1}/{num_chunks}: files {begin_index} to {chunk_end_index-1} ({files_in_chunk} files)"
            )
            typer.echo(f"  Output directory: {chunk_output_dir}")

            gt_dir = chunk_output_dir / "gt_dataset"
            eval_dir = chunk_output_dir / "eval_dataset"
            cvat_dir = chunk_output_dir / "cvat_dataset_preannotated"

            if cvat_dir.exists():
                typer.echo(
                    "    CVAT dataset already exists, treating chunk as complete and skipping."
                )
                continue

            if not gt_dir.exists():
                typer.echo(f"    Creating GT dataset...")
                create_gt(
                    benchmark=BenchMarkNames.PLAIN_FILES,
                    dataset_source=subdir,
                    output_dir=chunk_output_dir,
                    do_visualization=False,
                    begin_index=begin_index,
                    end_index=chunk_end_index,
                )
            else:
                typer.echo(f"    GT dataset already exists, skipping.")

            if use_predictions:
                if not eval_dir.exists():
                    typer.echo(f"    Creating prediction dataset (Docling)...")
                    create_eval(
                        benchmark=BenchMarkNames.PLAIN_FILES,
                        output_dir=chunk_output_dir,
                        prediction_provider=PredictionProviderType.DOCLING,
                        do_visualization=True,
                        image_scale_factor=2.0,
                        do_table_structure=False,
                        begin_index=begin_index,
                        end_index=chunk_end_index,
                    )
                else:
                    typer.echo(f"    Prediction dataset already exists, skipping.")
            else:
                typer.echo(
                    f"    Skipping prediction dataset creation (use_predictions=False)."
                )

            if not cvat_dir.exists():
                typer.echo(f"    Creating CVAT pre-annotated dataset...")
                # Use gt_dir when no predictions, eval_dir when using predictions
                source_dir = (
                    (eval_dir / "test") if use_predictions else (gt_dir / "test")
                )
                create_cvat(
                    gt_dir=source_dir,
                    output_dir=cvat_dir,
                    bucket_size=100,
                    use_predictions=use_predictions,
                    sliding_window=sliding_window,
                )
            else:
                typer.echo(f"    CVAT dataset already exists, skipping.")

            assert (
                gt_dir.exists()
            ), f"gt_dataset not created for {subdir_name} chunk {chunk_idx + 1}"
            assert (
                cvat_dir.exists()
            ), f"cvat_dataset_preannotated not created for {subdir_name} chunk {chunk_idx + 1}"
            typer.echo(
                f"    Successfully created all datasets for chunk {chunk_idx + 1}"
            )

        typer.echo(
            f"  Successfully processed all {num_chunks} chunk(s) for {subdir_name}"
        )


@app.command()
def batch_prepare(
    input_directory: Path = typer.Option(
        ..., help="Root directory with subdirectories to process."
    ),
    output_directory: Path = typer.Option(
        ..., help="Where to store generated datasets."
    ),
    sliding_window: int = typer.Option(
        1, help="Number of pages per CVAT task (default: 1)"
    ),
    use_predictions: bool = typer.Option(
        True, help="Whether to create prediction dataset and use predictions in CVAT"
    ),
    max_files_per_chunk: int = typer.Option(
        1000, help="Maximum number of files to process per chunk (default: 1000)"
    ),
    start_index: int = typer.Option(
        0,
        help="Start processing from this absolute file index within each sorted subdirectory (default: 0)",
    ),
    end_index: int = typer.Option(
        -1,
        help="Stop processing at this absolute file index, exclusive. Use -1 for all remaining files (default: -1)",
    ),
    chunk_number_offset: int = typer.Option(
        0,
        help="Add this offset to output chunk numbering so resumed runs continue from prior chunk numbers (default: 0)",
    ),
) -> None:
    """
    Batch-create Docling evaluation datasets for all subdirectories in input_directory.
    """
    process_subdirectories(
        input_directory,
        output_directory,
        sliding_window,
        use_predictions,
        max_files_per_chunk,
        start_index,
        end_index,
        chunk_number_offset,
    )
    typer.echo("\nAll benchmarks created successfully!")


if __name__ == "__main__":
    app()
