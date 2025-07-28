"""
CLI utility to batch-create datasets for annotation workflows from directores containing plain files.

Given a root input directory containing subdirectories (each with plain files: PDF, image, etc.),
this tool creates, for each subdirectory:
  - gt_dataset: ground truth dataset
  - eval_dataset: weak annotation dataset using Docling predictions
  - cvat_dataset_preannotated: CVAT-ready input structure for annotation

This is useful for preparing large-scale annotation tasks for CVAT or similar tools.

Usage:
    uv run python scratches/scratch_46.py --input-directory <input_dir> --output-directory <output_dir> [--sliding-window <int>]

Arguments:
    input_directory: Root directory containing subdirectories with files to process
    output_directory: Where to store the generated datasets (one subdir per input subdir)
    sliding_window: Number of pages per CVAT task (default: 1)
"""

from pathlib import Path
from typing import Optional

import typer

from docling_eval.cli.main import create_cvat, create_eval, create_gt
from docling_eval.datamodels.types import BenchMarkNames, PredictionProviderType

app = typer.Typer(add_completion=False)


def process_subdirectories(
    input_directory: Path, output_directory: Path, sliding_window: int = 1
) -> None:
    """
    For each subdirectory in input_directory, create gt_dataset, eval_dataset, and cvat_dataset_preannotated
    in the corresponding output_directory.

    Args:
        input_directory: Root directory with subdirectories to process
        output_directory: Where to store generated datasets
        sliding_window: Number of pages per CVAT task (default: 1)
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
        odir = output_directory / subdir_name
        odir.mkdir(parents=True, exist_ok=True)
        typer.echo(f"\nProcessing: {subdir_name}")

        gt_dir = odir / "gt_dataset"
        eval_dir = odir / "eval_dataset"
        cvat_dir = odir / "cvat_dataset_preannotated"

        if not gt_dir.exists():
            typer.echo(f"  Creating GT dataset...")
            create_gt(
                benchmark=BenchMarkNames.PLAIN_FILES,
                dataset_source=subdir,
                output_dir=odir,
                do_visualization=False,
            )
        else:
            typer.echo(f"  GT dataset already exists, skipping.")

        if not eval_dir.exists():
            typer.echo(f"  Creating prediction dataset (Docling)...")
            create_eval(
                benchmark=BenchMarkNames.PLAIN_FILES,
                output_dir=odir,
                prediction_provider=PredictionProviderType.DOCLING,
                do_visualization=True,
                image_scale_factor=2.0,
                do_table_structure=False,
            )
        else:
            typer.echo(f"  Prediction dataset already exists, skipping.")

        if not cvat_dir.exists():
            typer.echo(f"  Creating CVAT pre-annotated dataset...")
            create_cvat(
                gt_dir=eval_dir / "test",
                output_dir=cvat_dir,
                bucket_size=100,
                use_predictions=True,
                sliding_window=sliding_window,
            )
        else:
            typer.echo(f"  CVAT dataset already exists, skipping.")

        assert gt_dir.exists(), f"gt_dataset not created for {subdir_name}"
        assert (
            cvat_dir.exists()
        ), f"cvat_dataset_preannotated not created for {subdir_name}"
        typer.echo(f"  Successfully created all datasets for {subdir_name}")


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
) -> None:
    """
    Batch-create Docling evaluation datasets for all subdirectories in input_directory.
    """
    process_subdirectories(input_directory, output_directory, sliding_window)
    typer.echo("\nAll benchmarks created successfully!")


if __name__ == "__main__":
    app()
