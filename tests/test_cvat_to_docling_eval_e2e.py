#!/usr/bin/env python3
"""
Example pipeline that demonstrates chaining CVAT annotation processing 
through to layout evaluation.

This script:
1. Converts CVAT annotation files to DoclingDocument JSON
2. Creates a ground truth dataset using FileDatasetBuilder 
3. Creates an evaluation dataset using FilePredictionProvider
4. Runs layout evaluation on the results
"""

import json
import logging
import os
from pathlib import Path
from typing import List

import pytest
from dotenv import load_dotenv

from docling_eval.cli.main import evaluate, get_prediction_provider
from docling_eval.cvat_tools.cvat_to_docling import convert_cvat_to_docling
from docling_eval.datamodels.types import (
    BenchMarkNames,
    EvaluationModality,
    PredictionFormats,
    PredictionProviderType,
)
from docling_eval.dataset_builders.file_dataset_builder import FileDatasetBuilder
from docling_eval.prediction_providers.file_provider import FilePredictionProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)


IS_CI = bool(os.getenv("CI"))
load_dotenv()


def find_cvat_cases(root_dir: Path) -> List[tuple[Path, Path]]:
    """
    Find CVAT annotation XML files and corresponding input files.

    Args:
        root_dir: Root directory containing CVAT cases

    Returns:
        List of tuples (xml_path, input_path)
    """
    cases = []

    # Find all case directories (directories starting with "case")
    case_dirs = sorted(
        [d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("case")]
    )

    # Process each case directory
    for case_dir in case_dirs:
        # Extract case basename from directory name
        case_basename = case_dir.name
        # Construct XML path: case directory's parent + case_basename + "_annotations.xml"
        annotations_xml = case_dir.parent / f"{case_basename}_annotations.xml"

        if annotations_xml.exists():
            # Find all image/PDF files in case directory
            sample_paths = []
            for ext in [
                "*.pdf",
                "*.PDF",
                "*.png",
                "*.PNG",
                "*.jpg",
                "*.JPG",
                "*.jpeg",
                "*.JPEG",
                "*.tif",
                "*.TIF",
                "*.tiff",
                "*.TIFF",
                "*.bmp",
                "*.BMP",
            ]:
                sample_paths.extend(sorted(case_dir.glob(ext)))

            # Add each sample file as a case
            for sample_path in sorted(sample_paths):
                cases.append((annotations_xml, sample_path))

    return cases


def step1_convert_cvat_to_json(
    cvat_root_dir: Path, output_json_dir: Path
) -> List[Path]:
    """
    Step 1: Convert CVAT annotation files to DoclingDocument JSON.

    Args:
        cvat_root_dir: Directory containing CVAT annotation files
        output_json_dir: Directory to save JSON outputs

    Returns:
        List of paths to created JSON files
    """
    _log.info("=== Step 1: Converting CVAT annotations to DoclingDocument JSON ===")

    output_json_dir.mkdir(parents=True, exist_ok=True)
    json_files = []

    # Find CVAT cases
    cases = find_cvat_cases(cvat_root_dir)
    _log.info(f"Found {len(cases)} CVAT cases to process")

    for xml_path, input_path in cases:
        _log.info(f"Processing: {xml_path} with input {input_path}")

        try:
            # Convert CVAT to DoclingDocument
            doc = convert_cvat_to_docling(xml_path, input_path)

            if doc is not None:
                # Save as JSON
                output_name = f"{input_path.stem}_{xml_path.stem}.json"
                json_path = output_json_dir / output_name

                with open(json_path, "w", encoding="utf-8") as f:
                    doc.save_as_json(json_path)
                json_files.append(json_path)
                _log.info(f"✓ Saved DoclingDocument JSON to: {json_path}")
            else:
                _log.error(f"✗ Failed to convert {input_path}")

        except Exception as e:
            raise e
            _log.error(f"✗ Error processing {input_path}: {e}")

    _log.info(f"Step 1 complete: Created {len(json_files)} JSON files")
    return json_files


def step2_create_gt_dataset(json_dir: Path, gt_dataset_dir: Path) -> None:
    """
    Step 2: Create ground truth dataset from JSON files using FileDatasetBuilder.

    Args:
        json_dir: Directory containing DoclingDocument JSON files
        gt_dataset_dir: Directory to save ground truth dataset
    """
    _log.info("=== Step 2: Creating ground truth dataset ===")

    # Create FileDatasetBuilder with only JSON files
    dataset_builder = FileDatasetBuilder(
        name="CVAT_GT_Dataset",
        dataset_source=json_dir,
        target=gt_dataset_dir,
        split="test",
        file_extensions=["json"],  # Only process JSON files
    )

    # Save the dataset to disk with visualization
    dataset_builder.save_to_disk(chunk_size=50, do_visualization=True)

    _log.info(f"Step 2 complete: Ground truth dataset saved to {gt_dataset_dir}")


def step3_create_eval_dataset(
    json_dir: Path, gt_dataset_dir: Path, eval_dataset_dir: Path
) -> None:
    """
    Step 3: Create evaluation dataset using FilePredictionProvider.

    Args:
        json_dir: Directory containing JSON prediction files (same as GT for this example)
        gt_dataset_dir: Ground truth dataset directory
        eval_dataset_dir: Directory to save evaluation dataset
    """
    _log.info("=== Step 3: Creating evaluation dataset ===")

    # Create FilePredictionProvider
    file_provider = FilePredictionProvider(
        prediction_format=PredictionFormats.JSON,
        source_path=json_dir,
        do_visualization=True,
        ignore_missing_files=True,
        ignore_missing_predictions=True,
        use_ground_truth_page_images=True,
    )

    # Create prediction dataset
    file_provider.create_prediction_dataset(
        name="CVAT_Eval_Dataset",
        gt_dataset_dir=gt_dataset_dir,
        target_dataset_dir=eval_dataset_dir,
        split="test",
        chunk_size=50,
    )

    _log.info(f"Step 3 complete: Evaluation dataset saved to {eval_dataset_dir}")


def step4_run_evaluation(eval_dataset_dir: Path, evaluation_output_dir: Path) -> None:
    """
    Step 4: Run layout evaluation on the evaluation dataset.

    Args:
        eval_dataset_dir: Directory containing evaluation dataset
        evaluation_output_dir: Directory to save evaluation results
    """
    _log.info("=== Step 4: Running layout evaluation ===")

    evaluation_output_dir.mkdir(parents=True, exist_ok=True)

    # Run layout evaluation
    evaluation_result = evaluate(
        modality=EvaluationModality.LAYOUT,
        benchmark=BenchMarkNames.CVAT,
        idir=eval_dataset_dir,
        odir=evaluation_output_dir,
        split="test",
    )

    if evaluation_result:
        _log.info("✓ Layout evaluation completed successfully")
        _log.info(f"Results saved to: {evaluation_output_dir}")

        # Print some basic stats
        _log.info(f"Evaluated samples: {evaluation_result.evaluated_samples}")
        _log.info(f"Mean mAP: {evaluation_result.mAP:.4f}")
    else:
        _log.error("✗ Layout evaluation failed")

    # Run tree evaluation
    evaluation_result = evaluate(
        modality=EvaluationModality.DOCUMENT_STRUCTURE,
        benchmark=BenchMarkNames.CVAT,
        idir=eval_dataset_dir,
        odir=evaluation_output_dir,
        split="test",
    )

    if evaluation_result:
        _log.info("✓ Document structure evaluation completed successfully")
        _log.info(f"Results saved to: {evaluation_output_dir}")

        # Print some basic stats
        _log.info(f"Evaluated samples: {evaluation_result.evaluated_samples}")
        _log.info(
            f"Mean edit distance: {evaluation_result.edit_distance_stats.mean:.4f}"
        )
    else:
        _log.error("✗ Document structure evaluation failed")


@pytest.mark.skipif(IS_CI, reason="Skipping test in CI because the test is too heavy.")
def test_cvat_to_docling_eval_e2e():
    """Main pipeline execution."""

    # Define directory structure
    base_dir = Path("./scratch/cvat_evaluation_pipeline")

    # Input: CVAT annotation files (update this path to your CVAT data)
    cvat_input_dir = Path("./tests/data/cvat_pdfs_dataset_e2e")

    # Pipeline directories
    json_output_dir = base_dir / "docling_json_output"
    gt_dataset_dir = base_dir / "gt_dataset"
    eval_dataset_dir = base_dir / "eval_dataset"
    evaluation_results_dir = base_dir / "evaluation_results"

    _log.info("Starting CVAT to Evaluation Pipeline")
    _log.info(f"Input CVAT directory: {cvat_input_dir}")
    _log.info(f"Output base directory: {base_dir}")

    try:
        # Step 1: Convert CVAT annotations to JSON
        json_files = step1_convert_cvat_to_json(cvat_input_dir, json_output_dir)

        if not json_files:
            _log.error("No JSON files were created. Pipeline cannot continue.")
            return

        # Step 2: Create ground truth dataset
        step2_create_gt_dataset(json_output_dir, gt_dataset_dir)

        # Step 3: Create evaluation dataset
        step3_create_eval_dataset(json_output_dir, gt_dataset_dir, eval_dataset_dir)

        # Step 4: Run layout evaluation
        step4_run_evaluation(eval_dataset_dir, evaluation_results_dir)

        _log.info("=== Pipeline completed successfully! ===")
        _log.info(f"Results available in: {base_dir}")

    except Exception as e:
        _log.error(f"Pipeline failed with error: {e}")
        raise
