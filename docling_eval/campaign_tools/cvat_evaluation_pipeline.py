#!/usr/bin/env python3
"""
CVAT Evaluation Pipeline Utility

This utility provides a flexible pipeline for evaluating CVAT annotations by:
1. Converting CVAT XML annotations to DoclingDocument JSON format
2. Creating ground truth and prediction datasets
3. Running layout and document structure evaluations

The pipeline can be run in separate steps or end-to-end.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from docling_eval.campaign_tools.combine_cvat_evaluations import (
    combine_cvat_evaluations,
)
from docling_eval.cli.main import evaluate
from docling_eval.cvat_tools.cvat_to_docling import convert_cvat_to_docling
from docling_eval.cvat_tools.parser import MissingImageInCVATXML
from docling_eval.datamodels.types import (
    BenchMarkNames,
    EvaluationModality,
    PredictionFormats,
)
from docling_eval.dataset_builders.file_dataset_builder import FileDatasetBuilder
from docling_eval.prediction_providers.file_provider import FilePredictionProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
_log = logging.getLogger(__name__)


class CVATEvaluationPipeline:
    """Pipeline for CVAT annotation evaluation."""

    def __init__(self, images_dir: Path, output_dir: Path):
        """
        Initialize the pipeline.

        Args:
            images_dir: Directory containing PNG image files
            output_dir: Base directory for all pipeline outputs
        """
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)

        # Create subdirectories
        self.gt_json_dir = self.output_dir / "ground_truth_json"
        self.pred_json_dir = self.output_dir / "predictions_json"
        self.gt_dataset_dir = self.output_dir / "gt_dataset"
        self.eval_dataset_dir = self.output_dir / "eval_dataset"
        self.evaluation_results_dir = self.output_dir / "evaluation_results"

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _find_image_files(self) -> List[Path]:
        """Find all PNG image files recursively in the images directory."""
        image_extensions = ["*.png", "*.PNG"]
        image_files = []

        for ext in image_extensions:
            image_files.extend(sorted(self.images_dir.rglob(ext)))

        if not image_files:
            raise ValueError(f"No PNG files found recursively in {self.images_dir}")

        _log.info(f"Found {len(image_files)} image files (searched recursively)")
        return image_files

    def _convert_cvat_to_json(
        self, cvat_xml_path: Path, output_json_dir: Path, prefix: str = ""
    ) -> List[Path]:
        """
        Convert CVAT XML annotations to DoclingDocument JSON format.

        Args:
            cvat_xml_path: Path to CVAT XML annotation file
            output_json_dir: Directory to save JSON outputs
            prefix: Optional prefix for output filenames

        Returns:
            List of created JSON file paths
        """
        output_json_dir.mkdir(parents=True, exist_ok=True)
        json_files = []

        image_files = self._find_image_files()

        for image_path in image_files:
            _log.info(
                f"Converting {image_path.name} with annotations from {cvat_xml_path.name}"
            )

            try:
                # Convert CVAT to DoclingDocument
                doc = convert_cvat_to_docling(cvat_xml_path, image_path)

                if doc is not None:
                    # Create output filename
                    base_name = image_path.stem
                    if prefix:
                        output_name = f"{prefix}_{base_name}.json"
                    else:
                        output_name = f"{base_name}.json"

                    json_path = output_json_dir / output_name

                    # Save as JSON
                    doc.save_as_json(json_path)
                    json_files.append(json_path)
                    _log.info(f"\u2713 Saved DoclingDocument JSON to: {json_path}")
                else:
                    _log.warning(f"\u26a0 Failed to convert {image_path.name}")

            except MissingImageInCVATXML:
                _log.warning(
                    f"Image {image_path.name} not found in {cvat_xml_path.name}. "
                    "This is expected for partial annotation batches. Skipping."
                )
                continue
            except ValueError as ve:
                _log.error(f"\u2717 Error processing {image_path.name}: {ve}")
                continue
            except Exception as e:
                _log.error(f"\u2717 Error processing {image_path.name}: {e}")
                continue

        _log.info(f"Converted {len(json_files)} files to JSON format")
        return json_files

    def create_ground_truth_dataset(self, gt_cvat_xml: Path) -> None:
        """
        Step 1: Create ground truth dataset from CVAT XML.

        Args:
            gt_cvat_xml: Path to ground truth CVAT XML file
        """
        _log.info("=== Creating Ground Truth Dataset ===")

        # Convert CVAT XML to JSON
        gt_json_files = self._convert_cvat_to_json(gt_cvat_xml, self.gt_json_dir)

        if not gt_json_files:
            raise ValueError("No ground truth JSON files were created")

        # Create ground truth dataset
        dataset_builder = FileDatasetBuilder(
            name="CVAT_Ground_Truth_Dataset",
            dataset_source=self.gt_json_dir,
            target=self.gt_dataset_dir,
            split="test",
            file_extensions=["json"],
        )

        dataset_builder.save_to_disk(chunk_size=50, do_visualization=True)
        _log.info(f"✓ Ground truth dataset created: {self.gt_dataset_dir}")

    def create_prediction_dataset(self, pred_cvat_xml: Path) -> None:
        """
        Step 2: Create prediction dataset from CVAT XML using the ground truth dataset.

        Args:
            pred_cvat_xml: Path to prediction CVAT XML file
        """
        _log.info("=== Creating Prediction Dataset ===")

        if not self.gt_dataset_dir.exists():
            raise ValueError(
                f"Ground truth dataset not found at {self.gt_dataset_dir}. "
                "Please run create_ground_truth_dataset first."
            )

        # Convert prediction CVAT XML to JSON
        pred_json_files = self._convert_cvat_to_json(pred_cvat_xml, self.pred_json_dir)

        if not pred_json_files:
            raise ValueError("No prediction JSON files were created")

        # Create prediction dataset using FilePredictionProvider
        file_provider = FilePredictionProvider(
            prediction_format=PredictionFormats.JSON,
            source_path=self.pred_json_dir,
            do_visualization=True,
            ignore_missing_files=True,
            ignore_missing_predictions=True,
            use_ground_truth_page_images=True,
        )

        file_provider.create_prediction_dataset(
            name="CVAT_Prediction_Dataset",
            gt_dataset_dir=self.gt_dataset_dir,
            target_dataset_dir=self.eval_dataset_dir,
            split="test",
            chunk_size=50,
        )
        _log.info(f"✓ Prediction dataset created: {self.eval_dataset_dir}")

    def run_evaluation(
        self, modalities: Optional[List[str]] = None, user_csv: Optional[Path] = None
    ) -> None:
        """
        Step 3: Run evaluation on the prediction dataset.

        Args:
            modalities: List of evaluation modalities.
                       Options: ['layout', 'document_structure']
                       Default: both modalities
            user_csv: Path to user CSV file for provenance/self-confidence (optional)
        """
        _log.info("=== Running Evaluation ===")

        if not self.eval_dataset_dir.exists():
            raise ValueError(
                f"Evaluation dataset not found at {self.eval_dataset_dir}. "
                "Please run create_prediction_dataset first."
            )

        if modalities is None:
            modalities = ["layout", "document_structure", "key_value"]

        self.evaluation_results_dir.mkdir(parents=True, exist_ok=True)

        for modality_name in modalities:
            _log.info(f"Running {modality_name} evaluation...")

            if modality_name == "layout":
                modality = EvaluationModality.LAYOUT
            elif modality_name == "document_structure":
                modality = EvaluationModality.DOCUMENT_STRUCTURE
            elif modality_name == "key_value":
                modality = EvaluationModality.KEY_VALUE
            else:
                _log.warning(f"Unknown modality: {modality_name}. Skipping.")
                continue
            # TODO: add key-value evaluation, see https://github.com/docling-project/docling-eval/pull/140

            try:
                evaluation_result = evaluate(
                    modality=modality,
                    benchmark=BenchMarkNames.CVAT,
                    idir=self.eval_dataset_dir,
                    odir=self.evaluation_results_dir,
                    split="test",
                )

                if evaluation_result:
                    _log.info(
                        f"\u2713 {modality_name} evaluation completed successfully"
                    )
                    _log.info(
                        f"Evaluated samples: {evaluation_result.evaluated_samples}"
                    )

                    if modality_name == "layout":
                        _log.info(f"Mean mAP: {evaluation_result.mAP:.4f}")
                    elif modality_name == "document_structure":
                        _log.info(
                            f"Mean edit distance: {evaluation_result.edit_distance_stats.mean:.4f}"
                        )
                else:
                    _log.error(f"\u2717 {modality_name} evaluation failed")

            except Exception as e:
                _log.error(f"\u2717 Error in {modality_name} evaluation: {e}")
                raise e

        # Combine results if user_csv is provided
        combined_out = self.output_dir / "combined_evaluation.xlsx"
        layout_json = self.evaluation_results_dir / "evaluation_CVAT_layout.json"
        docstruct_json = (
            self.evaluation_results_dir / "evaluation_CVAT_document_structure.json"
        )
        _log.info(f"Combining evaluation results to {combined_out}")
        combine_cvat_evaluations(
            layout_json=layout_json,
            docstruct_json=docstruct_json,
            user_csv=user_csv,
            out=combined_out,
        )

    def run_full_pipeline(
        self,
        gt_cvat_xml: Path,
        pred_cvat_xml: Path,
        modalities: Optional[List[str]] = None,
        user_csv: Optional[Path] = None,
    ) -> None:
        """
        Run the complete pipeline: create datasets, run evaluation, and combine results.

        Args:
            gt_cvat_xml: Path to ground truth CVAT XML file
            pred_cvat_xml: Path to prediction CVAT XML file
            modalities: List of evaluation modalities to run
            user_csv: Path to user CSV file for provenance/self-confidence
            combined_out: Output file for combined evaluation (defaults to output_dir/combined_evaluation.xlsx)
        """
        _log.info("=== Running Full CVAT Evaluation Pipeline ===")

        try:
            self.create_ground_truth_dataset(gt_cvat_xml)
            self.create_prediction_dataset(pred_cvat_xml)
            self.run_evaluation(modalities, user_csv)

            # Combine results if user_csv is provided
            combined_out = self.output_dir / "combined_evaluation.xlsx"
            layout_json = self.evaluation_results_dir / "evaluation_CVAT_layout.json"
            docstruct_json = (
                self.evaluation_results_dir / "evaluation_CVAT_document_structure.json"
            )
            _log.info(f"Combining evaluation results to {combined_out}")
            combine_cvat_evaluations(
                layout_json=layout_json,
                docstruct_json=docstruct_json,
                user_csv=user_csv,
                out=combined_out,
            )

            _log.info("=== Pipeline completed successfully! ===")
            _log.info(f"Results available in: {self.output_dir}")

        except Exception as e:
            _log.error(f"Pipeline failed with error: {e}")
            raise e


def main():
    """Command line interface for the CVAT evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="CVAT Evaluation Pipeline - Convert CVAT annotations and run evaluations"
    )

    parser.add_argument(
        "images_dir", type=Path, help="Directory containing PNG image files"
    )

    parser.add_argument(
        "output_dir", type=Path, help="Output directory for pipeline results"
    )

    parser.add_argument(
        "--gt-xml", type=Path, help="Path to ground truth CVAT XML file"
    )

    parser.add_argument(
        "--pred-xml", type=Path, help="Path to prediction CVAT XML file"
    )

    parser.add_argument(
        "--user-csv",
        type=Path,
        default=None,
        help="Path to user CSV file for provenance/self-confidence (optional, used for combining evaluation results)",
    )

    parser.add_argument(
        "--step",
        choices=["gt", "pred", "eval", "full"],
        default="full",
        help="Pipeline step to run: gt (ground truth), pred (predictions), eval (evaluation), full (all steps)",
    )

    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=["layout", "document_structure"],
        default=["layout", "document_structure"],
        help="Evaluation modalities to run",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate inputs
    if not args.images_dir.exists():
        _log.error(f"Images directory does not exist: {args.images_dir}")
        sys.exit(1)

    # Initialize pipeline
    pipeline = CVATEvaluationPipeline(args.images_dir, args.output_dir)

    if args.step == "gt":
        if not args.gt_xml:
            _log.error("--gt-xml is required for ground truth step")
            sys.exit(1)
        if not args.gt_xml.exists():
            _log.error(f"Ground truth XML file does not exist: {args.gt_xml}")
            sys.exit(1)
        pipeline.create_ground_truth_dataset(args.gt_xml)

    elif args.step == "pred":
        if not args.pred_xml:
            _log.error("--pred-xml is required for prediction step")
            sys.exit(1)
        if not args.pred_xml.exists():
            _log.error(f"Prediction XML file does not exist: {args.pred_xml}")
            sys.exit(1)
        pipeline.create_prediction_dataset(args.pred_xml)

    elif args.step == "eval":
        pipeline.run_evaluation(args.modalities, user_csv=args.user_csv)

    elif args.step == "full":
        if not args.gt_xml or not args.pred_xml:
            _log.error("Both --gt-xml and --pred-xml are required for full pipeline")
            sys.exit(1)
        if not args.gt_xml.exists():
            _log.error(f"Ground truth XML file does not exist: {args.gt_xml}")
            sys.exit(1)
        if not args.pred_xml.exists():
            _log.error(f"Prediction XML file does not exist: {args.pred_xml}")
            sys.exit(1)
        pipeline.run_full_pipeline(
            args.gt_xml, args.pred_xml, args.modalities, user_csv=args.user_csv
        )


if __name__ == "__main__":
    main()
