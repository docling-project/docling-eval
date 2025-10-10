#!/usr/bin/env python3
from __future__ import annotations

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
from typing import Dict, List, Optional

import pandas as pd

from docling_eval.campaign_tools.combine_cvat_evaluations import (
    combine_cvat_evaluations,
)
from docling_eval.campaign_tools.evaluate_cvat_tables import evaluate_tables
from docling_eval.campaign_tools.merge_cvat_annotations import (
    create_merged_annotation_xml,
    extract_image_tags,
)
from docling_eval.cli.main import evaluate
from docling_eval.cvat_tools.cvat_to_docling import convert_cvat_folder_to_docling
from docling_eval.cvat_tools.folder_models import CVATFolderStructure
from docling_eval.cvat_tools.folder_parser import (
    find_xml_files_by_pattern,
    parse_cvat_folder,
)
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


GROUND_TRUTH_PATTERN: str = "task_{xx}_set_A"
PREDICTION_PATTERN: str = "task_{xx}_set_B"


class CVATEvaluationPipeline:
    """Pipeline for CVAT annotation evaluation."""

    def __init__(
        self,
        cvat_root: Path,
        output_dir: Path,
        *,
        strict: bool = False,
        tasks_root: Optional[Path] = None,
        force_ocr: bool = False,
        ocr_scale: float = 1.0,
    ):
        """
        Initialize the pipeline.

        Args:
            cvat_root: Root directory of the ``cvat_dataset_preannotated`` export
            output_dir: Base directory for all pipeline outputs
            strict: If True, treat conversion failures as fatal (default: False)
            tasks_root: Optional override directory containing ``cvat_tasks`` XMLs
            force_ocr: If True, force OCR on PDF page images instead of using native text layer (default: False)
            ocr_scale: Scale factor for rendering PDFs for OCR (default: 1.0 = 72 DPI).
                      Higher values increase OCR resolution. Coordinates are mapped back to 144 DPI.
        """
        self.cvat_root = Path(cvat_root)
        self.output_dir = Path(output_dir)
        self.strict = strict
        self.tasks_root = Path(tasks_root).resolve() if tasks_root else None
        self.force_ocr = force_ocr
        self.ocr_scale = ocr_scale
        self._folder_cache: Dict[str, CVATFolderStructure] = {}

        # Create subdirectories
        self.gt_json_dir = self.output_dir / "ground_truth_json"
        self.pred_json_dir = self.output_dir / "predictions_json"
        self.gt_dataset_dir = self.output_dir / "gt_dataset"
        self.eval_dataset_dir = self.output_dir / "eval_dataset"
        self.evaluation_results_dir = self.output_dir / "evaluation_results"
        self._intermediate_dir = self.output_dir / "intermediate"

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_folder_structure(self, xml_pattern: str) -> CVATFolderStructure:
        """Load and cache the CVAT folder structure for the provided pattern."""

        if xml_pattern in self._folder_cache:
            return self._folder_cache[xml_pattern]

        folder_structure = parse_cvat_folder(
            self.cvat_root,
            xml_pattern,
            tasks_root=self.tasks_root,
        )
        self._folder_cache[xml_pattern] = folder_structure
        return folder_structure

    def _convert_cvat_set_to_json(
        self,
        output_json_dir: Path,
        xml_pattern: str,
    ) -> List[Path]:
        """Convert all documents covered by ``xml_pattern`` into Docling JSON files."""

        folder_structure = self._load_folder_structure(xml_pattern)

        if output_json_dir.exists():
            for stale_json in output_json_dir.glob("*.json"):
                stale_json.unlink()
        output_json_dir.mkdir(parents=True, exist_ok=True)

        _log.info(
            "Converting %d document(s) matching %s",
            len(folder_structure.documents),
            xml_pattern,
        )

        results = convert_cvat_folder_to_docling(
            folder_path=self.cvat_root,
            xml_pattern=xml_pattern,
            output_dir=output_json_dir,
            save_formats=["json"],
            folder_structure=folder_structure,
            log_validation=self.strict,
            force_ocr=self.force_ocr,
            ocr_scale=self.ocr_scale,
        )

        json_files: List[Path] = []
        failed_docs: List[str] = []

        for doc_hash, result in results.items():
            cvat_doc = folder_structure.documents[doc_hash]
            json_path = output_json_dir / f"{cvat_doc.doc_name}.json"

            if result.document is None or not json_path.exists():
                failed_docs.append(cvat_doc.doc_name)
                if json_path.exists():
                    json_path.unlink()
                continue

            json_files.append(json_path)

        if failed_docs:
            _log.warning(
                "Skipped %d document(s) due to conversion issues: %s",
                len(failed_docs),
                ", ".join(sorted(failed_docs)[:5])
                + ("..." if len(failed_docs) > 5 else ""),
            )

        if self.strict and failed_docs:
            raise ValueError(
                "Strict mode enabled: conversion errors were encountered while converting documents."
            )

        json_files.sort()
        _log.info(
            "Converted %d/%d document(s) to JSON format",
            len(json_files),
            len(folder_structure.documents),
        )
        return json_files

    def _merge_task_xmls(
        self,
        xml_pattern: str,
        destination: Path,
    ) -> Path:
        """Merge all CVAT task XMLs for ``xml_pattern`` into a single file."""

        folder_structure = self._load_folder_structure(xml_pattern)
        xml_files = find_xml_files_by_pattern(folder_structure.tasks_dir, xml_pattern)
        if not xml_files:
            raise ValueError(
                f"No XML files matching pattern '{xml_pattern}' found in {folder_structure.tasks_dir}"
            )

        _log.info(
            "Merging %d CVAT task XMLs matching '%s'", len(xml_files), xml_pattern
        )

        image_elements = extract_image_tags(xml_files)
        if not image_elements:
            raise ValueError(
                f"No annotated images discovered while merging pattern '{xml_pattern}'"
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        create_merged_annotation_xml(image_elements, destination)
        _log.info("✓ Generated merged annotations at %s", destination)
        return destination

    def merge_annotation_xmls(
        self, destination_dir: Optional[Path] = None
    ) -> tuple[Path, Path]:
        """Merge all CVAT task XMLs for ground-truth and prediction sets."""

        if destination_dir is None:
            destination_dir = self._intermediate_dir / "merged_xml"

        destination_dir.mkdir(parents=True, exist_ok=True)

        gt_path = destination_dir / "combined_set_A.xml"
        pred_path = destination_dir / "combined_set_B.xml"

        gt_xml = self._merge_task_xmls(GROUND_TRUTH_PATTERN, gt_path)
        pred_xml = self._merge_task_xmls(PREDICTION_PATTERN, pred_path)

        return gt_xml, pred_xml

    def create_ground_truth_dataset(self) -> None:
        """
        Step 1: Create ground truth dataset from CVAT folder exports.
        """
        _log.info("=== Creating Ground Truth Dataset ===")

        # Convert CVAT XML to JSON
        gt_json_files = self._convert_cvat_set_to_json(
            self.gt_json_dir,
            GROUND_TRUTH_PATTERN,
        )

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

    def create_prediction_dataset(self) -> None:
        """
        Step 2: Create prediction dataset from CVAT folder exports using the ground truth dataset.
        """
        _log.info("=== Creating Prediction Dataset ===")

        if not self.gt_dataset_dir.exists():
            raise ValueError(
                f"Ground truth dataset not found at {self.gt_dataset_dir}. "
                "Please run create_ground_truth_dataset first."
            )

        # Convert prediction CVAT XML to JSON
        pred_json_files = self._convert_cvat_set_to_json(
            self.pred_json_dir,
            PREDICTION_PATTERN,
        )

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

    def run_table_evaluation(
        self,
        out_json: Optional[Path] = None,
        containment_thresh: float = 0.50,
        table_pair_iou: float = 0.20,
        sem_match_iou: float = 0.30,
        *,
        reuse_existing: bool = True,
    ) -> Path:
        """Run the table structure/semantics evaluation using merged CVAT task XMLs.

        When ``reuse_existing`` is ``True`` the method will reuse the previously merged
        ``combined_set_A/B.xml`` if present instead of re-parsing the CVAT exports.

        Writes a JSON file (default: evaluation_results/evaluation_CVAT_tables.json) and returns its path.
        """
        _log.info("=== Running Table Evaluation ===")

        if out_json is None:
            out_json = self.evaluation_results_dir / "evaluation_CVAT_tables.json"

        self.evaluation_results_dir.mkdir(parents=True, exist_ok=True)

        if reuse_existing and out_json.exists():
            _log.info("Reusing existing tables evaluation at %s", out_json)
            return out_json

        merged_dir = self._intermediate_dir / "merged_xml"
        gt_xml_path = merged_dir / "combined_set_A.xml"
        pred_xml_path = merged_dir / "combined_set_B.xml"

        if reuse_existing and gt_xml_path.exists() and pred_xml_path.exists():
            _log.info("Reusing existing merged table annotations at %s", merged_dir)
            gt_xml, pred_xml = gt_xml_path, pred_xml_path
        else:
            gt_xml, pred_xml = self.merge_annotation_xmls(destination_dir=merged_dir)

        result = evaluate_tables(
            set_a=gt_xml,
            set_b=pred_xml,
            containment_thresh=containment_thresh,
            table_pair_iou=table_pair_iou,
            sem_match_iou=sem_match_iou,
        )

        out_json.write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _log.info(f"✓ Tables evaluation written to: {out_json}")
        return out_json

    def run_evaluation(
        self,
        modalities: Optional[List[str]] = None,
        user_csv: Optional[Path] = None,
        *,
        subset_label: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
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

        overview_path = self.cvat_root / "cvat_overview.json"
        overview_for_eval = overview_path if overview_path.exists() else None

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
                    cvat_overview_path=overview_for_eval,
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
        key_value_json = self.evaluation_results_dir / "evaluation_CVAT_key_value.json"
        tables_json = self.evaluation_results_dir / "evaluation_CVAT_tables.json"
        _log.info(f"Combining evaluation results to {combined_out}")
        combined_df = combine_cvat_evaluations(
            layout_json=layout_json,
            docstruct_json=docstruct_json,
            keyvalue_json=key_value_json,
            user_csv=user_csv,
            tables_json=tables_json,
            out=combined_out,
            cvat_overview_path=overview_for_eval,
        )
        if subset_label is not None:
            combined_df = combined_df.copy()
            if "subset" not in combined_df.columns:
                combined_df.insert(0, "subset", subset_label)
            else:
                combined_df["subset"] = subset_label

        return combined_df

    def run_full_pipeline(
        self,
        modalities: Optional[List[str]] = None,
        user_csv: Optional[Path] = None,
    ) -> None:
        """
        Run the complete pipeline: create datasets, run evaluation, and combine results.

        Args:
            modalities: List of evaluation modalities to run
            user_csv: Path to user CSV file for provenance/self-confidence
        """
        _log.info("=== Running Full CVAT Evaluation Pipeline ===")

        try:
            self.create_ground_truth_dataset()
            self.create_prediction_dataset()
            self.run_table_evaluation()
            self.run_evaluation(modalities, user_csv)

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
        "cvat_root",
        type=Path,
        help="Path to the cvat_dataset_preannotated root directory",
    )

    parser.add_argument(
        "output_dir", type=Path, help="Output directory for pipeline results"
    )

    parser.add_argument(
        "--tasks-root",
        type=Path,
        default=None,
        help="Optional path whose 'cvat_tasks' directory should override the default annotations",
    )

    parser.add_argument(
        "--user-csv",
        type=Path,
        default=None,
        help="Path to user CSV file for provenance/self-confidence (optional, used for combining evaluation results)",
    )

    parser.add_argument(
        "--step",
        choices=["gt", "pred", "tables", "eval", "full"],
        default="full",
        help="Pipeline step to run: gt (ground truth), pred (predictions), tables (table eval only), eval, or full.",
    )

    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=["layout", "document_structure", "key_value"],
        default=["layout", "document_structure", "key_value"],
        help="Evaluation modalities to run",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode: abort if any conversion fails (default: log and continue)",
    )

    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Force OCR on PDF page images instead of using native text layer",
    )

    parser.add_argument(
        "--ocr-scale",
        type=float,
        default=1.0,
        help="Scale for rendering PDFs for OCR (default: 1.0 = 72 DPI, 2.0 = 144 DPI, 3.0 = 216 DPI). "
        "Higher values may improve OCR accuracy. Coordinates are mapped back to 144 DPI to match CVAT annotations.",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.cvat_root.exists():
        _log.error(f"CVAT root directory does not exist: {args.cvat_root}")
        sys.exit(1)

    if not args.cvat_root.is_dir():
        _log.error(f"CVAT root path is not a directory: {args.cvat_root}")
        sys.exit(1)

    overview_path = args.cvat_root / "cvat_overview.json"
    if not overview_path.exists():
        _log.error(
            "cvat_overview.json not found in %s. Please point to a cvat_dataset_preannotated root.",
            args.cvat_root,
        )
        sys.exit(1)

    tasks_root = args.tasks_root
    if tasks_root is not None:
        if not tasks_root.exists():
            _log.error(f"tasks-root does not exist: {tasks_root}")
            sys.exit(1)
        if not tasks_root.is_dir():
            _log.error(f"tasks-root is not a directory: {tasks_root}")
            sys.exit(1)
        tasks_root = tasks_root.resolve()

    pipeline = CVATEvaluationPipeline(
        cvat_root=args.cvat_root,
        output_dir=args.output_dir,
        strict=args.strict,
        tasks_root=tasks_root,
        force_ocr=args.force_ocr,
        ocr_scale=args.ocr_scale,
    )

    if args.step == "gt":
        pipeline.create_ground_truth_dataset()
    elif args.step == "pred":
        pipeline.create_prediction_dataset()
    elif args.step == "tables":
        pipeline.run_table_evaluation()
    elif args.step == "eval":
        pipeline.run_evaluation(args.modalities, user_csv=args.user_csv)
    elif args.step == "full":
        pipeline.run_full_pipeline(args.modalities, user_csv=args.user_csv)


if __name__ == "__main__":
    main()
