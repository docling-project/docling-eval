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
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from docling_eval.cvat_tools.models import (
    CVATValidationError,
    CVATValidationReport,
    CVATValidationRunReport,
    ValidationSeverity,
)
from docling_eval.cvat_tools.parser import get_all_images_from_cvat_xml, parse_cvat_file
from docling_eval.cvat_tools.validator import Validator, validate_cvat_sample
from docling_eval.datamodels.types import (
    BenchMarkNames,
    EvaluationModality,
    PredictionFormats,
)
from docling_eval.utils.json_dataset_joiner import join_docling_json_datasets

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
_log = logging.getLogger(__name__)


GROUND_TRUTH_PATTERN: str = "task_{xx}_set_A"
PREDICTION_PATTERN: str = "task_{xx}_set_B"

# Mapping from modality name to enum
_MODALITY_MAP = {
    "layout": EvaluationModality.LAYOUT,
    "document_structure": EvaluationModality.DOCUMENT_STRUCTURE,
    "key_value": EvaluationModality.KEY_VALUE,
}


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
        storage_scale: float = 2.0,
        gt_json_dirname: str = "ground_truth_json",
        pred_json_dirname: str = "predictions_json",
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
                      Higher values increase OCR resolution. Coordinates are mapped back to storage_scale.
            storage_scale: Scale for stored page images and coordinates (default: 2.0 for 144 DPI).
            gt_json_dirname: Name of the subdirectory for ground truth JSON exports (default: "ground_truth_json")
            pred_json_dirname: Name of the subdirectory for prediction JSON exports (default: "predictions_json")
        """
        self.cvat_root = Path(cvat_root)
        self.output_dir = Path(output_dir)
        self.strict = strict
        self.tasks_root = Path(tasks_root).resolve() if tasks_root else None
        self.force_ocr = force_ocr
        self.ocr_scale = ocr_scale
        self.storage_scale = storage_scale
        self._folder_cache: Dict[str, CVATFolderStructure] = {}

        # Create subdirectories
        self.gt_json_dir = self.output_dir / gt_json_dirname
        self.pred_json_dir = self.output_dir / pred_json_dirname
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
        save_validation_report: bool = False,
        reuse_existing: bool = False,
    ) -> List[Path]:
        """Convert all documents covered by ``xml_pattern`` into Docling JSON files.

        Args:
            output_json_dir: Directory to save JSON files
            xml_pattern: Pattern to match XML files
            save_validation_report: If True, save validation report to output_dir
            reuse_existing: When True, reuse existing JSON files if present

        Returns:
            List of created JSON file paths
        """
        folder_structure = self._load_folder_structure(xml_pattern)

        if output_json_dir.exists():
            existing_files = sorted(output_json_dir.glob("*.json"))
            if reuse_existing and existing_files:
                _log.info(
                    "Reusing existing JSON exports in %s (%d files)",
                    output_json_dir,
                    len(existing_files),
                )
                return existing_files
            for stale_json in output_json_dir.glob("*.json"):
                stale_json.unlink()
        output_json_dir.mkdir(parents=True, exist_ok=True)

        set_label = "ground truth" if "set_A" in xml_pattern else "predictions"
        _log.info("=" * 60)
        _log.info(
            f"Converting {len(folder_structure.documents)} {set_label} document(s)..."
        )
        _log.info("=" * 60)

        results = convert_cvat_folder_to_docling(
            folder_path=self.cvat_root,
            xml_pattern=xml_pattern,
            output_dir=output_json_dir,
            save_formats=["json"],
            folder_structure=folder_structure,
            log_validation=self.strict,
            force_ocr=self.force_ocr,
            ocr_scale=self.ocr_scale,
            storage_scale=self.storage_scale,
        )

        json_files: List[Path] = []
        failed_docs: List[str] = []
        all_validation_reports: List[Any] = []

        for doc_hash, result in results.items():
            cvat_doc = folder_structure.documents[doc_hash]
            json_path = output_json_dir / f"{cvat_doc.doc_name}.json"

            # Collect validation reports from all pages
            all_validation_reports.extend(result.per_page_reports.values())

            # Check if conversion succeeded (JSON file exists and no fatal error)
            if result.error is not None or not json_path.exists():
                failed_docs.append(cvat_doc.doc_name)
                if json_path.exists():
                    json_path.unlink()
                continue

            json_files.append(json_path)

        # Save validation report if requested
        if save_validation_report:
            set_label = "set_A" if "set_A" in xml_pattern else "set_B"
            validation_report_path = (
                self.output_dir / f"validation_report_{set_label}.json"
            )
            run_report = CVATValidationRunReport(
                samples=all_validation_reports,
                statistics=CVATValidationRunReport.compute_statistics(
                    all_validation_reports
                ),
            )
            validation_report_path.write_text(
                run_report.model_dump_json(indent=2),
                encoding="utf-8",
            )
            _log.info(f"✓ Validation report saved to: {validation_report_path}")

        # Summary
        json_files.sort()
        success_count = len(json_files)
        total_count = len(folder_structure.documents)
        failed_count = len(failed_docs)

        _log.info("=" * 60)
        if failed_count == 0:
            _log.info(
                f"✓ SUCCESS: All {success_count} documents converted successfully"
            )
        else:
            _log.warning(
                f"⚠ PARTIAL: {success_count}/{total_count} documents converted ({failed_count} failed)"
            )
            _log.warning(
                f"  Failed documents: {', '.join(sorted(failed_docs)[:5])}"
                + ("..." if len(failed_docs) > 5 else "")
            )
        _log.info("=" * 60)

        if self.strict and failed_docs:
            raise ValueError(
                "Strict mode enabled: conversion errors were encountered while converting documents."
            )

        return json_files

    def convert_ground_truth_to_json(
        self, *, reuse_existing: bool = False
    ) -> List[Path]:
        """Convert ground truth annotations to Docling JSON files."""
        return self._convert_cvat_set_to_json(
            self.gt_json_dir,
            GROUND_TRUTH_PATTERN,
            save_validation_report=True,
            reuse_existing=reuse_existing,
        )

    def convert_predictions_to_json(
        self, *, reuse_existing: bool = False
    ) -> List[Path]:
        """Convert prediction annotations to Docling JSON files."""
        return self._convert_cvat_set_to_json(
            self.pred_json_dir,
            PREDICTION_PATTERN,
            save_validation_report=True,
            reuse_existing=reuse_existing,
        )

    def create_json_exports(
        self, *, reuse_existing: bool = False
    ) -> tuple[List[Path], List[Path]]:
        """Convert both ground truth and prediction annotations to JSON."""
        with ProcessPoolExecutor(
            max_workers=2, mp_context=get_context("spawn")
        ) as pool:
            gt_future = pool.submit(
                self._convert_cvat_set_to_json,
                self.gt_json_dir,
                GROUND_TRUTH_PATTERN,
                True,
                reuse_existing,
            )
            pred_future = pool.submit(
                self._convert_cvat_set_to_json,
                self.pred_json_dir,
                PREDICTION_PATTERN,
                True,
                reuse_existing,
            )

            gt_files = gt_future.result()
            pred_files = pred_future.result()
        return gt_files, pred_files

    def ensure_json_exports_exist(self) -> None:
        """Verify that JSON exports for both ground truth and predictions exist."""
        missing: list[str] = []
        if not self.gt_json_dir.exists() or not any(self.gt_json_dir.glob("*.json")):
            missing.append(str(self.gt_json_dir))
        if not self.pred_json_dir.exists() or not any(
            self.pred_json_dir.glob("*.json")
        ):
            missing.append(str(self.pred_json_dir))

        if missing:
            raise FileNotFoundError(
                "Missing JSON exports: "
                + ", ".join(missing)
                + ". Run the pipeline without --resume-from-json to generate them."
            )

    def create_eval_dataset_from_json(
        self,
        *,
        reuse_existing: bool = False,
        ignore_missing_predictions: bool = True,
        do_visualization: bool = True,
    ) -> None:
        """Join existing JSON exports into a single evaluation dataset."""
        if reuse_existing and self.eval_dataset_dir.exists():
            existing_parquet = list((self.eval_dataset_dir / "test").glob("*.parquet"))
            if existing_parquet:
                _log.info(
                    "Reusing existing evaluation dataset at %s (found %d shards)",
                    self.eval_dataset_dir,
                    len(existing_parquet),
                )
                return

        self.ensure_json_exports_exist()
        self.eval_dataset_dir.mkdir(parents=True, exist_ok=True)

        join_docling_json_datasets(
            gt_json_dir=self.gt_json_dir,
            prediction_json_dir=self.pred_json_dir,
            target_dataset_dir=self.eval_dataset_dir,
            name="CVAT_Eval_Dataset",
            split="test",
            chunk_size=50,
            prediction_format=PredictionFormats.JSON,
            predictor_info={
                "asset": "cvat_json_joiner",
                "source_path": str(self.pred_json_dir),
            },
            ignore_missing_predictions=ignore_missing_predictions,
            do_visualization=do_visualization,
        )

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

    def regenerate_validation_reports_from_merged(
        self,
        merged_dir: Optional[Path] = None,
    ) -> None:
        """Rebuild validation reports using pre-merged annotation XMLs."""

        if merged_dir is None:
            merged_dir = self.output_dir / "merged_xml"

        set_to_filename = {
            "set_A": merged_dir / "combined_set_A.xml",
            "set_B": merged_dir / "combined_set_B.xml",
        }

        validator = Validator()

        for set_label, xml_path in set_to_filename.items():
            if not xml_path.exists():
                raise FileNotFoundError(
                    f"Missing merged annotations for {set_label}: {xml_path}"
                )

            parsed_file = parse_cvat_file(xml_path)
            image_names = sorted(parsed_file.image_names)

            reports: list[CVATValidationReport] = []
            for image_name in image_names:
                try:
                    validated = validate_cvat_sample(
                        xml_path,
                        image_name,
                        validator=validator,
                        parsed_file=parsed_file,
                    )
                    reports.append(validated.report)
                except Exception as exc:  # noqa: BLE001
                    _log.error(
                        "Validation failed for %s (%s): %s",
                        set_label,
                        image_name,
                        exc,
                    )
                    reports.append(
                        CVATValidationReport(
                            sample_name=image_name,
                            errors=[
                                CVATValidationError(
                                    error_type="processing_error",
                                    message=f"Validation failed: {exc}",
                                    severity=ValidationSeverity.FATAL,
                                )
                            ],
                        )
                    )

            run_report = CVATValidationRunReport(
                samples=reports,
                statistics=CVATValidationRunReport.compute_statistics(reports),
            )

            output_path = self.output_dir / f"validation_report_{set_label}.json"
            output_path.write_text(
                run_report.model_dump_json(indent=2),
                encoding="utf-8",
            )
            _log.info(
                "✓ Regenerated %s validation report with %d sample(s): %s",
                set_label,
                len(reports),
                output_path,
            )

    def create_ground_truth_dataset(self, *, reuse_existing_json: bool = False) -> None:
        """
        Step 1: Export ground truth annotations to Docling JSON.
        """
        _log.info("")
        _log.info("╔" + "=" * 58 + "╗")
        _log.info("║" + " STEP 1: CREATE GROUND TRUTH DATASET ".center(58) + "║")
        _log.info("╚" + "=" * 58 + "╝")

        gt_json_files = self.convert_ground_truth_to_json(
            reuse_existing=reuse_existing_json
        )
        if not gt_json_files:
            raise ValueError("No ground truth JSON files were created")

        _log.info(
            "✓ Ground truth JSON exports available at %s (%d files)",
            self.gt_json_dir,
            len(gt_json_files),
        )

    def create_prediction_dataset(
        self,
        *,
        reuse_existing_json: bool = False,
        reuse_existing_eval: bool = False,
    ) -> None:
        """
        Step 2: Export prediction annotations and build the evaluation dataset.
        """
        _log.info("")
        _log.info("╔" + "=" * 58 + "╗")
        _log.info("║" + " STEP 2: CREATE PREDICTION DATASET ".center(58) + "║")
        _log.info("╚" + "=" * 58 + "╝")

        if not reuse_existing_json:
            self.convert_ground_truth_to_json(reuse_existing=True)

        pred_json_files = self.convert_predictions_to_json(
            reuse_existing=reuse_existing_json
        )
        if not pred_json_files:
            raise ValueError("No prediction JSON files were created")

        _log.info(
            "✓ Prediction JSON exports available at %s (%d files)",
            self.pred_json_dir,
            len(pred_json_files),
        )

        _log.info("Building evaluation dataset from JSON exports...")
        self.create_eval_dataset_from_json(
            reuse_existing=reuse_existing_eval,
            ignore_missing_predictions=True,
            do_visualization=True,
        )
        _log.info(f"✓ Evaluation dataset created: {self.eval_dataset_dir}")

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
        _log.info("")
        _log.info("╔" + "=" * 58 + "╗")
        _log.info("║" + " RUNNING TABLE EVALUATION ".center(58) + "║")
        _log.info("╚" + "=" * 58 + "╝")

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
        _log.info("")
        _log.info("╔" + "=" * 58 + "╗")
        _log.info("║" + " STEP 3: RUNNING EVALUATION ".center(58) + "║")
        _log.info("╚" + "=" * 58 + "╝")

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

        for idx, modality_name in enumerate(modalities, start=1):
            _log.info(
                f"[{idx}/{len(modalities)}] Running {modality_name} evaluation..."
            )

            modality = _MODALITY_MAP.get(modality_name)
            if modality is None:
                _log.warning(f"Unknown modality: {modality_name}. Skipping.")
                continue

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

        # Combine results
        _log.info("")
        _log.info("=" * 60)
        _log.info("Combining evaluation results...")
        combined_out = self.output_dir / "combined_evaluation.xlsx"

        def _result_path(name: str) -> Path:
            return self.evaluation_results_dir / f"evaluation_CVAT_{name}.json"

        combined_df = combine_cvat_evaluations(
            layout_json=_result_path("layout"),
            docstruct_json=_result_path("document_structure"),
            keyvalue_json=_result_path("key_value"),
            tables_json=_result_path("tables"),
            user_csv=user_csv,
            out=combined_out,
            cvat_overview_path=overview_for_eval,
        )

        _log.info(f"✓ Combined evaluation saved to: {combined_out}")
        _log.info("=" * 60)

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
        *,
        stop_after_json: bool = False,
        resume_from_json: bool = False,
        reuse_existing_eval: bool = False,
    ) -> None:
        """
        Run the complete pipeline: create datasets, run evaluation, and combine results.

        Args:
            modalities: List of evaluation modalities to run
            user_csv: Path to user CSV file for provenance/self-confidence
            stop_after_json: Stop the pipeline after JSON exports are prepared
            resume_from_json: Reuse existing JSON exports instead of reconverting
            reuse_existing_eval: Reuse eval dataset shards when present
        """
        _log.info("=== Running Full CVAT Evaluation Pipeline ===")

        try:
            if resume_from_json:
                _log.info("Resuming from existing JSON exports.")
                self.ensure_json_exports_exist()
            else:
                self.create_json_exports(reuse_existing=False)

            if stop_after_json:
                _log.info(
                    "Stop-after-json flag set; skipping dataset join and evaluations."
                )
                return

            self.create_eval_dataset_from_json(
                reuse_existing=reuse_existing_eval,
                ignore_missing_predictions=True,
                do_visualization=True,
            )
            self.run_table_evaluation(reuse_existing=False)
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
        choices=["gt", "pred", "json", "tables", "eval", "full"],
        default="full",
        help=(
            "Pipeline step to run: "
            "gt (export ground truth JSON), "
            "pred (export predictions and build evaluation dataset), "
            "json (export both ground truth and prediction JSON), "
            "tables (run table evaluation only), "
            "eval (run evaluation only), "
            "full (complete pipeline)"
        ),
    )

    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=["layout", "document_structure", "key_value"],
        default=["layout", "document_structure", "key_value"],
        help="Evaluation modalities to run (used with --step=eval or --step=full)",
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
        "Higher values may improve OCR accuracy. Coordinates are mapped back to storage_scale.",
    )

    parser.add_argument(
        "--storage-scale",
        type=float,
        default=2.0,
        help="Scale for stored page images and coordinates (default: 2.0 = 144 DPI).",
    )

    parser.add_argument(
        "--stop-after-json",
        action="store_true",
        help="Stop after generating ground_truth_json and predictions_json directories.",
    )

    parser.add_argument(
        "--resume-from-json",
        action="store_true",
        help="Reuse existing ground_truth_json and predictions_json directories.",
    )

    parser.add_argument(
        "--reuse-eval-dataset",
        action="store_true",
        help="Reuse the existing eval_dataset parquet shards when present.",
    )

    parser.add_argument(
        "--gt-json-dirname",
        type=str,
        default="ground_truth_json",
        help="Directory name for ground truth JSON exports (default: ground_truth_json).",
    )

    parser.add_argument(
        "--pred-json-dirname",
        type=str,
        default="predictions_json",
        help="Directory name for prediction JSON exports (default: predictions_json).",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input paths
    def _validate_dir(path: Path, name: str) -> None:
        if not path.exists():
            _log.error(f"{name} does not exist: {path}")
            sys.exit(1)
        if not path.is_dir():
            _log.error(f"{name} is not a directory: {path}")
            sys.exit(1)

    _validate_dir(args.cvat_root, "CVAT root directory")

    overview_path = args.cvat_root / "cvat_overview.json"
    if not overview_path.exists():
        _log.error(
            "cvat_overview.json not found in %s. Please point to a cvat_dataset_preannotated root.",
            args.cvat_root,
        )
        sys.exit(1)

    tasks_root = args.tasks_root
    if tasks_root is not None:
        _validate_dir(tasks_root, "tasks-root")
        tasks_root = tasks_root.resolve()

    # Initialize pipeline
    pipeline = CVATEvaluationPipeline(
        cvat_root=args.cvat_root,
        output_dir=args.output_dir,
        strict=args.strict,
        tasks_root=tasks_root,
        force_ocr=args.force_ocr,
        ocr_scale=args.ocr_scale,
        storage_scale=args.storage_scale,
        gt_json_dirname=args.gt_json_dirname,
        pred_json_dirname=args.pred_json_dirname,
    )

    # Execute requested pipeline step
    if args.resume_from_json and args.stop_after_json:
        _log.info(
            "Resume-from-json and stop-after-json both enabled; no new artefacts will be generated."
        )

    if args.step == "gt":
        pipeline.create_ground_truth_dataset(reuse_existing_json=args.resume_from_json)
    elif args.step == "json":
        pipeline.create_json_exports(reuse_existing=args.resume_from_json)
        return
    elif args.step == "pred":
        if args.stop_after_json:
            pipeline.create_json_exports(reuse_existing=args.resume_from_json)
            return
        pipeline.create_prediction_dataset(
            reuse_existing_json=args.resume_from_json,
            reuse_existing_eval=args.reuse_eval_dataset,
        )
    elif args.step == "tables":
        pipeline.run_table_evaluation(reuse_existing=False)
    elif args.step == "eval":
        if args.resume_from_json:
            pipeline.ensure_json_exports_exist()
        else:
            pipeline.create_json_exports(reuse_existing=False)
        if args.stop_after_json:
            return
        pipeline.create_eval_dataset_from_json(
            reuse_existing=args.reuse_eval_dataset,
            ignore_missing_predictions=True,
            do_visualization=True,
        )
        pipeline.run_table_evaluation(reuse_existing=True)
        pipeline.run_evaluation(args.modalities, user_csv=args.user_csv)
    elif args.step == "full":
        pipeline.run_full_pipeline(
            args.modalities,
            user_csv=args.user_csv,
            stop_after_json=args.stop_after_json,
            resume_from_json=args.resume_from_json,
            reuse_existing_eval=args.reuse_eval_dataset,
        )


if __name__ == "__main__":
    main()
