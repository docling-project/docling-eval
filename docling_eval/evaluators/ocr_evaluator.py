import glob
import json
import logging
import os
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

import evaluate
import pandas as pd
from datasets import Dataset, load_dataset
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import DoclingDocument
from Levenshtein import distance as levenshtein_distance
from pydantic import BaseModel
from tqdm import tqdm

from docling_eval.datamodels.dataset_record import DatasetRecordWithPrediction
from docling_eval.datamodels.types import BenchMarkColumns, PredictionFormats
from docling_eval.evaluators.base_evaluator import BaseEvaluator
from docling_eval.evaluators.ocr.OcrPerformanceCalculator import OCRPerformanceEvaluator

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


_log = logging.getLogger(__name__)


class PageOcrEvaluation(BaseModel):
    doc_id: str
    true_text: str
    pred_text: str
    cer: Optional[float] = None  # Character Error Rate
    char_accuracy: Optional[float] = None  # Character Accuracy

    # Detection Metrics
    num_detected_cells: int
    num_gt_cells: int
    num_fp_detections: int
    num_tp_detections: int
    num_fn_detections: int
    detection_precision: float
    detection_recall: float
    detection_f1_score: float

    norm_ed_tp_only: float
    norm_ed_all_cells: float
    fn_without_ignored_chars: int
    fn_without_single_chars: int
    word_acc_intersect_cs: float
    word_acc_intersect_ci: float
    word_acc_union_cs: float
    word_acc_union_ci: float
    edit_score_intersect_cs: float
    edit_score_intersect_ci: float
    edit_score_union_cs: float
    edit_score_union_ci: float
    sum_edit_dist_intersect_cs: float
    sum_edit_dist_intersect_ci: float
    sum_max_len_intersection: float
    text_len_fp: int
    text_len_fn: int
    sum_norm_ed: float
    word_insertions: int
    word_deletions: int
    word_substitutions_cs: int
    word_hits_cs: int
    word_substitutions_ci: int
    word_hits_ci: int
    orientation_accuracy: float

    # Box/Refinement Metrics
    num_model_boxes_no_intersect: int
    num_gt_boxes_no_intersect: int
    num_gt_boxes_fn_refined: int
    num_model_boxes_fp_refined: int
    num_gt_boxes_low_iou: int
    num_model_boxes_low_iou: int
    model_boxes_merged: int
    gt_boxes_merged: int


class DatasetOcrEvaluation(BaseModel):
    evaluations: List[PageOcrEvaluation]
    mean_cer: Optional[float] = None
    mean_char_accuracy: Optional[float] = None

    total_detected_cells: int
    total_gt_cells: int
    total_fp_detections: int
    total_tp_detections: int
    total_fn_detections: int
    mean_detection_precision: float
    mean_detection_recall: float
    mean_detection_f1_score: float

    mean_norm_ed_tp_only: float
    mean_norm_ed_all_cells: float
    total_fn_without_ignored_chars: int
    total_fn_without_single_chars: int
    mean_word_acc_intersect_cs: float
    mean_word_acc_intersect_ci: float
    mean_word_acc_union_cs: float
    mean_word_acc_union_ci: float
    mean_edit_score_intersect_cs: float
    mean_edit_score_intersect_ci: float
    mean_edit_score_union_cs: float
    mean_edit_score_union_ci: float
    total_sum_edit_dist_intersect_cs: float
    total_sum_edit_dist_intersect_ci: float
    total_sum_max_len_intersection: float
    total_text_len_fp: int
    total_text_len_fn: int
    total_sum_norm_ed: float  # Sum of per-page sums
    total_word_insertions: int
    total_word_deletions: int
    total_word_substitutions_cs: int
    total_word_hits_cs: int
    total_word_substitutions_ci: int
    total_word_hits_ci: int
    mean_orientation_accuracy: float

    # Aggregated Box/Refinement Metrics
    total_model_boxes_no_intersect: int
    total_gt_boxes_no_intersect: int
    total_gt_boxes_fn_refined: int
    total_model_boxes_fp_refined: int
    total_gt_boxes_low_iou: int
    total_model_boxes_low_iou: int
    total_model_boxes_merged: int
    total_gt_boxes_merged: int


class OCREvaluator(BaseEvaluator):
    """Evaluator for OCR tasks that computes Character Accuracy"""

    def __init__(
        self,
        intermediate_evaluations_path: Optional[Path] = None,
        prediction_sources: List[PredictionFormats] = [
            PredictionFormats.DOCLING_DOCUMENT
        ],
        iou_threshold: float = 0.5,
    ):
        super().__init__(
            intermediate_evaluations_path=intermediate_evaluations_path,
            prediction_sources=prediction_sources,
            supported_prediction_formats=[PredictionFormats.DOCLING_DOCUMENT],
        )
        # Load the CER evaluation metric
        # https://huggingface.co/spaces/evaluate-metric/cer
        self._cer_eval = evaluate.load("cer")
        self.iou_threshold = iou_threshold

    def __call__(
        self,
        ds_path: Path,
        split: str = "test",
    ) -> DatasetOcrEvaluation:

        _log.info("Loading the split '%s' from: '%s'", split, ds_path)
        split_path = str(ds_path / split / "*.parquet")
        split_files = glob.glob(split_path)
        if not split_files:
            _log.warning(
                "No parquet files found for split '%s' in '%s'", split, ds_path
            )
            # Return empty evaluation
            empty_totals = {
                k: 0
                for k in DatasetOcrEvaluation.model_fields
                if k.startswith("total_")
            }
            empty_means = {
                k: 0.0
                for k in DatasetOcrEvaluation.model_fields
                if k.startswith("mean_") and k not in ["mean_cer", "mean_char_accuracy"]
            }
            empty_optionals = {"mean_cer": None, "mean_char_accuracy": None}
            return DatasetOcrEvaluation(
                evaluations=[], **empty_totals, **empty_means, **empty_optionals
            )

        _log.info("Found %d files: %s", len(split_files), split_files)
        ds = load_dataset("parquet", data_files={split: split_files})
        _log.info("Overview of dataset: %s", ds)

        ds_selection: Dataset = ds[split]

        page_evaluations_list: List[PageOcrEvaluation] = []

        total_detected_cells = 0
        total_gt_cells = 0
        total_fp_detections = 0
        total_tp_detections = 0
        total_fn_detections = 0
        total_fn_without_ignored_chars = 0
        total_fn_without_single_chars = 0
        total_sum_edit_dist_intersect_cs = 0.0
        total_sum_edit_dist_intersect_ci = 0.0
        total_sum_max_len_intersection = 0.0
        total_text_len_fp = 0
        total_text_len_fn = 0
        total_sum_norm_ed = 0.0
        total_word_insertions = 0
        total_word_deletions = 0
        total_word_substitutions_cs = 0
        total_word_hits_cs = 0
        total_word_substitutions_ci = 0
        total_word_hits_ci = 0
        total_model_boxes_no_intersect = 0
        total_gt_boxes_no_intersect = 0
        total_gt_boxes_fn_refined = 0
        total_model_boxes_fp_refined = 0
        total_gt_boxes_low_iou = 0
        total_model_boxes_low_iou = 0
        total_model_boxes_merged = 0
        total_gt_boxes_merged = 0

        metrics_for_mean = [
            "detection_precision",
            "detection_recall",
            "detection_f1_score",
            "norm_ed_tp_only",
            "norm_ed_all_cells",
            "word_acc_intersect_cs",
            "word_acc_intersect_ci",
            "word_acc_union_cs",
            "word_acc_union_ci",
            "edit_score_intersect_cs",
            "edit_score_intersect_ci",
            "edit_score_union_cs",
            "edit_score_union_ci",
            "orientation_accuracy",
            "cer",
            "char_accuracy",
        ]
        metric_lists = {key: [] for key in metrics_for_mean}

        num_valid_records = 0
        for i, data in tqdm(
            enumerate(ds_selection),
            desc="Evaluating OCR",
            ncols=120,
            total=len(ds_selection),
        ):
            try:
                data_record = DatasetRecordWithPrediction.model_validate(data)
            except Exception as e:
                _log.error("Failed to validate record %d: %s. Data: %s", i, e, data)
                continue

            doc_id = data_record.doc_id
            if data_record.status not in self._accepted_status:
                _log.warning(
                    "Skipping record %s due to status: %s", doc_id, data_record.status
                )
                continue

            true_doc = data_record.ground_truth_doc
            pred_doc = data_record.predicted_doc

            if not pred_doc:
                _log.warning("No prediction found for doc_id=%s. Skipping.", doc_id)
                continue
            if not true_doc:
                _log.warning("No ground truth found for doc_id=%s. Skipping.", doc_id)
                continue

            true_text_with_bboxes = self._extract_text_with_bboxes(true_doc)
            pred_text_with_bboxes = self._extract_text_with_bboxes(pred_doc)

            # matched_pairs, unmatched_true, unmatched_pred = self._match_text_regions(
            #     true_text_with_bboxes, pred_text_with_bboxes
            # )

            # evaluation_results = self._evaluate_matched_pairs(
            #     matched_pairs, unmatched_true, unmatched_pred
            # )

            true_text = " ".join([item["text"] for item in true_text_with_bboxes])
            pred_text = " ".join([item["text"] for item in pred_text_with_bboxes])

            try:
                evaluator = OCRPerformanceEvaluator(pred_doc=pred_doc, gt_doc=true_doc)
                evaluation_results = evaluator.calc_metrics()
            except Exception as e:
                _log.error("Error calculating metrics for doc_id=%s: %s", doc_id, e)
                continue
            # temp_dir = Path(
            #     "/Users/sami/Desktop/IBM/docling-eval/docling_eval/evaluators/ocr_evaluation_results"
            # )
            # temp_dir.mkdir(parents=True, exist_ok=True)
            # temp_file = temp_dir / f"{doc_id}_evaluation.json"
            # with temp_file.open("w") as f:
            #     json.dump(evaluation_results, f, indent=4)

            try:
                page_evaluation = PageOcrEvaluation(
                    doc_id=doc_id,
                    true_text=true_text,
                    pred_text=pred_text,
                    num_detected_cells=evaluation_results.get(
                        "number_of_detected_cells", 0
                    ),
                    num_gt_cells=evaluation_results.get("number_of_gt_cells", 0),
                    num_fp_detections=evaluation_results.get(
                        "number_of_false_positive_detections", 0
                    ),
                    norm_ed_tp_only=evaluation_results.get("Norm_ED (TP-Only)", 0.0),
                    norm_ed_all_cells=evaluation_results.get(
                        "Norm_ED (All-cells)", 0.0
                    ),
                    num_tp_detections=evaluation_results.get(
                        "number_of_true_positive_detections", 0
                    ),
                    num_fn_detections=evaluation_results.get(
                        "number_of_false_negative_detections", 0
                    ),
                    fn_without_ignored_chars=evaluation_results.get(
                        "without_ignored_chars_false_negatives", 0
                    ),
                    fn_without_single_chars=evaluation_results.get(
                        "without_single_chars_false_negatives", 0
                    ),
                    detection_precision=evaluation_results.get(
                        "detection_precision", 0.0
                    ),
                    detection_recall=evaluation_results.get("detection_recall", 0.0),
                    detection_f1_score=evaluation_results.get(
                        "detection_f1_score", 0.0
                    ),
                    word_acc_intersect_cs=evaluation_results.get(
                        "word_accuracy_intersection_case_sensitive", 0.0
                    ),
                    word_acc_intersect_ci=evaluation_results.get(
                        "word_accuracy_intersection_case_insensitive", 0.0
                    ),
                    word_acc_union_cs=evaluation_results.get(
                        "word_accuracy_union_case_sensitive", 0.0
                    ),
                    word_acc_union_ci=evaluation_results.get(
                        "word_accuracy_union_case_insensitive", 0.0
                    ),
                    edit_score_intersect_cs=evaluation_results.get(
                        "edit_score_intersection_case_sensitive_not_avg_over_words", 0.0
                    ),
                    edit_score_intersect_ci=evaluation_results.get(
                        "edit_score_intersection_case_insensitive_not_avg_over_words",
                        0.0,
                    ),
                    edit_score_union_cs=evaluation_results.get(
                        "edit_score_union_case_sensitive_not_avg_over_words", 0.0
                    ),
                    edit_score_union_ci=evaluation_results.get(
                        "edit_score_union_case_insensitive_not_avg_over_words", 0.0
                    ),
                    sum_edit_dist_intersect_cs=evaluation_results.get(
                        "sum_edit_distance_intersection_case_sensitive_not_avg_over_words",
                        0.0,
                    ),
                    sum_edit_dist_intersect_ci=evaluation_results.get(
                        "sum_edit_distance_intersection_case_insensitive_not_avg_over_words",
                        0.0,
                    ),
                    sum_max_len_intersection=evaluation_results.get(
                        "sum_max_length_intersection", 0.0
                    ),
                    text_len_fp=evaluation_results.get(
                        "text_length_false_positives", 0
                    ),
                    text_len_fn=evaluation_results.get(
                        "text_length_false_negatives", 0
                    ),
                    sum_norm_ed=evaluation_results.get("sum_norm_ed", 0.0),
                    word_insertions=evaluation_results.get("word_Insertions", 0),
                    word_deletions=evaluation_results.get("word_Deletions", 0),
                    word_substitutions_cs=evaluation_results.get(
                        "word_substitutions_case_sensitive", 0
                    ),
                    word_hits_cs=evaluation_results.get("word_Hits_case_sensitive", 0),
                    word_substitutions_ci=evaluation_results.get(
                        "word_substitutions_case_insensitive", 0
                    ),
                    word_hits_ci=evaluation_results.get(
                        "word_Hits_case_insensitive", 0
                    ),
                    num_model_boxes_no_intersect=evaluation_results.get(
                        "num_model_boxes_that_do_not_intersect_with_a_gt", 0
                    ),
                    num_gt_boxes_no_intersect=evaluation_results.get(
                        "num_gt_boxes_that_do_not_intersect_with_a_gt", 0
                    ),
                    num_gt_boxes_fn_refined=evaluation_results.get(
                        "num_gt_boxes_that_are_fn_after_refinement", 0
                    ),
                    num_model_boxes_fp_refined=evaluation_results.get(
                        "num_model_boxes_that_are_fp_after_refinement", 0
                    ),
                    num_gt_boxes_low_iou=evaluation_results.get(
                        "num_gt_boxes_with_low_iou", 0
                    ),
                    num_model_boxes_low_iou=evaluation_results.get(
                        "num_model_boxes_with_low_iou", 0
                    ),
                    model_boxes_merged=evaluation_results.get(
                        "model_boxes_that_were_merged", 0
                    ),
                    gt_boxes_merged=evaluation_results.get(
                        "gt_boxes_that_were_merged", 0
                    ),
                    orientation_accuracy=evaluation_results.get(
                        "orientation_accuracy", 0.0
                    ),
                    # cer=evaluation_results.get("CER"),
                    # char_accuracy=evaluation_results.get("Character-accuracy"),
                )

                page_evaluations_list.append(page_evaluation)
                num_valid_records += 1

                total_detected_cells += page_evaluation.num_detected_cells
                total_gt_cells += page_evaluation.num_gt_cells
                total_fp_detections += page_evaluation.num_fp_detections
                total_tp_detections += page_evaluation.num_tp_detections
                total_fn_detections += page_evaluation.num_fn_detections
                total_fn_without_ignored_chars += (
                    page_evaluation.fn_without_ignored_chars
                )
                total_fn_without_single_chars += page_evaluation.fn_without_single_chars
                total_sum_edit_dist_intersect_cs += (
                    page_evaluation.sum_edit_dist_intersect_cs
                )
                total_sum_edit_dist_intersect_ci += (
                    page_evaluation.sum_edit_dist_intersect_ci
                )
                total_sum_max_len_intersection += (
                    page_evaluation.sum_max_len_intersection
                )
                total_text_len_fp += page_evaluation.text_len_fp
                total_text_len_fn += page_evaluation.text_len_fn
                total_sum_norm_ed += page_evaluation.sum_norm_ed
                total_word_insertions += page_evaluation.word_insertions
                total_word_deletions += page_evaluation.word_deletions
                total_word_substitutions_cs += page_evaluation.word_substitutions_cs
                total_word_hits_cs += page_evaluation.word_hits_cs
                total_word_substitutions_ci += page_evaluation.word_substitutions_ci
                total_word_hits_ci += page_evaluation.word_hits_ci
                total_model_boxes_no_intersect += (
                    page_evaluation.num_model_boxes_no_intersect
                )
                total_gt_boxes_no_intersect += page_evaluation.num_gt_boxes_no_intersect
                total_gt_boxes_fn_refined += page_evaluation.num_gt_boxes_fn_refined
                total_model_boxes_fp_refined += (
                    page_evaluation.num_model_boxes_fp_refined
                )
                total_gt_boxes_low_iou += page_evaluation.num_gt_boxes_low_iou
                total_model_boxes_low_iou += page_evaluation.num_model_boxes_low_iou
                total_model_boxes_merged += page_evaluation.model_boxes_merged
                total_gt_boxes_merged += page_evaluation.gt_boxes_merged

                for key in metric_lists:
                    value = getattr(page_evaluation, key)
                    if (
                        value is not None
                    ):  # Handle optional metrics like CER/Char Accuracy
                        metric_lists[key].append(value)

                if self._intermediate_evaluations_path:
                    self.save_intermediate_evaluations(
                        evaluation_name="ocr_eval",
                        enumerate_id=i,
                        doc_id=doc_id,
                        evaluations=[page_evaluation.model_dump()],
                    )
            except Exception as e:
                _log.error(
                    "Failed to process or save PageOcrEvaluation for doc_id=%s: %s. Evaluation results: %s",
                    doc_id,
                    e,
                    evaluation_results,
                )
                continue

        _log.info(
            f"Processed {num_valid_records} valid records out of {len(ds_selection)}."
        )

        mean_metrics = {}
        for key in metric_lists:
            data_list = metric_lists[key]
            mean_key = f"mean_{key}"
            if data_list:
                mean_value = statistics.mean(data_list)
                mean_metrics[mean_key] = mean_value
            else:
                if key in ["cer", "char_accuracy"]:
                    mean_metrics[mean_key] = None
                else:
                    mean_metrics[mean_key] = 0.0

        _log.info(
            f"Mean Detection F1 Score: {mean_metrics.get('mean_detection_f1_score', 0.0):.4f}"
        )
        _log.info(
            f"Mean Norm ED (All Cells): {mean_metrics.get('mean_norm_ed_all_cells', 0.0):.4f}"
        )
        _log.info(
            f"Mean Word Accuracy (Union CI): {mean_metrics.get('mean_word_acc_union_ci', 0.0):.4f}"
        )
        if "mean_cer" in mean_metrics and mean_metrics["mean_cer"] is not None:
            _log.info(f"Mean CER: {mean_metrics['mean_cer']:.4f}")
        if (
            "mean_char_accuracy" in mean_metrics
            and mean_metrics["mean_char_accuracy"] is not None
        ):
            _log.info(
                f"Mean Character Accuracy: {mean_metrics['mean_char_accuracy']:.4f}"
            )
        _log.info(f"Total Detected Cells: {total_detected_cells}")

        return DatasetOcrEvaluation(
            evaluations=page_evaluations_list,
            total_detected_cells=total_detected_cells,
            total_gt_cells=total_gt_cells,
            total_fp_detections=total_fp_detections,
            total_tp_detections=total_tp_detections,
            total_fn_detections=total_fn_detections,
            total_fn_without_ignored_chars=total_fn_without_ignored_chars,
            total_fn_without_single_chars=total_fn_without_single_chars,
            total_sum_edit_dist_intersect_cs=total_sum_edit_dist_intersect_cs,
            total_sum_edit_dist_intersect_ci=total_sum_edit_dist_intersect_ci,
            total_sum_max_len_intersection=total_sum_max_len_intersection,
            total_text_len_fp=total_text_len_fp,
            total_text_len_fn=total_text_len_fn,
            total_sum_norm_ed=total_sum_norm_ed,
            total_word_insertions=total_word_insertions,
            total_word_deletions=total_word_deletions,
            total_word_substitutions_cs=total_word_substitutions_cs,
            total_word_hits_cs=total_word_hits_cs,
            total_word_substitutions_ci=total_word_substitutions_ci,
            total_word_hits_ci=total_word_hits_ci,
            total_model_boxes_no_intersect=total_model_boxes_no_intersect,
            total_gt_boxes_no_intersect=total_gt_boxes_no_intersect,
            total_gt_boxes_fn_refined=total_gt_boxes_fn_refined,
            total_model_boxes_fp_refined=total_model_boxes_fp_refined,
            total_gt_boxes_low_iou=total_gt_boxes_low_iou,
            total_model_boxes_low_iou=total_model_boxes_low_iou,
            total_model_boxes_merged=total_model_boxes_merged,
            total_gt_boxes_merged=total_gt_boxes_merged,
            **mean_metrics,
        )

    def _compute_cer_score(self, true_txt: str, pred_txt: str) -> float:
        result = self._cer_eval.compute(predictions=[pred_txt], references=[true_txt])
        return result

    def _compute_normalized_edit_distance(self, true_txt: str, pred_txt: str) -> float:
        if not true_txt:
            return 1.0 if not pred_txt else 0.0

        edit_dist = levenshtein_distance(pred_txt, true_txt)
        norm_ed = 1 - (edit_dist / max(len(true_txt), 1))

        return max(0.0, norm_ed)

    def _compute_word_metrics(self, true_txt: str, pred_txt: str) -> tuple:
        true_words = re.findall(r"\b\w+\b", true_txt.lower())
        pred_words = re.findall(r"\b\w+\b", pred_txt.lower())

        if not true_words:
            if not pred_words:
                return 1.0, 1.0, 1.0, 1.0, 0, 0, 0
            else:
                return 0.0, 0.0, 0.0, 0.0, 0, len(pred_words), 0

        true_words_copy = true_words.copy()
        pred_words_copy = pred_words.copy()
        matched_words = 0

        for word in true_words:
            if word in pred_words_copy:
                matched_words += 1
                pred_words_copy.remove(word)
                true_words_copy.remove(word)

        word_accuracy = matched_words / len(true_words)
        precision = matched_words / len(pred_words) if pred_words else 0.0
        recall = matched_words / len(true_words) if true_words else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return (
            word_accuracy,
            precision,
            recall,
            f1,
            matched_words,
            len(true_words_copy),
            len(pred_words_copy),
        )

    def _extract_text_with_bboxes(self, doc: DoclingDocument) -> List[Dict]:
        result = []

        for text_item in doc.texts:
            if not text_item.prov:
                continue

            for prov in text_item.prov:
                result.append(
                    {"text": text_item.text, "bbox": prov.bbox, "page_no": prov.page_no}
                )

        # for table_item in doc.tables:
        #     if not table_item.data or not table_item.data.table_cells:
        #         continue

        #     for cell in table_item.data.table_cells:
        #         if not cell.text or not cell.prov:
        #             continue

        #         for prov in cell.prov:
        #             result.append(
        #                 {"text": cell.text, "bbox": prov.bbox, "page_no": prov.page_no}
        #             )

        return result

    def _calculate_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        x_left = max(bbox1.l, bbox2.l)
        y_top = max(bbox1.t, bbox2.t)
        x_right = min(bbox1.r, bbox2.r)
        y_bottom = min(bbox1.b, bbox2.b)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        bbox1_area = (bbox1.r - bbox1.l) * (bbox1.b - bbox1.t)
        bbox2_area = (bbox2.r - bbox2.l) * (bbox2.b - bbox2.t)

        union_area = bbox1_area + bbox2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0

    def _match_text_regions(
        self, true_texts: List[Dict], pred_texts: List[Dict]
    ) -> Tuple[List, List, List]:
        matched_pairs = []
        unmatched_true = true_texts.copy()
        unmatched_pred = pred_texts.copy()

        for i, true_item in enumerate(true_texts):
            best_match_idx = -1
            best_iou = -1

            for j, pred_item in enumerate(pred_texts):
                if pred_item["page_no"] != true_item["page_no"]:
                    continue

                iou = self._calculate_iou(true_item["bbox"], pred_item["bbox"])

                if iou > self.iou_threshold and iou > best_iou:
                    best_match_idx = j
                    best_iou = iou

            if best_match_idx >= 0:
                matched_pairs.append((true_item, pred_texts[best_match_idx]))
                if true_item in unmatched_true:
                    unmatched_true.remove(true_item)
                if pred_texts[best_match_idx] in unmatched_pred:
                    unmatched_pred.remove(pred_texts[best_match_idx])

        return matched_pairs, unmatched_true, unmatched_pred

    def _evaluate_matched_pairs(
        self, matched_pairs: List, unmatched_true: List, unmatched_pred: List
    ) -> Dict:
        all_true_text = " ".join(
            [item["text"] for item in matched_pairs + unmatched_true]
        )
        all_pred_text = " ".join(
            [item["text"] for item in matched_pairs + unmatched_pred]
        )

        tp_only_true_text = " ".join([pair[0]["text"] for pair in matched_pairs])
        tp_only_pred_text = " ".join([pair[1]["text"] for pair in matched_pairs])

        without_fp_true_text = " ".join(
            [item["text"] for item in matched_pairs + unmatched_true]
        )
        without_fp_pred_text = " ".join([pair[1]["text"] for pair in matched_pairs])

        cer_all = self._compute_cer_score(all_true_text, all_pred_text)
        cer_tp_only = (
            self._compute_cer_score(tp_only_true_text, tp_only_pred_text)
            if tp_only_true_text
            else 1.0
        )

        norm_ed_all = self._compute_normalized_edit_distance(
            all_true_text, all_pred_text
        )
        norm_ed_tp_only = self._compute_normalized_edit_distance(
            tp_only_true_text, tp_only_pred_text
        )
        norm_ed_without_fp = self._compute_normalized_edit_distance(
            without_fp_true_text, without_fp_pred_text
        )

        word_acc_all, precision_all, recall_all, f1_all, word_hits, fn, fp = (
            self._compute_word_metrics(all_true_text, all_pred_text)
        )

        word_acc_tp, _, _, _, _, _, _ = self._compute_word_metrics(
            tp_only_true_text, tp_only_pred_text
        )

        word_subst = min(fn, fp)

        results = {
            "F1": f1_all,
            "Precision": precision_all,
            "Recall": recall_all,
            "Norm_ED (All-cells)": norm_ed_all,
            "Edit-score (All-cells)": norm_ed_all,
            "Word-accuracy (All-cells)": word_acc_all,
            "Norm_ED (TP-Only)": norm_ed_tp_only,
            "Norm_ED (Without FP)": norm_ed_without_fp,
            "Word-accuracy (TP-Only)": word_acc_tp,
            "Edit-score (TP-Only)": norm_ed_tp_only,
            "#Word-Hits": word_hits,
            "#Word-Substitutions": word_subst,
            "#FN": fn + len(unmatched_true),
            "#FP": fp + len(unmatched_pred),
            "CER": cer_all,
            "Character-accuracy": 1.0 - cer_all,
        }

        return results

    def evaluate_ocr(
        self, true_doc: DoclingDocument, pred_doc: DoclingDocument
    ) -> Dict:
        true_text_with_bboxes = self._extract_text_with_bboxes(true_doc)
        pred_text_with_bboxes = self._extract_text_with_bboxes(pred_doc)

        matched_pairs, unmatched_true, unmatched_pred = self._match_text_regions(
            true_text_with_bboxes, pred_text_with_bboxes
        )

        evaluation_results = self._evaluate_matched_pairs(
            matched_pairs, unmatched_true, unmatched_pred
        )

        return evaluation_results
