import glob
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

from datasets import Dataset, load_dataset
from pydantic import BaseModel
from tqdm import tqdm

from docling_eval.datamodels.dataset_record import DatasetRecordWithPrediction
from docling_eval.datamodels.types import PredictionFormats
from docling_eval.evaluators.base_evaluator import BaseEvaluator
from docling_eval.evaluators.ocr.pure_ocr_metrics import (
    calculate_aggregated_metrics,
    evaluate_single_pair,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


_log = logging.getLogger(__name__)


class PageOcrEvaluation(BaseModel):
    doc_id: str
    num_gt_original: float = 0.0
    num_pred_original: float = 0.0
    num_matches: float = 0.0
    num_tp: float = 0.0
    num_fp: float = 0.0
    num_fn: float = 0.0
    sum_edit_distance_intersection_sensitive: float = 0.0
    sum_max_length_intersection: float = 0.0
    word_hits_sensitive: float = 0.0
    word_substitutions_sensitive: float = 0.0
    total_gt_words_in_tps: float = 0.0
    sum_edit_distance_intersection_ignored_chars: float = 0.0
    sum_max_length_intersection_ignored_chars: float = 0.0
    sum_edit_distance_intersection_ignored_singles: float = 0.0
    sum_max_length_intersection_ignored_singles: float = 0.0
    text_length_false_positives: float = 0.0
    text_length_false_negatives: float = 0.0
    text_length_fp_ignored_chars: float = 0.0
    text_length_fn_ignored_chars: float = 0.0
    text_length_fp_ignored_singles: float = 0.0
    text_length_fn_ignored_singles: float = 0.0
    num_fn_ignoring_chars_becomes_empty: float = 0.0
    num_fn_ignoring_singles_becomes_empty: float = 0.0
    words_in_gt: float = 0.0
    fn_zero_iou: float = 0.0
    fp_low_iou: float = 0.0
    fn_ambiguous: float = 0.0
    fp_ambiguous: float = 0.0
    fp_zero_iou: float = 0.0
    model_merged: float = 0.0
    gt_merged: float = 0.0
    fn_low_iou: float = 0.0


class DatasetOcrEvaluation(BaseModel):
    evaluations: List[PageOcrEvaluation] = []

    f1_score: float = 0.0
    recall: float = 0.0
    precision: float = 0.0
    norm_ed_tp_only: float = 0.0
    norm_ed_without_fp: float = 0.0
    norm_ed_all_cells: float = 0.0
    norm_ed_ignoring_fp_and_chars: float = 0.0
    norm_ed_ignoring_chars: float = 0.0
    norm_ed_ignoring_singles_and_fp: float = 0.0
    norm_ed_ignoring_singles: float = 0.0
    word_accuracy_tp_only: float = 0.0
    word_accuracy_all_cells: float = 0.0
    edit_score_tp_only: float = 0.0
    edit_score_all_cells: float = 0.0
    hmean_norm_ed_without_fp_precision: float = 0.0

    total_word_hits: int = 0
    total_word_substitutions: int = 0
    total_fn: int = 0
    total_fp: int = 0
    total_fn_ignoring_chars: int = 0
    total_fn_ignoring_singles: int = 0
    total_words_in_gt: int = 0
    total_fn_zero_iou: int = 0
    total_fp_low_iou: int = 0
    total_fn_ambiguous_match: int = 0
    total_fp_ambiguous_match: int = 0
    total_fp_zero_iou: int = 0
    total_model_boxes_merged: int = 0
    total_gt_boxes_merged: int = 0
    total_fn_low_iou: int = 0

    raw_sum_num_tp_gt_based: int = 0
    raw_sum_num_fp: int = 0
    raw_sum_num_fn: int = 0
    raw_sum_num_gt_original: int = 0
    raw_sum_num_pred_original: int = 0
    raw_sum_num_matches: int = 0
    raw_sum_word_hits_sensitive: int = 0
    raw_sum_word_substitutions_sensitive: int = 0
    raw_sum_edit_distance_intersection_sensitive: float = 0.0
    raw_sum_max_length_intersection: float = 0.0
    raw_sum_text_length_false_positives: float = 0.0
    raw_sum_text_length_false_negatives: float = 0.0


class OCREvaluator(BaseEvaluator):
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
            return DatasetOcrEvaluation()

        _log.info("Found %d files: %s", len(split_files), split_files)
        ds = load_dataset("parquet", data_files={split: split_files})
        _log.info("Overview of dataset: %s", ds)

        ds_selection: Dataset = ds[split]

        page_evaluations_list: List[PageOcrEvaluation] = []
        dataset_sum_of_doc_results: defaultdict[str, float] = defaultdict(float)
        total_processed_page_pairs = 0
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

            true_segpages = data_record.ground_truth_segmented_pages
            pred_segpages = data_record.predicted_segmented_pages

            current_doc_aggregated_results: defaultdict[str, float] = defaultdict(float)
            doc_pages_processed_count = 0

            for page_no, true_seg_page in true_segpages.items():
                if page_no in pred_segpages:
                    pred_seg_page = pred_segpages[page_no]
                    page_pair_raw_results = evaluate_single_pair(
                        true_seg_page, pred_seg_page
                    )
                    for key, value in page_pair_raw_results.items():
                        current_doc_aggregated_results[key] += value
                    doc_pages_processed_count += 1
                else:
                    _log.warning(
                        f"Page {page_no} for doc {doc_id} not in predictions. Skipping page evaluation."
                    )

            if doc_pages_processed_count > 0:
                doc_eval_data = {
                    key.lstrip("_"): value
                    for key, value in current_doc_aggregated_results.items()
                }

                try:
                    doc_evaluation = PageOcrEvaluation(doc_id=doc_id, **doc_eval_data)
                    page_evaluations_list.append(doc_evaluation)
                    num_valid_records += 1

                    for key, value in current_doc_aggregated_results.items():
                        dataset_sum_of_doc_results[key] += value
                    total_processed_page_pairs += doc_pages_processed_count

                    if self._intermediate_evaluations_path:
                        self.save_intermediate_evaluations(
                            evaluation_name="ocr_eval",
                            enunumerate_id=i,
                            doc_id=doc_id,
                            evaluations=[doc_evaluation.model_dump()],
                        )
                except Exception as e:
                    _log.error(
                        "Failed to create or process PageOcrEvaluation for doc_id=%s: %s. Data: %s",
                        doc_id,
                        e,
                        doc_eval_data,
                    )
                    continue
            else:
                _log.warning(
                    f"No page pairs processed for document {doc_id}. Skipping document aggregation."
                )

        _log.info(
            f"Processed {num_valid_records} valid records (documents with at least one page pair) out of {len(ds_selection)}."
        )
        _log.info(f"Total page pairs processed: {total_processed_page_pairs}.")

        if total_processed_page_pairs == 0:
            _log.warning(
                "No page pairs processed across the dataset. Returning empty evaluation."
            )
            return DatasetOcrEvaluation()

        final_metrics_dict = calculate_aggregated_metrics(
            dataset_sum_of_doc_results, total_processed_page_pairs
        )

        dataset_eval_data = {}
        key_map_final_to_model = {
            "F1": "f1_score",
            "Recall": "recall",
            "Precision": "precision",
            "Norm_ED (TP-Only)": "norm_ed_tp_only",
            "Norm_ED (Without FP)": "norm_ed_without_fp",
            "Norm_ED (All-cells)": "norm_ed_all_cells",
            "Norm_ED (Ignoring FP and some chars)": "norm_ed_ignoring_fp_and_chars",
            "Norm_ED (Ignoring some chars)": "norm_ed_ignoring_chars",
            "Norm_ED (Ignoring single chars and FP)": "norm_ed_ignoring_singles_and_fp",
            "Norm_ED (Ignoring single chars)": "norm_ed_ignoring_singles",
            "Word-accuracy (TP-Only)": "word_accuracy_tp_only",
            "Word-accuracy (All-cells)": "word_accuracy_all_cells",
            "Edit-score (TP-Only)": "edit_score_tp_only",
            "Edit-score (All-cells)": "edit_score_all_cells",
            "Hmean: (Norm_ED [Without FP], precision)": "hmean_norm_ed_without_fp_precision",
            "#Word-Hits": "total_word_hits",
            "#Word-Substitutions": "total_word_substitutions",
            "#FN": "total_fn",
            "#FP": "total_fp",
            "#FN (Ignoring some chars)": "total_fn_ignoring_chars",
            "#FN (Ignoring single chars)": "total_fn_ignoring_singles",
            "#words-in-GT": "total_words_in_gt",
            "#FN - zero iou": "total_fn_zero_iou",
            "#FP - low iou": "total_fp_low_iou",
            "#FN - ambiguous match": "total_fn_ambiguous_match",
            "#FP - ambiguous match": "total_fp_ambiguous_match",
            "#FP - zero iou": "total_fp_zero_iou",
            "#Original Model boxes merged": "total_model_boxes_merged",
            "#Original GT boxes merged": "total_gt_boxes_merged",
            "#FN - low iou": "total_fn_low_iou",
            "_num_tp_gt_based": "raw_sum_num_tp_gt_based",
            "_num_fp": "raw_sum_num_fp",
            "_num_fn": "raw_sum_num_fn",
            "_num_gt_original": "raw_sum_num_gt_original",
            "_num_pred_original": "raw_sum_num_pred_original",
            "_num_matches": "raw_sum_num_matches",
            "_word_hits_sensitive": "raw_sum_word_hits_sensitive",
            "_word_substitutions_sensitive": "raw_sum_word_substitutions_sensitive",
            "_sum_edit_distance_intersection_sensitive": "raw_sum_edit_distance_intersection_sensitive",
            "_sum_max_length_intersection": "raw_sum_max_length_intersection",
            "_text_length_false_positives": "raw_sum_text_length_false_positives",
            "_text_length_false_negatives": "raw_sum_text_length_false_negatives",
        }

        for k_final, k_model in key_map_final_to_model.items():
            if k_final in final_metrics_dict:
                dataset_eval_data[k_model] = final_metrics_dict[k_final]
            else:
                _log.warning(
                    f"Key '{k_final}' not found in final_metrics_dict. Field '{k_model}' will use default."
                )

        dataset_evaluation_result = DatasetOcrEvaluation(
            evaluations=page_evaluations_list, **dataset_eval_data
        )

        _log.info(f"Final F1 Score: {dataset_evaluation_result.f1_score:.4f}")
        _log.info(f"Final Precision: {dataset_evaluation_result.precision:.4f}")
        _log.info(f"Final Recall: {dataset_evaluation_result.recall:.4f}")
        _log.info(
            f"Final Norm_ED (All-cells): {dataset_evaluation_result.norm_ed_all_cells:.4f}"
        )
        _log.info(f"Total Word Hits: {dataset_evaluation_result.total_word_hits}")

        return dataset_evaluation_result