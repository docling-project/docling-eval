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

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


_log = logging.getLogger(__name__)


class PageOcrEvaluation(BaseModel):
    doc_id: str
    true_text: str
    pred_text: str
    cer: float
    char_accuracy: float
    norm_ed: float
    word_accuracy: float
    f1: float
    precision: float
    recall: float
    word_hits: int
    word_substitutions: int
    fn: int
    fp: int


class DatasetOcrEvaluation(BaseModel):
    evaluations: List[PageOcrEvaluation]
    mean_character_accuracy: float
    mean_norm_ed: float
    mean_word_accuracy: float
    mean_f1: float
    mean_precision: float
    mean_recall: float
    total_word_hits: int
    total_word_substitutions: int
    total_fn: int
    total_fp: int


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
        _log.info("Files: %s", split_files)
        ds = load_dataset("parquet", data_files={split: split_files})
        _log.info("Overview of dataset: %s", ds)

        ds_selection: Dataset = ds[split]

        text_evaluations_list = []
        char_accuracy_list = []
        norm_ed_list = []
        word_accuracy_list = []
        f1_list = []
        precision_list = []
        recall_list = []
        total_word_hits = 0
        total_word_substitutions = 0
        total_fn = 0
        total_fp = 0

        for i, data in tqdm(
            enumerate(ds_selection),
            desc="Evaluating OCR",
            ncols=120,
            total=len(ds_selection),
        ):
            data_record = DatasetRecordWithPrediction.model_validate(data)
            doc_id = data_record.doc_id
            if data_record.status not in self._accepted_status:
                _log.error(
                    "Skipping record without successful conversion status: %s", doc_id
                )
                continue

            true_doc = data_record.ground_truth_doc
            pred_doc = data_record.predicted_doc

            if not pred_doc:
                _log.error("There is no prediction for doc_id=%s", doc_id)
                continue

            true_text_with_bboxes = self._extract_text_with_bboxes(true_doc)
            pred_text_with_bboxes = self._extract_text_with_bboxes(pred_doc)

            matched_pairs, unmatched_true, unmatched_pred = self._match_text_regions(
                true_text_with_bboxes, pred_text_with_bboxes
            )

            evaluation_results = self._evaluate_matched_pairs(
                matched_pairs, unmatched_true, unmatched_pred
            )

            true_text = " ".join([item["text"] for item in true_text_with_bboxes])
            pred_text = " ".join([item["text"] for item in pred_text_with_bboxes])

            char_accuracy_list.append(evaluation_results["Character-accuracy"])
            norm_ed_list.append(evaluation_results["Norm_ED (All-cells)"])
            word_accuracy_list.append(evaluation_results["Word-accuracy (All-cells)"])
            f1_list.append(evaluation_results["F1"])
            precision_list.append(evaluation_results["Precision"])
            recall_list.append(evaluation_results["Recall"])

            total_word_hits += evaluation_results["#Word-Hits"]
            total_word_substitutions += evaluation_results["#Word-Substitutions"]
            total_fn += evaluation_results["#FN"]
            total_fp += evaluation_results["#FP"]

            page_evaluation = PageOcrEvaluation(
                doc_id=doc_id,
                true_text=true_text,
                pred_text=pred_text,
                cer=evaluation_results["CER"],
                char_accuracy=evaluation_results["Character-accuracy"],
                norm_ed=evaluation_results["Norm_ED (All-cells)"],
                word_accuracy=evaluation_results["Word-accuracy (All-cells)"],
                f1=evaluation_results["F1"],
                precision=evaluation_results["Precision"],
                recall=evaluation_results["Recall"],
                word_hits=evaluation_results["#Word-Hits"],
                word_substitutions=evaluation_results["#Word-Substitutions"],
                fn=evaluation_results["#FN"],
                fp=evaluation_results["#FP"],
            )

            text_evaluations_list.append(page_evaluation)
            if self._intermediate_evaluations_path:
                self.save_intermediate_evaluations(
                    evaluation_name="ocr_eval",
                    enunumerate_id=i,
                    doc_id=doc_id,
                    evaluations=[page_evaluation],
                )

        mean_character_accuracy = (
            statistics.mean(char_accuracy_list) if char_accuracy_list else 0.0
        )
        mean_norm_ed = statistics.mean(norm_ed_list) if norm_ed_list else 0.0
        mean_word_accuracy = (
            statistics.mean(word_accuracy_list) if word_accuracy_list else 0.0
        )
        mean_f1 = statistics.mean(f1_list) if f1_list else 0.0
        mean_precision = statistics.mean(precision_list) if precision_list else 0.0
        mean_recall = statistics.mean(recall_list) if recall_list else 0.0

        _log.info(f"Mean Character Accuracy: {mean_character_accuracy:.4f}")
        _log.info(f"Mean Normalized Edit Distance: {mean_norm_ed:.4f}")
        _log.info(f"Mean Word Accuracy: {mean_word_accuracy:.4f}")
        _log.info(f"Mean F1: {mean_f1:.4f}")
        _log.info(f"Mean Precision: {mean_precision:.4f}")
        _log.info(f"Mean Recall: {mean_recall:.4f}")

        return DatasetOcrEvaluation(
            evaluations=text_evaluations_list,
            mean_character_accuracy=mean_character_accuracy,
            mean_norm_ed=mean_norm_ed,
            mean_word_accuracy=mean_word_accuracy,
            mean_f1=mean_f1,
            mean_precision=mean_precision,
            mean_recall=mean_recall,
            total_word_hits=total_word_hits,
            total_word_substitutions=total_word_substitutions,
            total_fn=total_fn,
            total_fp=total_fp,
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
