import copy
from collections import namedtuple
from typing import Dict, List, Tuple

from docling_core.types.doc.page import SegmentedPage

from docling_eval.evaluators.ocr.evaluation_models import (
    BenchmarkIntersectionInfo,
    OcrMetricsSummary,
    Word,
    _CalculationConstants,
)
from docling_eval.evaluators.ocr.geometry_utils import box_to_key
from docling_eval.evaluators.ocr.matching_logic import (
    current_detection_match_condition,
    match_ground_truth_to_prediction_words,
    refine_prediction_to_many_gt_boxes,
)
from docling_eval.evaluators.ocr.processing_utils import (
    convert_word_to_text_cell,
    merge_words_into_one,
)

BoxClassification = namedtuple(
    "BoxClassification", ["zero_iou", "low_iou", "ambiguous_match"]
)


class _OcrPerformanceCalculator:
    def __init__(
        self,
        prediction_words: List[Word],
        ground_truth_words: List[Word],
        prediction_segmented_page_metadata: SegmentedPage,
        ground_truth_segmented_page_metadata: SegmentedPage,
        add_space_between_merged_gt_words: bool = True,
        add_space_between_merged_prediction_words: bool = True,
    ) -> None:
        self.prediction_words_input: List[Word] = prediction_words
        self.ground_truth_words_input: List[Word] = ground_truth_words
        self.prediction_page_metadata: SegmentedPage = (
            prediction_segmented_page_metadata
        )
        self.ground_truth_page_metadata: SegmentedPage = (
            ground_truth_segmented_page_metadata
        )

        self.add_space_between_merged_gt_words: bool = add_space_between_merged_gt_words
        self.add_space_between_merged_prediction_words: bool = (
            add_space_between_merged_prediction_words
        )

        self.prediction_words_original: List[Word] = copy.deepcopy(
            self.prediction_words_input
        )
        self.ground_truth_words_original: List[Word] = copy.deepcopy(
            self.ground_truth_words_input
        )
        self._reset_matched_status()

        self.ground_truth_page_merged_words: SegmentedPage = (
            self.ground_truth_page_metadata.model_copy(deep=True)
        )
        self.ground_truth_page_merged_words.word_cells = []
        self.ground_truth_page_merged_words.has_words = False

        self.prediction_page_merged_words: SegmentedPage = (
            self.prediction_page_metadata.model_copy(deep=True)
        )
        self.prediction_page_merged_words.word_cells = []
        self.prediction_page_merged_words.has_words = False

        self.gt_boxes_flagged_as_fn_post_refinement: List[Word] = []
        self.prediction_boxes_flagged_as_fp_post_refinement: List[Word] = []
        self.prediction_boxes_that_were_merged: List[Word] = []
        self.gt_boxes_that_were_merged: List[Word] = []
        self.fp_box_classifications: BoxClassification = BoxClassification([], [], [])
        self.fn_box_classifications: BoxClassification = BoxClassification([], [], [])

        self.gt_to_prediction_overlap_map: Dict[
            Tuple[float, float, float, float],
            List[Tuple[Word, BenchmarkIntersectionInfo]],
        ]
        self.prediction_to_gt_overlap_map: Dict[
            Tuple[float, float, float, float],
            List[Tuple[Word, BenchmarkIntersectionInfo]],
        ]
        self._perform_evaluation()

    def _perform_evaluation(self) -> None:
        gt_to_pred_map, pred_to_gt_map = match_ground_truth_to_prediction_words(
            self.ground_truth_words_original, self.prediction_words_original
        )
        self.gt_to_prediction_overlap_map = gt_to_pred_map
        self.prediction_to_gt_overlap_map = pred_to_gt_map
        self._process_word_matches_and_merges()

    def get_processed_segmented_pages(self) -> Tuple[SegmentedPage, SegmentedPage]:
        return self.ground_truth_page_merged_words, self.prediction_page_merged_words

    def _get_overlapping_ground_truth_words(
        self, prediction_word: Word
    ) -> List[Tuple[Word, BenchmarkIntersectionInfo]]:
        box_key_val: Tuple[float, float, float, float] = box_to_key(
            prediction_word.bbox
        )
        return self.prediction_to_gt_overlap_map.get(box_key_val, [])

    def _get_overlapping_prediction_words(
        self, ground_truth_word: Word
    ) -> List[Tuple[Word, BenchmarkIntersectionInfo]]:
        gt_box_key_val: Tuple[float, float, float, float] = box_to_key(
            ground_truth_word.bbox
        )
        return self.gt_to_prediction_overlap_map.get(gt_box_key_val, [])

    def _reset_matched_status(self) -> None:
        for word_item in self.ground_truth_words_original:
            word_item.matched = False
        for word_item in self.prediction_words_original:
            word_item.matched = False

    def _process_word_matches_and_merges(self) -> None:
        gt_to_potential_prediction_matches: List[
            Tuple[Word, List[Tuple[Word, BenchmarkIntersectionInfo]]]
        ] = []
        self.current_false_negatives: List[Word] = []
        self.current_false_positives: List[Word] = []
        self.confirmed_gt_prediction_matches: List[Tuple[Word, Word]] = []

        prediction_to_many_gt_candidates: List[
            Tuple[Word, List[Tuple[Word, BenchmarkIntersectionInfo]]]
        ] = []

        for gt_word in self.ground_truth_words_original:
            valid_intersections_for_gt: List[Tuple[Word, BenchmarkIntersectionInfo]] = (
                []
            )
            overlapping_predictions: List[Tuple[Word, BenchmarkIntersectionInfo]] = (
                self._get_overlapping_prediction_words(gt_word)
            )

            if not overlapping_predictions:
                self.fn_box_classifications.zero_iou.append(gt_word)

            for pred_box, box_info in overlapping_predictions:
                gt_words_overlapping_with_pred: List[
                    Tuple[Word, BenchmarkIntersectionInfo]
                ] = self._get_overlapping_ground_truth_words(pred_box)
                is_better_match_for_pred_found: bool = False
                for other_gt_box, other_box_info in gt_words_overlapping_with_pred:
                    if box_info.iou < other_box_info.iou:
                        is_better_match_for_pred_found = True
                        break
                if not is_better_match_for_pred_found and (
                    current_detection_match_condition(box_info)
                ):
                    valid_intersections_for_gt.append((pred_box, box_info))

            if not valid_intersections_for_gt:
                self.current_false_negatives.append(gt_word)
                if overlapping_predictions:
                    self.fn_box_classifications.low_iou.append(gt_word)
            else:
                gt_to_potential_prediction_matches.append(
                    (gt_word, valid_intersections_for_gt)
                )
                gt_word.matched = True

        for pred_word in self.prediction_words_original:
            valid_intersections_for_pred: List[
                Tuple[Word, BenchmarkIntersectionInfo]
            ] = []
            overlapping_gts: List[Tuple[Word, BenchmarkIntersectionInfo]] = (
                self._get_overlapping_ground_truth_words(pred_word)
            )

            if not overlapping_gts:
                self.fp_box_classifications.zero_iou.append(pred_word)

            for gt_box, box_info in overlapping_gts:
                pred_words_overlapping_with_gt: List[
                    Tuple[Word, BenchmarkIntersectionInfo]
                ] = self._get_overlapping_prediction_words(gt_box)
                is_better_match_for_gt_found: bool = False
                for other_pred_box, other_box_info in pred_words_overlapping_with_gt:
                    if box_info.iou < other_box_info.iou:
                        is_better_match_for_gt_found = True
                        break
                if not is_better_match_for_gt_found and (
                    current_detection_match_condition(box_info)
                ):
                    valid_intersections_for_pred.append((gt_box, box_info))

            if not valid_intersections_for_pred:
                self.current_false_positives.append(pred_word)
                if overlapping_gts:
                    self.fp_box_classifications.low_iou.append(pred_word)
            elif len(valid_intersections_for_pred) > 1:
                prediction_to_many_gt_candidates.append(
                    (pred_word, valid_intersections_for_pred)
                )

        gt_boxes_consumed_by_prediction_merges: List[Word] = []

        for pred_word, gt_intersections in prediction_to_many_gt_candidates:
            valid_gt_merges, invalid_gt_merges = refine_prediction_to_many_gt_boxes(
                pred_word, gt_intersections
            )
            if valid_gt_merges:
                matched_gt_words_for_merge: List[Word] = [
                    gt_box for (gt_box, _) in valid_gt_merges
                ]
                gt_boxes_consumed_by_prediction_merges.extend(
                    matched_gt_words_for_merge
                )
                merged_gt_word: Word = merge_words_into_one(
                    matched_gt_words_for_merge,
                    add_space_between_words=self.add_space_between_merged_gt_words,
                )
                self.gt_boxes_that_were_merged.extend(matched_gt_words_for_merge)
                merged_gt_word.matched = True
                self.confirmed_gt_prediction_matches.append((merged_gt_word, pred_word))
            else:
                self.current_false_positives.append(pred_word)
                self.prediction_boxes_flagged_as_fp_post_refinement.append(pred_word)
                self.fp_box_classifications.ambiguous_match.append(pred_word)

            for (
                gt_box,
                _,
            ) in invalid_gt_merges:
                self.fn_box_classifications.ambiguous_match.append(gt_box)
                if gt_box.matched:
                    self.gt_boxes_flagged_as_fn_post_refinement.append(gt_box)

        gt_boxes_consumed_keys = {
            box_to_key(b.bbox) for b in gt_boxes_consumed_by_prediction_merges
        }

        gt_fn_post_refinement_keys = {
            box_to_key(b.bbox) for b in self.gt_boxes_flagged_as_fn_post_refinement
        }

        final_gt_to_prediction_matches: List[
            Tuple[Word, List[Tuple[Word, BenchmarkIntersectionInfo]]]
        ] = [
            (gt_word, intersections_list)
            for (gt_word, intersections_list) in gt_to_potential_prediction_matches
            if box_to_key(gt_word.bbox) not in gt_boxes_consumed_keys
            and box_to_key(gt_word.bbox) not in gt_fn_post_refinement_keys
        ]

        for gt_word_item in self.gt_boxes_flagged_as_fn_post_refinement:
            self.current_false_negatives.append(gt_word_item)
            gt_word_item.matched = False

        for gt_word, pred_intersections in final_gt_to_prediction_matches:
            if len(pred_intersections) > 1:
                prediction_boxes_for_merge: List[Word] = [
                    box for (box, _) in pred_intersections
                ]
                merged_prediction_word: Word = merge_words_into_one(
                    prediction_boxes_for_merge,
                    add_space_between_words=self.add_space_between_merged_prediction_words,
                )
                self.prediction_boxes_that_were_merged.extend(
                    prediction_boxes_for_merge
                )
                self.confirmed_gt_prediction_matches.append(
                    (gt_word, merged_prediction_word)
                )
            else:
                prediction_word_item: Word = pred_intersections[0][0]
                self.confirmed_gt_prediction_matches.append(
                    (gt_word, prediction_word_item)
                )

        merged_prediction_loc_keys = {
            box_to_key(b.bbox) for b in self.prediction_boxes_that_were_merged
        }

        self.current_false_positives = [
            pred_word
            for pred_word in self.current_false_positives
            if box_to_key(pred_word.bbox) not in merged_prediction_loc_keys
        ]

        self.current_false_negatives = [
            gt_word_item
            for gt_word_item in self.current_false_negatives
            if box_to_key(gt_word_item.bbox) not in gt_boxes_consumed_keys
        ]

        current_false_negatives_keys = {
            box_to_key(fn.bbox) for fn in self.current_false_negatives
        }
        current_false_positives_keys = {
            box_to_key(fp.bbox) for fp in self.current_false_positives
        }

        final_zero_iou_fn = [
            w
            for w in self.fn_box_classifications.zero_iou
            if box_to_key(w.bbox) in current_false_negatives_keys
        ]
        final_ambiguous_match_fn = [
            w
            for w in self.fn_box_classifications.ambiguous_match
            if box_to_key(w.bbox) in current_false_negatives_keys
        ]
        final_ambiguous_match_fn_keys = {
            box_to_key(fn.bbox) for fn in final_ambiguous_match_fn
        }
        final_low_iou_fn = [
            w
            for w in self.fn_box_classifications.low_iou
            if box_to_key(w.bbox) in current_false_negatives_keys
            and box_to_key(w.bbox) not in final_ambiguous_match_fn_keys
        ]
        self.fn_box_classifications = BoxClassification(
            final_zero_iou_fn, final_low_iou_fn, final_ambiguous_match_fn
        )

        final_zero_iou_fp = [
            w
            for w in self.fp_box_classifications.zero_iou
            if box_to_key(w.bbox) in current_false_positives_keys
        ]
        final_ambiguous_match_fp = [
            w
            for w in self.fp_box_classifications.ambiguous_match
            if box_to_key(w.bbox) in current_false_positives_keys
        ]
        final_ambiguous_match_fp_keys = {
            box_to_key(fp.bbox) for fp in final_ambiguous_match_fp
        }
        final_low_iou_fp = [
            w
            for w in self.fp_box_classifications.low_iou
            if box_to_key(w.bbox) in current_false_positives_keys
            and box_to_key(w.bbox) not in final_ambiguous_match_fp_keys
        ]
        self.fp_box_classifications = BoxClassification(
            final_zero_iou_fp, final_low_iou_fp, final_ambiguous_match_fp
        )

        self.ground_truth_words_final_set: List[Word] = []
        self.prediction_words_final_set: List[Word] = []

        self.ground_truth_words_final_set.extend(
            [gt_w for (gt_w, _) in self.confirmed_gt_prediction_matches]
        )
        self.ground_truth_words_final_set.extend(self.current_false_negatives)

        self.prediction_words_final_set.extend(
            [pred_w for (_, pred_w) in self.confirmed_gt_prediction_matches]
        )
        self.prediction_words_final_set.extend(self.current_false_positives)

        self.ground_truth_page_merged_words.word_cells = [
            convert_word_to_text_cell(w) for w in self.ground_truth_words_final_set
        ]
        self.ground_truth_page_merged_words.has_words = bool(
            self.ground_truth_page_merged_words.word_cells
        )

        self.prediction_page_merged_words.word_cells = [
            convert_word_to_text_cell(w) for w in self.prediction_words_final_set
        ]
        self.prediction_page_merged_words.has_words = bool(
            self.prediction_page_merged_words.word_cells
        )

    def calculate_image_metrics(self) -> OcrMetricsSummary:
        num_false_positives: int = len(self.current_false_positives)
        num_false_negatives: int = len(self.current_false_negatives)
        num_gt_cells_final: int = len(self.ground_truth_words_final_set)
        num_prediction_cells_final: int = len(self.prediction_words_final_set)
        number_of_true_positive_matches: int = len(self.confirmed_gt_prediction_matches)

        precision: float = number_of_true_positive_matches / max(
            _CalculationConstants.EPS, num_prediction_cells_final
        )
        recall: float = number_of_true_positive_matches / max(
            _CalculationConstants.EPS, num_gt_cells_final
        )
        f1_score: float = (2 * recall * precision) / max(
            recall + precision, _CalculationConstants.EPS
        )

        metrics_summary_data = {
            "number_of_prediction_cells": num_prediction_cells_final,
            "number_of_gt_cells": num_gt_cells_final,
            "number_of_false_positive_detections": num_false_positives,
            "number_of_true_positive_matches": number_of_true_positive_matches,
            "number_of_false_negative_detections": num_false_negatives,
            "detection_precision": 100.0 * precision,
            "detection_recall": 100.0 * recall,
            "detection_f1": 100.0 * f1_score,
        }

        summary_instance = OcrMetricsSummary.model_validate(metrics_summary_data)
        return summary_instance
