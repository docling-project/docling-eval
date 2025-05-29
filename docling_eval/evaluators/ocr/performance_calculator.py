import copy
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple

from docling_core.types.doc.page import SegmentedPage

from docling_eval.evaluators.ocr.benchmark_constants import (
    BenchmarkIntersectionInfo,
    ImageMetricsSummary,
    Location,
    SingleWordMetrics,
    Word,
)
from docling_eval.evaluators.ocr.geometry_utils import box_to_key, unify_words
from docling_eval.evaluators.ocr.text_utils import (
    _CalculationConstants,
    _convert_word_to_text_cell,
    calculate_edit_distance,
    match_ground_truth_to_prediction_words,
    refine_prediction_to_many_gt_boxes,
)

BoxesTypes = namedtuple("BoxesTypes", ["zero_iou", "low_iou", "ambiguous_match"])
detection_match_condition_1: Any = (
    lambda boxes_info: boxes_info.prediction_box_portion_covered > 0.5
    or boxes_info.gt_box_portion_covered > 0.5
)
detection_match_condition: Any = detection_match_condition_1


def get_metrics_per_single_word(
    prediction_word_obj: Word,
    matched_gt_word: Optional[Word] = None,
    iou: float = -1.0,
    distance: int = -1,
    distance_insensitive: int = -1,
    norm_ed: float = -1.0,
    norm_ed_insensitive: float = -1.0,
    is_orientation_correct: float = 0.0,
) -> SingleWordMetrics:

    metrics_data = {
        "prediction_text": prediction_word_obj.word,
        "word_weight": prediction_word_obj.word_weight,
        "prediction_location": prediction_word_obj.location,
        "prediction_word_confidence": prediction_word_obj.confidence,
        "prediction_character_confidence": prediction_word_obj.character_confidence,
        "correct_detection": False,
    }

    if matched_gt_word is None:
        return SingleWordMetrics(**metrics_data)

    metrics_data["correct_detection"] = True
    metrics_data["gt_text"] = matched_gt_word.word
    metrics_data["gt_location"] = matched_gt_word.location
    metrics_data["intersection_over_union"] = iou
    metrics_data["edit_distance"] = distance
    metrics_data["edit_distance_case_insensitive"] = distance_insensitive
    metrics_data["is_orientation_correct"] = is_orientation_correct
    max_word_length: int = max(len(prediction_word_obj.word), len(matched_gt_word.word))
    if max_word_length > 0:
        metrics_data["normalized_edit_distance"] = norm_ed
        metrics_data["normalized_edit_distance_case_insensitive"] = norm_ed_insensitive
    else:
        metrics_data["normalized_edit_distance"] = -1.0
        metrics_data["normalized_edit_distance_case_insensitive"] = -1.0

    return SingleWordMetrics(**metrics_data)


class _ModelPerformanceCalculator:

    def __init__(
        self,
        prediction_words: List[Word],
        gt_words: List[Word],
        prediction_segmented_page_meta: SegmentedPage,
        gt_segmented_page_meta: SegmentedPage,
        add_space_between_merged_gt_words: bool = True,
        add_space_between_merged_prediction_words: bool = True,
        string_normalize_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.prediction_words_input: List[Word] = prediction_words
        self.gt_words_input: List[Word] = gt_words
        self.prediction_page_meta: SegmentedPage = prediction_segmented_page_meta
        self.gt_page_meta: SegmentedPage = gt_segmented_page_meta

        self.add_space_between_merged_gt_words: bool = add_space_between_merged_gt_words
        self.add_space_between_merged_prediction_words: bool = (
            add_space_between_merged_prediction_words
        )
        self.string_normalize_map: Dict[str, str] = (
            string_normalize_map if string_normalize_map else {}
        )

        self.prediction_words_original: List[Word] = copy.deepcopy(
            self.prediction_words_input
        )
        self.gt_words_original: List[Word] = copy.deepcopy(self.gt_words_input)
        self.__reset_matched_property()

        self.gt_page_unified: SegmentedPage = self.gt_page_meta.model_copy(deep=True)
        self.gt_page_unified.word_cells = []
        self.gt_page_unified.has_words = False

        self.prediction_page_unified: SegmentedPage = (
            self.prediction_page_meta.model_copy(deep=True)
        )
        self.prediction_page_unified.word_cells = []
        self.prediction_page_unified.has_words = False

        self.gt_boxes_that_are_fn_after_refinement: List[Word] = []
        self.prediction_boxes_that_are_fp_after_refinement: List[Word] = []
        self.prediction_boxes_that_were_merged: List[Word] = []
        self.gt_boxes_that_were_merged: List[Word] = []
        self.fp_boxes: BoxesTypes = BoxesTypes([], [], [])
        self.fn_boxes: BoxesTypes = BoxesTypes([], [], [])
        self.gt_to_prediction_boxes_map: Dict[
            Tuple[float, float, float, float],
            List[Tuple[Word, BenchmarkIntersectionInfo]],
        ]
        self.prediction_to_gt_boxes_map: Dict[
            Tuple[float, float, float, float],
            List[Tuple[Word, BenchmarkIntersectionInfo]],
        ]
        self.__do_evaluation()

    def __do_evaluation(self) -> None:
        gt_to_prediction_boxes_map_val, prediction_to_gt_boxes_map_val = (
            match_ground_truth_to_prediction_words(
                self.gt_words_original, self.prediction_words_original
            )
        )
        self.gt_to_prediction_boxes_map = gt_to_prediction_boxes_map_val
        self.prediction_to_gt_boxes_map = prediction_to_gt_boxes_map_val
        self.num_prediction_boxes_without_gt_intersection_val: int = sum(
            [0 == len(v) for k, v in self.prediction_to_gt_boxes_map.items()]
        )
        self.num_gt_boxes_without_pred_intersection_val: int = sum(
            [0 == len(v) for k, v in self.gt_to_prediction_boxes_map.items()]
        )
        self.__create_unified_words()

    def num_prediction_boxes_without_gt_intersection(self) -> int:
        return self.num_prediction_boxes_without_gt_intersection_val

    def get_modified_jsons(self) -> Tuple[SegmentedPage, SegmentedPage]:
        return self.gt_page_unified, self.prediction_page_unified

    def __get_overlapped_gt_words(
        self, prediction_word: Word
    ) -> List[Tuple[Word, BenchmarkIntersectionInfo]]:
        box_key: Tuple[float, float, float, float] = box_to_key(
            prediction_word.location
        )
        if box_key in self.prediction_to_gt_boxes_map:
            return self.prediction_to_gt_boxes_map[box_key]
        return []

    def __get_overlapped_prediction_words(
        self, gt_word: Word
    ) -> List[Tuple[Word, BenchmarkIntersectionInfo]]:
        gt_box_key: Tuple[float, float, float, float] = box_to_key(gt_word.location)
        if gt_box_key in self.gt_to_prediction_boxes_map:
            return self.gt_to_prediction_boxes_map[gt_box_key]
        return []

    def __reset_matched_property(self) -> None:
        for word_item in self.gt_words_original:
            word_item.matched = False
        for word_item in self.prediction_words_original:
            word_item.matched = False

    def __create_unified_words(self) -> None:
        for pred_word in self.prediction_words_original:
            if pred_word.word_weight is None:
                pred_word.word_weight = 1
        for gt_word_item in self.gt_words_original:
            if gt_word_item.word_weight is None:
                gt_word_item.word_weight = 1

        gt_to_predictions_tmp: List[
            Tuple[Word, List[Tuple[Word, BenchmarkIntersectionInfo]]]
        ] = []
        self.false_negatives: List[Word] = []
        self.false_positives: List[Word] = []
        self.gt_and_prediction_matches: List[Tuple[Word, Word]] = []
        self.prediction_to_many_gt_boxes_map: List[
            Tuple[Word, List[Tuple[Word, BenchmarkIntersectionInfo]]]
        ] = []

        for gt_word in self.gt_words_original:
            intersections: List[Tuple[Word, BenchmarkIntersectionInfo]] = []
            words_overlapping_with_gt_word: List[
                Tuple[Word, BenchmarkIntersectionInfo]
            ] = self.__get_overlapped_prediction_words(gt_word)
            if 0 == len(words_overlapping_with_gt_word):
                self.fn_boxes.zero_iou.append(gt_word)
            for prediction_box, boxes_info in words_overlapping_with_gt_word:
                matched_gt_words: List[Tuple[Word, BenchmarkIntersectionInfo]] = (
                    self.__get_overlapped_gt_words(prediction_box)
                )
                found_better_match: bool = False
                for gt_box_2, boxes_info_2 in matched_gt_words:
                    if boxes_info.iou < boxes_info_2.iou:
                        found_better_match = True
                if not found_better_match and (detection_match_condition(boxes_info)):
                    intersections.append((prediction_box, boxes_info))
            if 0 == len(intersections):
                self.false_negatives.append(gt_word)
            else:
                gt_to_predictions_tmp.append((gt_word, intersections))
                gt_word.matched = True
            if len(words_overlapping_with_gt_word) > 0 and 0 == len(intersections):
                self.fn_boxes.low_iou.append(gt_word)

        for prediction_word in self.prediction_words_original:
            intersections: List[Tuple[Word, BenchmarkIntersectionInfo]] = []
            words_overlapping_with_prediction_word: List[
                Tuple[Word, BenchmarkIntersectionInfo]
            ] = self.__get_overlapped_gt_words(prediction_word)
            if 0 == len(words_overlapping_with_prediction_word):
                self.fp_boxes.zero_iou.append(prediction_word)
            for gt_box, boxes_info in words_overlapping_with_prediction_word:
                matched_prediction_words: List[
                    Tuple[Word, BenchmarkIntersectionInfo]
                ] = self.__get_overlapped_prediction_words(gt_box)
                found_better_match: bool = False
                for prediction_box_2, boxes_info_2 in matched_prediction_words:
                    if boxes_info.iou < boxes_info_2.iou:
                        found_better_match = True
                if not found_better_match and (detection_match_condition(boxes_info)):
                    intersections.append((gt_box, boxes_info))
            if 0 == len(intersections):
                self.false_positives.append(prediction_word)
            elif 1 == len(intersections):
                pass
            else:
                self.prediction_to_many_gt_boxes_map.append(
                    (prediction_word, intersections)
                )
            if len(words_overlapping_with_prediction_word) > 0 and 0 == len(
                intersections
            ):
                self.fp_boxes.low_iou.append(prediction_word)

        gt_boxes_to_be_removed_from_match_list: List[Word] = []

        for prediction_word, intersections in self.prediction_to_many_gt_boxes_map:
            valid_intersections: List[Tuple[Word, BenchmarkIntersectionInfo]]
            invalid_intersections: List[Tuple[Word, BenchmarkIntersectionInfo]]
            valid_intersections, invalid_intersections = (
                refine_prediction_to_many_gt_boxes(prediction_word, intersections)
            )
            if valid_intersections:
                matched_gt_words: List[Word] = [
                    gt_box for (gt_box, boxes_info) in valid_intersections
                ]
                gt_boxes_to_be_removed_from_match_list.extend(matched_gt_words)
                unified_gt_word: Word = unify_words(
                    matched_gt_words,
                    add_space_between_words=self.add_space_between_merged_gt_words,
                )
                self.gt_boxes_that_were_merged.extend(matched_gt_words)
                unified_gt_word.matched = True
                self.gt_and_prediction_matches.append(
                    (unified_gt_word, prediction_word)
                )
            else:
                self.false_positives.append(prediction_word)
                self.prediction_boxes_that_are_fp_after_refinement.append(
                    prediction_word
                )
                self.fp_boxes.ambiguous_match.append(prediction_word)
            for gt_box, _ in invalid_intersections:
                self.fn_boxes.ambiguous_match.append(gt_box)
                if gt_box.matched:
                    self.gt_boxes_that_are_fn_after_refinement.append(gt_box)
                else:
                    pass
        self.gt_with_predictions: List[
            Tuple[Word, List[Tuple[Word, BenchmarkIntersectionInfo]]]
        ] = []

        temp_gt_pred_list: List[
            Tuple[Word, List[Tuple[Word, BenchmarkIntersectionInfo]]]
        ] = [
            (gt_word, intersections_list)
            for (gt_word, intersections_list) in gt_to_predictions_tmp
            if not gt_word.location
            in [b.location for b in gt_boxes_to_be_removed_from_match_list]
        ]
        final_gt_pred_list: List[
            Tuple[Word, List[Tuple[Word, BenchmarkIntersectionInfo]]]
        ] = [
            (gt_word, intersections_list)
            for (gt_word, intersections_list) in temp_gt_pred_list
            if not gt_word.location
            in [b.location for b in self.gt_boxes_that_are_fn_after_refinement]
        ]
        for gt_word_item in self.gt_boxes_that_are_fn_after_refinement:
            self.false_negatives.append(gt_word_item)
            gt_word_item.matched = False
        self.gt_with_predictions = final_gt_pred_list

        for gt_word, intersections in self.gt_with_predictions:
            if len(intersections) > 1:
                prediction_boxes: List[Word] = [
                    box for (box, boxes_info) in intersections
                ]
                unified_prediction_word: Word = unify_words(
                    prediction_boxes,
                    add_space_between_words=self.add_space_between_merged_prediction_words,
                )
                self.prediction_boxes_that_were_merged.extend(prediction_boxes)
                self.gt_and_prediction_matches.append(
                    (gt_word, unified_prediction_word)
                )
            else:
                prediction_word_item: Word = intersections[0][0]
                self.gt_and_prediction_matches.append((gt_word, prediction_word_item))

        new_fp: List[Word] = [
            pred_word
            for pred_word in self.false_positives
            if not pred_word.location
            in [b.location for b in self.prediction_boxes_that_were_merged]
        ]
        self.false_positives = new_fp
        new_fn: List[Word] = [
            gt_word_item
            for gt_word_item in self.false_negatives
            if not gt_word_item.location
            in [b.location for b in gt_boxes_to_be_removed_from_match_list]
        ]
        self.false_negatives = new_fn

        zero_iou_fn: List[Word] = [
            word
            for word in self.fn_boxes.zero_iou
            if word.location in [b.location for b in self.false_negatives]
        ]
        ambiguous_match_fn: List[Word] = [
            word
            for word in self.fn_boxes.ambiguous_match
            if word.location in [b.location for b in self.false_negatives]
        ]
        low_iou_fn: List[Word] = [
            word
            for word in self.fn_boxes.low_iou
            if word.location in [b.location for b in self.false_negatives]
        ]
        low_iou_fn = [
            word
            for word in low_iou_fn
            if word.location not in [b.location for b in ambiguous_match_fn]
        ]
        self.fn_boxes = BoxesTypes(zero_iou_fn, low_iou_fn, ambiguous_match_fn)

        zero_iou_fp: List[Word] = [
            word
            for word in self.fp_boxes.zero_iou
            if word.location in [b.location for b in self.false_positives]
        ]
        low_iou_fp: List[Word] = [
            word
            for word in self.fp_boxes.low_iou
            if word.location in [b.location for b in self.false_positives]
        ]
        ambiguous_match_fp: List[Word] = [
            word
            for word in self.fp_boxes.ambiguous_match
            if word.location in [b.location for b in self.false_positives]
        ]
        self.fp_boxes = BoxesTypes(zero_iou_fp, low_iou_fp, ambiguous_match_fp)

        self.gt_words_unified: List[Word] = []
        self.prediction_words_unified: List[Word] = []
        self.gt_words_unified.extend(
            [gt_w for (gt_w, pred_w) in self.gt_and_prediction_matches]
        )
        self.gt_words_unified.extend(self.false_negatives)

        self.prediction_words_unified.extend(
            [pred_w for (gt_w, pred_w) in self.gt_and_prediction_matches]
        )
        self.prediction_words_unified.extend(self.false_positives)

        self.gt_page_unified.word_cells = [
            _convert_word_to_text_cell(w) for w in self.gt_words_unified
        ]
        self.gt_page_unified.has_words = bool(self.gt_page_unified.word_cells)

        self.prediction_page_unified.word_cells = [
            _convert_word_to_text_cell(w) for w in self.prediction_words_unified
        ]
        self.prediction_page_unified.has_words = bool(
            self.prediction_page_unified.word_cells
        )

        self.gt_words: List[Word] = self.gt_words_unified

    @staticmethod
    def split_fn_by_ignore_chars(
        false_negatives: List[Word],
    ) -> Tuple[List[Word], List[Word]]:
        fn_regular: List[Word] = []
        fn_ignore_chars: List[Word] = []
        for w in false_negatives:
            if w.word in "-,.*_+=":
                fn_ignore_chars.append(w)
            else:
                fn_regular.append(w)
        return fn_regular, fn_ignore_chars

    @staticmethod
    def split_fn_by_single_chars(
        false_negatives: List[Word],
    ) -> Tuple[List[Word], List[Word]]:
        fn_regular: List[Word] = []
        fn_ignore_chars: List[Word] = []
        for w in false_negatives:
            if 1 == len(w.word):
                fn_ignore_chars.append(w)
            else:
                fn_regular.append(w)
        return fn_regular, fn_ignore_chars

    def calc_metrics(self) -> ImageMetricsSummary:
        text_length_false_positives: float = _CalculationConstants.EPS
        sum_edit_distance_intersection_sensitive: float = _CalculationConstants.EPS
        sum_edit_distance_intersection_insensitive: float = _CalculationConstants.EPS
        sum_max_length_intersection: float = _CalculationConstants.EPS
        num_matched_pairs: int = 0
        sum_norm_ed: float = 0.0
        sum_correct_orientation: float = 0.0
        num_of_orientation_set: int = 0
        sum_norm_ed_insensitive: float = 0.0
        matched_pairs_perfect_recognition_sensitive_weighted_sum: float = 0.0
        matched_pairs_perfect_recognition_insensitive_weighted_sum: float = 0.0
        matched_pairs_imperfect_recognition_sensitive_weighted_sum: float = 0.0
        matched_pairs_imperfect_recognition_insensitive_weighted_sum: float = 0.0
        self.metrics_per_prediction_word: List[SingleWordMetrics] = []

        for gt_word, prediction_word in self.gt_and_prediction_matches:
            iou: float = -1.0
            gt_text: str = gt_word.word
            prediction_text: str = prediction_word.word
            num_matched_pairs += 1
            max_word_length: float = max(
                len(prediction_text), len(gt_text), _CalculationConstants.EPS
            )
            sum_max_length_intersection += max_word_length
            if max_word_length == _CalculationConstants.EPS:
                continue

            is_orientation_correct: float = 0.0
            if gt_word.vertical is not None and prediction_word.vertical is not None:
                if gt_word.vertical == prediction_word.vertical:
                    is_orientation_correct = 1.0
                sum_correct_orientation += is_orientation_correct
                num_of_orientation_set += 1

            word_edit_distance_insensitive: int = calculate_edit_distance(
                gt_text.upper(), prediction_text.upper(), self.string_normalize_map
            )
            norm_ed_insensitive_val: float = (
                word_edit_distance_insensitive / max_word_length
            )
            sum_norm_ed_insensitive += norm_ed_insensitive_val
            matched_pairs_perfect_recognition_insensitive_weighted_sum += (
                gt_word.word_weight if 0 == word_edit_distance_insensitive else 0
            )
            matched_pairs_imperfect_recognition_insensitive_weighted_sum += (
                0 if 0 == word_edit_distance_insensitive else gt_word.word_weight
            )
            sum_edit_distance_intersection_insensitive += word_edit_distance_insensitive

            word_edit_distance: int = calculate_edit_distance(
                gt_text, prediction_text, self.string_normalize_map
            )
            norm_ed_val: float = word_edit_distance / max_word_length
            sum_norm_ed += norm_ed_val
            matched_pairs_perfect_recognition_sensitive_weighted_sum += (
                gt_word.word_weight if 0 == word_edit_distance else 0
            )
            matched_pairs_imperfect_recognition_sensitive_weighted_sum += (
                0 if 0 == word_edit_distance else gt_word.word_weight
            )
            sum_edit_distance_intersection_sensitive += word_edit_distance

            prediction_cell_metric: SingleWordMetrics = get_metrics_per_single_word(
                prediction_word,
                gt_word,
                iou,
                word_edit_distance,
                word_edit_distance_insensitive,
                norm_ed_val,
                norm_ed_insensitive_val,
                is_orientation_correct,
            )
            self.metrics_per_prediction_word.append(prediction_cell_metric)

        for fp_word in self.false_positives:
            prediction_cell_metric: SingleWordMetrics = get_metrics_per_single_word(
                fp_word
            )
            self.metrics_per_prediction_word.append(prediction_cell_metric)

        fn_not_ignored_chars: List[Word]
        fn_ignore_chars: List[Word]
        fn_not_ignored_chars, fn_ignore_chars = self.split_fn_by_ignore_chars(
            self.false_negatives
        )
        fn_not_single_chars: List[Word]
        fn_single_chars: List[Word]
        fn_not_single_chars, fn_single_chars = self.split_fn_by_single_chars(
            self.false_negatives
        )
        unmatched_len: List[int] = [len(w.word) for w in self.false_negatives]
        text_length_false_negatives: float = (
            sum(unmatched_len) if unmatched_len else _CalculationConstants.EPS
        )

        text_length_false_positives = (
            sum(len(w.word) for w in self.false_positives)
            if self.false_positives
            else _CalculationConstants.EPS
        )

        denominator_union: float = (
            sum_max_length_intersection
            + text_length_false_positives
            + text_length_false_negatives
        )
        image_avg_edit_distance_union_sensitive: float = (
            (
                sum_edit_distance_intersection_sensitive
                + text_length_false_positives
                + text_length_false_negatives
            )
            / denominator_union
            if denominator_union > 0
            else 0
        )

        image_avg_edit_distance_union_insensitive: float = (
            (
                sum_edit_distance_intersection_insensitive
                + text_length_false_positives
                + text_length_false_negatives
            )
            / denominator_union
            if denominator_union > 0
            else 0
        )

        image_avg_edit_distance_intersection_sensitive: float = (
            (sum_edit_distance_intersection_sensitive / sum_max_length_intersection)
            if sum_max_length_intersection > 0
            else 0
        )

        image_avg_edit_distance_intersection_insensitive: float = (
            (sum_edit_distance_intersection_insensitive / sum_max_length_intersection)
            if sum_max_length_intersection > 0
            else 0
        )

        num_false_positives: int = len(self.false_positives)
        num_false_negatives_unfiltered: int = len(self.false_negatives)
        num_false_negatives_no_ignored_chars: int = len(fn_not_ignored_chars)
        num_false_negatives_no_single_chars: int = len(fn_not_single_chars)
        num_gt_cells: int = len(self.gt_words_unified)
        num_prediction_cells: int = len(self.prediction_words_unified)
        num_true_positive_matches: int = len(self.gt_and_prediction_matches)

        detection_precision: float = num_true_positive_matches / max(
            _CalculationConstants.EPS, num_prediction_cells
        )
        detection_recall: float = num_true_positive_matches / max(
            _CalculationConstants.EPS, num_gt_cells
        )
        detection_f1_score: float = (2 * detection_recall * detection_precision) / max(
            detection_recall + detection_precision, _CalculationConstants.EPS
        )

        denominator_wa_intersection_sensitive: float = max(
            _CalculationConstants.EPS,
            matched_pairs_perfect_recognition_sensitive_weighted_sum
            + matched_pairs_imperfect_recognition_sensitive_weighted_sum,
        )
        word_accuracy_intersection_case_sensitive: float = (
            (
                matched_pairs_perfect_recognition_sensitive_weighted_sum
                / denominator_wa_intersection_sensitive
            )
            if denominator_wa_intersection_sensitive > 0
            else 0
        )

        denominator_wa_intersection_insensitive: float = max(
            _CalculationConstants.EPS,
            matched_pairs_perfect_recognition_insensitive_weighted_sum
            + matched_pairs_imperfect_recognition_insensitive_weighted_sum,
        )
        word_accuracy_intersection_case_insensitive: float = (
            (
                matched_pairs_perfect_recognition_insensitive_weighted_sum
                / denominator_wa_intersection_insensitive
            )
            if denominator_wa_intersection_insensitive > 0
            else 0
        )

        denominator_wa_union_sensitive: float = max(
            _CalculationConstants.EPS,
            matched_pairs_perfect_recognition_sensitive_weighted_sum
            + matched_pairs_imperfect_recognition_sensitive_weighted_sum
            + num_false_positives
            + num_false_negatives_unfiltered,
        )
        word_accuracy_union_case_sensitive: float = (
            (
                matched_pairs_perfect_recognition_sensitive_weighted_sum
                / denominator_wa_union_sensitive
            )
            if denominator_wa_union_sensitive > 0
            else 0
        )

        denominator_wa_union_insensitive: float = max(
            _CalculationConstants.EPS,
            matched_pairs_perfect_recognition_insensitive_weighted_sum
            + matched_pairs_imperfect_recognition_insensitive_weighted_sum
            + num_false_positives
            + num_false_negatives_unfiltered,
        )
        word_accuracy_union_case_insensitive: float = (
            (
                matched_pairs_perfect_recognition_insensitive_weighted_sum
                / denominator_wa_union_insensitive
            )
            if denominator_wa_union_insensitive > 0
            else 0
        )

        calculated_norm_ed_tp_only: float = sum_norm_ed / max(
            _CalculationConstants.EPS, num_true_positive_matches
        )
        denominator_norm_ed_all_cells: float = max(
            _CalculationConstants.EPS,
            num_true_positive_matches
            + num_false_positives
            + num_false_negatives_unfiltered,
        )
        calculated_norm_ed_all_cells: float = (
            (sum_norm_ed + num_false_positives + num_false_negatives_unfiltered)
            / denominator_norm_ed_all_cells
            if denominator_norm_ed_all_cells > 0
            else 0
        )

        orientation_accuracy: float = sum_correct_orientation / max(
            1, num_of_orientation_set
        )

        metrics_summary_data = {
            "num_prediction_cells": num_prediction_cells,
            "number_of_gt_cells": num_gt_cells,
            "number_of_false_positive_detections": num_false_positives,
            "norm_ed_tp_only": 100 * (1 - calculated_norm_ed_tp_only),
            "norm_ed_all_cells": 100 * (1 - calculated_norm_ed_all_cells),
            "num_true_positive_matches": num_true_positive_matches,
            "number_of_false_negative_detections": num_false_negatives_unfiltered,
            "without_ignored_chars_false_negatives": num_false_negatives_no_ignored_chars,
            "without_single_chars_false_negatives": num_false_negatives_no_single_chars,
            "detection_precision": 100 * detection_precision,
            "detection_recall": 100 * detection_recall,
            "detection_f1_score": 100 * detection_f1_score,
            "word_accuracy_intersection_case_sensitive": 100
            * word_accuracy_intersection_case_sensitive,
            "word_accuracy_intersection_case_insensitive": 100
            * word_accuracy_intersection_case_insensitive,
            "word_accuracy_union_case_sensitive": 100
            * word_accuracy_union_case_sensitive,
            "word_accuracy_union_case_insensitive": 100
            * word_accuracy_union_case_insensitive,
            "edit_score_intersection_case_sensitive_not_avg_over_words": 100
            * (1 - image_avg_edit_distance_intersection_sensitive),
            "edit_score_intersection_case_insensitive_not_avg_over_words": 100
            * (1 - image_avg_edit_distance_intersection_insensitive),
            "edit_score_union_case_sensitive_not_avg_over_words": 100
            * (1 - image_avg_edit_distance_union_sensitive),
            "edit_score_union_case_insensitive_not_avg_over_words": 100
            * (1 - image_avg_edit_distance_union_insensitive),
            "sum_edit_distance_intersection_case_sensitive_not_avg_over_words": sum_edit_distance_intersection_sensitive,
            "sum_edit_distance_intersection_case_insensitive_not_avg_over_words": sum_edit_distance_intersection_insensitive,
            "sum_max_length_intersection": sum_max_length_intersection,
            "text_length_false_positives": text_length_false_positives,
            "text_length_false_negatives": text_length_false_negatives,
            "sum_norm_ed": sum_norm_ed,
            "word_insertions": num_false_negatives_unfiltered,
            "word_deletions": num_false_positives,
            "word_substitutions_case_sensitive": matched_pairs_imperfect_recognition_sensitive_weighted_sum,
            "word_hits_case_sensitive": matched_pairs_perfect_recognition_sensitive_weighted_sum,
            "word_substitutions_case_insensitive": matched_pairs_imperfect_recognition_insensitive_weighted_sum,
            "word_hits_case_insensitive": matched_pairs_perfect_recognition_insensitive_weighted_sum,
            "num_prediction_boxes_without_gt_intersection": len(self.fp_boxes.zero_iou),
            "num_gt_boxes_that_do_not_intersect_with_a_gt": len(self.fn_boxes.zero_iou),
            "num_gt_boxes_that_are_fn_after_refinement": len(
                self.fn_boxes.ambiguous_match
            ),
            "num_prediction_boxes_fp_after_refinement": len(
                self.fp_boxes.ambiguous_match
            ),
            "num_gt_boxes_with_low_iou": len(self.fn_boxes.low_iou),
            "num_prediction_boxes_with_low_iou": len(self.fp_boxes.low_iou),
            "prediction_boxes_that_were_merged": len(
                self.prediction_boxes_that_were_merged
            ),
            "gt_boxes_that_were_merged": len(self.gt_boxes_that_were_merged),
            "orientation_accuracy": 100 * orientation_accuracy,
        }

        summary_instance = ImageMetricsSummary.model_validate(metrics_summary_data)
        self.image_data_for_end2end: ImageMetricsSummary = summary_instance
        return summary_instance

    def get_metrics_per_word(self) -> List[SingleWordMetrics]:
        return self.metrics_per_prediction_word
