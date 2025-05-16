import copy
from collections import namedtuple

from docling_eval.evaluators.ocr.geometry_utils import box_to_key, unify_words
from docling_eval.evaluators.ocr.matching import (
    match_ground_truth_to_prediction_words,
    refine_prediction_to_many_gt_boxes,
)
from docling_eval.evaluators.ocr.text_utils import calculate_edit_distance

EPS = 1.0e-6
BoxesTypes = namedtuple("BoxesTypes", ["zero_iou", "low_iou", "ambiguous_match"])
detection_match_condition_1 = (
    lambda boxes_info: boxes_info.prediction_box_portion_covered > 0.5
    or boxes_info.gt_box_portion_covered > 0.5
)
detection_match_condition = detection_match_condition_1


def get_metrics_per_single_word(
    prediction_word_obj,
    matched_gt_word=None,
    iou=-1,
    distance=-1,
    distance_insensitive=-1,
    norm_ed=-1,
    norm_ed_insensitive=-1,
    is_orientation_correct=0,
):
    metrics_dict = {}
    metrics_dict["prediction_text"] = prediction_word_obj["word"]
    metrics_dict["word-weight"] = prediction_word_obj["word-weight"]
    metrics_dict["prediction_location"] = prediction_word_obj["location"]
    if "confidence" in prediction_word_obj:
        metrics_dict["prediction_word_confidence"] = prediction_word_obj["confidence"]
    else:
        metrics_dict["prediction_word_confidence"] = ""

    if "character_confidence" in prediction_word_obj:
        metrics_dict["prediction_character_confidence"] = prediction_word_obj[
            "character_confidence"
        ]
    else:
        metrics_dict["prediction_character_confidence"] = ""

    if matched_gt_word is None:
        metrics_dict["correct_detection"] = False
        return metrics_dict

    metrics_dict["correct_detection"] = True
    metrics_dict["gt_text"] = matched_gt_word["word"]
    metrics_dict["gt_location"] = matched_gt_word["location"]
    metrics_dict["intersection_over_union"] = iou
    metrics_dict["edit_distance"] = distance
    metrics_dict["edit_distance_case_insensitive"] = distance_insensitive
    metrics_dict["is_orientation_correct"] = is_orientation_correct
    max_word_length = max(
        len(prediction_word_obj["word"]), len(matched_gt_word["word"])
    )
    if max_word_length > 0:
        metrics_dict["normalized_edit_distance"] = norm_ed
        metrics_dict["normalized_edit_distance_case_insensitive"] = norm_ed_insensitive
    else:
        metrics_dict["normalized_edit_distance"] = -1
        metrics_dict["normalized_edit_distance_case_insensitive"] = -1

    return metrics_dict


class ModelPerformanceCalculator:

    def __init__(
        self,
        prediction_data: dict,
        gt_data: dict,
        add_space_between_merged_gt_words=True,
        add_space_between_merged_prediction_words=True,
        string_normalize_map=None,
    ):
        self.prediction_data = prediction_data
        self.gt_data = gt_data
        self.add_space_between_merged_gt_words = add_space_between_merged_gt_words
        self.add_space_between_merged_prediction_words = (
            add_space_between_merged_prediction_words
        )
        self.string_normalize_map = string_normalize_map if string_normalize_map else {}
        self.prediction_words_original = self.prediction_data["images"][0]["words"]
        self.gt_words_original = self.gt_data["images"][0]["words"]
        self.__reset_matched_property()
        self.gt_json_unified = copy.deepcopy(gt_data)
        self.prediction_json_unified = copy.deepcopy(prediction_data)
        self.gt_boxes_that_are_fn_after_refinement: list = []
        self.prediction_boxes_that_are_fp_after_refinement: list = []
        self.prediction_boxes_that_were_merged: list = []
        self.gt_boxes_that_were_merged: list = []
        self.fp_boxes = BoxesTypes([], [], [])
        self.fn_boxes = BoxesTypes([], [], [])
        self.__do_evaluation()

    def __do_evaluation(self):
        gt_to_prediction_boxes_map_val, prediction_to_gt_boxes_map_val = (
            match_ground_truth_to_prediction_words(
                self.gt_words_original, self.prediction_words_original
            )
        )
        self.gt_to_prediction_boxes_map = gt_to_prediction_boxes_map_val
        self.prediction_to_gt_boxes_map = prediction_to_gt_boxes_map_val
        self.num_prediction_boxes_without_gt_intersection_val = sum(
            [0 == len(v) for k, v in self.prediction_to_gt_boxes_map.items()]
        )
        self.num_gt_boxes_without_pred_intersection_val = sum(
            [0 == len(v) for k, v in self.gt_to_prediction_boxes_map.items()]
        )
        self.__create_unified_words()

    def num_prediction_boxes_without_gt_intersection(self):
        return self.num_prediction_boxes_without_gt_intersection_val

    def get_modified_jsons(self):
        return self.gt_json_unified, self.prediction_json_unified

    def __get_overlapped_gt_words(self, prediction_word):
        box_key = box_to_key(prediction_word["location"])
        if box_key in self.prediction_to_gt_boxes_map:
            return self.prediction_to_gt_boxes_map[box_key]
        return []

    def __get_overlapped_prediction_words(self, gt_word):
        gt_box_key = box_to_key(gt_word["location"])
        if gt_box_key in self.gt_to_prediction_boxes_map:
            return self.gt_to_prediction_boxes_map[gt_box_key]
        return []

    def __create_unified_words(self):
        for pred_word in self.prediction_words_original:
            if "word-weight" not in pred_word:
                pred_word["word-weight"] = 1
        for gt_word_item in self.gt_words_original:
            if "word-weight" not in gt_word_item:
                gt_word_item["word-weight"] = 1

        gt_to_predictions_tmp = []
        self.false_negatives = []
        self.false_positives = []
        self.gt_and_prediction_matches = []
        self.prediction_to_many_gt_boxes_map = []

        for gt_word in self.gt_words_original:
            intersections = []
            words_overlapping_with_gt_word = self.__get_overlapped_prediction_words(
                gt_word
            )
            if 0 == len(words_overlapping_with_gt_word):
                self.fn_boxes.zero_iou.append(gt_word)
            for prediction_box, boxes_info in words_overlapping_with_gt_word:
                matched_gt_words = self.__get_overlapped_gt_words(prediction_box)
                found_better_match = False
                for gt_box_2, boxes_info_2 in matched_gt_words:
                    if boxes_info.iou < boxes_info_2.iou:
                        found_better_match = True
                if not found_better_match and (detection_match_condition(boxes_info)):
                    intersections.append((prediction_box, boxes_info))
            if 0 == len(intersections):
                self.false_negatives.append(gt_word)
            else:
                gt_to_predictions_tmp.append((gt_word, intersections))
                gt_word["matched"] = True
            if len(words_overlapping_with_gt_word) > 0 and 0 == len(intersections):
                self.fn_boxes.low_iou.append(gt_word)

        for prediction_word in self.prediction_words_original:
            intersections = []
            words_overlapping_with_prediction_word = self.__get_overlapped_gt_words(
                prediction_word
            )
            if 0 == len(words_overlapping_with_prediction_word):
                self.fp_boxes.zero_iou.append(prediction_word)
            for gt_box, boxes_info in words_overlapping_with_prediction_word:
                matched_prediction_words = self.__get_overlapped_prediction_words(
                    gt_box
                )
                found_better_match = False
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

        gt_boxes_to_be_removed_from_match_list = []

        for prediction_word, intersections in self.prediction_to_many_gt_boxes_map:
            valid_intersections, invalid_intersections = (
                refine_prediction_to_many_gt_boxes(prediction_word, intersections)
            )
            if valid_intersections:
                matched_gt_words = [
                    gt_box for (gt_box, boxes_info) in valid_intersections
                ]
                gt_boxes_to_be_removed_from_match_list.extend(matched_gt_words)
                unified_gt_word = unify_words(
                    matched_gt_words,
                    add_space_between_words=self.add_space_between_merged_gt_words,
                )
                self.gt_boxes_that_were_merged.extend(matched_gt_words)
                unified_gt_word["matched"] = True
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
                if gt_box["matched"]:
                    self.gt_boxes_that_are_fn_after_refinement.append(gt_box)
                else:
                    pass
        self.gt_with_predictions = []

        temp_gt_pred_list = [
            (gt_word, intersections_list)
            for (gt_word, intersections_list) in gt_to_predictions_tmp
            if not gt_word["location"]
            in [b["location"] for b in gt_boxes_to_be_removed_from_match_list]
        ]
        final_gt_pred_list = [
            (gt_word, intersections_list)
            for (gt_word, intersections_list) in temp_gt_pred_list
            if not gt_word["location"]
            in [b["location"] for b in self.gt_boxes_that_are_fn_after_refinement]
        ]
        for gt_word_item in self.gt_boxes_that_are_fn_after_refinement:
            self.false_negatives.append(gt_word_item)
            gt_word_item["matched"] = False
        self.gt_with_predictions = final_gt_pred_list

        for gt_word, intersections in self.gt_with_predictions:
            if len(intersections) > 1:
                prediction_boxes = [box for (box, boxes_info) in intersections]
                unified_prediction_word = unify_words(
                    prediction_boxes,
                    add_space_between_words=self.add_space_between_merged_prediction_words,
                )
                self.prediction_boxes_that_were_merged.extend(prediction_boxes)
                self.gt_and_prediction_matches.append(
                    (gt_word, unified_prediction_word)
                )
            else:
                prediction_word_item = intersections[0][0]
                self.gt_and_prediction_matches.append((gt_word, prediction_word_item))

        new_fp = [
            pred_word
            for pred_word in self.false_positives
            if not pred_word["location"]
            in [b["location"] for b in self.prediction_boxes_that_were_merged]
        ]
        self.false_positives = new_fp
        new_fn = [
            gt_word_item
            for gt_word_item in self.false_negatives
            if not gt_word_item["location"]
            in [b["location"] for b in gt_boxes_to_be_removed_from_match_list]
        ]
        self.false_negatives = new_fn

        zero_iou_fn = [
            word
            for word in self.fn_boxes.zero_iou
            if word["location"] in [b["location"] for b in self.false_negatives]
        ]
        ambiguous_match_fn = [
            word
            for word in self.fn_boxes.ambiguous_match
            if word["location"] in [b["location"] for b in self.false_negatives]
        ]
        low_iou_fn = [
            word
            for word in self.fn_boxes.low_iou
            if word["location"] in [b["location"] for b in self.false_negatives]
        ]
        low_iou_fn = [
            word
            for word in low_iou_fn
            if word["location"] not in [b["location"] for b in ambiguous_match_fn]
        ]
        self.fn_boxes = BoxesTypes(zero_iou_fn, low_iou_fn, ambiguous_match_fn)

        zero_iou_fp = [
            word
            for word in self.fp_boxes.zero_iou
            if word["location"] in [b["location"] for b in self.false_positives]
        ]
        low_iou_fp = [
            word
            for word in self.fp_boxes.low_iou
            if word["location"] in [b["location"] for b in self.false_positives]
        ]
        ambiguous_match_fp = [
            word
            for word in self.fp_boxes.ambiguous_match
            if word["location"] in [b["location"] for b in self.false_positives]
        ]
        self.fp_boxes = BoxesTypes(zero_iou_fp, low_iou_fp, ambiguous_match_fp)

        self.gt_words_unified, self.prediction_words_unified = [], []
        self.gt_words_unified.extend(
            [gt_w for (gt_w, pred_w) in self.gt_and_prediction_matches]
        )
        self.gt_words_unified.extend(self.false_negatives)

        self.prediction_words_unified.extend(
            [pred_w for (gt_w, pred_w) in self.gt_and_prediction_matches]
        )
        self.prediction_words_unified.extend(self.false_positives)

        self.gt_json_unified["images"][0]["words"] = self.gt_words_unified
        self.prediction_json_unified["images"][0][
            "words"
        ] = self.prediction_words_unified
        self.gt_words = self.gt_words_unified

    @staticmethod
    def split_fn_by_ignore_chars(false_negatives):
        fn_regular = []
        fn_ignore_chars = []
        for w in false_negatives:
            if w["word"] in "-,.*_+=":
                fn_ignore_chars.append(w)
            else:
                fn_regular.append(w)
        return fn_regular, fn_ignore_chars

    @staticmethod
    def split_fn_by_single_chars(false_negatives):
        fn_regular = []
        fn_ignore_chars = []
        for w in false_negatives:
            if 1 == len(w["word"]):
                fn_ignore_chars.append(w)
            else:
                fn_regular.append(w)
        return fn_regular, fn_ignore_chars

    def calc_metrics(self):
        text_length_false_positives = EPS
        sum_edit_distance_intersection_sensitive = EPS
        sum_edit_distance_intersection_insensitive = EPS
        sum_max_length_intersection = EPS
        num_matched_pairs = 0
        sum_norm_ed = 0
        sum_correct_orientation = 0
        num_of_orientation_set = 0
        sum_norm_ed_insensitive = 0
        matched_pairs_perfect_recognition_sensitive_weighted_sum = 0
        matched_pairs_perfect_recognition_insensitive_weighted_sum = 0
        matched_pairs_imperfect_recognition_sensitive_weighted_sum = 0
        matched_pairs_imperfect_recognition_insensitive_weighted_sum = 0
        self.metrics_per_prediction_word = []

        for gt_word, prediction_word in self.gt_and_prediction_matches:
            iou = -1
            gt_text = gt_word["word"]
            prediction_text = prediction_word["word"]
            num_matched_pairs += 1
            max_word_length = max(len(prediction_text), len(gt_text), EPS)
            sum_max_length_intersection += max_word_length
            if max_word_length == EPS:
                continue

            is_orientation_correct = 0.0
            if "vertical" in gt_word and "vertical" in prediction_word:
                if gt_word["vertical"] == prediction_word["vertical"]:
                    is_orientation_correct = 1.0
                sum_correct_orientation += is_orientation_correct
                num_of_orientation_set += 1

            word_edit_distance_insensitive = calculate_edit_distance(
                gt_text.upper(), prediction_text.upper(), self.string_normalize_map
            )
            norm_ed_insensitive_val = word_edit_distance_insensitive / max_word_length
            sum_norm_ed_insensitive += norm_ed_insensitive_val
            matched_pairs_perfect_recognition_insensitive_weighted_sum += (
                gt_word["word-weight"] if 0 == word_edit_distance_insensitive else 0
            )
            matched_pairs_imperfect_recognition_insensitive_weighted_sum += (
                0 if 0 == word_edit_distance_insensitive else gt_word["word-weight"]
            )
            sum_edit_distance_intersection_insensitive += word_edit_distance_insensitive

            word_edit_distance = calculate_edit_distance(
                gt_text, prediction_text, self.string_normalize_map
            )
            norm_ed_val = word_edit_distance / max_word_length
            sum_norm_ed += norm_ed_val
            matched_pairs_perfect_recognition_sensitive_weighted_sum += (
                gt_word["word-weight"] if 0 == word_edit_distance else 0
            )
            matched_pairs_imperfect_recognition_sensitive_weighted_sum += (
                0 if 0 == word_edit_distance else gt_word["word-weight"]
            )
            sum_edit_distance_intersection_sensitive += word_edit_distance

            prediction_cell_metric = get_metrics_per_single_word(
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
            prediction_cell_metric = get_metrics_per_single_word(fp_word)
            self.metrics_per_prediction_word.append(prediction_cell_metric)

        fn_not_ignored_chars, fn_ignore_chars = self.split_fn_by_ignore_chars(
            self.false_negatives
        )
        fn_not_single_chars, fn_single_chars = self.split_fn_by_single_chars(
            self.false_negatives
        )
        unmatched_len = [len(w["word"]) for w in self.false_negatives]
        text_length_false_negatives = sum(unmatched_len) if unmatched_len else EPS

        text_length_false_positives = (
            sum(len(w["word"]) for w in self.false_positives)
            if self.false_positives
            else EPS
        )

        denominator_union = (
            sum_max_length_intersection
            + text_length_false_positives
            + text_length_false_negatives
        )
        image_avg_edit_distance_union_sensitive = (
            (
                sum_edit_distance_intersection_sensitive
                + text_length_false_positives
                + text_length_false_negatives
            )
            / denominator_union
            if denominator_union > 0
            else 0
        )

        image_avg_edit_distance_union_insensitive = (
            (
                sum_edit_distance_intersection_insensitive
                + text_length_false_positives
                + text_length_false_negatives
            )
            / denominator_union
            if denominator_union > 0
            else 0
        )

        image_avg_edit_distance_intersection_sensitive = (
            (sum_edit_distance_intersection_sensitive / sum_max_length_intersection)
            if sum_max_length_intersection > 0
            else 0
        )

        image_avg_edit_distance_intersection_insensitive = (
            (sum_edit_distance_intersection_insensitive / sum_max_length_intersection)
            if sum_max_length_intersection > 0
            else 0
        )

        num_false_positives = len(self.false_positives)
        num_false_negatives_unfiltered = len(self.false_negatives)
        num_false_negatives_no_ignored_chars = len(fn_not_ignored_chars)
        num_false_negatives_no_single_chars = len(fn_not_single_chars)
        num_gt_cells = len(self.gt_json_unified["images"][0]["words"])
        num_prediction_cells = len(self.prediction_json_unified["images"][0]["words"])
        num_true_positive_matches = len(self.gt_and_prediction_matches)

        detection_precision = num_true_positive_matches / max(EPS, num_prediction_cells)
        detection_recall = num_true_positive_matches / max(EPS, num_gt_cells)
        detection_f1_score = (2 * detection_recall * detection_precision) / max(
            detection_recall + detection_precision, EPS
        )

        denominator_wa_intersection_sensitive = max(
            EPS,
            matched_pairs_perfect_recognition_sensitive_weighted_sum
            + matched_pairs_imperfect_recognition_sensitive_weighted_sum,
        )
        word_accuracy_intersection_case_sensitive = (
            (
                matched_pairs_perfect_recognition_sensitive_weighted_sum
                / denominator_wa_intersection_sensitive
            )
            if denominator_wa_intersection_sensitive > 0
            else 0
        )

        denominator_wa_intersection_insensitive = max(
            EPS,
            matched_pairs_perfect_recognition_insensitive_weighted_sum
            + matched_pairs_imperfect_recognition_insensitive_weighted_sum,
        )
        word_accuracy_intersection_case_insensitive = (
            (
                matched_pairs_perfect_recognition_insensitive_weighted_sum
                / denominator_wa_intersection_insensitive
            )
            if denominator_wa_intersection_insensitive > 0
            else 0
        )

        denominator_wa_union_sensitive = max(
            EPS,
            matched_pairs_perfect_recognition_sensitive_weighted_sum
            + matched_pairs_imperfect_recognition_sensitive_weighted_sum
            + num_false_positives
            + num_false_negatives_unfiltered,
        )
        word_accuracy_union_case_sensitive = (
            (
                matched_pairs_perfect_recognition_sensitive_weighted_sum
                / denominator_wa_union_sensitive
            )
            if denominator_wa_union_sensitive > 0
            else 0
        )

        denominator_wa_union_insensitive = max(
            EPS,
            matched_pairs_perfect_recognition_insensitive_weighted_sum
            + matched_pairs_imperfect_recognition_insensitive_weighted_sum
            + num_false_positives
            + num_false_negatives_unfiltered,
        )
        word_accuracy_union_case_insensitive = (
            (
                matched_pairs_perfect_recognition_insensitive_weighted_sum
                / denominator_wa_union_insensitive
            )
            if denominator_wa_union_insensitive > 0
            else 0
        )

        calculated_norm_ed_tp_only = sum_norm_ed / max(EPS, num_true_positive_matches)
        denominator_norm_ed_all_cells = max(
            EPS,
            num_true_positive_matches
            + num_false_positives
            + num_false_negatives_unfiltered,
        )
        calculated_norm_ed_all_cells = (
            (sum_norm_ed + num_false_positives + num_false_negatives_unfiltered)
            / denominator_norm_ed_all_cells
            if denominator_norm_ed_all_cells > 0
            else 0
        )

        orientation_accuracy = sum_correct_orientation / max(1, num_of_orientation_set)

        metrics_summary_dict = {
            "num_prediction_cells": num_prediction_cells,
            "number_of_gt_cells": num_gt_cells,
            "number_of_false_positive_detections": num_false_positives,
            "Norm_ED (TP-Only)": 100 * (1 - calculated_norm_ed_tp_only),
            "Norm_ED (All-cells)": 100 * (1 - calculated_norm_ed_all_cells),
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
            "word_Insertions": num_false_negatives_unfiltered,
            "word_Deletions": num_false_positives,
            "word_substitutions_case_sensitive": matched_pairs_imperfect_recognition_sensitive_weighted_sum,
            "word_Hits_case_sensitive": matched_pairs_perfect_recognition_sensitive_weighted_sum,
            "word_substitutions_case_insensitive": matched_pairs_imperfect_recognition_insensitive_weighted_sum,
            "word_Hits_case_insensitive": matched_pairs_perfect_recognition_insensitive_weighted_sum,
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
        self.image_data_for_end2end = metrics_summary_dict
        return metrics_summary_dict

    def get_metrics_per_word(self):
        return self.metrics_per_prediction_word

    def __reset_matched_property(self):
        for word in self.gt_words_original:
            word["matched"] = False
        for word in self.prediction_words_original:
            word["matched"] = False
