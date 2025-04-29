from typing import List

from docling_core.types import DoclingDocument

from docling_eval.evaluators.ocr.bbox_utils import (
    BoxesTypes,
    box2key,
    create_polygon_from_rect,
    detection_match_condition,
    get_metrics_per_single_word,
    match_gt_2_model,
    my_edit_distance,
    refine_detection_to_many_gt_boxes,
    unify_words,
)

__eps = 1.0e-6


def _extract_words_from_docling(doc: DoclingDocument) -> List[dict]:
    """Extracts words in the expected format from a DoclingDocument."""
    words = []
    for text_item in doc.texts:
        if text_item.text and text_item.prov:
            primary_prov = text_item.prov[0]
            bbox = primary_prov.bbox
            location = {
                "top": bbox.t,
                "left": bbox.l,
                "right": bbox.r,
                "bottom": bbox.b,
                "width": bbox.r - bbox.l,
                "height": bbox.b - bbox.t,
            }
            polygon = create_polygon_from_rect(location)
            words.append(
                {
                    "crop_path": "",
                    "unreadable": False,
                    "vertical": False,
                    "ROI": False,
                    "handwritten": False,
                    "word": text_item.text,
                    "location": location,
                    "polygon": polygon,
                }
            )
    return words


class OCRPerformanceEvaluator:
    """
    Evaluates OCR model performance by comparing predicted boxes and text with ground truth,
    using DoclingDocument format.
    """

    def __init__(
        self,
        pred_doc: DoclingDocument,
        gt_doc: DoclingDocument,
        add_space_between_merged_gt_words=True,
        add_space_between_merged_model_words=True,
        string_normalize_map=None,
    ):
        """
        Args:
            pred_doc: DoclingDocument containing model predictions
            gt_doc: DoclingDocument containing ground truth data
            add_space_between_merged_gt_words: Add spaces when merging GT words
            add_space_between_merged_model_words: Add spaces when merging model words
            string_normalize_map: Character normalization mapping
        """
        # Removed self.pred_dict and self.gt_dict
        self.add_space_between_merged_gt_words = add_space_between_merged_gt_words
        self.add_space_between_merged_model_words = add_space_between_merged_model_words
        self.string_normalize_map = string_normalize_map

        # Extract words from DoclingDocuments
        self.model_words_original = _extract_words_from_docling(pred_doc)
        self.gt_words_original = _extract_words_from_docling(gt_doc)

        # Removed deepcopy of dicts and unified json structures
        # self.gt_json_unified = copy.deepcopy(gt_dict)
        # self.model_json_unified = copy.deepcopy(pred_dict)

        self.__reset_matched_property()
        self.gt_boxes_that_are_fn_after_refinement = []
        self.model_boxes_that_are_fp_after_refinement = []
        self.model_boxes_that_were_merged = []
        self.gt_boxes_that_were_merged = []
        self.fp_boxes = BoxesTypes([], [], [])
        self.fn_boxes = BoxesTypes([], [], [])
        self.__eps = 1.0e-6
        self.__do_evaluation()

    def __do_evaluation(self):
        gt2model_boxes, model2gt_boxes = match_gt_2_model(
            self.gt_words_original, self.model_words_original
        )
        self.gt2model_boxes = gt2model_boxes
        self.model2gt_boxes = model2gt_boxes
        self.num_model_boxes_that_do_not_intersect_with_a_gt = sum(
            [0 == len(v) for k, v in self.model2gt_boxes.items()]
        )
        self.num_gt_boxes_that_do_not_intersect_with_a_gt = sum(
            [0 == len(v) for k, v in self.gt2model_boxes.items()]
        )
        self.__create_unified_words_from_gt()

    def __get_overlapped_gt_words(self, word):
        box_key = box2key(word["location"])
        if box_key in self.model2gt_boxes:
            return self.model2gt_boxes[box_key]
        return []

    def __get_overlapped_model_words(self, gt_word):
        gt_box_key = box2key(gt_word["location"])
        if gt_box_key in self.gt2model_boxes:
            return self.gt2model_boxes[gt_box_key]
        return []

    def __create_unified_words_from_gt(self):
        # Initialize word weights and containers
        for w in self.model_words_original:
            w["word-weight"] = 1
        for w in self.gt_words_original:
            w["word-weight"] = 1

        gt_to_detections_tmp = []
        self.false_negatives = []
        self.false_positives = []
        self.gt_and_detection_matches = []
        self.detection_to_many_gtboxes = []

        # A. Map GT to detections and detection to GTs
        for gt_word in self.gt_words_original:
            intersections = []
            words_overlapping_with_gt_word = self.__get_overlapped_model_words(gt_word)
            if len(words_overlapping_with_gt_word) == 0:
                self.fn_boxes.zero_iou.append(gt_word)

            for model_box, boxes_info in words_overlapping_with_gt_word:
                matched_gt_words = self.__get_overlapped_gt_words(model_box)
                found_better_match = False
                for gt_box_2, boxes_info_2 in matched_gt_words:
                    if boxes_info.iou < boxes_info_2.iou:
                        found_better_match = (
                            True  # Another gt_box has higher match to the model_box
                        )

                if not found_better_match and detection_match_condition(boxes_info):
                    intersections.append((model_box, boxes_info))

            if len(intersections) == 0:
                self.false_negatives.append(gt_word)
            else:
                gt_to_detections_tmp.append((gt_word, intersections))
                gt_word["matched"] = True

            if len(words_overlapping_with_gt_word) > 0 and len(intersections) == 0:
                self.fn_boxes.low_iou.append(gt_word)

        for model_word in self.model_words_original:
            intersections = []
            words_overlapping_with_model_word = self.__get_overlapped_gt_words(
                model_word
            )

            if len(words_overlapping_with_model_word) == 0:
                self.fp_boxes.zero_iou.append(model_word)

            for gt_box, boxes_info in words_overlapping_with_model_word:
                matched_model_words = self.__get_overlapped_model_words(gt_box)
                found_better_match = False
                for model_box_2, boxes_info_2 in matched_model_words:
                    if boxes_info.iou < boxes_info_2.iou:
                        found_better_match = True

                if not found_better_match and detection_match_condition(boxes_info):
                    intersections.append((gt_box, boxes_info))

            if len(intersections) == 0:
                self.false_positives.append(model_word)
            elif len(intersections) > 1:
                self.detection_to_many_gtboxes.append((model_word, intersections))

            if len(words_overlapping_with_model_word) > 0 and len(intersections) == 0:
                self.fp_boxes.low_iou.append(model_word)

        # B. Merge many GTs to one detection box when applicable
        gt_boxes_to_be_removed_from_match_list = (
            []
        )  # GTs that will no longer have a 1-1 or 1-m gt-detection

        for model_word, intersections in self.detection_to_many_gtboxes:
            # Refine the gt_boxes that can be considered valid for a detection_box
            valid_intersections, invalid_intersections = (
                refine_detection_to_many_gt_boxes(model_word, intersections)
            )

            if valid_intersections:
                matched_gt_words = [
                    gt_box for (gt_box, boxes_info) in valid_intersections
                ]
                gt_boxes_to_be_removed_from_match_list.extend(matched_gt_words)

                # Merge the gt words
                unified_gt_word = unify_words(
                    matched_gt_words,
                    add_space_between_words=self.add_space_between_merged_gt_words,
                )
                self.gt_boxes_that_were_merged.extend(matched_gt_words)
                unified_gt_word["matched"] = True
                self.gt_and_detection_matches.append((unified_gt_word, model_word))
            else:
                self.false_positives.append(model_word)
                self.model_boxes_that_are_fp_after_refinement.append(model_word)
                self.fp_boxes.ambiguous_match.append(model_word)

            for gt_box, _ in invalid_intersections:
                self.fn_boxes.ambiguous_match.append(gt_box)
                if gt_box["matched"]:
                    self.gt_boxes_that_are_fn_after_refinement.append(gt_box)

        # Filter matches to remove gt_boxes that were merged or marked as FN after refinement
        self.gt_with_detections = []
        filtered_list = [
            (gt_word, intersections)
            for (gt_word, intersections) in gt_to_detections_tmp
            if not gt_word["location"]
            in [b["location"] for b in gt_boxes_to_be_removed_from_match_list]
        ]
        filtered_list = [
            (gt_word, intersections)
            for (gt_word, intersections) in filtered_list
            if not gt_word["location"]
            in [b["location"] for b in self.gt_boxes_that_are_fn_after_refinement]
        ]

        for gt_word in self.gt_boxes_that_are_fn_after_refinement:
            self.false_negatives.append(gt_word)
            gt_word["matched"] = False

        self.gt_with_detections = filtered_list

        # C. Handle one or many detections to one GT
        for gt_word, intersections in self.gt_with_detections:
            if len(intersections) > 1:
                model_boxes = [box for (box, boxes_info) in intersections]
                unified_model_word = unify_words(
                    model_boxes,
                    add_space_between_words=self.add_space_between_merged_model_words,
                )
                self.model_boxes_that_were_merged.extend(model_boxes)
                self.gt_and_detection_matches.append((gt_word, unified_model_word))
            else:
                model_word = intersections[0][0]
                self.gt_and_detection_matches.append((gt_word, model_word))

        # Update the FP and FN lists
        new_fp = [
            model_word
            for model_word in self.false_positives
            if not model_word["location"]
            in [b["location"] for b in self.model_boxes_that_were_merged]
        ]
        self.false_positives = new_fp

        new_fn = [
            gt_word
            for gt_word in self.false_negatives
            if not gt_word["location"]
            in [b["location"] for b in gt_boxes_to_be_removed_from_match_list]
        ]
        self.false_negatives = new_fn

        # Update box type classifications
        self._update_box_types()

        # Prepare unified words for final evaluation
        self.gt_words_unified = []
        self.model_words_unified = []

        self.gt_words_unified.extend(
            [gt_word for (gt_word, model_word) in self.gt_and_detection_matches]
        )
        self.gt_words_unified.extend(self.false_negatives)

        self.model_words_unified.extend(
            [model_word for (gt_word, model_word) in self.gt_and_detection_matches]
        )
        self.model_words_unified.extend(self.false_positives)

        # Update the JSON with unified words
        # self.gt_json_unified["images"][0]["words"] = self.gt_words_unified
        # self.model_json_unified["images"][0]["words"] = self.model_words_unified

        # Initialize for backward compatibility
        self.gt_words = self.gt_words_unified

    def _update_box_types(self):
        """Update box type classifications based on final matching results"""
        # Update FN box types
        zero_iou = [
            word
            for word in self.fn_boxes.zero_iou
            if word["location"] in [b["location"] for b in self.false_negatives]
        ]
        ambiguous_match = [
            word
            for word in self.fn_boxes.ambiguous_match
            if word["location"] in [b["location"] for b in self.false_negatives]
        ]
        low_iou = [
            word
            for word in self.fn_boxes.low_iou
            if word["location"] in [b["location"] for b in self.false_negatives]
            and word["location"] not in [b["location"] for b in ambiguous_match]
        ]
        self.fn_boxes = BoxesTypes(zero_iou, low_iou, ambiguous_match)

        # Update FP box types
        zero_iou = [
            word
            for word in self.fp_boxes.zero_iou
            if word["location"] in [b["location"] for b in self.false_positives]
        ]
        low_iou = [
            word
            for word in self.fp_boxes.low_iou
            if word["location"] in [b["location"] for b in self.false_positives]
        ]
        ambiguous_match = [
            word
            for word in self.fp_boxes.ambiguous_match
            if word["location"] in [b["location"] for b in self.false_positives]
        ]
        self.fp_boxes = BoxesTypes(zero_iou, low_iou, ambiguous_match)

        # Verify box type counts match
        fn_boxes_sum = sum([len(a) for a in self.fn_boxes])
        if fn_boxes_sum != len(self.false_negatives):
            print(
                f"Warning: FN boxes count mismatch: {fn_boxes_sum} != {len(self.false_negatives)}"
            )

        fp_boxes_sum = sum([len(a) for a in self.fp_boxes])
        if fp_boxes_sum != len(self.false_positives):
            print(
                f"Warning: FP boxes count mismatch: {fp_boxes_sum} != {len(self.false_positives)}"
            )

    @staticmethod
    def split_fn_by_ignore_chars(false_negatives):
        """Split false negatives into regular and ignored characters"""
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
        """Split false negatives into regular and single character words"""
        fn_regular = []
        fn_single_chars = []
        for w in false_negatives:
            if len(w["word"]) == 1:
                fn_single_chars.append(w)
            else:
                fn_regular.append(w)
        return fn_regular, fn_single_chars

    def calc_metrics(self):
        """Calculate performance metrics based on word matches"""
        # Initialize counters
        sum_edit_distance_sensitive = self.__eps
        sum_edit_distance_insensitive = self.__eps
        sum_max_length_intersection = self.__eps
        detected_words = 0
        sum_norm_ed = 0
        sum_correct_orientation = 0
        num_of_orientation_set = 0
        sum_norm_ed_insensitive = 0
        detected_words_with_perfect_recognition_sensitive = 0
        detected_words_with_perfect_recognition_insensitive = 0
        detected_words_without_perfect_recognition_sensitive = 0
        detected_words_without_perfect_recognition_insensitive = 0
        self.metrics_per_detections = []

        # Process matches to calculate metrics
        for gt_word, model_word in self.gt_and_detection_matches:
            iou = -1
            gt_text = gt_word["word"]
            detection_text = model_word["word"]
            detected_words += 1
            max_word_length = max(len(detection_text), len(gt_text))
            sum_max_length_intersection += max_word_length
            if max_word_length == 0:
                continue

            # Check orientation matching if available
            is_orientation_correct = 0.0
            if "vertical" in gt_word and "vertical" in model_word:
                if gt_word["vertical"] == model_word["vertical"]:
                    is_orientation_correct = 1.0
                sum_correct_orientation += is_orientation_correct
                num_of_orientation_set += 1

            # Calculate case insensitive metrics
            word_edit_distance_insensitive = my_edit_distance(
                gt_text.upper(), detection_text.upper(), self.string_normalize_map
            )
            norm_ed_insensitive = word_edit_distance_insensitive / max_word_length
            sum_norm_ed_insensitive += norm_ed_insensitive

            if word_edit_distance_insensitive == 0:
                detected_words_with_perfect_recognition_insensitive += gt_word[
                    "word-weight"
                ]
            else:
                detected_words_without_perfect_recognition_insensitive += gt_word[
                    "word-weight"
                ]

            sum_edit_distance_insensitive += word_edit_distance_insensitive

            # Calculate case sensitive metrics
            word_edit_distance = my_edit_distance(
                gt_text, detection_text, self.string_normalize_map
            )
            norm_ed = word_edit_distance / max_word_length
            sum_norm_ed += norm_ed

            if word_edit_distance == 0:
                detected_words_with_perfect_recognition_sensitive += gt_word[
                    "word-weight"
                ]
            else:
                detected_words_without_perfect_recognition_sensitive += gt_word[
                    "word-weight"
                ]

            sum_edit_distance_sensitive += word_edit_distance

            # Store metrics for this detection
            detection_metrics = get_metrics_per_single_word(
                model_word,
                gt_word,
                iou,
                word_edit_distance,
                word_edit_distance_insensitive,
                norm_ed,
                norm_ed_insensitive,
                is_orientation_correct,
            )
            self.metrics_per_detections.append(detection_metrics)

        # Add metrics for false positive detections
        for false_positive in self.false_positives:
            detection_metrics = get_metrics_per_single_word(false_positive)
            self.metrics_per_detections.append(detection_metrics)

        # Process false negatives
        fn_not_ignored_chars, fn_ignore_chars = self.split_fn_by_ignore_chars(
            self.false_negatives
        )
        fn_not_single_chars, fn_single_chars = self.split_fn_by_single_chars(
            self.false_negatives
        )

        unmatched_words = [w["word"] for w in self.false_negatives]
        unmatched_lengths = [len(w) for w in unmatched_words]
        text_length_false_negatives = sum(unmatched_lengths)
        text_length_false_positives = sum(len(w["word"]) for w in self.false_positives)

        # Calculate edit distance metrics
        total_text_length = (
            sum_max_length_intersection
            + text_length_false_positives
            + text_length_false_negatives
        )
        edit_distance_union_sensitive = (
            sum_edit_distance_sensitive
            + text_length_false_positives
            + text_length_false_negatives
        )
        edit_distance_union_insensitive = (
            sum_edit_distance_insensitive
            + text_length_false_positives
            + text_length_false_negatives
        )

        image_avg_edit_distance_union_sensitive = (
            edit_distance_union_sensitive / total_text_length
        )
        image_avg_edit_distance_union_insensitive = (
            edit_distance_union_insensitive / total_text_length
        )
        image_avg_edit_distance_intersection_sensitive = (
            sum_edit_distance_sensitive / sum_max_length_intersection
        )
        image_avg_edit_distance_intersection_insensitive = (
            sum_edit_distance_insensitive / sum_max_length_intersection
        )

        # Count statistics
        word_deletions = len(self.false_positives)
        word_insertions = len(self.false_negatives)
        false_negatives_without_filtering = len(self.false_negatives)
        false_negatives_without_ignored_chars = len(fn_not_ignored_chars)
        false_negatives_without_single_chars = len(fn_not_single_chars)

        # Calculate precision and recall
        num_gt_cells = len(self.gt_words_unified)
        num_detected_cells = len(self.model_words_unified)
        num_true_positive_detections = len(self.gt_and_detection_matches)

        detection_precision = num_true_positive_detections / max(
            self.__eps, num_detected_cells
        )
        detection_recall = num_true_positive_detections / max(self.__eps, num_gt_cells)
        detection_f1_score = (2 * detection_recall * detection_precision) / max(
            detection_recall + detection_precision, self.__eps
        )

        # Calculate word accuracy metrics
        word_accuracy_intersection_sensitive = (
            detected_words_with_perfect_recognition_sensitive
            / max(
                self.__eps,
                detected_words_with_perfect_recognition_sensitive
                + detected_words_without_perfect_recognition_sensitive,
            )
        )

        word_accuracy_intersection_insensitive = (
            detected_words_with_perfect_recognition_insensitive
            / max(
                self.__eps,
                detected_words_with_perfect_recognition_insensitive
                + detected_words_without_perfect_recognition_insensitive,
            )
        )

        word_accuracy_union_sensitive = (
            detected_words_with_perfect_recognition_sensitive
            / max(
                self.__eps,
                detected_words_with_perfect_recognition_sensitive
                + detected_words_without_perfect_recognition_sensitive
                + word_deletions
                + false_negatives_without_filtering,
            )
        )

        word_accuracy_union_insensitive = (
            detected_words_with_perfect_recognition_insensitive
            / max(
                self.__eps,
                detected_words_with_perfect_recognition_insensitive
                + detected_words_without_perfect_recognition_insensitive
                + word_deletions
                + false_negatives_without_filtering,
            )
        )

        # Calculate normalized edit distance metrics
        norm_ed_tp_only = sum_norm_ed / max(self.__eps, num_true_positive_detections)
        norm_ed_all = (
            sum_norm_ed + word_deletions + false_negatives_without_filtering
        ) / max(
            self.__eps,
            num_true_positive_detections
            + word_deletions
            + false_negatives_without_filtering,
        )

        orientation_accuracy = sum_correct_orientation / max(1, num_of_orientation_set)

        metrics = {
            "number_of_detected_cells": num_detected_cells,
            "number_of_gt_cells": num_gt_cells,
            "number_of_false_positive_detections": word_deletions,
            "Norm_ED (TP-Only)": 100 * (1 - norm_ed_tp_only),
            "Norm_ED (All-cells)": 100 * (1 - norm_ed_all),
            "number_of_true_positive_detections": num_true_positive_detections,
            "number_of_false_negative_detections": false_negatives_without_filtering,
            "without_ignored_chars_false_negatives": false_negatives_without_ignored_chars,
            "without_single_chars_false_negatives": false_negatives_without_single_chars,
            "detection_precision": 100 * detection_precision,
            "detection_recall": 100 * detection_recall,
            "detection_f1_score": 100 * detection_f1_score,
            "word_accuracy_intersection_case_sensitive": 100
            * word_accuracy_intersection_sensitive,
            "word_accuracy_intersection_case_insensitive": 100
            * word_accuracy_intersection_insensitive,
            "word_accuracy_union_case_sensitive": 100 * word_accuracy_union_sensitive,
            "word_accuracy_union_case_insensitive": 100
            * word_accuracy_union_insensitive,
            "edit_score_intersection_case_sensitive_not_avg_over_words": 100
            * (1 - image_avg_edit_distance_intersection_sensitive),
            "edit_score_intersection_case_insensitive_not_avg_over_words": 100
            * (1 - image_avg_edit_distance_intersection_insensitive),
            "edit_score_union_case_sensitive_not_avg_over_words": 100
            * (1 - image_avg_edit_distance_union_sensitive),
            "edit_score_union_case_insensitive_not_avg_over_words": 100
            * (1 - image_avg_edit_distance_union_insensitive),
            "sum_edit_distance_intersection_case_sensitive_not_avg_over_words": sum_edit_distance_sensitive,
            "sum_edit_distance_intersection_case_insensitive_not_avg_over_words": sum_edit_distance_insensitive,
            "sum_max_length_intersection": sum_max_length_intersection,
            "text_length_false_positives": text_length_false_positives,
            "text_length_false_negatives": text_length_false_negatives,
            "sum_norm_ed": sum_norm_ed,
            #
            "word_Insertions": false_negatives_without_filtering,
            "word_Deletions": word_deletions,
            "word_substitutions_case_sensitive": detected_words_without_perfect_recognition_sensitive,
            "word_Hits_case_sensitive": detected_words_with_perfect_recognition_sensitive,
            "word_substitutions_case_insensitive": detected_words_without_perfect_recognition_insensitive,
            "word_Hits_case_insensitive": detected_words_with_perfect_recognition_insensitive,
            "num_model_boxes_that_do_not_intersect_with_a_gt": len(
                self.fp_boxes.zero_iou
            ),
            "num_gt_boxes_that_do_not_intersect_with_a_gt": len(self.fn_boxes.zero_iou),
            "num_gt_boxes_that_are_fn_after_refinement": len(
                self.fn_boxes.ambiguous_match
            ),
            "num_model_boxes_that_are_fp_after_refinement": len(
                self.fp_boxes.ambiguous_match
            ),
            "num_gt_boxes_with_low_iou": len(self.fn_boxes.low_iou),
            "num_model_boxes_with_low_iou": len(self.fp_boxes.low_iou),
            "model_boxes_that_were_merged": len(self.model_boxes_that_were_merged),
            "gt_boxes_that_were_merged": len(self.gt_boxes_that_were_merged),
            "orientation_accuracy": 100 * orientation_accuracy,
        }

        """
        
        """
        self.image_data_for_end2end = metrics
        return metrics

    def get_metrics_per_word(self):
        return self.metrics_per_detections

    def get_model_boxes_that_were_merged(self):
        return self.model_boxes_that_were_merged

    def get_false_positives(self):
        self.num_model_boxes_that_do_not_intersect_with_a_gt = sum(
            [0 == len(v) for k, v in self.model2gt_boxes.items()]
        )
        self.num_gt_boxes_that_do_not_intersect_with_a_gt = sum(
            [0 == len(v) for k, v in self.gt2model_boxes.items()]
        )

    def get_image_data_for_end2end(self, calc_character_confusion_matrix=False):
        return (
            self.image_data_for_end2end,
            self.image_data_for_end2end,
            self.image_data_for_end2end,
        )

    def __reset_matched_property(self):
        for word in self.gt_words_original:
            word["matched"] = False
        for word in self.model_words_original:
            word["matched"] = False

    def get_detection_false_positives(self):
        metrics_per_word = self.get_metrics_per_word()
        false_positives_list = [
            word for word in metrics_per_word if not word["correct_detection"]
        ]
        return false_positives_list

    def get_detections_with_highest_edit_score(
        self, max_number_of_detections_to_return
    ):
        detection_true_positives = self.get_detection_true_positives()
        sorted_words = sorted(
            detection_true_positives, key=lambda k: k["edit_distance"], reverse=True
        )
        return sorted_words[:max_number_of_detections_to_return]
