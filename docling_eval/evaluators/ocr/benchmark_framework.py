import copy
import traceback
from collections import namedtuple

import numpy as np
from tqdm import tqdm

from docling_eval.evaluators.ocr.performance_calculator import (
    ModelPerformanceCalculator,
)

EPS = 1.0e-6

Metric = namedtuple("Metric", ["benchmark_internal_name", "readable_name", "function"])
metrics = [
    Metric("detection_f1_score", "F1", np.mean),
    Metric("detection_precision", "Precision", np.mean),
    Metric("detection_recall", "Recall", np.mean),
    Metric("Norm_ED (All-cells)", "Norm_ED (All-cells)", np.mean),
    Metric(
        "edit_score_union_case_sensitive_not_avg_over_words",
        "Edit-score (All-cells)",
        np.mean,
    ),
    Metric("word_accuracy_union_case_sensitive", "Word-accuracy (All-cells)", np.mean),
    Metric("Norm_ED (TP-Only)", "Norm_ED (TP-Only)", np.mean),
    Metric("Norm_ED (Without FP)", "Norm_ED (Without FP)", np.mean),
    Metric(
        "word_accuracy_intersection_case_sensitive", "Word-accuracy (TP-Only)", np.mean
    ),
    Metric(
        "edit_score_intersection_case_sensitive_not_avg_over_words",
        "Edit-score (TP-Only)",
        np.mean,
    ),
    Metric("word_Hits_case_sensitive", "#Word-Hits", np.sum),
    Metric("word_substitutions_case_sensitive", "#Word-Substitutions", np.sum),
    Metric("word_Insertions", "#FN", np.sum),
    Metric("word_Deletions", "#FP", np.sum),
    Metric("num_gt_boxes_that_do_not_intersect_with_a_gt", "#FN - zero iou", np.sum),
    Metric(
        "without_ignored_chars_false_negatives",
        "#FN - zero iou, Ignoring some chars",
        np.sum,
    ),
    Metric(
        "without_single_chars_false_negatives",
        "#FN - zero iou, Ignoring single chars",
        np.sum,
    ),
    Metric("num_prediction_boxes_without_gt_intersection", "#FP - zero iou", np.sum),
    Metric("num_gt_boxes_with_low_iou", "#FN - low iou", np.sum),
    Metric("num_prediction_boxes_with_low_iou", "#FP - low iou", np.sum),
    Metric(
        "num_gt_boxes_that_are_fn_after_refinement", "#FN - ambiguous match", np.sum
    ),
    Metric(
        "num_prediction_boxes_fp_after_refinement", "#FP -  ambiguous match", np.sum
    ),
    Metric("prediction_boxes_that_were_merged", "#Pred. Words Merged", np.sum),
    Metric("gt_boxes_that_were_merged", "#GT Words Merged", np.sum),
    Metric("detection_f1_score", "#images", len),
    Metric(
        "Hmean: (Norm_ED [Without FP], precision)",
        "Hmean: (Norm_ED [Without FP], precision)",
        np.mean,
    ),
    Metric(
        "Norm_ED (Ignoring FP and some chars)",
        "Norm_ED (Ignoring FP and some chars)",
        np.mean,
    ),
    Metric("Norm_ED (Ignoring some chars)", "Norm_ED (Ignoring some chars)", np.mean),
    Metric(
        "Norm_ED (Ignoring single chars and FP)",
        "Norm_ED (Ignoring single chars and FP)",
        np.mean,
    ),
    Metric(
        "Norm_ED (Ignoring single chars)", "Norm_ED (Ignoring single chars)", np.mean
    ),
]

CHAR_MAP_FOR_NORMALIZATION = {
    "Ĳ": "IJ",
    "ĳ": "ij",
    "Æ": "AE",
    "Œ": "OE",
    "æ": "ae",
    "œ": "oe",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "’": "'",
    "‘": "'",
    "”": '"',
    "“": '"',
    "—": "-",
    "‚": ",",
    "º": "°",
    "᾽": "'",
    "᾿": "'",
    "›": ">",
    "‹": "<",
    "،": ",",
    "׃": ":",
    "״": '"',
    "\xa0": " ",
}


class IgnoreZonesFilter:

    def __init__(self):
        pass

    def filter_ignore_zones(self, prediction_data: dict, gt_data: dict):
        prediction_words = prediction_data["images"][0]["words"]
        gt_words = gt_data["images"][0]["words"]
        ignore_zones = []
        for gt_word in gt_words:
            if "ignore_zone" in gt_word and gt_word["ignore_zone"] == True:
                ignore_zones.append(gt_word)

        for gt_word_zone in ignore_zones:
            self._mark_intersecting_to_ignore(gt_word_zone["location"], gt_words)
            self._mark_intersecting_to_ignore(
                gt_word_zone["location"], prediction_words
            )

        filtered_gt_words = [
            i for i in gt_words if "to_remove" not in i or not i["to_remove"]
        ]
        filtered_prediction_words = [
            i for i in prediction_words if "to_remove" not in i or not i["to_remove"]
        ]

        for w in ignore_zones:
            l = w["location"]
            if "width" in l and "height" in l:
                l["right"] = l["left"] + l["width"]
                l["bottom"] = l["top"] + l["height"]
        return filtered_gt_words, filtered_prediction_words, ignore_zones

    def _mark_intersecting_to_ignore(self, ignore_zone_location, words_list):
        for word in words_list:
            if self._intersects(word["location"], ignore_zone_location):
                word["to_remove"] = True

    def _get_y_axis_overlap(self, rect1, rect2):
        rect1_bottom = rect1["top"] + rect1["height"]
        rect2_bottom = rect2["top"] + rect2["height"]
        y_overlap = max(
            0,
            min(rect1_bottom, rect2_bottom) - max(rect1["top"], rect2["top"]),
        )
        return y_overlap

    def _get_x_axis_overlap(self, rect1, rect2):
        rect1_right = rect1["left"] + rect1["width"]
        rect2_right = rect2["left"] + rect2["width"]
        x_overlap = max(
            0,
            min(rect1_right, rect2_right) - max(rect1["left"], rect2["left"]),
        )
        return x_overlap

    def _intersects(self, rect1, rect2):
        det_x_len = rect1["width"]
        det_y_len = rect1["height"]

        x_overlap = self._get_x_axis_overlap(rect1, rect2)
        y_overlap = self._get_y_axis_overlap(rect1, rect2)

        x_overlap_ratio = 0 if det_x_len == 0 else x_overlap / det_x_len
        y_overlap_ratio = 0 if det_y_len == 0 else y_overlap / det_y_len

        if y_overlap_ratio < 0.1 or x_overlap_ratio < 0.1:
            return False
        else:
            return True


class ModelBenchmark:

    def __init__(
        self,
        model_name,
        performance_calculator="general",
        ignore_zone_filter="default",
        add_space_between_merged_prediction_words=True,
        add_space_between_merged_gt_words=True,
        string_normalize_map=CHAR_MAP_FOR_NORMALIZATION,
    ):
        self.model_name = model_name
        self.add_space_between_merged_prediction_words = (
            add_space_between_merged_prediction_words
        )
        self.add_space_between_merged_gt_words = add_space_between_merged_gt_words
        self.string_normalize_map = string_normalize_map if string_normalize_map else {}

        self.prediction_json_list = []
        self.gt_json_list = []
        self.gt_json_not_unified_list = []
        self.prediction_json_not_unified_list = []
        self.metrics_per_image = []
        self.image_to_mpc_map = {}
        self.image_to_ignore_zones_map = {}
        self.performance_calculator_type = performance_calculator
        self.ignore_zone_filter_instance = IgnoreZonesFilter()

    def load_res_and_gt_files(
        self,
        gt_words,
        pred_words,
    ):
        self.gt_json_not_unified_list.append(gt_words)
        self.prediction_json_not_unified_list.append(pred_words)

    def run_benchmark(self, min_intersection_to_match=0.5):
        desc = "Calculating metrics for {} images".format(
            len(self.gt_json_not_unified_list)
        )
        pbar = tqdm(total=len(self.gt_json_not_unified_list), desc=f"{desc:70s}")
        num_jsons = len(self.prediction_json_not_unified_list)
        for i, (pred_json_item, gt_json_item) in enumerate(
            zip(self.prediction_json_not_unified_list, self.gt_json_not_unified_list)
        ):
            image_name_key = "image_path"
            if (
                "images" in gt_json_item
                and gt_json_item["images"]
                and image_name_key in gt_json_item["images"][0]
            ):
                image_name = gt_json_item["images"][0][image_name_key]
            else:
                image_name = f"unknown_image_{i}"

            image_metrics_dict = {"image_name": image_name}
            pbar.set_description_str(
                f'{i + 1}/{num_jsons}.{image_metrics_dict["image_name"][:60]:70s}'
            )

            filtered_gt_words, filtered_prediction_words, ignore_zones = (
                self.ignore_zone_filter_instance.filter_ignore_zones(
                    copy.deepcopy(pred_json_item), copy.deepcopy(gt_json_item)
                )
            )
            self.image_to_ignore_zones_map[image_metrics_dict["image_name"]] = (
                ignore_zones
            )

            pred_filtered = copy.deepcopy(pred_json_item)
            gt_filtered = copy.deepcopy(gt_json_item)
            pred_filtered["images"][0]["words"] = filtered_prediction_words
            gt_filtered["images"][0]["words"] = filtered_gt_words

            mpc_instance = None
            if self.performance_calculator_type == "general":
                mpc_instance = ModelPerformanceCalculator(
                    pred_filtered,
                    gt_filtered,
                    add_space_between_merged_gt_words=self.add_space_between_merged_gt_words,
                    add_space_between_merged_prediction_words=self.add_space_between_merged_prediction_words,
                    string_normalize_map=self.string_normalize_map,
                )
            else:
                print(
                    f"Invalid performance calculator type: {self.performance_calculator_type}!!"
                )
                pbar.update(1)
                continue

            pbar.update(1)
            try:
                page_metrics = mpc_instance.calc_metrics()
                gt_json_unified, prediction_json_unified = (
                    mpc_instance.get_modified_jsons()
                )
                self.prediction_json_list.append(prediction_json_unified)
                self.gt_json_list.append(gt_json_unified)
                image_metrics_dict.update(page_metrics)
            except ZeroDivisionError:
                print(
                    f"Metrics for {image_metrics_dict} has failed due to ZeroDivisionError."
                )
                traceback.print_exc()
            except Exception:
                print(f"Metrics for {image_metrics_dict} has failed.")
                traceback.print_exc()
            self.metrics_per_image.append(image_metrics_dict)
            self.image_to_mpc_map[image_metrics_dict["image_name"]] = mpc_instance
        pbar.close()

    def metrics_per_data_slice(self, float_precision=2):
        images_metrics = self.metrics_per_image

        if not images_metrics:
            return None

        all_metrics_sum = {}
        for d_img_metrics in images_metrics:
            for k, v in d_img_metrics.items():
                if isinstance(v, (int, float)):
                    all_metrics_sum[k] = all_metrics_sum.get(k, 0) + v
                elif k == "image_name":
                    pass
                else:
                    if k not in all_metrics_sum:
                        all_metrics_sum[k] = ""

        denominator_union_sensitive = (
            all_metrics_sum.get("sum_max_length_intersection", EPS)
            + all_metrics_sum.get("text_length_false_positives", EPS)
            + all_metrics_sum.get("text_length_false_negatives", EPS)
        )
        edit_distance_union_sensitive = (
            all_metrics_sum.get(
                "sum_edit_distance_intersection_case_sensitive_not_avg_over_words",
                EPS,
            )
            + all_metrics_sum.get("text_length_false_positives", EPS)
            + all_metrics_sum.get("text_length_false_negatives", EPS)
        ) / max(EPS, denominator_union_sensitive)

        sum_max_len_intersection = all_metrics_sum.get(
            "sum_max_length_intersection", EPS
        )
        edit_distance_intersection_sensitive = all_metrics_sum.get(
            "sum_edit_distance_intersection_case_sensitive_not_avg_over_words", EPS
        ) / max(EPS, sum_max_len_intersection)

        hits_sensitive = all_metrics_sum.get("word_Hits_case_sensitive", EPS)
        substitutions_sensitive = all_metrics_sum.get(
            "word_substitutions_case_sensitive", EPS
        )
        deletions = all_metrics_sum.get("word_Deletions", EPS)
        insertions = all_metrics_sum.get("word_Insertions", EPS)

        word_accuracy_intersection_case_sensitive = hits_sensitive / max(
            EPS, hits_sensitive + substitutions_sensitive
        )
        word_accuracy_union_case_sensitive = hits_sensitive / max(
            EPS, hits_sensitive + substitutions_sensitive + deletions + insertions
        )

        num_tp_matches_sum = all_metrics_sum.get("num_true_positive_matches", EPS)
        num_prediction_cells_sum = all_metrics_sum.get("num_prediction_cells", EPS)
        num_gt_cells_sum = all_metrics_sum.get("number_of_gt_cells", EPS)

        detection_precision_overall = num_tp_matches_sum / max(
            EPS, num_prediction_cells_sum
        )
        detection_recall_overall = num_tp_matches_sum / max(EPS, num_gt_cells_sum)
        detection_f1_score_overall = (
            2 * detection_recall_overall * detection_precision_overall
        ) / max(detection_recall_overall + detection_precision_overall, EPS)

        sum_norm_ed_val = all_metrics_sum.get("sum_norm_ed", EPS)
        norm_ed_tp_only_overall = sum_norm_ed_val / max(EPS, num_tp_matches_sum)

        norm_ed_all_cells_denominator = (
            num_tp_matches_sum
            + all_metrics_sum.get("number_of_false_positive_detections", EPS)
            + all_metrics_sum.get("number_of_false_negative_detections", EPS)
        )
        norm_ed_all_cells_overall = (
            sum_norm_ed_val
            + all_metrics_sum.get("number_of_false_positive_detections", EPS)
            + all_metrics_sum.get("number_of_false_negative_detections", EPS)
        ) / max(EPS, norm_ed_all_cells_denominator)

        norm_ed_without_fp_denominator = num_tp_matches_sum + all_metrics_sum.get(
            "number_of_false_negative_detections", EPS
        )
        norm_ed_without_fp_overall = (
            sum_norm_ed_val
            + all_metrics_sum.get("number_of_false_negative_detections", EPS)
        ) / max(EPS, norm_ed_without_fp_denominator)

        without_ignored_chars_fn_sum = all_metrics_sum.get(
            "without_ignored_chars_false_negatives", EPS
        )
        norm_ed_ign_chars_denom = (
            num_tp_matches_sum
            + all_metrics_sum.get("number_of_false_positive_detections", EPS)
            + without_ignored_chars_fn_sum
        )
        without_ignored_chars_norm_ed_overall = (
            sum_norm_ed_val
            + all_metrics_sum.get("number_of_false_positive_detections", EPS)
            + without_ignored_chars_fn_sum
        ) / max(EPS, norm_ed_ign_chars_denom)

        norm_ed_ign_chars_no_fp_denom = (
            num_tp_matches_sum + without_ignored_chars_fn_sum
        )
        without_ignored_chars_norm_ed_without_fp_overall = (
            sum_norm_ed_val + without_ignored_chars_fn_sum
        ) / max(EPS, norm_ed_ign_chars_no_fp_denom)

        without_single_chars_fn_sum = all_metrics_sum.get(
            "without_single_chars_false_negatives", EPS
        )
        norm_ed_single_chars_denom = (
            num_tp_matches_sum
            + all_metrics_sum.get("number_of_false_positive_detections", EPS)
            + without_single_chars_fn_sum
        )
        without_single_chars_norm_ed_overall = (
            sum_norm_ed_val
            + all_metrics_sum.get("number_of_false_positive_detections", EPS)
            + without_single_chars_fn_sum
        ) / max(EPS, norm_ed_single_chars_denom)

        norm_ed_single_chars_no_fp_denom = (
            num_tp_matches_sum + without_single_chars_fn_sum
        )
        without_single_chars_norm_ed_without_fp_overall = (
            sum_norm_ed_val + without_single_chars_fn_sum
        ) / max(EPS, norm_ed_single_chars_no_fp_denom)

        num_images = len(images_metrics)
        avg_orientation_accuracy = (
            all_metrics_sum.get("orientation_accuracy", 0) / num_images
            if num_images > 0
            else 0
        )
        avg_precision_per_image = (
            all_metrics_sum.get("detection_precision", 0) / num_images
            if num_images > 0
            else 0
        )
        avg_recall_per_image = (
            all_metrics_sum.get("detection_recall", 0) / num_images
            if num_images > 0
            else 0
        )
        avg_f1_per_image = (
            all_metrics_sum.get("detection_f1_score", 0) / num_images
            if num_images > 0
            else 0
        )
        avg_edit_score_union_case_sensitive_per_image = (
            all_metrics_sum.get("edit_score_union_case_sensitive_not_avg_over_words", 0)
            / num_images
            if num_images > 0
            else 0
        )

        hmean_norm_ed_and_precision_val = (1 - norm_ed_without_fp_overall) * (
            avg_precision_per_image / 100.0
        )
        hmean_norm_ed_and_precision_sum = (1 - norm_ed_without_fp_overall) + (
            avg_precision_per_image / 100.0
        )
        hmean_norm_ed_and_precision = (2 * hmean_norm_ed_and_precision_val) / max(
            EPS, hmean_norm_ed_and_precision_sum
        )

        aggregated_metrics_dict = {
            "F1": 100 * detection_f1_score_overall,
            "Recall": 100 * detection_recall_overall,
            "Precision": 100 * detection_precision_overall,
            "Norm_ED (TP-Only)": 100 * (1 - norm_ed_tp_only_overall),
            "Norm_ED (Without FP)": 100 * (1 - norm_ed_without_fp_overall),
            "Norm_ED (All-cells)": 100 * (1 - norm_ed_all_cells_overall),
            "Norm_ED (Ignoring FP and some chars)": 100
            * (1 - without_ignored_chars_norm_ed_without_fp_overall),
            "Norm_ED (Ignoring some chars)": 100
            * (1 - without_ignored_chars_norm_ed_overall),
            "Norm_ED (Ignoring single chars and FP)": 100
            * (1 - without_single_chars_norm_ed_without_fp_overall),
            "Norm_ED (Ignoring single chars)": 100
            * (1 - without_single_chars_norm_ed_overall),
            "Word-accuracy (TP-Only)": 100 * word_accuracy_intersection_case_sensitive,
            "Word-accuracy (All-cells)": 100 * word_accuracy_union_case_sensitive,
            "Edit-score (TP-Only)": 100 * (1 - edit_distance_intersection_sensitive),
            "Edit-score (All-cells)": 100 * (1 - edit_distance_union_sensitive),
            "#Word-Hits": hits_sensitive,
            "#Word-Substitutions": substitutions_sensitive,
            "#FN": insertions,
            "#FP": deletions,
            "#FN - zero iou": all_metrics_sum.get(
                "num_gt_boxes_that_do_not_intersect_with_a_gt", 0
            ),
            "#FN - zero iou, Ignoring some chars": all_metrics_sum.get(
                "without_ignored_chars_false_negatives", 0
            ),
            "#FN - zero iou, Ignoring single chars": all_metrics_sum.get(
                "without_single_chars_false_negatives", 0
            ),
            "#FP - low iou": all_metrics_sum.get(
                "num_prediction_boxes_with_low_iou", 0
            ),
            "#FN - ambiguous match": all_metrics_sum.get(
                "num_gt_boxes_that_are_fn_after_refinement", 0
            ),
            "#FP -  ambiguous match": all_metrics_sum.get(
                "num_prediction_boxes_fp_after_refinement", 0
            ),
            "#FP - zero iou": all_metrics_sum.get(
                "num_prediction_boxes_without_gt_intersection", 0
            ),
            "#Pred. Words Merged": all_metrics_sum.get(
                "prediction_boxes_that_were_merged", 0
            ),
            "#GT Words Merged": all_metrics_sum.get("gt_boxes_that_were_merged", 0),
            "#FN - low iou": all_metrics_sum.get("num_gt_boxes_with_low_iou", 0),
            "#images": num_images,
            "avg_precision": avg_precision_per_image,
            "avg_recall": avg_recall_per_image,
            "avg_f1": avg_f1_per_image,
            "avg_orientation_accuracy": avg_orientation_accuracy,
            "avg_CACC": avg_edit_score_union_case_sensitive_per_image,
            "Hmean: (Norm_ED [Without FP], precision)": 100
            * hmean_norm_ed_and_precision,
        }
        aggregated_metrics_dict["#words-in-GT"] = (
            aggregated_metrics_dict["#Word-Hits"]
            + aggregated_metrics_dict["#Word-Substitutions"]
            + aggregated_metrics_dict["#FN"]
        )

        for k, v_val in aggregated_metrics_dict.items():
            try:
                y_precision = float(f"{{:.{float_precision}f}}".format(v_val))
                aggregated_metrics_dict[k] = y_precision
            except (ValueError, TypeError):
                pass
        return aggregated_metrics_dict

    def get_metrics_values(
        self,
        float_precision=1,
        flat_view=True,
    ):
        header_row_names = [m.readable_name for m in metrics]
        additional_metrics = [
            "#words-in-GT",
            "avg_precision",
            "avg_recall",
            "avg_f1",
            "avg_CACC",
            "avg_orientation_accuracy",
        ]
        header_row_names.extend(additional_metrics)

        table_data = [["Overall Category"] + header_row_names]
        data_only_list = []

        overall_metrics_results = self.metrics_per_data_slice(
            float_precision=float_precision
        )

        if overall_metrics_results:
            row_label = "Overall"
            row = [row_label] + [
                overall_metrics_results.get(metric_name, "N/A")
                for metric_name in header_row_names
            ]
            table_data.append(row)

            if flat_view:
                overall_metrics_results["category"] = "DOCUMENTS"
                overall_metrics_results["model_name"] = self.model_name
                overall_metrics_results["sub_category"] = "Overall"
                data_only_list.append(overall_metrics_results)
            else:
                metadata = {
                    "sub_category": "Overall",
                    "category": "DOCUMENTS",
                    "model_name": self.model_name,
                }
                data_only_list.append((metadata, overall_metrics_results))

        return data_only_list
