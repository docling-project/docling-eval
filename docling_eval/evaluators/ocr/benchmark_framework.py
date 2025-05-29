import copy
import traceback
from typing import Any, Dict, List, Optional, Tuple

from docling_core.types.doc.page import SegmentedPage
from tqdm import tqdm

from docling_eval.evaluators.ocr.benchmark_constants import (
    AggregatedBenchmarkMetrics,
    ImageBenchmarkEntry,
    ImageMetricsSummary,
    Location,
    Word,
)
from docling_eval.evaluators.ocr.performance_calculator import (
    _ModelPerformanceCalculator,
)
from docling_eval.evaluators.ocr.text_utils import (
    _CalculationConstants,
    _extract_word_details_from_text_cell,
)

CHAR_MAP_FOR_NORMALIZATION: Dict[str, str] = {
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


class _IgnoreZonesFilter:

    def __init__(self) -> None:
        pass

    def filter_ignore_zones(
        self, prediction_words: List[Word], gt_words: List[Word]
    ) -> Tuple[List[Word], List[Word], List[Word]]:
        ignore_zones: List[Word] = []

        temp_gt_words: List[Word] = list(gt_words)
        for gt_word in temp_gt_words:
            if gt_word.ignore_zone is True:
                ignore_zones.append(gt_word)
                gt_word.to_remove = True

        for gt_word_zone in ignore_zones:
            self._mark_intersecting_to_ignore(gt_word_zone.location, gt_words)
            self._mark_intersecting_to_ignore(gt_word_zone.location, prediction_words)

        filtered_gt_words: List[Word] = [i for i in gt_words if not i.to_remove]
        filtered_prediction_words: List[Word] = [
            i for i in prediction_words if not i.to_remove
        ]

        return filtered_gt_words, filtered_prediction_words, ignore_zones

    def _mark_intersecting_to_ignore(
        self, ignore_zone_location: Location, words_list: List[Word]
    ) -> None:
        for word_item in words_list:
            if self._intersects(word_item.location, ignore_zone_location):
                word_item.to_remove = True

    def _get_y_axis_overlap(self, rect1: Location, rect2: Location) -> float:
        y_overlap: float = max(
            0.0,
            min(rect1.bottom, rect2.bottom) - max(rect1.top, rect2.top),
        )
        return y_overlap

    def _get_x_axis_overlap(self, rect1: Location, rect2: Location) -> float:
        x_overlap: float = max(
            0.0,
            min(rect1.right, rect2.right) - max(rect1.left, rect2.left),
        )
        return x_overlap

    def _intersects(self, rect1: Location, rect2: Location) -> bool:
        det_x_len: float = rect1.width
        det_y_len: float = rect1.height

        x_overlap: float = self._get_x_axis_overlap(rect1, rect2)
        y_overlap: float = self._get_y_axis_overlap(rect1, rect2)

        x_overlap_ratio: float = 0.0 if det_x_len == 0 else x_overlap / det_x_len
        y_overlap_ratio: float = 0.0 if det_y_len == 0 else y_overlap / det_y_len

        if y_overlap_ratio < 0.1 or x_overlap_ratio < 0.1:
            return False
        else:
            return True


class _ModelBenchmark:

    def __init__(
        self,
        model_name: str,
        performance_calculator: str = "general",
        ignore_zone_filter: str = "default",
        add_space_between_merged_prediction_words: bool = True,
        add_space_between_merged_gt_words: bool = True,
        string_normalize_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.model_name: str = model_name
        self.add_space_between_merged_prediction_words: bool = (
            add_space_between_merged_prediction_words
        )
        self.add_space_between_merged_gt_words: bool = add_space_between_merged_gt_words
        if string_normalize_map is None:
            self.string_normalize_map: Dict[str, str] = (
                CHAR_MAP_FOR_NORMALIZATION.copy()
            )
        else:
            self.string_normalize_map = string_normalize_map

        self.prediction_pages_processed: List[SegmentedPage] = []
        self.gt_pages_processed: List[SegmentedPage] = []
        self.gt_pages_input: List[SegmentedPage] = []
        self.prediction_pages_input: List[SegmentedPage] = []
        self.metrics_per_image: List[ImageBenchmarkEntry] = []
        self.image_to_mpc_map: Dict[str, _ModelPerformanceCalculator] = {}
        self.image_to_ignore_zones_map: Dict[str, List[Word]] = {}
        self.performance_calculator_type: str = performance_calculator
        self.ignore_zone_filter_instance: _IgnoreZonesFilter = _IgnoreZonesFilter()

    def load_res_and_gt_files(
        self,
        gt_page: SegmentedPage,
        pred_page: SegmentedPage,
    ) -> None:
        self.gt_pages_input.append(gt_page)
        self.prediction_pages_input.append(pred_page)

    def run_benchmark(self, min_intersection_to_match: float = 0.5) -> None:
        desc: str = "Calculating metrics for {} images".format(len(self.gt_pages_input))
        pbar = tqdm(total=len(self.gt_pages_input), desc=f"{desc:70s}")
        num_pages: int = len(self.prediction_pages_input)
        for i, (pred_page_item, gt_page_item) in enumerate(
            zip(self.prediction_pages_input, self.gt_pages_input)
        ):
            image_name_for_entry = f"page_{i}"

            current_image_name_desc = (
                f"{i + 1}/{num_pages}.{image_name_for_entry[:60]:70s}"
            )
            pbar.set_description_str(current_image_name_desc)

            pred_words_list_raw: List[Word] = []
            if pred_page_item.has_words:
                page_height = pred_page_item.dimension.height
                for text_cell in pred_page_item.word_cells:
                    pred_words_list_raw.append(
                        _extract_word_details_from_text_cell(text_cell, page_height)
                    )

            gt_words_list_raw: List[Word] = []
            if gt_page_item.has_words:
                page_height = gt_page_item.dimension.height
                for text_cell in gt_page_item.word_cells:
                    gt_words_list_raw.append(
                        _extract_word_details_from_text_cell(text_cell, page_height)
                    )

            filtered_gt_words: List[Word]
            filtered_prediction_words: List[Word]
            ignore_zones: List[Word]

            copied_pred_words_list_raw = copy.deepcopy(pred_words_list_raw)
            copied_gt_words_list_raw = copy.deepcopy(gt_words_list_raw)

            filtered_gt_words, filtered_prediction_words, ignore_zones = (
                self.ignore_zone_filter_instance.filter_ignore_zones(
                    copied_pred_words_list_raw, copied_gt_words_list_raw
                )
            )
            self.image_to_ignore_zones_map[image_name_for_entry] = ignore_zones

            mpc_instance: Optional[_ModelPerformanceCalculator] = None
            if self.performance_calculator_type == "general":
                mpc_instance = _ModelPerformanceCalculator(
                    prediction_words=filtered_prediction_words,
                    gt_words=filtered_gt_words,
                    prediction_segmented_page_meta=pred_page_item,
                    gt_segmented_page_meta=gt_page_item,
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
                if mpc_instance:
                    page_metrics: ImageMetricsSummary = mpc_instance.calc_metrics()
                    image_benchmark_entry = ImageBenchmarkEntry(
                        image_name=image_name_for_entry, metrics=page_metrics
                    )
                    self.metrics_per_image.append(image_benchmark_entry)

                    gt_page_unified: SegmentedPage
                    prediction_page_unified: SegmentedPage
                    gt_page_unified, prediction_page_unified = (
                        mpc_instance.get_modified_jsons()
                    )
                    self.prediction_pages_processed.append(prediction_page_unified)
                    self.gt_pages_processed.append(gt_page_unified)
            except ZeroDivisionError:
                print(
                    f"Metrics for {image_name_for_entry} has failed due to ZeroDivisionError."
                )
                traceback.print_exc()
            except Exception:
                print(f"Metrics for {image_name_for_entry} has failed.")
                traceback.print_exc()
            if mpc_instance:
                self.image_to_mpc_map[image_name_for_entry] = mpc_instance
        pbar.close()

    def metrics_per_data_slice(
        self, float_precision: int = 2
    ) -> Optional[Dict[str, Any]]:
        image_benchmark_entries: List[ImageBenchmarkEntry] = self.metrics_per_image

        if not image_benchmark_entries:
            return None

        all_metrics_sum: Dict[str, Any] = {}
        for entry in image_benchmark_entries:
            for k, v in entry.metrics.model_dump(by_alias=False).items():
                if isinstance(v, (int, float)):
                    all_metrics_sum[k] = all_metrics_sum.get(k, 0) + v
                elif k == "image_name":
                    pass
                else:
                    if k not in all_metrics_sum:
                        all_metrics_sum[k] = ""

        num_tp_matches_sum: float = all_metrics_sum.get(
            "num_true_positive_matches", _CalculationConstants.EPS
        )
        num_prediction_cells_sum: float = all_metrics_sum.get(
            "num_prediction_cells", _CalculationConstants.EPS
        )
        num_gt_cells_sum: float = all_metrics_sum.get(
            "number_of_gt_cells", _CalculationConstants.EPS
        )

        detection_precision_overall: float = num_tp_matches_sum / max(
            _CalculationConstants.EPS, num_prediction_cells_sum
        )
        detection_recall_overall: float = num_tp_matches_sum / max(
            _CalculationConstants.EPS, num_gt_cells_sum
        )
        detection_f1_score_overall: float = (
            2 * detection_recall_overall * detection_precision_overall
        ) / max(
            detection_recall_overall + detection_precision_overall,
            _CalculationConstants.EPS,
        )

        sum_norm_ed_val: float = all_metrics_sum.get(
            "sum_norm_ed", _CalculationConstants.EPS
        )
        num_false_positives_sum: float = all_metrics_sum.get(
            "number_of_false_positive_detections", _CalculationConstants.EPS
        )
        num_false_negatives_sum: float = all_metrics_sum.get(
            "number_of_false_negative_detections", _CalculationConstants.EPS
        )

        norm_ed_all_cells_denominator: float = (
            num_tp_matches_sum + num_false_positives_sum + num_false_negatives_sum
        )
        norm_ed_all_cells_overall: float = (
            sum_norm_ed_val + num_false_positives_sum + num_false_negatives_sum
        ) / max(_CalculationConstants.EPS, norm_ed_all_cells_denominator)

        hits_sensitive: float = all_metrics_sum.get(
            "word_hits_case_sensitive", _CalculationConstants.EPS
        )
        substitutions_sensitive: float = all_metrics_sum.get(
            "word_substitutions_case_sensitive", _CalculationConstants.EPS
        )
        deletions: float = all_metrics_sum.get(
            "word_deletions", _CalculationConstants.EPS
        )
        insertions: float = all_metrics_sum.get(
            "word_insertions", _CalculationConstants.EPS
        )

        word_accuracy_union_case_sensitive: float = hits_sensitive / max(
            _CalculationConstants.EPS,
            hits_sensitive + substitutions_sensitive + deletions + insertions,
        )

        aggregated_metrics_model_data = {
            "f1": 100 * detection_f1_score_overall,
            "recall": 100 * detection_recall_overall,
            "precision": 100 * detection_precision_overall,
            "norm_ed_all_cells": 100 * (1 - norm_ed_all_cells_overall),
            "word_accuracy_all_cells": 100 * word_accuracy_union_case_sensitive,
        }

        aggregated_metrics_instance = AggregatedBenchmarkMetrics.model_validate(
            aggregated_metrics_model_data
        )

        output_dict = aggregated_metrics_instance.model_dump(by_alias=True)

        for k, v_val in output_dict.items():
            try:
                y_precision: float = float(f"{{:.{float_precision}f}}".format(v_val))
                output_dict[k] = y_precision
            except (ValueError, TypeError):
                pass
        return output_dict

    def get_metrics_values(
        self,
        float_precision: int = 1,
        flat_view: bool = True,
    ) -> List[Dict[str, Any]]:
        data_only_list: List[Dict[str, Any]] = []

        overall_metrics_results: Optional[Dict[str, Any]] = self.metrics_per_data_slice(
            float_precision=float_precision
        )

        if overall_metrics_results:
            overall_metrics_results["category"] = "DOCUMENTS"
            overall_metrics_results["model_name"] = self.model_name
            overall_metrics_results["sub_category"] = "Overall"
            data_only_list.append(overall_metrics_results)

        return data_only_list
