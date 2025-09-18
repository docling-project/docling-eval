import copy
import traceback
from typing import Any, Dict, List, Optional

from docling_core.types.doc.page import SegmentedPage

from docling_eval.evaluators.ocr.evaluation_models import (
    OcrBenchmarkEntry,
    OcrMetricsSummary,
    TextCellUnit,
    Word,
)
from docling_eval.evaluators.ocr.performance_calculator import _OcrPerformanceCalculator
from docling_eval.evaluators.ocr.processing_utils import (
    _CalculationConstants,
    _IgnoreZoneFilter,
    _IgnoreZoneFilterHWR,
    extract_word_from_text_cell,
)


class _OcrBenchmark:
    def __init__(
        self,
        model_identifier: str,
        performance_calculator_type: str = "general",
        ignore_zone_filter_type: str = "default",
        add_space_for_merged_prediction_words: bool = True,
        add_space_for_merged_gt_words: bool = True,
        aggregation_mode: str = "union",
        text_unit: TextCellUnit = TextCellUnit.WORD,
    ) -> None:
        self.model_identifier: str = model_identifier
        self.add_space_for_merged_prediction_words: bool = (
            add_space_for_merged_prediction_words
        )
        self.add_space_for_merged_gt_words: bool = add_space_for_merged_gt_words

        self.image_metrics_results: List[OcrBenchmarkEntry] = []
        self.image_to_performance_calculator_map: Dict[
            str, _OcrPerformanceCalculator
        ] = {}
        self.image_to_ignore_zones_map: Dict[str, List[Word]] = {}
        self.calculator_type: str = performance_calculator_type
        self.aggregation_mode: str = aggregation_mode
        self.text_unit = text_unit

        self.ignore_zone_filter: "_IgnoreZoneFilter | _IgnoreZoneFilterHWR"
        if ignore_zone_filter_type.lower() == "hwr":
            self.ignore_zone_filter = _IgnoreZoneFilterHWR()
        else:
            self.ignore_zone_filter = _IgnoreZoneFilter()

    def process_single_page_pair(
        self,
        ground_truth_page: SegmentedPage,
        prediction_page: SegmentedPage,
        image_identifier: str,
    ) -> None:
        if self.text_unit == TextCellUnit.LINE:
            prediction_cells = prediction_page.textline_cells
            prediction_has_cells = prediction_page.has_lines
            gt_cells = ground_truth_page.textline_cells
            gt_has_cells = ground_truth_page.has_lines
        else:
            prediction_cells = prediction_page.word_cells
            prediction_has_cells = prediction_page.has_words
            gt_cells = ground_truth_page.word_cells
            gt_has_cells = ground_truth_page.has_words

        raw_prediction_words: List[Word] = []
        if prediction_has_cells:
            page_height = prediction_page.dimension.height
            for text_cell_item in prediction_cells:
                raw_prediction_words.append(
                    extract_word_from_text_cell(text_cell_item, page_height)
                )

        raw_ground_truth_words: List[Word] = []
        if gt_has_cells:
            page_height = ground_truth_page.dimension.height
            for text_cell_item in gt_cells:
                raw_ground_truth_words.append(
                    extract_word_from_text_cell(text_cell_item, page_height)
                )

        copied_prediction_words = copy.deepcopy(raw_prediction_words)
        copied_ground_truth_words = copy.deepcopy(raw_ground_truth_words)

        (
            filtered_gt_words,
            filtered_prediction_words,
            ignored_zones,
        ) = self.ignore_zone_filter.filter_words_in_ignore_zones(
            copied_prediction_words, copied_ground_truth_words
        )
        self.image_to_ignore_zones_map[image_identifier] = ignored_zones

        perf_calculator: Optional[_OcrPerformanceCalculator] = None
        if self.calculator_type == "general":
            perf_calculator = _OcrPerformanceCalculator(
                prediction_words=filtered_prediction_words,
                ground_truth_words=filtered_gt_words,
                prediction_segmented_page_metadata=prediction_page,
                ground_truth_segmented_page_metadata=ground_truth_page,
                add_space_between_merged_gt_words=self.add_space_for_merged_gt_words,
                add_space_between_merged_prediction_words=self.add_space_for_merged_prediction_words,
            )
        else:
            print(f"Invalid performance calculator type: {self.calculator_type}!!")
            return

        try:
            if perf_calculator:
                page_metrics_summary: OcrMetricsSummary = (
                    perf_calculator.calculate_image_metrics()
                )
                image_benchmark_result = OcrBenchmarkEntry(
                    image_name=image_identifier, metrics=page_metrics_summary
                )
                self.image_metrics_results.append(image_benchmark_result)
                self.image_to_performance_calculator_map[image_identifier] = (
                    perf_calculator
                )
        except ZeroDivisionError:
            print(f"Metrics for {image_identifier} failed due to ZeroDivisionError.")
            traceback.print_exc()
        except Exception:
            print(f"Metrics for {image_identifier} failed.")
            traceback.print_exc()

    def calculate_aggregated_metrics(
        self, float_precision: int = 2
    ) -> Optional[Dict[str, Any]]:
        if not self.image_metrics_results:
            return None

        summed_metrics: Dict[str, Any] = {}
        for result_entry in self.image_metrics_results:
            for key, value in result_entry.metrics.model_dump(by_alias=False).items():
                if isinstance(value, (int, float)):
                    summed_metrics[key] = summed_metrics.get(key, 0) + value
                elif key == "image_name":
                    pass
                else:
                    if key not in summed_metrics:
                        summed_metrics[key] = ""

        num_images = len(self.image_metrics_results)
        # Recognition aggregation
        if self.aggregation_mode == "union":
            total_weighted_tp_words: float = summed_metrics.get(
                "tp_words_weighted", 0.0
            )
            total_fp: float = summed_metrics.get(
                "number_of_false_positive_detections", 0.0
            )
            total_fn: float = summed_metrics.get(
                "number_of_false_negative_detections", 0.0
            )
            total_union_words: float = total_weighted_tp_words + total_fp + total_fn
            total_perfect_sensitive: float = summed_metrics.get(
                "perfect_matches_sensitive_weighted", 0.0
            )
            total_perfect_insensitive: float = summed_metrics.get(
                "perfect_matches_insensitive_weighted", 0.0
            )
            avg_word_acc_sensitive = total_perfect_sensitive / max(
                _CalculationConstants.EPS, total_union_words
            )
            avg_word_acc_insensitive = total_perfect_insensitive / max(
                _CalculationConstants.EPS, total_union_words
            )
            # Character (union)
            sum_ed_sensitive_tp: float = summed_metrics.get("sum_ed_sensitive_tp", 0.0)
            sum_ed_insensitive_tp: float = summed_metrics.get(
                "sum_ed_insensitive_tp", 0.0
            )
            sum_max_len_tp: float = summed_metrics.get("sum_max_len_tp", 0.0)
            sum_text_len_fp: float = summed_metrics.get("text_len_fp", 0.0)
            sum_text_len_fn: float = summed_metrics.get("text_len_fn", 0.0)
            total_chars_union: float = (
                sum_max_len_tp + sum_text_len_fp + sum_text_len_fn
            )
            avg_ed_union_sensitive: float = (
                sum_ed_sensitive_tp + sum_text_len_fp + sum_text_len_fn
            ) / max(_CalculationConstants.EPS, total_chars_union)
            avg_ed_union_insensitive: float = (
                sum_ed_insensitive_tp + sum_text_len_fp + sum_text_len_fn
            ) / max(_CalculationConstants.EPS, total_chars_union)
            avg_char_acc_sensitive = 1 - avg_ed_union_sensitive
            avg_char_acc_insensitive = 1 - avg_ed_union_insensitive
            # Convert to percentage later
            avg_word_acc_sensitive *= 100.0
            avg_word_acc_insensitive *= 100.0
            avg_char_acc_sensitive *= 100.0
            avg_char_acc_insensitive *= 100.0
        else:
            # Per-image mean of already-percentage metrics
            avg_word_acc_sensitive = (
                summed_metrics.get("word_accuracy_sensitive", 0.0) / num_images
            )
            avg_word_acc_insensitive = (
                summed_metrics.get("word_accuracy_insensitive", 0.0) / num_images
            )
            avg_char_acc_sensitive = (
                summed_metrics.get("character_accuracy_sensitive", 0.0) / num_images
            )
            avg_char_acc_insensitive = (
                summed_metrics.get("character_accuracy_insensitive", 0.0) / num_images
            )

        total_true_positives: float = summed_metrics.get(
            "number_of_true_positive_matches", _CalculationConstants.EPS
        )
        total_predictions: float = summed_metrics.get(
            "number_of_prediction_cells", _CalculationConstants.EPS
        )
        total_ground_truths: float = summed_metrics.get(
            "number_of_gt_cells", _CalculationConstants.EPS
        )

        overall_precision: float = total_true_positives / max(
            _CalculationConstants.EPS, total_predictions
        )
        overall_recall: float = total_true_positives / max(
            _CalculationConstants.EPS, total_ground_truths
        )
        overall_f1_score: float = (2 * overall_recall * overall_precision) / max(
            overall_recall + overall_precision,
            _CalculationConstants.EPS,
        )

        avg_char_acc_sensitive = (
            summed_metrics.get("character_accuracy_sensitive", 0.0) / num_images
        )
        avg_char_acc_insensitive = (
            summed_metrics.get("character_accuracy_insensitive", 0.0) / num_images
        )

        aggregated_metrics_data = {
            "f1": 100 * overall_f1_score,
            "recall": 100 * overall_recall,
            "precision": 100 * overall_precision,
            "word_accuracy_sensitive": avg_word_acc_sensitive,
            "word_accuracy_insensitive": avg_word_acc_insensitive,
            "character_accuracy_sensitive": avg_char_acc_sensitive,
            "character_accuracy_insensitive": avg_char_acc_insensitive,
        }

        for key, val in aggregated_metrics_data.items():
            try:
                formatted_value: float = float(f"{{:.{float_precision}f}}".format(val))
                aggregated_metrics_data[key] = formatted_value
            except (ValueError, TypeError):
                pass

        return aggregated_metrics_data

    def get_formatted_metrics_summary(
        self,
        float_precision: int = 2,
    ) -> List[Dict[str, Any]]:
        summary_list: List[Dict[str, Any]] = []

        overall_aggregated_metrics: Optional[Dict[str, Any]] = (
            self.calculate_aggregated_metrics(float_precision=float_precision)
        )

        if overall_aggregated_metrics:
            overall_aggregated_metrics["category"] = "DOCUMENTS"
            overall_aggregated_metrics["model_name"] = self.model_identifier
            overall_aggregated_metrics["sub_category"] = "Overall"
            summary_list.append(overall_aggregated_metrics)

        return summary_list
