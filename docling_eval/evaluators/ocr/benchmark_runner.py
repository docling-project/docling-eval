import copy
import traceback
from typing import Any, Dict, List, Optional

from docling_core.types.doc.page import SegmentedPage

from docling_eval.evaluators.ocr.evaluation_models import (
    AggregatedBenchmarkMetrics,
    OcrBenchmarkEntry,
    OcrMetricsSummary,
    Word,
)
from docling_eval.evaluators.ocr.performance_calculator import _OcrPerformanceCalculator
from docling_eval.evaluators.ocr.processing_utils import (
    _CalculationConstants,
    _IgnoreZoneFilter,
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

        self.ignore_zone_filter: _IgnoreZoneFilter = _IgnoreZoneFilter()

    def process_single_page_pair(
        self,
        ground_truth_page: SegmentedPage,
        prediction_page: SegmentedPage,
        image_identifier: str,
    ) -> None:
        raw_prediction_words: List[Word] = []
        if prediction_page.has_words:
            page_height = prediction_page.dimension.height
            for text_cell_item in prediction_page.word_cells:
                raw_prediction_words.append(
                    extract_word_from_text_cell(text_cell_item, page_height)
                )

        raw_ground_truth_words: List[Word] = []
        if ground_truth_page.has_words:
            page_height = ground_truth_page.dimension.height
            for text_cell_item in ground_truth_page.word_cells:
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

        aggregated_metrics_data = {
            "f1": 100 * overall_f1_score,
            "recall": 100 * overall_recall,
            "precision": 100 * overall_precision,
        }

        aggregated_metrics = AggregatedBenchmarkMetrics.model_validate(
            aggregated_metrics_data
        )
        output_results = aggregated_metrics.model_dump(by_alias=True)

        for key, val in output_results.items():
            try:
                formatted_value: float = float(f"{{:.{float_precision}f}}".format(val))
                output_results[key] = formatted_value
            except (ValueError, TypeError):
                pass
        return output_results

    def get_formatted_metrics_summary(
        self,
        float_precision: int = 1,
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
