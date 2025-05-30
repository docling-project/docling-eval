import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset
from tqdm import tqdm

from docling_eval.datamodels.dataset_record import DatasetRecordWithPrediction
from docling_eval.datamodels.types import BenchMarkColumns, PredictionFormats
from docling_eval.evaluators.base_evaluator import (
    BaseEvaluator,
    DatasetEvaluation,
    EvaluationRejectionType,
    UnitEvaluation,
)
from docling_eval.evaluators.stats import DatasetStatistics, compute_stats

_log = logging.getLogger(__name__)


class DatasetTimingsEvaluation(DatasetEvaluation):
    """Dataset timing evaluation."""

    timing_per_document_stats: DatasetStatistics
    timing_per_page_stats: DatasetStatistics


class TimingsEvaluator(BaseEvaluator):
    """Timings evaluator."""

    def __init__(
        self,
        intermediate_evaluations_path: Optional[Path] = None,
        prediction_sources: List[PredictionFormats] = [],
    ):
        supported_prediction_formats: List[PredictionFormats] = [
            PredictionFormats.DOCLING_DOCUMENT,
        ]

        if not prediction_sources:
            prediction_sources = supported_prediction_formats
        super().__init__(
            intermediate_evaluations_path=intermediate_evaluations_path,
            prediction_sources=prediction_sources,
            supported_prediction_formats=supported_prediction_formats,
        )

    def __call__(
        self,
        ds_path: Path,
        split: str = "test",
    ) -> DatasetTimingsEvaluation:
        logging.info("Loading the split '%s' from: '%s'", split, ds_path)

        rejected_samples: Dict[EvaluationRejectionType, int] = {
            EvaluationRejectionType.INVALID_CONVERSION_STATUS: 0,
            EvaluationRejectionType.MISSING_PREDICTION: 0,
            EvaluationRejectionType.MISMATHCED_DOCUMENT: 0,
        }

        # Load the dataset
        split_path = str(ds_path / split / "*.parquet")
        split_files = glob.glob(split_path)
        logging.info("#-files: %s", len(split_files))
        ds = load_dataset("parquet", data_files={split: split_files})
        logging.info("Overview of dataset: %s", ds)

        # Select the split
        ds_selection: Dataset = ds[split]

        timings = []
        for i, data in tqdm(
            enumerate(ds_selection),
            desc="Timings evaluations",
            ncols=120,
            total=len(ds_selection),
        ):
            data_record = DatasetRecordWithPrediction.model_validate(data)

            doc_id = data_record.doc_id
            if data_record.status not in self._accepted_status:
                _log.error(
                    "Skipping record without successfull conversion status: %s", doc_id
                )
                rejected_samples[EvaluationRejectionType.INVALID_CONVERSION_STATUS] += 1
                continue

            # print(data_record.prediction_timings)
            timings.append(data_record.prediction_timings)

        if rejected_samples[EvaluationRejectionType.MISMATHCED_DOCUMENT] > 0:
            logging.error(
                "Total mismatched/skipped documents: %s over %s",
                rejected_samples[EvaluationRejectionType.MISMATHCED_DOCUMENT],
                len(ds_selection),
            )

        time_per_doc = []
        time_per_page = []

        for timing in timings:
            if timing is not None:
                for key, val in timing.items():
                    if key == "pipeline_total":
                        time_per_doc.extend(val)

                    if key == "layout":
                        _time_per_page = [0.0 for v in val]
                        for k2, v2 in timing.items():
                            if len(v2) == len(_time_per_page):
                                for i, v in enumerate(v2):
                                    _time_per_page[i] += v

                        time_per_page.extend(_time_per_page)

        dataset_timings_evaluation = DatasetTimingsEvaluation(
            timing_per_document_stats=compute_stats(
                time_per_doc,
                max_value_is_one=False,
                nr_bins=32,
            ),
            timing_per_page_stats=compute_stats(
                time_per_page,
                max_value_is_one=False,
                nr_bins=32,
            ),
        )
        return dataset_timings_evaluation
