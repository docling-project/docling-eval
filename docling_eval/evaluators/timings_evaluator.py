import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from datasets import Dataset, load_dataset

from docling_eval.evaluators.base_evaluator import (
    BaseEvaluator,
    DatasetEvaluation,
    EvaluationRejectionType,
    UnitEvaluation,
)
from docling_eval.evaluators.stats import DatasetStatistics, compute_stats

from docling_eval.datamodels.dataset_record import DatasetRecordWithPrediction
from docling_eval.datamodels.types import BenchMarkColumns, PredictionFormats

_log = logging.getLogger(__name__)

class DatasetTimingsEvaluation(DatasetEvaluation):
    """Dataset timing evaluation."""

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

            if data_record.status not in self._accepted_status:
                _log.error(
                    "Skipping record without successfull conversion status: %s", doc_id
                )
                rejected_samples[EvaluationRejectionType.INVALID_CONVERSION_STATUS] += 1
                continue

            print("data_record.prediction_timings: ", data_record.prediction_timings)
            
            timings.append(data_record.prediction_timings)
            
        dataset_timing_evaluation = DatasetTimingEvaluation(
            timing_per_page_stats=compute_stats([_.time for _ in timings])
        )
        return dataset_layout_evaluation
