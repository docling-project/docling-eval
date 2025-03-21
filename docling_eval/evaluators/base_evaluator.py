import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from docling_eval.benchmarks.constants import PredictionFormats
from docling_eval.evaluators.stats import DatasetStatistics

_log = logging.getLogger(__name__)


class DatasetEvaluation(BaseModel):
    pass


class BaseEvaluator:
    r"""
    Base class for all evaluators
    """

    def __init__(self, intermediate_evaluations_path: Optional[Path] = None):
        r"""
        Parameters
        ----------
        intermediate_evaluations_path: When True the evalution per example will be saved in a file
        """
        self._intermediate_evaluations_path = intermediate_evaluations_path

    def __call__(
        self,
        ds_path: Path,
        split: str = "test",
        ext_predictions: Optional[
            Dict[str, Any]
        ] = None,  # Optionally provided external predictions
    ) -> DatasetEvaluation:
        r"""
        Perform the evaluation
        """
        return None  # type: ignore

    def supported_prediction_formats(self) -> List[PredictionFormats]:
        r"""
        Return the supported formats for predictions
        """
        return [PredictionFormats.DOCLING_DOCUMENT]
