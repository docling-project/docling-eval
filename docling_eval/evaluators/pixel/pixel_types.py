from typing import Any, Dict, Optional

import numpy as np
from pydantic import BaseModel, model_serializer, model_validator

from docling_eval.evaluators.base_evaluator import EvaluationRejectionType
from docling_eval.evaluators.stats import DatasetStatistics


class LayoutResolution(BaseModel):
    r"""Single bbox resolution"""

    category_id: int

    # bbox coords: (x1, y1, x2, y2) with the origin(0, 0) at the top, left corner, no normalization
    bbox: list[float]


class MultiLabelMatrixAggMetrics(BaseModel):
    classes_precision: dict[str, float]
    classes_recall: dict[str, float]
    classes_f1: dict[str, float]

    classes_precision_mean: float
    classes_recall_mean: float
    classes_f1_mean: float


class MultiLabelMatrixMetrics(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    class_names: Dict[int, str]
    confusion_matrix: np.ndarray
    precision_matrix: np.ndarray
    recall_matrix: np.ndarray
    f1_matrix: np.ndarray

    agg_metrics: MultiLabelMatrixAggMetrics

    @model_serializer(mode="wrap")
    def serialize_model(self, serializer: Any) -> dict:
        data = serializer(self)
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, np.ndarray):
                data[field_name] = field_value.tolist()
        return data

    @model_validator(mode="before")
    @classmethod
    def deserialize_arrays(cls, data: Any) -> Any:
        if isinstance(data, dict):
            array_fields = [
                "confusion_matrix",
                "precision_matrix",
                "recall_matrix",
                "f1_matrix",
            ]
            for field_name in array_fields:
                if field_name in data:
                    data[field_name] = np.asarray(data[field_name])
        return data


class MultiLabelMatrixEvaluation(BaseModel):
    detailed: MultiLabelMatrixMetrics  # All classes included
    collapsed: MultiLabelMatrixMetrics  # Only the background the other classes summed up together


class PagePixelLayoutEvaluation(BaseModel):
    doc_id: str
    page_no: int
    matrix_evaluation: MultiLabelMatrixEvaluation


class DatasetPixelLayoutEvaluation(BaseModel):
    layout_model_name: Optional[str]
    num_pages: int
    num_pixels: int
    rejected_samples: Dict[EvaluationRejectionType, int]
    matrix_evaluation: MultiLabelMatrixEvaluation
    page_evaluations: Dict[str, PagePixelLayoutEvaluation]

    # Statistics across all images for f1 on all classes and on the collapsed classes
    f1_all_classes_stats: DatasetStatistics
    f1_collapsed_classes_stats: DatasetStatistics
