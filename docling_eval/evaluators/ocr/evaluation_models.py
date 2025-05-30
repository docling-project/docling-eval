from typing import Any, List, Optional

from docling_core.types.doc import BoundingBox
from docling_core.types.doc.page import TextCell
from pydantic import BaseModel, Field


class _CalculationConstants:
    EPS: float = 1.0e-6


class Word(TextCell):
    vertical: bool
    polygon: List[List[float]]
    matched: bool = Field(default=False)
    ignore_zone: Optional[bool] = None
    to_remove: Optional[bool] = None

    @property
    def bbox(self) -> BoundingBox:
        return self.rect.to_bounding_box()


class BenchmarkIntersectionInfo(BaseModel):
    x_axis_overlap: float
    y_axis_overlap: float
    intersection_area: float
    union_area: float
    iou: float
    gt_box_portion_covered: float
    prediction_box_portion_covered: float
    x_axis_iou: Optional[float] = None
    y_axis_iou: Optional[float] = None


class OcrMetricsSummary(BaseModel):
    number_of_prediction_cells: int
    number_of_gt_cells: int
    number_of_false_positive_detections: int
    number_of_true_positive_matches: int
    number_of_false_negative_detections: int
    detection_precision: float
    detection_recall: float
    detection_f1: float

    class Config:
        populate_by_name = True


class OcrBenchmarkEntry(BaseModel):
    image_name: str
    metrics: OcrMetricsSummary


class AggregatedBenchmarkMetrics(BaseModel):
    f1: float = Field(alias="F1")
    recall: float = Field(alias="Recall")
    precision: float = Field(alias="Precision")

    class Config:
        populate_by_name = True


class DocumentEvaluationEntry(BaseModel):
    doc_id: str

    class Config:
        extra = "allow"


class OcrDatasetEvaluationResult(BaseModel):
    f1_score: float = 0.0
    recall: float = 0.0
    precision: float = 0.0
