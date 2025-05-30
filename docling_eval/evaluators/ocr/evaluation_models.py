from typing import Any, List, Optional

from pydantic import BaseModel, Field


class Location(BaseModel):
    left: float
    top: float
    width: float
    height: float
    right: float
    bottom: float


class Word(BaseModel):
    word: str
    vertical: bool
    location: Location
    polygon: List[List[float]]
    matched: bool = Field(default=False)
    ignore_zone: Optional[bool] = None
    to_remove: Optional[bool] = None


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
    num_prediction_cells: int
    number_of_gt_cells: int
    number_of_false_positive_detections: int
    num_true_positive_matches: int
    number_of_false_negative_detections: int
    detection_precision: float
    detection_recall: float
    detection_f1_score: float

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
