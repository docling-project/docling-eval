from typing import Any, List, Optional

from docling_core.types.doc import BoundingBox
from docling_core.types.doc.page import TextCell
from pydantic import BaseModel, Field, field_validator


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


class BoundingBoxDict(BaseModel):
    """Bounding box represented as a dictionary with left, top, right, bottom coordinates."""

    l: float = Field(..., description="Left coordinate")
    t: float = Field(..., description="Top coordinate")
    r: float = Field(..., description="Right coordinate")
    b: float = Field(..., description="Bottom coordinate")

    def to_bounding_box(self) -> "BoundingBox":
        """Convert to your existing BoundingBox class."""
        return BoundingBox(l=self.l, t=self.t, r=self.r, b=self.b)


class WordBoundingBox(BaseModel):
    """Word with its bounding box coordinates."""

    word: str = Field(..., min_length=1, description="The extracted word text")
    bbox: BoundingBoxDict = Field(
        ..., description="Bounding box coordinates of the word"
    )


class LineTextInput(BaseModel):
    """Input parameters for smart weighted character distribution."""

    line_text: str = Field(..., description="The text line to process")
    line_bbox: BoundingBoxDict = Field(
        ..., description="Bounding box of the entire line"
    )

    @field_validator("line_text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Line text cannot be empty or whitespace only")
        return v

    @classmethod
    def from_existing_bbox(
        cls, line_text: str, line_bbox: "BoundingBox"
    ) -> "LineTextInput":
        """Create LineTextInput from your existing BoundingBox class."""
        return cls(
            line_text=line_text,
            line_bbox=BoundingBoxDict(
                l=line_bbox.l, t=line_bbox.t, r=line_bbox.r, b=line_bbox.b
            ),
        )


class WordSegmentationResult(BaseModel):
    """Result of segmenting a text line into individual words with precise positioning."""

    segmented_words: List[WordBoundingBox] = Field(
        default_factory=list,
        description="Words extracted from the line with their calculated bounding boxes",
    )
