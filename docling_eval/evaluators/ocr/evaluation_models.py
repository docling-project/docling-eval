from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from docling_core.types.doc import BoundingBox
from docling_core.types.doc.page import TextCell
from pydantic import BaseModel, Field


class _CalculationConstants:
    EPS: float = 1.0e-6
    CHAR_NORMALIZE_MAP: Dict[str, str] = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "—": "-",
        "–": "-",
        "\xa0": " ",
    }


class TextCellUnit(str, Enum):
    WORD = "word"
    LINE = "line"


class Word(TextCell):
    vertical: bool
    polygon: List[List[float]]
    matched: bool = Field(default=False)
    ignore_zone: Optional[bool] = None
    to_remove: Optional[bool] = None
    # number of GT words represented by this Word after merging
    word_weight: int = Field(default=1)

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
    # recognition metrics
    word_accuracy_sensitive: float = 0.0
    word_accuracy_insensitive: float = 0.0
    character_accuracy_sensitive: float = 0.0
    character_accuracy_insensitive: float = 0.0
    # for dataset-level union aggregation
    tp_words_weighted: float = 0.0
    perfect_matches_sensitive_weighted: float = 0.0
    perfect_matches_insensitive_weighted: float = 0.0
    sum_ed_sensitive_tp: float = 0.0
    sum_ed_insensitive_tp: float = 0.0
    sum_max_len_tp: float = 0.0
    text_len_fp: float = 0.0
    text_len_fn: float = 0.0

    class Config:
        populate_by_name = True


class OcrBenchmarkEntry(BaseModel):
    image_name: str
    metrics: OcrMetricsSummary


class DocumentEvaluationEntry(BaseModel):
    doc_id: str

    class Config:
        extra = "allow"


class OcrDatasetEvaluationResult(BaseModel):
    f1_score: float = 0.0
    recall: float = 0.0
    precision: float = 0.0
    word_accuracy_sensitive: float = 0.0
    word_accuracy_insensitive: float = 0.0
    character_accuracy_sensitive: float = 0.0
    character_accuracy_insensitive: float = 0.0


class WordEvaluationMetadata(BaseModel):
    text: str
    confidence: Optional[float] = None
    bounding_box: BoundingBox
    is_true_positive: bool = False
    is_false_positive: bool = False
    is_false_negative: bool = False
    edit_distance_sensitive: Optional[int] = None
    edit_distance_insensitive: Optional[int] = None


class TruePositiveMatch(BaseModel):
    pred: WordEvaluationMetadata
    gt: WordEvaluationMetadata


class DocumentEvaluationMetadata(BaseModel):
    doc_id: str
    true_positives: List[TruePositiveMatch]
    false_positives: List[WordEvaluationMetadata]
    false_negatives: List[WordEvaluationMetadata]
    metrics: OcrMetricsSummary
