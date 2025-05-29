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
    word_weight: int = Field(default=1)
    confidence: float = Field(default=1.0)
    character_confidence: Optional[str] = None
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


class SingleWordMetrics(BaseModel):
    prediction_text: str
    word_weight: Optional[int] = None
    prediction_location: Location
    prediction_word_confidence: Optional[float] = None
    prediction_character_confidence: Optional[str] = None
    correct_detection: bool
    gt_text: Optional[str] = None
    gt_location: Optional[Location] = None
    intersection_over_union: Optional[float] = None
    edit_distance: Optional[int] = None
    edit_distance_case_insensitive: Optional[int] = None
    is_orientation_correct: Optional[float] = None
    normalized_edit_distance: Optional[float] = None
    normalized_edit_distance_case_insensitive: Optional[float] = None


class ImageMetricsSummary(BaseModel):
    num_prediction_cells: int
    number_of_gt_cells: int
    number_of_false_positive_detections: int
    norm_ed_tp_only: float = Field(alias="Norm_ED (TP-Only)")
    norm_ed_all_cells: float = Field(alias="Norm_ED (All-cells)")
    num_true_positive_matches: int
    number_of_false_negative_detections: int
    without_ignored_chars_false_negatives: int
    without_single_chars_false_negatives: int
    detection_precision: float
    detection_recall: float
    detection_f1_score: float
    word_accuracy_intersection_case_sensitive: float
    word_accuracy_intersection_case_insensitive: float
    word_accuracy_union_case_sensitive: float
    word_accuracy_union_case_insensitive: float
    edit_score_intersection_case_sensitive_not_avg_over_words: float
    edit_score_intersection_case_insensitive_not_avg_over_words: float
    edit_score_union_case_sensitive_not_avg_over_words: float
    edit_score_union_case_insensitive_not_avg_over_words: float
    sum_edit_distance_intersection_case_sensitive_not_avg_over_words: float
    sum_edit_distance_intersection_case_insensitive_not_avg_over_words: float
    sum_max_length_intersection: float
    text_length_false_positives: float
    text_length_false_negatives: float
    sum_norm_ed: float
    word_insertions: int = Field(alias="word_Insertions")
    word_deletions: int = Field(alias="word_Deletions")
    word_substitutions_case_sensitive: float
    word_hits_case_sensitive: float = Field(alias="word_Hits_case_sensitive")
    word_substitutions_case_insensitive: float
    word_hits_case_insensitive: float = Field(alias="word_Hits_case_insensitive")
    num_prediction_boxes_without_gt_intersection: int
    num_gt_boxes_that_do_not_intersect_with_a_gt: int
    num_gt_boxes_that_are_fn_after_refinement: int
    num_prediction_boxes_fp_after_refinement: int
    num_gt_boxes_with_low_iou: int
    num_prediction_boxes_with_low_iou: int
    prediction_boxes_that_were_merged: int
    gt_boxes_that_were_merged: int
    orientation_accuracy: float

    class Config:
        populate_by_name = True


class ImageBenchmarkEntry(BaseModel):
    image_name: str
    metrics: ImageMetricsSummary


class AggregatedBenchmarkMetrics(BaseModel):
    f1: float = Field(alias="F1")
    recall: float = Field(alias="Recall")
    precision: float = Field(alias="Precision")
    norm_ed_all_cells: float = Field(alias="Norm_ED (All-cells)")
    word_accuracy_all_cells: float = Field(alias="Word-accuracy (All-cells)")

    class Config:
        populate_by_name = True


class OcrReportEvaluationEntry(BaseModel):
    doc_id: str

    class Config:
        extra = "allow"
