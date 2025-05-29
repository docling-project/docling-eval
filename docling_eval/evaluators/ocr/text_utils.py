from typing import Any, Dict, List, Optional, Tuple

import edit_distance
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import (
    BoundingRectangle,
    PageGeometry,
    SegmentedPage,
    TextCell,
)

from docling_eval.evaluators.ocr.benchmark_constants import (
    BenchmarkIntersectionInfo,
    Location,
    Word,
)
from docling_eval.evaluators.ocr.geometry_utils import (
    box_to_key,
    convert_locations_to_float,
    info_for_boxes,
    info_for_boxes_extended,
    is_horizontal,
)

edit_distance_chars_map: Dict[str, str] = {}


class _CalculationConstants:
    EPS: float = 1.0e-6


def _extract_word_details_from_text_cell(
    text_cell: TextCell, page_height: float
) -> Word:
    rect_to_process = text_cell.rect
    if rect_to_process.coord_origin != CoordOrigin.TOPLEFT:
        rect_to_process = rect_to_process.to_top_left_origin(page_height=page_height)

    polygon_points: List[List[float]] = [
        [float(rect_to_process.r_x0), float(rect_to_process.r_y0)],
        [float(rect_to_process.r_x1), float(rect_to_process.r_y1)],
        [float(rect_to_process.r_x2), float(rect_to_process.r_y2)],
        [float(rect_to_process.r_x3), float(rect_to_process.r_y3)],
    ]

    bbox = rect_to_process.to_bounding_box()
    left: float = float(bbox.l)
    top: float = float(bbox.t)
    right: float = float(bbox.r)
    bottom: float = float(bbox.b)
    width: float = right - left
    height: float = bottom - top

    is_vertical: bool = False
    if (
        width > _CalculationConstants.EPS
        and height > (2 * width)
        and len(text_cell.text) > 1
    ):
        is_vertical = True

    location_obj: Location = Location(
        left=left,
        top=top,
        width=width,
        height=height,
        right=right,
        bottom=bottom,
    )

    word_confidence: float = 1.0
    if hasattr(text_cell, "confidence") and text_cell.confidence is not None:
        word_confidence = float(text_cell.confidence)

    return Word(
        word=text_cell.text,
        vertical=is_vertical,
        location=location_obj,
        polygon=polygon_points,
        word_weight=1,
        confidence=word_confidence,
    )


def _convert_word_to_text_cell(word_obj: Word) -> TextCell:
    bbox = BoundingBox(
        l=word_obj.location.left,
        t=word_obj.location.top,
        r=word_obj.location.right,
        b=word_obj.location.bottom,
        coord_origin=CoordOrigin.TOPLEFT,
    )

    text_cell_rect = BoundingRectangle.from_bounding_box(bbox)

    return TextCell(
        rect=text_cell_rect,
        text=word_obj.word,
        orig=word_obj.word,
        confidence=word_obj.confidence if word_obj.confidence is not None else 1.0,
        from_ocr=True,
    )


def replace_chars_by_map(text: str, char_map: Dict[str, str]) -> str:
    processed_text: str = "".join(
        char if char not in char_map else char_map[char] for char in text
    )
    return processed_text


def calculate_edit_distance(
    hypothesis_text: str,
    reference_text: str,
    string_normalize_map: Optional[Dict[str, str]] = None,
) -> int:
    if string_normalize_map is None:
        string_normalize_map = {}

    hypothesis_text = replace_chars_by_map(hypothesis_text, string_normalize_map)
    reference_text = replace_chars_by_map(reference_text, string_normalize_map)

    if len(edit_distance_chars_map) > 0:
        hypothesis_text = "".join(
            (
                char
                if char not in edit_distance_chars_map
                else edit_distance_chars_map[char]
            )
            for char in hypothesis_text
        )
        reference_text = "".join(
            (
                char
                if char not in edit_distance_chars_map
                else edit_distance_chars_map[char]
            )
            for char in reference_text
        )
    sm = edit_distance.SequenceMatcher(hypothesis_text, reference_text)
    return sm.distance()


def match_ground_truth_to_prediction_words(
    gt_words: List[Word], prediction_words: List[Word]
) -> Tuple[
    Dict[
        Tuple[float, float, float, float], List[Tuple[Word, BenchmarkIntersectionInfo]]
    ],
    Dict[
        Tuple[float, float, float, float], List[Tuple[Word, BenchmarkIntersectionInfo]]
    ],
]:
    convert_locations_to_float(prediction_words)
    convert_locations_to_float(gt_words)

    gt_to_prediction_boxes_map: Dict[
        Tuple[float, float, float, float], List[Tuple[Word, BenchmarkIntersectionInfo]]
    ] = {}
    for gt_box in gt_words:
        intersections: List[Tuple[Word, BenchmarkIntersectionInfo]] = []
        for prediction_box in prediction_words:
            x_axis_overlap: float
            y_axis_overlap: float
            intersection_area: float
            union_area: float
            iou: float
            current_gt_box_portion_covered: float
            current_prediction_box_portion_covered: float
            (
                x_axis_overlap,
                y_axis_overlap,
                intersection_area,
                union_area,
                iou,
                current_gt_box_portion_covered,
                current_prediction_box_portion_covered,
            ) = info_for_boxes(gt_box.location, prediction_box.location)
            if intersection_area > 0:
                intersections.append(
                    (
                        prediction_box,
                        BenchmarkIntersectionInfo(
                            x_axis_overlap=x_axis_overlap,
                            y_axis_overlap=y_axis_overlap,
                            intersection_area=intersection_area,
                            union_area=union_area,
                            iou=iou,
                            gt_box_portion_covered=current_gt_box_portion_covered,
                            prediction_box_portion_covered=current_prediction_box_portion_covered,
                        ),
                    )
                )
        box_key_val: Tuple[float, float, float, float] = box_to_key(gt_box.location)
        if len(intersections) > 1:
            intersections = sorted(intersections, key=lambda x: x[1].intersection_area)
        gt_to_prediction_boxes_map[box_key_val] = intersections

    prediction_to_gt_boxes_map: Dict[
        Tuple[float, float, float, float], List[Tuple[Word, BenchmarkIntersectionInfo]]
    ] = {}
    for prediction_box in prediction_words:
        intersections: List[Tuple[Word, BenchmarkIntersectionInfo]] = []
        for gt_box in gt_words:
            x_axis_overlap: float
            y_axis_overlap: float
            intersection_area: float
            union_area: float
            iou: float
            current_gt_box_portion_covered: float
            current_prediction_box_portion_covered: float
            (
                x_axis_overlap,
                y_axis_overlap,
                intersection_area,
                union_area,
                iou,
                current_gt_box_portion_covered,
                current_prediction_box_portion_covered,
            ) = info_for_boxes(gt_box.location, prediction_box.location)
            if intersection_area > 0:
                intersections.append(
                    (
                        gt_box,
                        BenchmarkIntersectionInfo(
                            x_axis_overlap=x_axis_overlap,
                            y_axis_overlap=y_axis_overlap,
                            intersection_area=intersection_area,
                            union_area=union_area,
                            iou=iou,
                            gt_box_portion_covered=current_gt_box_portion_covered,
                            prediction_box_portion_covered=current_prediction_box_portion_covered,
                        ),
                    )
                )
        box_key_val: Tuple[float, float, float, float] = box_to_key(
            prediction_box.location
        )
        intersections = sorted(intersections, key=lambda x: x[1].intersection_area)
        prediction_to_gt_boxes_map[box_key_val] = intersections
    return gt_to_prediction_boxes_map, prediction_to_gt_boxes_map


def refine_prediction_to_many_gt_boxes(
    prediction_word: Word, intersections: List[Tuple[Word, BenchmarkIntersectionInfo]]
) -> Tuple[
    List[Tuple[Word, BenchmarkIntersectionInfo]],
    List[Tuple[Word, BenchmarkIntersectionInfo]],
]:
    s: List[Tuple[Word, BenchmarkIntersectionInfo]] = sorted(
        [(gt_box, boxes_info) for (gt_box, boxes_info) in intersections],
        key=lambda x: x[1].intersection_area,
        reverse=True,
    )
    a: List[bool] = [is_horizontal(x) for x, _ in s]
    num_horizontal: int = sum(a)
    num_vertical: int = len(a) - num_horizontal

    valid_intersections_line: List[Tuple[Word, BenchmarkIntersectionInfo]] = [s[0]]
    invalid_intersections_line: List[Tuple[Word, BenchmarkIntersectionInfo]] = []

    for gt_box, boxes_info in s[1:]:
        can_be_added: bool = True
        for b, _ in [s[0]]:
            x_axis_overlap: float
            y_axis_overlap: float
            intersection_area: float
            union_area: float
            iou: float
            x_axis_iou: float
            y_axis_iou: float
            (
                x_axis_overlap,
                y_axis_overlap,
                intersection_area,
                union_area,
                iou,
                x_axis_iou,
                y_axis_iou,
            ) = info_for_boxes_extended(gt_box.location, b.location)
            height_ratio: float = min(gt_box.location.height, b.location.height) / max(
                gt_box.location.height + _CalculationConstants.EPS,
                b.location.height + _CalculationConstants.EPS,
            )
            words_in_same_line: bool = (
                (x_axis_iou < 0.2 and y_axis_iou > 0)
                if height_ratio < 0.5
                else (x_axis_iou < 0.2 and y_axis_iou > 0.3)
            )
            if not words_in_same_line:
                can_be_added = False
        if can_be_added:
            valid_intersections_line.append((gt_box, boxes_info))
        else:
            invalid_intersections_line.append((gt_box, boxes_info))
    line_intersections_refined: List[List[Tuple[Word, BenchmarkIntersectionInfo]]] = [
        valid_intersections_line,
        invalid_intersections_line,
    ]

    valid_intersections_col: List[Tuple[Word, BenchmarkIntersectionInfo]] = [s[0]]
    invalid_intersections_col: List[Tuple[Word, BenchmarkIntersectionInfo]] = []
    for gt_box, boxes_info in s[1:]:
        can_be_added: bool = True
        for b, _ in [s[0]]:
            x_axis_overlap: float
            y_axis_overlap: float
            intersection_area: float
            union_area: float
            iou: float
            x_axis_iou: float
            y_axis_iou: float
            (
                x_axis_overlap,
                y_axis_overlap,
                intersection_area,
                union_area,
                iou,
                x_axis_iou,
                y_axis_iou,
            ) = info_for_boxes_extended(gt_box.location, b.location)
            width_ratio: float = min(gt_box.location.width, b.location.width) / max(
                gt_box.location.width + _CalculationConstants.EPS,
                b.location.width + _CalculationConstants.EPS,
            )

            words_in_same_column: bool = (
                (y_axis_iou < 0.2 and x_axis_iou > 0)
                if width_ratio < 0.5
                else (y_axis_iou < 0.2 and x_axis_iou > 0.5)
            )
            if not words_in_same_column:
                can_be_added = False
        if can_be_added:
            valid_intersections_col.append((gt_box, boxes_info))
        else:
            invalid_intersections_col.append((gt_box, boxes_info))
    column_intersections_refined: List[List[Tuple[Word, BenchmarkIntersectionInfo]]] = [
        valid_intersections_col,
        invalid_intersections_col,
    ]

    chosen_intersections: List[List[Tuple[Word, BenchmarkIntersectionInfo]]] = []
    if num_horizontal > num_vertical:
        chosen_intersections = line_intersections_refined
    else:
        chosen_intersections = column_intersections_refined

    if len(chosen_intersections[1]) > 0:
        return [], intersections
    else:
        return (
            chosen_intersections[0],
            chosen_intersections[1],
        )
