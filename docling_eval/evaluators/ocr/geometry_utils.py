from typing import List, Tuple

from docling_core.types.doc import BoundingBox, CoordOrigin

from docling_eval.evaluators.ocr.evaluation_models import Word, _CalculationConstants


def create_polygon_from_bbox(bbox: BoundingBox) -> List[List[float]]:
    return [
        [bbox.l, bbox.t],
        [bbox.r, bbox.t],
        [bbox.r, bbox.b],
        [bbox.l, bbox.b],
    ]


def calculate_box_intersection_info(
    bbox1: BoundingBox, bbox2: BoundingBox
) -> Tuple[float, float, float, float, float, float, float]:
    if bbox1.coord_origin != bbox2.coord_origin:
        raise ValueError("BoundingBoxes must have the same CoordOrigin.")

    x_axis_overlap_val: float = bbox1.x_overlap_with(bbox2)
    if x_axis_overlap_val <= _CalculationConstants.EPS:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    y_axis_overlap_val: float = bbox1.y_overlap_with(bbox2)
    if y_axis_overlap_val <= _CalculationConstants.EPS:
        return x_axis_overlap_val, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    intersection_area_val: float = bbox1.intersection_area_with(bbox2)

    box1_area: float = bbox1.area()
    box2_area: float = bbox2.area()

    union_area_val: float = bbox1.union_area_with(bbox2)

    iou_val: float = (
        intersection_area_val / union_area_val
        if union_area_val > _CalculationConstants.EPS
        else 0.0
    )

    box1_portion_covered: float = (
        intersection_area_val / box1_area
        if box1_area > _CalculationConstants.EPS
        else 0.0
    )
    box2_portion_covered: float = (
        intersection_area_val / box2_area
        if box2_area > _CalculationConstants.EPS
        else 0.0
    )

    return (
        x_axis_overlap_val,
        y_axis_overlap_val,
        intersection_area_val,
        union_area_val,
        iou_val,
        box1_portion_covered,
        box2_portion_covered,
    )


def calculate_box_intersection_info_extended(
    bbox1: BoundingBox, bbox2: BoundingBox
) -> Tuple[float, float, float, float, float, float, float]:
    if bbox1.coord_origin != bbox2.coord_origin:
        raise ValueError("BoundingBoxes must have the same CoordOrigin.")

    x_axis_overlap_val: float = bbox1.x_overlap_with(bbox2)
    y_axis_overlap_val: float = bbox1.y_overlap_with(bbox2)

    intersection_area_val: float = 0.0
    if (
        x_axis_overlap_val > _CalculationConstants.EPS
        and y_axis_overlap_val > _CalculationConstants.EPS
    ):
        intersection_area_val = bbox1.intersection_area_with(bbox2)

    union_area_val: float = bbox1.union_area_with(bbox2)

    iou_val: float = (
        intersection_area_val / union_area_val
        if union_area_val > _CalculationConstants.EPS
        else 0.0
    )

    x_axis_union_val: float = bbox1.x_union_with(bbox2)
    y_axis_union_val: float = bbox1.y_union_with(bbox2)

    x_axis_iou_val: float = (
        x_axis_overlap_val / x_axis_union_val
        if x_axis_union_val > _CalculationConstants.EPS
        else 0.0
    )
    y_axis_iou_val: float = (
        y_axis_overlap_val / y_axis_union_val
        if y_axis_union_val > _CalculationConstants.EPS
        else 0.0
    )

    return (
        x_axis_overlap_val,
        y_axis_overlap_val,
        intersection_area_val,
        union_area_val,
        iou_val,
        x_axis_iou_val,
        y_axis_iou_val,
    )


def box_to_key(bbox: BoundingBox) -> Tuple[float, float, float, float]:
    if bbox.coord_origin == CoordOrigin.TOPLEFT:
        return (bbox.t, bbox.l, bbox.r, bbox.b)
    elif bbox.coord_origin == CoordOrigin.BOTTOMLEFT:
        return (bbox.t, bbox.l, bbox.r, bbox.b)
    else:
        raise ValueError(f"Unsupported CoordOrigin: {bbox.coord_origin}")


def is_horizontal(word: Word) -> bool:
    bbox = word.bbox
    h: float = bbox.height
    w: float = bbox.width
    if w > _CalculationConstants.EPS and h > (2 * w) and len(word.text) > 1:
        return False
    return True
