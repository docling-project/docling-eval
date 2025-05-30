from typing import List, Tuple

from docling_eval.evaluators.ocr.evaluation_models import Location, Word


def create_polygon_from_rect(location: Location) -> List[List[float]]:
    return [
        [location.left, location.top],
        [location.left + location.width, location.top],
        [location.left + location.width, location.top + location.height],
        [location.left, location.top + location.height],
    ]


def convert_locations_to_float(words: List[Word]) -> None:
    for word_item in words:
        loc: Location = word_item.location
        loc.top = float(loc.top)
        loc.left = float(loc.left)
        loc.width = float(loc.width)
        loc.height = float(loc.height)
        loc.right = loc.left + loc.width
        loc.bottom = loc.top + loc.height
    return


def get_x_axis_overlap(rect1: Location, rect2: Location) -> float:
    x_overlap: float = max(
        0.0, min(rect1.right, rect2.right) - max(rect1.left, rect2.left)
    )
    return x_overlap


def get_y_axis_overlap(rect1: Location, rect2: Location) -> float:
    y_overlap: float = max(
        0.0, min(rect1.bottom, rect2.bottom) - max(rect1.top, rect2.top)
    )
    return y_overlap


def get_intersection_area(rect1: Location, rect2: Location) -> float:
    x_overlap: float = get_x_axis_overlap(rect1, rect2)
    y_overlap: float = get_y_axis_overlap(rect1, rect2)
    overlap_area: float = x_overlap * y_overlap
    return overlap_area


def get_union_area(rect1: Location, rect2: Location) -> float:
    union_area: float = (
        rect1.width * rect1.height
        + rect2.width * rect2.height
        - get_intersection_area(rect1, rect2)
    )
    return union_area


def get_x_axis_union(rect1: Location, rect2: Location) -> float:
    x_union: float = max(
        0.0, max(rect1.right, rect2.right) - min(rect1.left, rect2.left)
    )
    return x_union


def get_y_axis_union(rect1: Location, rect2: Location) -> float:
    y_union: float = max(
        0.0, max(rect1.bottom, rect2.bottom) - min(rect1.top, rect2.top)
    )
    return y_union


def calculate_box_intersection_info(
    box1: Location, box2: Location
) -> Tuple[float, float, float, float, float, float, float]:
    x_axis_overlap_val: float = get_x_axis_overlap(box1, box2)
    if x_axis_overlap_val == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    y_axis_overlap_val: float = get_y_axis_overlap(box1, box2)
    if y_axis_overlap_val == 0:
        return x_axis_overlap_val, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    intersection_area_val: float = x_axis_overlap_val * y_axis_overlap_val
    box1_area: float = box1.width * box1.height
    box2_area: float = box2.width * box2.height
    union_area_val: float = box1_area + box2_area - intersection_area_val
    iou_val: float = (
        intersection_area_val / union_area_val if union_area_val > 0 else 0.0
    )
    box1_portion_covered: float = (
        intersection_area_val / box1_area if box1_area > 0 else 0.0
    )
    box2_portion_covered: float = (
        intersection_area_val / box2_area if box2_area > 0 else 0.0
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
    box1: Location, box2: Location
) -> Tuple[float, float, float, float, float, float, float]:
    x_axis_overlap_val: float = get_x_axis_overlap(box1, box2)
    y_axis_overlap_val: float = get_y_axis_overlap(box1, box2)
    intersection_area_val: float = x_axis_overlap_val * y_axis_overlap_val
    box1_area: float = box1.width * box1.height
    box2_area: float = box2.width * box2.height
    union_area_val: float = box1_area + box2_area - intersection_area_val
    iou_val: float = (
        intersection_area_val / union_area_val if union_area_val > 0 else 0.0
    )

    x_axis_union_val: float = get_x_axis_union(box1, box2)
    y_axis_union_val: float = get_y_axis_union(box1, box2)
    x_axis_iou_val: float = (
        x_axis_overlap_val / x_axis_union_val if x_axis_union_val > 0 else 0.0
    )
    y_axis_iou_val: float = (
        y_axis_overlap_val / y_axis_union_val if y_axis_union_val > 0 else 0.0
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


def box_to_key(box: Location) -> Tuple[float, float, float, float]:
    return (box.top, box.left, box.right, box.bottom)


def is_horizontal(word: Word) -> bool:
    h: float = word.location.height
    w: float = word.location.width
    if h > int(2 * w) and len(word.word) > 1:
        return False
    return True
