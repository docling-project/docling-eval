from typing import List, Tuple

from docling_eval.evaluators.ocr.benchmark_constants import Location, Word


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


def info_for_boxes(
    box1: Location, box2: Location
) -> Tuple[float, float, float, float, float, float, float]:
    x_axis_overlap: float = get_x_axis_overlap(box1, box2)
    if x_axis_overlap == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    y_axis_overlap: float = get_y_axis_overlap(box1, box2)
    if y_axis_overlap == 0:
        return x_axis_overlap, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    intersection_area: float = x_axis_overlap * y_axis_overlap
    box1_area: float = box1.width * box1.height
    box2_area: float = box2.width * box2.height
    union_area: float = box1_area + box2_area - intersection_area
    iou: float = intersection_area / union_area if union_area > 0 else 0.0
    box1_portion_covered: float = (
        intersection_area / box1_area if box1_area > 0 else 0.0
    )
    box2_portion_covered: float = (
        intersection_area / box2_area if box2_area > 0 else 0.0
    )
    return (
        x_axis_overlap,
        y_axis_overlap,
        intersection_area,
        union_area,
        iou,
        box1_portion_covered,
        box2_portion_covered,
    )


def info_for_boxes_extended(
    box1: Location, box2: Location
) -> Tuple[float, float, float, float, float, float, float]:
    x_axis_overlap: float = get_x_axis_overlap(box1, box2)
    y_axis_overlap: float = get_y_axis_overlap(box1, box2)
    intersection_area: float = x_axis_overlap * y_axis_overlap
    box1_area: float = box1.width * box1.height
    box2_area: float = box2.width * box2.height
    union_area: float = box1_area + box2_area - intersection_area
    iou: float = intersection_area / union_area if union_area > 0 else 0.0

    x_axis_union: float = get_x_axis_union(box1, box2)
    y_axis_union: float = get_y_axis_union(box1, box2)
    x_axis_iou: float = x_axis_overlap / x_axis_union if x_axis_union > 0 else 0.0
    y_axis_iou: float = y_axis_overlap / y_axis_union if y_axis_union > 0 else 0.0
    return (
        x_axis_overlap,
        y_axis_overlap,
        intersection_area,
        union_area,
        iou,
        x_axis_iou,
        y_axis_iou,
    )


def box_to_key(box: Location) -> Tuple[float, float, float, float]:
    return (box.top, box.left, box.right, box.bottom)


def is_horizontal(word: Word) -> bool:
    h: float = word.location.height
    w: float = word.location.width
    if h > int(2 * w) and len(word.word) > 1:
        return False
    return True


def unify_words(words: List[Word], add_space_between_words: bool = True) -> Word:
    separator: str = " " if add_space_between_words else ""
    unified_text: str = ""

    min_left: float = float("inf")
    min_top: float = float("inf")
    max_right: float = 0.0
    max_bottom: float = 0.0

    sorted_words: List[Word] = sorted(words, key=lambda k: k.location.left)

    total_confidence: float = 0.0
    num_words_with_confidence: int = 0

    for word_item in sorted_words:
        unified_text = unified_text + separator + word_item.word
        min_left = min(min_left, word_item.location.left)
        min_top = min(min_top, word_item.location.top)
        max_right = max(max_right, word_item.location.right)
        max_bottom = max(max_bottom, word_item.location.bottom)
        if word_item.confidence is not None:
            total_confidence += word_item.confidence
            num_words_with_confidence += 1

    unified_text = unified_text.lstrip()

    unified_location = Location(
        top=min_top,
        left=min_left,
        width=max_right - min_left,
        height=max_bottom - min_top,
        right=max_right,
        bottom=max_bottom,
    )

    unified_polygon: List[List[float]] = create_polygon_from_rect(unified_location)

    avg_confidence: float = (
        (total_confidence / num_words_with_confidence)
        if num_words_with_confidence > 0
        else 1.0
    )

    is_vertical: bool = words[0].vertical if words else False

    return Word(
        word=unified_text,
        location=unified_location,
        polygon=unified_polygon,
        word_weight=len(sorted_words),
        confidence=avg_confidence,
        vertical=is_vertical,
    )
