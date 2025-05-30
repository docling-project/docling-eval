import json
import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple

from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import (
    BoundingRectangle,
    PageGeometry,
    SegmentedPage,
    TextCell,
)

from docling_eval.evaluators.ocr.evaluation_models import Location, Word
from docling_eval.evaluators.ocr.geometry_utils import (
    create_polygon_from_rect,
    get_x_axis_overlap,
    get_y_axis_overlap,
)

_log = logging.getLogger(__name__)


class _CalculationConstants:
    EPS: float = 1.0e-6


def extract_word_from_text_cell(text_cell: TextCell, page_height: float) -> Word:
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
    left_val: float = float(bbox.l)
    top_val: float = float(bbox.t)
    right_val: float = float(bbox.r)
    bottom_val: float = float(bbox.b)
    width_val: float = right_val - left_val
    height_val: float = bottom_val - top_val

    is_vertical_flag: bool = False
    if (
        width_val > _CalculationConstants.EPS
        and height_val > (2 * width_val)
        and len(text_cell.text) > 1
    ):
        is_vertical_flag = True

    location_obj: Location = Location(
        left=left_val,
        top=top_val,
        width=width_val,
        height=height_val,
        right=right_val,
        bottom=bottom_val,
    )

    return Word(
        word=text_cell.text,
        vertical=is_vertical_flag,
        location=location_obj,
        polygon=polygon_points,
    )


def convert_word_to_text_cell(word_obj: Word) -> TextCell:
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
        confidence=1.0,
        from_ocr=True,
    )


def merge_words_into_one(
    words: List[Word], add_space_between_words: bool = True
) -> Word:
    separator: str = " " if add_space_between_words else ""
    merged_text: str = ""

    min_left: float = float("inf")
    min_top: float = float("inf")
    max_right: float = 0.0
    max_bottom: float = 0.0

    sorted_words: List[Word] = sorted(words, key=lambda k: k.location.left)

    for word_item in sorted_words:
        merged_text = merged_text + separator + word_item.word
        min_left = min(min_left, word_item.location.left)
        min_top = min(min_top, word_item.location.top)
        max_right = max(max_right, word_item.location.right)
        max_bottom = max(max_bottom, word_item.location.bottom)

    merged_text = merged_text.lstrip()

    merged_location = Location(
        top=min_top,
        left=min_left,
        width=max_right - min_left,
        height=max_bottom - min_top,
        right=max_right,
        bottom=max_bottom,
    )

    merged_polygon: List[List[float]] = create_polygon_from_rect(merged_location)

    is_vertical_flag: bool = words[0].vertical if words else False

    return Word(
        word=merged_text,
        location=merged_location,
        polygon=merged_polygon,
        vertical=is_vertical_flag,
    )


class _IgnoreZoneFilter:
    def __init__(self) -> None:
        pass

    def filter_words_in_ignore_zones(
        self, prediction_words: List[Word], ground_truth_words: List[Word]
    ) -> Tuple[List[Word], List[Word], List[Word]]:
        ignore_zones: List[Word] = []

        temp_ground_truth_words: List[Word] = list(ground_truth_words)
        for gt_word in temp_ground_truth_words:
            if gt_word.ignore_zone is True:
                ignore_zones.append(gt_word)
                gt_word.to_remove = True

        for ignore_zone_word in ignore_zones:
            self._mark_intersecting_words_for_removal(
                ignore_zone_word.location, ground_truth_words
            )
            self._mark_intersecting_words_for_removal(
                ignore_zone_word.location, prediction_words
            )

        filtered_ground_truth_words: List[Word] = [
            word for word in ground_truth_words if not word.to_remove
        ]
        filtered_prediction_words: List[Word] = [
            word for word in prediction_words if not word.to_remove
        ]

        return filtered_ground_truth_words, filtered_prediction_words, ignore_zones

    def _mark_intersecting_words_for_removal(
        self, ignore_zone_location: Location, words_list: List[Word]
    ) -> None:
        for word_item in words_list:
            if self._check_intersection(word_item.location, ignore_zone_location):
                word_item.to_remove = True

    def _check_intersection(self, rect1: Location, rect2: Location) -> bool:
        rect1_width: float = rect1.width
        rect1_height: float = rect1.height

        x_overlap: float = get_x_axis_overlap(rect1, rect2)
        y_overlap: float = get_y_axis_overlap(rect1, rect2)

        x_overlap_ratio: float = 0.0 if rect1_width == 0 else x_overlap / rect1_width
        y_overlap_ratio: float = 0.0 if rect1_height == 0 else y_overlap / rect1_height

        if y_overlap_ratio < 0.1 or x_overlap_ratio < 0.1:
            return False
        else:
            return True


def parse_segmented_pages(
    segmented_pages_raw_data: Any, document_id: str
) -> Optional[Dict[int, SegmentedPage]]:
    segmented_pages_map: Dict[int, SegmentedPage] = {}
    if isinstance(segmented_pages_raw_data, (bytes, str)):
        try:
            segmented_pages_payload: Any = json.loads(segmented_pages_raw_data)
        except json.JSONDecodeError as e:
            _log.warning(
                f"JSONDecodeError for doc {document_id}: {e}. Data: {str(segmented_pages_raw_data)[:200]}"
            )
            return None
    elif isinstance(segmented_pages_raw_data, dict):
        segmented_pages_payload = segmented_pages_raw_data
    else:
        _log.warning(
            f"Unrecognized segmented_pages data format for doc {document_id}: {type(segmented_pages_raw_data)}"
        )
        return None

    if not isinstance(segmented_pages_payload, dict):
        _log.warning(
            f"Expected dict payload for segmented_pages for doc {document_id}, got {type(segmented_pages_payload)}"
        )
        return None

    for page_index_str, page_data in segmented_pages_payload.items():
        try:
            page_index: int = int(page_index_str)
        except ValueError:
            _log.warning(
                f"Invalid page index string '{page_index_str}' for doc {document_id}. Skipping page."
            )
            continue

        try:
            if isinstance(page_data, dict):
                segmented_pages_map[page_index] = SegmentedPage.model_validate(
                    page_data
                )
            elif isinstance(page_data, str):
                segmented_pages_map[page_index] = SegmentedPage.model_validate_json(
                    page_data
                )
            elif isinstance(page_data, SegmentedPage):
                segmented_pages_map[page_index] = page_data
            else:
                _log.warning(
                    f"Unrecognized page_data format for doc {document_id}, page {page_index}: {type(page_data)}"
                )
                continue
        except Exception as e_page_val:
            _log.error(
                f"Error validating page data for doc {document_id}, page {page_index}: {e_page_val}"
            )
            traceback.print_exc()
            continue
    return segmented_pages_map if segmented_pages_map else None
