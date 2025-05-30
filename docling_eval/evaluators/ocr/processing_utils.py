import json
import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple

from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import (
    BoundingRectangle,
    SegmentedPage,
    TextCell,
    TextDirection,
)

from docling_eval.evaluators.ocr.evaluation_models import Word, _CalculationConstants
from docling_eval.evaluators.ocr.geometry_utils import create_polygon_from_bbox

_log = logging.getLogger(__name__)


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

    width_val: float = bbox.width
    height_val: float = bbox.height

    is_vertical_flag: bool = False
    if (
        width_val > _CalculationConstants.EPS
        and height_val > (2 * width_val)
        and len(text_cell.text) > 1
    ):
        is_vertical_flag = True

    return Word(
        rect=rect_to_process,
        text=text_cell.text,
        orig=text_cell.orig,
        text_direction=text_cell.text_direction,
        confidence=text_cell.confidence,
        from_ocr=text_cell.from_ocr,
        vertical=is_vertical_flag,
        polygon=polygon_points,
    )


def convert_word_to_text_cell(word_obj: Word) -> TextCell:
    source_bbox = word_obj.bbox

    bbox_for_conversion = BoundingBox(
        l=source_bbox.l,
        t=source_bbox.t,
        r=source_bbox.r,
        b=source_bbox.b,
        coord_origin=CoordOrigin.TOPLEFT,
    )
    if source_bbox.coord_origin != CoordOrigin.TOPLEFT:
        pass

    text_cell_rect = BoundingRectangle.from_bounding_box(bbox_for_conversion)

    return TextCell(
        rect=text_cell_rect,
        text=word_obj.text,
        orig=word_obj.orig,
        confidence=word_obj.confidence,
        text_direction=word_obj.text_direction,
        from_ocr=word_obj.from_ocr,
    )


def merge_words_into_one(
    words: List[Word], add_space_between_words: bool = True
) -> Word:
    if not words:
        default_bbox = BoundingBox(l=0, t=0, r=0, b=0, coord_origin=CoordOrigin.TOPLEFT)
        default_rect = BoundingRectangle.from_bounding_box(default_bbox)
        return Word(
            text="",
            rect=default_rect,
            orig="",
            confidence=1.0,
            from_ocr=False,
            text_direction=TextDirection.LEFT_TO_RIGHT,
            vertical=False,
            polygon=create_polygon_from_bbox(default_bbox),
        )

    separator: str = " " if add_space_between_words else ""
    merged_text_parts = []

    min_left: float = float("inf")
    min_top: float = float("inf")
    max_right: float = -float("inf")
    max_bottom: float = -float("inf")

    sorted_words: List[Word] = sorted(words, key=lambda k: k.bbox.l)

    first_word_for_metadata = sorted_words[0]

    for word_item in sorted_words:
        merged_text_parts.append(word_item.text)
        current_bbox = word_item.bbox
        min_left = min(min_left, current_bbox.l)
        min_top = min(min_top, current_bbox.t)
        max_right = max(max_right, current_bbox.r)
        max_bottom = max(max_bottom, current_bbox.b)

    merged_text: str = separator.join(merged_text_parts)

    merged_bbox = BoundingBox(
        l=min_left,
        t=min_top,
        r=max_right,
        b=max_bottom,
        coord_origin=CoordOrigin.TOPLEFT,
    )

    merged_polygon: List[List[float]] = create_polygon_from_bbox(merged_bbox)
    merged_rect = BoundingRectangle.from_bounding_box(merged_bbox)

    return Word(
        text=merged_text,
        rect=merged_rect,
        orig=merged_text,
        confidence=first_word_for_metadata.confidence,
        from_ocr=any(w.from_ocr for w in sorted_words),
        text_direction=first_word_for_metadata.text_direction,
        vertical=first_word_for_metadata.vertical,
        polygon=merged_polygon,
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
                ignore_zone_word.bbox, ground_truth_words
            )
            self._mark_intersecting_words_for_removal(
                ignore_zone_word.bbox, prediction_words
            )

        filtered_ground_truth_words: List[Word] = [
            word for word in ground_truth_words if not word.to_remove
        ]
        filtered_prediction_words: List[Word] = [
            word for word in prediction_words if not word.to_remove
        ]

        return filtered_ground_truth_words, filtered_prediction_words, ignore_zones

    def _mark_intersecting_words_for_removal(
        self, ignore_zone_bbox: BoundingBox, words_list: List[Word]
    ) -> None:
        for word_item in words_list:
            if self._check_intersection(word_item.bbox, ignore_zone_bbox):
                word_item.to_remove = True

    def _check_intersection(self, bbox1: BoundingBox, bbox2: BoundingBox) -> bool:
        bbox1_width: float = bbox1.width
        bbox1_height: float = bbox1.height

        x_overlap: float = bbox1.x_overlap_with(bbox2)
        y_overlap: float = bbox1.y_overlap_with(bbox2)

        x_overlap_ratio: float = 0.0 if bbox1_width == 0 else x_overlap / bbox1_width
        y_overlap_ratio: float = 0.0 if bbox1_height == 0 else y_overlap / bbox1_height

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
