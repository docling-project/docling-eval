import json
import traceback
from typing import Any, Dict, Optional

from docling_core.types.doc import CoordOrigin
from docling_core.types.doc.page import PageGeometry, SegmentedPage, TextCell

from docling_eval.evaluators.ocr.benchmark_framework import EPS


def _process_segpages_data(
    segpages_data_raw: Any, doc_id: str
) -> Optional[Dict[int, SegmentedPage]]:
    segmented_pages_map: Dict[int, SegmentedPage] = {}
    if isinstance(segpages_data_raw, (bytes, str)):
        try:
            segpages_dict_payload = json.loads(segpages_data_raw)
        except json.JSONDecodeError as e:
            print(
                f"JSONDecodeError for doc {doc_id}: {e}. Data: {str(segpages_data_raw)[:200]}"
            )
            return None
    elif isinstance(segpages_data_raw, dict):
        segpages_dict_payload = segpages_data_raw
    else:
        print(
            f"Unrecognized segmented_pages data format for doc {doc_id}: {type(segpages_data_raw)}"
        )
        return None

    if not isinstance(segpages_dict_payload, dict):
        print(
            f"Expected dict payload for segmented_pages for doc {doc_id}, got {type(segpages_dict_payload)}"
        )
        return None

    for page_idx_str, page_data_item in segpages_dict_payload.items():
        try:
            page_idx = int(page_idx_str)
        except ValueError:
            print(
                f"Invalid page index string '{page_idx_str}' for doc {doc_id}. Skipping page."
            )
            continue

        try:
            if isinstance(page_data_item, dict):
                segmented_pages_map[page_idx] = SegmentedPage.model_validate(
                    page_data_item
                )
            elif isinstance(page_data_item, str):
                segmented_pages_map[page_idx] = SegmentedPage.model_validate_json(
                    page_data_item
                )
            elif isinstance(page_data_item, SegmentedPage):
                segmented_pages_map[page_idx] = page_data_item
            else:
                print(
                    f"Unrecognized page_data format for doc {doc_id}, page {page_idx}: {type(page_data_item)}"
                )
                continue
        except Exception as e_page_val:
            print(
                f"Error validating page data for doc {doc_id}, page {page_idx}: {e_page_val}"
            )
            traceback.print_exc()
            continue
    return segmented_pages_map if segmented_pages_map else None


def _extract_word_details_from_text_cell(
    text_cell: TextCell, page_height: float
) -> Dict[str, Any]:
    rect_to_process = text_cell.rect
    if rect_to_process.coord_origin != CoordOrigin.TOPLEFT:
        rect_to_process = rect_to_process.to_top_left_origin(page_height=page_height)

    polygon_points = [
        [float(rect_to_process.r_x0), float(rect_to_process.r_y0)],
        [float(rect_to_process.r_x1), float(rect_to_process.r_y1)],
        [float(rect_to_process.r_x2), float(rect_to_process.r_y2)],
        [float(rect_to_process.r_x3), float(rect_to_process.r_y3)],
    ]

    bbox = rect_to_process.to_bounding_box()
    left = float(bbox.l)
    top = float(bbox.t)
    right = float(bbox.r)
    bottom = float(bbox.b)
    width = right - left
    height = bottom - top

    is_vertical = False
    if width > EPS and height > (2 * width) and len(text_cell.text) > 1:
        is_vertical = True

    word_entry = {
        "word": text_cell.text,
        "vertical": is_vertical,
        "location": {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "right": right,
            "bottom": bottom,
        },
        "polygon": polygon_points,
        "word-weight": 1,
    }

    if hasattr(text_cell, "confidence") and text_cell.confidence is not None:
        word_entry["confidence"] = float(text_cell.confidence)
    else:
        word_entry["confidence"] = 1.0

    return word_entry


def _create_ocr_dictionary_from_segmented_page(
    segmented_page: SegmentedPage,
    doc_id: str,
    page_number: int,
    image_path_override: Optional[str] = None,
) -> Dict[str, Any]:
    page_width = segmented_page.dimension.width
    page_height = segmented_page.dimension.height

    actual_image_path = image_path_override
    if actual_image_path is None:
        actual_image_path = f"{doc_id}_page_{page_number + 1}.png"

    word_dictionaries = []
    if segmented_page.has_words:
        for word_cell in segmented_page.word_cells:
            word_dictionaries.append(
                _extract_word_details_from_text_cell(word_cell, page_height)
            )

    return {
        "image_path": actual_image_path,
        "page_dimensions": {"width": page_width, "height": page_height},
        "words": word_dictionaries,
        "category": "",
        "sub-category": "",
    }
