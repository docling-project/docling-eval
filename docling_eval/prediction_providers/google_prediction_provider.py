import copy
import importlib.metadata
import json
import logging
import os
from typing import Dict, List, Optional, Set, Tuple

from docling.datamodel.base_models import ConversionStatus
from docling_core.types import DoclingDocument
from docling_core.types.doc import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import (
    ImageRef,
    PageItem,
    ProvenanceItem,
    TableCell,
    TableData,
)
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc.page import (
    BoundingRectangle,
    PageGeometry,
    SegmentedPage,
    TextCell,
)
from docling_core.types.io import DocumentStream
from google.cloud import documentai
from google.oauth2 import service_account
from google.protobuf.json_format import MessageToDict

from docling_eval.datamodels.dataset_record import (
    DatasetRecord,
    DatasetRecordWithPrediction,
)
from docling_eval.datamodels.types import PredictionFormats, PredictionProviderType
from docling_eval.prediction_providers.base_prediction_provider import (
    BasePredictionProvider,
)
from docling_eval.utils.utils import (
    does_intersection_area_exceed_threshold,
    from_pil_to_base64uri,
)

_log = logging.getLogger(__name__)


class _WordMerger:
    SPECIAL_CHARS: List[str] = list("*:;,.?()!@#$%^&[]{}/\\\"'~+-_<>=")
    CLOSE_THRESHOLD: float = 0.3
    NUMBERS_CLOSE_THRESHOLD: float = 0.7
    CLOSE_LEFT_THRESHOLD: float = 0.7
    INSIDE_THRESHOLD: float = 0.15

    @staticmethod
    def _get_y_axis_iou(rect1: BoundingRectangle, rect2: BoundingRectangle) -> float:
        bb1 = rect1.to_bounding_box()
        bb2 = rect2.to_bounding_box()

        y_overlap = max(0.0, min(bb1.b, bb2.b) - max(bb1.t, bb2.t))
        y_union_span = max(bb1.b, bb2.b) - min(bb1.t, bb2.t)

        return y_overlap / y_union_span if y_union_span > 0 else 0.0

    def _find_close_right_and_left(
        self,
        special_char_cell: TextCell,
        word_cells: List[TextCell],
        threshold: float,
        only_numbers: bool = False,
    ) -> Tuple[bool, Optional[TextCell], Optional[TextCell]]:
        special_char_bb = special_char_cell.rect.to_bounding_box()
        special_char_left_coord = special_char_bb.l
        special_char_right_coord = special_char_bb.r

        left_found_cell: Optional[TextCell] = None
        right_found_cell: Optional[TextCell] = None

        for word_cell in word_cells:
            y_axis_iou = self._get_y_axis_iou(special_char_cell.rect, word_cell.rect)
            if y_axis_iou < 0.6:
                continue

            word_bb = word_cell.rect.to_bounding_box()
            word_left_coord = word_bb.l
            word_right_coord = word_bb.r

            height = (word_bb.b - word_bb.t) + 1.0
            margin = int(threshold * height + 0.5)
            inside_margin_offset = int(self.INSIDE_THRESHOLD * height + 0.5)

            if not left_found_cell:
                left_diff = special_char_left_coord - word_right_coord
                end_char = word_cell.text[-1] if word_cell.text else ""
                if (left_diff <= margin) and (left_diff >= 0):
                    if (not only_numbers) or (only_numbers and end_char.isdigit()):
                        left_found_cell = word_cell
                        if right_found_cell:
                            return True, left_found_cell, right_found_cell

            if not right_found_cell:
                right_diff = word_left_coord - special_char_right_coord
                start_char = word_cell.text[0] if word_cell.text else ""
                if (right_diff <= margin) and (right_diff >= -inside_margin_offset):
                    if (not only_numbers) or (only_numbers and start_char.isdigit()):
                        right_found_cell = word_cell
                        if left_found_cell:
                            return True, left_found_cell, right_found_cell

        return (
            (left_found_cell is not None and right_found_cell is not None),
            left_found_cell,
            right_found_cell,
        )

    def _find_close_right(
        self, special_char_cell: TextCell, word_cells: List[TextCell], threshold: float
    ) -> Tuple[bool, Optional[TextCell]]:
        special_char_bb = special_char_cell.rect.to_bounding_box()
        special_char_right_coord = special_char_bb.r

        right_found_cell: Optional[TextCell] = None

        for word_cell in word_cells:
            y_axis_iou = self._get_y_axis_iou(special_char_cell.rect, word_cell.rect)
            if y_axis_iou < 0.6:
                continue

            word_bb = word_cell.rect.to_bounding_box()
            word_left_coord = word_bb.l
            height = (word_bb.b - word_bb.t) + 1.0
            margin = int(threshold * height + 0.5)
            inside_margin_offset = int(self.INSIDE_THRESHOLD * height + 0.5)

            right_diff = word_left_coord - special_char_right_coord
            if (right_diff <= margin) and (right_diff >= -inside_margin_offset):
                right_found_cell = word_cell
                return True, right_found_cell

        return False, None

    def _find_close_left(
        self, special_char_cell: TextCell, word_cells: List[TextCell], threshold: float
    ) -> Tuple[bool, Optional[TextCell]]:
        special_char_bb = special_char_cell.rect.to_bounding_box()
        special_char_left_coord = special_char_bb.l

        left_found_cell: Optional[TextCell] = None

        for word_cell in word_cells:
            y_axis_iou = self._get_y_axis_iou(special_char_cell.rect, word_cell.rect)
            if y_axis_iou < 0.6:
                continue

            word_bb = word_cell.rect.to_bounding_box()
            word_right_coord = word_bb.r
            height = (word_bb.b - word_bb.t) + 1.0
            margin = int(threshold * height + 0.5)
            inside_margin_offset = int(self.INSIDE_THRESHOLD * height + 0.5)

            left_diff = special_char_left_coord - word_right_coord
            if (left_diff <= margin) and (left_diff >= -inside_margin_offset):
                left_found_cell = word_cell
                return True, left_found_cell

        return False, None

    def _merge_close_left_and_right(
        self,
        current_word_cells: List[TextCell],
        special_char_list: List[str],
        threshold: float,
        only_numbers: bool,
    ) -> List[TextCell]:
        active_word_cells = copy.deepcopy(current_word_cells)

        processed_list_changed_in_iteration = True
        while processed_list_changed_in_iteration:
            processed_list_changed_in_iteration = False
            for special_char_cell_candidate in active_word_cells:
                if special_char_cell_candidate.text in special_char_list:
                    candidate_neighbors = [
                        cell
                        for cell in active_word_cells
                        if cell is not special_char_cell_candidate
                    ]

                    (
                        found_neighbors,
                        left_neighbor_cell,
                        right_neighbor_cell,
                    ) = self._find_close_right_and_left(
                        special_char_cell_candidate,
                        candidate_neighbors,
                        threshold,
                        only_numbers,
                    )

                    if found_neighbors and left_neighbor_cell and right_neighbor_cell:
                        special_char_bb = (
                            special_char_cell_candidate.rect.to_bounding_box()
                        )
                        left_bb = left_neighbor_cell.rect.to_bounding_box()
                        right_bb = right_neighbor_cell.rect.to_bounding_box()

                        new_l = left_bb.l
                        new_r = right_bb.r
                        new_t = min(left_bb.t, special_char_bb.t, right_bb.t)
                        new_b = max(left_bb.b, special_char_bb.b, right_bb.b)

                        current_coord_origin = (
                            special_char_cell_candidate.rect.coord_origin
                        )
                        new_bounding_rect = BoundingRectangle(
                            r_x0=new_l,
                            r_y0=new_t,
                            r_x1=new_r,
                            r_y1=new_t,
                            r_x2=new_r,
                            r_y2=new_b,
                            r_x3=new_l,
                            r_y3=new_b,
                            coord_origin=current_coord_origin,
                        )

                        new_text_val = (
                            left_neighbor_cell.text
                            + special_char_cell_candidate.text
                            + right_neighbor_cell.text
                        )
                        new_orig_val = (
                            left_neighbor_cell.orig
                            + special_char_cell_candidate.orig
                            + right_neighbor_cell.orig
                        )
                        new_from_ocr_val = (
                            left_neighbor_cell.from_ocr
                            or special_char_cell_candidate.from_ocr
                            or right_neighbor_cell.from_ocr
                        )

                        newly_merged_cell = TextCell(
                            rect=new_bounding_rect,
                            text=new_text_val,
                            orig=new_orig_val,
                            from_ocr=new_from_ocr_val,
                            confidence=special_char_cell_candidate.confidence,
                            text_direction=special_char_cell_candidate.text_direction,
                        )

                        active_word_cells.remove(left_neighbor_cell)
                        active_word_cells.remove(special_char_cell_candidate)
                        active_word_cells.remove(right_neighbor_cell)
                        active_word_cells.append(newly_merged_cell)

                        processed_list_changed_in_iteration = True
                        break
        return active_word_cells

    def _merge_to_the_right(
        self,
        current_word_cells: List[TextCell],
        special_char_list: List[str],
        threshold: float,
    ) -> List[TextCell]:
        active_word_cells = copy.deepcopy(current_word_cells)
        processed_list_changed_in_iteration = True
        while processed_list_changed_in_iteration:
            processed_list_changed_in_iteration = False
            for leading_cell_candidate in active_word_cells:
                if (
                    leading_cell_candidate.text
                    and leading_cell_candidate.text[-1] in special_char_list
                ):
                    candidate_neighbors = [
                        cell
                        for cell in active_word_cells
                        if cell is not leading_cell_candidate
                    ]

                    found_neighbor, right_neighbor_cell = self._find_close_right(
                        leading_cell_candidate, candidate_neighbors, threshold
                    )

                    if found_neighbor and right_neighbor_cell:
                        leading_bb = leading_cell_candidate.rect.to_bounding_box()
                        right_bb = right_neighbor_cell.rect.to_bounding_box()

                        new_l = leading_bb.l
                        new_r = right_bb.r
                        new_t = min(leading_bb.t, right_bb.t)
                        new_b = max(leading_bb.b, right_bb.b)

                        current_coord_origin = leading_cell_candidate.rect.coord_origin
                        new_bounding_rect = BoundingRectangle(
                            r_x0=new_l,
                            r_y0=new_t,
                            r_x1=new_r,
                            r_y1=new_t,
                            r_x2=new_r,
                            r_y2=new_b,
                            r_x3=new_l,
                            r_y3=new_b,
                            coord_origin=current_coord_origin,
                        )

                        new_text_val = (
                            leading_cell_candidate.text + right_neighbor_cell.text
                        )
                        new_orig_val = (
                            leading_cell_candidate.orig + right_neighbor_cell.orig
                        )
                        new_from_ocr_val = (
                            leading_cell_candidate.from_ocr
                            or right_neighbor_cell.from_ocr
                        )

                        newly_merged_cell = TextCell(
                            rect=new_bounding_rect,
                            text=new_text_val,
                            orig=new_orig_val,
                            from_ocr=new_from_ocr_val,
                            confidence=leading_cell_candidate.confidence,
                            text_direction=leading_cell_candidate.text_direction,
                        )

                        active_word_cells.remove(leading_cell_candidate)
                        active_word_cells.remove(right_neighbor_cell)
                        active_word_cells.append(newly_merged_cell)

                        processed_list_changed_in_iteration = True
                        break
        return active_word_cells

    def _merge_to_the_left(
        self,
        current_word_cells: List[TextCell],
        special_char_list: List[str],
        threshold: float,
    ) -> List[TextCell]:
        active_word_cells = copy.deepcopy(current_word_cells)
        processed_list_changed_in_iteration = True
        while processed_list_changed_in_iteration:
            processed_list_changed_in_iteration = False
            for trailing_cell_candidate in active_word_cells:
                if (
                    trailing_cell_candidate.text
                    and trailing_cell_candidate.text[0] in special_char_list
                ):
                    candidate_neighbors = [
                        cell
                        for cell in active_word_cells
                        if cell is not trailing_cell_candidate
                    ]

                    found_neighbor, left_neighbor_cell = self._find_close_left(
                        trailing_cell_candidate, candidate_neighbors, threshold
                    )

                    if found_neighbor and left_neighbor_cell:
                        trailing_bb = trailing_cell_candidate.rect.to_bounding_box()
                        left_bb = left_neighbor_cell.rect.to_bounding_box()

                        new_l = left_bb.l
                        new_r = trailing_bb.r
                        new_t = min(left_bb.t, trailing_bb.t)
                        new_b = max(left_bb.b, trailing_bb.b)

                        current_coord_origin = trailing_cell_candidate.rect.coord_origin
                        new_bounding_rect = BoundingRectangle(
                            r_x0=new_l,
                            r_y0=new_t,
                            r_x1=new_r,
                            r_y1=new_t,
                            r_x2=new_r,
                            r_y2=new_b,
                            r_x3=new_l,
                            r_y3=new_b,
                            coord_origin=current_coord_origin,
                        )

                        new_text_val = (
                            left_neighbor_cell.text + trailing_cell_candidate.text
                        )
                        new_orig_val = (
                            left_neighbor_cell.orig + trailing_cell_candidate.orig
                        )
                        new_from_ocr_val = (
                            left_neighbor_cell.from_ocr
                            or trailing_cell_candidate.from_ocr
                        )

                        newly_merged_cell = TextCell(
                            rect=new_bounding_rect,
                            text=new_text_val,
                            orig=new_orig_val,
                            from_ocr=new_from_ocr_val,
                            confidence=trailing_cell_candidate.confidence,
                            text_direction=trailing_cell_candidate.text_direction,
                        )

                        active_word_cells.remove(left_neighbor_cell)
                        active_word_cells.remove(trailing_cell_candidate)
                        active_word_cells.append(newly_merged_cell)

                        processed_list_changed_in_iteration = True
                        break
        return active_word_cells

    def apply_word_merging_to_page(self, page: SegmentedPage) -> SegmentedPage:
        initial_word_cells: List[TextCell] = list(page.word_cells)

        merged_cells_step1 = self._merge_close_left_and_right(
            initial_word_cells,
            special_char_list=self.SPECIAL_CHARS,
            threshold=self.CLOSE_THRESHOLD,
            only_numbers=False,
        )
        merged_cells_step2 = self._merge_close_left_and_right(
            merged_cells_step1,
            special_char_list=list(",.-/"),
            threshold=self.NUMBERS_CLOSE_THRESHOLD,
            only_numbers=True,
        )
        merged_cells_step3 = self._merge_to_the_left(
            merged_cells_step2,
            special_char_list=list(",."),
            threshold=self.CLOSE_LEFT_THRESHOLD,
        )
        merged_cells_step4 = self._merge_to_the_left(
            merged_cells_step3,
            special_char_list=self.SPECIAL_CHARS,
            threshold=self.CLOSE_THRESHOLD,
        )
        merged_cells_step5 = self._merge_to_the_right(
            merged_cells_step4,
            special_char_list=self.SPECIAL_CHARS,
            threshold=self.CLOSE_THRESHOLD,
        )
        merged_cells_step6 = self._merge_to_the_left(
            merged_cells_step5,
            special_char_list=list(")]}"),
            threshold=self.CLOSE_LEFT_THRESHOLD,
        )
        final_merged_cells = self._merge_to_the_right(
            merged_cells_step6,
            special_char_list=list("([{"),
            threshold=self.CLOSE_LEFT_THRESHOLD,
        )

        page.word_cells = final_merged_cells
        return page


class GoogleDocAIPredictionProvider(BasePredictionProvider):
    def __init__(
        self,
        do_visualization: bool = False,
        ignore_missing_predictions: bool = True,
        true_labels: Optional[Set[DocItemLabel]] = None,
        pred_labels: Optional[Set[DocItemLabel]] = None,
    ):
        super().__init__(
            do_visualization=do_visualization,
            ignore_missing_predictions=ignore_missing_predictions,
            true_labels=true_labels,
            pred_labels=pred_labels,
        )

        if not hasattr(documentai, "DocumentProcessorServiceClient"):
            raise ValueError(
                "Error: google-cloud-documentai library not installed. Google Doc AI functionality will be disabled."
            )

        google_location = os.getenv("GOOGLE_LOCATION", "us")
        google_processor_id = os.getenv("GOOGLE_PROCESSOR_ID")

        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if credentials_path is None:
            raise ValueError(
                "GOOGLE_APPLICATION_CREDENTIALS must be set in environment variables."
            )
        with open(credentials_path) as f:
            creds_json = json.load(f)
            google_project_id = creds_json.get("project_id")

        if not google_processor_id:
            raise ValueError(
                "GOOGLE_PROCESSOR_ID must be set in environment variables."
            )

        credentials = service_account.Credentials.from_service_account_file(
            credentials_path
        )

        self.doc_ai_client = documentai.DocumentProcessorServiceClient(
            credentials=credentials
        )

        self.google_processor_name = f"projects/{google_project_id}/locations/{google_location}/processors/{google_processor_id}"
        self._word_merger = _WordMerger()

    def extract_bbox_from_vertices(self, vertices):
        if len(vertices) >= 4:
            return {
                "l": vertices[0].get("x", 0),
                "t": vertices[0].get("y", 0),
                "r": vertices[2].get("x", 0),
                "b": vertices[2].get("y", 0),
            }
        return {"l": 0, "t": 0, "r": 0, "b": 0}

    def extract_bbox_from_normalized_vertices(self, normalized_vertices, width, height):
        if len(normalized_vertices) >= 4:
            return {
                "l": normalized_vertices[0].get("x", 0) * width,
                "t": normalized_vertices[0].get("y", 0) * height,
                "r": normalized_vertices[2].get("x", 0) * width,
                "b": normalized_vertices[2].get("y", 0) * height,
            }
        return {"l": 0, "t": 0, "r": 0, "b": 0}

    def process_table_row(
        self,
        row,
        row_index,
        document,
        table_data,
        page_width,
        page_height,
        is_header=False,
    ):
        for cell_index, cell in enumerate(row.get("cells", [])):
            cell_text_content = ""
            if "layout" in cell and "textAnchor" in cell["layout"]:
                for text_segment in cell["layout"]["textAnchor"].get(
                    "textSegments", []
                ):
                    start_index = int(text_segment.get("startIndex", 0))
                    end_index = int(text_segment.get("endIndex", 0))
                    if document.get("text") and start_index < len(document["text"]):
                        cell_text_content += document["text"][start_index:end_index]

            vertices = (
                cell.get("layout", {}).get("boundingPoly", {}).get("vertices", [])
            )
            normalized_vertices = (
                cell.get("layout", {})
                .get("boundingPoly", {})
                .get("normalizedVertices", [])
            )

            if vertices:
                cell_bbox = self.extract_bbox_from_vertices(vertices)
            else:
                cell_bbox = self.extract_bbox_from_normalized_vertices(
                    normalized_vertices, page_width, page_height
                )

            row_span = cell.get("rowSpan", 1)
            col_span = cell.get("colSpan", 1)

            table_cell = TableCell(
                bbox=BoundingBox(
                    l=cell_bbox["l"],
                    t=cell_bbox["t"],
                    r=cell_bbox["r"],
                    b=cell_bbox["b"],
                    coord_origin=CoordOrigin.TOPLEFT,
                ),
                row_span=row_span,
                col_span=col_span,
                start_row_offset_idx=row_index,
                end_row_offset_idx=row_index + row_span,
                start_col_offset_idx=cell_index,
                end_col_offset_idx=cell_index + col_span,
                text=cell_text_content.strip(),
                column_header=is_header,
                row_header=not is_header and cell_index == 0,
                row_section=False,
            )

            table_data.table_cells.append(table_cell)

    def convert_google_output_to_docling(self, document, record: DatasetRecord):
        doc = DoclingDocument(name=record.doc_id)
        segmented_pages: Dict[int, SegmentedPage] = {}

        for page in document.get("pages", []):
            page_no = page.get("pageNumber", 1)
            page_width = page.get("dimension", {}).get("width", 0)
            page_height = page.get("dimension", {}).get("height", 0)

            im = record.ground_truth_page_images[page_no - 1]

            image_ref = ImageRef(
                mimetype=f"image/png",
                dpi=72,
                size=Size(width=float(im.width), height=float(im.height)),
                uri=from_pil_to_base64uri(im),
            )
            page_item = PageItem(
                page_no=page_no,
                size=Size(width=float(page_width), height=float(page_height)),
                image=image_ref,
            )
            doc.pages[page_no] = page_item

            if page_no not in segmented_pages.keys():
                seg_page = SegmentedPage(
                    dimension=PageGeometry(
                        angle=0,
                        rect=BoundingRectangle.from_bounding_box(
                            BoundingBox(
                                l=0,
                                t=0,
                                r=page_item.size.width,
                                b=page_item.size.height,
                            )
                        ),
                    )
                )
                segmented_pages[page_no] = seg_page

            for table in page.get("tables", []):

                vertices = (
                    table.get("layout", {}).get("boundingPoly", {}).get("vertices", [])
                )
                normalized_vertices = (
                    table.get("layout", {})
                    .get("boundingPoly", {})
                    .get("normalizedVertices", [])
                )

                if vertices:
                    table_bbox = self.extract_bbox_from_vertices(vertices)
                else:
                    table_bbox = self.extract_bbox_from_normalized_vertices(
                        normalized_vertices, page_width, page_height
                    )

                num_rows = len(table.get("headerRows", [])) + len(
                    table.get("bodyRows", [])
                )
                num_cols = 0
                table_bbox_obj = BoundingBox(
                    l=table_bbox["l"],
                    t=table_bbox["t"],
                    r=table_bbox["r"],
                    b=table_bbox["b"],
                    coord_origin=CoordOrigin.TOPLEFT,
                )

                table_prov = ProvenanceItem(
                    page_no=page_no, bbox=table_bbox_obj, charspan=(0, 0)
                )

                table_data = TableData(
                    table_cells=[],
                    num_rows=num_rows,
                    num_cols=0,
                    grid=[],
                )

                for row_index, row in enumerate(table.get("headerRows", [])):
                    num_cols = max(table_data.num_cols, len(row.get("cells", [])))
                    table_data.num_cols = num_cols

                    self.process_table_row(
                        row,
                        row_index,
                        document,
                        table_data,
                        page_width,
                        page_height,
                        is_header=True,
                    )

                header_row_count = len(table.get("headerRows", []))
                for row_index, row in enumerate(table.get("bodyRows", [])):
                    actual_row_index = header_row_count + row_index
                    num_cols = max(table_data.num_cols, len(row.get("cells", [])))
                    table_data.num_cols = num_cols

                    self.process_table_row(
                        row,
                        actual_row_index,
                        document,
                        table_data,
                        page_width,
                        page_height,
                        is_header=False,
                    )

                doc.add_table(data=table_data, prov=table_prov)

            for paragraph in page.get("paragraphs", []):
                text_content = ""
                if "layout" in paragraph and "textAnchor" in paragraph["layout"]:
                    for text_segment in paragraph["layout"]["textAnchor"].get(
                        "textSegments", []
                    ):
                        if "endIndex" in text_segment:
                            start_index = int(text_segment.get("startIndex", 0))
                            end_index = int(text_segment.get("endIndex", 0))
                            if document.get("text") and start_index < len(
                                document["text"]
                            ):
                                text_content += document["text"][start_index:end_index]

                vertices = (
                    paragraph.get("layout", {})
                    .get("boundingPoly", {})
                    .get("vertices", [])
                )
                normalized_vertices = (
                    paragraph.get("layout", {})
                    .get("boundingPoly", {})
                    .get("normalizedVertices", [])
                )

                if vertices:
                    para_bbox = self.extract_bbox_from_vertices(vertices)
                else:
                    para_bbox = self.extract_bbox_from_normalized_vertices(
                        normalized_vertices, page_width, page_height
                    )

                bbox_obj = BoundingBox(
                    l=para_bbox["l"],
                    t=para_bbox["t"],
                    r=para_bbox["r"],
                    b=para_bbox["b"],
                    coord_origin=CoordOrigin.TOPLEFT,
                )

                if any(
                    does_intersection_area_exceed_threshold(
                        bbox_obj, table.prov[0].bbox, 0.8
                    )
                    for table in doc.tables
                ):
                    continue

                prov = ProvenanceItem(
                    page_no=page_no, bbox=bbox_obj, charspan=(0, len(text_content))
                )

                doc.add_text(label=DocItemLabel.TEXT, text=text_content, prov=prov)

            for token in page.get("tokens", []):
                text_content = ""
                if "layout" in token and "textAnchor" in token["layout"]:
                    for text_segment in token["layout"]["textAnchor"].get(
                        "textSegments", []
                    ):
                        if "endIndex" in text_segment:
                            start_index = int(text_segment.get("startIndex", 0))
                            end_index = int(text_segment.get("endIndex", 0))
                            if document.get("text") and start_index < len(
                                document["text"]
                            ):
                                text_content += document["text"][start_index:end_index]

                token_bbox = (
                    None if not vertices else self.extract_bbox_from_vertices(vertices)
                )
                vertices = (
                    token.get("layout", {}).get("boundingPoly", {}).get("vertices", [])
                )
                normalized_vertices = (
                    token.get("layout", {})
                    .get("boundingPoly", {})
                    .get("normalizedVertices", [])
                )

                if vertices:
                    token_bbox = self.extract_bbox_from_vertices(vertices)
                else:
                    token_bbox = self.extract_bbox_from_normalized_vertices(
                        normalized_vertices, page_width, page_height
                    )

                if text_content and token_bbox:
                    bbox_obj = BoundingBox(
                        l=token_bbox["l"],
                        t=token_bbox["t"],
                        r=token_bbox["r"],
                        b=token_bbox["b"],
                        coord_origin=CoordOrigin.TOPLEFT,
                    )
                    segmented_pages[page_no].word_cells.append(
                        TextCell(
                            rect=BoundingRectangle.from_bounding_box(bbox_obj),
                            text=text_content,
                            orig=text_content,
                            from_ocr=False,
                        )
                    )

            segmented_pages[page_no] = self._word_merger.apply_word_merging_to_page(
                segmented_pages[page_no]
            )
        return doc, segmented_pages

    @property
    def prediction_format(self) -> PredictionFormats:
        """Get the prediction format."""
        return PredictionFormats.JSON

    def predict(self, record: DatasetRecord) -> DatasetRecordWithPrediction:
        """For the given document stream (single document), run the API and create the doclingDocument."""

        status = ConversionStatus.SUCCESS
        assert record.original is not None

        if not isinstance(record.original, DocumentStream):
            raise RuntimeError(
                "Original document must be a DocumentStream for PDF or image files"
            )

        result_json = {}
        pred_doc = None
        pred_segmented_pages = {}

        try:
            if record.mime_type in ["application/pdf", "image/png", "image/jpeg"]:
                file_content = record.original.stream.read()
                record.original.stream.seek(0)
                raw_document = documentai.RawDocument(
                    content=file_content, mime_type=record.mime_type
                )

                # Optional: Additional configurations for Document OCR Processor.
                # For more information: https://cloud.google.com/document-ai/docs/enterprise-document-ocr
                process_options = documentai.ProcessOptions(
                    ocr_config=documentai.OcrConfig(
                        enable_native_pdf_parsing=True,
                        enable_image_quality_scores=True,
                        enable_symbol=True,
                        # OCR Add Ons https://cloud.google.com/document-ai/docs/ocr-add-ons
                        # If these are not specified, tables are not output
                        premium_features=documentai.OcrConfig.PremiumFeatures(
                            compute_style_info=False,
                            enable_math_ocr=False,
                            enable_selection_mark_detection=True,
                        ),
                    ),
                    # Although the docs say this is not applicable to OCR and FORM parser, it actually works with OCR parser and outputs the tables
                    layout_config=documentai.ProcessOptions.LayoutConfig(
                        chunking_config=documentai.ProcessOptions.LayoutConfig.ChunkingConfig(
                            include_ancestor_headings=True
                        )
                    ),
                )
                request = documentai.ProcessRequest(
                    name=self.google_processor_name,
                    raw_document=raw_document,
                    process_options=process_options,
                )
                response = self.doc_ai_client.process_document(request=request)
                result_json = MessageToDict(response.document._pb)
                _log.info(
                    f"Successfully processed [{record.doc_id}] using Google Document AI API!"
                )
                pred_doc, pred_segmented_pages = self.convert_google_output_to_docling(
                    result_json, record
                )
            else:
                raise RuntimeError(
                    f"Unsupported mime type: {record.mime_type}. GoogleDocAIPredictionProvider supports 'application/pdf' and 'image/png'"
                )
        except Exception as e:
            _log.error(f"Error in Google DocAI prediction: {str(e)}")
            status = ConversionStatus.FAILURE
            if not self.ignore_missing_predictions:
                raise
            pred_doc = record.ground_truth_doc.model_copy(deep=True)

        pred_record = self.create_dataset_record_with_prediction(
            record, pred_doc, json.dumps(result_json)
        )
        pred_record.predicted_segmented_pages = pred_segmented_pages
        pred_record.status = status
        return pred_record

    def info(self) -> Dict:
        return {
            "asset": PredictionProviderType.GOOGLE,
            "version": importlib.metadata.version("google-cloud-documentai"),
        }
