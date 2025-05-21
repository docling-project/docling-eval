import copy
import importlib.metadata
import json
import logging
import os
from typing import Dict, Optional, Set

from docling.datamodel.base_models import ConversionStatus
from docling_core.types import DoclingDocument
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import (
    DoclingDocument,
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
from google.cloud import documentai  # type: ignore
from google.oauth2 import service_account
from google.protobuf.json_format import MessageToDict
from PIL.Image import Image

from docling_eval.datamodels.dataset_record import (
    DatasetRecord,
    DatasetRecordWithPrediction,
)
from docling_eval.datamodels.types import PredictionFormats, PredictionProviderType
from docling_eval.prediction_providers.base_prediction_provider import (
    BasePredictionProvider,
)
from docling_eval.utils.utils import from_pil_to_base64uri

_log = logging.getLogger(__name__)

SPECIAL_CHARS = list("*:;,.?()!@#$%^&[]{}/\\\"'~+-_<>=")
CLOSE_THRESHOLD = 0.3
NUMBERS_CLOSE_THRESHOLD = 0.7
CLOSE_LEFT_THRESHOLD = 0.7
INSIDE_THRESHOLD = 0.15


def get_y_axis_iou(rect1: BoundingRectangle, rect2: BoundingRectangle):
    """Calculate the IoU (Intersection over Union) in the y-axis between two BoundingRectangle objects."""
    top1 = min(rect1.r_y0, rect1.r_y1, rect1.r_y2, rect1.r_y3)
    bottom1 = max(rect1.r_y0, rect1.r_y1, rect1.r_y2, rect1.r_y3)

    top2 = min(rect2.r_y0, rect2.r_y1, rect2.r_y2, rect2.r_y3)
    bottom2 = max(rect2.r_y0, rect2.r_y1, rect2.r_y2, rect2.r_y3)

    y_overlap = max(0, min(bottom1, bottom2) - max(top1, top2))
    y_union = max(bottom1, bottom2) - min(top1, top2)

    return y_overlap / y_union if y_union > 0 else 0


def text_cell_to_word_dict(cell: TextCell):
    """Convert a TextCell to a word dictionary format used by the merging functions."""
    left = min(cell.rect.r_x0, cell.rect.r_x1, cell.rect.r_x2, cell.rect.r_x3)
    top = min(cell.rect.r_y0, cell.rect.r_y1, cell.rect.r_y2, cell.rect.r_y3)
    right = max(cell.rect.r_x0, cell.rect.r_x1, cell.rect.r_x2, cell.rect.r_x3)
    bottom = max(cell.rect.r_y0, cell.rect.r_y1, cell.rect.r_y2, cell.rect.r_y3)

    bbox = (left, top, right, bottom)
    return {
        "word": cell.text,
        "bbox": bbox,
        "rect": cell.rect,
        "orig": cell.orig,
        "from_ocr": cell.from_ocr,
    }


def word_dict_to_text_cell(word_dict):
    """Convert a word dictionary back to a TextCell."""
    if "rect" in word_dict:
        rect = word_dict["rect"]
    else:
        bbox = word_dict["bbox"]
        rect = BoundingRectangle(
            r_x0=bbox[0],
            r_y0=bbox[1],
            r_x1=bbox[2],
            r_y1=bbox[1],
            r_x2=bbox[2],
            r_y2=bbox[3],
            r_x3=bbox[0],
            r_y3=bbox[3],
            coord_origin=CoordOrigin.TOPLEFT,
        )

    return TextCell(
        rect=rect,
        text=word_dict["word"],
        orig=word_dict.get("orig", word_dict["word"]),
        from_ocr=word_dict.get("from_ocr", False),
    )


def find_close_right_and_left(
    special_char_word, words_array, threshold, only_numbers=False
):
    special_char_left = special_char_word["bbox"][0]
    special_char_right = special_char_word["bbox"][2]

    left_found = False
    right_found = False

    left_word = None
    right_word = None
    found_flag = False

    for word in words_array:
        y_axis_iou = get_y_axis_iou(special_char_word["rect"], word["rect"])
        if y_axis_iou < 0.6:
            continue

        bbox = word["bbox"]
        left = bbox[0]
        right = bbox[2]
        top = bbox[1]
        bottom = bbox[3]
        height = bottom - top + 1
        margin = int(threshold * height + 0.5)
        inside_margin = -int(INSIDE_THRESHOLD * height + 0.5)

        if not left_found:
            left_diff = special_char_left - right
            end_char = word["word"][-1] if word["word"] else ""
            if (left_diff <= margin) and (left_diff >= 0):
                if (not only_numbers) or (only_numbers and end_char.isdigit()):
                    left_found = True
                    left_word = word
                    if right_found:
                        found_flag = True
                        break

        if not right_found:
            right_diff = left - special_char_right
            start_char = word["word"][0] if word["word"] else ""
            if (right_diff <= margin) and (right_diff >= inside_margin):
                if (not only_numbers) or (only_numbers and start_char.isdigit()):
                    right_found = True
                    right_word = word
                    if left_found:
                        found_flag = True
                        break

    return found_flag, left_word, right_word


def find_close_right(special_char_word, words_array, threshold):
    special_char_right = special_char_word["bbox"][2]

    right_word = None
    found_flag = False

    for word in words_array:
        y_axis_iou = get_y_axis_iou(special_char_word["rect"], word["rect"])
        if y_axis_iou < 0.6:
            continue

        bbox = word["bbox"]
        left = bbox[0]
        top = bbox[1]
        bottom = bbox[3]
        height = bottom - top + 1
        margin = int(threshold * height + 0.5)
        inside_margin = -int(INSIDE_THRESHOLD * height + 0.5)

        right_diff = left - special_char_right
        if (right_diff <= margin) and (right_diff >= inside_margin):
            right_word = word
            found_flag = True
            break

    return found_flag, right_word


def find_close_left(special_char_word, words_array, threshold):
    special_char_left = special_char_word["bbox"][0]

    left_word = None
    found_flag = False

    for word in words_array:
        y_axis_iou = get_y_axis_iou(special_char_word["rect"], word["rect"])
        if y_axis_iou < 0.6:
            continue

        bbox = word["bbox"]
        right = bbox[2]
        top = bbox[1]
        bottom = bbox[3]
        height = bottom - top + 1
        margin = int(threshold * height + 0.5)
        inside_margin = -int(INSIDE_THRESHOLD * height + 0.5)

        left_diff = special_char_left - right
        if (left_diff <= margin) and (left_diff >= inside_margin):
            left_word = word
            found_flag = True
            break

    return found_flag, left_word


def merge_close_left_and_right(
    words_array, special_char_array, threshold, only_numbers
):
    words_array = copy.deepcopy(words_array)
    found = True
    while found:
        found = False
        for word in words_array:
            if word["word"] in special_char_array:
                words_array_minus_special = copy.deepcopy(words_array)
                words_array_minus_special.remove(word)
                find_flag, left_word, right_word = find_close_right_and_left(
                    word, words_array_minus_special, threshold, only_numbers
                )
                if find_flag:
                    new_content = left_word["word"] + word["word"] + right_word["word"]
                    new_left = left_word["bbox"][0]
                    new_right = right_word["bbox"][2]
                    new_top = min(
                        left_word["bbox"][1], word["bbox"][1], right_word["bbox"][1]
                    )
                    new_bottom = max(
                        left_word["bbox"][3], word["bbox"][3], right_word["bbox"][3]
                    )

                    new_rect = BoundingRectangle(
                        r_x0=new_left,
                        r_y0=new_top,
                        r_x1=new_right,
                        r_y1=new_top,
                        r_x2=new_right,
                        r_y2=new_bottom,
                        r_x3=new_left,
                        r_y3=new_bottom,
                        coord_origin=CoordOrigin.TOPLEFT,
                    )

                    new_word = copy.deepcopy(word)
                    new_word["word"] = new_content
                    new_word["rect"] = new_rect
                    new_word["bbox"] = (new_left, new_top, new_right, new_bottom)
                    new_word["merge"] = True

                    words_array.remove(left_word)
                    words_array.remove(word)
                    words_array.remove(right_word)
                    words_array.append(new_word)
                    found = True
                    break
    return words_array


def merge_to_the_right(words_array, special_char_array, threshold):
    words_array = copy.deepcopy(words_array)
    found = True
    while found:
        found = False
        for word in words_array:
            if word["word"] and word["word"][-1] in special_char_array:
                words_array_minus_special = copy.deepcopy(words_array)
                words_array_minus_special.remove(word)
                find_flag, right_word = find_close_right(
                    word, words_array_minus_special, threshold
                )
                if find_flag:
                    new_content = word["word"] + right_word["word"]
                    new_left = word["bbox"][0]
                    new_right = right_word["bbox"][2]
                    new_top = min(word["bbox"][1], right_word["bbox"][1])
                    new_bottom = max(word["bbox"][3], right_word["bbox"][3])

                    new_rect = BoundingRectangle(
                        r_x0=new_left,
                        r_y0=new_top,
                        r_x1=new_right,
                        r_y1=new_top,
                        r_x2=new_right,
                        r_y2=new_bottom,
                        r_x3=new_left,
                        r_y3=new_bottom,
                        coord_origin=CoordOrigin.TOPLEFT,
                    )

                    new_word = copy.deepcopy(word)
                    new_word["word"] = new_content
                    new_word["rect"] = new_rect
                    new_word["bbox"] = (new_left, new_top, new_right, new_bottom)
                    new_word["merge"] = True

                    words_array.remove(word)
                    words_array.remove(right_word)
                    words_array.append(new_word)
                    found = True
                    break
    return words_array


def merge_to_the_left(words_array, special_char_array, threshold):
    words_array = copy.deepcopy(words_array)
    found = True
    while found:
        found = False
        for word in words_array:
            if word["word"] and word["word"][0] in special_char_array:
                words_array_minus_special = copy.deepcopy(words_array)
                words_array_minus_special.remove(word)
                find_flag, left_word = find_close_left(
                    word, words_array_minus_special, threshold
                )
                if find_flag:
                    new_content = left_word["word"] + word["word"]
                    new_left = left_word["bbox"][0]
                    new_right = word["bbox"][2]
                    new_top = min(left_word["bbox"][1], word["bbox"][1])
                    new_bottom = max(left_word["bbox"][3], word["bbox"][3])

                    new_rect = BoundingRectangle(
                        r_x0=new_left,
                        r_y0=new_top,
                        r_x1=new_right,
                        r_y1=new_top,
                        r_x2=new_right,
                        r_y2=new_bottom,
                        r_x3=new_left,
                        r_y3=new_bottom,
                        coord_origin=CoordOrigin.TOPLEFT,
                    )

                    new_word = copy.deepcopy(word)
                    new_word["word"] = new_content
                    new_word["rect"] = new_rect
                    new_word["bbox"] = (new_left, new_top, new_right, new_bottom)
                    new_word["merge"] = True

                    words_array.remove(left_word)
                    words_array.remove(word)
                    words_array.append(new_word)
                    found = True
                    break
    return words_array


def _apply_word_merging_to_page(page: SegmentedPage) -> SegmentedPage:
    words_arr = [text_cell_to_word_dict(cell) for cell in page.word_cells]

    new_words_arr = merge_close_left_and_right(
        words_arr,
        special_char_array=SPECIAL_CHARS,
        threshold=CLOSE_THRESHOLD,
        only_numbers=False,
    )
    new_words_arr2 = merge_close_left_and_right(
        new_words_arr,
        special_char_array=list(",.-/"),
        threshold=NUMBERS_CLOSE_THRESHOLD,
        only_numbers=True,
    )
    new_words_arr3 = merge_to_the_left(
        new_words_arr2,
        special_char_array=list(",."),
        threshold=CLOSE_LEFT_THRESHOLD,
    )
    new_words_arr4 = merge_to_the_left(
        new_words_arr3, special_char_array=SPECIAL_CHARS, threshold=CLOSE_THRESHOLD
    )
    new_words_arr5 = merge_to_the_right(
        new_words_arr4, special_char_array=SPECIAL_CHARS, threshold=CLOSE_THRESHOLD
    )
    new_words_arr6 = merge_to_the_left(
        new_words_arr5,
        special_char_array=list(")]}"),
        threshold=CLOSE_LEFT_THRESHOLD,
    )
    new_words_arr7 = merge_to_the_right(
        new_words_arr6,
        special_char_array=list("([{"),
        threshold=CLOSE_LEFT_THRESHOLD,
    )

    merged_cells = [word_dict_to_text_cell(word) for word in new_words_arr7]

    page.word_cells = merged_cells
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

    def extract_bbox_from_vertices(self, vertices):
        if len(vertices) >= 4:
            return {
                "l": vertices[0].get("x", 0),
                "t": vertices[0].get("y", 0),
                "r": vertices[2].get("x", 0),
                "b": vertices[2].get("y", 0),
            }
        return {"l": 0, "t": 0, "r": 0, "b": 0}

    def process_table_row(self, row, row_index, document, table_data, is_header=False):
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

            cell_bbox = self.extract_bbox_from_vertices(
                cell.get("layout", {}).get("boundingPoly", {}).get("vertices", [])
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

                para_bbox = self.extract_bbox_from_vertices(
                    paragraph.get("layout", {})
                    .get("boundingPoly", {})
                    .get("vertices", [])
                )

                bbox_obj = BoundingBox(
                    l=para_bbox["l"],
                    t=para_bbox["t"],
                    r=para_bbox["r"],
                    b=para_bbox["b"],
                    coord_origin=CoordOrigin.TOPLEFT,
                )

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

                vertices = (
                    token.get("layout", {}).get("boundingPoly", {}).get("vertices", [])
                )
                token_bbox = (
                    None if not vertices else self.extract_bbox_from_vertices(vertices)
                )

                if text_content and token_bbox is not None:
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

            for table in page.get("tables", []):
                table_bbox = self.extract_bbox_from_vertices(
                    table.get("layout", {}).get("boundingPoly", {}).get("vertices", [])
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
                        row, row_index, document, table_data, is_header=True
                    )

                header_row_count = len(table.get("headerRows", []))
                for row_index, row in enumerate(table.get("bodyRows", [])):
                    actual_row_index = header_row_count + row_index
                    num_cols = max(table_data.num_cols, len(row.get("cells", [])))
                    table_data.num_cols = num_cols

                    self.process_table_row(
                        row, actual_row_index, document, table_data, is_header=False
                    )

                doc.add_table(data=table_data, prov=table_prov)

            segmented_pages[page_no] = _apply_word_merging_to_page(
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
            if record.mime_type in ["application/pdf", "image/png"]:
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
                            enable_math_ocr=False,  # Enable to use Math OCR Model
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
