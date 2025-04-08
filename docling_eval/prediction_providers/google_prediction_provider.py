import json
import logging
import os
from io import BytesIO
from pathlib import Path
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
from docling_core.types.io import DocumentStream
from google.cloud import documentai
from google.protobuf.json_format import MessageToDict  # Convert to JSON for storage
from PIL.Image import Image

from docling_eval.datamodels.dataset_record import (
    DatasetRecord,
    DatasetRecordWithPrediction,
)
from docling_eval.prediction_providers.base_prediction_provider import (
    BasePredictionProvider,
)
from docling_eval.utils.utils import from_pil_to_base64uri

_log = logging.getLogger(__name__)


class GoogleDocAIPredictionProvider(BasePredictionProvider):
    """Provider that calls the Google Document AI API for predicting the tables in document."""

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
        google_project_id = os.getenv("GOOGLE_PROJECT_ID")
        google_location = os.getenv("GOOGLE_LOCATION", "us")
        google_processor_id = os.getenv("GOOGLE_PROCESSOR_ID")

        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            raise ValueError(
                "GOOGLE_APPLICATION_CREDENTIALS must be set in environment variables."
            )

        if not google_project_id or not google_processor_id:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT_ID and GOOGLE_DOCAI_PROCESSOR_ID must be set in environment variables."
            )

        self.doc_ai_client = documentai.DocumentProcessorServiceClient()
        self.google_processor_name = f"projects/{google_project_id}/locations/{google_location}/processors/{google_processor_id}"

    def extract_bbox_from_vertices(self, vertices):
        """Helper function to extract bbox coordinates from vertices."""
        if len(vertices) >= 4:
            return {
                "l": vertices[0].get("x", 0),
                "t": vertices[0].get("y", 0),
                "r": vertices[2].get("x", 0),
                "b": vertices[2].get("y", 0),
            }
        return {"l": 0, "t": 0, "r": 0, "b": 0}

    def process_table_row(self, row, row_index, document, table_data, is_header=False):
        """Process a table row and add cells to table_data."""
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
                end_row_offset_idx=row_index + row_span - 1,
                start_col_offset_idx=cell_index,
                end_col_offset_idx=cell_index + col_span - 1,
                text=cell_text_content.strip(),
                column_header=is_header,
                row_header=not is_header
                and cell_index == 0,  # First column might be row header
                row_section=False,
            )

            table_data.table_cells.append(table_cell)

    def convert_google_output_to_docling(self, document, doc_id, file_bytes):
        """Converts Google Document AI output to DoclingDocument format."""
        doc = DoclingDocument(name=doc_id)

        image = Image.open(BytesIO(file_bytes))

        # Add page with image
        image_ref = ImageRef(
            mimetype=f"image/png",
            dpi=72,
            size=Size(width=float(image.width), height=float(image.height)),
            uri=from_pil_to_base64uri(im),
        )
        page_item = PageItem(
            page_no=1,
            size=Size(width=float(image.width), height=float(image.height)),
            image=image_ref,
        )
        doc.pages[1] = page_item

        for page in document.get("pages", []):
            page_no = page.get("pageNumber", 1)
            page_width = page.get("dimension", {}).get("width", 0)
            page_height = page.get("dimension", {}).get("height", 0)
            doc.pages[page_no] = PageItem(
                size=Size(width=float(page_width), height=float(page_height)),
                page_no=page_no,
            )

            for paragraph in page.get("paragraphs", []):
                # Extract text content from text_anchor and text_segments
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

                # Extract paragraph bounding box
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

                doc.add_text(label=DocItemLabel.PARAGRAPH, text=text_content, prov=prov)

            for table in page.get("tables", []):
                table_bbox = self.extract_bbox_from_vertices(
                    table.get("layout", {}).get("boundingPoly", {}).get("vertices", [])
                )

                num_rows = len(table.get("headerRows", [])) + len(
                    table.get("bodyRows", [])
                )
                num_cols = 0  # Will be calculated based on cells
                table_bbox_obj = BoundingBox(
                    l=table_bbox["l"],
                    t=table_bbox["t"],
                    r=table_bbox["r"],
                    b=table_bbox["b"],
                    coord_origin=CoordOrigin.TOPLEFT,
                )

                table_prov = ProvenanceItem(
                    page_no=page_no, bbox=table_bbox_obj, charspan=[]
                )

                table_data = TableData(
                    table_cells=[],
                    num_rows=num_rows,
                    num_cols=0,  # Will update as we process cells
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

                doc.add_table(table_data=table_data, prov=table_prov)

        return doc

    def predict(self, record: DatasetRecord) -> DatasetRecordWithPrediction:
        """For the given document stream (single document), run the API and create the doclingDocument."""

        status = ConversionStatus.SUCCESS
        assert record.original is not None

        if not isinstance(record.original, DocumentStream):
            raise RuntimeError(
                "Original document must be a DocumentStream for PDF or image files"
            )

        try:
            if record.mime_type in ["application/pdf", "image/png"]:
                # Get file content and mime type
                file_content = record.original.stream.read()

                # Reset stream position
                record.original.stream.seek(0)

                # Process the document
                raw_document = documentai.RawDocument(
                    content=file_content, mime_type=record.mime_type
                )
                request = documentai.ProcessRequest(
                    name=self.google_processor_name, raw_document=raw_document
                )
                response = self.doc_ai_client.process_document(request=request)
                result_json = MessageToDict(response.document._pb)
                _log.info(
                    f"Successfully processed [{record.doc_id}] using Google Document AI API!"
                )

                pred_doc = self.convert_google_output_to_docling(
                    result_json, record.doc_id, file_content
                )
            else:
                raise RuntimeError(
                    f"Unsupported mime type: {record.mime_type}. GoogleDocAIPredictionProvider supports 'application/pdf' and 'image/png'"
                )
        except Exception as e:
            _log.error(f"Error in AWS Textract prediction: {str(e)}")
            status = ConversionStatus.FAILURE
            if not self.ignore_missing_predictions:
                raise
            pred_doc = record.ground_truth_doc.model_copy(
                deep=True
            )  # Use copy of ground truth as fallback

        pred_record = self.create_dataset_record_with_prediction(
            record, pred_doc, json.dumps(result_json)
        )
        pred_record.status = status
        return pred_record

    def info(self) -> Dict:
        return {"asset": "Google Document AI", "version": "1.0.0"}
