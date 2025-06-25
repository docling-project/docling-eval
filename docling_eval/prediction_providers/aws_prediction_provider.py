import importlib.metadata
import json
import logging
import os
from typing import Dict, Optional, Set, Tuple

import boto3
from docling.datamodel.base_models import ConversionStatus
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

from docling_eval.datamodels.dataset_record import (
    DatasetRecord,
    DatasetRecordWithPrediction,
)
from docling_eval.datamodels.types import (
    EvaluationModality,
    PredictionFormats,
    PredictionProviderType,
)
from docling_eval.prediction_providers.base_prediction_provider import (
    BasePredictionProvider,
)
from docling_eval.utils.utils import from_pil_to_base64uri

_log = logging.getLogger(__name__)


class AWSTextractPredictionProvider(BasePredictionProvider):
    """Provider that calls the AWS Textract API for predicting the tables in document."""

    prediction_provider_type: PredictionProviderType = PredictionProviderType.AWS

    prediction_modalities = [EvaluationModality.TABLE_STRUCTURE]

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

        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

        if not aws_access_key_id:
            raise ValueError(
                "Environment variable AWS_ACCESS_KEY_ID is not set. This is required to authenticate with AWS."
            )
        if not aws_secret_access_key:
            raise ValueError(
                "Environment variable AWS_SECRET_ACCESS_KEY is not set. This is required to authenticate with AWS."
            )

        region_name = os.getenv("AWS_REGION", "us-east-1")

        self.textract_client = boto3.client(
            "textract",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )

    def extract_bbox_from_geometry(self, geometry):
        """Helper function to extract bbox coordinates from AWS Textract geometry data."""
        if not geometry or not geometry.get("BoundingBox"):
            return {"l": 0, "t": 0, "r": 0, "b": 0}

        bbox = geometry.get("BoundingBox")

        left = bbox.get("Left", 0)
        top = bbox.get("Top", 0)
        width = bbox.get("Width", 0)
        height = bbox.get("Height", 0)

        return {"l": left, "t": top, "r": left + width, "b": top + height}

    def get_cell_content(self, cell, blocks_map):
        """Extract text content from a table cell."""
        text = ""
        if "Relationships" in cell:
            for relationship in cell.get("Relationships", []):
                if relationship["Type"] == "CHILD":
                    for child_id in relationship["Ids"]:
                        child = blocks_map[child_id]
                        if child["BlockType"] == "WORD":
                            text += child["Text"] + " "
        return text.strip()

    def process_table(self, table, blocks_map, page_no, width, height):
        """Process a single table from AWS Textract output."""
        table_cells = []

        table_bbox = self.extract_bbox_from_geometry(table.get("Geometry", {}))
        # Scale normalized coordinates to the page dimensions
        table_bbox_obj = BoundingBox(
            l=table_bbox["l"] * width,
            t=table_bbox["t"] * height,
            r=table_bbox["r"] * width,
            b=table_bbox["b"] * height,
            coord_origin=CoordOrigin.TOPLEFT,
        )

        table_prov = ProvenanceItem(
            page_no=page_no, bbox=table_bbox_obj, charspan=(0, 0)
        )

        cells = []
        if "Relationships" in table:
            for relationship in table.get("Relationships", []):
                if relationship["Type"] == "CHILD":
                    for cell_id in relationship["Ids"]:
                        cell = blocks_map[cell_id]
                        if cell["BlockType"] == "CELL":
                            cells.append(cell)

        max_row = 0
        max_col = 0
        for cell in cells:
            row = cell.get("RowIndex", 0)
            col = cell.get("ColumnIndex", 0)
            max_row = max(max_row, row)
            max_col = max(max_col, col)

        for cell in cells:
            row_index = cell.get("RowIndex", 1) - 1
            col_index = cell.get("ColumnIndex", 1) - 1
            row_span = cell.get("RowSpan", 1)
            col_span = cell.get("ColumnSpan", 1)

            # Get cell content
            cell_text = self.get_cell_content(cell, blocks_map)

            # Get cell bbox
            cell_bbox = self.extract_bbox_from_geometry(cell.get("Geometry", {}))

            # Check if the cell is a column header
            entity_types = cell.get("EntityTypes", [])
            is_column_header = True if "COLUMN_HEADER" in entity_types else False

            # Create TableCell object by scaling normalized coordinates to the page dimensions
            table_cell = TableCell(
                bbox=BoundingBox(
                    l=cell_bbox["l"] * width,
                    t=cell_bbox["t"] * height,
                    r=cell_bbox["r"] * width,
                    b=cell_bbox["b"] * height,
                    coord_origin=CoordOrigin.TOPLEFT,
                ),
                row_span=row_span,
                col_span=col_span,
                start_row_offset_idx=row_index,
                end_row_offset_idx=row_index + row_span,
                start_col_offset_idx=col_index,
                end_col_offset_idx=col_index + col_span,
                text=cell_text,
                column_header=is_column_header,
                # AWS does not identify row headers
                row_header=False,
                row_section=False,
            )

            table_cells.append(table_cell)

        table_data = TableData(
            table_cells=table_cells, num_rows=max_row, num_cols=max_col
        )

        return table_prov, table_data

    def convert_aws_output_to_docling(
        self, analyze_result, record: DatasetRecord, file_bytes
    ) -> Tuple[DoclingDocument, Dict[int, SegmentedPage]]:
        """Converts AWS Textract output to DoclingDocument format."""
        doc = DoclingDocument(name=record.doc_id)

        blocks_map = {block["Id"]: block for block in analyze_result.get("Blocks", [])}

        processed_pages = set()
        segmented_pages: Dict[int, SegmentedPage] = {}

        # Get page dimensions from page block
        # AWS provides normalized coordinates, so we need to multiply by a typical page size
        # width = 8.5 * 72  # Standard US Letter width in points
        # height = 11 * 72  # Standard US Letter height in points
        im = record.ground_truth_page_images[0]
        width, height = im.size
        blocks = analyze_result.get("Blocks", [])
        for block in blocks:
            if block["BlockType"] == "PAGE":
                page_no = int(block.get("Page", 1))
                processed_pages.add(page_no)
                im = record.ground_truth_page_images[page_no - 1]

                # Add page with image
                image_ref = ImageRef(
                    mimetype=f"image/png",
                    dpi=72,
                    size=Size(width=float(im.width), height=float(im.height)),
                    uri=from_pil_to_base64uri(im),
                )
                page_item = PageItem(
                    page_no=page_no,
                    size=Size(width=float(im.width), height=float(im.height)),
                    image=image_ref,
                )

                doc.pages[page_no] = page_item

                # Create SegmentedPage Entry if not already present for the page number
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

            elif block["BlockType"] == "WORD" and block.get("Page", 1) == page_no:
                text_content = block.get("Text", None)
                geometry = block.get("Geometry", None)

                if text_content is not None and geometry is not None:
                    bbox = self.extract_bbox_from_geometry(geometry)
                    # Scale normalized coordinates to the page dimensions
                    bbox_obj = BoundingBox(
                        l=bbox["l"] * width,
                        t=bbox["t"] * height,
                        r=bbox["r"] * width,
                        b=bbox["b"] * height,
                        coord_origin=CoordOrigin.TOPLEFT,
                    )

                    segmented_pages[page_no].word_cells.append(
                        TextCell(
                            rect=BoundingRectangle.from_bounding_box(bbox_obj),
                            text=text_content,
                            orig=text_content,
                            # Keeping from_ocr flag False since AWS output doesn't indicate whether the given word is programmatic or OCR
                            from_ocr=False,
                        )
                    )

            elif block["BlockType"] == "LAYOUT_TITLE":
                self._add_title(block, doc, height, page_no, width, blocks_map)

            elif block["BlockType"] == "LAYOUT_HEADER":
                self._add_page_header(block, doc, height, page_no, width, blocks_map)

            elif block["BlockType"] == "LAYOUT_FOOTER":
                self._add_page_footer(block, doc, height, page_no, width, blocks_map)

            elif block["BlockType"] == "LAYOUT_SECTION_HEADER":
                self._add_heading(block, doc, height, page_no, width, blocks_map)

            elif block["BlockType"] == "LAYOUT_PAGE_NUMBER":
                self._add_page_number(block, doc, height, page_no, width, blocks_map)

            elif block["BlockType"] == "LAYOUT_LIST":
                self._add_list(block, doc, height, page_no, width, blocks_map)

            elif block["BlockType"] == "LAYOUT_FIGURE":
                self._add_figure(block, doc, height, page_no, width, blocks_map)

            elif block["BlockType"] == "LAYOUT_KEY_VALUE":
                self._add_key_value(block, doc, height, page_no, width, blocks_map)

            # This condition is to add only the layout of the table as predicted, doesn't contain the cell structure
            elif block["BlockType"] == "LAYOUT_TABLE":
                # We are skipping adding table layout here since this information is duplicated through BlockType TABLE
                continue
                # self._add_table_layout(block, doc, height, page_no, width, blocks_map)

            elif block["BlockType"] == "LAYOUT_TEXT":
                self._add_text(block, doc, height, page_no, width, blocks_map)

            # This condition is to add output from actual tables API which adds detailed table
            elif block["BlockType"] == "TABLE":
                page_no = int(block.get("Page", 1))
                table_prov, table_data = self.process_table(
                    block, blocks_map, page_no, width, height
                )
                doc.add_table(prov=table_prov, data=table_data, caption=None)

        return doc, segmented_pages

    def _add_text(self, block, doc, height, page_no, width, blocks_map):
        """Maps AWS text to Docling text."""
        text_content = self._get_layout_element_text(block, blocks_map)
        bbox = self.extract_bbox_from_geometry(block.get("Geometry", {}))
        # Scale normalized coordinates to the page dimensions
        bbox_obj = BoundingBox(
            l=bbox["l"] * width,
            t=bbox["t"] * height,
            r=bbox["r"] * width,
            b=bbox["b"] * height,
            coord_origin=CoordOrigin.TOPLEFT,
        )
        prov = ProvenanceItem(
            page_no=page_no,
            bbox=bbox_obj,
            charspan=(0, len(text_content)),
        )
        doc.add_text(label=DocItemLabel.TEXT, text=text_content, prov=prov)

    def _add_table_layout(self, block, doc, height, page_no, width, blocks_map):
        """Maps AWS table layout to Docling table layout"""
        text_content = self._get_layout_element_text(block, blocks_map)
        bbox = self.extract_bbox_from_geometry(block.get("Geometry", {}))
        # Scale normalized coordinates to the page dimensions
        bbox_obj = BoundingBox(
            l=bbox["l"] * width,
            t=bbox["t"] * height,
            r=bbox["r"] * width,
            b=bbox["b"] * height,
            coord_origin=CoordOrigin.TOPLEFT,
        )
        prov = ProvenanceItem(
            page_no=page_no,
            bbox=bbox_obj,
            charspan=(0, len(text_content)),
        )
        doc.add_table(data=TableData(), prov=prov)

    def _add_key_value(self, block, doc, height, page_no, width, blocks_map):
        """Maps AWS Kew-Value pairs to Docling text"""
        text_content = self._get_layout_element_text(block, blocks_map)
        bbox = self.extract_bbox_from_geometry(block.get("Geometry", {}))
        # Scale normalized coordinates to the page dimensions
        bbox_obj = BoundingBox(
            l=bbox["l"] * width,
            t=bbox["t"] * height,
            r=bbox["r"] * width,
            b=bbox["b"] * height,
            coord_origin=CoordOrigin.TOPLEFT,
        )
        prov = ProvenanceItem(
            page_no=page_no,
            bbox=bbox_obj,
            charspan=(0, len(text_content)),
        )
        doc.add_text(label=DocItemLabel.TEXT, text=text_content, prov=prov)

    def _add_figure(self, block, doc, height, page_no, width, blocks_map):
        """Maps AWS Figure to Docling picture"""
        text_content = self._get_layout_element_text(block, blocks_map)
        bbox = self.extract_bbox_from_geometry(block.get("Geometry", {}))
        # Scale normalized coordinates to the page dimensions
        bbox_obj = BoundingBox(
            l=bbox["l"] * width,
            t=bbox["t"] * height,
            r=bbox["r"] * width,
            b=bbox["b"] * height,
            coord_origin=CoordOrigin.TOPLEFT,
        )
        prov = ProvenanceItem(
            page_no=page_no,
            bbox=bbox_obj,
            charspan=(0, len(text_content)),
        )
        doc.add_picture(prov=prov)

    def _add_list(self, block, doc, height, page_no, width, blocks_map):
        """Maps AWS List to Docling List"""
        text_content = self._get_layout_element_text(block, blocks_map)
        bbox = self.extract_bbox_from_geometry(block.get("Geometry", {}))
        # Scale normalized coordinates to the page dimensions
        bbox_obj = BoundingBox(
            l=bbox["l"] * width,
            t=bbox["t"] * height,
            r=bbox["r"] * width,
            b=bbox["b"] * height,
            coord_origin=CoordOrigin.TOPLEFT,
        )
        prov = ProvenanceItem(
            page_no=page_no,
            bbox=bbox_obj,
            charspan=(0, len(text_content)),
        )
        doc.add_list_item(text=text_content, prov=prov)

    def _add_page_number(self, block, doc, height, page_no, width, blocks_map):
        """Maps AWS page number to Docling text"""
        text_content = self._get_layout_element_text(block, blocks_map)
        bbox = self.extract_bbox_from_geometry(block.get("Geometry", {}))
        # Scale normalized coordinates to the page dimensions
        bbox_obj = BoundingBox(
            l=bbox["l"] * width,
            t=bbox["t"] * height,
            r=bbox["r"] * width,
            b=bbox["b"] * height,
            coord_origin=CoordOrigin.TOPLEFT,
        )
        prov = ProvenanceItem(
            page_no=page_no,
            bbox=bbox_obj,
            charspan=(0, len(text_content)),
        )
        doc.add_text(label=DocItemLabel.TEXT, text=text_content, prov=prov)

    def _add_heading(self, block, doc, height, page_no, width, blocks_map):
        """Maps AWS section header to Docling section header"""
        text_content = self._get_layout_element_text(block, blocks_map)
        bbox = self.extract_bbox_from_geometry(block.get("Geometry", {}))
        # Scale normalized coordinates to the page dimensions
        bbox_obj = BoundingBox(
            l=bbox["l"] * width,
            t=bbox["t"] * height,
            r=bbox["r"] * width,
            b=bbox["b"] * height,
            coord_origin=CoordOrigin.TOPLEFT,
        )
        prov = ProvenanceItem(
            page_no=page_no,
            bbox=bbox_obj,
            charspan=(0, len(text_content)),
        )
        doc.add_heading(text=text_content, prov=prov)

    def _add_page_footer(self, block, doc, height, page_no, width, blocks_map):
        """Maps AWS page footer to Docling page footer"""
        text_content = self._get_layout_element_text(block, blocks_map)
        bbox = self.extract_bbox_from_geometry(block.get("Geometry", {}))
        # Scale normalized coordinates to the page dimensions
        bbox_obj = BoundingBox(
            l=bbox["l"] * width,
            t=bbox["t"] * height,
            r=bbox["r"] * width,
            b=bbox["b"] * height,
            coord_origin=CoordOrigin.TOPLEFT,
        )
        prov = ProvenanceItem(
            page_no=page_no,
            bbox=bbox_obj,
            charspan=(0, len(text_content)),
        )
        doc.add_text(label=DocItemLabel.PAGE_FOOTER, text=text_content, prov=prov)

    def _add_page_header(self, block, doc, height, page_no, width, blocks_map):
        """Maps AWS page header to Docling page header"""
        text_content = self._get_layout_element_text(block, blocks_map)
        bbox = self.extract_bbox_from_geometry(block.get("Geometry", {}))
        # Scale normalized coordinates to the page dimensions
        bbox_obj = BoundingBox(
            l=bbox["l"] * width,
            t=bbox["t"] * height,
            r=bbox["r"] * width,
            b=bbox["b"] * height,
            coord_origin=CoordOrigin.TOPLEFT,
        )
        prov = ProvenanceItem(
            page_no=page_no,
            bbox=bbox_obj,
            charspan=(0, len(text_content)),
        )
        doc.add_text(label=DocItemLabel.PAGE_HEADER, text=text_content, prov=prov)

    def _add_title(self, block, doc, height, page_no, width, blocks_map):
        """Maps AWS title to Docling title"""
        text_content = self._get_layout_element_text(block, blocks_map)
        bbox = self.extract_bbox_from_geometry(block.get("Geometry", {}))
        # Scale normalized coordinates to the page dimensions
        bbox_obj = BoundingBox(
            l=bbox["l"] * width,
            t=bbox["t"] * height,
            r=bbox["r"] * width,
            b=bbox["b"] * height,
            coord_origin=CoordOrigin.TOPLEFT,
        )
        prov = ProvenanceItem(
            page_no=page_no,
            bbox=bbox_obj,
            charspan=(0, len(text_content)),
        )
        doc.add_title(text=text_content, prov=prov)

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

        try:
            if record.mime_type in [
                "application/pdf",
                "image/png",
                "image/jpg",
                "image/jpeg",
            ]:
                # Call the AWS Textract API by passing in the image for prediction

                file_bytes = record.original.stream.read()
                response = self.textract_client.analyze_document(
                    Document={"Bytes": file_bytes},
                    FeatureTypes=["TABLES", "FORMS", "LAYOUT"],
                )
                result_orig = json.dumps(response, default=str)
                result_json = json.loads(result_orig)
                _log.info(
                    f"Successfully processed [{record.doc_id}] using AWS Textract API!"
                )

                pred_doc, pred_segmented_pages = self.convert_aws_output_to_docling(
                    result_json, record, file_bytes
                )
            else:
                raise RuntimeError(
                    f"Unsupported mime type: {record.mime_type}. AWSTextractPredictionProvider supports 'application/pdf' 'image/png', 'image/jpeg', and 'image/jpg'"
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
            record, pred_doc, result_orig
        )
        pred_record.predicted_segmented_pages = pred_segmented_pages
        pred_record.status = status
        return pred_record

    def info(self) -> Dict:
        return {
            "asset": PredictionProviderType.AWS,
            "version": importlib.metadata.version("boto3"),
        }

    def _get_layout_element_text(self, block, blocks_map) -> str:
        text_content = ""
        separator = " "
        relationships = block.get("Relationships", [])
        children_ids = []
        for relationship in relationships:
            if relationship.get("Type") == "CHILD":
                # Extend the list, in case there are multiple Ids
                children_ids.extend(relationship.get("Ids", []))

        for child_id in children_ids:
            child_block = blocks_map.get(child_id)
            if child_block["BlockType"] == "LINE":
                text_content += child_block["Text"] + separator
            elif child_block["BlockType"] == "LAYOUT_TEXT":
                text_content += self._get_layout_element_text(child_block, blocks_map)
            else:
                _log.warning("Encountered unhandled BlockType!")

            # Remove the trailing separator
        text_content = text_content[: -len(separator)]
        return text_content
