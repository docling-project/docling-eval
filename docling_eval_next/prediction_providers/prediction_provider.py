import os
from pathlib import Path
from abc import abstractmethod
from typing import Dict

from docling.document_converter import DocumentConverter
from docling_core.types import DoclingDocument
from docling_core.types.io import DocumentStream

from docling_eval.benchmarks.utils import docling_version

from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import (
    DoclingDocument,
    PageItem,
    ProvenanceItem,
    TableCell,
    TableData,
    TableItem,
)
from docling_core.types.doc.labels import DocItemLabel
from PIL import Image

class BasePredictionProvider:
    def __init__(
        self, **kwargs
    ):  # could be the docling converter options, the remote credentials for MS/Google, etc.
        self.provider_args = kwargs

    @abstractmethod
    def predict(self, stream: DocumentStream, **extra_args) -> DoclingDocument:
        return DoclingDocument(name="dummy")

    @abstractmethod
    def info(self) -> Dict:
        return {}


class DoclingPredictionProvider(BasePredictionProvider):
    def __init__(
        self, **kwargs
    ):  # could be the docling converter options, the remote credentials for MS/Google, etc.
        super().__init__(**kwargs)

        if kwargs is not None:
            if "format_options" in kwargs:
                self.doc_converter = DocumentConverter(
                    format_options=kwargs["format_options"]
                )
            else:
                self.doc_converter = DocumentConverter()

    def predict(self, stream: DocumentStream, **extra_args) -> DoclingDocument:
        return self.doc_converter.convert(stream).document

    def info(self) -> Dict:
        return {"asset": "Docling", "version": docling_version()}


class AzureDocIntelligencePredictionProvider(BasePredictionProvider):
    def __init__(
        self, **kwargs
    ):  # could be the docling converter options, the remote credentials for MS/Google, etc.
        super().__init__(**kwargs)

        # TODO - Need a temp directory to save Azure outputs
        # Validate the required library
        try:
            from azure.ai.formrecognizer import DocumentAnalysisClient, AnalysisFeature
            from azure.core.credentials import AzureKeyCredential
        except ImportError:
            raise ImportError("azure-ai-formrecognizer library is not installed..")


        # Validate the required endpoints to call the API
        endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

        if not endpoint or not key:
            raise ValueError(
                "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY must be set in environment variables."
            )

        self.doc_intelligence_client = DocumentAnalysisClient(endpoint, AzureKeyCredential(key))

    def extract_bbox_from_polygon(self, polygon):
        """Helper function to extract bbox coordinates from polygon data."""
        if not polygon or not isinstance(polygon, list):
            return {"l": 0, "t": 0, "r": 0, "b": 0}

        # Handle flat array format: [x1, y1, x2, y2, x3, y3, x4, y4]
        if len(polygon) >= 8 and all(isinstance(p, (int, float)) for p in polygon):
            return {"l": polygon[0], "t": polygon[1], "r": polygon[4], "b": polygon[5]}
        # Handle array of point objects: [{x, y}, {x, y}, ...]
        elif len(polygon) >= 4 and all(
            isinstance(p, dict) and "x" in p and "y" in p for p in polygon
        ):
            return {
                "l": polygon[0]["x"],
                "t": polygon[0]["y"],
                "r": polygon[2]["x"],
                "b": polygon[2]["y"],
            }
        else:
            return {"l": 0, "t": 0, "r": 0, "b": 0}


    def convert_azure_output_to_docling(self, analyze_result, filename) -> DoclingDocument:
        """Converts Azure Document Intelligence output to DoclingDocument format."""
        doc = DoclingDocument(name=filename)
       
        for page in analyze_result.get("pages", []):
            page_no = page.get("page_number", 1)

            page_width = page.get("width")
            page_height = page.get("height")
            doc.pages[page_no] = PageItem(
                size=Size(width=float(page_width), height=float(page_height)),
                page_no=page_no,
            )

            for word in page.get("words", []):
                polygon = word.get("polygon", [])
                bbox = self.extract_bbox_from_polygon(polygon)

                text_content = word.get("content", "")

                bbox_obj = BoundingBox(
                    l=bbox["l"],
                    t=bbox["t"],
                    r=bbox["r"],
                    b=bbox["b"],
                    coord_origin=CoordOrigin.TOPLEFT,
                )

                prov = ProvenanceItem(
                    page_no=page_no, bbox=bbox_obj, charspan=(0, len(text_content))
                )

                doc.add_text(label=DocItemLabel.TEXT, text=text_content, prov=prov)

        for table in analyze_result.get("tables", []):
            page_no = table.get("page_range", {}).get("first_page_number", 1)
            row_count = table.get("row_count", 0)
            col_count = table.get("column_count", 0)

            table_bounds = (
                table.get("boundingRegions", [{}])[0]
                if table.get("boundingRegions")
                else {}
            )
            table_polygon = table_bounds.get("polygon", [])
            table_bbox = self.extract_bbox_from_polygon(table_polygon)

            table_bbox_obj = BoundingBox(
                l=table_bbox["l"],
                t=table_bbox["t"],
                r=table_bbox["r"],
                b=table_bbox["b"],
                coord_origin=CoordOrigin.TOPLEFT,
            )

            table_prov = ProvenanceItem(page_no=page_no, bbox=table_bbox_obj, charspan=(0, 0))

            table_data = TableData(
                table_cells=[], num_rows=row_count, num_cols=col_count, grid=[]
            )

            for cell in table.get("cells", []):

                cell_text = cell.get("content", "").strip()
                row_index = cell.get("row_index", 0)
                col_index = cell.get("column_index", 0)
                row_span = cell.get("row_span", 1)
                col_span = cell.get("column_span", 1)

                cell_bounds = (
                    cell.get("boundingRegions", [{}])[0]
                    if cell.get("boundingRegions")
                    else {}
                )
                cell_polygon = cell_bounds.get("polygon", [])
                cell_bbox = self.extract_bbox_from_polygon(cell_polygon)

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
                    start_col_offset_idx=col_index,
                    end_col_offset_idx=col_index + col_span - 1,
                    text=cell_text,
                    column_header=False,
                    row_header=False,
                    row_section=False,
                )

                table_data.table_cells.append(table_cell)

            doc.add_table(label=DocItemLabel.TABLE, prov=table_prov, data=table_data)

        return doc


    def predict(self, stream: DocumentStream, **extra_args) -> DoclingDocument:
        # For the given document stream (single document), run the API and create the doclingDocument

        # Get the image from the stream
        # TODO - Convert the given stream?
        print(f"\nstream - {type(stream)}")
        print(f"stream.model_dump() type - {type(stream.model_dump())}")
        print(f"stream.stream type - {type(stream.stream)}")
        poller = self.doc_intelligence_client.begin_analyze_document("prebuilt-layout", stream.stream, features=[])
        result = poller.result()
        result_json = result.to_dict()
        print("Successfully processed using Azure API..!!")
        # print(result_json)
        # poller = document_analysis_client.begin_analyze_document(
        #             model_name, document=f, features=features
        #         )
        #         result = poller.result()

        #     # Convert the result to a dictionary
        #     result_json = result.to_dict()
        return self.convert_azure_output_to_docling(result_json, stream.name)

        
    def info(self) -> Dict:
        return {"asset": "Azure AI Document Intelligence", "version": "1.0.0"}
