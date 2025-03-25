import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple

from docling_core.types import DoclingDocument
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import (
    DoclingDocument,
    PageItem,
    ProvenanceItem,
    TableCell,
    TableData,
)
from docling_core.types.io import DocumentStream

from docling_eval_next.datamodels.dataset_record import DatasetRecord
from docling_eval_next.prediction_providers.prediction_provider import (
    BasePredictionProvider,
)

# from docling_core.types.doc.labels import DocItemLabel

_log = logging.getLogger(__name__)


class AzureDocIntelligencePredictionProvider(BasePredictionProvider):
    """Provider that calls the Microsoft Azure Document Intelligence API for predicting the tables in document."""

    def __init__(
        self, **kwargs
    ):  # could be the docling converter options, the remote credentials for MS/Google, etc.
        super().__init__(**kwargs)

        # TODO - Need a temp directory to save Azure outputs
        # Validate the required library
        try:
            from azure.ai.formrecognizer import AnalysisFeature, DocumentAnalysisClient
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

        self.doc_intelligence_client = DocumentAnalysisClient(
            endpoint, AzureKeyCredential(key)
        )

        # Save the flags for intermediate results and processing
        self.skip_api_if_prediction_is_present = bool(
            kwargs.get("skip_api_if_prediction_is_present", False) is True
        )
        # self.predictions_dir = kwargs.get("predictions_dir")
        # os.makedirs(self.predictions_dir, exist_ok=True)

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

    def convert_azure_output_to_docling(
        self, analyze_result, filename
    ) -> DoclingDocument:
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

                # TODO this needs to be developed.
                # doc.add_text(label=DocItemLabel.TEXT, text=text_content, prov=prov)

        for table in analyze_result.get("tables", []):
            page_no = table.get("page_range", {}).get("first_page_number", 1)
            row_count = table.get("row_count", 0)
            col_count = table.get("column_count", 0)

            table_polygon = table.get("bounding_regions", [{}])[0].get("polygon", [])
            table_bbox = self.extract_bbox_from_polygon(table_polygon)

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

            table_cells = []

            for cell in table.get("cells", []):

                cell_text = cell.get("content", "").strip()
                row_index = cell.get("row_index", 0)
                col_index = cell.get("column_index", 0)
                row_span = cell.get("row_span", 1)
                col_span = cell.get("column_span", 1)

                cell_polygon = cell.get("bounding_regions", [{}])[0].get("polygon", [])
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
                    end_row_offset_idx=row_index + row_span,
                    start_col_offset_idx=col_index,
                    end_col_offset_idx=col_index + col_span,
                    text=cell_text,
                    column_header=False,
                    row_header=False,
                    row_section=False,
                )

                table_cells.append(table_cell)

            table_data = TableData(
                table_cells=table_cells, num_rows=row_count, num_cols=col_count
            )

            doc.add_table(prov=table_prov, data=table_data, caption=None)

        return doc

    def predict(
        self,
        record: DatasetRecord,
    ) -> Tuple[DoclingDocument, Optional[str]]:
        """For the given document stream (single document), run the API and create the doclingDocument."""

        # TODO: Remove this code. Caching prediction results should not be handled in a PredictionProvider.
        # _log.info(f"Creating prediction for file - {record.original.name}..")
        # stream_file_basename = Path(record.original.name).stem

        # prediction_file_name = os.path.join(self.predictions_dir, f"{stream_file_basename}.json")
        # _log.debug(f"{prediction_file_name=}")

        # prediction_file_exists = False
        # Check if the prediction exists, if so - reuse it

        # if (self.skip_api_if_prediction_is_present and os.path.exists(prediction_file_name)):
        #     prediction_file_exists = True
        #     print(f"Skipping Azure API call and re-using existing prediction from [{prediction_file_name}].")
        #     with open(prediction_file_name, "r", encoding="utf-8") as f:
        #         result_json = json.load(f)
        #     result_json
        # else:

        if record.original:  # there is a PDF in here.
            # Call the Azure API by passing in the image for prediction
            poller = self.doc_intelligence_client.begin_analyze_document(
                "prebuilt-layout", record.original.stream, features=[]
            )
            result = poller.result()
            result_json = result.to_dict()
            _log.info(
                f"Successfully processed [{record.original.name}] using Azure API..!!"
            )
        elif len(record.ground_truth_page_images) > 0:
            # Call the Azure API by passing in the image for prediction
            buf = BytesIO()

            # TODO do this in a loop for all page images in the doc, not just the first.
            record.ground_truth_page_images[0].save(buf, format="PNG")

            poller = self.doc_intelligence_client.begin_analyze_document(
                "prebuilt-layout", buf, features=[]
            )
            result = poller.result()
            result_json = result.to_dict()
            _log.info(
                f"Successfully processed [{record.original.name}] using Azure API..!!"
            )

        # Convert the prediction to doclingDocument
        pred_docling_doc = self.convert_azure_output_to_docling(
            result_json, record.doc_id
        )
        result_orig = json.dumps(result_json)

        # TODO: Remove this code.
        # save both the prediction json as well as converted docling_document into the subfolders underneath
        # if not prediction_file_exists:
        #     with open(prediction_file_name, 'w', encoding="utf-8") as f:
        #         json.dump(result_json, f, indent=2)
        #     print(f"Saved Prediction output to - {prediction_file_name}")
        #
        # # Directory for storing docling_document output
        # output_dir = os.path.join(self.predictions_dir, "docling_document")
        # os.makedirs(output_dir, exist_ok=True)
        # docling_document_file_name = os.path.join(output_dir, f"{record.original.name}.json")    # include full name
        # with open(docling_document_file_name, 'w', encoding="utf-8") as f:
        #     json.dump(pred_docling_doc.export_to_dict(), f, indent=2)
        # print(f"Saved Docling Document output of prediction to - {docling_document_file_name}")

        return pred_docling_doc, result_orig

    def info(self) -> Dict:
        return {"asset": "Azure AI Document Intelligence", "version": "1.0.0"}
