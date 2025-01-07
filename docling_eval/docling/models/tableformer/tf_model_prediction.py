import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from docling.datamodel.base_models import Cluster, LayoutPrediction, Page, Table
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.models.table_structure_model import TableStructureModel
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling_core.types.doc import DocItemLabel
from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.document import (
    DoclingDocument,
    TableCell,
    TableData,
    TableItem,
)
from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor
from docling_parse.pdf_parsers import pdf_parser_v2
from huggingface_hub import snapshot_download

# import cv2
from PIL import Image
from pydantic import BaseModel

from docling_eval.benchmarks.utils import get_input_document
from docling_eval.docling.models.tableformer.tf_constants import tf_config
from docling_eval.docling.utils import crop_bounding_box, map_to_records

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class PageToken(BaseModel):
    bbox: BoundingBox

    text: str
    id: int


class PageTokens(BaseModel):
    tokens: List[PageToken]

    height: float
    width: float


def get_iocr_page(parsed_page: Dict, table_bbox: Tuple[float, float, float, float]):

    height = parsed_page["sanitized"]["dimension"]["height"]
    width = parsed_page["sanitized"]["dimension"]["width"]

    records = map_to_records(parsed_page["sanitized"]["cells"])

    cnt = 0

    tokens = []
    text_lines = []
    for i, rec in enumerate(records):
        tokens.append(
            {
                "bbox": {
                    "l": rec["x0"],
                    "t": height - rec["y1"],
                    "r": rec["x1"],
                    "b": height - rec["y0"],
                },
                "text": rec["text"],
                "id": i,
            }
        )

        text_lines.append(
            {
                "bbox": [rec["x0"], height - rec["y1"], rec["x1"], height - rec["y0"]],
                "text": rec["text"],
            }
        )

        """
        if table_bbox[0]<=tokens[-1]["bbox"]["l"] and \
           table_bbox[2]>=tokens[-1]["bbox"]["r"] and \
           table_bbox[1]<=tokens[-1]["bbox"]["b"] and \
           table_bbox[3]>=tokens[-1]["bbox"]["t"]:
            cnt += 1
            print(f"text-cell [{cnt}]: ", tokens[-1]["text"], "\t", tokens[-1]["bbox"])
        """

    iocr_page = {"tokens": tokens, "height": height, "width": width}

    return iocr_page


def to_np(pil_image: Image.Image):
    # Convert to NumPy array
    np_image = np.array(pil_image)

    # Handle different formats
    if np_image.ndim == 3:  # RGB or RGBA image
        if np_image.shape[2] == 4:  # RGBA image
            # Discard alpha channel and convert to BGR
            np_image = np_image[:, :, :3]  # Keep only RGB channels

        # Convert RGB to BGR by reversing the last axis
        np_image = np_image[:, :, ::-1]

        return np_image
    else:
        raise ValueError("Unsupported image format")


def tf_predict(
    config,
    page_image: Image.Image,
    parsed_page: dict,
    table_bbox: Tuple[float, float, float, float],
    viz: bool = True,
    device: str = "cpu",
    num_threads: int = 2,
    image_scale: float = 1.0,
):
    r"""
    Test the TFPredictor
    """

    table_bboxes = [[table_bbox[0], table_bbox[1], table_bbox[2], table_bbox[3]]]

    iocr_page = get_iocr_page(parsed_page, table_bbox=table_bbox)

    iocr_page["image"] = to_np(page_image)
    iocr_page["table_bboxes"] = table_bboxes

    # Loop over the iocr_pages
    predictor = TFPredictor(config, device=device, num_threads=num_threads)

    tf_output = predictor.multi_table_predict(
        iocr_page,
        table_bboxes=table_bboxes,
        do_matching=True,
        correct_overlapping_cells=False,
        sort_row_col_indexes=True,
    )
    # print("tf-output: ", json.dumps(tf_output, indent=2))

    table_out = tf_output[0]

    do_cell_matching = True

    table_cells = []
    for element in table_out["tf_responses"]:

        tc = TableCell.model_validate(element)
        if do_cell_matching and tc.bbox is not None:
            tc.bbox = tc.bbox.scaled(1 / image_scale)
        table_cells.append(tc)

    # Retrieving cols/rows, after post processing:
    num_rows = table_out["predict_details"]["num_rows"]
    num_cols = table_out["predict_details"]["num_cols"]
    otsl_seq = table_out["predict_details"]["prediction"]["rs_seq"]

    table_data = TableData(
        num_rows=num_rows, num_cols=num_cols, table_cells=table_cells
    )

    return table_data

# TODO: This method must be dropped.
def tf_predict_with_page_tokens(
    config,
    page_image: Image.Image,
    page_tokens: PageTokens,
    table_bbox: Tuple[float, float, float, float],
    viz: bool = True,
    device: str = "cpu",
    num_threads: int = 2,
    image_scale: float = 1.0,
):
    r"""
    Test the TFPredictor
    """

    table_bboxes = [[table_bbox[0], table_bbox[1], table_bbox[2], table_bbox[3]]]

    ocr_page = page_tokens.dict()

    ocr_page["image"] = to_np(page_image)
    ocr_page["table_bboxes"] = table_bboxes

    # Loop over the iocr_pages
    predictor = TFPredictor(config, device=device, num_threads=num_threads)

    tf_output = predictor.multi_table_predict(
        ocr_page,
        table_bboxes=table_bboxes,
        do_matching=True,
        correct_overlapping_cells=False,
        sort_row_col_indexes=True,
    )
    # print("tf-output: ", json.dumps(tf_output, indent=2))

    table_out = tf_output[0]

    do_cell_matching = True

    table_cells = []
    for element in table_out["tf_responses"]:

        tc = TableCell.model_validate(element)
        if do_cell_matching and tc.bbox is not None:
            tc.bbox = tc.bbox.scaled(1 / image_scale)
        table_cells.append(tc)

    # Retrieving cols/rows, after post processing:
    num_rows = table_out["predict_details"]["num_rows"]
    num_cols = table_out["predict_details"]["num_cols"]
    otsl_seq = table_out["predict_details"]["prediction"]["rs_seq"]

    table_data = TableData(
        num_rows=num_rows, num_cols=num_cols, table_cells=table_cells
    )

    return table_data


# TODO remove this method once `replace_tabledata_with_page_tokens` does no longer need it.
def init_tf_model() -> dict:
    r"""
    Initialize the testing environment
    """
    config: Dict[str, Any] = tf_config

    # Download models from HF
    download_path = snapshot_download(repo_id="ds4sd/docling-models", revision="v2.1.0")
    save_dir = os.path.join(download_path, "model_artifacts/tableformer/fast")

    config["model"]["save_dir"] = save_dir
    return config


class TableFormerUpdater:

    def __init__(self):
        # Init the TableFormer model
        # Download models from HF
        download_path = StandardPdfPipeline.download_models_hf()
        pdf_pipeline_opts = PdfPipelineOptions()
        self.docling_tf_model = TableStructureModel(
            enabled=True,
            artifacts_path=download_path / StandardPdfPipeline._table_model_path,
            options=pdf_pipeline_opts.table_structure_options,
            accelerator_options=pdf_pipeline_opts.accelerator_options,
        )

        # TODO make this obsolete, only needed for `replace_tabledata_with_page_tokens`
        self.tf_config = init_tf_model()

    def get_page_cells(self, filename: str):

        parser = pdf_parser_v2("fatal")

        try:
            key = "key"
            parser.load_document(key=key, filename=filename)

            parsed_doc = parser.parse_pdf_from_key(key=key)

            parser.unload_document(key)
            return parsed_doc

        except Exception as exc:
            logging.error(exc)

        return None

    def _make_internal_page(self, input_doc, prov):
        page = Page(page_no=prov.page_no - 1)
        page._backend = input_doc._backend.load_page(page.page_no)
        page.cells = list(page._backend.get_text_cells())
        page.size = page._backend.get_size()

        if page._backend is not None and page._backend.is_valid():
            cluster = Cluster(
                id=0,
                label=DocItemLabel.TABLE,
                bbox=prov.bbox.to_top_left_origin(page.size.height),
            )
            for cell in page.cells:
                overlap = cell.bbox.intersection_area_with(cluster.bbox)
                overlap_ratio = overlap / cell.bbox.area()
                if overlap_ratio > 0.2:
                    cluster.cells.append(cell)

            page.predictions.layout = LayoutPrediction(clusters=[cluster])

        return page

    def replace_tabledata(
        self,
        pdf_path: Path,
        true_doc: DoclingDocument,
        # true_page_images: List[Image.Image],
    ) -> Tuple[bool, DoclingDocument]:

        updated = False

        # deep copy of the true-document
        pred_doc = copy.deepcopy(true_doc)

        input_doc = get_input_document(pdf_path)
        if not input_doc.valid:
            logging.error("could not parse pdf-file")
            return False, pred_doc

        conv_res = ConversionResult(input=input_doc)

        # parsed_doc = self.get_page_cells(str(pdf_path))
        # if parsed_doc is None:
        #    logging.error("could not parse pdf-file")
        #    return False, pred_doc

        # Replace the groundtruth tables with predictions from TableFormer
        for item, level in pred_doc.iterate_items():
            if isinstance(item, TableItem):
                for prov in item.prov:
                    page = self._make_internal_page(input_doc, prov)

                    page = next(self.docling_tf_model(conv_res, [page]))
                    tbl: Table = page.predictions.tablestructure.table_map[0]
                    table_data: TableData = TableData(
                        num_rows=tbl.num_rows,
                        num_cols=tbl.num_cols,
                        table_cells=tbl.table_cells,
                    )

                    item.data = table_data
                    page._backend.unload()

                    updated = True

                    # md = item.export_to_markdown()
                    # print("prediction from table-former: \n\n", md)

        return updated, pred_doc

    # TODO: This method must be re-written to use the TableStructureModel instance instead. See above.
    def replace_tabledata_with_page_tokens(
        self,
        page_tokens: PageTokens,
        true_doc: DoclingDocument,
        true_page_images: List[Image.Image],
    ) -> Tuple[bool, DoclingDocument]:

        updated = False

        # deep copy of the true-document
        pred_doc = copy.deepcopy(true_doc)

        # Replace the groundtruth tables with predictions from TableFormer
        for item, level in pred_doc.iterate_items():
            if isinstance(item, TableItem):
                for prov in item.prov:

                    # md = item.export_to_markdown()
                    # print("groundtruth: \n\n", md)

                    page_image = true_page_images[prov.page_no - 1]
                    # page_image.show()

                    table_data = tf_predict_with_page_tokens(
                        config=self.tf_config,
                        page_image=page_image,
                        page_tokens=page_tokens,
                        table_bbox=(prov.bbox.l, prov.bbox.b, prov.bbox.r, prov.bbox.t),
                    )
                    item.data = table_data

                    updated = True

                    # md = item.export_to_markdown()
                    # print("prediction from table-former: \n\n", md)

        return updated, pred_doc
