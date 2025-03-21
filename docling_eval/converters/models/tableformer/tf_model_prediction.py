import copy
import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from docling.datamodel.base_models import Cluster, LayoutPrediction, Page, Table
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.models.table_structure_model import TableStructureModel
from docling_core.types.doc import DocItemLabel
from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.document import (
    DoclingDocument,
    TableCell,
    TableData,
    TableItem,
)
from docling_parse.pdf_parsers import pdf_parser_v2
from PIL import Image
from pydantic import BaseModel

from docling_eval.benchmarks.utils import get_input_document, map_to_records

# Get logger
log = logging.getLogger(__name__)


class PageToken(BaseModel):
    bbox: BoundingBox

    text: str
    id: int


class PageTokens(BaseModel):
    tokens: List[PageToken]

    height: float
    width: float


class TableFormerUpdater:

    def __init__(
        self,
        mode: TableFormerMode,
        num_threads: int = 16,
        artifacts_path: Optional[Path] = None,
    ):
        r""" """
        # Init the TableFormer model
        table_structure_options = TableStructureOptions(mode=mode)
        accelerator_options = AcceleratorOptions(
            num_threads=num_threads, device=AcceleratorDevice.AUTO
        )
        self._docling_tf_model = TableStructureModel(
            enabled=True,
            artifacts_path=artifacts_path,
            options=table_structure_options,
            accelerator_options=accelerator_options,
        )
        log.info("Initialize %s", mode)

    def to_np(self, pil_image: Image.Image):
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

    def get_page_cells(self, filename: str):

        parser = pdf_parser_v2("fatal")

        try:
            key = "key"
            parser.load_document(key=key, filename=filename)

            parsed_doc = parser.parse_pdf_from_key(key=key)

            parser.unload_document(key)
            return parsed_doc

        except Exception as exc:
            log.error(exc)

        return None

    def _make_internal_page_with_table(self, input_doc, prov):
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
                if cell.bbox.area() > 0:
                    overlap_ratio = overlap / cell.bbox.area()
                    if overlap_ratio > 0.2:
                        cluster.cells.append(cell)

            page.predictions.layout = LayoutPrediction(clusters=[cluster])

        return page

    def replace_tabledata(
        self,
        pdf_path: Path | BytesIO,
        true_doc: DoclingDocument,
    ) -> Tuple[bool, DoclingDocument]:

        updated = False

        # deep copy of the true-document
        pred_doc = true_doc.model_copy(deep=True)

        input_doc = get_input_document(pdf_path)
        if not input_doc.valid:
            log.error("could not parse pdf-file")
            return False, pred_doc

        conv_res = ConversionResult(input=input_doc)

        # parsed_doc = self.get_page_cells(str(pdf_path))
        # if parsed_doc is None:
        #    log.error("could not parse pdf-file")
        #    return False, pred_doc

        # Replace the groundtruth tables with predictions from TableFormer
        for item, level in pred_doc.iterate_items():
            if isinstance(item, TableItem):
                for prov in item.prov:
                    page = self._make_internal_page_with_table(input_doc, prov)

                    page = next(self._docling_tf_model(conv_res, [page]))  # type: ignore
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

    def _tf_predict_with_page_tokens(
        self,
        page_image: Image.Image,
        page_tokens: PageTokens,
        table_bbox: Tuple[float, float, float, float],
        image_scale: float = 1.0,
    ):
        r"""
        Test the TFPredictor
        """
        table_bboxes = [[table_bbox[0], table_bbox[1], table_bbox[2], table_bbox[3]]]

        ocr_page = page_tokens.dict()

        ocr_page["image"] = self.to_np(page_image)
        ocr_page["table_bboxes"] = table_bboxes

        # TODO: Here we bypass docling API and we steal the tf_preditor private object :-(
        predictor = self._docling_tf_model.tf_predictor

        # Loop over the iocr_pages
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

        table_data = TableData(
            num_rows=num_rows, num_cols=num_cols, table_cells=table_cells
        )

        return table_data

    def replace_tabledata_with_page_tokens(
        self,
        page_tokens: PageTokens,
        true_doc: DoclingDocument,
        true_page_images: List[Image.Image],
    ) -> Tuple[bool, DoclingDocument]:

        updated = False

        # deep copy of the true-document
        pred_doc = copy.deepcopy(true_doc)

        assert len(pred_doc.pages) == 1
        page_size = pred_doc.pages[1].size

        # Replace the groundtruth tables with predictions from TableFormer
        for item, level in pred_doc.iterate_items():
            if isinstance(item, TableItem):
                for prov in item.prov:

                    # md = item.export_to_markdown()
                    # print("groundtruth: \n\n", md)

                    page_image = true_page_images[prov.page_no - 1]
                    # page_image.show()

                    # Ensure that the bbox will be inside the min/max ranges
                    table_bbox = (
                        max(prov.bbox.l, 0.0),
                        max(prov.bbox.b, 0.0),
                        min(prov.bbox.r, page_size.width),
                        min(prov.bbox.t, page_size.height),
                    )

                    table_data = self._tf_predict_with_page_tokens(
                        page_image=page_image,
                        page_tokens=page_tokens,
                        table_bbox=table_bbox,
                    )
                    item.data = table_data

                    updated = True

                    # md = item.export_to_markdown()
                    # print("prediction from table-former: \n\n", md)

        return updated, pred_doc
