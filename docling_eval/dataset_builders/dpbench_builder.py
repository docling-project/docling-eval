import json
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List

from docling_core.types import DoclingDocument
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    ImageRef,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)
from docling_core.types.io import DocumentStream
from PIL.Image import Image
from tqdm import tqdm

from docling_eval.datamodels.dataset_record import DatasetRecord
from docling_eval.dataset_builders.dataset_builder import (
    BaseEvaluationDatasetBuilder,
    HFSource,
)
from docling_eval.utils.utils import (
    add_pages_to_true_doc,
    convert_html_table_into_docling_tabledata,
    crop_bounding_box,
    from_pil_to_base64uri,
    get_binary,
    get_binhash,
)

TRUE_HTML_EXPORT_LABELS = {
    DocItemLabel.TITLE,
    DocItemLabel.DOCUMENT_INDEX,
    DocItemLabel.SECTION_HEADER,
    DocItemLabel.PARAGRAPH,
    DocItemLabel.TABLE,
    DocItemLabel.PICTURE,
    DocItemLabel.FORMULA,
    DocItemLabel.CHECKBOX_UNSELECTED,
    DocItemLabel.CHECKBOX_SELECTED,
    DocItemLabel.TEXT,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.CODE,
    DocItemLabel.REFERENCE,
    # Additional
    DocItemLabel.CAPTION,
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.PAGE_FOOTER,
    DocItemLabel.FOOTNOTE,
}

PRED_HTML_EXPORT_LABELS = {
    DocItemLabel.TITLE,
    DocItemLabel.DOCUMENT_INDEX,
    DocItemLabel.SECTION_HEADER,
    DocItemLabel.PARAGRAPH,
    DocItemLabel.TABLE,
    DocItemLabel.PICTURE,
    DocItemLabel.FORMULA,
    DocItemLabel.CHECKBOX_UNSELECTED,
    DocItemLabel.CHECKBOX_SELECTED,
    DocItemLabel.TEXT,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.CODE,
    DocItemLabel.REFERENCE,
    # Additional
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.PAGE_FOOTER,
    DocItemLabel.FOOTNOTE,
}


class DPBenchDatasetBuilder(BaseEvaluationDatasetBuilder):
    def __init__(
        self,
        # prediction_provider: BasePredictionProvider,
        target: Path,
        split: str = "test",
    ):
        super().__init__(
            name="DPBench",
            dataset_source=HFSource(repo_id="upstage/dp-bench"),
            target=target,
            split=split,
        )

    def _update_gt_doc(
        self,
        doc: DoclingDocument,
        annots: Dict,
        page,
        page_image: Image,
        page_width: float,
        page_height: float,
    ):

        label = annots["category"]

        min_x = annots["coordinates"][0]["x"]
        max_x = annots["coordinates"][0]["x"]

        min_y = annots["coordinates"][0]["y"]
        max_y = annots["coordinates"][0]["y"]

        for coor in annots["coordinates"]:
            min_x = min(min_x, coor["x"])
            max_x = max(max_x, coor["x"])

            min_y = min(min_y, coor["y"])
            max_y = max(max_y, coor["y"])

        text = annots["content"]["text"].replace("\n", " ")
        html = annots["content"]["html"]

        bbox = BoundingBox(
            l=min_x * page_width,
            r=max_x * page_width,
            t=min_y * page_height,
            b=max_y * page_height,
            coord_origin=CoordOrigin.TOPLEFT,
        )

        prov = ProvenanceItem(page_no=1, bbox=bbox, charspan=(0, len(text)))

        img = crop_bounding_box(page_image=page_image, page=page, bbox=bbox)

        if label == "Header":
            doc.add_text(
                label=DocItemLabel.PAGE_HEADER, text=text, orig=text, prov=prov
            )

        elif label == "Footer":
            doc.add_text(
                label=DocItemLabel.PAGE_FOOTER, text=text, orig=text, prov=prov
            )

        elif label == "Paragraph":
            doc.add_text(label=DocItemLabel.TEXT, text=text, orig=text, prov=prov)

        elif label == "Index":

            # FIXME: ultra approximate solution
            text = annots["content"]["text"]
            rows = text.split("\n")

            num_rows = len(rows)
            num_cols = 2

            row_span = 1
            col_span = 1

            cells = []
            for row_idx, row in enumerate(rows):
                parts = row.split(" ")

                col_idx = 0
                cell = TableCell(
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=row_idx,
                    end_row_offset_idx=row_idx + row_span,
                    start_col_offset_idx=col_idx,
                    end_col_offset_idx=col_idx + col_span,
                    text=" ".join(parts[:-1]),
                )
                cells.append(cell)

                col_idx = 1
                cell = TableCell(
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=row_idx,
                    end_row_offset_idx=row_idx + row_span,
                    start_col_offset_idx=col_idx,
                    end_col_offset_idx=col_idx + col_span,
                    text=parts[-1],
                )
                cells.append(cell)

            table_data = TableData(
                num_rows=num_rows, num_cols=num_cols, table_cells=cells
            )
            doc.add_table(
                data=table_data,
                caption=None,
                prov=prov,
                label=DocItemLabel.DOCUMENT_INDEX,
            )

        elif label == "List":
            doc.add_list_item(text=text, orig=text, prov=prov)
            # doc.add_text(label=DocItemLabel.TEXT, text=text, orig=text, prov=prov)

        elif label == "Caption":
            doc.add_text(label=DocItemLabel.CAPTION, text=text, orig=text, prov=prov)

        elif label == "Equation":
            doc.add_text(label=DocItemLabel.FORMULA, text=text, orig=text, prov=prov)

        elif label == "Figure":
            uri = from_pil_to_base64uri(img)

            imgref = ImageRef(
                mimetype="image/png",
                dpi=72,
                size=Size(width=img.width, height=img.height),
                uri=uri,
            )

            doc.add_picture(prov=prov, image=imgref)

        elif label == "Table":

            table_data = convert_html_table_into_docling_tabledata(table_html=html)

            doc.add_table(data=table_data, caption=None, prov=prov)

        elif label == "Chart":
            uri = from_pil_to_base64uri(img)

            imgref = ImageRef(
                mimetype="image/png",
                dpi=72,
                size=Size(width=img.width, height=img.height),
                uri=uri,
            )

            doc.add_picture(prov=prov, image=imgref)

            # doc.add_picture(prov=prov)

        elif label == "Footnote":
            doc.add_text(label=DocItemLabel.FOOTNOTE, text=text, orig=text, prov=prov)

        elif label == "Heading1":
            doc.add_heading(text=text, orig=text, level=1, prov=prov)

        else:
            return

    def iterate(self) -> Iterable[DatasetRecord]:
        if not self.retrieved:
            raise RuntimeError(
                "You must first retrieve the source dataset. Call retrieve_input_dataset()."
            )

        assert self.dataset_local_path is not None

        # load the groundtruth
        with open(self.dataset_local_path / f"dataset/reference.json", "r") as fr:
            gt = json.load(fr)

        cnt = 0
        for filename, annots in tqdm(
            gt.items(),
            desc="Processing files for DP-Bench with end-to-end",
            total=len(gt),
            ncols=128,
        ):
            cnt += 1

            # if cnt == 10:
            #    break

            pdf_path = self.dataset_local_path / f"dataset/pdfs/{filename}"

            # Create the groundtruth Document
            true_doc = DoclingDocument(
                name=f"ground-truth {os.path.basename(pdf_path)}"
            )
            true_doc, true_page_images = add_pages_to_true_doc(
                pdf_path=pdf_path, true_doc=true_doc, image_scale=2.0
            )
            assert len(true_page_images) == 1, "len(true_page_images)==1"

            page_width = true_doc.pages[1].size.width
            page_height = true_doc.pages[1].size.height

            for elem in annots["elements"]:
                self._update_gt_doc(
                    true_doc,
                    elem,
                    page=true_doc.pages[1],
                    page_image=true_page_images[0],
                    page_width=page_width,
                    page_height=page_height,
                )

            pdf_bytes = get_binary(pdf_path)

            pdf_stream = DocumentStream(name=pdf_path.name, stream=BytesIO(pdf_bytes))

            record = DatasetRecord(
                doc_id=str(filename),
                doc_hash=get_binhash(pdf_bytes),
                ground_truth_doc=true_doc,
                original=pdf_stream,
                mime_type="application/pdf",
            )

            yield record
