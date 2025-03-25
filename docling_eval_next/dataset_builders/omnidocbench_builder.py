import glob
import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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

from docling_eval.benchmarks.utils import (
    add_pages_to_true_doc,
    convert_html_table_into_docling_tabledata,
    crop_bounding_box,
    from_pil_to_base64uri,
    get_binary,
    get_binhash,
    set_selection_range,
)
from docling_eval.converters.models.tableformer.tf_model_prediction import (
    TableFormerUpdater,
)
from docling_eval.visualisation.visualisations import save_comparison_html_with_clusters
from docling_eval_next.datamodels.dataset_record import DatasetRecord
from docling_eval_next.dataset_builders.dataset_builder import (
    BaseEvaluationDatasetBuilder,
    HFSource,
)
from docling_eval_next.prediction_providers.prediction_provider import (
    BasePredictionProvider,
)

# Get logger
_log = logging.getLogger(__name__)

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


class OmniDocBenchDatasetBuilder(BaseEvaluationDatasetBuilder):
    def __init__(
        self,
        # prediction_provider: BasePredictionProvider,
        target: Path,
        do_visualization: bool = True,
    ):
        super().__init__(
            name="OmniDocBench: end-to-end",
            dataset_source=HFSource(repo_id="opendatalab/OmniDocBench"),
            # prediction_provider=prediction_provider,
            target=target,
        )
        self.do_visualization = do_visualization

    def update_gt_into_map(self, gt):

        result = {}

        for item in gt:
            path = item["page_info"]["image_path"]
            result[path] = item

        return result

    def get_filenames(self, omnidocbench_dir: Path) -> List[Tuple[str, str]]:

        page_images = sorted(glob.glob(str(omnidocbench_dir / "images/*.jpg")))
        page_pdfs = sorted(glob.glob(str(omnidocbench_dir / "ori_pdfs/*.pdf")))

        assert len(page_images) == len(
            page_pdfs
        ), f"len(page_images)!=len(page_pdfs) => {len(page_images)}!={len(page_pdfs)}"

        return list(zip(page_images, page_pdfs))

    def update_doc_with_gt(
        self,
        gt,
        true_doc,
        page,
        page_image: Image,
        page_width: float,
        page_height: float,
    ):

        gt_width = float(gt["page_info"]["width"])
        gt_height = float(gt["page_info"]["height"])

        for item in gt["layout_dets"]:

            label = item["category_type"]

            text = f"&lt;omitted text for {label}&gt;"
            if "text" in item:
                text = item["text"]

            min_x = item["poly"][0]
            max_x = item["poly"][0]

            min_y = item["poly"][1]
            max_y = item["poly"][1]

            for i in range(0, 4):
                min_x = min(min_x, item["poly"][2 * i])
                max_x = max(max_x, item["poly"][2 * i])

                min_y = min(min_y, item["poly"][2 * i + 1])
                max_y = max(max_y, item["poly"][2 * i + 1])

            bbox = BoundingBox(
                l=min_x * page_width / gt_width,
                r=max_x * page_width / gt_width,
                t=min_y * page_height / gt_height,
                b=max_y * page_height / gt_height,
                coord_origin=CoordOrigin.TOPLEFT,
            )

            prov = ProvenanceItem(page_no=1, bbox=bbox, charspan=(0, len(text)))

            img = crop_bounding_box(page_image=page_image, page=page, bbox=bbox)
            # img.show()

            if label == "title":
                true_doc.add_heading(text=text, orig=text, level=1, prov=prov)

            elif label == "text_block":
                true_doc.add_text(
                    label=DocItemLabel.TEXT, text=text, orig=text, prov=prov
                )

            elif label == "text_mask":
                true_doc.add_text(
                    label=DocItemLabel.TEXT, text=text, orig=text, prov=prov
                )

            elif label == "table":

                table_data = convert_html_table_into_docling_tabledata(
                    table_html=item["html"]
                )
                true_doc.add_table(data=table_data, caption=None, prov=prov)

            elif label == "table_caption":
                true_doc.add_text(
                    label=DocItemLabel.CAPTION, text=text, orig=text, prov=prov
                )

            elif label == "table_footnote":
                true_doc.add_text(
                    label=DocItemLabel.FOOTNOTE, text=text, orig=text, prov=prov
                )

            elif label == "table_mask":
                true_doc.add_text(
                    label=DocItemLabel.TEXT, text=text, orig=text, prov=prov
                )

            elif label == "figure":

                uri = from_pil_to_base64uri(img)

                imgref = ImageRef(
                    mimetype="image/png",
                    dpi=72,
                    size=Size(width=img.width, height=img.height),
                    uri=uri,
                )

                true_doc.add_picture(prov=prov, image=imgref)

            elif label == "figure_caption":
                true_doc.add_text(
                    label=DocItemLabel.CAPTION, text=text, orig=text, prov=prov
                )

            elif label == "figure_footnote":
                true_doc.add_text(
                    label=DocItemLabel.FOOTNOTE, text=text, orig=text, prov=prov
                )

            elif label == "equation_isolated":
                true_doc.add_text(
                    label=DocItemLabel.FORMULA, text=text, orig=text, prov=prov
                )

            elif label == "equation_caption":
                true_doc.add_text(
                    label=DocItemLabel.CAPTION, text=text, orig=text, prov=prov
                )

            elif label == "code_txt":
                true_doc.add_text(
                    label=DocItemLabel.TEXT, text=text, orig=text, prov=prov
                )

            elif label == "abandon":
                true_doc.add_text(
                    label=DocItemLabel.TEXT, text=text, orig=text, prov=prov
                )

            elif label == "need_mask":
                true_doc.add_text(
                    label=DocItemLabel.TEXT, text=text, orig=text, prov=prov
                )

            elif label == "header":
                true_doc.add_text(
                    label=DocItemLabel.PAGE_HEADER, text=text, orig=text, prov=prov
                )

            elif label == "footer":
                true_doc.add_text(
                    label=DocItemLabel.PAGE_FOOTER, text=text, orig=text, prov=prov
                )

            elif label == "reference":
                true_doc.add_text(
                    label=DocItemLabel.TEXT, text=text, orig=text, prov=prov
                )

            elif label == "page_footnote":
                true_doc.add_text(
                    label=DocItemLabel.FOOTNOTE, text=text, orig=text, prov=prov
                )

            elif label == "page_number":
                true_doc.add_text(
                    label=DocItemLabel.PAGE_FOOTER, text=text, orig=text, prov=prov
                )

            else:
                logging.error(f"label {label} is not assigned!")

        return true_doc

    def iterate(self) -> Iterable[DatasetRecord]:
        if not self.retrieved:
            raise RuntimeError(
                "You must first retrieve the source dataset. Call retrieve_input_dataset()."
            )

        assert self.dataset_local_path is not None
        # load the groundtruth
        with open(self.dataset_local_path / "OmniDocBench.json", "r") as fr:
            gt = json.load(fr)

        gt = self.update_gt_into_map(gt)

        viz_dir = self.target / "vizualisations"
        os.makedirs(viz_dir, exist_ok=True)

        page_tuples = self.get_filenames(self.dataset_local_path)

        # Apply index ranges
        total_ds_len = len(page_tuples)

        begin_index = 0
        end_index = -1
        # begin_index, end_index = set_selection_range(
        #    begin_index, end_index, total_ds_len
        # )
        page_tuples = page_tuples[begin_index:end_index]
        selected_ds_len = len(page_tuples)
        _log.info(
            "Dataset len: %s. Selected range: [%s, %s] = %s",
            total_ds_len,
            begin_index,
            end_index,
            selected_ds_len,
        )

        for page_tuple in tqdm(
            page_tuples,
            total=selected_ds_len,
            ncols=128,
            desc="Processing files for OmniDocBench with end-to-end",
        ):

            jpg_path = page_tuple[0]
            pdf_path = Path(page_tuple[1])

            # logging.info(f"file: {pdf_path}")
            if os.path.basename(jpg_path) not in gt:
                logging.error(
                    f"did not find ground-truth for {os.path.basename(jpg_path)}"
                )
                continue

            gt_doc = gt[os.path.basename(jpg_path)]

            # Create the groundtruth Document
            true_doc = DoclingDocument(
                name=f"ground-truth {os.path.basename(jpg_path)}"
            )
            true_doc, true_page_images = add_pages_to_true_doc(
                pdf_path=pdf_path, true_doc=true_doc, image_scale=2.0
            )

            assert len(true_page_images) == 1, "len(true_page_images)==1"

            # The true_doc.pages is a dict with the page numbers as indices starting at 1
            page_width = true_doc.pages[1].size.width
            page_height = true_doc.pages[1].size.height

            true_doc = self.update_doc_with_gt(
                gt=gt_doc,
                true_doc=true_doc,
                page=true_doc.pages[1],
                page_image=true_page_images[0],
                page_width=page_width,
                page_height=page_height,
            )

            pdf_bytes = get_binary(pdf_path)

            pdf_stream = DocumentStream(name=pdf_path.name, stream=BytesIO(pdf_bytes))

            record = DatasetRecord(
                # predictor_info=self.prediction_provider.info(),
                doc_id=str(os.path.basename(jpg_path)),
                doc_hash=get_binhash(pdf_bytes),
                ground_truth_doc=true_doc,
                original=pdf_stream,
                mime_type="application/pdf",
            )

            # self.update_prediction(record)

            # if self.do_visualization and record.predicted_doc is not None:
            #     save_comparison_html_with_clusters(
            #         filename=viz_dir / f"{os.path.basename(pdf_path)}-clusters.html",
            #         true_doc=true_doc,
            #         pred_doc=record.predicted_doc,
            #         page_image=true_page_images[0],
            #         true_labels=TRUE_HTML_EXPORT_LABELS,
            #         pred_labels=PRED_HTML_EXPORT_LABELS,
            #     )

            yield record
