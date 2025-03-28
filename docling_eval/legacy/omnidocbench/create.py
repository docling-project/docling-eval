import glob
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

from docling.datamodel.pipeline_options import TableFormerMode
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import DoclingDocument, ImageRef, ProvenanceItem
from docling_core.types.doc.labels import DocItemLabel
from PIL import Image  # as PILImage
from tqdm import tqdm  # type: ignore

from docling_eval.datamodels.types import (
    BenchMarkColumns,
    ConverterTypes,
    EvaluationModality,
)
from docling_eval.legacy.converters.conversion import (
    create_pdf_docling_converter,
    create_smol_docling_converter,
)
from docling_eval.prediction_providers.tableformer_provider import TableFormerUpdater
from docling_eval.utils.utils import (
    add_pages_to_true_doc,
    convert_html_table_into_docling_tabledata,
    crop_bounding_box,
    docling_version,
    extract_images,
    from_pil_to_base64uri,
    get_binary,
    save_shard_to_disk,
    set_selection_range,
    write_datasets_info,
)
from docling_eval.visualisation.visualisations import (
    save_comparison_html,
    save_comparison_html_with_clusters,
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


def get_filenames(omnidocbench_dir: Path) -> List[Tuple[str, str]]:

    page_images = sorted(glob.glob(str(omnidocbench_dir / "images/*.jpg")))
    page_pdfs = sorted(glob.glob(str(omnidocbench_dir / "ori_pdfs/*.pdf")))

    assert len(page_images) == len(
        page_pdfs
    ), f"len(page_images)!=len(page_pdfs) => {len(page_images)}!={len(page_pdfs)}"

    return list(zip(page_images, page_pdfs))


def update_gt_into_map(gt):

    result = {}

    for item in gt:
        path = item["page_info"]["image_path"]
        result[path] = item

    return result


def update_doc_with_gt(
    gt, true_doc, page, page_image: Image.Image, page_width: float, page_height: float
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
            true_doc.add_text(label=DocItemLabel.TEXT, text=text, orig=text, prov=prov)

        elif label == "text_mask":
            true_doc.add_text(label=DocItemLabel.TEXT, text=text, orig=text, prov=prov)

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
            true_doc.add_text(label=DocItemLabel.TEXT, text=text, orig=text, prov=prov)

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
            true_doc.add_text(label=DocItemLabel.TEXT, text=text, orig=text, prov=prov)

        elif label == "abandon":
            true_doc.add_text(label=DocItemLabel.TEXT, text=text, orig=text, prov=prov)

        elif label == "need_mask":
            true_doc.add_text(label=DocItemLabel.TEXT, text=text, orig=text, prov=prov)

        elif label == "header":
            true_doc.add_text(
                label=DocItemLabel.PAGE_HEADER, text=text, orig=text, prov=prov
            )

        elif label == "footer":
            true_doc.add_text(
                label=DocItemLabel.PAGE_FOOTER, text=text, orig=text, prov=prov
            )

        elif label == "reference":
            true_doc.add_text(label=DocItemLabel.TEXT, text=text, orig=text, prov=prov)

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


def create_omnidocbench_e2e_dataset(
    omnidocbench_dir: Path,
    output_dir: Path,
    begin_index: int = 0,
    end_index: int = -1,  # If -1 take the whole split
    converter_type: ConverterTypes = ConverterTypes.DOCLING,
    image_scale: float = 1.0,
    do_viz: bool = False,
    artifacts_path: Optional[Path] = None,
):

    # Create Converter
    if converter_type == ConverterTypes.DOCLING:
        converter = create_pdf_docling_converter(page_image_scale=1.0)
    else:
        converter = create_smol_docling_converter()

    # load the groundtruth
    with open(omnidocbench_dir / "OmniDocBench.json", "r") as fr:
        gt = json.load(fr)

    gt = update_gt_into_map(gt)

    viz_dir = output_dir / "vizualisations"
    os.makedirs(viz_dir, exist_ok=True)

    records = []

    page_tuples = get_filenames(omnidocbench_dir)

    # Apply index ranges
    total_ds_len = len(page_tuples)
    begin_index, end_index = set_selection_range(begin_index, end_index, total_ds_len)
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
            logging.error(f"did not find ground-truth for {os.path.basename(jpg_path)}")
            continue

        gt_doc = gt[os.path.basename(jpg_path)]

        # Create the predicted Document
        conv_results = converter.convert(source=pdf_path, raises_on_error=True)
        pred_doc = conv_results.document

        # Create the groundtruth Document
        true_doc = DoclingDocument(name=f"ground-truth {os.path.basename(jpg_path)}")
        true_doc, true_page_images = add_pages_to_true_doc(
            pdf_path=pdf_path, true_doc=true_doc, image_scale=image_scale
        )

        assert len(true_page_images) == 1, "len(true_page_images)==1"

        # The true_doc.pages is a dict with the page numbers as indices starting at 1
        page_width = true_doc.pages[1].size.width
        page_height = true_doc.pages[1].size.height

        true_doc = update_doc_with_gt(
            gt=gt_doc,
            true_doc=true_doc,
            page=true_doc.pages[1],
            page_image=true_page_images[0],
            page_width=page_width,
            page_height=page_height,
        )

        if do_viz:
            save_comparison_html_with_clusters(
                filename=viz_dir / f"{os.path.basename(pdf_path)}-clusters.html",
                true_doc=true_doc,
                pred_doc=pred_doc,
                page_image=true_page_images[0],
                true_labels=TRUE_HTML_EXPORT_LABELS,
                pred_labels=PRED_HTML_EXPORT_LABELS,
            )

        true_doc, true_pictures, true_page_images = extract_images(
            true_doc,  # conv_results.document,
            pictures_column=BenchMarkColumns.GROUNDTRUTH_PICTURES,  # pictures_column,
            page_images_column=BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES,  # page_images_column,
        )

        pred_doc, pred_pictures, pred_page_images = extract_images(
            pred_doc,  # conv_results.document,
            pictures_column=BenchMarkColumns.PREDICTION_PICTURES,  # pictures_column,
            page_images_column=BenchMarkColumns.PREDICTION_PAGE_IMAGES,  # page_images_column,
        )

        record = {
            BenchMarkColumns.CONVERTER_TYPE: converter_type,
            BenchMarkColumns.CONVERTER_VERSION: docling_version(),
            BenchMarkColumns.STATUS: "SUCCESS",
            BenchMarkColumns.DOC_ID: str(os.path.basename(jpg_path)),
            BenchMarkColumns.GROUNDTRUTH: json.dumps(true_doc.export_to_dict()),
            BenchMarkColumns.PREDICTION: json.dumps(pred_doc.export_to_dict()),
            BenchMarkColumns.ORIGINAL: get_binary(pdf_path),
            BenchMarkColumns.MIMETYPE: "application/pdf",
            # BenchMarkColumns.PAGE_IMAGES: pred_page_images,
            BenchMarkColumns.PREDICTION_PAGE_IMAGES: pred_page_images,
            BenchMarkColumns.PREDICTION_PICTURES: pred_pictures,
            BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES: true_page_images,
            BenchMarkColumns.GROUNDTRUTH_PICTURES: true_pictures,
            BenchMarkColumns.MODALITIES: [
                EvaluationModality.LAYOUT,
                EvaluationModality.MARKDOWN_TEXT,
                EvaluationModality.READING_ORDER,
                EvaluationModality.TABLE_STRUCTURE,
            ],
        }
        records.append(record)

    test_dir = output_dir / "test"
    os.makedirs(test_dir, exist_ok=True)

    save_shard_to_disk(items=records, dataset_path=test_dir)

    write_datasets_info(
        name="OmniDocBench: end-to-end",
        output_dir=output_dir,
        num_train_rows=0,
        num_test_rows=len(records),
    )


def create_omnidocbench_tableformer_dataset(
    omnidocbench_dir: Path,
    output_dir: Path,
    begin_index: int = 0,
    end_index: int = -1,  # If -1 take the whole split
    image_scale: float = 1.0,
    mode: TableFormerMode = TableFormerMode.ACCURATE,
    artifacts_path: Optional[Path] = None,
):
    # Init the TableFormer model
    tf_updater = TableFormerUpdater(mode, artifacts_path=artifacts_path)

    # load the groundtruth
    with open(omnidocbench_dir / "OmniDocBench.json", "r") as fr:
        gt = json.load(fr)

    gt = update_gt_into_map(gt)

    viz_dir = output_dir / "vizualisations"
    os.makedirs(viz_dir, exist_ok=True)

    records = []

    page_tuples = get_filenames(omnidocbench_dir)

    # Apply index ranges
    total_ds_len = len(page_tuples)
    begin_index, end_index = set_selection_range(begin_index, end_index, total_ds_len)
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
            logging.error(f"did not find ground-truth for {os.path.basename(jpg_path)}")
            continue

        gt_doc = gt[os.path.basename(jpg_path)]

        # Create the groundtruth Document
        true_doc = DoclingDocument(name=f"ground-truth {os.path.basename(jpg_path)}")
        true_doc, true_page_images = add_pages_to_true_doc(
            pdf_path=pdf_path, true_doc=true_doc, image_scale=image_scale
        )

        assert len(true_page_images) == 1, "len(true_page_images)==1"

        page_width = true_doc.pages[1].size.width
        page_height = true_doc.pages[1].size.height

        true_doc = update_doc_with_gt(
            gt=gt_doc,
            true_doc=true_doc,
            page=true_doc.pages[1],
            page_image=true_page_images[0],
            page_width=page_width,
            page_height=page_height,
        )

        updated, pred_doc = tf_updater.replace_tabledata(
            pdf_path=pdf_path, true_doc=true_doc
        )

        if updated:

            if True:
                save_comparison_html(
                    filename=viz_dir / f"{os.path.basename(pdf_path)}-comp.html",
                    true_doc=true_doc,
                    pred_doc=pred_doc,
                    page_image=true_page_images[0],
                    true_labels=TRUE_HTML_EXPORT_LABELS,
                    pred_labels=PRED_HTML_EXPORT_LABELS,
                )

            true_doc, true_pictures, true_page_images = extract_images(
                true_doc,  # conv_results.document,
                pictures_column=BenchMarkColumns.GROUNDTRUTH_PICTURES,  # pictures_column,
                page_images_column=BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES,  # page_images_column,
            )

            pred_doc, pred_pictures, pred_page_images = extract_images(
                pred_doc,  # conv_results.document,
                pictures_column=BenchMarkColumns.PREDICTION_PICTURES,  # pictures_column,
                page_images_column=BenchMarkColumns.PREDICTION_PAGE_IMAGES,  # page_images_column,
            )

            record = {
                BenchMarkColumns.CONVERTER_TYPE: ConverterTypes.DOCLING,
                BenchMarkColumns.CONVERTER_VERSION: docling_version(),
                BenchMarkColumns.STATUS: "SUCCESS",
                BenchMarkColumns.DOC_ID: str(os.path.basename(jpg_path)),
                BenchMarkColumns.GROUNDTRUTH: json.dumps(true_doc.export_to_dict()),
                BenchMarkColumns.PREDICTION: json.dumps(pred_doc.export_to_dict()),
                BenchMarkColumns.ORIGINAL: get_binary(pdf_path),
                BenchMarkColumns.MIMETYPE: "application/pdf",
                BenchMarkColumns.PREDICTION_PAGE_IMAGES: pred_page_images,
                BenchMarkColumns.PREDICTION_PICTURES: pred_pictures,
                BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES: true_page_images,
                BenchMarkColumns.GROUNDTRUTH_PICTURES: true_pictures,
                BenchMarkColumns.MODALITIES: [EvaluationModality.TABLE_STRUCTURE],
            }
            records.append(record)

    test_dir = output_dir / "test"
    os.makedirs(test_dir, exist_ok=True)

    save_shard_to_disk(items=records, dataset_path=test_dir)

    write_datasets_info(
        name="OmniDocBench: tableformer",
        output_dir=output_dir,
        num_train_rows=0,
        num_test_rows=len(records),
    )
