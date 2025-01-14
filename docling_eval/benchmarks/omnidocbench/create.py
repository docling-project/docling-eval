import argparse
import glob
import json
import logging
import os
from pathlib import Path

from bs4 import BeautifulSoup  # type: ignore
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import DoclingDocument, ImageRef, ProvenanceItem
from docling_core.types.doc.labels import DocItemLabel
from PIL import Image  # as PILImage
from tqdm import tqdm  # type: ignore

from docling_eval.benchmarks.constants import BenchMarkColumns
from docling_eval.benchmarks.utils import (
    add_pages_to_true_doc,
    convert_html_table_into_docling_tabledata,
    save_comparison_html,
    save_comparison_html_with_clusters,
    write_datasets_info,
)
from docling_eval.docling.conversion import create_converter
from docling_eval.docling.models.tableformer.tf_model_prediction import (
    TableFormerUpdater,
)
from docling_eval.docling.utils import (
    crop_bounding_box,
    docling_version,
    extract_images,
    from_pil_to_base64uri,
    get_binary,
    save_shard_to_disk,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
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


def get_filenames(omnidocbench_dir: Path):

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
    omnidocbench_dir: Path, output_dir: Path, image_scale: float = 1.0
):

    # Create Converter
    doc_converter = create_converter(page_image_scale=image_scale)

    # load the groundtruth
    with open(omnidocbench_dir / f"OmniDocBench.json", "r") as fr:
        gt = json.load(fr)

    gt = update_gt_into_map(gt)

    viz_dir = output_dir / "vizualisations"
    os.makedirs(viz_dir, exist_ok=True)

    records = []

    page_tuples = get_filenames(omnidocbench_dir)

    cnt = 0

    for page_tuple in tqdm(
        page_tuples,
        total=len(page_tuples),
        ncols=128,
        desc="Processing files for OmniDocBench with end-to-end",
    ):

        jpg_path = page_tuple[0]
        pdf_path = Path(page_tuple[1])

        # logging.info(f"file: {pdf_path}")
        if not os.path.basename(jpg_path) in gt:
            logging.error(f"did not find ground-truth for {os.path.basename(jpg_path)}")
            continue

        gt_doc = gt[os.path.basename(jpg_path)]

        # Create the predicted Document
        conv_results = doc_converter.convert(source=pdf_path, raises_on_error=True)
        pred_doc = conv_results.document

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

        if True:
            """
            save_comparison_html(
                filename=viz_dir / f"{os.path.basename(pdf_path)}-comp.html",
                true_doc=true_doc,
                pred_doc=pred_doc,
                page_image=true_page_images[0],
                true_labels=TRUE_HTML_EXPORT_LABELS,
                pred_labels=PRED_HTML_EXPORT_LABELS,
            )
            """

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
            BenchMarkColumns.DOCLING_VERSION: docling_version(),
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
    omnidocbench_dir: Path, output_dir: Path, image_scale: float = 1.0
):
    # Init the TableFormer model
    tf_updater = TableFormerUpdater()

    # load the groundtruth
    with open(omnidocbench_dir / f"OmniDocBench.json", "r") as fr:
        gt = json.load(fr)

    gt = update_gt_into_map(gt)

    viz_dir = output_dir / "vizualisations"
    os.makedirs(viz_dir, exist_ok=True)

    records = []

    page_tuples = get_filenames(omnidocbench_dir)

    for page_tuple in tqdm(
        page_tuples,
        total=len(page_tuples),
        ncols=128,
        desc="Processing files for OmniDocBench with end-to-end",
    ):

        jpg_path = page_tuple[0]
        pdf_path = Path(page_tuple[1])

        # logging.info(f"file: {pdf_path}")
        if not os.path.basename(jpg_path) in gt:
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
                BenchMarkColumns.DOCLING_VERSION: docling_version(),
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


def parse_arguments():
    """Parse arguments for DP-Bench parsing."""

    parser = argparse.ArgumentParser(
        description="Process DP-Bench benchmark from directory into HF dataset."
    )
    parser.add_argument(
        "-i",
        "--omnidocbench-directory",
        help="input directory with documents",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        help="output directory with shards",
        required=False,
        default="./benchmarks/omnidocbench",
    )
    parser.add_argument(
        "-s",
        "--image-scale",
        help="image-scale of the pages",
        required=False,
        default=1.0,
    )
    parser.add_argument(
        "-m",
        "--mode",
        help="mode of dataset",
        required=False,
        choices=["end-2-end", "table", "formula", "all"],
    )
    args = parser.parse_args()

    return (
        Path(args.omnidocbench_directory),
        Path(args.output_directory),
        float(args.image_scale),
        args.mode,
    )


def main():

    omnidocbench_dir, output_dir, image_scale, mode = parse_arguments()

    # Create the directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    odir_e2e = Path(output_dir) / "end_to_end"
    odir_tab = Path(output_dir) / "tables"
    odir_eqn = Path(output_dir) / "formulas"

    os.makedirs(odir_e2e, exist_ok=True)
    os.makedirs(odir_tab, exist_ok=True)
    # os.makedirs(odir_eqn, exist_ok=True)

    for _ in ["test", "train"]:
        os.makedirs(odir_e2e / _, exist_ok=True)
        os.makedirs(odir_tab / _, exist_ok=True)

    if mode == "end-2-end":
        create_omnidocbench_e2e_dataset(
            omnidocbench_dir=omnidocbench_dir,
            output_dir=odir_e2e,
            image_scale=image_scale,
        )

    elif mode == "table":
        create_omnidocbench_tableformer_dataset(
            omnidocbench_dir=omnidocbench_dir,
            output_dir=odir_tab,
            image_scale=image_scale,
        )

    elif mode == "all":
        create_omnidocbench_e2e_dataset(
            omnidocbench_dir=omnidocbench_dir,
            output_dir=odir_e2e,
            image_scale=image_scale,
        )

        create_omnidocbench_tableformer_dataset(
            omnidocbench_dir=omnidocbench_dir,
            output_dir=odir_tab,
            image_scale=image_scale,
        )


if __name__ == "__main__":
    main()
