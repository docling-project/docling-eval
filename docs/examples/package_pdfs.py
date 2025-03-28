import copy
import json
import logging
from pathlib import Path

from docling.datamodel.base_models import ConversionStatus
from docling_core.types.doc.document import ImageRefMode
from docling_core.types.doc.labels import DocItemLabel
from tqdm import tqdm  # type: ignore

from docling_eval.datamodels.types import BenchMarkColumns
from docling_eval.legacy.converters.conversion import create_pdf_docling_converter
from docling_eval.utils.utils import (
    docling_version,
    from_pil_to_base64,
    get_binary,
    save_shard_to_disk,
)
from docling_eval.visualisation.constants import HTML_INSPECTION
from docling_eval.visualisation.visualisations import draw_clusters_with_reading_order

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
)

MAX_RECORDS = 1000

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


import argparse
import os


def parse_arguments():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Process arguments for parsing PDFs.")

    # Add arguments
    parser.add_argument(
        "-n", "--name", required=True, type=str, help="The name of the process."
    )
    parser.add_argument(
        "-d",
        "--directory",
        required=True,
        type=str,
        help="Path to the directory containing PDFs.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        type=bool,
        default=True,
        help="Whether to search directories recursively (default: True).",
    )
    parser.add_argument(
        "-s",
        "--image_scale",
        type=float,
        default=1.0,
        help="Scaling factor for images (default: 1.0).",
    )
    parser.add_argument(
        "--ocr", type=bool, default=False, help="Do OCR (default: False)."
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate the directory
    if not os.path.isdir(args.directory):
        raise ValueError(f"The specified directory '{args.directory}' does not exist.")

    # Collect PDF files
    pdf_files = []
    if args.recursive:
        for root, _, files in os.walk(args.directory):
            pdf_files.extend(
                [os.path.join(root, f) for f in files if f.lower().endswith(".pdf")]
            )
    else:
        pdf_files = [
            os.path.join(args.directory, f)
            for f in os.listdir(args.directory)
            if f.lower().endswith(".pdf")
        ]

    # Return parsed arguments and the PDF files
    return (
        args.name,
        args.directory,
        args.recursive,
        args.image_scale,
        pdf_files,
        args.ocr,
    )


def main():

    # Set logging level for the 'docling' package
    # logging.getLogger('docling').setLevel(logging.WARNING)

    name, directory, recursive, image_scale, pdf_files, do_ocr = parse_arguments()

    odir = Path(f"./benchmarks/{name}")

    pqt_dir = odir / "dataset"
    viz_dir = odir / "visualization"

    os.makedirs(pqt_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    # Create Converter
    doc_converter = create_pdf_docling_converter(
        page_image_scale=image_scale, do_ocr=do_ocr
    )

    records = []

    tid, sid = 0, 0

    for pdf_file in tqdm(pdf_files, total=len(pdf_files), ncols=128):

        # Create the predicted Document
        try:
            conv_results = doc_converter.convert(source=pdf_file, raises_on_error=True)
            pred_doc = conv_results.document
        except:
            record = {
                BenchMarkColumns.CONVERTER_VERSION: docling_version(),
                BenchMarkColumns.STATUS: str(ConversionStatus.FAILURE.value),
                BenchMarkColumns.DOC_ID: str(os.path.basename(pdf_file)),
                BenchMarkColumns.PREDICTION: json.dumps(None),
                BenchMarkColumns.ORIGINAL: get_binary(pdf_file),
                BenchMarkColumns.MIMETYPE: "application/pdf",
                BenchMarkColumns.TIMINGS: json.dumps(None),
            }
            records.append(record)
            continue

        timings = {}
        for key, item in conv_results.timings.items():
            timings[key] = json.loads(item.model_dump_json())

        html_doc = pred_doc.export_to_html(
            image_mode=ImageRefMode.EMBEDDED,
            # html_head=HTML_DEFAULT_HEAD_FOR_COMP,
            # labels=pred_labels,
        )

        html_doc = html_doc.replace("'", "&#39;")

        page_images = []
        page_template = '<div class="image-wrapper"><img src="data:image/png;base64,BASE64PAGE" alt="Example Image"></div>'
        for page_no, page in pred_doc.pages.items():
            page_img = page.image.pil_image

            page_img = draw_clusters_with_reading_order(
                doc=pred_doc,
                page_image=page_img,
                labels=PRED_HTML_EXPORT_LABELS,
                page_no=page_no,
                reading_order=True,
            )

            page_base64 = from_pil_to_base64(page_img)
            page_images.append(page_template.replace("BASE64PAGE", page_base64))

        page = copy.deepcopy(HTML_INSPECTION)
        page = page.replace("PREDDOC", html_doc)
        page = page.replace("PAGE_IMAGES", "\n".join(page_images))

        filename = viz_dir / f"{os.path.basename(pdf_file)}.html"
        with open(str(filename), "w") as fw:
            fw.write(page)

        record = {
            BenchMarkColumns.CONVERTER_VERSION: docling_version(),
            BenchMarkColumns.STATUS: str(conv_results.status.value),
            BenchMarkColumns.DOC_ID: str(os.path.basename(pdf_file)),
            BenchMarkColumns.PREDICTION: json.dumps(pred_doc.export_to_dict()),
            BenchMarkColumns.ORIGINAL: get_binary(pdf_file),
            BenchMarkColumns.MIMETYPE: "application/pdf",
            BenchMarkColumns.TIMINGS: json.dumps(timings),
        }
        records.append(record)

        if len(records) == MAX_RECORDS:
            save_shard_to_disk(
                items=records,
                dataset_path=pqt_dir,
                thread_id=tid,
                shard_id=sid,
                shard_format="parquet",
            )
            sid += 1
            records = []

    if len(records) > 0:
        save_shard_to_disk(
            items=records,
            dataset_path=pqt_dir,
            thread_id=tid,
            shard_id=sid,
            shard_format="parquet",
        )
        sid += 1
        records = []


if __name__ == "__main__":
    main()
