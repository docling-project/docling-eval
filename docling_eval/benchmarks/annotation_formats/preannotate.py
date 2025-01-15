import argparse
import copy
import glob
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import Dataset, load_dataset
from docling_core.types.doc.document import (
    DocItem,
    DoclingDocument,
    PictureItem,
    TableItem,
)
from docling_core.types.doc.labels import (
    DocItemLabel,
    GroupLabel,
    PictureClassificationLabel,
    TableCellLabel,
)
from tqdm import tqdm  # type: ignore

from docling_eval.benchmarks.annotation_formats.utils import (
    DocLinkLabel,
    TableComponentLabel,
)
from docling_eval.benchmarks.constants import BenchMarkColumns
from docling_eval.docling.utils import insert_images

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def create_cvat_project_properties(project_file: Path):

    def rgb_to_hex(r, g, b):
        """
        Converts RGB values to a HEX color code.

        Args:
            r (int): Red value (0-255)
            g (int): Green value (0-255)
            b (int): Blue value (0-255)

        Returns:
            str: HEX color code (e.g., "#RRGGBB")
        """
        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
            raise ValueError("RGB values must be in the range 0-255")

        return f"#{r:02X}{g:02X}{b:02X}"

    def line_to_rgba_color(line: str):

        if line == "reading_order":
            return (255, 0, 0)
        elif line == "next_text":
            return (255, 0, 255)
        elif line == "next_figure":
            return (255, 0, 255)
        elif line == "to_caption":
            return (0, 255, 0)
        elif line == "to_footnote":
            return (0, 255, 0)
        elif line == "to_value":
            return (0, 0, 255)
        else:
            exit(-1)

    def line_to_hex_color(line: str):

        r, g, b = line_to_rgba_color(line)
        return rgb_to_hex(r, g, b)

    results = []

    for item in DocItemLabel:

        r, g, b = DocItemLabel.get_color(item)

        results.append(
            {
                "name": item.value,
                "color": rgb_to_hex(r, g, b),
                "type": "rectangle",
                "attributes": [],
            }
        )

        if item in [DocItemLabel.LIST_ITEM, DocItemLabel.SECTION_HEADER]:
            results[-1]["attributes"].append(
                {
                    "name": "level",
                    "input_type": "number",
                    "mutable": True,
                    "values": ["1", "10", "1"],
                    "default_value": "1",
                }
            )

        if item == [DocItemLabel.FORMULA]:
            results[-1]["attributes"].append(
                {
                    "name": "latex",
                    "mutable": True,
                    "input_type": "text",
                    "mutable": False,
                    "values": [""],
                    "default_value": "",
                }
            )

        if item == [DocItemLabel.CODE]:
            results[-1]["attributes"].append(
                {
                    "name": "code",
                    "mutable": True,
                    "input_type": "text",
                    "mutable": False,
                    "values": [""],
                    "default_value": "",
                }
            )

        if item == DocItemLabel.PICTURE:

            labels = []
            for label in PictureClassificationLabel:
                labels.append(str(label))

            results[-1]["attributes"].append(
                {
                    "name": f"{item.value}-class",
                    "input_type": "select",
                    "mutable": True,
                    "values": labels,
                    "default_value": str(PictureClassificationLabel.OTHER),
                }
            )
            results[-1]["attributes"].append(
                {
                    "name": "json",
                    "mutable": True,
                    "input_type": "text",
                    "mutable": False,
                    "values": [""],
                    "default_value": "",
                }
            )

    for table_item in TableComponentLabel:

        r, g, b = TableComponentLabel.get_color(table_item)

        results.append(
            {
                "name": table_item.value,
                "color": rgb_to_hex(r, g, b),
                "type": "rectangle",
                "attributes": [],
            }
        )

    """    
    for item in TableCellLabel:

        r, g, b  = TableCellLabel.get_color(item)
        
        results.append({
            "name": item.value,
            "color": rgb_to_hex(r, g, b),
            "type": "rectangle",
            "attributes": []
        })
    """

    for link_item in DocLinkLabel:

        r, g, b = DocLinkLabel.get_color(link_item)

        results.append(
            {
                "name": link_item.value,
                "color": rgb_to_hex(r, g, b),
                "type": "polyline",
                "attributes": [],
            }
        )

    logging.info(f"writing project description: {str(project_file)}")
    with open(str(project_file), "w") as fw:
        fw.write(json.dumps(results, indent=2))


def create_cvat_preannotation_file_for_single_page(
    docs: List[DoclingDocument],
    overview: List[dict],
    output_dir: Path,
    imgs_dir: Path,
    page_imgs_dir: Path,
):

    assert len(docs) == len(overview)

    results = []

    results.append('<?xml version="1.0" encoding="utf-8"?>')
    results.append("<annotations>")

    img_to_doc = {}

    img_id = 0
    for doc_id, doc in tqdm(
        enumerate(docs), ncols=128, desc="creating the CVAT file", total=len(docs)
    ):

        doc_overview = overview[doc_id]

        doc_name = doc.name

        page_images = []
        page_fnames = []
        for j, page in doc.pages.items():
            filename = f"doc_{doc_name}_page_{j:06}.png"

            img_file = str(imgs_dir / filename)
            page_img_file = str(page_imgs_dir / filename)

            page_image = page.image.pil_image

            page_image.save(img_file)
            page_image.save(page_img_file)

            page_images.append(page_image)
            page_fnames.append(filename)

            img_to_doc[filename] = {
                "basename": filename,
                "img_w": page_image.width,
                "img_h": page_image.height,
                "img_id": img_id,
                "img_file": img_file,
                "page_img_files": [page_img_file],
                "pdf_file": doc_overview["pdf_file"],
                "true_file": doc_overview["true_file"],
                "pred_file": doc_overview["pred_file"],
                "page_nos": [j],
                "page_inds": [j - 1],
            }

        page_bboxes: Dict[int, List[dict]] = {}
        for i, fname in enumerate(page_fnames):
            page_bboxes[i] = []

        for item, level in doc.iterate_items():
            if isinstance(item, DocItem):  # and item.label in labels:
                for prov in item.prov:
                    page_no = prov.page_no

                    page_w = doc.pages[prov.page_no].size.width
                    page_h = doc.pages[prov.page_no].size.height

                    img_w = page_images[page_no - 1].width
                    img_h = page_images[page_no - 1].height

                    page_bbox = prov.bbox.to_top_left_origin(page_height=page_h)

                    img_bbox = [
                        page_bbox.l / page_w * img_w,
                        page_bbox.b / page_h * img_h,
                        page_bbox.r / page_w * img_w,
                        page_bbox.t / page_h * img_h,
                    ]

                    page_bboxes[page_no - 1].append(
                        {
                            "label": item.label.value,
                            "l": img_bbox[0],
                            "r": img_bbox[2],
                            "b": img_bbox[1],
                            "t": img_bbox[3],
                        }
                    )

        for page_no, page_file in enumerate(page_fnames):
            img_w = page_images[page_no].width
            img_h = page_images[page_no].height

            results.append(
                f'<image id="{img_id}" name="{page_file}" width="{img_w}" height="{img_h}">'
            )

            for bbox_id, bbox in enumerate(page_bboxes[page_no]):
                label = bbox["label"]
                l = round(bbox["l"])
                r = round(bbox["r"])
                t = round(bbox["t"])
                b = round(bbox["b"])
                results.append(
                    f'<box label="{label}" source="docling" occluded="0" xtl="{l}" ytl="{t}" xbr="{r}" ybr="{b}" z_order="{bbox_id}"></box>'
                )

            results.append("</image>")

    results.append("</annotations>")

    with open(str(output_dir / "pre-annotations.xml"), "w") as fw:
        fw.write("\n".join(results))

    with open(str(output_dir / "overview_map.json"), "w") as fw:
        fw.write(json.dumps(img_to_doc, indent=2))


def export_from_dataset_supplementary_files(
    idir: Path, imgs_dir: Path, pdfs_dir: Path, json_true_dir: Path, json_pred_dir: Path
):

    # benchmark_path = Path("./benchmarks/DPBench-dataset/layout/test")

    test_files = sorted(glob.glob(str(idir / "*.parquet")))
    ds = load_dataset("parquet", data_files={"test": test_files})

    logging.info(f"oveview of dataset: {ds}")

    if ds is not None:
        ds_selection = ds["test"]

    docs, overview = [], []
    for i, data in tqdm(
        enumerate(ds_selection),
        desc="iterating dataset",
        ncols=120,
        total=len(ds_selection),
    ):
        # Get the Docling predicted document
        pred_doc_dict = data[BenchMarkColumns.PREDICTION]
        pred_doc = DoclingDocument.model_validate_json(pred_doc_dict)

        page_images = data[BenchMarkColumns.PREDICTION_PAGE_IMAGES]
        pics_images = data[BenchMarkColumns.PREDICTION_PICTURES]

        insert_images(pred_doc, page_images=page_images, pictures=pics_images)

        # Get the groundtruth document (to cherry pick table structure later ...)
        true_doc_dict = data[BenchMarkColumns.GROUNDTRUTH]

        # FIXME: make the unique name in a column
        doc_name = f"{pred_doc.name}"

        # Write groundtruth and predicted document. The groundtruth will
        # be replaced/updated by the annoted ones later on
        true_file = str(json_true_dir / f"{doc_name}.json")
        with open(true_file, "w") as fw:
            fw.write(json.dumps(true_doc_dict, indent=2))

        pred_file = str(json_pred_dir / f"{doc_name}.json")
        with open(str(json_pred_dir / f"{doc_name}.json"), "w") as fw:
            fw.write(json.dumps(pred_doc_dict, indent=2))

        # Write original pdf ...
        pdf_name = doc_name
        if not pdf_name.endswith(".pdf"):
            pdf_name = f"{pdf_name}.pdf"

        pdf_file = str(pdfs_dir / pdf_name)

        bindoc = data[BenchMarkColumns.ORIGINAL]
        with open(pdf_file, "wb") as fw:
            fw.write(bindoc)

        docs.append(pred_doc)

        overview.append(
            {
                "true_file": true_file,
                "pred_file": pred_file,
                "pdf_file": pdf_file,
            }
        )

    return docs, overview


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process input and output directories and a pre-annotation file."
    )

    parser.add_argument(
        "-i",
        "--input_dir",
        required=True,
        help="Path to the input dataset directory with parquet files.",
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, help="Path to the output directory."
    )
    parser.add_argument(
        "-a",
        "--preannot_file",
        required=False,
        help="Path to the pre-annotation file.",
        default="cvat-pre-annotations.xml",
    )
    parser.add_argument(
        "-p",
        "--project_file",
        required=False,
        help="Path to the project descriptiom file.",
        default="cvat-project.json",
    )

    args = parser.parse_args()
    return (
        Path(args.input_dir),
        Path(args.output_dir),
        Path(args.output_dir) / args.preannot_file,
        Path(args.output_dir) / args.project_file,
    )


def main():

    input_dir, output_dir, preannot_file, project_file = parse_args()

    imgs_dir = output_dir / "imgs"
    page_imgs_dir = output_dir / "page_imgs"
    pdfs_dir = output_dir / "pdfs"

    json_true_dir = output_dir / "json-groundtruth"
    json_pred_dir = output_dir / "json-predictions"
    json_anno_dir = output_dir / "json-annotations"

    for _ in [
        output_dir,
        imgs_dir,
        page_imgs_dir,
        pdfs_dir,
        json_true_dir,
        json_pred_dir,
        json_anno_dir,
    ]:
        os.makedirs(_, exist_ok=True)

    create_cvat_project_properties(project_file=project_file)

    docs, overview = export_from_dataset_supplementary_files(
        idir=input_dir,
        pdfs_dir=pdfs_dir,
        imgs_dir=imgs_dir,
        json_true_dir=json_true_dir,
        json_pred_dir=json_pred_dir,
    )

    assert len(docs) == len(overview)

    create_cvat_preannotation_file_for_single_page(
        docs=docs,
        overview=overview,
        output_dir=output_dir,
        imgs_dir=imgs_dir,
        page_imgs_dir=page_imgs_dir,
    )


if __name__ == "__main__":
    main()
