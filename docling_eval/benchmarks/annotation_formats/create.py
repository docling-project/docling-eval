import argparse
import glob
import copy
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import xmltodict

from PIL import Image  # as PILImage

from tqdm import tqdm  # type: ignore

from datasets import Dataset, load_dataset

from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.labels import (
    DocItemLabel,
    GroupLabel,
    TableCellLabel,
    PictureClassificationLabel,
)

from docling_parse.pdf_parsers import pdf_parser_v2  # type: ignore[import]

from docling_core.types.doc.document import (
    DoclingDocument,
    DocItem,
    FloatingItem,
    PictureItem,
    TableItem,
    ImageRef,
    PageItem,
    ProvenanceItem,
    TableData,
)

from docling_eval.docling.utils import from_pil_to_base64uri, crop_bounding_box
from docling_eval.docling.utils import (
    insert_images,
    extract_images,
    docling_version,
    get_binary,
    save_shard_to_disk,
)

from docling_eval.benchmarks.constants import BenchMarkColumns
from docling_eval.benchmarks.utils import (
    draw_clusters_with_reading_order,
    save_inspection_html,
    save_comparison_html_with_clusters,
    write_datasets_info,
)

from docling_eval.docling.conversion import create_converter

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def find_box(boxes: List, point: Tuple[float, float]):

    index = -1
    area = 1e6

    for i, box in enumerate(boxes):
        assert box["l"] < box["r"]
        assert box["b"] > box["t"]

        if (
            box["l"] <= point[0]
            and point[0] <= box["r"]
            and box["t"] <= point[1]
            and point[1] <= box["b"]
        ):
            # if abs(box["r"]-box["l"])*(box["b"]-box["t"])<area:
            # area = abs(box["r"]-box["l"])*(box["b"]-box["t"])
            index = i

    if index == -1:
        loggging.error(f"point {point} is not in a bounding-box!")
        for i, box in enumerate(boxes):
            x = point[0]
            y = point[1]

            l = box["l"]
            r = box["r"]
            t = box["t"]
            b = box["b"]

            logging.infor(
                f"=> bbox: {l:.3f}, {r:.3f}, ({(l<x) and (x<r)}), {t:.3f}, {b:.3f}, ({(t<y) and (y<b)})"
            )

    return index, boxes[index]


def parse_annotation(image_annot: dict):

    basename = image_annot["@name"]

    keep = False

    boxes = []
    lines = []

    reading_order = None

    to_captions = []
    to_footnotes = []
    to_values = []

    merges = []
    group = []

    if "box" not in image_annot or "polyline" not in image_annot:
        return (
            basename,
            keep,
            boxes,
            lines,
            reading_order,
            to_captions,
            to_footnotes,
            next_text,
        )

    if isinstance(image_annot["box"], dict):
        boxes = [image_annot["box"]]
    elif isinstance(image_annot["box"], list):
        boxes = image_annot["box"]
    else:
        logging.error("could not get boxes")
        return (
            basename,
            keep,
            boxes,
            lines,
            reading_order,
            to_captions,
            to_footnotes,
            next_text,
        )

    if isinstance(image_annot["polyline"], dict):
        lines = [image_annot["polyline"]]
    elif isinstance(image_annot["polyline"], list):
        lines = image_annot["polyline"]
    else:
        logging.error("could not get boxes")
        return (
            basename,
            keep,
            boxes,
            lines,
            reading_order,
            to_captions,
            to_footnotes,
            next_text,
        )

    for i, box in enumerate(boxes):
        boxes[i]["b"] = float(box["@ybr"])
        boxes[i]["t"] = float(box["@ytl"])
        boxes[i]["l"] = float(box["@xtl"])
        boxes[i]["r"] = float(box["@xbr"])

    assert boxes[i]["b"] > boxes[i]["t"]

    for i, line in enumerate(lines):

        points = []
        for _ in line["@points"].split(";"):
            __ = _.split(",")
            points.append((float(__[0]), float(__[1])))

        boxids = []
        for point in points:
            bind, box = find_box(boxes=boxes, point=point)

            if 0 <= bind and bind < len(boxes):
                boxids.append(bind)

        lines[i]["points"] = points
        lines[i]["boxids"] = boxids

        # print(line["@label"], ": ", len(points), "\t", len(boxids))

    for i, line in enumerate(lines):
        if line["@label"] == "reading_order":
            assert reading_order is None  # you can only have 1 reading order
            keep = True
            reading_order = line

        elif line["@label"] == "to_caption":
            to_captions.append(line)
        elif line["@label"] == "to_footnote":
            to_footnotes.append(line)
        elif line["@label"] == "to_value":
            to_values.append(line)
        elif line["@label"] == "next_text" or line["@label"] == "merge":
            merges.append(line)
        elif line["@label"] == "next_figure" or line["@label"] == "group":
            group.append(line)

    return (
        basename,
        keep,
        boxes,
        lines,
        reading_order,
        to_captions,
        to_footnotes,
        to_values,
        merges,
        group,
    )


def create_prov(
    box: Dict,
    page_no: int,
    img_width: int,
    img_height: int,
    pdf_width: float,
    pdf_height: float,
    origin: CoordOrigin = CoordOrigin.TOPLEFT,
):

    bbox = BoundingBox(
        l=pdf_width * box["l"] / float(img_width),
        r=pdf_width * box["r"] / float(img_width),
        b=pdf_height * box["b"] / float(img_height),
        t=pdf_height * box["t"] / float(img_height),
        coord_origin=origin,
    )
    prov = ProvenanceItem(page_no=page_no, bbox=bbox, charspan=(0, 0))

    return prov, bbox


def get_label_prov_and_text(
    box: dict,
    page_no: int,
    img_width: float,
    img_height: float,
    pdf_width: float,
    pdf_height: float,
    parser: pdf_parser_v2,
    parsed_page: dict,
):

    assert page_no > 0

    prov, bbox = create_prov(
        box=box,
        page_no=page_no,
        img_width=img_width,
        img_height=img_height,
        pdf_width=pdf_width,
        pdf_height=pdf_height,
    )

    label = DocItemLabel(box["@label"])

    assert pdf_height - prov.bbox.b < pdf_height - prov.bbox.t

    pdf_text = parser.sanitize_cells_in_bbox(
        page=parsed_page,
        bbox=[
            prov.bbox.l,
            pdf_height - prov.bbox.b,
            prov.bbox.r,
            pdf_height - prov.bbox.t,
        ],
        cell_overlap=0.9,
        horizontal_cell_tolerance=1.0,
        enforce_same_font=False,
        space_width_factor_for_merge=1.5,
        space_width_factor_for_merge_with_space=0.33,
    )

    text = ""
    try:
        texts = []
        for row in pdf_text["data"]:
            texts.append(row[pdf_text["header"].index("text")])

        text = " ".join(texts)
    except:
        text = ""

    text = text.replace("  ", " ")

    return label, prov, text


def compute_iou(box_1: BoundingBox, box_2: BoundingBox, page_height: float):

    bbox1 = box_1.to_top_left_origin(page_height=page_height)
    bbox2 = box_2.to_top_left_origin(page_height=page_height)

    # Intersection coordinates
    inter_left = max(bbox1.l, bbox2.l)
    inter_top = max(bbox1.t, bbox2.t)
    inter_right = min(bbox1.r, bbox2.r)
    inter_bottom = min(bbox1.b, bbox2.b)

    # Intersection area
    if inter_left < inter_right and inter_top < inter_bottom:
        inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)
    else:
        inter_area = 0  # No intersection

    # Union area
    bbox1_area = (bbox1.r - bbox1.l) * (bbox1.b - bbox1.t)
    bbox2_area = (bbox2.r - bbox2.l) * (bbox2.b - bbox2.t)
    union_area = bbox1_area + bbox2_area - inter_area

    # IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def find_table_data(doc: DoclingDocument, prov: BoundingBox, iou_cutoff: float = 0.90):

    # logging.info(f"annot-table: {prov}")

    for item, level in doc.iterate_items():
        if isinstance(item, TableItem):
            for prov_ in item.prov:
                # logging.info(f"table: {prov_}")

                if prov_.page_no != prov.page_no:
                    continue

                page_height = doc.pages[prov_.page_no].size.height

                iou = compute_iou(
                    box_1=prov_.bbox, box_2=prov.bbox, page_height=page_height
                )

                if iou > iou_cutoff:
                    logging.info(f" => found table-data! {iou}")
                    return item.data

    logging.warning(" => missing table-data!")

    table_data = TableData(num_rows=-1, num_cols=-1, table_cells=[])
    return table_data


def get_next_provs(
    page_no: int,
    boxid: int,
    text: str,
    boxes: list,
    merges: list,
    already_added: list[int],
    true_doc: DoclingDocument,
    parser: pdf_parser_v2,
    parsed_page: dict,
):

    next_provs = []
    for merge in merges:
        if len(merge["boxids"]) > 1 and merge["boxids"][0] == boxid:

            for l in range(1, len(merge["boxids"])):
                boxid_ = merge["boxids"][l]
                already_added.append(boxid_)

                label_, prov_, text_ = get_label_prov_and_text(
                    box=boxes[boxid_],
                    page_no=page_no,
                    img_width=true_doc.pages[page_no].image.size.width,
                    img_height=true_doc.pages[page_no].image.size.height,
                    pdf_width=true_doc.pages[page_no].size.width,
                    pdf_height=true_doc.pages[page_no].size.height,
                    parser=parser,
                    parsed_page=parsed_page,
                )

                prov_.charspan = (len(text) + 1, len(text_))

                text = text + " " + text_

                next_provs.append(prov_)

    return next_provs, text, already_added


def add_captions_to_item(
    to_captions: list,
    item: FloatingItem,
    page_no: int,
    boxid: int,
    boxes: list,
    already_added: list[int],
    true_doc: DoclingDocument,
    parser: pdf_parser_v2,
    parsed_page: dict,
):

    for to_caption in to_captions:
        if to_caption["boxids"][0] == boxid:
            for l in range(1, len(to_caption["boxids"])):
                boxid_ = to_caption["boxids"][l]
                already_added.append(boxid_)

                caption_box = boxes[boxid_]

                label, prov, text = get_label_prov_and_text(
                    box=caption_box,
                    page_no=page_no,
                    img_width=true_doc.pages[page_no].image.size.width,
                    img_height=true_doc.pages[page_no].image.size.height,
                    pdf_width=true_doc.pages[page_no].size.width,
                    pdf_height=true_doc.pages[page_no].size.height,
                    parser=parser,
                    parsed_page=parsed_page,
                )

                caption_ref = true_doc.add_text(
                    label=DocItemLabel.CAPTION, prov=prov, text=text
                )
                item.captions.append(caption_ref.get_ref())

                if label != DocItemLabel.CAPTION:
                    logging.error(f"{label}!=DocItemLabel.CAPTION for {basename}")

    return true_doc, already_added


def add_footnotes_to_item(
    to_footnotes: list,
    item: FloatingItem,
    page_no: int,
    boxid: int,
    boxes: list,
    already_added: list[int],
    true_doc: DoclingDocument,
    parser: pdf_parser_v2,
    parsed_page: dict,
):

    for to_footnote in to_footnotes:
        if to_footnote["boxids"][0] == boxid:
            for l in range(1, len(to_footnote["boxids"])):
                boxid_ = to_footnote["boxids"][l]
                already_added.append(boxid_)

                footnote_box = boxes[boxid_]

                label, prov, text = get_label_prov_and_text(
                    box=footnote_box,
                    page_no=page_no,
                    img_width=true_doc.pages[page_no].image.size.width,
                    img_height=true_doc.pages[page_no].image.size.height,
                    pdf_width=true_doc.pages[page_no].size.width,
                    pdf_height=true_doc.pages[page_no].size.height,
                    parser=parser,
                    parsed_page=parsed_page,
                )

                footnote_ref = true_doc.add_text(
                    label=DocItemLabel.FOOTNOTE, prov=prov, text=text
                )
                item.footnotes.append(footnote_ref.get_ref())

                if label != DocItemLabel.FOOTNOTE:
                    logging.error(f"{label}!=DocItemLabel.FOOTNOTE for {basename}")

    return true_doc, already_added


def create_true_document(basename: str, annot: dict, desc: dict):

    (
        _,
        keep,
        boxes,
        lines,
        reading_order,
        to_captions,
        to_footnotes,
        to_values,
        merges,
        group,
    ) = parse_annotation(annot)
    assert _ == basename

    if not keep:
        logging.error(f"incorrect annotation for {basename}")
        return None

    logging.info(f"analyzing {basename}")

    # ========== Original Groundtruth
    orig_file = Path(desc["true_file"])
    assert os.path.exists(orig_file)

    with open(orig_file, "r") as fr:
        orig_doc = DoclingDocument.model_validate_json(json.load(fr))

    # ========== Original Prediction (to pre-annotate)
    pred_file = Path(desc["pred_file"])
    assert os.path.exists(pred_file)

    with open(pred_file, "r") as fr:
        pred_doc = DoclingDocument.model_validate_json(json.load(fr))

    # ========== Original PDF page
    pdf_file: Path = Path(desc["pdf_file"])
    assert os.path.exists(pdf_file)

    # Init the parser to extract the text-cells
    parser = pdf_parser_v2(level="fatal")
    success = parser.load_document(key=basename, filename=str(pdf_file))

    parsed_pages = {}
    for i, page_no in enumerate(desc["page_nos"]):
        parsed_doc = parser.parse_pdf_from_key_on_page(key=basename, page=page_no - 1)
        parsed_pages[page_no] = parsed_doc["pages"][0]

    parser.unload_document(basename)

    # ========== Create Ground Truth document
    true_doc = DoclingDocument(name=f"{basename}")

    for i, page_no in enumerate(desc["page_nos"]):

        # --- PDF
        assert len(parsed_doc["pages"]) == 1
        pdf_width = parsed_pages[page_no]["sanitized"]["dimension"]["width"]
        pdf_height = parsed_pages[page_no]["sanitized"]["dimension"]["height"]

        # --- PNG
        img_file = desc["page_img_files"][i]

        page_image = Image.open(str(img_file))
        # page_image.show()

        img_width = page_image.width
        img_height = page_image.height

        assert pred_doc.pages[page_no].image.size.width == img_width
        assert pred_doc.pages[page_no].image.size.height == img_height

        image_ref = ImageRef(
            mimetype="image/png",
            dpi=pred_doc.pages[page_no].image.dpi,
            size=Size(width=float(img_width), height=float(img_height)),
            uri=from_pil_to_base64uri(page_image),
        )
        page_item = PageItem(
            page_no=page_no,
            size=Size(width=float(pdf_width), height=float(pdf_height)),
            image=image_ref,
        )
        true_doc.pages[page_no] = page_item

    # Build the true-doc

    logging.info(reading_order)

    already_added = []

    for boxid in reading_order["boxids"]:

        if boxid in already_added:
            logging.warning(f"{boxid} is already added: {already_added}")
            continue

        # FIXME for later ...
        page_no = 1
        page_image = true_doc.pages[page_no].image.pil_image

        label, prov, text = get_label_prov_and_text(
            box=boxes[boxid],
            page_no=page_no,
            img_width=true_doc.pages[page_no].image.size.width,
            img_height=true_doc.pages[page_no].image.size.height,
            pdf_width=true_doc.pages[page_no].size.width,
            pdf_height=true_doc.pages[page_no].size.height,
            parser=parser,
            parsed_page=parsed_pages[page_no],
        )

        next_provs, text, already_added = get_next_provs(
            page_no=page_no,
            boxid=boxid,
            text=text,
            boxes=boxes,
            merges=merges,
            already_added=already_added,
            true_doc=true_doc,
            parser=parser,
            parsed_page=parsed_pages[page_no],
        )

        if label in [
            DocItemLabel.TEXT,
            DocItemLabel.PARAGRAPH,
            DocItemLabel.REFERENCE,
            DocItemLabel.PAGE_HEADER,
            DocItemLabel.PAGE_FOOTER,
            DocItemLabel.TITLE,
            DocItemLabel.FOOTNOTE,
        ]:
            current_item = true_doc.add_text(label=label, prov=prov, text=text)

            for next_prov in next_provs:
                current_item.prov.append(next_prov)

        elif label == DocItemLabel.SECTION_HEADER:
            true_doc.add_text(label=label, prov=prov, text=text)

        elif label == DocItemLabel.CAPTION:
            pass

        elif label == DocItemLabel.CHECKBOX_SELECTED:
            true_doc.add_text(label=label, prov=prov, text=text)

        elif label == DocItemLabel.CHECKBOX_UNSELECTED:
            true_doc.add_text(label=label, prov=prov, text=text)

        elif label == DocItemLabel.LIST_ITEM:
            true_doc.add_list_item(prov=prov, text=text)

        elif label == DocItemLabel.FORMULA:
            true_doc.add_text(label=label, prov=prov, text=text)

        elif label == DocItemLabel.CODE:
            true_doc.add_text(label=label, prov=prov, text=text)

        elif label == DocItemLabel.FORM:
            true_doc.add_text(label=label, prov=prov, text=text)

        elif label == DocItemLabel.KEY_VALUE_REGION:
            true_doc.add_text(label=label, prov=prov, text=text)

        elif label in [DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX]:

            table_data = find_table_data(doc=orig_doc, prov=prov)

            table_item = true_doc.add_table(label=label, data=table_data, prov=prov)

            true_doc, already_added = add_captions_to_item(
                to_captions=to_captions,
                item=table_item,
                page_no=page_no,
                boxid=boxid,
                boxes=boxes,
                already_added=already_added,
                true_doc=true_doc,
                parser=parser,
                parsed_page=parsed_pages[page_no],
            )

            true_doc, already_added = add_footnotes_to_item(
                to_footnotes=to_footnotes,
                item=table_item,
                page_no=page_no,
                boxid=boxid,
                boxes=boxes,
                already_added=already_added,
                true_doc=true_doc,
                parser=parser,
                parsed_page=parsed_pages[page_no],
            )

        elif label == DocItemLabel.PICTURE:

            crop_image = crop_bounding_box(
                page_image=page_image, page=true_doc.pages[page_no], bbox=prov.bbox
            )

            imgref = ImageRef(
                mimetype="image/png",
                dpi=true_doc.pages[page_no].image.dpi,
                size=Size(width=crop_image.width, height=crop_image.height),
                uri=from_pil_to_base64uri(crop_image),
            )

            picture_item = true_doc.add_picture(prov=prov, image=imgref)

            true_doc, already_added = add_captions_to_item(
                to_captions=to_captions,
                item=picture_item,
                page_no=page_no,
                boxid=boxid,
                boxes=boxes,
                already_added=already_added,
                true_doc=true_doc,
                parser=parser,
                parsed_page=parsed_pages[page_no],
            )

            true_doc, already_added = add_footnotes_to_item(
                to_footnotes=to_footnotes,
                item=picture_item,
                page_no=page_no,
                boxid=boxid,
                boxes=boxes,
                already_added=already_added,
                true_doc=true_doc,
                parser=parser,
                parsed_page=parsed_pages[page_no],
            )

    return true_doc


def contains_reading_order(image_annot: dict):

    if "box" not in image_annot:
        return False

    if "polyline" not in image_annot:
        return False

    if isinstance(image_annot["polyline"], dict):
        lines = [image_annot["polyline"]]
    elif isinstance(image_annot["polyline"], list):
        lines = image_annot["polyline"]
    else:
        return False

    cnt = 0
    for i, line in enumerate(lines):
        if line["@label"] == "reading_order":
            cnt += 1

    return cnt == 1


def from_cvat_to_docling_document(
    annotation_filenames: List[Path],
    overview: dict,
    imgs_dir: Path,
    pdfs_dir: Path,
    image_scale: float = 1.0,
):

    for annot_file in annotation_filenames:

        with open(str(annot_file), "r") as fr:
            xml_data = fr.read()

        # Convert XML to a Python dictionary
        annot_data = xmltodict.parse(xml_data)

        for image_annot in annot_data["annotations"]["image"]:

            basename = image_annot["@name"]

            if basename not in overview:
                logging.warning(f"Skipping {basename}: not in overview_file")
                yield overview[basename], None

            elif not contains_reading_order(image_annot):
                logging.warning(f"Skipping {basename}: no reading-order detected")
                yield overview[basename], None

            else:
                true_doc = create_true_document(
                    basename=basename, annot=image_annot, desc=overview[basename]
                )
                yield overview[basename], true_doc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create new evaluation dataset using annotation file."
    )

    parser.add_argument(
        "-i", "--input_dir", required=True, help="Path to the input directory"
    )
    parser.add_argument(
        "-a",
        "--annot_file",
        required=False,
        help="Path to the CVAT annotation file.",
        default="annotations.xml",
    )

    args = parser.parse_args()
    return (
        Path(args.input_dir),
        Path(args.input_dir) / args.annot_file,
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
    # DocItemLabel.CAPTION,
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


def create_layout_dataset_from_annotations(input_dir: Path, annot_file: Path):

    output_dir = input_dir / "layout"

    imgs_dir = input_dir / "imgs"
    page_imgs_dir = input_dir / "page_imgs"
    pdfs_dir = input_dir / "pdfs"

    json_true_dir = input_dir / "json-groundtruth"
    json_pred_dir = input_dir / "json-predictions"
    json_anno_dir = input_dir / "json-annotations"

    html_anno_dir = input_dir / "html-annotations"
    html_viz_dir = input_dir / "html-annotatations-viz"

    overview_file = input_dir / "overview_map.json"

    with open(overview_file, "r") as fr:
        overview = json.load(fr)

    for _ in [
        input_dir,
        output_dir,
        imgs_dir,
        page_imgs_dir,
        pdfs_dir,
        json_true_dir,
        json_pred_dir,
        json_anno_dir,
        html_anno_dir,
        html_viz_dir,
    ]:
        os.makedirs(_, exist_ok=True)

    image_scale = 2.0

    # Create Converter
    doc_converter = create_converter(page_image_scale=image_scale)

    records = []
    for desc, true_doc in tqdm(
        from_cvat_to_docling_document(
            annotation_filenames=[annot_file],
            overview=overview,
            pdfs_dir=pdfs_dir,
            imgs_dir=imgs_dir,
        ),
        total=len(overview),
        ncols=128,
        desc="Creating documents from annotations",
    ):

        basename = desc["basename"]

        """
        save_inspection_html(filename=str(html_viz_dir / f"{basename}.html"), doc = true_doc,
                             labels=TRUE_HTML_EXPORT_LABELS)
        """

        pdf_file = desc["pdf_file"]

        # Create the predicted Document
        conv_results = doc_converter.convert(source=pdf_file, raises_on_error=True)
        pred_doc = conv_results.document

        true_doc, true_pictures, true_page_images = extract_images(
            document=true_doc,
            pictures_column=BenchMarkColumns.GROUNDTRUTH_PICTURES.value,  # pictures_column,
            page_images_column=BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES.value,  # page_images_column,
        )

        pred_doc, pred_pictures, pred_page_images = extract_images(
            document=pred_doc,
            pictures_column=BenchMarkColumns.PREDICTION_PICTURES.value,  # pictures_column,
            page_images_column=BenchMarkColumns.PREDICTION_PAGE_IMAGES.value,  # page_images_column,
        )

        if True:
            save_comparison_html_with_clusters(
                filename=html_viz_dir / f"{basename}-clusters.html",
                true_doc=true_doc,
                pred_doc=pred_doc,
                page_image=true_page_images[0],
                true_labels=TRUE_HTML_EXPORT_LABELS,
                pred_labels=PRED_HTML_EXPORT_LABELS,
            )

        record = {
            BenchMarkColumns.DOCLING_VERSION: docling_version(),
            BenchMarkColumns.STATUS: str(conv_results.status),
            BenchMarkColumns.DOC_ID: str(basename),
            BenchMarkColumns.GROUNDTRUTH: json.dumps(true_doc.export_to_dict()),
            BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES: true_page_images,
            BenchMarkColumns.GROUNDTRUTH_PICTURES: true_pictures,
            BenchMarkColumns.PREDICTION: json.dumps(pred_doc.export_to_dict()),
            BenchMarkColumns.PREDICTION_PAGE_IMAGES: pred_page_images,
            BenchMarkColumns.PREDICTION_PICTURES: pred_pictures,
            BenchMarkColumns.ORIGINAL: get_binary(pdf_file),
            BenchMarkColumns.MIMETYPE: "application/pdf",
        }
        records.append(record)

    test_dir = output_dir / "test"
    os.makedirs(test_dir, exist_ok=True)

    save_shard_to_disk(items=records, dataset_path=test_dir)

    write_datasets_info(
        name="DPBench: end-to-end",
        output_dir=output_dir,
        num_train_rows=0,
        num_test_rows=len(records),
    )


def main():

    input_dir, preannot_file = parse_args()

    imgs_dir = input_dir / "imgs"
    page_imgs_dir = input_dir / "page_imgs"
    pdfs_dir = input_dir / "pdfs"

    json_true_dir = input_dir / "json-groundtruth"
    json_pred_dir = input_dir / "json-predictions"
    json_anno_dir = input_dir / "json-annotations"

    html_anno_dir = input_dir / "html-annotations"
    html_viz_dir = input_dir / "html-annotatations-viz"

    overview_file = input_dir / "overview_map.json"

    with open(overview_file, "r") as fr:
        overview = json.load(fr)

    for _ in [
        input_dir,
        imgs_dir,
        page_imgs_dir,
        pdfs_dir,
        json_true_dir,
        json_pred_dir,
        json_anno_dir,
        html_anno_dir,
        html_viz_dir,
    ]:
        os.makedirs(_, exist_ok=True)

    image_scale = 2.0

    # Create Converter
    doc_converter = create_converter(page_image_scale=image_scale)

    for desc, true_doc in tqdm(
        from_cvat_to_docling_document(
            annotation_filenames=[preannot_file],
            overview=overview,
            pdfs_dir=pdfs_dir,
            imgs_dir=imgs_dir,
        ),
        total=len(overview),
        ncols=128,
        desc="Creating documents from annotations",
    ):

        basename = desc["basename"]

        save_inspection_html(
            filename=str(html_viz_dir / f"{basename}.html"),
            doc=true_doc,
            labels=TRUE_HTML_EXPORT_LABELS,
        )


if __name__ == "__main__":
    # main()

    input_dir, annot_file = parse_args()

    create_layout_dataset_from_annotations(input_dir=input_dir, annot_file=annot_file)
