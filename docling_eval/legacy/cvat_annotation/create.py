import argparse
import glob
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, cast

import xmltodict  # type: ignore[import]
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import (
    DoclingDocument,
    FloatingItem,
    GraphData,
    ImageRef,
    PageItem,
    ProvenanceItem,
    TableData,
    TableItem,
)
from docling_core.types.doc.labels import DocItemLabel
from docling_parse.pdf_parsers import pdf_parser_v2  # type: ignore[import]
from PIL import Image  # as PILImage
from tqdm import tqdm  # type: ignore

from docling_eval.datamodels.types import (
    BenchMarkColumns,
    ConverterTypes,
    EvaluationModality,
)
from docling_eval.legacy.converters.conversion import create_pdf_docling_converter
from docling_eval.legacy.cvat_annotation.utils import (
    AnnotatedImage,
    AnnotationOverview,
    BenchMarkDirs,
)
from docling_eval.utils.utils import (
    crop_bounding_box,
    docling_version,
    extract_images,
    from_pil_to_base64uri,
    get_binary,
    get_binhash,
    save_shard_to_disk,
    write_datasets_info,
)
from docling_eval.visualisation.visualisations import save_comparison_html_with_clusters

# from pydantic import


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def find_box(boxes: List, point: Tuple[float, float]):

    index = -1
    area = 1e9

    for i, box in enumerate(boxes):
        assert box["l"] < box["r"]
        assert box["b"] > box["t"]

        if (
            box["l"] <= point[0]
            and point[0] <= box["r"]
            and box["t"] <= point[1]
            and point[1] <= box["b"]
        ):
            if index == -1 or abs(box["r"] - box["l"]) * (box["b"] - box["t"]) < area:
                area = abs(box["r"] - box["l"]) * (box["b"] - box["t"])
                index = i

    if index == -1:
        logging.error(f"point {point} is not in a bounding-box!")
        for i, box in enumerate(boxes):
            x = point[0]
            y = point[1]

            l = box["l"]
            r = box["r"]
            t = box["t"]
            b = box["b"]

            # logging.info(f"=> bbox: {l:.3f}, {r:.3f}, ({(l<x) and (x<r)}), {t:.3f}, {b:.3f}, ({(t<y) and (y<b)})")

    return index, boxes[index]


def parse_annotation(image_annot: dict):

    basename: str = image_annot["@name"]
    # logging.info(f"parsing annotations for {basename}")

    keep: bool = False

    boxes: List[dict] = []
    lines: List[dict] = []

    reading_order: dict = {}

    to_captions: List[dict] = []
    to_footnotes: List[dict] = []
    to_values: List[dict] = []

    merges: List[dict] = []
    group: List[dict] = []

    if "box" not in image_annot or "polyline" not in image_annot:
        logging.warning("skipping because no `box` nor `polyline` is found")
        return (
            basename,
            keep,
            boxes,
            lines,
            reading_order,
            to_captions,
            to_footnotes,
            merges,
            group,
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
            merges,
            group,
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
            merges,
            group,
        )

    for i, box in enumerate(boxes):
        boxes[i]["b"] = float(box["@ybr"])
        boxes[i]["t"] = float(box["@ytl"])
        boxes[i]["l"] = float(box["@xtl"])
        boxes[i]["r"] = float(box["@xbr"])

    assert boxes[i]["b"] > boxes[i]["t"]

    for i, line in enumerate(lines):

        # print(line)

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

        """
        for i in range(0, len(lines[i]["points"])):
            print(i, "\t", points[i], "\t", boxids[i])
        
        print(line["@label"], ": ", len(points), "\t", len(boxids))
        """

    for i, line in enumerate(lines):
        if line["@label"] == "reading_order":
            assert len(reading_order) == 0  # you can only have 1 reading order
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
    img_width: float,
    img_height: float,
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


def find_table_data(
    doc: DoclingDocument, prov: ProvenanceItem, iou_cutoff: float = 0.90
):

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


def get_page_imageref(page_no: int, true_doc: DoclingDocument) -> ImageRef:

    if page_no not in true_doc.pages:
        raise ValueError(f"{page_no} in not in the document")

    if not isinstance(true_doc.pages[page_no].image, ImageRef):
        raise ValueError(f"{page_no} in not in the document has no ImageRef")

    true_page_imageref: ImageRef = cast(ImageRef, true_doc.pages[page_no].image)

    return true_page_imageref


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
    """
    if true_doc.pages[page_no].image is None:
        logging.error("true_doc.pages[page_no].image is None, skipping ...")
        return

    true_page_imageref: ImageRef = true_doc.pages[page_no].image
    """

    true_page_imageref = get_page_imageref(page_no=page_no, true_doc=true_doc)

    next_provs = []
    for merge in merges:
        if len(merge["boxids"]) > 1 and merge["boxids"][0] == boxid:

            for l in range(1, len(merge["boxids"])):
                boxid_ = merge["boxids"][l]
                already_added.append(boxid_)

                label_, prov_, text_ = get_label_prov_and_text(
                    box=boxes[boxid_],
                    page_no=page_no,
                    # img_width=true_doc.pages[page_no].image.size.width,
                    img_width=true_page_imageref.size.width,
                    # img_height=true_doc.pages[page_no].image.size.height,
                    img_height=true_page_imageref.size.height,
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
    basename: str,
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
    true_page_imageref = get_page_imageref(page_no=page_no, true_doc=true_doc)

    for to_caption in to_captions:
        if to_caption["boxids"][0] == boxid:
            for l in range(1, len(to_caption["boxids"])):
                boxid_ = to_caption["boxids"][l]
                already_added.append(boxid_)

                caption_box = boxes[boxid_]

                label, prov, text = get_label_prov_and_text(
                    box=caption_box,
                    page_no=page_no,
                    img_width=true_page_imageref.size.width,
                    img_height=true_page_imageref.size.height,
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
    basename: str,
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
    true_page_imageref = get_page_imageref(page_no=page_no, true_doc=true_doc)

    for to_footnote in to_footnotes:
        if to_footnote["boxids"][0] == boxid:
            for l in range(1, len(to_footnote["boxids"])):
                boxid_ = to_footnote["boxids"][l]
                already_added.append(boxid_)

                footnote_box = boxes[boxid_]

                label, prov, text = get_label_prov_and_text(
                    box=footnote_box,
                    page_no=page_no,
                    img_width=true_page_imageref.size.width,
                    img_height=true_page_imageref.size.height,
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


def create_true_document(basename: str, annot: dict, desc: AnnotatedImage):

    # logging.info(f"creating ground-truth document for {basename}")
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

    # logging.info(f"analyzing {basename}")

    # ========== Original Groundtruth
    orig_file = desc.true_file
    assert os.path.exists(orig_file)

    orig_doc = DoclingDocument.load_from_json(filename=orig_file)
    # with open(orig_file, "r") as fr:
    #    orig_doc = DoclingDocument.model_validate_json(json.load(fr))

    # ========== Original Prediction (to pre-annotate)
    pred_file = desc.pred_file
    assert os.path.exists(str(pred_file))

    pred_doc = DoclingDocument.load_from_json(filename=pred_file)
    # with open(pred_file, "r") as fr:
    #    pred_doc = DoclingDocument.model_validate_json(json.load(fr))

    # ========== Original PDF page
    pdf_file: Path = desc.bin_file
    assert os.path.exists(pdf_file)

    # Init the parser to extract the text-cells
    parser = pdf_parser_v2(level="fatal")
    success = parser.load_document(key=basename, filename=str(pdf_file))

    parsed_pages = {}
    for i, page_no in enumerate(desc.page_nos):
        parsed_doc = parser.parse_pdf_from_key_on_page(key=basename, page=page_no - 1)
        parsed_pages[page_no] = parsed_doc["pages"][0]

    parser.unload_document(basename)

    # ========== Create Ground Truth document
    true_doc = DoclingDocument(name=f"{basename}")

    # Copy the page-images from the predicted pages
    for i, page_no in enumerate(desc.page_nos):

        # --- PDF
        assert len(parsed_doc["pages"]) == 1
        pdf_width = parsed_pages[page_no]["sanitized"]["dimension"]["width"]
        pdf_height = parsed_pages[page_no]["sanitized"]["dimension"]["height"]

        # --- PNG
        img_file = desc.page_img_files[i]

        page_image = Image.open(str(img_file))
        # page_image.show()

        img_width = page_image.width
        img_height = page_image.height

        if pred_doc.pages[page_no] is None:
            logging.error("Page item is None, skipping ...")
            continue

        pred_page_item = pred_doc.pages[page_no]

        pred_page_imageref = pred_page_item.image
        if pred_page_imageref is None:
            logging.error("Page ImageRef is None, skipping ...")
            continue

        assert pred_page_imageref.size.width == img_width
        assert pred_page_imageref.size.height == img_height

        image_ref = ImageRef(
            mimetype="image/png",
            # dpi=pred_doc.pages[page_no].image.dpi,
            dpi=pred_page_imageref.dpi,
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
    # logging.info(f"reading-oder from annotations: {reading_order}")

    already_added: List[int] = []
    for boxid in reading_order["boxids"]:
        # print("reading order => ", boxid, ": ", boxes[boxid])

        if boxid in already_added:
            logging.warning(f"{boxid} is already added: {already_added}")
            continue

        # FIXME for later ...
        page_no = 1

        if (page_no not in true_doc.pages) or (true_doc.pages[page_no] is None):
            logging.error("Page item is None, skipping ...")
            continue

        true_page_item = true_doc.pages[page_no]

        true_page_imageref = true_page_item.image
        if true_page_imageref is None:
            logging.error("Page ImageRef is None, skipping ...")
            continue

        true_page_pilimage = true_page_imageref.pil_image

        label, prov, text = get_label_prov_and_text(
            box=boxes[boxid],
            page_no=page_no,
            img_width=true_page_imageref.size.width,
            img_height=true_page_imageref.size.height,
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
            # true_doc.add_text(label=label, prov=prov, text=text)

            code_item = true_doc.add_code(text=text, prov=prov)

            true_doc, already_added = add_captions_to_item(
                basename=basename,
                to_captions=to_captions,
                item=code_item,
                page_no=page_no,
                boxid=boxid,
                boxes=boxes,
                already_added=already_added,
                true_doc=true_doc,
                parser=parser,
                parsed_page=parsed_pages[page_no],
            )

        elif label == DocItemLabel.FORM:
            graph = GraphData(cells=[], links=[])
            true_doc.add_form(graph=graph, prov=prov)

        elif label == DocItemLabel.KEY_VALUE_REGION:
            graph = GraphData(cells=[], links=[])
            true_doc.add_key_values(graph=graph, prov=prov)

        elif label in [DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX]:

            table_data = find_table_data(doc=orig_doc, prov=prov)

            table_item = true_doc.add_table(label=label, data=table_data, prov=prov)

            true_doc, already_added = add_captions_to_item(
                basename=basename,
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
                basename=basename,
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
                dpi=72,
                size=Size(width=crop_image.width, height=crop_image.height),
                uri=from_pil_to_base64uri(crop_image),
            )
            if true_doc.pages[page_no].image is not None:
                _ = true_doc.pages[page_no].image
                imgref.dpi = _.dpi

            picture_item = true_doc.add_picture(prov=prov, image=imgref)

            true_doc, already_added = add_captions_to_item(
                basename=basename,
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
                basename=basename,
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
    overview: AnnotationOverview,
    image_scale: float = 1.0,
) -> Iterator[Tuple[str, AnnotatedImage, Optional[DoclingDocument]]]:

    for annot_file in annotation_filenames:

        with open(str(annot_file), "r") as fr:
            xml_data = fr.read()

        # Convert XML to a Python dictionary
        annot_data = xmltodict.parse(xml_data)

        for image_annot in annot_data["annotations"]["image"]:

            basename = image_annot["@name"]
            # logging.info(basename)

            """
            if basename != "doc_5387a06d7e31d738c4bdb64b1936ac6fa09246b6a7e8506e1ee86691ff37155c_page_000001.png":
                continue
            """

            if basename not in overview.img_annotations:
                logging.warning(f"Skipping {basename}: not in overview_file")
                yield basename, overview.img_annotations[basename], None

            elif not contains_reading_order(image_annot):
                logging.warning(f"Skipping {basename}: no reading-order detected")
                yield basename, overview.img_annotations[basename], None

            else:
                true_doc = create_true_document(
                    basename=basename,
                    annot=image_annot,
                    desc=overview.img_annotations[basename],
                )
                yield basename, overview.img_annotations[basename], true_doc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create new evaluation dataset using CVAT annotation files."
    )

    parser.add_argument(
        "-i", "--input_dir", required=True, help="Path to the input directory"
    )

    args = parser.parse_args()
    return Path(args.input_dir)


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


def create_layout_dataset_from_annotations(
    benchmark_dirs: BenchMarkDirs, annot_files: List[Path]
):

    overview = AnnotationOverview.load_from_json(filename=benchmark_dirs.overview_file)

    # Create Converter
    image_scale = 2.0
    doc_converter = create_pdf_docling_converter(page_image_scale=image_scale)

    records = []
    for basename, desc, true_doc in tqdm(
        from_cvat_to_docling_document(
            annotation_filenames=annot_files,
            overview=overview,
        ),
        total=len(overview.img_annotations),
        ncols=128,
        desc="Creating documents from annotations",
    ):

        if true_doc is None:
            continue
        else:
            true_doc.save_as_json(
                filename=benchmark_dirs.json_anno_dir / f"{basename}.json"
            )

        """
        save_inspection_html(filename=str(html_comp_dir / f"{basename}.html"), doc = true_doc,
                             labels=TRUE_HTML_EXPORT_LABELS)
        """

        pdf_file = desc.bin_file

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
            vizname = benchmark_dirs.html_comp_dir / f"{basename}-clusters.html"
            # logging.info(f"creating visualization: {vizname}")

            save_comparison_html_with_clusters(
                filename=vizname,
                true_doc=true_doc,
                pred_doc=pred_doc,
                page_image=true_page_images[0],
                true_labels=TRUE_HTML_EXPORT_LABELS,
                pred_labels=PRED_HTML_EXPORT_LABELS,
            )

        record = {
            BenchMarkColumns.CONVERTER_TYPE: ConverterTypes.DOCLING,
            BenchMarkColumns.CONVERTER_VERSION: docling_version(),
            BenchMarkColumns.STATUS: str(conv_results.status),
            BenchMarkColumns.DOC_ID: str(basename),
            BenchMarkColumns.DOC_PATH: str(basename),
            BenchMarkColumns.DOC_HASH: get_binhash(get_binary(pdf_file)),
            BenchMarkColumns.GROUNDTRUTH: json.dumps(true_doc.export_to_dict()),
            BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES: true_page_images,
            BenchMarkColumns.GROUNDTRUTH_PICTURES: true_pictures,
            BenchMarkColumns.PREDICTION: json.dumps(pred_doc.export_to_dict()),
            BenchMarkColumns.PREDICTION_PAGE_IMAGES: pred_page_images,
            BenchMarkColumns.PREDICTION_PICTURES: pred_pictures,
            BenchMarkColumns.ORIGINAL: get_binary(pdf_file),
            BenchMarkColumns.MIMETYPE: "application/pdf",
            BenchMarkColumns.MODALITIES: [
                EvaluationModality.LAYOUT,
                EvaluationModality.READING_ORDER,
                EvaluationModality.CAPTIONING,
            ],
        }
        records.append(record)

    save_shard_to_disk(items=records, dataset_path=benchmark_dirs.dataset_test_dir)

    write_datasets_info(
        name="CVAT: end-to-end",
        output_dir=benchmark_dirs.dataset_dir,
        num_train_rows=0,
        num_test_rows=len(records),
    )


import zipfile


def unzip_files(zip_files, output_dir):
    """
    Unzips a list of zip files into a specified directory, avoiding filename collisions.

    Args:
        zip_files (list): List of paths to zip files to unzip.
        output_dir (str): Path to the directory where files will be unzipped.

    Returns:
        list: List of paths to all unzipped files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    unzipped_files = []

    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file, "r") as zf:
            for file_name in zf.namelist():
                # Resolve filename collisions
                original_file_name = file_name
                file_path = os.path.join(output_dir, file_name)
                base, ext = os.path.splitext(file_name)
                counter = 1
                while os.path.exists(file_path):
                    # Append a numeric suffix to resolve collisions
                    file_name = f"{base}_{counter}{ext}"
                    file_path = os.path.join(output_dir, file_name)
                    counter += 1

                # Extract file and add to the list
                with open(file_path, "wb") as f:
                    f.write(zf.read(original_file_name))
                unzipped_files.append(file_path)

    return unzipped_files


def get_annotation_files(benchmark_dirs):

    xml_files = []
    zip_files = sorted(glob.glob(str(benchmark_dirs.annotations_zip_dir / "*.zip")))

    if len(zip_files) > 0:
        logging.info(f"#-zips: {len(zip_files)}")

        xml_files = sorted(glob.glob(str(benchmark_dirs.annotations_xml_dir / "*.xml")))
        logging.info(f"#-xml: {len(xml_files)}")

        for xml_file in xml_files:
            os.remove(xml_file)

        xml_files = unzip_files(
            zip_files=zip_files, output_dir=benchmark_dirs.annotations_xml_dir
        )
    else:
        xml_files = sorted(glob.glob(str(benchmark_dirs.annotations_xml_dir / "*.xml")))

    logging.info(f"#-xml: {len(xml_files)}")
    return xml_files


def main():

    source_dir = parse_args()

    benchmark_dirs = BenchMarkDirs()
    benchmark_dirs.set_up_directory_structure(source=source_dir, target=source_dir)

    # Get all annotation files
    annot_files = get_annotation_files(benchmark_dirs)

    create_layout_dataset_from_annotations(
        benchmark_dirs=benchmark_dirs, annot_files=annot_files
    )


if __name__ == "__main__":
    main()
