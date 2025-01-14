import argparse
import copy
import glob
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import xmltodict
from datasets import Dataset, load_dataset
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import (
    DocItem,
    DoclingDocument,
    ImageRef,
    ImageRefMode,
    PageItem,
    PictureItem,
    ProvenanceItem,
    TableCell,
    TableData,
    TableItem,
)
from docling_core.types.doc.labels import DocItemLabel, GroupLabel
from docling_parse.pdf_parsers import pdf_parser_v2  # type: ignore[import]
from PIL import Image  # as PILImage
from tqdm import tqdm  # type: ignore

from docling_eval.benchmarks.constants import BenchMarkColumns
from docling_eval.benchmarks.utils import (
    draw_clusters_with_reading_order,
    save_inspection_html,
)
from docling_eval.docling.utils import (
    crop_bounding_box,
    from_pil_to_base64,
    from_pil_to_base64uri,
    insert_images,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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


def label_to_rgba_color(label: DocItemLabel):

    if label == DocItemLabel.TEXT:
        return (255, 255, 153)  # Light Yellow
    elif label == DocItemLabel.PARAGRAPH:
        return (255, 255, 153)  # Light Yellow
    elif label == DocItemLabel.REFERENCE:
        return (173, 216, 230)  # Light Yellow
    elif label == DocItemLabel.CAPTION:
        return (255, 204, 153)  # Light Orange
    elif label == DocItemLabel.LIST_ITEM:
        return (153, 153, 255)  # Light Purple
    elif label == DocItemLabel.FORMULA:
        return (192, 192, 192)  # Gray
    elif label == DocItemLabel.TABLE:
        return (255, 204, 204)  # Light Pink
    elif label == DocItemLabel.PICTURE:
        return (255, 204, 164)  # Light Beige
    elif label == DocItemLabel.SECTION_HEADER:
        return (255, 153, 153)  # Light Red
    elif label == DocItemLabel.PAGE_HEADER:
        return (204, 255, 204)  # Light Green
    elif label == DocItemLabel.PAGE_FOOTER:
        return (204, 255, 204)  # Light Green
    elif label == DocItemLabel.TITLE:
        return (255, 153, 153)  # Light Red (same as Section-Header)
    elif label == DocItemLabel.FOOTNOTE:
        return (200, 200, 255)  # Light Blue
    elif label == DocItemLabel.DOCUMENT_INDEX:
        return (220, 220, 220)  # Light Gray
    elif label == DocItemLabel.CODE:
        return (125, 125, 125)  # Gray
    elif label == DocItemLabel.CHECKBOX_SELECTED:
        return (255, 182, 193)  # Pale Green
    elif label == DocItemLabel.CHECKBOX_UNSELECTED:
        return (255, 182, 193)  # Light Pink
    elif label == DocItemLabel.FORM:
        return (200, 255, 255)  # Light Cyan
    elif label == DocItemLabel.KEY_VALUE_REGION:
        return (183, 65, 14)  # Rusty orange
    elif label == DocItemLabel.FORM:
        return (64, 64, 64)
    elif label == DocItemLabel.KEY_VALUE_REGION:
        return (255, 255, 0)
    elif label == DocItemLabel.PICTURE:
        return (150, 123, 182)
    else:
        logger.warning(f"no color for label: {label}")


def label_to_hex_color(label: DocItemLabel):

    r, g, b = label_to_rgba_color(label)
    return rgb_to_hex(r, g, b)


def line_to_rgba_color(line: str):

    if line == "reading_order":
        return (255, 0, 0)
    elif line == "next_text":
        return (255, 0, 255)
    elif line == "to_caption":
        return (0, 255, 0)
    elif line == "to_footnote":
        return (0, 255, 0)
    elif line == "to_value":
        return (0, 0, 255)


def line_to_hex_color(line: str):

    r, g, b = line_to_rgba_color(line)
    return rgb_to_hex(r, g, b)


def create_labels_for_cvat_in_xml():

    results.append("<labels>")
    for item in DocItemLabel:
        results.append("<label>")
        results.append(f"<name>{item.value}</name>")
        results.append(f"<color>{label_to_hex_color(item)}</color>")
        results.append(f"<type>rectangle</type>")
        results.append(f"<attributes></attributes>")
        results.append("</label>")

    for value in [
        "reading_order",
        "next_text",
        "to_caption",
        "to_footnote",
        "to_value",
    ]:
        results.append("<label>")
        results.append(f"<name>{value}</name>")
        results.append(f"<color>{line_to_hex_color(value)}</color>")
        results.append(f"<type>polyline</type>")
        results.append(f"<attributes></attributes>")
        results.append("</label>")

    results.append("</labels>")


def create_labels_for_cvat_in_json():

    results = []

    for item in DocItemLabel:
        results.append(
            {
                "name": item.value,
                "color": label_to_hex_color(item),
                "type": "rectangle",
                "attributes": [],
            }
        )

    """"
    for item in GroupItemLabel:
        results.append({
            "name": item.value,
            "color": label_to_hex_color(item),
            "type": "rectangle",
            "attributes": []
        })
    """

    for value in [
        "reading_order",
        "next_text",
        "to_caption",
        "to_footnote",
        "to_value",
    ]:
        results.append(
            {
                "name": value,
                "color": line_to_hex_color(value),
                "type": "polyline",
                "attributes": [],
            }
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
        print("point: ", point)
        for i, box in enumerate(boxes):
            x = point[0]
            y = point[1]

            l = box["l"]
            r = box["r"]
            t = box["t"]
            b = box["b"]

            print(
                f"bbox: {l:.3f}, {r:.3f}, ({(l<x) and (x<r)}), {t:.3f}, {b:.3f}, ({(t<y) and (y<b)})"
            )

    return index, boxes[index]


def parse_annotations(image_annot: dict):

    doc_name = image_annot["@name"]

    keep = False
    boxes = []
    lines = []
    reading_order = None
    to_captions = []
    to_footnotes = []
    next_text = []

    if "box" not in image_annot or "polyline" not in image_annot:
        return (
            doc_name,
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
            doc_name,
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
            doc_name,
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
        elif line["@label"] == "next_text":
            next_text.append(line)

    return (
        doc_name,
        keep,
        boxes,
        lines,
        reading_order,
        to_captions,
        to_footnotes,
        next_text,
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
                """
                if prov_.page_no==prov.page_no and \
                   abs(prov_.bbox.l-prov.bbox.l)<eps and \
                   abs(prov_.bbox.r-prov.bbox.r)<eps and \
                   abs(prov_.bbox.t-prov.bbox.t)<eps and \
                   abs(prov_.bbox.b-prov.bbox.b)<eps:
                    logging.info(" => found table-data!")
                    return item.data
                """

    logging.warning(" => missing table-data!")

    table_data = TableData(num_rows=-1, num_cols=-1, table_cells=[])
    return table_data


"""
def crop_text_from_pdf(pdf_height:float, bbox:BoundingBox, parser:pdf_parser_v2, parsed_page:dict):
    
    assert pdf_height-prov.bbox.b<pdf_height-prov.bbox.t # verify we are topleft                            

    pdf_text = parser.sanitize_cells_in_bbox(
        page=parsed_page,
        bbox=[bbox.l, pdf_height-bbox.b, bbox.r, pdf_height-bbox.t],
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
        
    return text
"""


def from_cvat_to_docling_document(
    annotation_filenames: List[Path],
    img_to_pdf_file: Path,
    imgs_dir: Path,
    pdfs_dir: Path,
    image_scale: float = 1.0,
):

    with open(str(img_to_pdf_file), "r") as fr:
        img_to_pdf = json.load(fr)

    for annot_file in annotation_filenames:

        with open(str(annot_file), "r") as fr:
            xml_data = fr.read()

        # Convert XML to a Python dictionary
        annot_data = xmltodict.parse(xml_data)

        for image_annot in annot_data["annotations"]["image"]:

            (
                doc_name,
                keep,
                boxes,
                lines,
                reading_order,
                to_captions,
                to_footnotes,
                next_texts,
            ) = parse_annotations(image_annot)
            logging.info(f"analyzing {doc_name}")

            if not keep:
                continue

            # Original Groundtruth
            orig_file = Path(img_to_pdf[doc_name]["true_file"])
            assert os.path.exists(orig_file)

            with open(orig_file, "r") as fr:
                orig_doc = DoclingDocument.model_validate_json(json.load(fr))

            assert len(orig_doc.pages) == 1

            # Page image
            img_file = imgs_dir / doc_name

            page_image = Image.open(str(img_file))

            img_width = page_image.width
            img_height = page_image.height

            # PDF page
            pdf_page_no: int = img_to_pdf[doc_name]["page_no"]

            pdf_file: Path = Path(img_to_pdf[doc_name]["pdf_file"])
            assert os.path.exists(pdf_file)

            # logging.info(f"img: {img_file} => pdf: {pdf_file}")

            # Init the parser to extract the text-cells
            parser = pdf_parser_v2(level="fatal")
            success = parser.load_document(key=doc_name, filename=str(pdf_file))
            parsed_doc = parser.parse_pdf_from_key_on_page(
                key=doc_name, page=pdf_page_no - 1
            )

            assert len(parsed_doc["pages"]) == 1
            pdf_width = parsed_doc["pages"][0]["sanitized"]["dimension"]["width"]
            pdf_height = parsed_doc["pages"][0]["sanitized"]["dimension"]["height"]

            # Create Ground Truth document
            true_doc = DoclingDocument(name=f"{doc_name}")

            page_no = 1

            image_ref = ImageRef(
                mimetype="image/png",
                dpi=round(72 * image_scale),
                size=Size(width=float(img_width), height=float(img_height)),
                uri=from_pil_to_base64uri(page_image),
            )
            page_item = PageItem(
                page_no=page_no,
                size=Size(width=float(pdf_width), height=float(pdf_height)),
                image=image_ref,
            )
            true_doc.pages[page_no] = page_item

            to_be_skipped = []

            for boxid in reading_order["boxids"]:

                if boxid in to_be_skipped:
                    logging.warning(f"{boxid} is already added: {to_be_skipped}")
                    continue

                """
                box = boxes[boxid]
                label = DocItemLabel(box["@label"])
                
                prov, bbox = create_prov(box=box, page_no=page_no,
                                         img_width=page_image.width,
                                         img_height=page_image.height,
                                         pdf_width=pdf_width,
                                         pdf_height=pdf_height)

                text = crop_text_from_pdf(pdf_height=pdf_height, bbox, parser=parser, parsed_page=parsed_page)
                prov.charspan = (0, len(text))
                """

                label, prov, text = get_label_prov_and_text(
                    box=boxes[boxid],
                    page_no=page_no,
                    img_width=img_width,
                    img_height=img_height,
                    pdf_width=pdf_width,
                    pdf_height=pdf_height,
                    parser=parser,
                    parsed_page=parsed_doc["pages"][0],
                )

                next_provs = []
                for next_text in next_texts:
                    if len(next_text["boxids"]) > 1 and next_text["boxids"][0] == boxid:

                        for l in range(1, len(next_text["boxids"])):
                            boxid_ = next_text["boxids"][l]
                            to_be_skipped.append(boxid_)

                            """
                            box_ = boxes[boxid_]
                            assert label==DocItemLabel(box_["@label"])
                            
                            prov_, bbox_ = create_prov(box=box_, page_no=page_no,
                                                       img_width=page_image.width,
                                                       img_height=page_image.height,
                                                       pdf_width=pdf_width,
                                                       pdf_height=pdf_height)

                            text_ = crop_text_from_pdf(pdf_height=pdf_height, bbox, parser=parser, parsed_page=parsed_page)
                            """

                            label_, prov_, text_ = get_label_prov_and_text(
                                box=boxes[boxid_],
                                page_no=page_no,
                                img_width=img_width,
                                img_height=img_height,
                                pdf_width=pdf_width,
                                pdf_height=pdf_height,
                                parser=parser,
                                parsed_page=parsed_doc["pages"][0],
                            )

                            prov_.charspan = (len(text) + 1, len(text_))

                            text = text + " " + text_

                            next_provs.append(prov_)
                            # texts.append(text_)

                """
                assert pdf_height-prov.bbox.b<pdf_height-prov.bbox.t # verify we are topleft
                
                pdf_text = parser.sanitize_cells_in_bbox(
                    page=parsed_doc["pages"][page_no-1],
                    bbox=[prov.bbox.l, pdf_height-prov.bbox.b, prov.bbox.r, pdf_height-prov.bbox.t],
                    cell_overlap=0.9,
                    horizontal_cell_tolerance=1.0,
                    enforce_same_font=False,
                    space_width_factor_for_merge=1.5,
                    space_width_factor_for_merge_with_space=0.33,
                )
                #print(f"pdf_text: {json.dumps(pdf_text)}")

                text = ""
                try:
                    texts = []
                    for row in pdf_text["data"]:
                        texts.append(row[pdf_text["header"].index("text")])
                                
                    text = " ".join(texts)
                except:
                    text = ""
                """

                if label == DocItemLabel.TEXT:
                    current_item = true_doc.add_text(label=label, prov=prov, text=text)

                    for next_prov in next_provs:
                        current_item.prov.append(next_prov)

                elif label == DocItemLabel.PARAGRAPH:
                    true_doc.add_text(label=label, prov=prov, text=text)

                elif label == DocItemLabel.REFERENCE:
                    true_doc.add_text(label=label, prov=prov, text=text)

                elif label == DocItemLabel.CAPTION:
                    pass

                elif label == DocItemLabel.LIST_ITEM:
                    true_doc.add_list_item(prov=prov, text=text)

                elif label == DocItemLabel.FORMULA:
                    true_doc.add_text(label=label, prov=prov, text=text)

                elif label == DocItemLabel.DOCUMENT_INDEX:

                    table_data = find_table_data(doc=orig_doc, prov=prov)

                    true_doc.add_table(
                        label=DocItemLabel.DOCUMENT_INDEX, data=table_data, prov=prov
                    )

                elif label == DocItemLabel.TABLE:

                    table_data = find_table_data(doc=orig_doc, prov=prov)

                    # table_data = TableData(num_rows=-1, num_cols=-1, table_cells=[])
                    table_item = true_doc.add_table(
                        label=DocItemLabel.TABLE, data=table_data, prov=prov
                    )

                    for to_caption in to_captions:
                        if to_caption["boxids"][0] == boxid:
                            for l in range(1, len(to_caption["boxids"])):
                                boxid_ = to_caption["boxids"][l]
                                to_be_skipped.append(boxid_)

                                caption_box = boxes[boxid_]

                                label, prov, text = get_label_prov_and_text(
                                    box=caption_box,
                                    page_no=page_no,
                                    img_width=img_width,
                                    img_height=img_height,
                                    pdf_width=pdf_width,
                                    pdf_height=pdf_height,
                                    parser=parser,
                                    parsed_page=parsed_doc["pages"][0],
                                )

                                caption_ref = true_doc.add_text(
                                    label=DocItemLabel.CAPTION, prov=prov, text=text
                                )
                                table_item.captions.append(caption_ref.get_ref())

                                if label != DocItemLabel.CAPTION:
                                    logging.error(
                                        f"{label}!=DocItemLabel.CAPTION for {doc_name}"
                                    )

                    for to_footnote in to_footnotes:
                        if to_footnote["boxids"][0] == boxid:
                            for l in range(1, len(to_footnote["boxids"])):
                                boxid_ = to_footnote["boxids"][l]
                                to_be_skipped.append(boxid_)

                                footnote_box = boxes[boxid_]

                                label, prov, text = get_label_prov_and_text(
                                    box=footnote_box,
                                    page_no=page_no,
                                    img_width=img_width,
                                    img_height=img_height,
                                    pdf_width=pdf_width,
                                    pdf_height=pdf_height,
                                    parser=parser,
                                    parsed_page=parsed_doc["pages"][0],
                                )

                                footnote_ref = true_doc.add_text(
                                    label=DocItemLabel.FOOTNOTE, prov=prov, text=text
                                )
                                table_item.footnotes.append(footnote_ref.get_ref())

                                if label != DocItemLabel.FOOTNOTE:
                                    logging.error(
                                        f"{label}!=DocItemLabel.FOOTNOTE for {doc_name}"
                                    )

                elif label == DocItemLabel.PICTURE:

                    pic = crop_bounding_box(
                        page_image=page_image,
                        page=true_doc.pages[page_no],
                        bbox=prov.bbox,
                    )

                    uri = from_pil_to_base64uri(pic)

                    imgref = ImageRef(
                        mimetype="image/png",
                        dpi=72 * image_scale,
                        size=Size(width=pic.width, height=pic.height),
                        uri=uri,
                    )

                    picture_item = true_doc.add_picture(prov=prov, image=imgref)

                    for to_caption in to_captions:
                        if to_caption["boxids"][0] == boxid:
                            for l in range(1, len(to_caption["boxids"])):
                                boxid_ = to_caption["boxids"][l]
                                to_be_skipped.append(boxid_)

                                caption_box = boxes[boxid_]

                                label, prov, text = get_label_prov_and_text(
                                    box=caption_box,
                                    page_no=page_no,
                                    img_width=img_width,
                                    img_height=img_height,
                                    pdf_width=pdf_width,
                                    pdf_height=pdf_height,
                                    parser=parser,
                                    parsed_page=parsed_doc["pages"][0],
                                )

                                caption_ref = true_doc.add_text(
                                    label=DocItemLabel.CAPTION, prov=prov, text=text
                                )
                                picture_item.captions.append(caption_ref.get_ref())

                                if label != DocItemLabel.CAPTION:
                                    logging.error(
                                        f"{label}!=DocItemLabel.CAPTION for {doc_name}"
                                    )

                    for to_footnote in to_footnotes:
                        if to_footnote["boxids"][0] == boxid:
                            for l in range(1, len(to_footnote["boxids"])):
                                boxid_ = to_footnote["boxids"][l]
                                to_be_skipped.append(boxid_)

                                footnote_box = boxes[boxid_]

                                label, prov, text = get_label_prov_and_text(
                                    box=footnote_box,
                                    page_no=page_no,
                                    img_width=img_width,
                                    img_height=img_height,
                                    pdf_width=pdf_width,
                                    pdf_height=pdf_height,
                                    parser=parser,
                                    parsed_page=parsed_doc["pages"][0],
                                )

                                footnote_ref = true_doc.add_text(
                                    label=DocItemLabel.FOOTNOTE, prov=prov, text=text
                                )
                                picture_item.footnotes.append(footnote_ref.get_ref())

                                if label != DocItemLabel.FOOTNOTE:
                                    logging.error(
                                        f"{label}!=DocItemLabel.FOOTNOTE for {doc_name}"
                                    )

                elif label == DocItemLabel.SECTION_HEADER:
                    true_doc.add_text(label=label, prov=prov, text=text)

                elif label == DocItemLabel.PAGE_HEADER:
                    true_doc.add_text(label=label, prov=prov, text=text)

                elif label == DocItemLabel.PAGE_FOOTER:
                    true_doc.add_text(label=label, prov=prov, text=text)

                elif label == DocItemLabel.TITLE:
                    true_doc.add_text(label=label, prov=prov, text=text)

                elif label == DocItemLabel.FOOTNOTE:
                    true_doc.add_text(label=label, prov=prov, text=text)

                elif label == DocItemLabel.CODE:
                    true_doc.add_text(label=label, prov=prov, text=text)

                elif label == DocItemLabel.CHECKBOX_SELECTED:
                    true_doc.add_text(label=label, prov=prov, text=text)

                elif label == DocItemLabel.CHECKBOX_UNSELECTED:
                    true_doc.add_text(label=label, prov=prov, text=text)

                elif label == DocItemLabel.FORM:
                    true_doc.add_text(label=label, prov=prov, text=text)

                elif label == DocItemLabel.KEY_VALUE_REGION:
                    true_doc.add_text(label=label, prov=prov, text=text)

                elif label == DocItemLabel.FORM:
                    true_doc.add_text(label=label, prov=prov, text=text)

                elif label == DocItemLabel.KEY_VALUE_REGION:
                    true_doc.add_text(label=label, prov=prov, text=text)

                else:
                    logging.error(f"unknown label={label}")

            yield doc_name, true_doc


def create_cvat_annotations_file(
    docs: List[DoclingDocument], overview: List[dict], output_dir: Path
):

    assert len(docs) == len(overview)

    output_page_dir = output_dir / "page_images"

    for _ in [output_dir, output_page_dir]:
        os.makedirs(_, exist_ok=True)

    results = []

    results.append('<?xml version="1.0" encoding="utf-8"?>')
    results.append("<annotations>")

    img_to_doc = {}

    img_id = 0
    for doc_id, doc in enumerate(docs):

        doc_overview = overview[doc_id]

        doc_name = doc.name

        page_images = []
        page_fnames = []
        for j, page in doc.pages.items():
            filename = f"doc_{doc_name}_page_{j:06}.png"

            img_file = str(output_page_dir / filename)

            img_to_doc[filename] = {
                "img_id": img_id,
                "img_file": img_file,
                "pdf_file": doc_overview["pdf_file"],
                "true_file": doc_overview["true_file"],
                "pred_file": doc_overview["pred_file"],
                "page_no": j,
                "page_ind": j - 1,
            }

            page_image = page.image.pil_image
            page_image.save(str(output_page_dir / filename))

            page_images.append(page_image)
            page_fnames.append(filename)

        page_bboxes = {i: [] for i, fname in enumerate(page_fnames)}
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

            for bbox_id, page_bbox in enumerate(page_bboxes[page_no]):
                label = page_bbox["label"]
                l = round(page_bbox["l"])
                r = round(page_bbox["r"])
                t = round(page_bbox["t"])
                b = round(page_bbox["b"])
                results.append(
                    f'<box label="{label}" source="docling" occluded="0" xtl="{l}" ytl="{t}" xbr="{r}" ybr="{b}" z_order="{bbox_id}"></box>'
                )

            results.append("</image>")

    results.append("</annotations>")

    with open(str(output_dir / "pre-annotations.xml"), "w") as fw:
        fw.write("\n".join(results))

    with open(str(output_dir / "overview_map.json"), "w") as fw:
        fw.write(json.dumps(img_to_doc, indent=2))


def export_to_preannotations():

    output_dir = Path("./benchmarks/DPBench-dataset/preannotations")

    imgs_dir = output_dir / "page_images"
    pdfs_dir = output_dir / "pdfs"
    json_true_dir = output_dir / "json-groundtruth"
    json_pred_dir = output_dir / "json-predictions"
    json_anno_dir = output_dir / "json-annotations"

    for _ in [
        output_dir,
        imgs_dir,
        pdfs_dir,
        json_true_dir,
        json_pred_dir,
        json_anno_dir,
    ]:
        os.makedirs(_, exist_ok=True)

    benchmark_path = Path("./benchmarks/DPBench-dataset/layout/test")

    test_files = sorted(glob.glob(str(benchmark_path / "*.parquet")))
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

    assert len(docs) == len(overview)

    create_cvat_annotations_file(docs=docs, overview=overview, output_dir=output_dir)


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


def import_from_cvat():

    output_dir = Path("./benchmarks/DPBench-dataset/preannotations/")

    annots = [output_dir / "annotations.xml"]
    imgs_dir = output_dir / "page_images"
    pdfs_dir = output_dir / "pdfs"

    json_anno_dir = output_dir / "json-annotations"
    html_anno_dir = output_dir / "html-annotations"
    html_inspections_dir = output_dir / "html-annotations-inspections"

    img_to_pdf_file = output_dir / "overview_map.json"

    for _ in [
        output_dir,
        imgs_dir,
        pdfs_dir,
        json_anno_dir,
        html_anno_dir,
        html_inspections_dir,
    ]:
        os.makedirs(_, exist_ok=True)

    for doc_name, true_doc in from_cvat_to_docling_document(
        annots, img_to_pdf_file=img_to_pdf_file, pdfs_dir=pdfs_dir, imgs_dir=imgs_dir
    ):

        """
        for item, level in true_doc.iterate_items():
            if isinstance(item, DocItem): # and item.label in labels:
                for prov in item.prov:
                    print(item.label, " => ", prov.bbox)
        """

        """
        print(" ============================================ ")
        
        md = true_doc.export_to_markdown()
        print(md)
        
        res = draw_clusters_with_reading_order(doc=true_doc,
                                               page_image=true_page_images[0],
                                               labels=TRUE_HTML_EXPORT_LABELS)
        res.show()

        print(" ============================================ ")
        """

        true_doc.save_as_html(
            filename=html_anno_dir / f"{doc_name}.html",
            image_mode=ImageRefMode.EMBEDDED,
        )

        save_inspection_html(
            filename=str(html_inspections_dir / f"{doc_name}.html"),
            doc=true_doc,
            labels=TRUE_HTML_EXPORT_LABELS,
        )

    logging.info(f"dumped all output here: {html_inspections_dir}")


def parse_arguments():
    """Parse arguments for CVAT annotations."""

    parser = argparse.ArgumentParser(
        description="Process DP-Bench benchmark from directory into HF dataset."
    )


def main():
    # export_to_preannotations()

    import_from_cvat()


if __name__ == "__main__":
    main()
