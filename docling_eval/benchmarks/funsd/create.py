import io
import itertools
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types import DoclingDocument
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    GroupLabel,
    ImageRef,
    PageItem,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)
from docling_core.types.doc.document import GraphCell, GraphData, GraphLink
from docling_core.types.doc.labels import GraphCellLabel, GraphLinkLabel
from docling_core.types.doc.tokens import TableToken
from docling_core.types.io import DocumentStream
from PIL import Image
from tqdm import tqdm  # type: ignore

from datasets import load_from_disk
from docling_eval.benchmarks.constants import BenchMarkColumns
from docling_eval.benchmarks.doclaynet_v1.create import (
    PRED_HTML_EXPORT_LABELS,
    TRUE_HTML_EXPORT_LABELS,
)
from docling_eval.benchmarks.utils import (
    save_comparison_html_with_clusters,
    write_datasets_info,
)
from docling_eval.docling.conversion import create_image_converter
from docling_eval.docling.utils import (
    crop_bounding_box,
    docling_version,
    extract_images,
    from_pil_to_base64uri,
    save_shard_to_disk,
)

SHARD_SIZE = 1000


def parse_texts(texts, tokens):
    split_word = TableToken.OTSL_NL.value
    split_row_tokens = [
        list(y) for x, y in itertools.groupby(tokens, lambda z: z == split_word) if not x
    ]
    table_cells = []
    r_idx = 0
    c_idx = 0

    def count_right(tokens, c_idx, r_idx, which_tokens):
        span = 0
        c_idx_iter = c_idx
        while tokens[r_idx][c_idx_iter] in which_tokens:
            c_idx_iter += 1
            span += 1
            if c_idx_iter >= len(tokens[r_idx]):
                return span
        return span

    def count_down(tokens, c_idx, r_idx, which_tokens):
        span = 0
        r_idx_iter = r_idx
        while tokens[r_idx_iter][c_idx] in which_tokens:
            r_idx_iter += 1
            span += 1
            if r_idx_iter >= len(tokens):
                return span
        return span

    for i, text in enumerate(texts):
        cell_text = ""
        if text in [
            TableToken.OTSL_FCEL.value,
            TableToken.OTSL_ECEL.value,
            TableToken.OTSL_CHED.value,
            TableToken.OTSL_RHED.value,
            TableToken.OTSL_SROW.value,
        ]:
            row_span = 1
            col_span = 1
            right_offset = 1
            if text != TableToken.OTSL_ECEL.value:
                cell_text = texts[i + 1]
                right_offset = 2

            # Check next element(s) for lcel / ucel / xcel, set properly row_span, col_span
            next_right_cell = texts[i + right_offset]

            next_bottom_cell = ""
            if r_idx + 1 < len(split_row_tokens):
                next_bottom_cell = split_row_tokens[r_idx + 1][c_idx]

            if next_right_cell in [
                TableToken.OTSL_LCEL.value,
                TableToken.OTSL_XCEL.value,
            ]:
                # we have horisontal spanning cell or 2d spanning cell
                col_span += count_right(
                    split_row_tokens,
                    c_idx + 1,
                    r_idx,
                    [TableToken.OTSL_LCEL.value, TableToken.OTSL_XCEL.value],
                )
            if next_bottom_cell in [
                TableToken.OTSL_UCEL.value,
                TableToken.OTSL_XCEL.value,
            ]:
                # we have a vertical spanning cell or 2d spanning cell
                row_span += count_down(
                    split_row_tokens,
                    c_idx,
                    r_idx + 1,
                    [TableToken.OTSL_UCEL.value, TableToken.OTSL_XCEL.value],
                )

            table_cells.append(
                TableCell(
                    text=cell_text.strip(),
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=r_idx,
                    end_row_offset_idx=r_idx + row_span,
                    start_col_offset_idx=c_idx,
                    end_col_offset_idx=c_idx + col_span,
                )
            )
        if text in [
            TableToken.OTSL_FCEL.value,
            TableToken.OTSL_ECEL.value,
            TableToken.OTSL_CHED.value,
            TableToken.OTSL_RHED.value,
            TableToken.OTSL_SROW.value,
            TableToken.OTSL_LCEL.value,
            TableToken.OTSL_UCEL.value,
            TableToken.OTSL_XCEL.value,
        ]:
            c_idx += 1
        if text == TableToken.OTSL_NL.value:
            r_idx += 1
            c_idx = 0
    return table_cells, split_row_tokens


def extract_tokens_and_text(s: str):
    # Pattern to match anything enclosed by < > (including the angle brackets themselves)
    pattern = r"(<[^>]+>)"
    # Find all tokens (e.g. "<otsl>", "<loc_140>", etc.)
    tokens = re.findall(pattern, s)
    # Remove any tokens that start with "<loc_"
    tokens = [
        token
        for token in tokens
        if not (token.startswith("<loc_") or token in ["<otsl>", "</otsl>"])
    ]
    # Split the string by those tokens to get the in-between text
    text_parts = re.split(pattern, s)
    text_parts = [
        token
        for token in text_parts
        if not (token.startswith("<loc_") or token in ["<otsl>", "</otsl>"])
    ]
    # Remove any empty or purely whitespace strings from text_parts
    text_parts = [part for part in text_parts if part.strip()]

    return tokens, text_parts


def parse_table_content(otsl_content: str) -> TableData:
    tokens, mixed_texts = extract_tokens_and_text(otsl_content)
    table_cells, split_row_tokens = parse_texts(mixed_texts, tokens)

    return TableData(
        num_rows=len(split_row_tokens),
        num_cols=(max(len(row) for row in split_row_tokens) if split_row_tokens else 0),
        table_cells=table_cells,
    )


def update(true_doc, current_list, img, label, segment, bb):
    bbox = BoundingBox.from_tuple(tuple(bb), CoordOrigin.TOPLEFT).to_bottom_left_origin(
        page_height=true_doc.pages[1].size.height
    )
    prov = ProvenanceItem(page_no=1, bbox=bbox, charspan=(0, len(segment["text"])))
    img_elem = crop_bounding_box(page_image=img, page=true_doc.pages[1], bbox=bbox)
    if label == DocItemLabel.PICTURE:
        current_list = None
        try:
            uri = from_pil_to_base64uri(img_elem)
            imgref = ImageRef(
                mimetype="image/png",
                dpi=72,
                size=Size(width=img_elem.width, height=img_elem.height),
                uri=uri,
            )
        except Exception as e:
            print(
                "Warning: failed to resolve image uri for segment {} of doc {}. Caught exception is {}:{}. Setting null ImageRef".format(
                    str(segment), str(true_doc.name), type(e).__name__, e
                )
            )
            imgref = None

        true_doc.add_picture(prov=prov, image=imgref)
    elif label in [DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX]:
        current_list = None
        if segment["data"] is not None:
            otsl_str = "".join(segment["data"]["otsl_seq"])
            tbl_data = parse_table_content(otsl_str)
            true_doc.add_table(data=tbl_data, prov=prov, label=label)
    elif label in [DocItemLabel.FORM, DocItemLabel.KEY_VALUE_REGION]:
        group_label = GroupLabel.UNSPECIFIED
        if label == DocItemLabel.FORM:
            group_label = GroupLabel.FORM_AREA
        elif label == DocItemLabel.KEY_VALUE_REGION:
            group_label = GroupLabel.KEY_VALUE_AREA
        true_doc.add_group(label=group_label)
    elif label == DocItemLabel.LIST_ITEM:
        if current_list is None:
            current_list = true_doc.add_group(label=GroupLabel.LIST, name="list")

        true_doc.add_list_item(
            text=segment["text"],
            enumerated=False,
            prov=prov,
            parent=current_list,
        )
    elif label == DocItemLabel.SECTION_HEADER:
        current_list = None
        true_doc.add_heading(text=segment["text"], prov=prov)
    else:
        current_list = None
        true_doc.add_text(label=label, text=segment["text"], prov=prov)


def convert_bbox(bbox_data) -> BoundingBox:
    if isinstance(bbox_data, list) and len(bbox_data) == 4:
        return BoundingBox(l=bbox_data[0], t=bbox_data[1], r=bbox_data[2], b=bbox_data[3])
    elif isinstance(bbox_data, BoundingBox):
        return bbox_data
    else:
        raise ValueError(
            "Invalid bounding box data; expected a list of four numbers or a BoundingBox instance."
        )


def create_graph_cell(cell_data: Dict, label: GraphCellLabel) -> GraphCell:
    bbox_instance = None
    if "bbox" in cell_data and cell_data["bbox"] is not None:
        bbox_instance = convert_bbox(cell_data["bbox"])

    return GraphCell(
        cell_id=cell_data["cell_id"],
        text=cell_data["text"],
        orig=cell_data.get("orig", cell_data["text"]),
        bbox=bbox_instance,
        label=label,
    )


def create_graph_link(
    key_cell: GraphCell,
    value_cell: GraphCell,
    label: GraphLinkLabel = GraphLinkLabel.TO_VALUE,
) -> GraphLink:
    return GraphLink(
        source_cell_id=key_cell.cell_id,
        target_cell_id=value_cell.cell_id,
        label=label,
    )


def get_overall_bbox(
    links: List[GraphLink], cell_dict: Dict[int, GraphCell]
) -> Optional[BoundingBox]:
    """
    Compute the overall bounding box (min_x, min_y, max_x, max_y)
    from all cell ids found in the links using element_dict.
    """
    all_bboxes = []  # type: List[BoundingBox]
    for link in links:
        src_prov = cell_dict[link.source_cell_id].prov
        tgt_prov = cell_dict[link.target_cell_id].prov
        if src_prov is not None:
            all_bboxes.append(src_prov.bbox)
        if tgt_prov is not None:
            all_bboxes.append(tgt_prov.bbox)

    if len(all_bboxes) == 0:
        return None
    bbox_instance = BoundingBox.union(all_bboxes)
    return bbox_instance


# creation of K/V pairs
def create_kv_pairs(data):
    link_pairs = []
    seg_with_id = {}
    bbox_with_id = {}

    _ids = data["annotation_ids"]
    bboxes = data["boxes"]
    segments = data["segments"]
    links = data["links"]

    # str to integer id mapping
    int_ids = {id: i for i, id in enumerate(_ids)}

    for i, seg in enumerate(segments):
        seg_with_id[_ids[i]] = seg
        bbox_with_id[_ids[i]] = bboxes[i]

    for i, segment in enumerate(segments):
        if links[i] is not None and links[i] in seg_with_id:
            link_pairs.append(
                {
                    "value": {
                        "cell_id": int_ids[_ids[i]],
                        "bbox": bboxes[i],  # or segment["bbox"]
                        "text": segment["text"],
                        "label": GraphCellLabel.VALUE,
                    },
                    "key": {
                        "cell_id": int_ids[links[i]],
                        "bbox": bbox_with_id[links[i]],  # or seg_with_id[links[i]]["bbox"]
                        "text": seg_with_id[links[i]]["text"],
                        "label": GraphCellLabel.KEY,
                    },
                }
            )
    return link_pairs


def populate_key_value_item_from_funsd(doc: DoclingDocument, funsd_data: dict) -> DoclingDocument:
    """
    example structure given in FUNSD JSON data:

      {
         "form": [
             {
                "id": 0,
                "text": "Registration No.",
                "box": [94, 169, 191, 186],
                "linking": [[0, 1]],
                "label": "question",
                "words": [ ... ] # for our case it is too fine-grained, so we ignore it
             },
             {
                "id": 1,
                "text": "533",
                "box": [209, 169, 236, 182],
                "linking": [[0, 1]],
                "label": "answer",
                "words": [ ... ] # for our case it is too fine-grained, so we ignore it
             }
         ]
      }

    This function is to populate the key_value_items for each FUNSD JSON annotation.
    """
    if "form" not in funsd_data:
        raise ValueError("Invalid FUNSD data: missing 'form' key.")

    form_items = funsd_data["form"]

    cell_by_id = {}
    for item in form_items:
        cell_id = item["id"]
        # Use the text as both the sanitized and original text (or adjust if needed).
        cell_text = item.get("text", "")

        bbox_instance = None
        if item.get("box") is not None:
            bbox_instance = convert_bbox(item.get("box"))
            cell_prov = ProvenanceItem(
                page_no=doc.pages[1].page_no,
                charspan=(0, 0),
                bbox=bbox_instance,
            )
        else:
            cell_prov = None

        cell = GraphCell(
            cell_id=cell_id,
            text=cell_text,
            orig=cell_text,
            prov=cell_prov,
            label=GraphCellLabel.KEY,  # later to be updated by label_cells
        )
        cell_by_id[cell_id] = cell

    # unique linking pairs
    linking_set = set()
    for item in form_items:
        linking = item.get("linking", [])
        for pair in linking:
            if isinstance(pair, list) and len(pair) == 2:
                linking_set.add(tuple(pair))  # (source_id, target_id)

    # creation of links using the cell mapping.
    links = []
    for src, tgt in linking_set:
        if src in cell_by_id and tgt in cell_by_id:
            # later to be updated by label_cells
            cell_by_id[tgt].label = GraphCellLabel.VALUE
            kv_link = create_graph_link(cell_by_id[src], cell_by_id[tgt])
            links.append(kv_link)

    cells = list(cell_by_id.values())

    overal_bbox = get_overall_bbox(links, cell_dict={cell.cell_id: cell for cell in cells})

    if overal_bbox is not None:
        prov = ProvenanceItem(
            page_no=doc.pages[1].page_no,
            charspan=(0, 0),
            bbox=overal_bbox,
        )
    else:
        prov = None

    graph = GraphData(cells=cells, links=links)

    doc.add_key_value_item(graph=graph, prov=prov)

    return doc


def create_funsd_dataset(
    input_dir: Path,
    output_dir: Path,
    splits: List[str] = ["train", "test"],
    max_items: int = -1,
):
    doc_converter = create_image_converter(do_ocr=True, ocr_lang=["en"])

    num_train_rows = 0
    num_test_rows = 0
    for split in splits:
        if split == "train":
            image_dir = input_dir / "training_data" / "images"
        elif split == "test":
            image_dir = input_dir / "testing_data" / "images"

        split_dir = output_dir / split

        images = list(image_dir.glob("*.png"))

        if max_items > 0:
            random.seed(42)  # for reproducibility
            images = random.sample(images, max_items)

        if split == "train":
            num_train_rows = len(images)
        elif split == "test":
            num_test_rows = len(images)

        os.makedirs(split_dir, exist_ok=True)
        records = []
        count = 0
        for img_path in tqdm(images, total=len(images)):
            img = Image.open(img_path)
            data_path = (
                img_path.parent.parent / "annotations" / img_path.name.replace(".png", ".json")
            )
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            with io.BytesIO() as img_byte_stream:
                img.save(img_byte_stream, format=img.format)
                img_byte_stream.seek(0)
                img_byte_stream.seek(0)
                img_bytes = img_byte_stream.getvalue()

            # process image with docling, as there is no groundtruth
            # for the funsd dataset, we will use the docling conversion as true_doc
            true_doc = doc_converter.convert(img_path).document
            image_ref = ImageRef(
                mimetype="image/png",
                dpi=72,
                size=Size(width=float(img.width), height=float(img.height)),
                uri=from_pil_to_base64uri(img),
            )
            page_item = PageItem(
                page_no=1,
                size=Size(width=float(img.width), height=float(img.height)),
                image=image_ref,
            )

            true_doc.pages[1] = page_item

            true_doc = populate_key_value_item_from_funsd(true_doc, data)

            true_doc, true_pictures, true_page_images = extract_images(
                document=true_doc,
                pictures_column=BenchMarkColumns.GROUNDTRUTH_PICTURES.value,  # pictures_column,
                page_images_column=BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES.value,  # page_images_column,
            )

            record = {
                BenchMarkColumns.DOCLING_VERSION: docling_version(),
                BenchMarkColumns.DOC_ID: img_path.stem,
                BenchMarkColumns.GROUNDTRUTH: json.dumps(true_doc.export_to_dict()),
                BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES: true_page_images,
                BenchMarkColumns.GROUNDTRUTH_PICTURES: true_pictures,
                BenchMarkColumns.ORIGINAL: img_bytes,
                BenchMarkColumns.MIMETYPE: "image/png",
            }
            records.append(record)
            count += 1
            if count % SHARD_SIZE == 0:
                shard_id = count // SHARD_SIZE - 1
                save_shard_to_disk(items=records, dataset_path=split_dir, shard_id=shard_id)
                records = []

        shard_id = count // SHARD_SIZE
        save_shard_to_disk(items=records, dataset_path=split_dir, shard_id=shard_id)
    write_datasets_info(
        name="FUNSD",
        output_dir=output_dir,
        num_train_rows=num_train_rows,
        num_test_rows=num_test_rows,
    )


"""
def main():
    input_dir = Path("path/to/funsd")
    output_dir = Path("path/to/output)

    create_funsd_dataset(input_dir, output_dir, splits=["train", "test"], max_items=-1)


if __name__ == "__main__":
    main()
"""
