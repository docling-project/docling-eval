import argparse
import io
import itertools
import json
import os
import re
from pathlib import Path

from datasets import load_from_disk
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
from docling_core.types.doc.tokens import TableToken
from docling_core.types.io import DocumentStream
from tqdm import tqdm  # type: ignore

from docling_eval.benchmarks.constants import BenchMarkColumns
from docling_eval.benchmarks.utils import write_datasets_info
from docling_eval.docling.conversion import create_converter
from docling_eval.docling.utils import (
    crop_bounding_box,
    docling_version,
    extract_images,
    from_pil_to_base64uri,
    save_shard_to_disk,
)


def parse_arguments():
    """Parse arguments for DP-Bench parsing."""

    parser = argparse.ArgumentParser(
        description="Convert DocLayNet v2 into DoclingDocument ground truth data"
    )
    parser.add_argument(
        "-i",
        "--input-directory",
        help="input directory with documents",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        help="output directory with shards",
        required=False,
        default="./benchmarks/dlnv2",
    )
    args = parser.parse_args()

    return (
        Path(args.input_directory),
        Path(args.output_directory),
    )


def parse_texts(texts, tokens):
    split_word = TableToken.OTSL_NL.value
    split_row_tokens = [
        list(y)
        for x, y in itertools.groupby(tokens, lambda z: z == split_word)
        if not x
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
            text=segment["text"], enumerated=False, prov=prov, parent=current_list
        )
    elif label == DocItemLabel.SECTION_HEADER:
        current_list = None
        true_doc.add_heading(text=segment["text"], prov=prov)
    else:
        current_list = None
        true_doc.add_text(label=label, text=segment["text"], prov=prov)


def create_dlnv2_e2e_dataset(input_dir, output_dir):
    converter = create_converter(
        page_image_scale=1.0, do_ocr=True, ocr_lang=["en", "fr", "es", "de", "jp", "cn"]
    )
    ds = load_from_disk(input_dir)
    records = []
    for doc in tqdm(ds):
        img = doc["image"]
        with io.BytesIO() as img_byte_stream:
            img.save(img_byte_stream, format=img.format)
            img_byte_stream.seek(0)
            conv_results = converter.convert(
                source=DocumentStream(name="foo.png", stream=img_byte_stream),
                raises_on_error=True,
            )
            img_byte_stream.seek(0)
            img_bytes = img_byte_stream.getvalue()

        pred_doc = conv_results.document

        true_doc = DoclingDocument(name=Path(doc["extra"]["filename"]).stem)
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

        current_list = None
        boxes = doc["boxes"]
        labels = list(
            map(
                lambda label: label.lower().replace("-", "_").replace(" ", "_"),
                doc["labels"],
            )
        )
        segments = doc["segments"]
        for l, s, b in zip(labels, segments, boxes):
            update(true_doc, current_list, img, l, s, b)

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

        record = {
            BenchMarkColumns.DOCLING_VERSION: docling_version(),
            BenchMarkColumns.STATUS: str(conv_results.status),
            BenchMarkColumns.DOC_ID: doc["extra"]["filename"],
            BenchMarkColumns.GROUNDTRUTH: json.dumps(true_doc.export_to_dict()),
            BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES: true_page_images,
            BenchMarkColumns.GROUNDTRUTH_PICTURES: true_pictures,
            BenchMarkColumns.PREDICTION: json.dumps(pred_doc.export_to_dict()),
            BenchMarkColumns.PREDICTION_PAGE_IMAGES: pred_page_images,
            BenchMarkColumns.PREDICTION_PICTURES: pred_pictures,
            BenchMarkColumns.ORIGINAL: img_bytes,
            BenchMarkColumns.MIMETYPE: "image/png",
        }
        records.append(record)

    test_dir = output_dir / "test"
    os.makedirs(test_dir, exist_ok=True)
    save_shard_to_disk(items=records, dataset_path=test_dir)
    write_datasets_info(
        name="DocLayNetV2: end-to-end",
        output_dir=output_dir,
        num_train_rows=0,
        num_test_rows=len(records),
    )


def main():
    input_dir, output_dir = parse_arguments()
    os.makedirs(output_dir, exist_ok=True)

    odir_e2e = Path(output_dir) / "end_to_end"
    os.makedirs(odir_e2e, exist_ok=True)
    for _ in ["test", "train"]:
        os.makedirs(odir_e2e / _, exist_ok=True)

    create_dlnv2_e2e_dataset(input_dir=input_dir, output_dir=odir_e2e)


if __name__ == "__main__":
    main()
