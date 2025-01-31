import argparse
import io
import json
import math
import os
from pathlib import Path

from datasets import load_dataset
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
from docling_core.types.io import DocumentStream
from tqdm import tqdm  # type: ignore

from docling_eval.benchmarks.constants import BenchMarkColumns
from docling_eval.benchmarks.utils import (
    write_datasets_info,
    save_comparison_html_with_clusters,
)
from docling_eval.docling.conversion import create_converter
from docling_eval.docling.utils import (
    crop_bounding_box,
    docling_version,
    extract_images,
    from_pil_to_base64uri,
    save_shard_to_disk,
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

SHARD_SIZE = 1000

category_map = {
    1: "caption",
    2: "footnote",
    3: "formula",
    4: "list_item",
    5: "page_footer",
    6: "page_header",
    7: "picture",
    8: "section_header",
    9: "table",
    10: "text",
    11: "title",
}


def parse_arguments():
    """Parse arguments for DP-Bench parsing."""

    parser = argparse.ArgumentParser(
        description="Convert DocLayNet v1 into DoclingDocument ground truth data"
    )
    parser.add_argument(
        "-s",
        "--split",
        help="train test or val",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        help="output directory with shards",
        required=False,
        default="./benchmarks/dlnv1",
    )
    args = parser.parse_args()

    return args.split, Path(args.output_directory)


def ltwh_to_ltrb(box):
    l = box[0]
    t = box[1]
    w = box[2]
    h = box[3]
    r = l + w
    b = t + h
    return l, t, r, b


def update(true_doc, current_list, img, label, box, content):
    bbox = BoundingBox.from_tuple(
        tuple(ltwh_to_ltrb(box)), CoordOrigin.TOPLEFT
    ).to_bottom_left_origin(page_height=true_doc.pages[1].size.height)
    prov = ProvenanceItem(page_no=1, bbox=bbox, charspan=(0, len(content)))
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
                "Warning: failed to resolve image uri for content {} of doc {}. Caught exception is {}:{}. Setting null ImageRef".format(
                    str(content), str(true_doc.name), type(e).__name__, e
                )
            )
            imgref = None

        true_doc.add_picture(prov=prov, image=imgref)
    elif label in [DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX]:
        current_list = None
        tbl_cell = TableCell(
            start_row_offset_idx=0,
            end_row_offset_idx=0,
            start_col_offset_idx=0,
            end_col_offset_idx=0,
            text=content,
        )
        tbl_data = TableData(table_cells=[tbl_cell])

        true_doc.add_table(data=tbl_data, prov=prov, label=label)
    elif label == DocItemLabel.LIST_ITEM:
        if current_list is None:
            current_list = true_doc.add_group(label=GroupLabel.LIST, name="list")

            # TODO: Infer if this is a numbered or a bullet list item
        true_doc.add_list_item(
            text=content, enumerated=False, prov=prov, parent=current_list
        )
    elif label == DocItemLabel.SECTION_HEADER:
        current_list = None
        true_doc.add_heading(text=content, prov=prov)
    else:
        current_list = None
        true_doc.add_text(label=label, text=content, prov=prov)


def create_dlnv1_e2e_dataset(split, output_dir, do_viz=False, max_items=None):
    converter = create_converter(
        page_image_scale=1.0, do_ocr=True, ocr_lang=["en", "fr", "es", "de", "jp", "cn"]
    )
    ds = load_dataset("ds4sd/DocLayNet-v1.1", trust_remote_code=True)

    if do_viz:
        viz_dir = output_dir / "visualizations"
        os.makedirs(viz_dir, exist_ok=True)

    test_dir = output_dir / split
    os.makedirs(test_dir, exist_ok=True)
    records = []
    count = 0
    for doc in tqdm(
        ds[split],
        total=min(len(ds[split]), max_items if max_items is not None else math.inf),
    ):
        img = doc["image"]
        with io.BytesIO() as img_byte_stream:
            img.save(img_byte_stream, format=img.format)
            img_byte_stream.seek(0)
            conv_results = converter.convert(
                source=DocumentStream(
                    name=doc["metadata"]["page_hash"], stream=img_byte_stream
                ),
                raises_on_error=True,
            )
            img_byte_stream.seek(0)
            img_bytes = img_byte_stream.getvalue()

        pred_doc = conv_results.document

        true_doc = DoclingDocument(name=doc["metadata"]["page_hash"])
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
        labels = list(map(lambda cid: category_map[int(cid)], doc["category_id"]))
        bboxes = doc["bboxes"]
        segments = doc["pdf_cells"]
        contents = [
            " ".join(map(lambda cell: cell["text"], cells)) for cells in segments
        ]
        for l, b, c in zip(labels, bboxes, contents):
            update(true_doc, current_list, img, l, b, c)

        if do_viz:
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
                filename=viz_dir / f"{true_doc.name}-clusters.html",
                true_doc=true_doc,
                pred_doc=pred_doc,
                page_image=img,
                true_labels=TRUE_HTML_EXPORT_LABELS,
                pred_labels=PRED_HTML_EXPORT_LABELS,
            )
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
            BenchMarkColumns.DOC_ID: doc["metadata"]["page_hash"],
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
        count += 1
        if count % SHARD_SIZE == 0:
            shard_id = count // SHARD_SIZE - 1
            save_shard_to_disk(items=records, dataset_path=test_dir, shard_id=shard_id)
            records = []
        if count is not None and count > max_items:
            break

    if len(records) > 0:
        shard_id = count // SHARD_SIZE
        save_shard_to_disk(items=records, dataset_path=test_dir, shard_id=shard_id)

    write_datasets_info(
        name="DocLayNetV1: end-to-end",
        output_dir=output_dir,
        num_train_rows=0,
        num_test_rows=len(records) + count,
    )


def main():
    split, output_dir = parse_arguments()
    os.makedirs(output_dir, exist_ok=True)

    odir_e2e = Path(output_dir) / "end_to_end"
    os.makedirs(odir_e2e, exist_ok=True)
    for _ in ["test", "train"]:
        os.makedirs(odir_e2e / _, exist_ok=True)

    create_dlnv1_e2e_dataset(split=split, output_dir=odir_e2e)


if __name__ == "__main__":
    main()
