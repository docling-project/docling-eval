import io
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

from docling_core.types import DoclingDocument
from docling_core.types.doc import BoundingBox, ImageRef, PageItem, ProvenanceItem, Size
from docling_core.types.doc.document import GraphCell, GraphData, GraphLink
from docling_core.types.doc.labels import GraphCellLabel, GraphLinkLabel
from PIL import Image
from tqdm import tqdm  # type: ignore

from docling_eval.datamodels.types import BenchMarkColumns, ConverterTypes
from docling_eval.legacy.converters.conversion import create_image_docling_converter
from docling_eval.utils.utils import (
    docling_version,
    extract_images,
    from_pil_to_base64uri,
    save_shard_to_disk,
    write_datasets_info,
)

SHARD_SIZE = 1000


def convert_bbox(bbox_data) -> BoundingBox:
    if isinstance(bbox_data, list) and len(bbox_data) == 4:
        return BoundingBox(
            l=bbox_data[0], t=bbox_data[1], r=bbox_data[2], b=bbox_data[3]
        )
    elif isinstance(bbox_data, BoundingBox):
        return bbox_data
    else:
        raise ValueError(
            "Invalid bounding box data; expected a list of four numbers or a BoundingBox instance."
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
    bbox_instance = BoundingBox.enclosing_bbox(all_bboxes)
    return bbox_instance


def populate_key_value_item_from_funsd(
    doc: DoclingDocument, funsd_data: dict
) -> DoclingDocument:
    """Populate the key-value item from the FUNSD data."""
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

    overal_bbox = get_overall_bbox(
        links, cell_dict={cell.cell_id: cell for cell in cells}
    )

    if overal_bbox is not None:
        prov = ProvenanceItem(
            page_no=doc.pages[1].page_no,
            charspan=(0, 0),
            bbox=overal_bbox,
        )
    else:
        prov = None

    graph = GraphData(cells=cells, links=links)

    doc.add_key_values(graph=graph, prov=prov)

    return doc


def create_funsd_dataset(
    input_dir: Path,
    output_dir: Path,
    splits: List[str] = ["train", "test"],
    max_items: int = -1,
):
    doc_converter = create_image_docling_converter(do_ocr=True, ocr_lang=["en"])

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
                img_path.parent.parent
                / "annotations"
                / img_path.name.replace(".png", ".json")
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
                BenchMarkColumns.CONVERTER_TYPE: ConverterTypes.DOCLING,
                BenchMarkColumns.CONVERTER_VERSION: docling_version(),
                BenchMarkColumns.DOC_ID: img_path.stem,
                BenchMarkColumns.GROUNDTRUTH: json.dumps(true_doc.export_to_dict()),
                BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES: true_page_images,
                BenchMarkColumns.GROUNDTRUTH_PICTURES: true_pictures,
                BenchMarkColumns.ORIGINAL: img_bytes,
                BenchMarkColumns.MIMETYPE: "image/png",
                BenchMarkColumns.MODALITIES: [],
            }
            records.append(record)
            count += 1
            if count % SHARD_SIZE == 0:
                shard_id = count // SHARD_SIZE - 1
                save_shard_to_disk(
                    items=records, dataset_path=split_dir, shard_id=shard_id
                )
                records = []

        shard_id = count // SHARD_SIZE
        save_shard_to_disk(items=records, dataset_path=split_dir, shard_id=shard_id)
    write_datasets_info(
        name="FUNSD",
        output_dir=output_dir,
        num_train_rows=num_train_rows,
        num_test_rows=num_test_rows,
    )
