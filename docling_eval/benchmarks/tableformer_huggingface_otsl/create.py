import glob
import json
import os
from pathlib import Path
from typing import Any, List, Optional

from datasets import load_dataset
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.pipeline_options import TableFormerMode
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import (
    DoclingDocument,
    ImageRef,
    PageItem,
    ProvenanceItem,
)
from docling_core.types.doc.labels import DocItemLabel
from tqdm import tqdm  # type: ignore

from docling_eval.benchmarks.constants import BenchMarkColumns
from docling_eval.benchmarks.utils import (
    convert_html_table_into_docling_tabledata,
    save_comparison_html,
)
from docling_eval.docling.models.tableformer.tf_model_prediction import (
    PageTokens,
    TableFormerUpdater,
)
from docling_eval.docling.utils import (
    docling_version,
    extract_images,
    from_pil_to_base64uri,
    save_shard_to_disk,
)

HTML_EXPORT_LABELS = {
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


def create_page_tokens(data: List[Any], height: float, width: float) -> PageTokens:

    tokens = []
    # text_lines = []

    cnt = 0
    for i, row in enumerate(data):
        for j, item in enumerate(row):
            text = "".join(item["tokens"])

            tokens.append(
                {
                    "bbox": {
                        "l": item["bbox"][0],
                        # "t": height - item["bbox"][3],
                        "t": item["bbox"][1],
                        "r": item["bbox"][2],
                        # "b": height - item["bbox"][1],
                        "b": item["bbox"][3],
                        # "coord_origin": str(CoordOrigin.BOTTOMLEFT.value)
                        "coord_origin": str(CoordOrigin.TOPLEFT.value),
                    },
                    "text": "".join(item["tokens"]),
                    "id": cnt,
                }
            )
            cnt += 1

    result = {"tokens": tokens, "height": height, "width": width}
    return PageTokens.parse_obj(result)


def create_huggingface_otsl_tableformer_dataset(
    name: str,
    output_dir: Path,
    image_scale: float = 1.0,
    max_records: int = 1000,
    split: str = "test",
    do_viz: bool = False,
    max_items: int = -1,
    mode: TableFormerMode = TableFormerMode.ACCURATE,
    artifacts_path: Optional[Path] = None,
):

    # Create the directories
    viz_dir = output_dir / "vizualisations"
    os.makedirs(viz_dir, exist_ok=True)

    test_dir = output_dir / f"{split}"
    os.makedirs(test_dir, exist_ok=True)

    # Use glob to find all .parquet files in the directory
    parquet_files = glob.glob(os.path.join(str(test_dir), "*.parquet"))

    # Loop through and remove each file
    for file in parquet_files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")

    # Init the TableFormer model
    tf_updater = TableFormerUpdater(mode, artifacts_path=artifacts_path)

    ds = load_dataset(name, split=split)

    if max_items == -1:
        max_items = len(ds)

    records = []
    tid, sid = 0, 0

    for i, item in tqdm(
        enumerate(ds),
        total=max_items,
        ncols=128,
        desc=f"create {name} tableformer dataset",
    ):

        if i >= max_items:
            break

        filename = item["filename"]
        table_image = item["image"]

        true_page_images = [table_image]
        page_tokens = create_page_tokens(
            data=item["cells"], height=table_image.height, width=table_image.width
        )

        # Ground truth document
        true_doc = DoclingDocument(name=f"ground-truth {filename}")

        page_index = 1

        image_ref = ImageRef(
            mimetype="image/png",
            dpi=round(72 * image_scale),
            size=Size(width=float(table_image.width), height=float(table_image.height)),
            uri=from_pil_to_base64uri(table_image),
        )
        page_item = PageItem(
            page_no=page_index,
            size=Size(width=float(table_image.width), height=float(table_image.height)),
            image=image_ref,
        )

        html = "<table>" + "".join(item["html"]) + "</table>"
        table_data = convert_html_table_into_docling_tabledata(
            html, text_cells=item["cells"][0]
        )

        l = 0.0
        b = 0.0
        r = table_image.width
        t = table_image.height
        if "table_bbox" in item:
            l = item["table_bbox"][0]
            b = table_image.height - item["table_bbox"][3]
            r = item["table_bbox"][2]
            t = table_image.height - item["table_bbox"][1]

        bbox = BoundingBox(
            l=l,
            r=r,
            b=b,
            t=t,
            coord_origin=CoordOrigin.BOTTOMLEFT,
        )

        prov = ProvenanceItem(page_no=page_index, bbox=bbox, charspan=(0, 0))
        true_doc.pages[1] = page_item

        true_doc.add_table(data=table_data, caption=None, prov=prov)

        # Create the updated Document
        updated, pred_doc = tf_updater.replace_tabledata_with_page_tokens(
            page_tokens=page_tokens,
            true_doc=true_doc,
            true_page_images=true_page_images,
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

        if updated:

            if do_viz:
                save_comparison_html(
                    filename=viz_dir / f"{os.path.basename(filename)}-comp.html",
                    true_doc=true_doc,
                    pred_doc=pred_doc,
                    page_image=true_page_images[0],
                    true_labels=HTML_EXPORT_LABELS,
                    pred_labels=HTML_EXPORT_LABELS,
                )

            record = {
                BenchMarkColumns.DOCLING_VERSION: docling_version(),
                BenchMarkColumns.STATUS: str(ConversionStatus.SUCCESS.value),
                BenchMarkColumns.DOC_ID: str(os.path.basename(filename)),
                BenchMarkColumns.GROUNDTRUTH: json.dumps(true_doc.export_to_dict()),
                BenchMarkColumns.PREDICTION: json.dumps(pred_doc.export_to_dict()),
                BenchMarkColumns.ORIGINAL: item["image"],
                BenchMarkColumns.MIMETYPE: "image/png",
                BenchMarkColumns.PREDICTION_PAGE_IMAGES: pred_page_images,
                BenchMarkColumns.PREDICTION_PICTURES: pred_pictures,
                BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES: true_page_images,
                BenchMarkColumns.GROUNDTRUTH_PICTURES: true_pictures,
            }
            records.append(record)
        else:
            record = {
                BenchMarkColumns.DOCLING_VERSION: docling_version(),
                BenchMarkColumns.STATUS: str(ConversionStatus.FAILURE.value),
                BenchMarkColumns.DOC_ID: str(os.path.basename(filename)),
                BenchMarkColumns.GROUNDTRUTH: json.dumps(true_doc.export_to_dict()),
                BenchMarkColumns.PREDICTION: json.dumps(None),
                BenchMarkColumns.ORIGINAL: item["image"],
                BenchMarkColumns.MIMETYPE: "image/png",
                BenchMarkColumns.PREDICTION_PAGE_IMAGES: pred_page_images,
                BenchMarkColumns.PREDICTION_PICTURES: pred_pictures,
                BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES: true_page_images,
                BenchMarkColumns.GROUNDTRUTH_PICTURES: true_pictures,
            }
            records.append(record)

        if len(records) == max_records:
            save_shard_to_disk(
                items=records,
                dataset_path=test_dir,
                thread_id=tid,
                shard_id=sid,
                shard_format="parquet",
            )
            sid += 1
            records = []

    if len(records) > 0:
        save_shard_to_disk(
            items=records,
            dataset_path=test_dir,
            thread_id=tid,
            shard_id=sid,
            shard_format="parquet",
        )
        sid += 1
        records = []


def create_fintabnet_tableformer_dataset(
    output_dir: Path,
    image_scale: float = 1.0,
    max_records: int = 1000,
    do_viz: bool = False,
    max_items: int = 1000,
    mode: TableFormerMode = TableFormerMode.ACCURATE,
    artifacts_path: Optional[Path] = None,
):
    create_huggingface_otsl_tableformer_dataset(
        name="ds4sd/FinTabNet_OTSL",
        output_dir=output_dir,
        image_scale=image_scale,
        max_records=max_records,
        split="test",
        do_viz=do_viz,
        max_items=max_items,
        mode=mode,
        artifacts_path=artifacts_path,
    )


def create_pubtabnet_tableformer_dataset(
    output_dir: Path,
    image_scale: float = 1.0,
    max_records: int = 1000,
    do_viz: bool = False,
    max_items: int = 1000,
    mode: TableFormerMode = TableFormerMode.ACCURATE,
    artifacts_path: Optional[Path] = None,
):
    create_huggingface_otsl_tableformer_dataset(
        name="ds4sd/PubTabNet_OTSL",
        output_dir=output_dir,
        image_scale=image_scale,
        max_records=max_records,
        split="val",
        do_viz=do_viz,
        max_items=max_items,
        mode=mode,
        artifacts_path=artifacts_path,
    )


def create_p1m_tableformer_dataset(
    output_dir: Path,
    image_scale: float = 1.0,
    max_records: int = 1000,
    do_viz: bool = True,
    max_items: int = 1000,
    mode: TableFormerMode = TableFormerMode.ACCURATE,
    artifacts_path: Optional[Path] = None,
):
    create_huggingface_otsl_tableformer_dataset(
        name="ds4sd/PubTables-1M_OTSL",
        output_dir=output_dir,
        image_scale=image_scale,
        max_records=max_records,
        split="test",
        do_viz=do_viz,
        max_items=max_items,
        mode=mode,
        artifacts_path=artifacts_path,
    )
