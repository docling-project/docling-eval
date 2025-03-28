import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

from docling.datamodel.pipeline_options import TableFormerMode
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import (
    DoclingDocument,
    ImageRef,
    ProvenanceItem,
    TableCell,
    TableData,
)
from docling_core.types.doc.labels import DocItemLabel
from PIL import Image  # as PILImage
from tqdm import tqdm

from docling_eval.datamodels.types import (
    BenchMarkColumns,
    ConverterTypes,
    EvaluationModality,
)
from docling_eval.legacy.converters.conversion import (
    create_pdf_docling_converter,
    create_smol_docling_converter,
)
from docling_eval.prediction_providers.tableformer_provider import TableFormerUpdater
from docling_eval.utils.utils import (
    add_pages_to_true_doc,
    convert_html_table_into_docling_tabledata,
    crop_bounding_box,
    docling_version,
    extract_images,
    from_pil_to_base64uri,
    get_binary,
    save_shard_to_disk,
    set_selection_range,
    write_datasets_info,
)
from docling_eval.visualisation.visualisations import (  # type: ignore
    save_comparison_html,
    save_comparison_html_with_clusters,
)

# Get logger
_log = logging.getLogger(__name__)

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


def update(
    doc: DoclingDocument,
    annots: Dict,
    page,
    page_image: Image.Image,
    page_width: float,
    page_height: float,
):

    label = annots["category"]

    min_x = annots["coordinates"][0]["x"]
    max_x = annots["coordinates"][0]["x"]

    min_y = annots["coordinates"][0]["y"]
    max_y = annots["coordinates"][0]["y"]

    for coor in annots["coordinates"]:
        min_x = min(min_x, coor["x"])
        max_x = max(max_x, coor["x"])

        min_y = min(min_y, coor["y"])
        max_y = max(max_y, coor["y"])

    text = annots["content"]["text"].replace("\n", " ")
    html = annots["content"]["html"]

    bbox = BoundingBox(
        l=min_x * page_width,
        r=max_x * page_width,
        t=min_y * page_height,
        b=max_y * page_height,
        coord_origin=CoordOrigin.TOPLEFT,
    )

    prov = ProvenanceItem(page_no=1, bbox=bbox, charspan=(0, len(text)))

    img = crop_bounding_box(page_image=page_image, page=page, bbox=bbox)

    if label == "Header":
        doc.add_text(label=DocItemLabel.PAGE_HEADER, text=text, orig=text, prov=prov)

    elif label == "Footer":
        doc.add_text(label=DocItemLabel.PAGE_FOOTER, text=text, orig=text, prov=prov)

    elif label == "Paragraph":
        doc.add_text(label=DocItemLabel.TEXT, text=text, orig=text, prov=prov)

    elif label == "Index":

        # FIXME: ultra approximate solution
        text = annots["content"]["text"]
        rows = text.split("\n")

        num_rows = len(rows)
        num_cols = 2

        row_span = 1
        col_span = 1

        cells = []
        for row_idx, row in enumerate(rows):

            parts = row.split(" ")

            col_idx = 0
            cell = TableCell(
                row_span=row_span,
                col_span=col_span,
                start_row_offset_idx=row_idx,
                end_row_offset_idx=row_idx + row_span,
                start_col_offset_idx=col_idx,
                end_col_offset_idx=col_idx + col_span,
                text=" ".join(parts[:-1]),
            )
            cells.append(cell)

            col_idx = 1
            cell = TableCell(
                row_span=row_span,
                col_span=col_span,
                start_row_offset_idx=row_idx,
                end_row_offset_idx=row_idx + row_span,
                start_col_offset_idx=col_idx,
                end_col_offset_idx=col_idx + col_span,
                text=parts[-1],
            )
            cells.append(cell)

        table_data = TableData(num_rows=num_rows, num_cols=num_cols, table_cells=cells)
        doc.add_table(
            data=table_data, caption=None, prov=prov, label=DocItemLabel.DOCUMENT_INDEX
        )

    elif label == "List":
        doc.add_list_item(text=text, orig=text, prov=prov)
        # doc.add_text(label=DocItemLabel.TEXT, text=text, orig=text, prov=prov)

    elif label == "Caption":
        doc.add_text(label=DocItemLabel.CAPTION, text=text, orig=text, prov=prov)

    elif label == "Equation":
        doc.add_text(label=DocItemLabel.FORMULA, text=text, orig=text, prov=prov)

    elif label == "Figure":
        uri = from_pil_to_base64uri(img)

        imgref = ImageRef(
            mimetype="image/png",
            dpi=72,
            size=Size(width=img.width, height=img.height),
            uri=uri,
        )

        doc.add_picture(prov=prov, image=imgref)

    elif label == "Table":

        table_data = convert_html_table_into_docling_tabledata(table_html=html)

        doc.add_table(data=table_data, caption=None, prov=prov)

    elif label == "Chart":
        uri = from_pil_to_base64uri(img)

        imgref = ImageRef(
            mimetype="image/png",
            dpi=72,
            size=Size(width=img.width, height=img.height),
            uri=uri,
        )

        doc.add_picture(prov=prov, image=imgref)

        # doc.add_picture(prov=prov)

    elif label == "Footnote":
        doc.add_text(label=DocItemLabel.FOOTNOTE, text=text, orig=text, prov=prov)

    elif label == "Heading1":
        doc.add_heading(text=text, orig=text, level=1, prov=prov)

    else:
        return


def create_dpbench_e2e_dataset(
    dpbench_dir: Path,
    output_dir: Path,
    begin_index: int = 0,
    end_index: int = -1,  # If -1 take the whole split
    converter_type: ConverterTypes = ConverterTypes.DOCLING,
    image_scale: float = 1.0,
    do_viz: bool = False,
    artifacts_path: Optional[Path] = None,
):
    # Create Converter
    if converter_type == ConverterTypes.DOCLING:
        converter = create_pdf_docling_converter(page_image_scale=1.0)
    else:
        converter = create_smol_docling_converter()

    # load the groundtruth
    with open(dpbench_dir / "dataset/reference.json", "r") as fr:
        gt = json.load(fr)

    viz_dir = output_dir / "vizualisations"
    os.makedirs(viz_dir, exist_ok=True)

    records = []

    # Apply index ranges
    gt_selection = list(gt.items())
    total_ds_len = len(gt_selection)
    begin_index, end_index = set_selection_range(begin_index, end_index, total_ds_len)
    gt_selection = gt_selection[begin_index:end_index]
    selected_ds_len = len(gt_selection)
    _log.info(
        "Dataset len: %s. Selected range: [%s, %s] = %s",
        total_ds_len,
        begin_index,
        end_index,
        selected_ds_len,
    )

    for filename, annots in tqdm(
        gt_selection,
        desc="Processing files for DP-Bench with end-to-end",
        total=selected_ds_len,
        ncols=128,
    ):
        pdf_path = dpbench_dir / f"dataset/pdfs/{filename}"

        # Create the predicted Document
        conv_results = converter.convert(source=pdf_path, raises_on_error=True)
        pred_doc = conv_results.document

        # Create the groundtruth Document
        true_doc = DoclingDocument(name=f"ground-truth {os.path.basename(pdf_path)}")
        true_doc, true_page_images = add_pages_to_true_doc(
            pdf_path=pdf_path, true_doc=true_doc, image_scale=image_scale
        )

        assert len(true_page_images) == 1, "len(true_page_images)==1"

        page_width = true_doc.pages[1].size.width
        page_height = true_doc.pages[1].size.height

        for elem in annots["elements"]:
            update(
                true_doc,
                elem,
                page=true_doc.pages[1],
                page_image=true_page_images[0],
                page_width=page_width,
                page_height=page_height,
            )

        if do_viz:
            save_comparison_html_with_clusters(
                filename=viz_dir / f"{os.path.basename(pdf_path)}-clusters.html",
                true_doc=true_doc,
                pred_doc=pred_doc,
                page_image=true_page_images[0],
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
            BenchMarkColumns.CONVERTER_TYPE: converter_type,
            BenchMarkColumns.CONVERTER_VERSION: docling_version(),
            BenchMarkColumns.STATUS: str(conv_results.status),
            BenchMarkColumns.DOC_ID: str(filename),
            BenchMarkColumns.GROUNDTRUTH: json.dumps(true_doc.export_to_dict()),
            BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES: true_page_images,
            BenchMarkColumns.GROUNDTRUTH_PICTURES: true_pictures,
            BenchMarkColumns.PREDICTION: json.dumps(pred_doc.export_to_dict()),
            BenchMarkColumns.PREDICTION_PAGE_IMAGES: pred_page_images,
            BenchMarkColumns.PREDICTION_PICTURES: pred_pictures,
            BenchMarkColumns.ORIGINAL: get_binary(pdf_path),
            BenchMarkColumns.MIMETYPE: "application/pdf",
            BenchMarkColumns.MODALITIES: [
                EvaluationModality.LAYOUT,
                EvaluationModality.MARKDOWN_TEXT,
                EvaluationModality.READING_ORDER,
                EvaluationModality.TABLE_STRUCTURE,
            ],
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


def create_dpbench_tableformer_dataset(
    dpbench_dir: Path,
    output_dir: Path,
    begin_index: int = 0,
    end_index: int = -1,  # If -1 take the whole split
    image_scale: float = 1.0,
    mode: TableFormerMode = TableFormerMode.ACCURATE,
    artifacts_path: Optional[Path] = None,
):
    # Init the TableFormer model
    tf_updater = TableFormerUpdater(mode, artifacts_path=artifacts_path)

    # load the groundtruth
    with open(dpbench_dir / "dataset/reference.json", "r") as fr:
        gt = json.load(fr)

    viz_dir = output_dir / "vizualisations"
    os.makedirs(viz_dir, exist_ok=True)

    records = []

    # Apply index ranges
    gt_selection = list(gt.items())
    total_ds_len = len(gt_selection)
    begin_index, end_index = set_selection_range(begin_index, end_index, total_ds_len)
    gt_selection = gt_selection[begin_index:end_index]
    selected_ds_len = len(gt_selection)
    _log.info(
        "Dataset len: %s. Selected range: [%s, %s] = %s",
        total_ds_len,
        begin_index,
        end_index,
        selected_ds_len,
    )

    for filename, annots in tqdm(
        gt_selection,
        desc="Processing files for DP-Bench with end-to-end",
        total=selected_ds_len,
        ncols=128,
    ):
        pdf_path = dpbench_dir / f"dataset/pdfs/{filename}"

        # Create the groundtruth Document
        true_doc = DoclingDocument(name=f"{os.path.basename(pdf_path)}")
        true_doc, true_page_images = add_pages_to_true_doc(
            pdf_path=pdf_path, true_doc=true_doc, image_scale=image_scale
        )

        assert len(true_page_images) == 1, "len(true_page_images)==1"

        page_width = true_doc.pages[1].size.width
        page_height = true_doc.pages[1].size.height

        for elem in annots["elements"]:
            update(
                true_doc,
                elem,
                page=true_doc.pages[1],
                page_image=true_page_images[0],
                page_width=page_width,
                page_height=page_height,
            )

        # Create the updated Document
        updated, pred_doc = tf_updater.replace_tabledata(
            pdf_path=pdf_path, true_doc=true_doc
        )

        if updated:

            if True:
                save_comparison_html(
                    filename=viz_dir / f"{os.path.basename(pdf_path)}-comp.html",
                    true_doc=true_doc,
                    pred_doc=pred_doc,
                    page_image=true_page_images[0],
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
                BenchMarkColumns.CONVERTER_TYPE: ConverterTypes.DOCLING,
                BenchMarkColumns.CONVERTER_VERSION: docling_version(),
                BenchMarkColumns.STATUS: "SUCCESS",
                BenchMarkColumns.DOC_ID: str(os.path.basename(pdf_path)),
                BenchMarkColumns.GROUNDTRUTH: json.dumps(true_doc.export_to_dict()),
                BenchMarkColumns.PREDICTION: json.dumps(pred_doc.export_to_dict()),
                BenchMarkColumns.ORIGINAL: get_binary(pdf_path),
                BenchMarkColumns.MIMETYPE: "application/pdf",
                BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES: true_page_images,
                BenchMarkColumns.GROUNDTRUTH_PICTURES: true_pictures,
                BenchMarkColumns.PREDICTION_PAGE_IMAGES: pred_page_images,
                BenchMarkColumns.PREDICTION_PICTURES: pred_pictures,
                BenchMarkColumns.MODALITIES: [
                    EvaluationModality.TABLE_STRUCTURE,
                ],
            }
            records.append(record)

    test_dir = output_dir / "test"
    os.makedirs(test_dir, exist_ok=True)

    save_shard_to_disk(items=records, dataset_path=test_dir)

    write_datasets_info(
        name="DPBench: tableformer",
        output_dir=output_dir,
        num_train_rows=0,
        num_test_rows=len(records),
    )
