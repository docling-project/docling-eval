import io
import json
import logging
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
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)
from docling_core.types.io import DocumentStream
from tqdm import tqdm  # type: ignore

from docling_eval.datamodels.types import (
    BenchMarkColumns,
    ConverterTypes,
    EvaluationModality,
)
from docling_eval.legacy.converters.conversion import (
    create_pdf_docling_converter,
    create_smol_docling_converter,
)
from docling_eval.utils.utils import (
    add_pages_to_true_doc,
    crop_bounding_box,
    docling_version,
    extract_images,
    from_pil_to_base64uri,
    save_shard_to_disk,
    set_selection_range,
    write_datasets_info,
)
from docling_eval.visualisation.visualisations import save_comparison_html_with_clusters

# Get logger
_log = logging.getLogger(__name__)


###########################################################################################
BLACKLISTED_DOC_IDS = [
    "f556167ac3284665652050b1b0bc1e6f5af27f54f17f27566c60c80f6f134a92",
    "dbc51622cbe9b8766f44db3b3fda8d0a745da06b9bfec9935bd003d2bdd494c8",
    "d4c0401fffc04d24e629a9fada23266a3b492ea63e889641b3c33adf815d44e3",
    "cc93b556f49af1f2e366719ec98a131186c16385545d8062d21e4d38b6bf7686",
    "c9755e6972e3150a1c02565ec8070bfc26503d0fe09d056e418d6dcd6ea43cd9",
    "c90d298ac9493e3804baf1b62c9321cdabf388c29eb504c5ad12106b3cdf530b",
    "c2b513a5611d3138726e679c6e2e9e5383e4d3d82a2c588bbe3d5802797e2765",
    "b72bb61059b06ff9859ae023aa66cdb3ff706c354ac72ca5d3c837e107d0a384",
    "b4f5d430d89499474a31f39fe8eb615fdcd7aa682eb0b959a0384206d5c8174c",
    "ab9315a0610ec0e5446a7062cd99a9e137efe3d7da9a7bffa2523894ac68751a",
    "99723d3d3c61db030dbd813faec67579ceb50c6b5dd8c2f500c6e073849e9784",
    "87c7dc9ca13016fafa4a7539efa1bf00401ba27323a473094b4184bc42cb36c0",
    "7c1fa2e7c81a81888c18eb95cfe37edb82a91dd340e75c8123618a6774081f2e",
    "7a231e9b7d841935a142d972ea1c7546d613cba18e301b0e07415f9eb44e3382",
    "5793282eaaa089d0dc71e67c951c68b4157a212cc43edbc3106323e96b385190",
    "55f9167173149b0b4c8d8951baca190ee756450d6565a91655ec04967a08c798",
    "5003688e1ae61558cbeda741d246804b59fe89dac29cf508b4b6ce56d1a4342b",
    "4f6e20223b7bc8436c623b9e6282db6ebd5f221aeb880a8db9b4544326d5a8a6",
    "4232e47097e6ecfdf53d4097cb90bdd56cc63c31508a2f91a6d3908770a4d1ed",
    "3361796dba75fe2c641c43db12ab31a0eb9dbcbbc7c99721288d36c41d759bcd",
    "1fadb433bffa31c43817d1f6bafbb10dff53422ad046d391ed560ebef13d9f83",
    "1a8f46903dbe89dc5b6df43389b4895a376e00ab3b90c7c37f1a1b561d3d51a1",
    "1763e54be635759ccb66ebb462548f8a40d44567f62cecc5ca26f22acd28e823",
    "048a570b2e415b653a62313ef82504adfda480c99f69826fcbeb67758ea3c7a4",
    "0261791e343389682847c913a16789776d0ba41a584901571846c7ddab3cbaa6",
]
###########################################################################################

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

SHARD_SIZE = 100

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


def ltwh_to_ltrb(box):
    l = box[0]
    t = box[1]
    w = box[2]
    h = box[3]
    r = l + w
    b = t + h
    return l, t, r, b


def update(true_doc, current_list, img, old_size, label, box, content):
    w, h = img.size
    new_size = Size(width=w, height=h)
    bbox = (
        BoundingBox.from_tuple(tuple(ltwh_to_ltrb(box)), CoordOrigin.TOPLEFT)
        .to_bottom_left_origin(page_height=old_size.height)
        .scale_to_size(old_size=old_size, new_size=new_size)
    )
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
            _log.error(
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


def create_dlnv1_e2e_dataset(
    name: str,
    split: str,
    output_dir: Path,
    converter_type: ConverterTypes = ConverterTypes.DOCLING,
    do_viz: bool = False,
    begin_index: int = 0,
    end_index: int = -1,  # If -1 take the whole split
    do_debug: bool = False,
):
    ds = load_dataset(name, split=split)
    total_ds_len = len(ds)

    # Select the asked rows
    begin_index, end_index = set_selection_range(begin_index, end_index, total_ds_len)
    ds = ds.select(range(begin_index, end_index))
    selected_ds_len = len(ds)
    _log.info(
        "Dataset len: %s. Selected range: [%s, %s] = %s",
        total_ds_len,
        begin_index,
        end_index,
        selected_ds_len,
    )

    # Decide which converter type to initialize
    if converter_type == ConverterTypes.DOCLING:
        converter = create_pdf_docling_converter(page_image_scale=1.0)
    else:
        converter = create_smol_docling_converter()

    if do_viz:
        viz_dir = output_dir / "visualizations"
        os.makedirs(viz_dir, exist_ok=True)

    test_dir = output_dir / split
    os.makedirs(test_dir, exist_ok=True)
    records = []
    exported_rows = 0
    skipped_rows = 0
    saved_shards = 0

    black_page_hashes = set(BLACKLISTED_DOC_IDS)

    for doc in tqdm(ds, total=selected_ds_len):
        try:
            page_hash = doc["metadata"]["page_hash"]
            if page_hash in black_page_hashes:
                _log.info("Skip blacklisted doc id: %s", page_hash)
                continue

            if do_debug:
                _log.info("Converting: %s", page_hash)

            pdf = doc["pdf"]
            pdf_stream = io.BytesIO(pdf)
            pdf_stream.seek(0)
            conv_results = converter.convert(
                source=DocumentStream(
                    name=doc["metadata"]["page_hash"], stream=pdf_stream
                ),
                raises_on_error=True,
            )
            pdf_stream = io.BytesIO(pdf)

            pred_doc = conv_results.document

            # Debugging code that dumps the VLM predicted text in files
            if do_debug and converter_type == ConverterTypes.SMOL_DOCLING:
                debug_dir = output_dir / "debug"
                os.makedirs(debug_dir, exist_ok=True)
                if len(conv_results.pages):
                    for page_id, page in enumerate(conv_results.pages):
                        predicted_txt = page.predictions.vlm_response.text
                        if predicted_txt is None:
                            continue
                        page_text_fn = debug_dir / f"{page_hash}_{page_id}.txt"
                        with open(page_text_fn, "w") as fd:
                            fd.write(predicted_txt)

            true_doc = DoclingDocument(name=page_hash)
            true_doc, true_page_images = add_pages_to_true_doc(
                pdf_path=pdf_stream, true_doc=true_doc, image_scale=1.0
            )
            img = true_page_images[0]
            old_w, old_h = doc["image"].size
            old_size = Size(width=old_w, height=old_h)

            current_list = None
            labels = list(map(lambda cid: category_map[int(cid)], doc["category_id"]))
            bboxes = doc["bboxes"]
            segments = doc["pdf_cells"]
            contents = [
                " ".join(map(lambda cell: cell["text"], cells)) for cells in segments
            ]
            for l, b, c in zip(labels, bboxes, contents):
                update(true_doc, current_list, img, old_size, l, b, c)

            if do_viz:
                save_comparison_html_with_clusters(
                    filename=viz_dir / f"{true_doc.name}-clusters.html",
                    true_doc=true_doc,
                    pred_doc=pred_doc,
                    page_image=img,
                    true_labels=TRUE_HTML_EXPORT_LABELS,
                    pred_labels=PRED_HTML_EXPORT_LABELS,
                    draw_reading_order=False,  # Disable reading-order visualization
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
                BenchMarkColumns.DOC_ID: page_hash,
                BenchMarkColumns.GROUNDTRUTH: json.dumps(true_doc.export_to_dict()),
                BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES: true_page_images,
                BenchMarkColumns.GROUNDTRUTH_PICTURES: true_pictures,
                BenchMarkColumns.PREDICTION: json.dumps(pred_doc.export_to_dict()),
                BenchMarkColumns.PREDICTION_PAGE_IMAGES: pred_page_images,
                BenchMarkColumns.PREDICTION_PICTURES: pred_pictures,
                BenchMarkColumns.ORIGINAL: pdf_stream.getvalue(),
                BenchMarkColumns.MIMETYPE: "image/png",
                BenchMarkColumns.MODALITIES: [
                    EvaluationModality.LAYOUT,
                    EvaluationModality.MARKDOWN_TEXT,
                ],
            }
            pdf_stream.close()
            records.append(record)
            exported_rows += 1
            if exported_rows % SHARD_SIZE == 0:
                shard_id = exported_rows // SHARD_SIZE - 1
                save_shard_to_disk(
                    items=records, dataset_path=test_dir, shard_id=shard_id
                )
                saved_shards += 1
                records = []
        except Exception as ex:
            _log.error(str(ex))
            skipped_rows += 1

    if len(records) > 0:
        shard_id = exported_rows // SHARD_SIZE
        save_shard_to_disk(items=records, dataset_path=test_dir, shard_id=shard_id)
        saved_shards += 1

    if selected_ds_len > 0:
        write_datasets_info(
            name="DocLayNetV1: end-to-end",
            output_dir=output_dir,
            num_train_rows=0,
            num_test_rows=len(records) + exported_rows,
        )

    _log.info(
        "Dataset len: %s. Selected range: [%s, %s]. Exported rows: %s. Skipped_rows: %s. Saved shards: %s",
        total_ds_len,
        begin_index,
        end_index,
        exported_rows,
        skipped_rows,
        saved_shards,
    )
