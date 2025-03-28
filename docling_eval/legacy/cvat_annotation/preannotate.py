import argparse
import glob
import json
import logging
import os
from pathlib import Path
from typing import List

from datasets import load_dataset
from docling_core.types.doc.base import BoundingBox, ImageRefMode
from docling_core.types.doc.document import DocItem, DoclingDocument
from docling_core.types.doc.labels import DocItemLabel, PictureClassificationLabel
from tqdm import tqdm  # type: ignore

from docling_eval.datamodels.types import BenchMarkColumns
from docling_eval.legacy.cvat_annotation.utils import (
    AnnotatedDoc,
    AnnotatedImage,
    AnnotationBBox,
    AnnotationOverview,
    BenchMarkDirs,
    DocLinkLabel,
    TableComponentLabel,
    rgb_to_hex,
)
from docling_eval.utils.utils import get_binhash, insert_images

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def create_cvat_project_properties(project_file: Path):

    results = []

    for item in DocItemLabel:

        r, g, b = DocItemLabel.get_color(item)

        results.append(
            {
                "name": item.value,
                "color": rgb_to_hex(r, g, b),
                "type": "rectangle",
                "attributes": [],
            }
        )

        if item in [DocItemLabel.LIST_ITEM, DocItemLabel.SECTION_HEADER]:
            results[-1]["attributes"].append(
                {
                    "name": "level",
                    "input_type": "number",
                    "mutable": True,
                    "values": ["1", "10", "1"],
                    "default_value": "1",
                }
            )

        if item == [DocItemLabel.FORMULA]:
            results[-1]["attributes"].append(
                {
                    "name": "latex",
                    "mutable": True,
                    "input_type": "text",
                    "mutable": False,
                    "values": [""],
                    "default_value": "",
                }
            )

        if item == [DocItemLabel.CODE]:
            results[-1]["attributes"].append(
                {
                    "name": "code",
                    "mutable": True,
                    "input_type": "text",
                    "mutable": False,
                    "values": [""],
                    "default_value": "",
                }
            )

        if item == DocItemLabel.PICTURE:

            labels = []
            for label in PictureClassificationLabel:
                labels.append(str(label))

            results[-1]["attributes"].append(
                {
                    "name": f"{item.value}-class",
                    "input_type": "select",
                    "mutable": True,
                    "values": labels,
                    "default_value": str(PictureClassificationLabel.OTHER),
                }
            )
            results[-1]["attributes"].append(
                {
                    "name": "json",
                    "mutable": True,
                    "input_type": "text",
                    "mutable": False,
                    "values": [""],
                    "default_value": "",
                }
            )

    for table_item in TableComponentLabel:

        r, g, b = TableComponentLabel.get_color(table_item)

        results.append(
            {
                "name": table_item.value,
                "color": rgb_to_hex(r, g, b),
                "type": "rectangle",
                "attributes": [],
            }
        )

    """    
    for item in TableCellLabel:

        r, g, b  = TableCellLabel.get_color(item)
        
        results.append({
            "name": item.value,
            "color": rgb_to_hex(r, g, b),
            "type": "rectangle",
            "attributes": []
        })
    """

    for link_item in DocLinkLabel:

        r, g, b = DocLinkLabel.get_color(link_item)

        results.append(
            {
                "name": link_item.value,
                "color": rgb_to_hex(r, g, b),
                "type": "polyline",
                "attributes": [],
            }
        )

    logging.info(f"writing project description: {str(project_file)}")
    with open(str(project_file), "w") as fw:
        fw.write(json.dumps(results, indent=2))


def create_cvat_preannotation_file_for_single_page(
    benchmark_dirs: BenchMarkDirs,
    overview: AnnotationOverview,
    bucket_size: int = 200,
):

    cvat_annots: List[str] = []

    img_id, img_cnt, bucket_id, bucket_cnt = 0, 0, 0, 0
    for doc_id, doc_overview in tqdm(
        enumerate(overview.doc_annotations),
        ncols=128,
        desc="creating the CVAT file",
        total=len(overview.doc_annotations),
    ):
        doc = DoclingDocument.load_from_json(doc_overview.pred_file)

        for page_no, page in doc.pages.items():
            img_cnt += 1

            bucket_id = int((img_cnt - 1) / float(bucket_size))
            bucket_dir = benchmark_dirs.tasks_dir / f"task_{bucket_id:02}"

            if (
                not os.path.exists(bucket_dir) and len(cvat_annots) > 0
            ):  # write the pre-annotation files

                logging.info(f"#-annots: {len(cvat_annots)}")

                prev_bucket_id = int((img_cnt - 1) / float(bucket_size))
                preannot_file = (
                    benchmark_dirs.tasks_dir
                    / f"task_{prev_bucket_id:02}_preannotate.xml"
                )

                fw = open(preannot_file, "w")
                fw.write('<?xml version="1.0" encoding="utf-8"?>\n')
                fw.write("<annotations>\n")
                for cvat_annot in cvat_annots:
                    fw.write(f"{cvat_annot}\n")
                fw.write("</annotations>\n")
                fw.close()

                cvat_annots = []

            os.makedirs(bucket_dir, exist_ok=True)

            doc_name = doc_overview.doc_name
            doc_hash = doc_overview.doc_hash

            # annotated_image = doc_overview.model_copy()
            filename = f"doc_{doc_hash}_page_{page_no:06}.png"

            annotated_image = AnnotatedImage(
                img_id=img_cnt,
                mime_type=doc_overview.mime_type,
                true_file=doc_overview.true_file,
                pred_file=doc_overview.pred_file,
                bin_file=doc_overview.bin_file,
                doc_name=doc_name,
                doc_hash=doc_hash,
                bucket_dir=bucket_dir,
                filename=filename,
            )

            annotated_image.img_file = bucket_dir / filename

            page_img_file = benchmark_dirs.page_imgs_dir / filename
            annotated_image.page_img_files = [page_img_file]

            page_image_ref = page.image
            if page_image_ref is not None:

                page_image = page_image_ref.pil_image

                if page_image is not None:
                    page_image.save(annotated_image.img_file)
                    page_image.save(annotated_image.page_img_files[0])

                    annotated_image.img_w = page_image.width
                    annotated_image.img_h = page_image.height

                    annotated_image.page_nos = [page_no]
                    overview.img_annotations[filename] = annotated_image
                else:
                    logging.warning("missing pillow image of the page, skipping ...")
                    continue

            else:
                logging.warning("missing image-ref of the page, skipping ...")
                continue

            page_bboxes: List[AnnotationBBox] = []

            for item, level in doc.iterate_items():
                if isinstance(item, DocItem):
                    for prov in item.prov:
                        if page_no == prov.page_no:

                            page_w = doc.pages[prov.page_no].size.width
                            page_h = doc.pages[prov.page_no].size.height

                            img_w = (
                                annotated_image.img_w
                            )  # page_images[page_no - 1].width
                            img_h = (
                                annotated_image.img_h
                            )  # page_images[page_no - 1].height

                            page_bbox = prov.bbox.to_top_left_origin(page_height=page_h)

                            page_bboxes.append(
                                AnnotationBBox(
                                    bbox_id=len(page_bboxes),
                                    label=item.label,
                                    bbox=BoundingBox(
                                        l=page_bbox.l / page_w * img_w,
                                        r=page_bbox.r / page_w * img_w,
                                        t=page_bbox.t / page_h * img_h,
                                        b=page_bbox.b / page_h * img_h,
                                        coord_origin=page_bbox.coord_origin,
                                    ),
                                )
                            )

            annotated_image.pred_boxes = page_bboxes

            cvat_annots.append(annotated_image.to_cvat())

    if (
        os.path.exists(bucket_dir) and len(cvat_annots) > 0
    ):  # write the pre-annotation files

        # logging.info(f"#-annots: {len(cvat_annots)}")

        preannot_file = (
            benchmark_dirs.tasks_dir / f"task_{bucket_id:02}_preannotate.xml"
        )

        fw = open(preannot_file, "w")
        fw.write('<?xml version="1.0" encoding="utf-8"?>\n')
        fw.write("<annotations>\n")
        for cvat_annot in cvat_annots:
            fw.write(f"{cvat_annot}\n")
        fw.write("</annotations>\n")
        fw.close()

    overview.save_as_json(benchmark_dirs.overview_file)


def export_from_dataset_supplementary_files(
    benchmark_dirs: BenchMarkDirs,
) -> AnnotationOverview:

    test_files = sorted(glob.glob(str(benchmark_dirs.source_dir / "*.parquet")))
    # print(json.dumps(test_files, indent=2))

    ds = load_dataset("parquet", data_files={"test": test_files})
    # logging.info(f"oveview of dataset: {ds}")

    if ds is not None:
        ds_selection = ds["test"]

    overview = AnnotationOverview()

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

        pred_doc = insert_images(
            pred_doc, page_images=page_images, pictures=pics_images
        )

        # Get the groundtruth document (to cherry pick table structure later ...)
        true_doc = pred_doc
        if BenchMarkColumns.GROUNDTRUTH in data:
            true_doc_dict = data[BenchMarkColumns.GROUNDTRUTH]
            true_doc = DoclingDocument.model_validate_json(true_doc_dict)

            true_page_images = data[BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES]
            true_pics_images = data[BenchMarkColumns.GROUNDTRUTH_PICTURES]

            true_doc = insert_images(
                true_doc, page_images=true_page_images, pictures=true_pics_images
            )

        # FIXME: make the unique name in a column
        doc_name = f"{pred_doc.name}"

        bin_doc = data[BenchMarkColumns.ORIGINAL]
        if BenchMarkColumns.DOC_HASH in data:
            doc_hash = data[BenchMarkColumns.DOC_HASH]
        else:
            doc_hash = get_binhash(binary_data=bin_doc)

        # Write groundtruth and predicted document. The groundtruth will
        # be replaced/updated by the annoted ones later on
        true_file = benchmark_dirs.json_true_dir / f"{doc_name}.json"
        true_doc.save_as_json(filename=true_file, image_mode=ImageRefMode.EMBEDDED)

        pred_file = benchmark_dirs.json_pred_dir / f"{doc_name}.json"
        pred_doc.save_as_json(filename=pred_file, image_mode=ImageRefMode.EMBEDDED)

        mime_type = data[BenchMarkColumns.MIMETYPE]

        bin_name = None
        if mime_type == "application/pdf":  # Write original pdf ...
            bin_name = f"{doc_hash}.pdf"
        elif mime_type == "image/png":  # Write original png ...
            bin_name = f"{doc_hash}.png"
        elif mime_type == "image/jpg":  # Write original jpg ...
            bin_name = f"{doc_hash}.jpg"
        else:
            exit(-1)

        bin_file = str(benchmark_dirs.bins_dir / bin_name)
        with open(bin_file, "wb") as fw:
            fw.write(data[BenchMarkColumns.ORIGINAL])

        overview.doc_annotations.append(
            AnnotatedDoc(
                mime_type=mime_type,
                true_file=true_file,
                pred_file=pred_file,
                bin_file=bin_file,
                doc_hash=doc_hash,
                doc_name=doc_name,
            )
        )

    return overview


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process input and output directories and a pre-annotation file."
    )

    parser.add_argument(
        "-i",
        "--input_dir",
        required=True,
        help="Path to the input dataset directory with parquet files.",
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, help="Path to the output directory."
    )
    parser.add_argument(
        "-b", "--bucket-size", required=True, help="Numbers of documents in the bucket."
    )

    args = parser.parse_args()

    return (Path(args.input_dir), Path(args.output_dir), int(args.bucket_size))


def main():

    source_dir, target_dir, bucket_size = parse_args()

    benchmark_dirs = BenchMarkDirs()
    benchmark_dirs.set_up_directory_structure(source=source_dir, target=target_dir)

    create_cvat_project_properties(project_file=benchmark_dirs.project_desc_file)

    overview = export_from_dataset_supplementary_files(benchmark_dirs)

    create_cvat_preannotation_file_for_single_page(
        benchmark_dirs, overview, bucket_size=bucket_size
    )


if __name__ == "__main__":
    main()
