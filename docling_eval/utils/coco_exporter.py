import argparse
import glob
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset
from docling_core.types.doc.base import BoundingBox, Size
from docling_core.types.doc.document import (
    ContentLayer,
    DocItem,
    DoclingDocument,
    PageItem,
)
from docling_core.types.doc.labels import DocItemLabel
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm  # type: ignore

from docling_eval.datamodels.dataset_record import DatasetRecordWithPrediction

_log = logging.getLogger(__name__)


# If the original COCO dataset does not call all categories (e.g. DLNv1), the mapping is ignored
VALID_DOCLING_LABELS_TO_COCO_CATEGORIES: Dict[DocItemLabel, str] = {
    DocItemLabel.CAPTION: "Caption",
    DocItemLabel.FOOTNOTE: "Footnote",
    DocItemLabel.FORMULA: "Formula",
    DocItemLabel.LIST_ITEM: "List-item",
    DocItemLabel.PAGE_FOOTER: "Page-footer",
    DocItemLabel.PAGE_HEADER: "Page-header",
    DocItemLabel.PICTURE: "Picture",
    DocItemLabel.SECTION_HEADER: "Section-header",
    DocItemLabel.TABLE: "Table",
    DocItemLabel.TEXT: "Text",
    DocItemLabel.TITLE: "Title",
    DocItemLabel.DOCUMENT_INDEX: "Document Index",
    DocItemLabel.CODE: "Code",
    DocItemLabel.CHECKBOX_SELECTED: "Checkbox-Selected",
    DocItemLabel.CHECKBOX_UNSELECTED: "Checkbox-Unselected",
    DocItemLabel.FORM: "Form",
    DocItemLabel.KEY_VALUE_REGION: "Key-Value Region",
}


def detect_coco_annotation_files(
    annotations_dir: Path,
    split_options: List[str] = ["train", "val", "test"],
) -> Dict[str, Path]:
    """
    Scan a directory and detect the json files for the COCO annotations.
    """
    split_files: Dict[str, Path] = {}
    for file_path in annotations_dir.iterdir():
        if not file_path.is_file():
            continue
        if file_path.suffix != ".json":
            continue

        for split_name in split_options:
            if split_name in file_path.stem:
                split_files[split_name] = file_path
    return split_files


def detect_coco_images_split_dirs(
    coco_root: Path,
    split_options: List[str] = ["train", "val", "test"],
) -> Dict[str, Path]:
    r"""
    Scan the coco_root path and detect the directory that has the images for the given split
    """
    images_split_dirs: Dict[str, Path] = {}
    for images_dir in coco_root.iterdir():
        if not images_dir.is_dir():
            continue
        for split_name in split_options:
            if split_name in images_dir.stem:
                images_split_dirs[split_name] = images_dir
    return images_split_dirs


def load_coco_annotations(split: str, coco_dir: Path) -> COCO:
    r"""Load the COCO dataset"""
    coco_ann_dir = coco_dir / "annotations"
    if not coco_ann_dir.is_dir():
        raise Exception(f"COCO annotations dir is missing: {str(coco_ann_dir)}")

    coco_ann_fns = detect_coco_annotation_files(coco_ann_dir)
    if split not in coco_ann_fns:
        raise Exception(f"Split '{split}' does not exist in COCO dataset")

    ann_fn = coco_ann_fns[split]
    with open(ann_fn, "r") as fd:
        ann_data = json.load(fd)
    return ann_data


class DoclingEvalCOCOExporter:
    r"""
    Receive as input an HF parquet dataset with GT and predictions in DoclingDocument format
    Export the predictions as a json file in the format expected by pycocotools
    """

    def __init__(self, docling_eval_ds_path: Path):
        r""" """
        self._docling_eval_ds_path = docling_eval_ds_path

        # Tolerance in size diff between page size and page image size measured in pixels
        self._page_image_pixels_tolerance = 2

    def export_COCO_and_predictions(
        self,
        split: str,
        save_dir: Path,
    ):
        r"""
        Export a docling-eval HF parquet dataset in COCO:
        - Export gt_doc in COCO format.
        - Export pred_doc in pycocotools json format.
        """
        # TODO
        pass

    def export_COCO(
        self,
        split: str,
        save_dir: Path,
        extra_doc_label_to_valid_label_mapping: dict[
            DocItemLabel, Optional[DocItemLabel]
        ],
        source_doc_column: str = "GT",
    ):
        r"""
        Export COCO dataset

        Parameters
        ----------
        save_dir: Location to save the exported COCO dataset
        split: COCO split to be created: One of ['train', 'test', 'val']
        doc_label_to_valid_label_mapping: Exta mappings from docling document to valid docling labels.
                                          If a mapping value is None, it means to ignore this key.
        source_doc_column: Which column from the parquet file should be used to generate the COCO dataset.
                           It should be one of ["GT", "pred"]. By default "GT"
        """
        # Build the info and licenses
        info: dict = self._build_info()
        licenses: list[dict] = self._build_licenses()

        # Generate mapping from document labels to category_id
        label_to_category_id: dict[DocItemLabel, int] = {
            label: cat_id
            for cat_id, (label, category) in enumerate(
                VALID_DOCLING_LABELS_TO_COCO_CATEGORIES.items()
            )
        }
        # Apply the corrections given in the doc_label_to_valid_label_mapping
        for doc_label, valid_label in extra_doc_label_to_valid_label_mapping.items():
            if valid_label is not None:
                label_to_category_id[doc_label] = label_to_category_id[valid_label]
            elif doc_label in label_to_category_id:
                del label_to_category_id[doc_label]

        # Build the categories
        categories: list[dict] = self._build_categories()

        # Get the images dir
        images_dir = save_dir / split
        images_dir.mkdir(parents=True, exist_ok=True)

        # Build images and annotations
        images: list[dict] = []
        anns: list[dict] = []
        ds = self._load_ds(split)
        ds_selection = ds[split]
        image_id = 0
        annotation_id = 0
        for i, data in enumerate(ds_selection):
            data_record = DatasetRecordWithPrediction.model_validate(data)
            doc_id = data_record.doc_id

            if data_record.predicted_doc is not None and source_doc_column == "pred":
                doc = data_record.predicted_doc
                _log.info("Dataset document to export: 'predicted_doc'")
            else:
                doc = data_record.ground_truth_doc
                _log.info("Dataset document to export: 'ground_truth_doc'")

            # Convert the doc in a COCO-dataset
            doc_images: list[dict]
            doc_anns: list[dict]
            doc_images, doc_anns, image_id, annotation_id = (
                self._extract_layout_coco_annotations(
                    doc_id,
                    doc,
                    label_to_category_id,
                    images_dir,
                    image_id,
                    annotation_id,
                )
            )
            images.extend(doc_images)
            anns.extend(doc_anns)

        # Save the annotations
        annotations: dict = {
            "info": info,
            "categories": categories,
            "images": images,
            "annotations": anns,
            "licenses": licenses,
        }
        annotations_dir = save_dir / "annotations"
        annotations_dir.mkdir(parents=True, exist_ok=True)
        annotations_fn = annotations_dir / f"{split}2017.json"
        _log.info("Saving the exported COCO annotations in: %s", str(annotations_fn))
        with open(annotations_fn, "w") as fd:
            json.dump(annotations, fd)

        return annotations

    def _extract_layout_coco_annotations(
        self,
        doc_id: str,
        doc: DoclingDocument,
        labels_to_category_ids: Dict[DocItemLabel, int],
        images_dir: Path,
        image_id_offset: int,
        annotation_id_offset: int,
    ) -> Tuple[List[Dict], List[Dict], int, int]:
        r"""
        Extract layout information from DoclingDocument into coco-tools format

        Returns
        -------
        images: List of dict in COCO format with the images in the document
        annotations: List of dict in COCO format with the annotations in the document
        last_image_id: The last image id generated by that document
        last_annotation_id: The last annotation_id generated by that document
        """
        doc_images: list[dict] = []  # Images of the document
        doc_anns: list[dict] = []  # Annotations of the document

        included_content_layers = {c for c in ContentLayer}
        image_id = image_id_offset
        annotation_id = annotation_id_offset
        for item, _ in doc.iterate_items(
            included_content_layers=included_content_layers
        ):
            if not isinstance(item, DocItem):
                continue
            label = item.label
            category_id = labels_to_category_ids.get(label, -1)

            # Skip label without mapping into a COCO categories
            if category_id == -1:
                _log.warning(
                    "Skip prediction with label that does not map to COCO categories: '%s'",
                    label,
                )
                continue

            # Use only the first provenance of the item
            if len(item.prov) == 0:
                _log.error("Skip item without provenance: %s: %s", doc_id, label.value)
                continue

            prov = item.prov[0]
            page_no = prov.page_no
            page = doc.pages[page_no]
            page_size = page.size

            # Save the page image
            if page.image is not None and page_no > len(doc_images):
                img: Image.Image = page.image.pil_image  # type: ignore
                if img:
                    # Check the tolerance for the page/image size mismatch
                    page_size = self._check_page_image_size(page)
                    if not page_size:
                        continue

                    image_filename = (
                        f"{doc_id}.png"
                        if "page" in doc_id
                        else f"{doc_id}_page_{page_no:06d}.png"
                    )
                    image_fn = images_dir / image_filename
                    _log.info("Saving image: %s", str(image_fn))
                    img.save(image_fn)

                    doc_images.append(
                        {
                            "licence": 1,
                            "file_name": image_filename,
                            "height": img.height,
                            "width": img.width,
                            "id": image_id,
                        }
                    )
                    image_id += 1

            # Get the bbox in [x,y,w,h] COCO format
            bbox: BoundingBox = prov.bbox
            bbox = bbox.to_top_left_origin(page_height=page_size.height)
            doc_anns.append(
                {
                    "image_id": image_id - 1,
                    "category_id": category_id,
                    "bbox": [bbox.l, bbox.t, bbox.width, bbox.height],
                    "iscrowd": 0,
                    "area": bbox.area(),
                    "id": annotation_id,
                }
            )
            annotation_id += 1

        return doc_images, doc_anns, image_id, annotation_id

    def _build_licenses(self) -> list[dict]:
        r""" """
        license = {
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
        }
        return [license]

    def _load_ds(self, split: str) -> Dataset:
        r"""Load the dataset from the parquet files"""
        split_path = str(self._docling_eval_ds_path / split / "*.parquet")
        split_files = glob.glob(split_path)
        ds = load_dataset("parquet", data_files={split: split_files})
        return ds

    def _build_info(self):
        r""" """
        info = {
            "description": "COCO 2017 Dataset",
            "url": "http://cocodataset.org",
            "version": "1.0",
            "year": 2017,
            "contributor": "COCO Consortium",
            "date_created": "2017/09/01",
        }
        return info

    def _build_categories(
        self,
        supercategory: str = "DoclingDocument",
    ) -> list[dict]:
        r""" """
        categories: list[dict] = []
        for cat_id, (label, category_name) in enumerate(
            VALID_DOCLING_LABELS_TO_COCO_CATEGORIES.items()
        ):
            categories.append(
                {
                    "supercategory": supercategory,
                    "id": cat_id,
                    "name": category_name,
                }
            )
        return categories

    def export_predictions_wrt_original_COCO(
        self,
        split: str,
        save_dir: Path,
        original_coco_dir: Path,
    ) -> List[Dict]:
        r"""
        Export the predictions as a json file in pycocotools format:

        ```
        [
            {
                "image_id": int,           // ID of the image as in the COCO dataset
                "category_id": int,        // Category ID predicted
                "bbox": [x, y, width, height], // Bounding box in COCO format
                "score": float             // Confidence score of the prediction
            },
            ...
        ]
        ```

        Original COCO dataset
        ---------------------
        We assume that the docling-eval HF dataset originates from an external COCO dataset if both
        conditions are met:
        1. The `document_id` field of the HF dataset matches an image filename of the COCO dataset.
        2. The DocItemLabel labels appearing in pred_doc can be mapped to a category from
           the COCO dataset using the provided labels_to_categories mapping.

        Return
        ------
        Generate a Dict in pycocotools format using the "image_id", "category_id" from the original
        COCO dataset. Dump it as a json file.
        """
        # Ensure save_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        # Load the COCO dataset
        _log.info("Loading COCO dataset")
        coco_dataset = load_coco_annotations(split, original_coco_dir)
        coco_image_dirs = detect_coco_images_split_dirs(original_coco_dir)
        coco_image_dir = coco_image_dirs[split]

        # Build indices for COCO images/categories
        img_name_to_id_idx: Dict[str, int] = {
            Path(img_info["file_name"]).stem: img_info["id"]  # filename stem -> imag_id
            for img_info in coco_dataset["images"]
        }
        category_to_id: Dict[str, int] = {
            cat_info["name"]: cat_info["id"] for cat_info in coco_dataset["categories"]
        }
        labels_to_category_ids: Dict[DocItemLabel, int] = {
            label: category_to_id[category]
            for label, category in VALID_DOCLING_LABELS_TO_COCO_CATEGORIES.items()
            if category in category_to_id
        }

        # Load the HF dataset
        ds = self._load_ds(split)
        ds_selection: Dataset = ds[split]

        # Debug
        # ds_selection = ds_selection.select(range(0, 10))

        # Build the predictions
        predictions: List[Dict] = []
        for i, data in tqdm(
            enumerate(ds_selection),
            desc="Export HF predictions to pycocotools format",
            ncols=120,
            total=len(ds_selection),
        ):
            data_record = DatasetRecordWithPrediction.model_validate(data)
            doc_id = data_record.doc_id
            pred_doc = data_record.predicted_doc
            assert pred_doc is not None

            # Check if the doc_id exists as an image_filename in COCO
            if doc_id not in img_name_to_id_idx:
                continue
            img_id = img_name_to_id_idx[doc_id]

            # Load the COCO image to get the original image dimensions
            coco_image_fn = coco_image_dir / f"{doc_id}.png"
            if not coco_image_fn.is_file():
                _log.error(
                    "Skipping document because COCO image is missing: %s", doc_id
                )
                continue
            with Image.open(coco_image_fn) as im:
                coco_img_width = im.width
                coco_img_height = im.height

            # Extract labels, bboxes, scores
            category_ids, scores, bboxes = self._extract_layout_predictions(
                doc_id,
                pred_doc,
                coco_img_width,
                coco_img_height,
                labels_to_category_ids,
            )
            for (
                category_id,
                score,
                bbox,
            ) in zip(category_ids, scores, bboxes):
                predictions.append(
                    {
                        "image_id": img_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": score,
                    }
                )

        # Save the predictions
        prediction_fn = save_dir / f"coco_predictions.json"
        _log.info("Saving COCO predictions: %s", prediction_fn)
        with open(prediction_fn, "w") as fd:
            json.dump(predictions, fd)
        return predictions

    def _extract_layout_predictions(
        self,
        doc_id: str,
        pred_doc: DoclingDocument,
        coco_img_width: int,
        coco_img_height: int,
        labels_to_category_ids: Dict[DocItemLabel, int],
    ) -> Tuple[List[int], List[float], List[List[float]]]:
        r"""
        Extract the layout data

        Returns
        -------
        categories_ids, scores, bboxes
        """
        category_ids: List[int] = []
        scores: List[float] = []
        bboxes: List[List[float]] = []  # [x,y,w,h] COCO format
        new_size = Size(width=coco_img_width, height=coco_img_height)
        included_content_layers = {c for c in ContentLayer}
        for item, _ in pred_doc.iterate_items(
            included_content_layers=included_content_layers
        ):
            if not isinstance(item, DocItem):
                continue
            label = item.label
            category_id = labels_to_category_ids.get(label, -1)

            # Skip label without mapping into a COCO categories
            if category_id == -1:
                _log.error(
                    "Skip prediction with label that does not map to COCO categories: '%s'",
                    label,
                )
                continue

            for prov in item.prov:
                page_no = prov.page_no
                if page_no != 1:
                    _log.error("Skip pages after the first one")
                    continue
                old_size = pred_doc.pages[page_no].size

                # Scale bbox to the original COCO image dimensions and save in COCO format
                bbox: BoundingBox = prov.bbox.to_top_left_origin(
                    page_height=old_size.height
                )
                if old_size != new_size:
                    bbox = bbox.scale_to_size(old_size, new_size)
                bboxes.append([bbox.l, bbox.t, bbox.width, bbox.height])

                scores.append(1.0)
                category_ids.append(category_id)
        return category_ids, scores, bboxes

    def _check_page_image_size(self, page: PageItem) -> Optional[Size]:
        r"""
        Check if the page size and page image size are within the allowed tolerance
        If the tolerance is respected, return the smaller size, otherwise return None
        """
        page_size = page.size
        img: Image.Image = page.image.pil_image  # type: ignore
        if (
            abs(img.width - page_size.width) > self._page_image_pixels_tolerance
            or abs(img.height - page_size.height) > self._page_image_pixels_tolerance
        ):
            _log.error(
                "Page/image size diff exceeds tolerance (%f): (%d, %d) vs (%d, %d)",
                self._page_image_pixels_tolerance,
                page_size.width,
                page_size.height,
                img.width,
                img.height,
            )
            return None
        return Size(
            width=min(page_size.width, img.width),
            height=min(page_size.height, img.height),
        )


def main():
    r""" """
    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--operation",
        required=True,
        type=str,
        help="Operation to perform. One of ['coco']",
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        required=True,
        type=Path,
        help="Output directory to save files",
    )
    parser.add_argument(
        "-d",
        "--docling_eval_dir",
        required=True,
        type=Path,
        help="Root dir with the docling-eval parquet dataset with the predictions",
    )
    parser.add_argument(
        "-c",
        "--coco_dir",
        required=False,
        type=Path,
        help="Root dir of the COCO dataset",
    )
    args = parser.parse_args()

    # Setup logger
    logging.getLogger("docling").setLevel(logging.WARNING)
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    _log.info("Operation: %s", args.operation)
    _log.info("eval-dataset: %s", str(args.docling_eval_dir))
    _log.info("Save path: %s", str(args.save_dir))

    # Create the COCO exporter
    exporter = DoclingEvalCOCOExporter(args.docling_eval_dir)

    # Run the operation
    if args.operation.upper() == "COCO":
        # Mapping from the parquet document label to the valid docling labels
        doc_label_to_valid_label_mapping: dict[DocItemLabel, DocItemLabel] = {
            DocItemLabel.PAGE_FOOTER: DocItemLabel.TEXT,
            DocItemLabel.PAGE_HEADER: DocItemLabel.TEXT,
            DocItemLabel.HANDWRITTEN_TEXT: DocItemLabel.PICTURE,
            DocItemLabel.EMPTY_VALUE: None,
            DocItemLabel.KEY_VALUE_REGION: None,
            DocItemLabel.PARAGRAPH: DocItemLabel.TEXT,
            DocItemLabel.REFERENCE: DocItemLabel.TEXT,
        }
        exporter.export_COCO(
            "test",
            args.save_dir,
            doc_label_to_valid_label_mapping,
        )
    elif args.operation.upper() == "PREDICTIONS":
        exporter.export_predictions_wrt_original_COCO(
            "test",
            args.save_dir,
            args.coco_dir,
        )
    else:
        raise ValueError(f"Not supported operation: {args.operation}")


if __name__ == "__main__":
    main()
