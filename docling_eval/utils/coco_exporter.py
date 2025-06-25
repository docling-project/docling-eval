import argparse
import glob
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import Dataset, load_dataset
from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.document import DocItem, DoclingDocument
from docling_core.types.doc.labels import DocItemLabel
from pycocotools.coco import COCO
from tqdm import tqdm  # type: ignore

from docling_eval.datamodels.dataset_record import DatasetRecordWithPrediction

_log = logging.getLogger(__name__)


# If the original COCO dataset does not call all categories (e.g. DLNv1), the mapping is ignored
DOCLING_LABELS_TO_COCO_CATEGORIES: Dict[DocItemLabel, str] = {
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


def load_coco(split: str, coco_dir: Path) -> COCO:
    r"""Load the COCO dataset"""
    coco_ann_dir = coco_dir / "annotations"
    if not coco_ann_dir.is_dir():
        raise Exception(f"COCO annotations dir is missing: {str(coco_ann_dir)}")

    coco_ann_fns = detect_coco_annotation_files(coco_ann_dir)
    if split not in coco_ann_fns:
        raise Exception(f"Split '{split}' does not exist in COCO dataset")

    coco = COCO(coco_ann_fns[split])
    return coco


class DoclingEvalCOCOExporter:
    r"""
    Receive as input an HF parquet dataset with GT and predictions in DoclingDocument format
    Export the predictions as a json file in the format expected by pycocotools
    """

    def __init__(self, docling_eval_ds_path: Path):
        r""" """
        self._docling_eval_ds_path = docling_eval_ds_path

    def export_COCO_and_predictions(
        self,
        save_dir: Path,
    ):
        r"""
        Export a docling-eval HF parquet dataset in COCO:
        - Export gt_doc in COCO format.
        - Export pred_doc in pycocotools json format.
        """
        # TODO
        pass

    def export_predictions_wrt_original_COCO(
        self,
        split: str,
        save_dir: Path,
        original_coco_dir: Path,
        labels_to_categories: Dict[DocItemLabel, str],
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
        # Load the COCO dataset
        _log.info("Loading COCO dataset")
        coco = load_coco(split, original_coco_dir)
        coco_dataset = coco.dataset

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
            for label, category in labels_to_categories.items()
            if category in category_to_id
        }

        # Load the HF dataset
        split_path = str(self._docling_eval_ds_path / split / "*.parquet")
        split_files = glob.glob(split_path)
        ds = load_dataset("parquet", data_files={split: split_files})
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

            # Extract labels, bboxes, scores
            category_ids, scores, bboxes = self._extract_layout_data(
                pred_doc, labels_to_category_ids
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

    def _extract_layout_data(
        self,
        pred_doc: DoclingDocument,
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

        for item, _ in pred_doc.iterate_items():
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

                page_w, page_h = pred_doc.pages[page_no].size.as_tuple()

                # Save bbox in COCO format
                bbox: BoundingBox = prov.bbox.to_top_left_origin(page_height=page_h)
                bboxes.append([bbox.l, bbox.t, bbox.width, bbox.height])

                scores.append(1.0)
                category_ids.append(category_id)
        return category_ids, scores, bboxes


def main(args):
    r""" """
    # Get args
    docling_eval_path = Path(args.docling_eval_dir)
    coco_path = Path(args.coco_dir)
    save_path = Path(args.save_dir)

    # Setup logger
    logging.getLogger("docling").setLevel(logging.WARNING)
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    # Create the COCO exporter
    exporter = DoclingEvalCOCOExporter(docling_eval_path)
    exporter.export_predictions_wrt_original_COCO(
        "test",
        save_path,
        coco_path,
        DOCLING_LABELS_TO_COCO_CATEGORIES,
    )


if __name__ == "__main__":
    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--docling_eval_dir",
        required=False,
        help="Root dir with the docling-eval parquet dataset with the predictions",
    )
    parser.add_argument(
        "-c", "--coco_dir", required=False, help="Root dir of the COCO dataset"
    )
    parser.add_argument(
        "-s", "--save_dir", required=True, help="Output directory to save files"
    )
    args = parser.parse_args()
    main(args)
