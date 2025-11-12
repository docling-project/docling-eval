import glob
import logging
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from datasets import Dataset, load_dataset
from docling_core.types.doc.document import (
    DEFAULT_EXPORT_LABELS,
    ContentLayer,
    DocItem,
    DoclingDocument,
)
from docling_core.types.doc.labels import DocItemLabel
from tqdm import tqdm  # type: ignore

from docling_eval.datamodels.dataset_record import DatasetRecordWithPrediction
from docling_eval.datamodels.types import BenchMarkColumns, PredictionFormats
from docling_eval.evaluators.base_evaluator import (
    BaseEvaluator,
    EvaluationRejectionType,
    docling_document_from_doctags,
)
from docling_eval.evaluators.layout_evaluator import MissingPredictionStrategy
from docling_eval.evaluators.pixel.multi_label_confusion_matrix import (
    LayoutResolution,
    MultiLabelConfusionMatrix,
)

_log = logging.getLogger(__name__)


class PixelLayoutEvaluator(BaseEvaluator):
    r"""
    Evaluate the document layout by computing a pixel-level confusion matrix and derivative matrices
    (precision, recall, f1).
    """

    def __init__(
        self,
        label_mapping: Optional[Dict[DocItemLabel, Optional[DocItemLabel]]] = None,
        intermediate_evaluations_path: Optional[Path] = None,
        prediction_sources: List[PredictionFormats] = [],
        missing_prediction_strategy: MissingPredictionStrategy = MissingPredictionStrategy.PENALIZE,
    ):
        r"""

        Parameters:
        -----------
        label_mapping: Optional parameter to map DocItemLabels to other DocItemLabels.
                       If a label is mapped to None, it means not to use that label
        """
        supported_prediction_formats: List[PredictionFormats] = [
            PredictionFormats.DOCLING_DOCUMENT,
            PredictionFormats.DOCTAGS,
            PredictionFormats.JSON,
            PredictionFormats.YAML,
        ]
        if not prediction_sources:
            prediction_sources = supported_prediction_formats
        super().__init__(
            intermediate_evaluations_path=intermediate_evaluations_path,
            prediction_sources=prediction_sources,
            supported_prediction_formats=supported_prediction_formats,
        )

        self._missing_prediction_strategy = missing_prediction_strategy

        # Initialize the multi label confusion matrix calculator
        self._mlcm = MultiLabelConfusionMatrix(validation_mode="disabled")

        self._set_categories(label_mapping)

    def _set_categories(
        self,
        label_mapping: Optional[Dict[DocItemLabel, Optional[DocItemLabel]]] = None,
    ):
        r"""
        Set the categories index and reversed index
        """
        label_to_id: dict[str, int] = {
            label: i for i, label in enumerate(DEFAULT_EXPORT_LABELS)
        }

        self._category_name_to_id: Dict[str, int] = {}
        if label_mapping:
            for label in DEFAULT_EXPORT_LABELS:
                if label in label_mapping:
                    mapped_label = label_mapping.get(label)
                    if not mapped_label:  # Skip a label that maps to None
                        continue
                    self._category_name_to_id[label] = label_to_id[mapped_label]
                else:
                    self._category_name_to_id[label] = label_to_id[label]
        else:
            self._category_name_to_id = label_to_id

        self._category_id_to_name: Dict[int, str] = {
            cat_id: cat_name for cat_name, cat_id in self._category_name_to_id.items()
        }

    def __call__(
        self,
        ds_path: Path,
        split: str = "test",
    ):
        _log.info("Loading the split '%s' from: '%s'", split, ds_path)

        # Load the dataset
        split_path = str(ds_path / split / "*.parquet")
        split_files = glob.glob(split_path)
        _log.info("#-files: %s", len(split_files))
        ds = load_dataset("parquet", data_files={split: split_files})
        _log.info("Overview of dataset: %s", ds)

        # Select the split
        ds_selection: Dataset = ds[split]

        # Results containers
        rejected_samples: Dict[EvaluationRejectionType, int] = {
            EvaluationRejectionType.INVALID_CONVERSION_STATUS: 0,
            EvaluationRejectionType.MISSING_PREDICTION: 0,
            EvaluationRejectionType.MISMATHCED_DOCUMENT: 0,
        }
        doc_stats: Dict[str, Dict[str, int]] = {}

        matrix_categories_ids: List[int] = list(self._category_id_to_name.keys())
        num_categories = len(matrix_categories_ids)
        confusion_matrix_sum = np.zeros((num_categories, num_categories))

        for i, data in tqdm(
            enumerate(ds_selection),
            desc="Layout evaluations",
            ncols=120,
            total=len(ds_selection),
        ):
            data_record = DatasetRecordWithPrediction.model_validate(data)
            doc_id = data_record.doc_id
            if data_record.status not in self._accepted_status:
                _log.error(
                    "Skipping record without successfull conversion status: %s", doc_id
                )
                rejected_samples[EvaluationRejectionType.INVALID_CONVERSION_STATUS] += 1
                continue

            true_doc = data_record.ground_truth_doc
            pred_doc = self._get_pred_doc(data_record)
            if not pred_doc:
                _log.error("There is no prediction for doc_id=%s", doc_id)
                rejected_samples[EvaluationRejectionType.MISSING_PREDICTION] += 1
                continue

            # TODO: Check in the end if to optimize the memory allocation for the intermediate CMs
            doc_cm = self._compute_document_confusion_matrix(true_doc, pred_doc)

            # TODO: Check if to compute metrics per document
            confusion_matrix_sum += doc_cm

        # TODO: Compute metrics
        ds_metrics = self._mlcm.compute_metrics(
            confusion_matrix_sum,
            self._category_id_to_name,
            True,
        )

    def _compute_document_confusion_matrix(
        self,
        true_doc: DoclingDocument,
        pred_doc: DoclingDocument,
    ) -> np.ndarray:
        r"""
        Compute the confusion matrix for the given documents.
        This is the sum of the confusion matrices of the document pages.
        """

        # Collect all DocItems by page for both GT and predictions
        true_pages_to_objects = self._collect_items_by_page(true_doc)
        pred_pages_to_objects = self._collect_items_by_page(pred_doc)

        # Get all pages that have GT data (we evaluate based on GT pages)
        gt_pages = set(true_pages_to_objects.keys())
        pred_pages = set(pred_pages_to_objects.keys())
        _log.debug(f"GT pages: {sorted(gt_pages)}, Pred pages: {sorted(pred_pages)}")

        matrix_categories_ids: List[int] = list(self._category_id_to_name.keys())
        num_categories = len(matrix_categories_ids)
        off_diagonal_cells = num_categories * num_categories - num_categories
        confusion_matrix_sum = np.zeros((num_categories, num_categories))
        # num_images = 0
        # num_pixels = 0
        # all_image_metrics: dict[str, dict] = {}  # image_filename -> image_metrics

        for page_no in sorted(gt_pages):
            page_size = true_doc.pages[page_no].size
            pg_width = page_size.width
            pg_height = page_size.height

            # Always process GT for this page
            gt_layouts = self._get_page_layout_resolution(
                page_no=page_no,
                items=true_pages_to_objects[page_no],
                doc=true_doc,
            )

            # Handle prediction for this page based on strategy
            if page_no in pred_pages:
                # We have prediction data for this page
                pred_layouts = self._get_page_layout_resolution(
                    page_no=page_no,
                    items=pred_pages_to_objects[page_no],
                    doc=pred_doc,
                )

                # Compute the confusion matrix
                gt_binary = self._mlcm.make_binary_representation(
                    pg_width, pg_height, gt_layouts
                )
                preds_binary = self._mlcm.make_binary_representation(
                    pg_width, pg_height, pred_layouts
                )
                confusion_matrix_sum += self._mlcm.generate_confusion_matrix(
                    gt_binary, preds_binary, matrix_categories_ids
                )
            else:
                # No prediction data for this page
                if (
                    self._missing_prediction_strategy
                    == MissingPredictionStrategy.PENALIZE
                ):
                    # Create a penalty confusion matrix
                    image_pixels = pg_width * pg_height
                    penalty_value = image_pixels / off_diagonal_cells
                    confusion_matrix_sum += penalty_value * (
                        np.ones((num_categories, num_categories))
                        - np.eye(num_categories)
                    )
                elif (
                    self._missing_prediction_strategy
                    == MissingPredictionStrategy.IGNORE
                ):
                    # Skip this page entirely
                    continue
                else:
                    raise ValueError(
                        f"Unknown missing prediction strategy: {self._missing_prediction_strategy}"
                    )
        return confusion_matrix_sum

    def _get_page_layout_resolution(
        self,
        page_no: int,
        items: List[DocItem],
        doc: DoclingDocument,
    ) -> List[LayoutResolution]:
        r"""
        Generate a list of LayoutResolution objects for the given document page
        Each LayoutResolution corresponds to one bbox and its category_id
        """
        page_size = doc.pages[page_no].size
        page_height = page_size.height

        resolutions: List[LayoutResolution] = []
        for item in items:
            for prov in item.prov:
                if prov.page_no != page_no:
                    # Only process provenances for this specific page
                    continue

                category_id = self._category_name_to_id[item.label]
                bbox: List[int] = list(
                    prov.bbox.to_top_left_origin(page_height=page_height).as_tuple()
                )
                resolutions.append(LayoutResolution(category_id=category_id, bbox=bbox))
        return resolutions

    def _collect_items_by_page(
        self,
        doc: DoclingDocument,
    ) -> Dict[int, List[DocItem]]:
        """
        Collect DocItems by page number for the given document and filter labels.

        Args:
            doc: The DoclingDocument to process

        Returns:
            Dictionary mapping page numbers to lists of DocItems
        """
        pages_to_objects: Dict[int, List[DocItem]] = defaultdict(list)

        for item, level in doc.iterate_items(
            included_content_layers={
                c for c in ContentLayer if c != ContentLayer.BACKGROUND
            },
            traverse_pictures=True,
            with_groups=True,
        ):
            if isinstance(item, DocItem):
                for prov in item.prov:
                    pages_to_objects[prov.page_no].append(item)

        return pages_to_objects

    def _get_pred_doc(
        self, data_record: DatasetRecordWithPrediction
    ) -> Optional[DoclingDocument]:
        r"""
        Get the predicted DoclingDocument
        """
        # TODO: Duplicated code from LayoutEvaluator
        pred_doc = None
        for prediction_format in self._prediction_sources:
            if prediction_format == PredictionFormats.DOCLING_DOCUMENT:
                pred_doc = data_record.predicted_doc
            elif prediction_format == PredictionFormats.JSON:
                if data_record.original_prediction:
                    pred_doc = DoclingDocument.load_from_json(
                        data_record.original_prediction
                    )
            elif prediction_format == PredictionFormats.YAML:
                if data_record.original_prediction:
                    pred_doc = DoclingDocument.load_from_yaml(
                        data_record.original_prediction
                    )
            elif prediction_format == PredictionFormats.DOCTAGS:
                pred_doc = docling_document_from_doctags(data_record)
            if pred_doc is not None:
                break

        return pred_doc
