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
from docling_ibm_models.layoutmodel.labels import LayoutLabels
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


def category_name_to_docitemlabel(category_name: str) -> DocItemLabel:
    r""" """
    label = DocItemLabel(category_name.lower().replace(" ", "_").replace("-", "_"))
    return label


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

        # Initialize the mappings between DocItemLabel <-> category_id <-> category_name
        self._matrix_doclabelitem_to_id: Dict[
            DocItemLabel, int
        ]  # DocLabelItem to cat_id (shifted to make space for Background)
        self._matrix_id_to_name: Dict[
            int, str
        ]  # shifted cat_id to string (to include Background)
        self._matrix_doclabelitem_to_id, self._matrix_id_to_name = (
            self._build_matrix_categories(label_mapping)
        )

    def _build_matrix_categories(
        self,
        label_mapping: Optional[Dict[DocItemLabel, Optional[DocItemLabel]]] = None,
    ) -> Tuple[
        Dict[DocItemLabel, int],
        Dict[int, str],
    ]:
        r"""
        Create mappings for the matrix categories including the background (shifted) while taking
        into account the label_mappings:

        Returns:
        --------
        matrix_doclabelitem_to_id: Dict[DocItemLabel, int]
            From DocItemLabel to shifted category_id (the values do NOT contain zero)
            If the label_mapping maps to None, this entry is omitted

        matrix_id_to_name: Dict[int, str]
            From shifted_category_id to string.
            For key==0 the value is Background, otherwise the value of the corresponding DocItemLabel
            taking into account any label mapping
            If the label_mapping maps to None, this entry is omitted

        """
        layout_labels = LayoutLabels()

        # Auxiliary mapping: DocItemLabel -> canonical_category_id
        canonical_to_id: Dict[str, int] = layout_labels.canonical_to_int()
        label_to_id: Dict[DocItemLabel, int] = {
            DocItemLabel(cat_name.lower().replace(" ", "_").replace("-", "_")): cat_id
            for cat_name, cat_id in canonical_to_id.items()
        }

        # Populate the matrix_doclabelitem_to_id
        matrix_doclabelitem_to_id: Dict[DocItemLabel, int] = (
            {}
        )  # The values are shifted (not including zero)
        label_id_offset = 1

        # TODO: If label_mappings are provided, we end up having more than one DocItemLabel with the same cat_id
        for label, canonical_cat_id in label_to_id.items():
            effective_label = label
            if label_mapping and label in label_mapping:
                effective_label = label_mapping.get(label)
                if not effective_label:  # Skip labels that map to None
                    continue
            matrix_doclabelitem_to_id[label] = (
                label_to_id[effective_label] + label_id_offset
            )

        # Populate the matrix_id_to_name
        matrix_id_to_name: Dict[int, str] = {}  # The keys start from 0 to include BG
        shifted_canonical: Dict[int, str] = layout_labels.shifted_canonical_categories()

        # TODO: If label_mappings are provided we end up having more than 1 cat_id with the same name
        for shifted_cat_id, cat_name in shifted_canonical.items():
            label = None
            if cat_name != shifted_canonical[0]:
                label = category_name_to_docitemlabel(cat_name)
                if label_mapping and label in label_mapping:
                    label = label_mapping.get(label)
                    if not label:  # Skip labels that map to None
                        continue
            matrix_id_to_name[shifted_cat_id] = label.value if label else cat_name

        return matrix_doclabelitem_to_id, matrix_id_to_name

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

        matrix_categories_ids: List[int] = list(self._matrix_id_to_name.keys())
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
            self._matrix_id_to_name,
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

        matrix_categories_ids: List[int] = list(self._matrix_id_to_name.keys())
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
                    # Create a penalty confusion matrix by distributing all pixels outside of diagonal
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

                category_id = self._matrix_doclabelitem_to_id[item.label]
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
