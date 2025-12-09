import logging
from pathlib import Path
from typing import List, Optional, Tuple

from datasets import Dataset, load_dataset
from docling.datamodel.base_models import ConversionStatus
from docling_core.types.doc.document import DoclingDocument
from PIL import Image
from tqdm import tqdm  # type: ignore

from docling_eval.datamodels.dataset_record import DatasetRecordWithPrediction
from docling_eval.datamodels.types import BenchMarkColumns, PredictionFormats
from docling_eval.utils.external_docling_document_loader import (
    ExternalDoclingDocumentLoader,
)
from docling_eval.utils.utils import extract_images, insert_images_from_pil
from docling_eval.visualisation.visualisations import save_comparison_html_with_clusters

_LOGGER = logging.getLogger(__name__)


class PredictionsVisualizer:
    """
    Render ground-truth vs. prediction visualizations for an existing dataset.

    Works with either:
    - A dataset that already embeds predictions (DatasetRecordWithPrediction parquet)
    - A ground-truth-only dataset paired with an external predictions directory
      containing DoclingDocument files named <doc_id>.[json|dt|yaml|yml]
    """

    def __init__(
        self,
        visualizations_dir: Path,
        *,
        external_predictions_dir: Optional[Path] = None,
        ignore_missing_predictions: bool = False,
    ):
        self._loader = (
            ExternalDoclingDocumentLoader(external_predictions_dir)
            if external_predictions_dir is not None
            else None
        )
        self._visualizations_dir = visualizations_dir
        self._ignore_missing_predictions = ignore_missing_predictions

    def create_visualizations(
        self,
        dataset_dir: Path,
        split: str = "test",
        begin_index: int = 0,
        end_index: int = -1,
    ) -> None:
        """
        Generate paired HTML visualizations between ground truth and predictions.
        """
        dataset = self._load_split(dataset_dir, split)
        dataset = self._slice_dataset(dataset, begin_index, end_index)
        self._visualizations_dir.mkdir(parents=True, exist_ok=True)

        for _, row in tqdm(
            enumerate(dataset),
            desc="Rendering visualizations",
            total=len(dataset),
            ncols=120,
        ):
            record = DatasetRecordWithPrediction.model_validate(row)
            pred_doc = self._resolve_prediction_document(record)
            if pred_doc is None:
                message = f"Missing prediction for document {record.doc_id}"
                if self._ignore_missing_predictions:
                    _LOGGER.warning(message)
                    continue
                raise FileNotFoundError(message)

            pred_doc, pred_pictures, pred_page_images = self._prepare_prediction_assets(
                record, pred_doc
            )

            record_for_viz = record.model_copy(deep=True)
            record_for_viz.predicted_doc = pred_doc
            record_for_viz.predicted_pictures = pred_pictures
            record_for_viz.predicted_page_images = pred_page_images
            record_for_viz.prediction_format = PredictionFormats.DOCLING_DOCUMENT
            record_for_viz.status = ConversionStatus.SUCCESS

            self._save_visualization(record_for_viz)

    def _resolve_prediction_document(
        self, record: DatasetRecordWithPrediction
    ) -> Optional[DoclingDocument]:
        if self._loader is not None:
            return self._loader(record)
        return record.predicted_doc

    def _prepare_prediction_assets(
        self, record: DatasetRecordWithPrediction, pred_doc: DoclingDocument
    ) -> Tuple[DoclingDocument, List[Image.Image], List[Image.Image]]:
        if self._loader is None and (
            record.predicted_pictures or record.predicted_page_images
        ):
            return (
                pred_doc.model_copy(deep=True),
                list(record.predicted_pictures),
                list(record.predicted_page_images),
            )

        prepared_doc, pred_pictures, pred_page_images = extract_images(
            document=pred_doc.model_copy(deep=True),
            pictures_column=BenchMarkColumns.PREDICTION_PICTURES.value,
            page_images_column=BenchMarkColumns.PREDICTION_PAGE_IMAGES.value,
        )
        return prepared_doc, pred_pictures, pred_page_images

    def _load_split(self, dataset_dir: Path, split: str) -> Dataset:
        split_dir = dataset_dir / split
        split_files = sorted(split_dir.glob("*.parquet"))
        if not split_files:
            raise FileNotFoundError(f"No parquet files found under {split_dir}")
        dataset = load_dataset(
            "parquet", data_files={split: [str(path) for path in split_files]}
        )
        return dataset[split]

    def _slice_dataset(
        self, dataset: Dataset, begin_index: int, end_index: int
    ) -> Dataset:
        total = len(dataset)
        begin = max(begin_index, 0)
        end = total if end_index < 0 else min(end_index, total)

        if begin >= end:
            return dataset.select([])
        if begin == 0 and end == total:
            return dataset
        return dataset.select(range(begin, end))

    def _save_visualization(self, record: DatasetRecordWithPrediction) -> None:
        if record.predicted_doc is None:
            return

        gt_doc = insert_images_from_pil(
            record.ground_truth_doc.model_copy(deep=True),
            record.ground_truth_pictures,
            record.ground_truth_page_images,
        )
        pred_doc = insert_images_from_pil(
            record.predicted_doc.model_copy(deep=True),
            record.predicted_pictures,
            record.predicted_page_images,
        )

        try:
            save_comparison_html_with_clusters(
                filename=self._visualizations_dir / f"{record.doc_id}.html",
                true_doc=gt_doc,
                pred_doc=pred_doc,
                draw_reading_order=True,
            )
        except (IndexError, ValueError) as e:
            _LOGGER.warning(
                f"Failed to save visualization for doc_id {record.doc_id}: {e}"
            )
