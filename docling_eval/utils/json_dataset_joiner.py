from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterator, Optional

from docling.datamodel.base_models import ConversionStatus
from docling.utils.utils import chunkify
from docling_core.types import DoclingDocument
from docling_core.types.io import DocumentStream

from docling_eval.datamodels.dataset_record import (
    DatasetRecord,
    DatasetRecordWithPrediction,
)
from docling_eval.datamodels.types import BenchMarkColumns, PredictionFormats
from docling_eval.utils.docling_json_loader import iter_docling_json_records
from docling_eval.utils.utils import (
    extract_images,
    insert_images_from_pil,
    save_shard_to_disk,
    write_datasets_info,
)
from docling_eval.visualisation.visualisations import save_comparison_html_with_clusters

_LOGGER = logging.getLogger(__name__)


def _load_prediction_json(
    prediction_record: DatasetRecord,
) -> tuple[DoclingDocument, list, list, Optional[str]]:
    document = prediction_record.ground_truth_doc.model_copy(deep=True)
    prediction_doc, pictures, page_images = extract_images(
        document=document,
        pictures_column=BenchMarkColumns.PREDICTION_PICTURES.value,
        page_images_column=BenchMarkColumns.PREDICTION_PAGE_IMAGES.value,
    )

    prediction_text: Optional[str] = None
    original = prediction_record.original
    if original is not None:
        try:
            raw_bytes: bytes
            if isinstance(original, DocumentStream):
                stream = original.stream
                stream.seek(0)
                raw_bytes = stream.read()
                stream.seek(0)
            elif isinstance(original, Path):
                raw_bytes = original.read_bytes()
            else:  # pragma: no cover
                raise TypeError(f"Unsupported prediction source: {type(original)!r}")

            prediction_text = raw_bytes.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            _LOGGER.debug(
                "Failed to decode prediction JSON for %s: %s",
                prediction_record.doc_id,
                exc,
            )

    return prediction_doc, pictures, page_images, prediction_text


def _build_prediction_record(
    gt_record: DatasetRecord,
    prediction_doc: DoclingDocument,
    pred_pictures: list,
    pred_page_images: list,
    *,
    prediction_format: PredictionFormats,
    predictor_info: Dict,
    original_prediction: Optional[str],
) -> DatasetRecordWithPrediction:
    record_data = gt_record.model_dump()
    record_data["original"] = gt_record.original
    record_data["ground_truth_doc"] = gt_record.ground_truth_doc
    record_data["ground_truth_pictures"] = gt_record.ground_truth_pictures
    record_data["ground_truth_page_images"] = gt_record.ground_truth_page_images
    record_data["doc_path"] = gt_record.doc_path
    record_data.update(
        {
            "predicted_doc": prediction_doc,
            "predicted_pictures": pred_pictures,
            "predicted_page_images": pred_page_images,
            "prediction_format": prediction_format,
            "predictor_info": predictor_info,
            "prediction_timings": None,
            "original_prediction": original_prediction,
            "status": ConversionStatus.SUCCESS,
        }
    )
    return DatasetRecordWithPrediction.model_validate(record_data)


def _visualize_record(
    record: DatasetRecordWithPrediction, visualizations_dir: Path
) -> None:
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

    save_comparison_html_with_clusters(
        filename=visualizations_dir / f"{record.doc_id}.html",
        true_doc=gt_doc,
        pred_doc=pred_doc,
        draw_reading_order=True,
    )


def join_docling_json_datasets(
    gt_json_dir: Path,
    prediction_json_dir: Path,
    target_dataset_dir: Path,
    *,
    name: str = "DoclingJSONJoin",
    split: str = "test",
    chunk_size: int = 80,
    prediction_format: PredictionFormats = PredictionFormats.JSON,
    predictor_info: Optional[Dict] = None,
    ignore_missing_predictions: bool = True,
    do_visualization: bool = False,
) -> None:
    """
    Join two Docling JSON directories into a single evaluation parquet dataset.
    """
    predictor_info = predictor_info or {
        "asset": "docling_json_joiner",
        "prediction_format": prediction_format.value,
    }

    gt_records = list(iter_docling_json_records(gt_json_dir))
    prediction_records = {
        record.doc_id: record
        for record in iter_docling_json_records(prediction_json_dir)
    }

    test_dir = target_dataset_dir / split
    target_dataset_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    visualizations_dir: Optional[Path] = None
    if do_visualization:
        visualizations_dir = target_dataset_dir / "visualizations"
        visualizations_dir.mkdir(parents=True, exist_ok=True)

    def _generate_records() -> Iterator[DatasetRecordWithPrediction]:
        for gt_record in gt_records:
            prediction_record = prediction_records.get(gt_record.doc_id)
            if prediction_record is None:
                message = f"Missing prediction for document {gt_record.doc_id}"
                if ignore_missing_predictions:
                    _LOGGER.debug(message)
                    continue
                raise ValueError(message)

            prediction_doc, pictures, page_images, original_prediction = (
                _load_prediction_json(prediction_record)
            )

            joined = _build_prediction_record(
                gt_record,
                prediction_doc,
                pictures,
                page_images,
                prediction_format=prediction_format,
                predictor_info=predictor_info,
                original_prediction=original_prediction,
            )

            if do_visualization and visualizations_dir is not None:
                try:
                    _visualize_record(joined, visualizations_dir)
                except Exception as exc:  # noqa: BLE001
                    _LOGGER.warning(
                        "Failed to build visualization for %s: %s", joined.doc_id, exc
                    )

            yield joined

    count = 0
    chunk_count = 0
    for prediction_chunk in chunkify(_generate_records(), chunk_size):
        chunk_list = list(prediction_chunk)
        if not chunk_list:
            continue

        save_shard_to_disk(
            items=[record.as_record_dict() for record in chunk_list],
            dataset_path=test_dir,
            schema=DatasetRecordWithPrediction.pyarrow_schema(),
            shard_id=chunk_count,
        )

        count += len(chunk_list)
        chunk_count += 1

    write_datasets_info(
        name=name,
        output_dir=target_dataset_dir,
        num_train_rows=0,
        num_test_rows=count,
        features=DatasetRecordWithPrediction.features(),
    )

    _LOGGER.info(
        "Joined %s records into dataset %s (chunks: %s)",
        count,
        target_dataset_dir,
        chunk_count,
    )
