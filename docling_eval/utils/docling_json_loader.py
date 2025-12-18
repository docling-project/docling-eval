from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import Iterator, List

from docling_core.types import DoclingDocument
from docling_core.types.io import DocumentStream
from pydantic import ValidationError

from docling_eval.datamodels.dataset_record import DatasetRecord
from docling_eval.datamodels.types import BenchMarkColumns
from docling_eval.utils.utils import extract_images, get_binhash

_LOGGER = logging.getLogger(__name__)


def _select_range(files: List[Path], begin_index: int, end_index: int) -> List[Path]:
    if begin_index < 0:
        begin_index = 0

    total = len(files)
    effective_end = total if end_index < 0 or end_index > total else end_index

    if begin_index >= effective_end:
        return []

    return files[begin_index:effective_end]


def iter_docling_json_records(
    json_dir: Path,
    *,
    begin_index: int = 0,
    end_index: int = -1,
) -> Iterator[DatasetRecord]:
    json_files: List[Path] = sorted(json_dir.glob("*.json"))
    selected_files = _select_range(json_files, begin_index, end_index)

    for json_path in selected_files:
        try:
            document: DoclingDocument = DoclingDocument.load_from_json(json_path)
        except ValidationError as exc:
            _LOGGER.error(
                "Validation error loading document %s: %s. Skipping this document.",
                json_path,
                exc,
            )
            continue
        except Exception as exc:  # noqa: BLE001
            _LOGGER.error(
                "Unexpected error loading document %s: %s. Skipping this document.",
                json_path,
                exc,
            )
            continue

        try:
            document, pictures, page_images = extract_images(
                document=document,
                pictures_column=BenchMarkColumns.GROUNDTRUTH_PICTURES.value,
                page_images_column=BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES.value,
            )

            doc_bytes = json_path.read_bytes()

            yield DatasetRecord(
                doc_id=json_path.stem,
                doc_path=json_path,
                doc_hash=get_binhash(doc_bytes),
                ground_truth_doc=document,
                ground_truth_pictures=pictures,
                ground_truth_page_images=page_images,
                original=DocumentStream(
                    name=json_path.name,
                    stream=BytesIO(doc_bytes),
                ),
                mime_type="application/json",
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.error(
                "Error processing document %s: %s. Skipping this document.",
                json_path,
                exc,
            )
            continue
