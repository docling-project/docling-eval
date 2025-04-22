import glob
import json
import logging
import os
import statistics
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar

import evaluate
import pandas as pd
from datasets import Dataset, load_dataset
from docling_core.types.doc.document import DoclingDocument
from pydantic import BaseModel
from tqdm import tqdm

from docling_eval.datamodels.dataset_record import DatasetRecordWithPrediction
from docling_eval.datamodels.types import BenchMarkColumns, PredictionFormats
from docling_eval.evaluators.base_evaluator import BaseEvaluator

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

_log = logging.getLogger(__name__)


class PageOcrEvaluation(BaseModel):
    doc_id: str
    true_text: str
    pred_text: str
    cer: float
    char_accuracy: float


class DatasetOcrEvaluation(BaseModel):
    evaluations: List[PageOcrEvaluation]
    mean_character_accuracy: float


class OCREvaluator(BaseEvaluator):
    """Evaluator for OCR tasks that computes Character Accuracy"""

    def __init__(
        self,
        intermediate_evaluations_path: Optional[Path] = None,
        prediction_sources: List[PredictionFormats] = [
            PredictionFormats.DOCLING_DOCUMENT
        ],
    ):
        """Initialize the OCR evaluator"""
        super().__init__(
            intermediate_evaluations_path=intermediate_evaluations_path,
            prediction_sources=prediction_sources,
            supported_prediction_formats=[PredictionFormats.DOCLING_DOCUMENT],
        )
        # Load the CER evaluation metric
        # https://huggingface.co/spaces/evaluate-metric/cer
        self._cer_eval = evaluate.load("cer")

    def __call__(
        self,
        ds_path: Path,
        split: str = "test",
    ) -> DatasetOcrEvaluation:

        _log.info("Loading the split '%s' from: '%s'", split, ds_path)
        split_path = str(ds_path / split / "*.parquet")
        split_files = glob.glob(split_path)
        _log.info("Files: %s", split_files)
        ds = load_dataset("parquet", data_files={split: split_files})
        _log.info("Overview of dataset: %s", ds)

        # Select the split
        ds_selection: Dataset = ds[split]

        text_evaluations_list = []
        char_accuracy_list = []

        for i, data in tqdm(
            enumerate(ds_selection),
            desc="Evaluating OCR",
            ncols=120,
            total=len(ds_selection),
        ):
            data_record = DatasetRecordWithPrediction.model_validate(data)
            doc_id = data_record.doc_id
            if data_record.status not in self._accepted_status:
                _log.error(
                    "Skipping record without successfull conversion status: %s", doc_id
                )
                continue

            true_doc = data_record.ground_truth_doc
            pred_doc = data_record.predicted_doc

            if not pred_doc:
                _log.error("There is no prediction for doc_id=%s", doc_id)
                continue

            true_text = self._extract_text(true_doc)
            pred_text = self._extract_text(pred_doc)

            if true_text and pred_text:
                cer = self._compute_cer_score(true_text, pred_text)
                char_accuracy = 1.0 - cer
            else:
                cer = 1.0  # max error when text is missing
                char_accuracy = 0.0  # zero accuracy when text is missing

            char_accuracy_list.append(char_accuracy)

            page_evaluation = PageOcrEvaluation(
                doc_id=doc_id,
                true_text=true_text,
                pred_text=pred_text,
                cer=cer,
                char_accuracy=char_accuracy,
            )

            text_evaluations_list.append(page_evaluation)
            if self._intermediate_evaluations_path:
                self.save_intermediate_evaluations(
                    evaluation_name="ocr_eval",
                    enunumerate_id=i,
                    doc_id=doc_id,
                    evaluations=[page_evaluation],
                )

        mean_character_accuracy = (
            statistics.mean(char_accuracy_list) if char_accuracy_list else 0.0
        )

        _log.info(f"Mean Character Accuracy: {mean_character_accuracy:.4f}")

        return DatasetOcrEvaluation(
            evaluations=text_evaluations_list,
            mean_character_accuracy=mean_character_accuracy,
        )

    def _compute_cer_score(self, true_txt: str, pred_txt: str) -> float:
        """Compute Character Error Rate"""
        result = self._cer_eval.compute(predictions=[pred_txt], references=[true_txt])
        return result

    def _extract_text(self, doc: DoclingDocument) -> str:
        """Extract text from document JSON structure"""
        extracted_text = ""
        for text_item in doc.texts:
            extracted_text += text_item.text + " "
        return extracted_text.strip()
