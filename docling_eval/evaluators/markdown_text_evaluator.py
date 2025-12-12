import logging
from concurrent.futures import Executor, Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import evaluate
import nltk
from datasets import load_dataset
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import ContentLayer, DoclingDocument
from docling_core.types.doc.labels import DocItemLabel
from nltk import edit_distance, word_tokenize
from nltk.metrics import f_measure, precision, recall
from nltk.translate import meteor_score
from pydantic import BaseModel
from tqdm import tqdm  # type: ignore

from docling_eval.datamodels.dataset_record import DatasetRecordWithPrediction
from docling_eval.datamodels.types import (  # type: ignore
    BenchMarkColumns,
    PredictionFormats,
)
from docling_eval.evaluators.base_evaluator import (
    BaseEvaluator,
    DatasetEvaluation,
    EvaluationRejectionType,
    UnitEvaluation,
)
from docling_eval.evaluators.stats import DatasetStatistics, compute_stats
from docling_eval.utils.external_docling_document_loader import (
    ExternalDoclingDocumentLoader,
)

_log = logging.getLogger(__name__)


def compute_bleu_score(bleu_eval, true_txt: str, pred_txt: str) -> float:
    r"""
    Compute BLEU score with the HF evaluate and the default Tokenizer_13
    """
    result = bleu_eval.compute(predictions=[pred_txt], references=[[true_txt]])
    bleu = result["bleu"]
    return bleu


def compute_nltk_scores(true_txt: str, pred_txt: str) -> dict[str, float]:
    r"""
    Returns:
    --------
    dict with keys: ["f_measure", "precision", "recall", "edit_dist"]
    """
    true_tokens = word_tokenize(true_txt)
    true_tokens_set = set(true_tokens)
    pred_tokens = word_tokenize(pred_txt)
    pred_tokens_set = set(pred_tokens)

    f1_score = f_measure(true_tokens_set, pred_tokens_set)
    precision_score = precision(true_tokens_set, pred_tokens_set)
    recall_score = recall(true_tokens_set, pred_tokens_set)
    edit_dist = edit_distance(pred_tokens, true_tokens) / max(
        len(pred_tokens), len(true_tokens)
    )
    meteor = meteor_score.meteor_score([true_tokens], pred_tokens)

    metrics: dict[str, float] = {
        "f1_score": f1_score,
        "precision": precision_score,
        "recall": recall_score,
        "edit_distance": edit_dist,
        "meteor": meteor,
    }
    return metrics


def evaluate_page(bleu_eval, true_md: str, pred_md: str) -> dict[str, float]:
    r"""Compute the bleu and the nltk scores"""
    scores = compute_nltk_scores(true_md, pred_md)
    scores["bleu"] = compute_bleu_score(bleu_eval, true_md, pred_md)
    return scores


class PageMarkdownEvaluation(UnitEvaluation):
    doc_id: str

    true_md: str
    pred_md: str
    bleu: float
    f1_score: float
    precision: float
    recall: float
    edit_distance: float
    meteor: float


class DatasetMarkdownEvaluation(DatasetEvaluation):
    evaluations: List[PageMarkdownEvaluation]

    bleu_stats: DatasetStatistics
    f1_score_stats: DatasetStatistics
    precision_stats: DatasetStatistics
    recall_stats: DatasetStatistics
    edit_distance_stats: DatasetStatistics
    meteor_stats: DatasetStatistics


class MarkdownTextEvaluator(BaseEvaluator):
    def __init__(
        self,
        intermediate_evaluations_path: Optional[Path] = None,
        prediction_sources: List[PredictionFormats] = [],
        concurrency: int = 4,
    ):
        r""" """
        supported_prediction_formats: List[PredictionFormats] = [
            PredictionFormats.DOCLING_DOCUMENT,
            PredictionFormats.MARKDOWN,
        ]
        if not prediction_sources:
            prediction_sources = supported_prediction_formats
        super().__init__(
            intermediate_evaluations_path=intermediate_evaluations_path,
            prediction_sources=prediction_sources,
            supported_prediction_formats=supported_prediction_formats,
            concurrency=concurrency,
        )

        self._bleu_eval = evaluate.load("bleu")

        # Download the NLTK data
        nltk.download("popular", quiet=True)
        nltk.download("punkt_tab", quiet=True)

        # Select which DocItemLabels should be exported to markdown
        self._labels: Set[DocItemLabel] = set(
            [
                DocItemLabel.CAPTION,
                DocItemLabel.FOOTNOTE,
                DocItemLabel.FORMULA,
                DocItemLabel.LIST_ITEM,
                DocItemLabel.PAGE_FOOTER,
                DocItemLabel.PAGE_HEADER,
                DocItemLabel.PICTURE,
                DocItemLabel.SECTION_HEADER,
                # DocItemLabel.TABLE,
                DocItemLabel.TEXT,
                DocItemLabel.TITLE,
                DocItemLabel.DOCUMENT_INDEX,
                DocItemLabel.CODE,
                DocItemLabel.CHECKBOX_SELECTED,
                DocItemLabel.CHECKBOX_UNSELECTED,
                DocItemLabel.FORM,
                DocItemLabel.KEY_VALUE_REGION,
                DocItemLabel.PARAGRAPH,
                DocItemLabel.REFERENCE,
            ]
        )

    def __call__(
        self,
        ds_path: Path,
        split: str = "test",
        external_predictions_path: Optional[Path] = None,
    ) -> DatasetMarkdownEvaluation:
        r"""
        Parameters
        ----------
        ds_path: Path to load the parquet files of the dataset
        split: Split of the dataset to load
        """
        if external_predictions_path is not None:
            external_docling_doc_loader = ExternalDoclingDocumentLoader(
                external_predictions_path
            )

        parquet_files = str(ds_path / split / "*.parquet")
        ds = load_dataset("parquet", data_files={split: parquet_files})
        _log.info(f"Overview of the dataset: {ds}")
        if ds is not None:
            ds_selection = ds[split]

        evaluations: list[PageMarkdownEvaluation] = []
        rejected_samples: Dict[EvaluationRejectionType, int] = {
            EvaluationRejectionType.INVALID_CONVERSION_STATUS: 0,
            EvaluationRejectionType.MISSING_PREDICTION: 0,
        }

        # Metrics per page
        ds_metrics: dict[str, list[float]] = {
            "bleu": [],
            "f1_score": [],
            "precision": [],
            "recall": [],
            "edit_distance": [],
            "meteor": [],
        }

        with ProcessPoolExecutor(max_workers=self._concurrency) as executor:
            futures: list[Future] = []

            # Submit the evaluation tasks
            _log.info("Submitting the documents for evaluation...")
            for data in ds_selection:
                data_record = DatasetRecordWithPrediction.model_validate(data)
                doc_id = data_record.doc_id
                true_doc = data_record.ground_truth_doc
                true_md = self._docling_document_to_md(true_doc)

                # Get the predicted markdown from the external predictions path
                if external_predictions_path is not None:
                    pred_doc = external_docling_doc_loader(data_record)
                    if pred_doc is None:
                        _log.error("No external prediction found for doc_id=%s", doc_id)
                        rejected_samples[
                            EvaluationRejectionType.MISSING_PREDICTION
                        ] += 1
                        continue
                    pred_md = self._docling_document_to_md(pred_doc)
                else:
                    if data_record.status not in self._accepted_status:
                        _log.error(
                            "Skipping record without successfull conversion status: %s",
                            doc_id,
                        )
                        rejected_samples[
                            EvaluationRejectionType.INVALID_CONVERSION_STATUS
                        ] += 1
                        continue
                    pred_md = self._get_pred_md(data_record)  # type: ignore

                if pred_md is None:
                    _log.error("There is no markdown prediction for doc_id=%s", doc_id)
                    rejected_samples[EvaluationRejectionType.MISSING_PREDICTION] += 1
                    continue

                if true_md != "" and pred_md != "":
                    futures.append(
                        executor.submit(
                            evaluate_page, self._bleu_eval, true_md, pred_md
                        )
                    )

            # Collect the futures
            _log.info("Collecting the documents for evaluations...")
            for i, future in tqdm(
                enumerate(as_completed(futures)),
                desc="Markdown text evaluations",
                ncols=120,
                total=len(futures),
            ):
                doc_metrics = future.result()

                # Collect metrics across pages
                for score_name, score in doc_metrics.items():
                    ds_metrics[score_name].append(score)

                md_evaluation = PageMarkdownEvaluation(
                    doc_id=doc_id,
                    true_md=true_md,
                    pred_md=pred_md,
                    bleu=doc_metrics["bleu"],
                    f1_score=doc_metrics["f1_score"],
                    precision=doc_metrics["precision"],
                    recall=doc_metrics["recall"],
                    edit_distance=doc_metrics["edit_distance"],
                    meteor=doc_metrics["meteor"],
                )
                evaluations.append(md_evaluation)

                if self._intermediate_evaluations_path:
                    self.save_intermediate_evaluations("MD", i, doc_id, evaluations)

        ds_md_evalutions = DatasetMarkdownEvaluation(
            evaluated_samples=len(evaluations),
            rejected_samples=rejected_samples,
            evaluations=evaluations,
            bleu_stats=compute_stats(ds_metrics["bleu"]),
            f1_score_stats=compute_stats(ds_metrics["f1_score"]),
            precision_stats=compute_stats(ds_metrics["precision"]),
            recall_stats=compute_stats(ds_metrics["recall"]),
            edit_distance_stats=compute_stats(ds_metrics["edit_distance"]),
            meteor_stats=compute_stats(ds_metrics["meteor"]),
        )
        return ds_md_evalutions

    def _docling_document_to_md(self, doc: DoclingDocument) -> str:
        r"""
        Export DoclingDocument to markdown
        """
        md = doc.export_to_markdown(
            image_mode=ImageRefMode.PLACEHOLDER,
            image_placeholder="",
            labels=self._labels,
            included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE},
        )
        return md

    def _get_pred_md(self, data_record: DatasetRecordWithPrediction) -> Optional[str]:
        r"""
        Get the predicted markdown
        """
        pred_md = None
        for prediction_format in self._prediction_sources:
            if prediction_format == PredictionFormats.DOCLING_DOCUMENT:
                pred_doc = data_record.predicted_doc
                if pred_doc is not None:
                    pred_md = self._docling_document_to_md(pred_doc)
            elif prediction_format == PredictionFormats.MARKDOWN:
                pred_md = data_record.original_prediction
            if pred_md is not None:
                break

        return pred_md
