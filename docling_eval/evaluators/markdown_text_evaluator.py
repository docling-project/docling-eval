import logging
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import DoclingDocument
from nltk.tokenize import word_tokenize
from pydantic import BaseModel
from tqdm import tqdm  # type: ignore

from docling_eval.benchmarks.constants import BenchMarkColumns  # type: ignore
from docling_eval.utils.bleu import compute_bleu_score
from docling_eval.utils.stats import DatasetStatistics, compute_stats

_log = logging.getLogger(__name__)


class PageMarkdownEvaluation(BaseModel):
    doc_id: str

    true_md: str
    pred_md: str
    bleu: float


class DatasetMarkdownEvaluation(BaseModel):
    evaluations: List[PageMarkdownEvaluation]
    bleu_stats: DatasetStatistics


class MarkdownTextEvaluator:
    def __init__(self):
        pass

    def __call__(self, ds_path: Path, split: str = "test") -> DatasetMarkdownEvaluation:
        parquet_files = str(ds_path / split / "*.parquet")
        ds = load_dataset("parquet", data_files={split: parquet_files})
        _log.info(f"oveview of dataset: {ds}")
        if ds is not None:
            ds_selection = ds[split]

        evaluations: list[PageMarkdownEvaluation] = []
        bleus = []

        broken_inputs = 0
        for i, data in tqdm(
            enumerate(ds_selection),
            desc="Markdown text evaluations",
            ncols=120,
            total=len(ds_selection),
        ):
            doc_id = data[BenchMarkColumns.DOC_ID]
            true_doc_dict = data[BenchMarkColumns.GROUNDTRUTH]
            true_doc: DoclingDocument = DoclingDocument.model_validate_json(
                true_doc_dict
            )
            pred_doc_dict = data[BenchMarkColumns.PREDICTION]
            pred_doc: DoclingDocument = DoclingDocument.model_validate_json(
                pred_doc_dict
            )

            # Export to markdown and tokenize
            true_md = true_doc.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER)
            true_tokens = word_tokenize(true_md)
            pred_md = pred_doc.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER)
            pred_tokens = word_tokenize(pred_md)

            bleu = compute_bleu_score(true_tokens, pred_tokens)
            bleus.append(bleu)
            md_evaluation = PageMarkdownEvaluation(
                doc_id=doc_id, true_md=true_md, pred_md=pred_md, bleu=bleu
            )
            evaluations.append(md_evaluation)
        bleu_stats = compute_stats(bleus)
        ds_md_evalutions = DatasetMarkdownEvaluation(
            evaluations=evaluations, bleu_stats=bleu_stats
        )
        return ds_md_evalutions
