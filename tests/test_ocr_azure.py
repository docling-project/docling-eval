import logging
import os
from pathlib import Path

import pytest

from docling_eval.cli.main import evaluate, visualize
from docling_eval.datamodels.types import BenchMarkNames, EvaluationModality
from docling_eval.dataset_builders.pixparse_builder import PixparseDatasetBuilder
from docling_eval.prediction_providers.azure_prediction_provider import (
    AzureDocIntelligencePredictionProvider,
)

IS_CI = bool(os.getenv("CI"))

logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_run_pixparse_builder():
    target_path = Path(f"./scratch/{BenchMarkNames.PIXPARSEIDL.value}_azure/")
    azure_provider = AzureDocIntelligencePredictionProvider(
        do_visualization=True, ignore_missing_predictions=True
    )

    dataset = PixparseDatasetBuilder(
        target=target_path / "gt_dataset",
        end_index=5,
    )

    dataset.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset.save_to_disk()  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    azure_provider.create_prediction_dataset(
        name=dataset.name,
        gt_dataset_dir=target_path / "gt_dataset",
        target_dataset_dir=target_path / "eval_dataset",
    )

    evaluate(
        modality=EvaluationModality.OCR,
        benchmark=BenchMarkNames.PIXPARSEIDL,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.OCR.value,
    )

    visualize(
        modality=EvaluationModality.OCR,
        benchmark=BenchMarkNames.PIXPARSEIDL,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.OCR.value,
    )
