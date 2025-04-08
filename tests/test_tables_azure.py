import logging
import os
from pathlib import Path

import pytest

from docling_eval.cli.main import evaluate, visualize
from docling_eval.datamodels.types import BenchMarkNames, EvaluationModality
from docling_eval.dataset_builders.otsl_table_dataset_builder import (
    FintabNetDatasetBuilder,
)
from docling_eval.prediction_providers.azure_prediction_provider import (
    AzureDocIntelligencePredictionProvider,
)

IS_CI = os.getenv("RUN_IN_CI") == "1"

logging.getLogger("azure").setLevel(logging.WARNING)


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_run_fintabnet_builder():
    target_path = Path(f"./scratch/{BenchMarkNames.FINTABNET.value}/")
    azure_provider = AzureDocIntelligencePredictionProvider(
        do_visualization=True, ignore_missing_predictions=True
    )

    dataset = FintabNetDatasetBuilder(
        target=target_path / "gt_dataset",
        end_index=5,
    )

    # dataset.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset.save_to_disk()  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    azure_provider.create_prediction_dataset(
        name=dataset.name,
        gt_dataset_dir=target_path / "gt_dataset",
        target_dataset_dir=target_path / "eval_dataset",
    )

    evaluate(
        modality=EvaluationModality.TABLE_STRUCTURE,
        benchmark=BenchMarkNames.FINTABNET,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.TABLE_STRUCTURE.value,
    )

    visualize(
        modality=EvaluationModality.TABLE_STRUCTURE,
        benchmark=BenchMarkNames.FINTABNET,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.TABLE_STRUCTURE.value,
    )
