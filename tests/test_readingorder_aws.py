import logging
import os
from pathlib import Path

import pytest

from docling_eval.cli.main import evaluate, visualize
from docling_eval.datamodels.types import BenchMarkNames, EvaluationModality
from docling_eval.dataset_builders.dpbench_builder import DPBenchDatasetBuilder
from docling_eval.dataset_builders.omnidocbench_builder import (
    OmniDocBenchDatasetBuilder,
)
from docling_eval.prediction_providers.aws_prediction_provider import (
    AWSTextractPredictionProvider,
)

IS_CI = bool(os.getenv("CI"))

logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_run_dpbench_builder():
    target_path = Path(f"./scratch/{BenchMarkNames.DPBENCH.value}_aws/")
    aws_provider = AWSTextractPredictionProvider(
        do_visualization=True, ignore_missing_predictions=False
    )

    dataset = DPBenchDatasetBuilder(
        target=target_path / "gt_dataset",
        end_index=15,
    )

    dataset.retrieve_input_dataset()
    dataset.save_to_disk()  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    aws_provider.create_prediction_dataset(
        name=dataset.name,
        gt_dataset_dir=target_path / "gt_dataset",
        target_dataset_dir=target_path / "eval_dataset",
    )

    evaluate(
        modality=EvaluationModality.READING_ORDER,
        benchmark=BenchMarkNames.DPBENCH,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.READING_ORDER.value,
    )

    visualize(
        modality=EvaluationModality.READING_ORDER,
        benchmark=BenchMarkNames.DPBENCH,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.READING_ORDER.value,
    )


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_run_omnidocbench_builder():
    target_path = Path(f"./scratch/{BenchMarkNames.OMNIDOCBENCH.value}_aws/")
    aws_provider = AWSTextractPredictionProvider(
        do_visualization=True, ignore_missing_predictions=False
    )

    dataset = OmniDocBenchDatasetBuilder(
        target=target_path / "gt_dataset",
        end_index=15,
    )

    dataset.retrieve_input_dataset()
    dataset.save_to_disk()  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    aws_provider.create_prediction_dataset(
        name=dataset.name,
        gt_dataset_dir=target_path / "gt_dataset",
        target_dataset_dir=target_path / "eval_dataset",
    )

    evaluate(
        modality=EvaluationModality.READING_ORDER,
        benchmark=BenchMarkNames.OMNIDOCBENCH,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.READING_ORDER.value,
    )

    visualize(
        modality=EvaluationModality.READING_ORDER,
        benchmark=BenchMarkNames.OMNIDOCBENCH,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.READING_ORDER.value,
    )
