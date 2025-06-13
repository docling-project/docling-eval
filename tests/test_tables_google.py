import logging
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from docling_eval.cli.main import evaluate, visualize
from docling_eval.datamodels.types import BenchMarkNames, EvaluationModality
from docling_eval.dataset_builders.dpbench_builder import DPBenchDatasetBuilder
from docling_eval.dataset_builders.otsl_table_dataset_builder import (
    FintabNetDatasetBuilder,
)
from docling_eval.prediction_providers.google_prediction_provider import (
    GoogleDocAIPredictionProvider,
)

IS_CI = bool(os.getenv("CI"))

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)

load_dotenv()


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_run_fintabnet_builder():
    target_path = Path(f"./scratch/{BenchMarkNames.FINTABNET.value}_google/")
    google_provider = GoogleDocAIPredictionProvider(
        do_visualization=True, ignore_missing_predictions=False
    )

    dataset = FintabNetDatasetBuilder(
        target=target_path / "gt_dataset",
        end_index=15,
    )

    dataset.save_to_disk()  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    google_provider.create_prediction_dataset(
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


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_run_dpbench_builder():
    target_path = Path(f"./scratch/{BenchMarkNames.DPBENCH.value}_google/")
    google_provider = GoogleDocAIPredictionProvider(
        do_visualization=True, ignore_missing_predictions=False
    )

    dataset = DPBenchDatasetBuilder(
        target=target_path / "gt_dataset",
        begin_index=185,
        end_index=201,
    )

    dataset.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset.save_to_disk()  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    google_provider.create_prediction_dataset(
        name=dataset.name,
        gt_dataset_dir=target_path / "gt_dataset",
        target_dataset_dir=target_path / "eval_dataset",
    )

    evaluate(
        modality=EvaluationModality.TABLE_STRUCTURE,
        benchmark=BenchMarkNames.DPBENCH,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.TABLE_STRUCTURE.value,
    )

    visualize(
        modality=EvaluationModality.TABLE_STRUCTURE,
        benchmark=BenchMarkNames.DPBENCH,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.TABLE_STRUCTURE.value,
    )
