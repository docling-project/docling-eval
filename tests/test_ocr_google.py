import logging
import os
from pathlib import Path

import pytest

from docling_eval.cli.main import evaluate, visualize
from docling_eval.datamodels.types import BenchMarkNames, EvaluationModality
from docling_eval.dataset_builders.pixparse_builder import PixparseDatasetBuilder
from docling_eval.dataset_builders.xfund_builder import XFUNDDatasetBuilder
from docling_eval.prediction_providers.google_prediction_provider import (
    GoogleDocAIPredictionProvider,
)
from tests.test_utils import validate_evaluation_results

IS_CI = bool(os.getenv("CI"))

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_run_pixparse_builder():
    target_path = Path(f"./scratch/{BenchMarkNames.PIXPARSEIDL.value}_google/")
    google_provider = GoogleDocAIPredictionProvider(
        do_visualization=True, ignore_missing_predictions=False
    )

    dataset = PixparseDatasetBuilder(
        target=target_path / "gt_dataset",
        end_index=5,
    )

    dataset.retrieve_input_dataset()
    dataset.save_to_disk()  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    google_provider.create_prediction_dataset(
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


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_run_xfund_builder():
    target_path = Path(f"./scratch/{BenchMarkNames.XFUND.value}_google/")
    google_provider = GoogleDocAIPredictionProvider(
        do_visualization=True, ignore_missing_predictions=False
    )

    dataset = XFUNDDatasetBuilder(
        dataset_source=target_path / "input_dataset",
        target=target_path / "gt_dataset",
        end_index=2,
    )

    dataset.retrieve_input_dataset()
    dataset.save_to_disk()  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    google_provider.create_prediction_dataset(
        name=dataset.name,
        gt_dataset_dir=target_path / "gt_dataset",
        split="val",  # NOTE: Xfund has val split instead of test
        target_dataset_dir=target_path / "eval_dataset",
    )

    evaluate(
        modality=EvaluationModality.OCR,
        benchmark=BenchMarkNames.XFUND,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.OCR.value,
        split="val",
    )
    validate_evaluation_results(
        target_path=target_path,
        benchmark=BenchMarkNames.XFUND.value,
        modality=EvaluationModality.OCR.value,
    )
    visualize(
        modality=EvaluationModality.OCR,
        benchmark=BenchMarkNames.XFUND,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.OCR.value,
        split="val",
    )
