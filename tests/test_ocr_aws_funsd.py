import logging
import os
from pathlib import Path

import pytest

from docling_eval.cli.main import evaluate, visualize
from docling_eval.datamodels.types import BenchMarkNames, EvaluationModality
from docling_eval.dataset_builders.funsd_builder import FUNSDDatasetBuilder
from docling_eval.prediction_providers.aws_prediction_provider import (
    AWSTextractPredictionProvider,
)
from tests.test_utils import validate_evaluation_results

IS_CI = os.getenv("RUN_IN_CI") == "1"

logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_run_funsd_builder():
    target_path = Path(f"./scratch/{BenchMarkNames.FUNSD.value}_aws/")
    dataset_source = Path(target_path, "input_dataset")
    aws_provider = AWSTextractPredictionProvider(
        do_visualization=True, ignore_missing_predictions=False
    )

    dataset = FUNSDDatasetBuilder(
        dataset_source=dataset_source,
        target=target_path / "gt_dataset",
        end_index=1,
    )
    dataset.retrieve_input_dataset()
    dataset.save_to_disk()

    aws_provider.create_prediction_dataset(
        # name="Funsd",
        name=dataset.name,
        gt_dataset_dir=target_path / "gt_dataset",
        target_dataset_dir=target_path / "eval_dataset",
    )

    evaluate(
        modality=EvaluationModality.OCR,
        benchmark=BenchMarkNames.FUNSD,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.OCR.value,
    )

    validate_evaluation_results(
        target_path=target_path,
        benchmark=BenchMarkNames.FUNSD.value,
        modality=EvaluationModality.OCR.value,
    )
    visualize(
        modality=EvaluationModality.OCR,
        benchmark=BenchMarkNames.FUNSD,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.OCR.value,
    )
