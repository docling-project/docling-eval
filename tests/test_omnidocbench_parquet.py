"""
Test OmniDocBench Parquet mode functionality.

This test verifies that the OmniDocBenchDatasetBuilder can load data
from a Parquet-format dataset via load_dataset, avoiding rate limits
from downloading many individual files.
"""

import os
from pathlib import Path

import pytest

from docling_eval.cli.main import evaluate, get_prediction_provider, visualize
from docling_eval.datamodels.types import (
    BenchMarkNames,
    EvaluationModality,
    PredictionProviderType,
)
from docling_eval.dataset_builders.omnidocbench_builder import (
    OmniDocBenchDatasetBuilder,
)

IS_CI = bool(os.getenv("CI"))


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_run_omnidocbench_parquet_e2e():
    """
    Test OmniDocBench with Parquet mode (use_parquet=True).

    This uses the samiuc/OmniDocBench-parquet dataset which contains
    filename, image, pdf, and ground_truth columns in Parquet format,
    avoiding HuggingFace rate limits from downloading individual files.
    """
    target_path = Path(f"./scratch/{BenchMarkNames.OMNIDOCBENCH.value}-parquet/")
    docling_provider = get_prediction_provider(PredictionProviderType.DOCLING)

    dataset_layout = OmniDocBenchDatasetBuilder(
        target=target_path / "gt_dataset",
        repo_id="samiuc/OmniDocBench-parquet",
        use_parquet=True,
        end_index=5,
    )

    dataset_layout.retrieve_input_dataset()  # No-op in Parquet mode
    dataset_layout.save_to_disk()  # Iterates dataset and saves as parquet shards

    docling_provider.create_prediction_dataset(
        name=dataset_layout.name,
        gt_dataset_dir=target_path / "gt_dataset",
        target_dataset_dir=target_path / "eval_dataset",
    )

    # Evaluate Layout
    evaluate(
        modality=EvaluationModality.LAYOUT,
        benchmark=BenchMarkNames.OMNIDOCBENCH,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.LAYOUT.value,
    )

    visualize(
        modality=EvaluationModality.LAYOUT,
        benchmark=BenchMarkNames.OMNIDOCBENCH,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.LAYOUT.value,
    )

    # Evaluate Reading Order
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

    # Evaluate Markdown Text
    evaluate(
        modality=EvaluationModality.MARKDOWN_TEXT,
        benchmark=BenchMarkNames.OMNIDOCBENCH,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.MARKDOWN_TEXT.value,
    )

    visualize(
        modality=EvaluationModality.MARKDOWN_TEXT,
        benchmark=BenchMarkNames.OMNIDOCBENCH,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.MARKDOWN_TEXT.value,
    )
