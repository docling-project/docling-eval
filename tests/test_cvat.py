import os
import shutil
from pathlib import Path

import pytest

from docling_eval.cli.main import PredictionProviderType, get_prediction_provider
from docling_eval.datamodels.types import BenchMarkNames, EvaluationModality
from docling_eval.dataset_builders.cvat_dataset_builder import CvatDatasetBuilder
from docling_eval.dataset_builders.cvat_preannotation_builder import (
    CvatPreannotationBuilder,
)
from docling_eval.dataset_builders.dpbench_builder import DPBenchDatasetBuilder

IS_CI = os.getenv("RUN_IN_CI") == "1"


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_run_cvat_on_gt():
    gt_path = Path(f"./scratch/{BenchMarkNames.DPBENCH.value}-GT/")
    cvat_path = Path(f"./scratch/{BenchMarkNames.DPBENCH.value}-CVAT/")

    # Stage 1: Create and pre-annotate dataset
    dataset_layout = DPBenchDatasetBuilder(
        target=gt_path,
        begin_index=15,
        end_index=20,
    )  # 10-25 is a small range which has samples with tables included.

    dataset_layout.retrieve_input_dataset()
    dataset_layout.save_to_disk()  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    builder = CvatPreannotationBuilder(
        dataset_source=gt_path / "test", target=cvat_path, bucket_size=20
    )
    builder.prepare_for_annotation()

    ## Stage 2: Re-build dataset
    shutil.copy(
        "./tests/data/annotations_cvat.zip",
        str(cvat_path / "cvat_annotations" / "zips"),
    )

    # Create dataset from CVAT annotations
    dataset_builder = CvatDatasetBuilder(
        name="MyCVATAnnotations",
        dataset_source=cvat_path,
        target=cvat_path / "datasets",
        split="test",
    )
    dataset_builder.retrieve_input_dataset()
    dataset_builder.save_to_disk(do_visualization=True)


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_run_cvat_on_pred():
    target_path = Path(f"./scratch/{BenchMarkNames.DPBENCH.value}-GT/")
    cvat_path = Path(f"./scratch/{BenchMarkNames.DPBENCH.value}-CVAT/")

    docling_provider = get_prediction_provider(PredictionProviderType.DOCLING)

    dataset_layout = DPBenchDatasetBuilder(
        target=target_path / "gt_dataset",
        begin_index=15,
        end_index=20,
    )  # 10-25 is a small range which has samples with tables included.

    dataset_layout.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset_layout.save_to_disk()  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    docling_provider.create_prediction_dataset(
        name=dataset_layout.name,
        gt_dataset_dir=target_path / "gt_dataset",
        target_dataset_dir=target_path / "eval_dataset_e2e",
    )

    builder = CvatPreannotationBuilder(
        dataset_source=target_path / "eval_dataset_e2e" / "test",
        target=cvat_path,
        bucket_size=20,
    )
    builder.prepare_for_annotation()

    ## Stage 2: Re-build dataset
    shutil.copy(
        "./tests/data/annotations_cvat.zip",
        str(cvat_path / "cvat_annotations" / "zips"),
    )

    # Create dataset from CVAT annotations
    dataset_builder = CvatDatasetBuilder(
        name="MyCVATAnnotations",
        dataset_source=cvat_path,
        target=cvat_path / "datasets",
        split="test",
    )
    dataset_builder.retrieve_input_dataset()
    dataset_builder.save_to_disk(do_visualization=True)
