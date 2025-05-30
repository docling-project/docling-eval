import logging
import os
import shutil
from pathlib import Path

import pytest

from docling_eval.dataset_builders.dataset_builder import (
    BaseEvaluationDatasetBuilder,
    S3Source,
)

IS_CI = bool(os.getenv("CI"))

# Get logger
_log = logging.getLogger(__name__)


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI if the dataset in cos is very large."
)
def test_s3source():
    # Define the COS(s3) endpoints and buckets to pull the data from;
    # Make sure there is some data in there.
    endpoint = os.environ.get("S3_ENDPOINT")
    access_key = os.environ.get("S3_ACCESS_KEY")
    secret_key = os.environ.get("S3_SECRET_KEY")
    bucket = os.environ.get("S3_BUCKET")
    key_prefix = os.environ.get("S3_KEY_PREFIX")

    root_dir = Path("./scratch/s3source")
    target_path = root_dir / "evaluation_data"  # path for GT+Predictions on the dataset
    dataset_local_path = root_dir / "data_from_cos"  # path to download the dataset

    #  Clean the directory
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    if os.path.exists(dataset_local_path):
        shutil.rmtree(dataset_local_path)

    if not endpoint:
        raise ValueError("Please set the S3_ENDPOINT environment variable")
    if not access_key:
        raise ValueError("Please set the S3_ACCESS_KEY environment variable")
    if not secret_key:
        raise ValueError("Please set the S3_SECRET_KEY environment variable")
    if not bucket:
        raise ValueError("Please set the S3_BUCKET environment variable")
    if not key_prefix:
        raise ValueError("Please set the S3_KEY_PREFIX environment variable")

    dataset_source = S3Source(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        bucket=bucket,
        key_prefix=key_prefix,
        overwrite_downloads=True,
    )

    # Test 1: Specify separate target and dataset_local_path
    dataset_builder = BaseEvaluationDatasetBuilder(
        name="s3_dataset",
        dataset_source=dataset_source,
        target=target_path,
        dataset_local_path=dataset_local_path,
        end_index=-1,
    )

    output_dir = dataset_builder.retrieve_input_dataset()
    assert output_dir is not None

    assert (
        len(os.listdir(dataset_local_path)) > 0
    ), f"The directory {dataset_local_path} is empty."

    assert not (
        os.path.exists(target_path)
    ), f"Target directory {target_path} should NOT exist."

    # Test 2: Specify only target
    #  Clean the directory
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    if os.path.exists(dataset_local_path):
        shutil.rmtree(dataset_local_path)

    dataset_builder = BaseEvaluationDatasetBuilder(
        name="s3_dataset",
        dataset_source=dataset_source,
        target=target_path,
        end_index=-1,
    )

    output_dir = dataset_builder.retrieve_input_dataset()
    assert output_dir == target_path / "source_data"

    assert len(os.listdir(output_dir)) > 0, f"The directory {output_dir} is empty."
