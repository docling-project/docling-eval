import os
import sys
from abc import abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from docling.utils.utils import chunkify
from docling_core.types.io import DocumentStream
from huggingface_hub import snapshot_download
from pydantic import BaseModel

from docling_eval.benchmarks.utils import save_shard_to_disk, write_datasets_info
from docling_eval_next.datamodels.dataset_record import DatasetRecord
from docling_eval_next.prediction_providers.prediction_provider import (
    BasePredictionProvider,
)


class HFSource(BaseModel):
    repo_id: str
    revision: Optional[str] = None
    hf_token: Optional[str] = os.getenv("HF_TOKEN", None)


class S3Source(BaseModel):
    endpoint: str
    access_key: str
    secret_key: str
    cos_bucket: str  # Bucket of interest inside of COS.
    cos_dir: str  # Path to dataset "directory" of interest in COS.
    cos_resource: Optional[Any] = None
    cos_client: Optional[Any] = None
    overwrite_downloads: Optional[bool] = True

    def __init__(self, **data):
        super().__init__(**data)
        self.cos_resource = self.initialize_s3_resource()
        self.cos_client = self.initialize_s3_client()

    def initialize_s3_client(self):
        """Initializes boto3 resource - s3 instance
        Returns the s3 instance
        """
        import ibm_boto3

        return ibm_boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )

    def initialize_s3_resource(self):
        """Initializes boto3 resource - s3 instance
        Returns the s3 instance
        """
        import ibm_boto3

        return ibm_boto3.resource(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )

    def download_objects(self, download_dir):
        """Downloads the objects from the bucket to the given download directory."""
        print(
            f"Download objects from {self.cos_bucket}/{self.cos_dir} to {download_dir}"
        )
        paginator = self.cos_client.get_paginator("list_objects_v2")
        pagination_params = {
            "Bucket": self.cos_bucket,
            "Prefix": self.cos_dir,
            "MaxKeys": 100,
        }
        page_iterator = paginator.paginate(**pagination_params)
        for page in page_iterator:
            for file_meta in page["Contents"]:
                # print(file_meta)
                relative_path = file_meta["Key"][len(self.cos_dir) + 1 :]
                if len(relative_path) == 0:
                    continue
                if file_meta["Size"] == 0:
                    continue

                # Identify the path to the file on disk.
                local_file_path = os.path.join(download_dir, relative_path)
                print(f"Download {file_meta['Key']} to {local_file_path}")

                # If the option to overwrite downloads is ON, and the file already exists, skip it.
                if self.overwrite_downloads and os.path.exists(local_file_path):
                    print(f"File {local_file_path} already exists. Skipping.")
                    continue

                # Create the directories as required
                local_dir = os.path.dirname(local_file_path)
                if not os.path.exists(local_dir):
                    os.makedirs(local_dir)

                self.cos_resource.Bucket(self.cos_bucket).download_file(
                    file_meta["Key"], local_file_path
                )

        return download_dir


class BaseEvaluationDatasetBuilder:
    def __init__(
        self,
        name: str,
        dataset_source: Union[HFSource, S3Source, Path],
        target: Path,
    ):
        self.name = name
        self.target: Path = target
        # self.prediction_provider = prediction_provider
        self.dataset_source = dataset_source

        self.dataset_local_path: Optional[Path] = None  # TBD

        self.retrieved = False

    def retrieve_input_dataset(self) -> Path:
        if isinstance(self.dataset_source, HFSource):
            if not self.dataset_local_path:
                path_str = snapshot_download(
                    repo_id=self.dataset_source.repo_id,
                    repo_type="dataset",
                    token=self.dataset_source.hf_token,
                    # local_dir=self.target,
                )
                path: Path = Path(path_str)
                self.dataset_local_path = path
            else:
                path_str = snapshot_download(
                    repo_id=self.dataset_source.repo_id,
                    repo_type="dataset",
                    token=self.dataset_source.hf_token,
                    local_dir=self.dataset_local_path,
                )
                path = Path(path_str)
        elif isinstance(self.dataset_source, Path):
            path = self.dataset_source
        elif isinstance(self.dataset_source, S3Source):
            # Download the data from S3 bucket to the target folder
            self.dataset_source.download_objects(self.target)
            path = Path(self.target)
            self.dataset_local_path = path
        else:
            raise RuntimeError(
                f"Unknown dataset_source type {type(self.dataset_source)}"
            )

        self.retrieved = True

        return path

    @abstractmethod
    def iterate(self) -> Iterable[DatasetRecord]:
        pass

    # def update_prediction(self, record: DatasetRecord):
    #     # This might need customization depending on the input the dataset has.
    #     # The default implementation assumes that there is an original file in binary format which is accepted.
    #     input_data = record.original
    #
    #     if not isinstance(input_data, DocumentStream):
    #         if isinstance(input_data, Path):
    #             input_data = DocumentStream(
    #                 name=input_data.name, stream=BytesIO(input_data.open("rb").read())
    #             )
    #
    #     pred_doc = self.prediction_provider.predict(
    #         record.ground_truth_doc, stream=input_data
    #     )
    #
    #     record.predicted_doc = pred_doc
    #
    #     record.validate_model()  # type: ignore

    def save_to_disk(self, chunk_size: int = 80, max_num_chunks: int = sys.maxsize):
        if not self.retrieved:
            raise RuntimeError(
                "You must first retrieve the source dataset. Call retrieve_input_dataset()."
            )

        test_dir = self.target / "test"
        os.makedirs(test_dir, exist_ok=True)

        count = 0
        chunk_count = 0
        for record_chunk in chunkify(self.iterate(), chunk_size):
            record_chunk = [r.as_record_dict() for r in record_chunk]
            save_shard_to_disk(
                items=record_chunk, dataset_path=test_dir, shard_id=chunk_count
            )
            count += len(record_chunk)
            chunk_count += 1

            if chunk_count >= max_num_chunks:
                break

        write_datasets_info(
            name=self.name,
            output_dir=self.target,
            num_train_rows=0,
            num_test_rows=count,
        )
