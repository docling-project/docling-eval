import copy
import logging
import os
import sys
from abc import abstractmethod
from pathlib import Path
from typing import Iterable, Optional, Union

import ibm_boto3  # type: ignore
from docling.utils.utils import chunkify
from docling_core.types.doc.document import ImageRefMode
from huggingface_hub import snapshot_download
from pydantic import BaseModel

from docling_eval.datamodels.dataset_record import (
    DatasetRecord,
    DatasetRecordWithPrediction,
)
from docling_eval.prediction_providers.base_prediction_provider import (
    TRUE_HTML_EXPORT_LABELS,
)
from docling_eval.utils.utils import (
    insert_images_from_pil,
    save_shard_to_disk,
    write_datasets_info,
)
from docling_eval.visualisation.visualisations import save_single_document_html

# Get logger
_log = logging.getLogger(__name__)


class HFSource(BaseModel):
    repo_id: str
    revision: Optional[str] = None
    hf_token: Optional[str] = os.getenv("HF_TOKEN", None)


class S3Source(BaseModel):
    endpoint: str
    access_key: str
    secret_key: str
    bucket: str  # Bucket of interest inside of COS.
    key_prefix: str  # Path to dataset "directory" of interest in COS.
    overwrite_downloads: bool = True

    def __init__(self, **data):
        super().__init__(**data)
        self._cos_resource: ibm_boto3.resource = self.initialize_s3_resource()
        self._cos_client: ibm_boto3.client = self.initialize_s3_client()

    def initialize_s3_client(self) -> ibm_boto3.client:
        """Initializes boto3 client - s3 instance
        Returns the s3 client
        """
        return ibm_boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )

    def initialize_s3_resource(self) -> ibm_boto3.resource:
        """Initializes boto3 resource - s3 instance
        Returns the s3 instance
        """

        return ibm_boto3.resource(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )

    def download_objects(self, download_dir: Path) -> Path:
        """Downloads the objects from the bucket to the given download directory."""
        _log.info(
            f"Download objects from {self.bucket}/{self.key_prefix} to {download_dir}"
        )
        paginator = self._cos_client.get_paginator("list_objects_v2")
        pagination_params = {
            "Bucket": self.bucket,
            "Prefix": self.key_prefix,
            "MaxKeys": 100,
        }
        page_iterator = paginator.paginate(**pagination_params)
        for page in page_iterator:
            for file_meta in page["Contents"]:
                relative_path = file_meta["Key"][len(self.key_prefix) + 1 :]
                if len(relative_path) == 0:
                    continue
                if file_meta["Size"] == 0:
                    continue

                # Identify the path to the file on disk.
                local_file_path = os.path.join(download_dir, relative_path)

                # If the option to overwrite downloads is ON, and the file already exists, skip it.
                if self.overwrite_downloads and os.path.exists(local_file_path):
                    _log.info(f"File {local_file_path} already exists. Skipping.")
                    continue

                # Create the directories as required
                local_dir = os.path.dirname(local_file_path)
                if not os.path.exists(local_dir):
                    os.makedirs(local_dir)

                self._cos_resource.Bucket(self.bucket).download_file(
                    file_meta["Key"], local_file_path
                )
                _log.info(f"Downloaded {file_meta['Key']} to {local_file_path}")

        return download_dir


class BaseEvaluationDatasetBuilder:
    """
    Base class for dataset builders that create evaluation datasets.

    This class provides common functionality for retrieving datasets,
    applying index ranges, and saving processed data to disk.
    """

    def __init__(
        self,
        name: str,
        dataset_source: Union[HFSource, S3Source, Path],
        target: Path,
        dataset_local_path: Optional[Path] = None,
        split: str = "test",
        begin_index: int = 0,
        end_index: int = -1,
    ):
        """
        Initialize the dataset builder.

        Args:
            name: Name of the dataset
            dataset_source: Source of the dataset (HuggingFace, S3, or local path)
            target: Path where processed dataset will be saved
            dataset_local_path: Path where the dataset will be saved as-is locally
            split: Dataset split to use (train, test, etc.)
            begin_index: Start index for processing (inclusive)
            end_index: End index for processing (exclusive), -1 means process all
        """
        self.name = name
        self.target: Path = target
        self.dataset_source = dataset_source
        self.dataset_local_path = dataset_local_path
        self.split = split
        self.begin_index = begin_index
        self.end_index = end_index
        self.retrieved = False

        self.must_retrieve = False

    def retrieve_input_dataset(self) -> Path:
        """
        Download and retrieve the input dataset.

        Returns:
            Path to the retrieved dataset
        """
        if isinstance(self.dataset_source, HFSource):
            if not self.dataset_local_path:
                path_str = snapshot_download(
                    repo_id=self.dataset_source.repo_id,
                    repo_type="dataset",
                    token=self.dataset_source.hf_token,
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
            if not self.dataset_local_path:
                self.dataset_local_path = self.target / "source_data"

            _log.info("Dataset local path = [%s]", self.dataset_local_path)
            _log.info("Target path = [%s]", self.target)

            # Download the data from S3 bucket
            self.dataset_source.download_objects(self.dataset_local_path)
            path = self.dataset_local_path
        else:
            raise RuntimeError(
                f"Unknown dataset_source type {type(self.dataset_source)}"
            )

        self.retrieved = True
        return path

    def get_effective_indices(self, total_items: int) -> tuple[int, int]:
        """
        Calculate the effective begin and end indices based on dataset size.

        Args:
            total_items: Total number of items available

        Returns:
            Tuple of (effective_begin_index, effective_end_index)
        """
        begin = self.begin_index if self.begin_index >= 0 else 0
        end = self.end_index if self.end_index > 0 else total_items
        end = min(end, total_items)

        if begin >= total_items:
            _log.warning(
                f"Begin index ({begin}) is greater than or equal to dataset size ({total_items}). "
                f"No items will be processed."
            )
            begin = total_items

        _log.info(
            f"Processing range [{begin}:{end}] out of {total_items} total items "
            f"({end - begin} items)"
        )

        return begin, end

    def log_dataset_stats(self, total_items: int, selected_items: int) -> None:
        """
        Log dataset statistics for debugging.

        Args:
            total_items: Total number of items in the dataset
            selected_items: Number of items selected after applying indices
        """
        _log.info(
            f"Dataset '{self.name}' total items: {total_items}. "
            f"Selected range: [{self.begin_index}, {self.end_index}] = {selected_items} items"
        )

    @abstractmethod
    def iterate(self) -> Iterable[DatasetRecord]:
        """
        Iterate through the dataset and yield DatasetRecord objects.

        Implementations should respect begin_index and end_index.

        Returns:
            Iterable of DatasetRecord objects
        """
        pass

    def save_to_disk(
        self,
        chunk_size: int = 80,
        max_num_chunks: int = sys.maxsize,
        do_visualization: bool = False,
    ) -> None:
        """
        Save the dataset to disk in chunks.

        Args:
            chunk_size: Number of records per chunk
            max_num_chunks: Maximum number of chunks to save
            do_visualization: Whether to generate visualizations
        """
        if not self.retrieved and self.must_retrieve:
            raise RuntimeError(
                "You must first retrieve the source dataset. Call retrieve_input_dataset()."
            )

        test_dir = self.target / self.split
        test_dir.mkdir(parents=True, exist_ok=True)

        if do_visualization:
            viz_path = self.target / "visualizations"
            viz_path.mkdir(exist_ok=True)

        count = 0
        chunk_count = 0

        for record_chunk in chunkify(self.iterate(), chunk_size):
            record_list = []
            for r in record_chunk:
                record_list.append(r.as_record_dict())
                if do_visualization:
                    viz_path_split = self.target / "visualizations" / f"{r.doc_id}.html"

                    # Create a visualization using the same approach as BasePredictionProvider
                    # but only for the ground truth document
                    tmp = insert_images_from_pil(
                        document=copy.deepcopy(r.ground_truth_doc),
                        pictures=r.ground_truth_pictures,
                        page_images=r.ground_truth_page_images,
                    )

                    # Save visualization using the single document template
                    save_single_document_html(
                        filename=viz_path_split,
                        doc=tmp,
                        draw_reading_order=True,
                    )

            save_shard_to_disk(
                items=record_list,
                dataset_path=test_dir,
                schema=DatasetRecord.pyarrow_schema(),
                shard_id=chunk_count,
            )
            count += len(record_list)
            chunk_count += 1

            if chunk_count >= max_num_chunks:
                _log.info(
                    f"Reached maximum number of chunks ({max_num_chunks}). Stopping."
                )
                break

        _log.info(f"Saved {count} records in {chunk_count} chunks to {test_dir}")

        write_datasets_info(
            name=self.name,
            output_dir=self.target,
            num_train_rows=0,
            num_test_rows=count,
            features=DatasetRecord.features(),
        )
