import logging
import os
import sys
from abc import abstractmethod
from pathlib import Path
from typing import Iterable, Optional, Union

from docling.utils.utils import chunkify
from huggingface_hub import snapshot_download
from pydantic import BaseModel

from docling_eval.datamodels.dataset_record import DatasetRecord
from docling_eval.utils.utils import save_shard_to_disk, write_datasets_info

# Get logger
_log = logging.getLogger(__name__)


class HFSource(BaseModel):
    repo_id: str
    revision: Optional[str] = None
    hf_token: Optional[str] = os.getenv("HF_TOKEN", None)


class S3Source(BaseModel):
    # TBD
    pass


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
            split: Dataset split to use (train, test, etc.)
            begin_index: Start index for processing (inclusive)
            end_index: End index for processing (exclusive), -1 means process all
        """
        self.name = name
        self.target: Path = target
        self.dataset_source = dataset_source
        self.dataset_local_path: Optional[Path] = None
        self.split = split
        self.begin_index = begin_index
        self.end_index = end_index
        self.retrieved = False

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
        self, chunk_size: int = 80, max_num_chunks: int = sys.maxsize
    ) -> None:
        """
        Save the dataset to disk in chunks.

        Args:
            chunk_size: Number of records per chunk
            max_num_chunks: Maximum number of chunks to save
        """
        if not self.retrieved:
            raise RuntimeError(
                "You must first retrieve the source dataset. Call retrieve_input_dataset()."
            )

        test_dir = self.target / self.split
        test_dir.mkdir(parents=True, exist_ok=True)

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
        )
