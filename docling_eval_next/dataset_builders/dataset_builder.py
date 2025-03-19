import os
from abc import abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional, Union

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
    # TBD
    pass


class BaseEvaluationDatasetBuilder:
    def __init__(
        self,
        name: str,
        dataset_source: Union[HFSource, S3Source, Path],
        prediction_provider: BasePredictionProvider,
        target: Path,
    ):
        self.name = name
        self.target: Path = target
        self.prediction_provider = prediction_provider
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
        else:
            raise RuntimeError(
                f"Unknown dataset_source type {type(self.dataset_source)}"
            )

        self.retrieved = True

        return path

    @abstractmethod
    def iterate(self) -> Iterable[DatasetRecord]:
        pass

    def update_prediction(self, record: DatasetRecord):
        # This might need customization depending on the input the dataset has.
        # The default implementation assumes that there is an original file in binary format which is accepted.
        input_data = record.original

        assert input_data is not None

        if not isinstance(input_data, DocumentStream):
            if isinstance(input_data, Path):
                input_data = DocumentStream(
                    name=input_data.name, stream=BytesIO(input_data.open("rb").read())
                )

        pred_doc = self.prediction_provider.predict(input_data)

        record.predicted_doc = pred_doc

        record.validate_model()  # type: ignore

    def save_to_disk(self):
        if not self.retrieved:
            raise RuntimeError(
                "You must first retrieve the source dataset. Call retrieve_input_dataset()."
            )

        test_dir = self.target / "test"
        os.makedirs(test_dir, exist_ok=True)

        count = 0
        for record_chunk in chunkify(self.iterate(), 80):
            record_chunk = [r.as_record_dict() for r in record_chunk]
            save_shard_to_disk(items=record_chunk, dataset_path=test_dir)
            count += len(record_chunk)

        write_datasets_info(
            name=self.name,
            output_dir=self.target,
            num_train_rows=0,
            num_test_rows=count,
        )
