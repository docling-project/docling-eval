import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, Set

from datasets import load_dataset
from docling_core.types import DoclingDocument
from docling_core.types.io import DocumentStream
from PIL import Image as PILImage

from docling_eval.datamodels.dataset_record import DatasetRecord
from docling_eval.dataset_builders.dataset_builder import (
    BaseEvaluationDatasetBuilder,
    HFSource,
)
from docling_eval.utils.utils import get_binary, get_binhash

# Get logger
_log = logging.getLogger(__name__)


class DoclingDPBenchDatasetBuilder(BaseEvaluationDatasetBuilder):
    """
    DoclingDPBench dataset builder implementing the base dataset builder interface.

    This builder processes the DoclingDPBench dataset, which contains document
    understanding benchmarks for various document types.
    """

    def __init__(
        self,
        target: Path,
        split: str = "test",
        begin_index: int = 0,
        end_index: int = -1,
    ):
        """
        Initialize the DoclingDPBench dataset builder.

        Args:
            target: Path where processed dataset will be saved
            split: Dataset split to use
            begin_index: Start index for processing (inclusive)
            end_index: End index for processing (exclusive), -1 means process all
        """
        super().__init__(
            name="DoclingDPBench",
            dataset_source=HFSource(repo_id="ds4sd/docling-dpbench"),
            target=target,
            split=split,
            begin_index=begin_index,
            end_index=end_index,
        )

        self.must_retrieve = True

    def iterate(self) -> Iterable[DatasetRecord]:
        """
        Iterate through the dataset and yield DatasetRecord objects.

        Yields:
            DatasetRecord objects
        """
        if not self.retrieved and self.must_retrieve:
            raise RuntimeError(
                "You must first retrieve the source dataset. Call retrieve_input_dataset()."
            )

        assert self.dataset_local_path is not None
        _log.info(f"dataset_local_path: {self.dataset_local_path}")

        # Login using e.g. `huggingface-cli login` to access this dataset
        ds = load_dataset("ds4sd/docling-dpbench")

        for idx, _ in enumerate(ds["test"]):
            doc_hash = str(get_binhash(_["BinaryDocument"]))
            doc = (DoclingDocument.model_validate_json(_["GroundTruthDocument"]),)

            page_images = [
                PILImage.open(BytesIO(__["bytes"])) for __ in _["GroundTruthPageImages"]
            ]
            pictures = [
                PILImage.open(BytesIO(__["bytes"])) for __ in _["GroundTruthPictures"]
            ]

            pdf_stream = DocumentStream(
                name=f"ds4sd/docling-dpbench/{idx}", stream=BytesIO(_["BinaryDocument"])
            )

            # Create dataset record
            record = DatasetRecord(
                doc_id=str(_["document_id"]),
                doc_hash=doc_hash,
                ground_truth_doc=doc[0],
                ground_truth_pictures=pictures,
                ground_truth_page_images=page_images,
                original=pdf_stream,
                mime_type=_["mimetype"],
            )

            yield record
