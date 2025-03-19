import os
from pathlib import Path
from abc import abstractmethod
from typing import Dict

from docling.document_converter import DocumentConverter
from docling_core.types import DoclingDocument
from docling_core.types.io import DocumentStream

from docling_eval.benchmarks.utils import docling_version

from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import (
    DoclingDocument,
    PageItem,
    ProvenanceItem,
    TableCell,
    TableData,
    TableItem,
)
from docling_core.types.doc.labels import DocItemLabel
from PIL import Image

class BasePredictionProvider:
    def __init__(
        self, **kwargs
    ):  # could be the docling converter options, the remote credentials for MS/Google, etc.
        self.provider_args = kwargs

    @abstractmethod
    def predict(self, stream: DocumentStream, **extra_args) -> DoclingDocument:
        return DoclingDocument(name="dummy")

    @abstractmethod
    def info(self) -> Dict:
        return {}


class DoclingPredictionProvider(BasePredictionProvider):
    def __init__(
        self, **kwargs
    ):  # could be the docling converter options, the remote credentials for MS/Google, etc.
        super().__init__(**kwargs)

        if kwargs is not None:
            if "format_options" in kwargs:
                self.doc_converter = DocumentConverter(
                    format_options=kwargs["format_options"]
                )
            else:
                self.doc_converter = DocumentConverter()

    def predict(self, stream: DocumentStream, **extra_args) -> DoclingDocument:
        return self.doc_converter.convert(stream).document

    def info(self) -> Dict:
        return {"asset": "Docling", "version": docling_version()}

