from abc import abstractmethod
from typing import Dict, Optional

from docling.datamodel.pipeline_options import TableFormerMode
from docling.document_converter import DocumentConverter
from docling_core.types import DoclingDocument
from docling_core.types.io import DocumentStream

from docling_eval.benchmarks.utils import docling_models_version, docling_version
from docling_eval.converters.models.tableformer.tf_model_prediction import (
    PageToken,
    TableFormerUpdater,
)


class BasePredictionProvider:
    def __init__(
        self, **kwargs
    ):  # could be the docling converter options, the remote credentials for MS/Google, etc.
        self.provider_args = kwargs

    @abstractmethod
    def predict(
        self,
        gt_doc: DoclingDocument,
        stream: Optional[DocumentStream] = None,
        **extra_kwargs,
    ) -> DoclingDocument:
        return DoclingDocument(name="dummy")

    @abstractmethod
    def info(self) -> Dict:
        return {}


class NullPredictionProvider(BasePredictionProvider):
    def predict(
        self,
        gt_doc: DoclingDocument,
        stream: Optional[DocumentStream] = None,
        **extra_kwargs,
    ) -> DoclingDocument:
        return gt_doc

    def info(self) -> Dict:
        return {"asset": "NullProvider", "version": "0.0.0"}


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

    def predict(
        self,
        gt_doc: DoclingDocument,
        stream: Optional[DocumentStream] = None,
        **extra_kwargs,
    ) -> DoclingDocument:
        assert (
            stream is not None
        ), "stream must be given for docling prediction provider to work."

        return self.doc_converter.convert(stream).document

    def info(self) -> Dict:
        return {"asset": "Docling", "version": docling_version()}


class TableFormerPredictionProvider(BasePredictionProvider):
    def __init__(
        self, **kwargs
    ):  # could be the docling converter options, the remote credentials for MS/Google, etc.
        super().__init__(**kwargs)

        self.tf_updater = TableFormerUpdater(TableFormerMode.ACCURATE)

    def predict(
        self,
        gt_doc: DoclingDocument,
        stream: Optional[DocumentStream] = None,
        page_tokens: Optional[PageToken] = None,
        **extra_kwargs,
    ) -> DoclingDocument:

        assert (
            gt_doc is not None
        ), "true_doc must be given for TableFormer prediction provider to work."

        if stream is not None and page_tokens is None:
            updated, pred_doc = self.tf_updater.replace_tabledata(stream.stream, gt_doc)
        elif page_tokens is not None:
            updated, pred_doc = self.tf_updater.replace_tabledata_with_page_tokens(
                page_tokens, gt_doc, []
            )  # FIXME: Must not expect page images.
        else:
            raise RuntimeError(
                "TableFormerPredictionProvider.predict must be called with a stream or page_tokens."
            )
        return pred_doc

    def info(self) -> Dict:
        return {"asset": "TableFormer", "version": docling_models_version()}
