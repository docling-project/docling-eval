import copy
import json
import logging
from pathlib import Path
from typing import Optional

from deepsearch_glm.andromeda_nlp import nlp_model  # type: ignore
from docling.utils.glm_utils import to_docling_document
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel
from docling_core.utils.legacy import (
    doc_item_label_to_legacy_name,
    docling_document_to_legacy,
)

from docling_eval.benchmarks.utils import get_input_document

_log = logging.getLogger(__name__)


class ReadingOrderUpdater:
    def __init__(self):
        r""" """
        self._nlp_model = nlp_model(loglevel="error", text_ordering=True)
        self._labels_forward_mapping = {
            doc_item_label_to_legacy_name(v): v.value for v in DocItemLabel
        }

    def __call__(
        self, pdf_path: Path, true_doc: DoclingDocument
    ) -> Optional[DoclingDocument]:
        r""" """
        print(true_doc.name)
        # deep copy of the true-document
        pred_doc = copy.deepcopy(true_doc)
        pred_doc_legacy = docling_document_to_legacy(pred_doc)
        ds_doc_dict = pred_doc_legacy.model_dump(by_alias=True, exclude_none=True)
        try:
            # TODO: Understand why some documents fail here
            glm_doc = self._nlp_model.apply_on_doc(ds_doc_dict)
        except RuntimeError as ex:
            # print("nlp_model.apply_on_doc()")
            return None

        # Map from value to key.value before calling to_docling_document
        for page_element in glm_doc["page-elements"]:
            page_element["name"] = self._labels_forward_mapping[page_element["name"]]

        # When true_doc.name == "ground-truth 01030000000016.pdf"
        # pydantic_core._pydantic_core.ValidationError: 1 validation error for TextItem label
        # Input should be <DocItemLabel.CAPTION: 'caption'>, <DocItemLabel.CHECKBOX_SELECTED: 'checkbox_selected'>,
        #  <DocItemLabel.CHECKBOX_UNSELECTED: 'checkbox_unselected'>, <DocItemLabel.CODE: 'code'>,
        #  <DocItemLabel.FOOTNOTE: 'footnote'>, <DocItemLabel.FORMULA: 'formula'>, <DocItemLabel.PAGE_FOOTER: 'page_footer'>,
        #  <DocItemLabel.PAGE_HEADER: 'page_header'>, <DocItemLabel.PARAGRAPH: 'paragraph'>, <DocItemLabel.REFERENCE: 'reference'>,
        #  <DocItemLabel.TEXT: 'text'> or <DocItemLabel.TITLE: 'title'>
        #  [type=literal_error, input_value=<DocItemLabel.DOCUMENT_INDEX: 'document_index'>, input_type=DocItemLabel]
        pred_doc = to_docling_document(glm_doc)

        return pred_doc
