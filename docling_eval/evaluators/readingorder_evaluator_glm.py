import copy
import logging
from typing import Dict, Optional

from deepsearch_glm.andromeda_nlp import nlp_model  # type: ignore
from docling_core.types.doc.document import DoclingDocument
from docling_core.utils.legacy import docling_document_to_legacy

from docling_eval.evaluators.base_readingorder_evaluator import (
    BaseReadingOrderEvaluator,
)

_log = logging.getLogger(__name__)


class ReadingOrderEvaluatorGlm(BaseReadingOrderEvaluator):
    r"""
    Evaluate the reading order using the Average Relative Distance metric
    """

    def __init__(self):
        self._nlp_model = nlp_model(loglevel="error", text_ordering=True)

    def _get_reading_order_preds(
        self, doc_id: str, true_doc: DoclingDocument
    ) -> Optional[dict]:
        r"""
        Return dict with the bboxes and the predicted reading order or None if something goes wrong.
        None is also returned if the document contains items with multiple provenances

        Returns
        -------
        reading_order: Keys are "bboxes" and "pred_order". Return None if the document is broken.
        """
        try:
            page_size = true_doc.pages[1].size

            # Convert the bboxes to bottom-left coords before running the GLM
            bboxes = []
            for item, level in true_doc.iterate_items():
                pred_len = len(item.prov)  # type: ignore
                if pred_len > 1:
                    _log.warning(
                        "Skipping document %s as it has %s provenances",
                        doc_id,
                        pred_len,
                    )
                    return None

                # Convert the bbox to BOTTOM-LEFT origin
                bbox = item.prov[0].bbox.to_bottom_left_origin(page_size.height)  # type: ignore
                item.prov[0].bbox = bbox  # type: ignore
                bboxes.append(copy.deepcopy(bbox))

            # Run the reading order model
            legacy_doc = docling_document_to_legacy(true_doc)
            legacy_doc_dict = legacy_doc.model_dump(by_alias=True, exclude_none=True)
            legacy_doc_dict = self._ensure_bboxes_in_legacy_tables(legacy_doc_dict)
            glm_doc = self._nlp_model.apply_on_doc(legacy_doc_dict)

            # pred_to_origin_order: predicted order -> original order
            pred_to_origin_order: Dict[int, int] = {}
            for po, pe in enumerate(glm_doc["page-elements"]):
                oo = pe["orig-order"]
                pred_to_origin_order[po] = oo

            # pred_order: The index is the predicted order and the value is the original order
            pred_order = [
                pred_to_origin_order[x] for x in range(len(pred_to_origin_order))
            ]

            reading_order = {"bboxes": bboxes, "pred_order": pred_order}
            return reading_order
        except RuntimeError as ex:
            _log.error(str(ex))
            return None
