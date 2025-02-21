import copy
import logging
import random
from typing import Dict, List, Optional

from docling_core.types.doc.document import DocItem, DoclingDocument, RefItem
from docling_ibm_models.reading_order.reading_order_rb import (
    PageElement as ReadingOrderPageElement,
)
from docling_ibm_models.reading_order.reading_order_rb import ReadingOrderPredictor

from docling_eval.evaluators.base_readingorder_evaluator import (
    BaseReadingOrderEvaluator,
)

_log = logging.getLogger(__name__)


class ReadingOrderEvaluator(BaseReadingOrderEvaluator):
    r"""
    Evaluate the reading order using the Average Relative Distance metric
    """

    def __init__(self):
        self.ro_model = ReadingOrderPredictor()

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
            ro_elements: List[ReadingOrderPageElement] = []

            for ix, (item, level) in enumerate(true_doc.iterate_items()):
                assert isinstance(item, DocItem)  # this is satisfied, make mypy happy.

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
                page_no = item.prov[0].page_no

                # item.prov[0].bbox = bbox  # type: ignore
                bboxes.append(copy.deepcopy(bbox))
                ro_elements.append(
                    ReadingOrderPageElement(
                        cid=len(ro_elements),
                        ref=RefItem(cref=f"#/{ix}"),
                        text="dummy",
                        page_no=page_no,
                        page_size=true_doc.pages[page_no].size,
                        label=item.label,
                        l=bbox.l,
                        r=bbox.r,
                        b=bbox.b,
                        t=bbox.t,
                        coord_origin=bbox.coord_origin,
                    )
                )
            random.shuffle(ro_elements)
            sorted_elements = self.ro_model.predict_reading_order(
                page_elements=ro_elements
            )

            # pred_to_origin_order: predicted order -> original order
            pred_to_origin_order: Dict[int, int] = {}

            for ix, el in enumerate(sorted_elements):
                pred_to_origin_order[ix] = el.cid

            # pred_order: The index is the predicted order and the value is the original order
            pred_order = [
                pred_to_origin_order[x] for x in range(len(pred_to_origin_order))
            ]

            reading_order = {"bboxes": bboxes, "pred_order": pred_order}
            return reading_order
        except RuntimeError as ex:
            _log.error(str(ex))
            return None
