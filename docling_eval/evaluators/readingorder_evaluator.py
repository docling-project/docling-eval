import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset
from deepsearch_glm.andromeda_nlp import nlp_model  # type: ignore
from docling.datamodel.base_models import BoundingBox
from docling_core.types.doc.document import DocItem, DoclingDocument, TextItem
from docling_core.utils.legacy import docling_document_to_legacy
from pydantic import BaseModel
from tqdm import tqdm  # type: ignore

from docling_eval.benchmarks.constants import BenchMarkColumns  # type: ignore
from docling_eval.utils.stats import DatasetStatistics, compute_stats

_log = logging.getLogger(__name__)


class PageReadingOrderEvaluation(BaseModel):
    # BBoxes are in BOTTOMLEFT origin and in the true order
    bboxes: List[Tuple[float, float, float, float]]
    pred_order: List[int]
    ard: float


class DatasetReadingOrderEvaluation(BaseModel):
    evaluations: List[PageReadingOrderEvaluation]
    ard_stats: DatasetStatistics


class ReadingOrderEvaluator:
    r"""
    Evaluate the reading order using the Average Relative Distance metric
    """

    def __init__(self):
        self._nlp_model = nlp_model(loglevel="error", text_ordering=True)

    def __call__(
        self, ds_path: Path, split: str = "test"
    ) -> DatasetReadingOrderEvaluation:
        parquet_files = str(ds_path / split / "*.parquet")
        ds = load_dataset("parquet", data_files={split: parquet_files})
        _log.info(f"oveview of dataset: {ds}")
        if ds is not None:
            ds_selection = ds[split]

        evaluations: list[PageReadingOrderEvaluation] = []
        ards = []

        broken_inputs = 0
        for i, data in tqdm(
            enumerate(ds_selection),
            desc="Reading order evaluations",
            ncols=120,
            total=len(ds_selection),
        ):
            doc_id = data[BenchMarkColumns.DOC_ID]
            true_doc_dict = data[BenchMarkColumns.GROUNDTRUTH]
            true_doc: DoclingDocument = DoclingDocument.model_validate_json(
                true_doc_dict
            )
            # print(f"\n{i} - doc_id: {doc_id}")
            # self._show_items(true_doc)

            reading_order = self._get_reading_order_preds(true_doc)
            if reading_order is None:
                print(f"Broken input: {doc_id}")
                broken_inputs += 1
                continue

            ard = self._compute_ard(reading_order)
            ards.append(ard)

            page_evaluation = PageReadingOrderEvaluation(
                bboxes=[b.as_tuple() for b in reading_order["bboxes"]],
                pred_order=reading_order["pred_order"],
                ard=ard,
            )
            # print("pred_reading_order")
            # print(page_evaluation)
            # print(f"ard={ard}")

            evaluations.append(page_evaluation)

        if broken_inputs > 0:
            _log.error(f"broken_inputs={broken_inputs}")

        # Compute statistics for metrics
        ard_stats = compute_stats(ards)

        ds_reading_order_evaluation = DatasetReadingOrderEvaluation(
            evaluations=evaluations, ard_stats=ard_stats
        )

        return ds_reading_order_evaluation

    def _get_reading_order_preds(self, true_doc: DoclingDocument):
        r""" """
        try:
            # Run the reading order model
            legacy_doc = docling_document_to_legacy(true_doc)
            legacy_doc_dict = legacy_doc.model_dump(by_alias=True, exclude_none=True)
            legacy_doc_dict = self._ensure_bboxes_in_legacy_tables(legacy_doc_dict)
            glm_doc = self._nlp_model.apply_on_doc(legacy_doc_dict)

            # Prepare index origin to predicted reading order
            orig_to_pred_order: Dict[int, int] = {}
            for po, pe in enumerate(glm_doc["page-elements"]):
                orig_to_pred_order[pe["orig-order"]] = po

            # Make index with the bbox objects in BOTTOM-LEFT origin
            # original_order -> bbox
            page_size = true_doc.pages[1].size
            bboxes = []
            pred_order = []
            for true_idx, (item, level) in enumerate(true_doc.iterate_items()):
                # Convert the bbox to BOTTOM-LEFT origin
                bbox = item.prov[0].bbox.to_bottom_left_origin(page_size.height)  # type: ignore
                bboxes.append(bbox)
                pred_order.append(orig_to_pred_order[true_idx])

            reading_order = {"bboxes": bboxes, "pred_order": pred_order}
            return reading_order
        except RuntimeError as ex:
            _log.error(str(ex))
            return None

    def _compute_ard(self, reading_order: Dict) -> float:
        r"""
        Compute the Average Relative Distance (ARD)
        """
        n = len(reading_order["bboxes"])
        if n == 0:
            return 0.0
        ard = 0.0
        for true_ro, pred_ro in enumerate(reading_order["pred_order"]):
            ard += math.fabs(true_ro - pred_ro)
        ard /= n
        return ard

    def _ensure_bboxes_in_legacy_tables(self, legacy_doc_dict: Dict):
        r"""
        Ensure bboxes for all table cells
        """
        for table in legacy_doc_dict["tables"]:
            for row in table["data"]:
                for cell in row:
                    if "bbox" not in cell:
                        cell["bbox"] = [0, 0, 0, 0]
        return legacy_doc_dict

    def _show_items(self, true_doc: DoclingDocument):
        r""" """
        page_size = true_doc.pages[1].size
        for i, (item, level) in enumerate(true_doc.iterate_items()):
            bbox = (
                item.prov[0].bbox.to_bottom_left_origin(page_size.height)
                if isinstance(item, DocItem)
                else None
            )
            text = item.text if isinstance(item, TextItem) else None
            label = item.label  # type: ignore
            print(f"True {i}: {level} - {label}: {bbox} - {text}")
