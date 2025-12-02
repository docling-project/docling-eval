import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from datasets import load_dataset
from docling_core.types.doc.document import DoclingDocument, KeyValueItem
from editdistance import eval as _edit_distance_eval
from pydantic import Field
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm  # type: ignore

from docling_eval.datamodels.cvat_page_mapping import CvatPageMapping, PageRef
from docling_eval.datamodels.dataset_record import DatasetRecordWithPrediction
from docling_eval.datamodels.types import PredictionFormats
from docling_eval.evaluators.base_evaluator import (
    BaseEvaluator,
    DatasetEvaluation,
    EvaluationRejectionType,
    UnitEvaluation,
    docling_document_from_doctags,
)
from docling_eval.evaluators.stats import DatasetStatistics, compute_stats

_log = logging.getLogger(__name__)


# --- Levenshtein-compatible helpers using 'editdistance' --------------------
def distance(s1: str, s2: str) -> int:
    return _edit_distance_eval(s1, s2)


def ratio(s1: str, s2: str) -> float:
    total = len(s1) + len(s2)
    if total == 0:
        return 1.0
    return (total - distance(s1, s2)) / total


# ---------- basic utilities ----------
def normalize_text(text: str) -> str:
    return text.strip().lower()


def precision(tp: int, fp: int) -> float:
    if tp == 0 and fp == 0:
        return 1.0
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(tp: int, fn: int) -> float:
    if tp == 0 and fn == 0:
        return 1.0
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(p: float, r: float) -> float:
    if p == 1.0 and r == 1.0:
        return 1.0
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def is_text_matching(pred_text: str, gt_text: str, is_strict: bool = True) -> bool:
    return (
        pred_text == gt_text
        if is_strict
        else (distance(pred_text, gt_text) <= 3 or ratio(pred_text, gt_text) >= 0.8)
    )


# ---------- entity recognition ----------
def evaluate_entity_recognition(
    gt_doc: DoclingDocument, pred_doc: DoclingDocument, is_strict: bool = False
):
    if (
        gt_doc.key_value_items is None
        or pred_doc.key_value_items is None
        or len(gt_doc.key_value_items) == 0
        or len(pred_doc.key_value_items) == 0
    ):
        return 0, 0, 0, 0.0, 0.0, 0.0
    gt_item = gt_doc.key_value_items[0]
    pred_item = pred_doc.key_value_items[0]

    gt_texts = [normalize_text(c.text) for c in gt_item.graph.cells]
    pred_texts = [normalize_text(c.text) for c in pred_item.graph.cells]

    if is_strict:
        gt_texts_set = set(gt_texts)
        pred_texts_set = set(pred_texts)
        tp = len(gt_texts_set.intersection(pred_texts_set))
        fp = len(pred_texts_set - gt_texts_set)
        fn = len(gt_texts_set - pred_texts_set)
    else:
        matched_gt = [False] * len(gt_texts)
        tp = 0
        for p_text in pred_texts:
            for idx, g_text in enumerate(gt_texts):
                if not matched_gt[idx] and is_text_matching(p_text, g_text, is_strict):
                    matched_gt[idx] = True
                    tp += 1
                    break
        fp = len(pred_texts) - tp
        fn = len(gt_texts) - tp
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    f1 = f1_score(prec, rec)
    return tp, fp, fn, prec, rec, f1


# ---------- relation helpers ----------
def extract_relation_set(item: KeyValueItem):
    cell_map = {c.cell_id: (normalize_text(c.text), c.label) for c in item.graph.cells}
    rels = set()
    for link in item.graph.links:
        if link.source_cell_id in cell_map and link.target_cell_id in cell_map:
            rels.add((cell_map[link.source_cell_id], cell_map[link.target_cell_id]))
    return rels


# ---------- relation extraction ----------
def evaluate_relation_extraction(
    gt_doc: DoclingDocument, pred_doc: DoclingDocument, is_strict: bool = False
):
    if (
        gt_doc.key_value_items is None
        or pred_doc.key_value_items is None
        or len(gt_doc.key_value_items) == 0
        or len(pred_doc.key_value_items) == 0
    ):
        return 0, 0, 0, 0.0, 0.0, 0.0
    gt_rels = extract_relation_set(gt_doc.key_value_items[0])
    pred_rels = extract_relation_set(pred_doc.key_value_items[0])

    tp = 0
    for p_src, p_tgt in pred_rels:
        for g_src, g_tgt in gt_rels:
            if is_text_matching(p_src[0], g_src[0], is_strict) and is_text_matching(
                p_tgt[0], g_tgt[0], is_strict
            ):
                tp += 1
                break

    fp = len(pred_rels) - tp
    fn = len(gt_rels) - tp
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    f1 = f1_score(prec, rec)
    return tp, fp, fn, prec, rec, f1


# ---------- layout-aware entity ----------
def evaluate_entity_extraction_with_bbox(
    gt_doc: DoclingDocument, pred_doc: DoclingDocument, is_strict: bool = False
):
    if (
        gt_doc.key_value_items is None
        or pred_doc.key_value_items is None
        or len(gt_doc.key_value_items) == 0
        or len(pred_doc.key_value_items) == 0
    ):
        return 0, 0, 0, 0.0, 0.0, 0.0
    gt_cells = gt_doc.key_value_items[0].graph.cells
    pred_cells = pred_doc.key_value_items[0].graph.cells

    # Build cost matrix (negative IoU for matches; large cost otherwise)
    cost = np.full((len(pred_cells), len(gt_cells)), 1e6)
    for i, p in enumerate(pred_cells):
        for j, g in enumerate(gt_cells):
            if not is_text_matching(
                normalize_text(p.text), normalize_text(g.text), is_strict
            ):
                continue
            if p.prov is None or g.prov is None:
                continue
            iou = p.prov.bbox.intersection_over_union(g.prov.bbox)
            if iou > 0:
                cost[i, j] = -iou

    row_ind, col_ind = linear_sum_assignment(cost)
    tp = sum(cost[r, c] < 1e6 for r, c in zip(row_ind, col_ind))
    fp = len(pred_cells) - tp
    fn = len(gt_cells) - tp
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    f1 = f1_score(prec, rec)
    return tp, fp, fn, prec, rec, f1


# ---------- layout-aware relation ----------
def extract_relation_set_with_bbox(item: KeyValueItem):
    cell_map = {}
    for c in item.graph.cells:
        cell_map[c.cell_id] = (
            normalize_text(c.text),
            c.label,
            c.prov.bbox if c.prov else None,
        )
    rels = []
    for link in item.graph.links:
        if link.source_cell_id in cell_map and link.target_cell_id in cell_map:
            rels.append((cell_map[link.source_cell_id], cell_map[link.target_cell_id]))
    return rels


def evaluate_relation_extraction_with_bbox(
    gt_doc: DoclingDocument, pred_doc: DoclingDocument, is_strict: bool = False
):
    if (
        gt_doc.key_value_items is None
        or pred_doc.key_value_items is None
        or len(gt_doc.key_value_items) == 0
        or len(pred_doc.key_value_items) == 0
    ):
        return 0, 0, 0, 0.0, 0.0, 0.0
    gt_rels = extract_relation_set_with_bbox(gt_doc.key_value_items[0])
    pred_rels = extract_relation_set_with_bbox(pred_doc.key_value_items[0])

    cost = np.full((len(pred_rels), len(gt_rels)), 1e6)
    for i, (p_src, p_tgt) in enumerate(pred_rels):
        for j, (g_src, g_tgt) in enumerate(gt_rels):
            if (
                p_src[2] is None
                or g_src[2] is None
                or p_tgt[2] is None
                or g_tgt[2] is None
            ):
                continue
            if not is_text_matching(
                p_src[0], g_src[0], is_strict
            ) or not is_text_matching(p_tgt[0], g_tgt[0], is_strict):
                continue
            iou_src = p_src[2].intersection_over_union(g_src[2])
            iou_tgt = p_tgt[2].intersection_over_union(g_tgt[2])
            if iou_src > 0 and iou_tgt > 0:
                cost[i, j] = -(iou_src + iou_tgt) / 2.0

    row_ind, col_ind = linear_sum_assignment(cost)
    tp = sum(cost[r, c] < 1e6 for r, c in zip(row_ind, col_ind))
    fp = len(pred_rels) - tp
    fn = len(gt_rels) - tp
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    f1 = f1_score(prec, rec)
    return tp, fp, fn, prec, rec, f1


# ---------- count helpers ----------
def count_entities(doc: DoclingDocument) -> int:
    """Count the number of entities (keys and values) in the document."""
    if doc.key_value_items is None or len(doc.key_value_items) == 0:
        return 0
    return len(doc.key_value_items[0].graph.cells)


def count_links(doc: DoclingDocument) -> int:
    """Count the number of links in the document."""
    if doc.key_value_items is None or len(doc.key_value_items) == 0:
        return 0
    return len(doc.key_value_items[0].graph.links)


# --------------------------------------------------------------------------- #
# Per-document evaluation record
# --------------------------------------------------------------------------- #
class KeyValueEvaluation(UnitEvaluation):
    doc_id: str

    # -------- entities (text only) --------
    entity_tp: int
    entity_fp: int
    entity_fn: int
    entity_precision: float
    entity_recall: float
    entity_f1: float

    # -------- entities (layout-aware) ----
    entity_tp_bbox: int
    entity_fp_bbox: int
    entity_fn_bbox: int
    entity_precision_bbox: float
    entity_recall_bbox: float
    entity_f1_bbox: float

    # -------- relations (text only) ------
    relation_tp: int
    relation_fp: int
    relation_fn: int
    relation_precision: float
    relation_recall: float
    relation_f1: float

    # -------- relations (layout-aware) ---
    relation_tp_bbox: int
    relation_fp_bbox: int
    relation_fn_bbox: int
    relation_precision_bbox: float
    relation_recall_bbox: float
    relation_f1_bbox: float

    # -------- count differences ----------
    num_entity_diff: int
    num_link_diff: int
    num_entity_diff_normalized: float
    num_link_diff_normalized: float


class KeyValuePageEvaluation(UnitEvaluation):
    doc_id: str
    page_no: int
    image_name: str

    entity_tp: int
    entity_fp: int
    entity_fn: int
    entity_precision: float
    entity_recall: float
    entity_f1: float

    entity_tp_bbox: int
    entity_fp_bbox: int
    entity_fn_bbox: int
    entity_precision_bbox: float
    entity_recall_bbox: float
    entity_f1_bbox: float

    relation_tp: int
    relation_fp: int
    relation_fn: int
    relation_precision: float
    relation_recall: float
    relation_f1: float

    relation_tp_bbox: int
    relation_fp_bbox: int
    relation_fn_bbox: int
    relation_precision_bbox: float
    relation_recall_bbox: float
    relation_f1_bbox: float

    num_entity_diff: int
    num_link_diff: int
    num_entity_diff_normalized: float
    num_link_diff_normalized: float


# --------------------------------------------------------------------------- #
# Dataset-level evaluation
# --------------------------------------------------------------------------- #
class DatasetKeyValueEvaluation(DatasetEvaluation):
    evaluations: List[KeyValueEvaluation]

    # --- statistics ---
    entity_precision_stats: DatasetStatistics
    entity_recall_stats: DatasetStatistics
    entity_f1_stats: DatasetStatistics

    entity_precision_bbox_stats: DatasetStatistics
    entity_recall_bbox_stats: DatasetStatistics
    entity_f1_bbox_stats: DatasetStatistics

    relation_precision_stats: DatasetStatistics
    relation_recall_stats: DatasetStatistics
    relation_f1_stats: DatasetStatistics

    relation_precision_bbox_stats: DatasetStatistics
    relation_recall_bbox_stats: DatasetStatistics
    relation_f1_bbox_stats: DatasetStatistics

    # --- count difference statistics ---
    num_entity_diff_stats: DatasetStatistics
    num_link_diff_stats: DatasetStatistics
    num_entity_diff_normalized_stats: DatasetStatistics
    num_link_diff_normalized_stats: DatasetStatistics

    evaluations_per_page: List[KeyValuePageEvaluation] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Main evaluator
# --------------------------------------------------------------------------- #
class KeyValueEvaluator(BaseEvaluator):
    r"""
    Key-Value evaluator that plugs into the existing benchmarking framework.
    It supports multiple prediction sources (DoclingDocument, JSON, YAML, Doctags)
    and computes both traditional and layout-aware scores.
    """

    def __init__(
        self,
        intermediate_evaluations_path: Optional[Path] = None,
        prediction_sources: List[PredictionFormats] | None = None,
        strict_matching: bool = False,
        page_mapping_path: Optional[Path] = None,
    ):
        supported_prediction_formats = [
            PredictionFormats.DOCLING_DOCUMENT,
            PredictionFormats.JSON,
            PredictionFormats.YAML,
            PredictionFormats.DOCTAGS,
        ]
        if prediction_sources is None or len(prediction_sources) == 0:
            prediction_sources = supported_prediction_formats

        super().__init__(
            intermediate_evaluations_path=intermediate_evaluations_path,
            prediction_sources=prediction_sources,
            supported_prediction_formats=supported_prediction_formats,
        )

        self._strict = strict_matching
        self._page_mapping_path = page_mapping_path
        self._page_mapping: Optional[CvatPageMapping] = None

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def __call__(self, ds_path: Path, split: str = "test") -> DatasetKeyValueEvaluation:
        split_glob = str(ds_path / split / "*.parquet")
        ds = load_dataset("parquet", data_files={split: split_glob})
        _log.info("Loaded split '%s' – %d samples", split, len(ds[split]))

        rejected_samples: Dict[EvaluationRejectionType, int] = {
            EvaluationRejectionType.INVALID_CONVERSION_STATUS: 0,
            EvaluationRejectionType.MISSING_PREDICTION: 0,
        }

        # Dataset-level bucket for every scalar metric
        ds_metrics: Dict[str, List[float]] = {  # type: ignore
            k: []  # creates identical lists on demand
            for k in [
                "entity_precision",
                "entity_recall",
                "entity_f1",
                "entity_precision_bbox",
                "entity_recall_bbox",
                "entity_f1_bbox",
                "relation_precision",
                "relation_recall",
                "relation_f1",
                "relation_precision_bbox",
                "relation_recall_bbox",
                "relation_f1_bbox",
                "num_entity_diff",
                "num_link_diff",
                "num_entity_diff_normalized",
                "num_link_diff_normalized",
            ]
        }

        all_evals: List[KeyValueEvaluation] = []
        page_evaluations: List[KeyValuePageEvaluation] = []

        for i, data in tqdm(
            enumerate(ds[split]),
            total=len(ds[split]),
            ncols=120,
            desc="Key-Value evaluations",
        ):
            record = DatasetRecordWithPrediction.model_validate(data)
            doc_id = record.doc_id

            # ----- sanity checks --------------------------------------------------
            if record.status not in self._accepted_status:
                rejected_samples[EvaluationRejectionType.INVALID_CONVERSION_STATUS] += 1
                _log.error("Skipping %s – conversion failed", doc_id)
                continue

            gt_doc = record.ground_truth_doc
            pred_doc = self._get_pred_doc(record)
            if pred_doc is None:
                rejected_samples[EvaluationRejectionType.MISSING_PREDICTION] += 1
                _log.error("Skipping %s – missing prediction", doc_id)
                continue

            # ----- compute metrics ------------------------------------------------
            ent_tp, ent_fp, ent_fn, ent_prec, ent_rec, ent_f1 = (
                evaluate_entity_recognition(gt_doc, pred_doc, is_strict=self._strict)
            )
            (
                ent_tp_b,
                ent_fp_b,
                ent_fn_b,
                ent_prec_b,
                ent_rec_b,
                ent_f1_b,
            ) = evaluate_entity_extraction_with_bbox(
                gt_doc, pred_doc, is_strict=self._strict
            )

            rel_tp, rel_fp, rel_fn, rel_prec, rel_rec, rel_f1 = (
                evaluate_relation_extraction(gt_doc, pred_doc, is_strict=self._strict)
            )
            (
                rel_tp_b,
                rel_fp_b,
                rel_fn_b,
                rel_prec_b,
                rel_rec_b,
                rel_f1_b,
            ) = evaluate_relation_extraction_with_bbox(
                gt_doc, pred_doc, is_strict=self._strict
            )

            # ----- compute count differences -------------------------------------
            gt_entity_count = count_entities(gt_doc)
            pred_entity_count = count_entities(pred_doc)
            entity_diff = abs(gt_entity_count - pred_entity_count)
            max_entity_count = max(gt_entity_count, pred_entity_count)
            entity_diff_normalized = (
                (entity_diff / max_entity_count) if max_entity_count > 0 else 0.0
            )

            gt_link_count = count_links(gt_doc)
            pred_link_count = count_links(pred_doc)
            link_diff = abs(gt_link_count - pred_link_count)
            max_link_count = max(gt_link_count, pred_link_count)
            link_diff_normalized = (
                (link_diff / max_link_count) if max_link_count > 0 else 0.0
            )

            # ----- accumulate dataset metrics ------------------------------------
            for key, val in [
                ("entity_precision", ent_prec),
                ("entity_recall", ent_rec),
                ("entity_f1", ent_f1),
                ("entity_precision_bbox", ent_prec_b),
                ("entity_recall_bbox", ent_rec_b),
                ("entity_f1_bbox", ent_f1_b),
                ("relation_precision", rel_prec),
                ("relation_recall", rel_rec),
                ("relation_f1", rel_f1),
                ("relation_precision_bbox", rel_prec_b),
                ("relation_recall_bbox", rel_rec_b),
                ("relation_f1_bbox", rel_f1_b),
                ("num_entity_diff", entity_diff),
                ("num_link_diff", link_diff),
                ("num_entity_diff_normalized", entity_diff_normalized),
                ("num_link_diff_normalized", link_diff_normalized),
            ]:
                ds_metrics[key].append(float(val))

            # ----- per-document record -------------------------------------------
            evaluation = KeyValueEvaluation(
                doc_id=doc_id,
                # entities
                entity_tp=ent_tp,
                entity_fp=ent_fp,
                entity_fn=ent_fn,
                entity_precision=ent_prec,
                entity_recall=ent_rec,
                entity_f1=ent_f1,
                entity_tp_bbox=ent_tp_b,
                entity_fp_bbox=ent_fp_b,
                entity_fn_bbox=ent_fn_b,
                entity_precision_bbox=ent_prec_b,
                entity_recall_bbox=ent_rec_b,
                entity_f1_bbox=ent_f1_b,
                # relations
                relation_tp=rel_tp,
                relation_fp=rel_fp,
                relation_fn=rel_fn,
                relation_precision=rel_prec,
                relation_recall=rel_rec,
                relation_f1=rel_f1,
                relation_tp_bbox=rel_tp_b,
                relation_fp_bbox=rel_fp_b,
                relation_fn_bbox=rel_fn_b,
                relation_precision_bbox=rel_prec_b,
                relation_recall_bbox=rel_rec_b,
                relation_f1_bbox=rel_f1_b,
                # count differences
                num_entity_diff=entity_diff,
                num_link_diff=link_diff,
                num_entity_diff_normalized=entity_diff_normalized,
                num_link_diff_normalized=link_diff_normalized,
            )
            all_evals.append(evaluation)

            page_mapping = self._get_page_mapping()
            if page_mapping is not None:
                page_evals = self._evaluate_per_page(
                    doc_id,
                    gt_doc,
                    pred_doc,
                    page_mapping,
                )
                page_evaluations.extend(page_evals)

            # optional: dump intermediate JSON
            if self._intermediate_evaluations_path is not None:
                self.save_intermediate_evaluations("KeyValue", i, doc_id, [evaluation])

        # ----------------------------------------------------------------- #
        # Aggregate dataset-level statistics
        # ----------------------------------------------------------------- #
        dataset_eval = DatasetKeyValueEvaluation(
            evaluated_samples=len(all_evals),
            rejected_samples=rejected_samples,
            evaluations=all_evals,
            # ---------- entity -------------
            entity_precision_stats=compute_stats(ds_metrics["entity_precision"]),
            entity_recall_stats=compute_stats(ds_metrics["entity_recall"]),
            entity_f1_stats=compute_stats(ds_metrics["entity_f1"]),
            entity_precision_bbox_stats=compute_stats(
                ds_metrics["entity_precision_bbox"]
            ),
            entity_recall_bbox_stats=compute_stats(ds_metrics["entity_recall_bbox"]),
            entity_f1_bbox_stats=compute_stats(ds_metrics["entity_f1_bbox"]),
            # ---------- relation -----------
            relation_precision_stats=compute_stats(ds_metrics["relation_precision"]),
            relation_recall_stats=compute_stats(ds_metrics["relation_recall"]),
            relation_f1_stats=compute_stats(ds_metrics["relation_f1"]),
            relation_precision_bbox_stats=compute_stats(
                ds_metrics["relation_precision_bbox"]
            ),
            relation_recall_bbox_stats=compute_stats(
                ds_metrics["relation_recall_bbox"]
            ),
            relation_f1_bbox_stats=compute_stats(ds_metrics["relation_f1_bbox"]),
            # ---------- count differences --
            num_entity_diff_stats=compute_stats(ds_metrics["num_entity_diff"]),
            num_link_diff_stats=compute_stats(ds_metrics["num_link_diff"]),
            num_entity_diff_normalized_stats=compute_stats(
                ds_metrics["num_entity_diff_normalized"]
            ),
            num_link_diff_normalized_stats=compute_stats(
                ds_metrics["num_link_diff_normalized"]
            ),
            evaluations_per_page=page_evaluations,
        )
        return dataset_eval

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _get_pred_doc(
        self, data_record: DatasetRecordWithPrediction
    ) -> Optional[DoclingDocument]:
        """Fetch the prediction in the first available format declared by `prediction_sources`."""
        pred_doc: Optional[DoclingDocument] = None

        for fmt in self._prediction_sources:
            if fmt == PredictionFormats.DOCLING_DOCUMENT:
                pred_doc = data_record.predicted_doc
            elif fmt == PredictionFormats.JSON and data_record.original_prediction:
                pred_doc = DoclingDocument.load_from_json(
                    data_record.original_prediction
                )
            elif fmt == PredictionFormats.YAML and data_record.original_prediction:
                pred_doc = DoclingDocument.load_from_yaml(
                    data_record.original_prediction
                )
            elif fmt == PredictionFormats.DOCTAGS:
                pred_doc = docling_document_from_doctags(data_record)

            if pred_doc is not None:
                break

        return pred_doc

    def _get_page_mapping(self) -> Optional[CvatPageMapping]:
        if self._page_mapping_path is None:
            return None
        if self._page_mapping is None:
            self._page_mapping = CvatPageMapping.from_overview_path(
                self._page_mapping_path
            )
        return self._page_mapping

    def _evaluate_per_page(
        self,
        doc_id: str,
        gt_doc: DoclingDocument,
        pred_doc: DoclingDocument,
        page_mapping: CvatPageMapping,
    ) -> List[KeyValuePageEvaluation]:
        results: List[KeyValuePageEvaluation] = []

        page_refs: List[PageRef] = page_mapping.pages_for_doc_name(doc_id)
        if not page_refs:
            return results

        for ref in page_refs:
            gt_page_doc = self._filter_doc_to_page(gt_doc, ref.page_no)
            pred_page_doc = self._filter_doc_to_page(pred_doc, ref.page_no)

            (
                ent_tp,
                ent_fp,
                ent_fn,
                ent_prec,
                ent_rec,
                ent_f1,
            ) = evaluate_entity_recognition(
                gt_page_doc, pred_page_doc, is_strict=self._strict
            )
            (
                ent_tp_b,
                ent_fp_b,
                ent_fn_b,
                ent_prec_b,
                ent_rec_b,
                ent_f1_b,
            ) = evaluate_entity_extraction_with_bbox(
                gt_page_doc, pred_page_doc, is_strict=self._strict
            )

            (
                rel_tp,
                rel_fp,
                rel_fn,
                rel_prec,
                rel_rec,
                rel_f1,
            ) = evaluate_relation_extraction(
                gt_page_doc, pred_page_doc, is_strict=self._strict
            )
            (
                rel_tp_b,
                rel_fp_b,
                rel_fn_b,
                rel_prec_b,
                rel_rec_b,
                rel_f1_b,
            ) = evaluate_relation_extraction_with_bbox(
                gt_page_doc, pred_page_doc, is_strict=self._strict
            )

            gt_entity_count = count_entities(gt_page_doc)
            pred_entity_count = count_entities(pred_page_doc)
            entity_diff = abs(gt_entity_count - pred_entity_count)
            max_entity_count = max(gt_entity_count, pred_entity_count)
            entity_diff_normalized = (
                (entity_diff / max_entity_count) if max_entity_count > 0 else 0.0
            )

            gt_link_count = count_links(gt_page_doc)
            pred_link_count = count_links(pred_page_doc)
            link_diff = abs(gt_link_count - pred_link_count)
            max_link_count = max(gt_link_count, pred_link_count)
            link_diff_normalized = (
                (link_diff / max_link_count) if max_link_count > 0 else 0.0
            )

            results.append(
                KeyValuePageEvaluation(
                    doc_id=doc_id,
                    page_no=ref.page_no,
                    image_name=ref.image_name,
                    entity_tp=ent_tp,
                    entity_fp=ent_fp,
                    entity_fn=ent_fn,
                    entity_precision=ent_prec,
                    entity_recall=ent_rec,
                    entity_f1=ent_f1,
                    entity_tp_bbox=ent_tp_b,
                    entity_fp_bbox=ent_fp_b,
                    entity_fn_bbox=ent_fn_b,
                    entity_precision_bbox=ent_prec_b,
                    entity_recall_bbox=ent_rec_b,
                    entity_f1_bbox=ent_f1_b,
                    relation_tp=rel_tp,
                    relation_fp=rel_fp,
                    relation_fn=rel_fn,
                    relation_precision=rel_prec,
                    relation_recall=rel_rec,
                    relation_f1=rel_f1,
                    relation_tp_bbox=rel_tp_b,
                    relation_fp_bbox=rel_fp_b,
                    relation_fn_bbox=rel_fn_b,
                    relation_precision_bbox=rel_prec_b,
                    relation_recall_bbox=rel_rec_b,
                    relation_f1_bbox=rel_f1_b,
                    num_entity_diff=entity_diff,
                    num_link_diff=link_diff,
                    num_entity_diff_normalized=entity_diff_normalized,
                    num_link_diff_normalized=link_diff_normalized,
                )
            )

        return results

    @staticmethod
    def _filter_doc_to_page(doc: DoclingDocument, page_no: int) -> DoclingDocument:
        if doc.pages and page_no in doc.pages:
            page_doc = doc.filter(page_nrs={page_no})
            page_doc.name = doc.name
            return page_doc

        return DoclingDocument(name=doc.name)
