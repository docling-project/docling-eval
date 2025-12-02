import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from apted import APTED, PerEditOperationConfig
from datasets import load_dataset
from docling_core.types.doc.document import DoclingDocument
from pydantic import Field
from tqdm import tqdm  # type: ignore

from docling_eval.datamodels.cvat_page_mapping import CvatPageMapping, PageRef
from docling_eval.datamodels.dataset_record import DatasetRecordWithPrediction
from docling_eval.datamodels.types import PredictionFormats  # type: ignore
from docling_eval.evaluators.base_evaluator import (
    BaseEvaluator,
    DatasetEvaluation,
    EvaluationRejectionType,
    UnitEvaluation,
)
from docling_eval.evaluators.stats import DatasetStatistics, compute_stats

_log = logging.getLogger(__name__)


class DocStructureEvaluation(UnitEvaluation):
    doc_id: str

    edit_distance: float


class DocStructurePageEvaluation(UnitEvaluation):
    doc_id: str
    page_no: int
    image_name: str

    edit_distance: float


class DatasetDocStructureEvaluation(DatasetEvaluation):
    evaluations: List[DocStructureEvaluation]

    edit_distance_stats: DatasetStatistics

    evaluations_per_page: List[DocStructurePageEvaluation] = Field(default_factory=list)
    page_edit_distance_stats: DatasetStatistics


class DocStructureEvaluator(BaseEvaluator):
    def __init__(
        self,
        intermediate_evaluations_path: Optional[Path] = None,
        prediction_sources: List[PredictionFormats] = [],
        page_mapping_path: Optional[Path] = None,
    ):
        r""" """
        supported_prediction_formats: List[PredictionFormats] = [
            PredictionFormats.DOCLING_DOCUMENT,
        ]
        if not prediction_sources:
            prediction_sources = supported_prediction_formats
        super().__init__(
            intermediate_evaluations_path=intermediate_evaluations_path,
            prediction_sources=prediction_sources,
            supported_prediction_formats=supported_prediction_formats,
        )

        self._page_mapping_path = page_mapping_path
        self._page_mapping: Optional[CvatPageMapping] = None

    def __call__(
        self,
        ds_path: Path,
        split: str = "test",
    ) -> DatasetDocStructureEvaluation:
        r"""
        Parameters
        ----------
        ds_path: Path to load the parquet files of the dataset
        split: Split of the dataset to load
        """
        parquet_files = str(ds_path / split / "*.parquet")
        ds = load_dataset("parquet", data_files={split: parquet_files})
        _log.info(f"Overview of the dataset: {ds}")
        if ds is not None:
            ds_selection = ds[split]

        evaluations: list[DocStructureEvaluation] = []
        rejected_samples: Dict[EvaluationRejectionType, int] = {
            EvaluationRejectionType.INVALID_CONVERSION_STATUS: 0,
            EvaluationRejectionType.MISSING_PREDICTION: 0,
        }

        # Metrics per page
        ds_metrics: dict[str, list[float]] = {
            "edit_distance": [],
        }

        page_evaluations: List[DocStructurePageEvaluation] = []
        page_edit_distances: List[float] = []

        for i, data in tqdm(
            enumerate(ds_selection),
            desc="Document structure evaluations",
            ncols=120,
            total=len(ds_selection),
        ):
            data_record = DatasetRecordWithPrediction.model_validate(data)
            doc_id = data_record.doc_id
            if data_record.status not in self._accepted_status:
                _log.error(
                    "Skipping record without successfull conversion status: %s", doc_id
                )
                rejected_samples[EvaluationRejectionType.INVALID_CONVERSION_STATUS] += 1
                continue

            true_doc = data_record.ground_truth_doc
            pred_doc = data_record.predicted_doc

            if pred_doc is None:
                _log.error("There is no prediction for doc_id=%s", doc_id)
                rejected_samples[EvaluationRejectionType.MISSING_PREDICTION] += 1
                continue

            struct_scores = self._compute_struct_scores(true_doc, pred_doc)

            # Collect metrics across pages
            for score_name, score in struct_scores.items():
                ds_metrics[score_name].append(score)

            struct_evaluation = DocStructureEvaluation(
                doc_id=doc_id,
                edit_distance=struct_scores["edit_distance"],
            )
            evaluations.append(struct_evaluation)

            page_mapping = self._get_page_mapping()
            if page_mapping is not None:
                page_evals, metrics = self._evaluate_per_page(
                    doc_id,
                    true_doc,
                    pred_doc,
                    page_mapping,
                )
                page_evaluations.extend(page_evals)
                page_edit_distances.extend(metrics)

            if self._intermediate_evaluations_path:
                self.save_intermediate_evaluations("DOCSTRUCT", i, doc_id, evaluations)

        ds_struct_evalutions = DatasetDocStructureEvaluation(
            evaluated_samples=len(evaluations),
            rejected_samples=rejected_samples,
            evaluations=evaluations,
            edit_distance_stats=compute_stats(ds_metrics["edit_distance"]),
            evaluations_per_page=page_evaluations,
            page_edit_distance_stats=(
                compute_stats(page_edit_distances)
                if page_edit_distances
                else compute_stats([])
            ),
        )
        return ds_struct_evalutions

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
        true_doc: DoclingDocument,
        pred_doc: DoclingDocument,
        page_mapping: CvatPageMapping,
    ) -> tuple[List[DocStructurePageEvaluation], List[float]]:
        evaluations: List[DocStructurePageEvaluation] = []
        distances: List[float] = []

        page_refs: List[PageRef] = page_mapping.pages_for_doc_name(doc_id)
        if not page_refs:
            return evaluations, distances

        for ref in page_refs:
            try:
                gt_page_doc = self._filter_doc_to_page(true_doc, ref.page_no)
            except Exception as exc:
                _log.error(
                    "Failed to filter ground-truth doc '%s' (page %s) for doc_id=%s: %s",
                    true_doc.name,
                    ref.page_no,
                    doc_id,
                    exc,
                )
                continue
                raise

            try:
                pred_page_doc = self._filter_doc_to_page(pred_doc, ref.page_no)
            except Exception as exc:
                _log.error(
                    "Failed to filter predicted doc '%s' (page %s) for doc_id=%s: %s",
                    pred_doc.name if pred_doc is not None else "<missing>",
                    ref.page_no,
                    doc_id,
                    exc,
                )
                continue
                raise

            scores = self._compute_struct_scores(gt_page_doc, pred_page_doc)
            distances.append(scores["edit_distance"])

            evaluations.append(
                DocStructurePageEvaluation(
                    doc_id=doc_id,
                    page_no=ref.page_no,
                    image_name=ref.image_name,
                    edit_distance=scores["edit_distance"],
                )
            )

        return evaluations, distances

    @staticmethod
    def _filter_doc_to_page(doc: DoclingDocument, page_no: int) -> DoclingDocument:
        if doc.pages and page_no in doc.pages:
            page_doc = doc.filter(page_nrs={page_no})
            page_doc.name = doc.name
            return page_doc

        empty_doc = DoclingDocument(name=doc.name)
        return empty_doc

    def _compute_struct_scores(
        self,
        true_doc: DoclingDocument,
        pred_doc: DoclingDocument,
    ) -> dict[str, float]:
        r"""
        Returns:
        --------
        dict with keys: ["edit_distance"]
        """

        class LabeledTreeWrapper:
            def __init__(self, node, source: str):
                self.node = node  # your actual node
                self.source = source  # "source" or "target"

            @property
            def label(self):
                return self.node.label

            @property
            def children(self):
                return [
                    LabeledTreeWrapper(child, self.source)
                    for child in self.node.children
                ]

        class StructConfig(PerEditOperationConfig):

            def __init__(
                self,
                del_cost: float,
                ins_cost: float,
                ren_cost: float,
                source_doc: DoclingDocument,
                target_doc: DoclingDocument,
            ):
                super().__init__(del_cost, ins_cost, ren_cost)
                self.source_doc = source_doc
                self.target_doc = target_doc

            def children(self, node):
                if node.source == "source":
                    return [
                        LabeledTreeWrapper(
                            child.node.resolve(self.source_doc), "source"
                        )
                        for child in node.children
                    ]
                else:
                    return [
                        LabeledTreeWrapper(
                            child.node.resolve(self.target_doc), "target"
                        )
                        for child in node.children
                    ]

            def rename(self, n1, n2):
                """
                Return the cost of changing n1 into n2.
                Here: 0 when labels match, 1 otherwise.
                """
                return 0 if n1.label == n2.label else 1

        apted = APTED(
            LabeledTreeWrapper(true_doc.body, "source"),
            LabeledTreeWrapper(pred_doc.body, "target"),
            StructConfig(
                del_cost=1.0,
                ins_cost=1.0,
                ren_cost=1.0,
                source_doc=true_doc,
                target_doc=pred_doc,
            ),
        )
        edit_dist = apted.compute_edit_distance()

        metrics: dict[str, float] = {
            "edit_distance": edit_dist,
        }
        return metrics
