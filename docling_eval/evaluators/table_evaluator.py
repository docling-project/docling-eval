import glob
import logging
import random
from concurrent.futures import Executor, Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from datasets import Dataset, load_dataset
from docling_core.types.doc.document import DoclingDocument, TableItem
from docling_core.types.doc.labels import DocItemLabel
from lxml import html
from pydantic import BaseModel
from tqdm import tqdm  # type: ignore

from docling_eval.datamodels.dataset_record import DatasetRecordWithPrediction
from docling_eval.datamodels.types import BenchMarkColumns, PredictionFormats
from docling_eval.evaluators.base_evaluator import (
    BaseEvaluator,
    DatasetEvaluation,
    EvaluationRejectionType,
    UnitEvaluation,
    docling_document_from_doctags,
)
from docling_eval.evaluators.stats import DatasetStatistics, compute_stats
from docling_eval.evaluators.table.teds import TEDScorer
from docling_eval.utils.external_docling_document_loader import (
    ExternalDoclingDocumentLoader,
)

_log = logging.getLogger(__name__)


class TableEvaluation(UnitEvaluation):
    filename: str = "<unknown>"
    table_id: int = -1
    TEDS: float
    is_complex: bool = False
    structure_only_evaluation: bool = False

    true_ncols: int = -1
    pred_ncols: int = -1

    true_nrows: int = -1
    pred_nrows: int = -1


class DatasetTableEvaluation(DatasetEvaluation):
    evaluations: List[TableEvaluation]
    table_structure_evaluations: List[TableEvaluation]

    TEDS: DatasetStatistics
    TEDS_struct: DatasetStatistics
    TEDS_simple: DatasetStatistics
    TEDS_complex: DatasetStatistics

    def save_histogram_delta_row_col(self, figname: Path):
        delta_row = {i: 0 for i in range(-10, 11)}
        delta_col = {i: 0 for i in range(-10, 11)}

        for evaluation in self.evaluations:
            if evaluation.true_nrows - evaluation.pred_nrows in delta_row:
                delta_row[evaluation.true_nrows - evaluation.pred_nrows] += 1

            if evaluation.true_ncols - evaluation.pred_ncols in delta_col:
                delta_col[evaluation.true_ncols - evaluation.pred_ncols] += 1

        x_row, y_row = [], []
        for k, v in delta_row.items():
            x_row.append(k)
            if v == 0:
                y_row.append(1.0e-6)
            else:
                y_row.append(v / float(len(self.evaluations)))

        x_col, y_col = [], []
        for k, v in delta_col.items():
            x_col.append(k)
            if v == 0:
                y_col.append(1.0e-6)
            else:
                y_col.append(v / float(len(self.evaluations)))

        fignum = int(1000 * random.random())
        plt.figure(fignum)

        plt.semilogy(x_row, y_row, "k.-", label="rows_{true} - rows_{pred}")
        plt.semilogy(x_col, y_col, "r.-", label="cols_{true} - cols_{pred}")

        plt.xlabel("delta")
        plt.ylabel("%")
        plt.legend(loc="upper right")

        _log.info(f"saving figure to {figname}")
        plt.savefig(figname)


def is_complex_table(table: TableItem) -> bool:
    r"""
    Implement the logic to check if table is complex
    """
    for cell in table.data.table_cells:
        if cell.row_span > 1 or cell.col_span > 1:
            return True
    return False


def evaluate_tables(
    teds_scorer: TEDScorer,
    stopwords: list[str],
    doc_id: str,
    table_id: int,
    true_html: str,
    true_num_rows: int,
    true_num_cols: int,
    pred_html: str,
    pred_num_rows: int,
    pred_num_cols: int,
    is_complex: bool,
    structure_only: bool,
) -> Optional[TableEvaluation]:
    r"""
    Execution function
    Receive 2 tables as html-formatted string. Compute the TEDS score

    Return
    ------
    teds: float
    is_complex: bool
    structure_only: bool
    """
    try:
        for stopword in stopwords:
            pred_html = pred_html.replace(stopword, "")
        for stopword in stopwords:
            true_html = true_html.replace(stopword, "")

        pred_html_obj = html.fromstring(pred_html)
        true_html_obj = html.fromstring(true_html)

        teds = teds_scorer(
            gt_table=true_html_obj,
            pred_table=pred_html_obj,
            structure_only=structure_only,
        )
        teds = round(teds, 3)

        # Prepare output
        table_evaluation = TableEvaluation(
            TEDS=teds,
            is_complex=is_complex,
            filename=doc_id,
            table_id=table_id,
            true_ncols=true_num_cols,
            pred_ncols=pred_num_cols,
            true_nrows=true_num_rows,
            pred_nrows=pred_num_rows,
            structure_only_evaluation=structure_only,
        )
        return table_evaluation
    except Exception:
        _log.error("Cannot evaluate doc_id: %s table: %d ", doc_id, table_id)
        return None


class TableEvaluator(BaseEvaluator):
    r"""
    Evaluate table predictions from HF dataset with the columns:
    """

    def __init__(
        self,
        intermediate_evaluations_path: Optional[Path] = None,
        structure_only: bool = False,
        prediction_sources: List[PredictionFormats] = [],
        concurrency: int = 4,
    ):
        supported_prediction_formats: List[PredictionFormats] = [
            PredictionFormats.DOCLING_DOCUMENT,
            PredictionFormats.DOCTAGS,
        ]
        if not prediction_sources:
            prediction_sources = supported_prediction_formats
        super().__init__(
            concurrency=concurrency,
            intermediate_evaluations_path=intermediate_evaluations_path,
            prediction_sources=prediction_sources,
            supported_prediction_formats=supported_prediction_formats,
        )

        self._structure_only = structure_only
        self._teds_scorer: TEDScorer = TEDScorer()
        self._stopwords = ["<i>", "</i>", "<b>", "</b>", "<u>", "</u>"]

    def __call__(
        self,
        ds_path: Path,
        split: str = "test",
        ext_docdoc_loader: Optional[ExternalDoclingDocumentLoader] = None,
    ) -> DatasetTableEvaluation:
        r"""
        Load a dataset in HF format. Expected columns with DoclingDocuments
        "GTDoclingDocument"
        "PredictionDoclingDocument"
        """
        self._begin_message(ds_path, split, ext_docdoc_loader)

        # Load the dataset
        split_path = str(ds_path / split / "*.parquet")
        split_files = glob.glob(split_path)
        _log.info("Files: %s", split_files)
        ds = load_dataset("parquet", data_files={split: split_files})
        _log.info("Overview of dataset: %s", ds)

        # Select the split
        ds_selection: Dataset = ds[split]

        table_evaluations = []
        table_struct_evaluations = []
        rejected_samples: Dict[EvaluationRejectionType, int] = {
            EvaluationRejectionType.MISSING_PREDICTION: 0,
            EvaluationRejectionType.EVALUATION_ERROR: 0,
            EvaluationRejectionType.MISMATHCED_DOCUMENT: 0,
        }

        with ProcessPoolExecutor(max_workers=self._concurrency) as executor:
            futures: list[Future] = []
            table_futures: list[Future]
            table_rejection: Optional[EvaluationRejectionType]

            # Submit pages for execution
            _log.info("Submitting the tables for evaluation...")
            for i, data in enumerate(ds_selection):
                data_record = DatasetRecordWithPrediction.model_validate(data)
                doc_id = data_record.doc_id
                gt_doc = data_record.ground_truth_doc
                pred_doc = self._get_pred_doc(data_record, ext_docdoc_loader)
                if not pred_doc:
                    _log.error("There is no prediction for doc_id=%s", doc_id)
                    rejected_samples[EvaluationRejectionType.MISSING_PREDICTION] += 1
                    continue

                if not self._structure_only:
                    # Evaluate the tables with structure + content
                    table_futures, table_rejection = self._evaluate_tables_in_documents(
                        executor,
                        doc_id=doc_id,
                        true_doc=gt_doc,
                        pred_doc=pred_doc,
                        structure_only=False,
                    )
                    if table_rejection != None:
                        rejected_samples[table_rejection] += 1
                        continue
                    futures.extend(table_futures)

                # Always evaluate the tables with structure
                table_futures, table_rejection = self._evaluate_tables_in_documents(
                    executor,
                    doc_id=doc_id,
                    true_doc=gt_doc,
                    pred_doc=pred_doc,
                    structure_only=True,
                )
                if table_rejection != None:
                    rejected_samples[table_rejection] += 1
                    continue
                futures.extend(table_futures)

            # Collect the futures
            _log.info("Collecting the tables for evaluations...")
            for future in tqdm(
                as_completed(futures),
                desc="Table evaluations",
                ncols=120,
                total=len(futures),
            ):
                table_evaluation: Optional[TableEvaluation] = future.result()
                if table_evaluation is None:
                    rejected_samples[EvaluationRejectionType.EVALUATION_ERROR] += 1
                    continue

                table_id: int = table_evaluation.table_id
                doc_id = table_evaluation.filename

                if not table_evaluation.structure_only_evaluation:
                    table_evaluations.append(table_evaluation)
                    if self._intermediate_evaluations_path:
                        self.save_intermediate_evaluations(
                            "TEDs_struct_content", table_id, doc_id, [table_evaluation]
                        )

                table_struct_evaluations.append(table_evaluation)
                if self._intermediate_evaluations_path:
                    self.save_intermediate_evaluations(
                        "TEDs_struct", table_id, doc_id, [table_evaluation]
                    )

        # Summary log
        _log.info(
            (
                "Finish. Missing prediction documents: %d."
                + " Documents with mismatch in number of tables between GT/predictions: %d."
                + " Skipped tables due to evaluation errors: %d"
            ),
            rejected_samples[EvaluationRejectionType.MISSING_PREDICTION],
            rejected_samples[EvaluationRejectionType.MISMATHCED_DOCUMENT],
            rejected_samples[EvaluationRejectionType.EVALUATION_ERROR],
        )

        # Compute TED statistics for the entire dataset
        teds_simple = []
        teds_complex = []
        teds_all = []
        if not self._structure_only:
            for te in table_evaluations:
                teds_all.append(te.TEDS)

                if te.is_complex:
                    teds_complex.append(te.TEDS)
                else:
                    teds_simple.append(te.TEDS)

        teds_struct = []
        for te in table_struct_evaluations:
            teds_struct.append(te.TEDS)

        dataset_evaluation = DatasetTableEvaluation(
            evaluated_samples=len(table_evaluations),
            rejected_samples=rejected_samples,
            evaluations=table_evaluations,
            table_structure_evaluations=table_struct_evaluations,
            TEDS=compute_stats(teds_all),
            TEDS_struct=compute_stats(teds_struct),
            TEDS_simple=compute_stats(teds_simple),
            TEDS_complex=compute_stats(teds_complex),
        )
        return dataset_evaluation

    def _evaluate_tables_in_documents(
        self,
        executor: Executor,
        doc_id: str,
        true_doc: DoclingDocument,
        pred_doc: DoclingDocument,
        structure_only: bool = False,
    ) -> tuple[list[Future], Optional[EvaluationRejectionType]]:
        r"""
        1. Extract the tables from true/pred document
        2. Reject if the number of tables differs across true/pred
        3. Export table as html-formatted string.
        4. Submit the tables for evaluation
        5. Return futures (one per table)

        Return
        ------

        """
        futures: list[Future] = []
        true_tables: list[TableItem] = true_doc.tables
        pred_tables: list[TableItem] = pred_doc.tables
        true_tables_len = len(true_tables)
        pred_tables_len = len(pred_tables)
        _log.debug(
            "#-true-tables: %s, #-pred-tables: %s", true_tables_len, pred_tables_len
        )
        # Reject the document is there is a mismatch in the number of tables between true/pred doc
        if true_tables_len != pred_tables_len:
            _log.error(
                "Mismatched number of tables between GT and predictions: [%d, %d]. Skipping doc: %s",
                true_tables_len,
                pred_tables_len,
                doc_id,
            )
            return futures, EvaluationRejectionType.MISMATHCED_DOCUMENT

        for table_id in range(len(true_tables)):  # , len(pred_tables)):
            # Avoid items of type DocItemLabel.DOCUMENT_INDEX
            if true_tables[table_id].label != DocItemLabel.TABLE:
                _log.warning(f"Skipping table with label {true_tables[table_id].label}")
                continue

            try:
                true_table = true_tables[table_id]
                pred_table = pred_tables[table_id]

                is_complex = is_complex_table(true_table)

                true_html: str = true_table.export_to_html(true_doc)
                pred_html: str = pred_table.export_to_html(pred_doc)

                # Submit table for evaluation
                futures.append(
                    executor.submit(
                        evaluate_tables,
                        self._teds_scorer,
                        self._stopwords,
                        doc_id,
                        table_id,
                        true_html,
                        true_table.data.num_rows,
                        true_table.data.num_cols,
                        pred_html,
                        pred_table.data.num_rows,
                        pred_table.data.num_cols,
                        is_complex,
                        structure_only,
                    )
                )
            except Exception:
                _log.error(
                    f"Table {table_id} from document {doc_id} could not be compared!"
                )
        return futures, None

    def _get_pred_doc(
        self,
        data_record: DatasetRecordWithPrediction,
        ext_docdoc_loader: Optional[ExternalDoclingDocumentLoader] = None,
    ) -> Optional[DoclingDocument]:
        r"""
        Get the predicted DoclingDocument
        """
        pred_doc = None
        if ext_docdoc_loader is not None:
            pred_doc = ext_docdoc_loader.get(data_record)
            return pred_doc

        for prediction_format in self._prediction_sources:
            if prediction_format == PredictionFormats.DOCLING_DOCUMENT:
                pred_doc = data_record.predicted_doc
            elif prediction_format == PredictionFormats.DOCTAGS:
                pred_doc = docling_document_from_doctags(data_record)
            if pred_doc is not None:
                break

        return pred_doc
