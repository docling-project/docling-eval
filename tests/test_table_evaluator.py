from pathlib import Path
from typing import List

import pytest
from docling_core.types.doc.document import DoclingDocument, TableCell, TableData

from docling_eval.datamodels.types import BenchMarkColumns, PredictionFormats
from docling_eval.evaluators.table_evaluator import TableEvaluation, TableEvaluator


def test_evaluate_tables():
    r"""
    Testing the internal _evaluate_tables_in_documents
    """
    data_table_cells = []
    num_cols = 6
    num_rows = 5
    # ======================================
    data_table_cells.append(
        TableCell(
            text="AB",
            row_span=1,
            col_span=2,
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=0,
            end_col_offset_idx=3,
            col_header=False,
            row_header=True,
        )
    )

    data_table_cells.append(
        TableCell(
            text="C",
            row_span=1,
            col_span=1,
            start_row_offset_idx=0,
            end_row_offset_idx=1,
            start_col_offset_idx=2,
            end_col_offset_idx=3,
            col_header=False,
            row_header=True,
        )
    )

    # ======================================
    data_table_cells.append(
        TableCell(
            text="1",
            row_span=1,
            col_span=1,
            start_row_offset_idx=1,
            end_row_offset_idx=2,
            start_col_offset_idx=0,
            end_col_offset_idx=1,
            col_header=False,
            row_header=True,
        )
    )

    data_table_cells.append(
        TableCell(
            text="2",
            row_span=1,
            col_span=1,
            start_row_offset_idx=1,
            end_row_offset_idx=2,
            start_col_offset_idx=1,
            end_col_offset_idx=2,
            col_header=False,
            row_header=False,
        )
    )

    data_table_cells.append(
        TableCell(
            text="3",
            row_span=1,
            col_span=1,
            start_row_offset_idx=1,
            end_row_offset_idx=2,
            start_col_offset_idx=2,
            end_col_offset_idx=3,
            col_header=False,
            row_header=False,
        )
    )

    # ======================================
    data_table_cells.append(
        TableCell(
            text="2D",
            row_span=2,
            col_span=3,
            start_row_offset_idx=0,
            end_row_offset_idx=2,
            start_col_offset_idx=3,
            end_col_offset_idx=6,
            col_header=True,
            row_header=False,
        )
    )

    # ======================================
    data_table_cells.append(
        TableCell(
            text="4",
            row_span=2,
            col_span=1,
            start_row_offset_idx=2,
            end_row_offset_idx=4,
            start_col_offset_idx=0,
            end_col_offset_idx=1,
            col_header=False,
            row_header=True,
        )
    )

    data_table_cells.append(
        TableCell(
            text="5",
            row_span=1,
            col_span=1,
            start_row_offset_idx=2,
            end_row_offset_idx=3,
            start_col_offset_idx=1,
            end_col_offset_idx=2,
            col_header=False,
            row_header=False,
        )
    )

    data_table_cells.append(
        TableCell(
            text="6",
            row_span=1,
            col_span=1,
            start_row_offset_idx=2,
            end_row_offset_idx=3,
            start_col_offset_idx=2,
            end_col_offset_idx=3,
            col_header=False,
            row_header=False,
        )
    )

    data_table_cells.append(
        TableCell(
            text="next 2 cells empty",
            row_span=1,
            col_span=1,
            start_row_offset_idx=2,
            end_row_offset_idx=3,
            start_col_offset_idx=3,
            end_col_offset_idx=4,
            col_header=False,
            row_header=False,
        )
    )

    data_table_cells.append(
        TableCell(
            text="",
            row_span=1,
            col_span=1,
            start_row_offset_idx=2,
            end_row_offset_idx=3,
            start_col_offset_idx=4,
            end_col_offset_idx=5,
            col_header=False,
            row_header=False,
        )
    )

    data_table_cells.append(
        TableCell(
            text="",
            row_span=1,
            col_span=1,
            start_row_offset_idx=2,
            end_row_offset_idx=3,
            start_col_offset_idx=5,
            end_col_offset_idx=6,
            col_header=False,
            row_header=False,
        )
    )

    # ======================================

    data_table_cells.append(
        TableCell(
            text="Q",
            row_span=1,
            col_span=1,
            start_row_offset_idx=3,
            end_row_offset_idx=4,
            start_col_offset_idx=1,
            end_col_offset_idx=2,
            col_header=False,
            row_header=False,
        )
    )

    data_table_cells.append(
        TableCell(
            text="W",
            row_span=1,
            col_span=1,
            start_row_offset_idx=3,
            end_row_offset_idx=4,
            start_col_offset_idx=2,
            end_col_offset_idx=3,
            col_header=False,
            row_header=False,
        )
    )

    data_table_cells.append(
        TableCell(
            text="E",
            row_span=1,
            col_span=1,
            start_row_offset_idx=3,
            end_row_offset_idx=4,
            start_col_offset_idx=3,
            end_col_offset_idx=4,
            col_header=False,
            row_header=False,
        )
    )

    data_table_cells.append(
        TableCell(
            text="R",
            row_span=1,
            col_span=1,
            start_row_offset_idx=3,
            end_row_offset_idx=4,
            start_col_offset_idx=4,
            end_col_offset_idx=5,
            col_header=False,
            row_header=False,
        )
    )

    data_table_cells.append(
        TableCell(
            text="T",
            row_span=1,
            col_span=1,
            start_row_offset_idx=3,
            end_row_offset_idx=4,
            start_col_offset_idx=5,
            end_col_offset_idx=6,
            col_header=False,
            row_header=False,
        )
    )

    # ======================================
    data_table_cells.append(
        TableCell(
            text="Section header",
            row_span=1,
            col_span=6,
            start_row_offset_idx=4,
            end_row_offset_idx=5,
            start_col_offset_idx=0,
            end_col_offset_idx=6,
            col_header=False,
            row_header=False,
            row_section=True,
        )
    )

    # ======================================
    doc = DoclingDocument(name="test_eval")
    data = TableData(num_rows=num_rows, num_cols=num_cols, table_cells=data_table_cells)
    doc.add_table(data=data)

    table_evaluator = TableEvaluator()

    # Evaluate equal tables
    evaluations: List[TableEvaluation] = table_evaluator._evaluate_tables_in_documents(
        doc_id="test", true_doc=doc, pred_doc=doc
    )
    assert len(evaluations) == 1
    evaluation = evaluations[0]
    assert evaluation.TEDS == 1.0
    assert evaluation.is_complex


@pytest.mark.dependency(
    depends=["tests/test_dataset_builder.py::test_run_dpbench_tables"],
    scope="session",
)
def test_table_evaluator():
    r""" """
    test_dataset_dir = Path("scratch/DPBench/eval_dataset_tables")

    # Default evaluator
    eval1 = TableEvaluator()
    v1 = eval1(test_dataset_dir)
    assert v1 is not None

    # # Specify order in prediction_sources
    # eval2 = TableEvaluator(prediction_sources=[PredictionFormats.DOCTAGS])
    # v2 = eval2(test_dataset_dir)
    # assert v2 is not None

    # Specify invalid order in prediction_sources
    is_exception = False
    try:
        eval3 = TableEvaluator(prediction_sources=[PredictionFormats.MARKDOWN])
        eval3(test_dataset_dir)
    except RuntimeError as ex:
        is_exception = True
    assert is_exception


@pytest.mark.dependency(
    depends=["tests/test_dataset_builder.py::test_run_dpbench_e2e"],
    scope="session",
)
def test_table_evaluator_external_predictions():
    r"""Testing the evaluator with external predictions"""
    from docling_eval.utils.external_docling_document_loader import (
        ExternalDoclingDocumentLoader,
    )

    eval = TableEvaluator()
    gt_path = Path("scratch/DPBench/gt_dataset")
    preds_path = Path("scratch/DPBench/predicted_documents/json")

    loader = ExternalDoclingDocumentLoader(preds_path)
    v = eval(gt_path, external_document_loader=loader)
    assert v is not None


if __name__ == "__main__":
    test_table_evaluator_external_predictions()
