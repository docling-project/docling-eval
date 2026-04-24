from pathlib import Path

import pytest
from docling.datamodel.base_models import ConversionStatus
from docling_core.types import DoclingDocument
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    PageItem,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)

from docling_eval.datamodels.types import PredictionFormats
from docling_eval.evaluators.layout_evaluator import LayoutEvaluator


def test_layout_item_counts_are_page_level() -> None:
    document = DoclingDocument(name="multi-page")
    document.pages[1] = PageItem(page_no=1, size=Size(width=100, height=100))
    document.pages[2] = PageItem(page_no=2, size=Size(width=100, height=100))

    bbox = BoundingBox(l=0, t=0, r=10, b=10, coord_origin=CoordOrigin.TOPLEFT)
    page_1_prov = ProvenanceItem(page_no=1, bbox=bbox, charspan=(0, 4))
    page_2_prov = ProvenanceItem(page_no=2, bbox=bbox, charspan=(0, 4))

    document.add_text(
        label=DocItemLabel.TEXT,
        text="text",
        orig="text",
        prov=page_1_prov,
    )
    document.add_table(
        data=TableData(
            table_cells=[
                TableCell(
                    start_row_offset_idx=0,
                    end_row_offset_idx=0,
                    start_col_offset_idx=0,
                    end_col_offset_idx=0,
                    text="cell",
                )
            ]
        ),
        caption=None,
        prov=page_2_prov,
    )
    document.add_picture(prov=page_2_prov)

    counts_by_page = LayoutEvaluator()._count_layout_items_by_page(document)

    assert counts_by_page[1].element_count == 1
    assert counts_by_page[1].table_count == 0
    assert counts_by_page[1].picture_count == 0
    assert counts_by_page[2].element_count == 2
    assert counts_by_page[2].table_count == 1
    assert counts_by_page[2].picture_count == 1


@pytest.mark.dependency(
    depends=["tests/test_dataset_builder.py::test_run_dpbench_e2e"],
    scope="session",
)
def test_layout_evaluator():
    r""" """
    test_dataset_dir = Path("scratch/DPBench/eval_dataset_e2e")

    # Default evaluator
    eval1 = LayoutEvaluator()
    v1 = eval1(test_dataset_dir)
    assert v1 is not None

    # Specify order in prediction_sources
    eval2 = LayoutEvaluator(prediction_sources=[PredictionFormats.JSON])
    v2 = eval2(test_dataset_dir)
    assert v2 is not None

    # Specify invalid order in prediction_sources
    is_exception = False
    try:
        eval3 = LayoutEvaluator(prediction_sources=[PredictionFormats.MARKDOWN])
        eval3(test_dataset_dir)
    except RuntimeError as ex:
        is_exception = True
    assert is_exception


@pytest.mark.dependency(
    depends=["tests/test_dataset_builder.py::test_run_dpbench_e2e"],
    scope="session",
)
def test_failed_conversions():
    r"""Test if the evaluator skips invalid data samples"""
    test_dataset_dir = Path("scratch/DPBench/eval_dataset_e2e")

    # Create a manipulated evaluator to accept only failed data samples
    evaluator = LayoutEvaluator()
    evaluator._accepted_status = [ConversionStatus.FAILURE]

    v1 = evaluator(test_dataset_dir)
    assert v1 is not None
    assert len(v1.evaluations_per_class) == 0
    assert len(v1.evaluations_per_image) == 0


@pytest.mark.dependency(
    depends=["tests/test_dataset_builder.py::test_run_dpbench_e2e"],
    scope="session",
)
def test_layout_evaluator_external_predictions():
    r"""Testing the evaluator with external predictions"""
    from docling_eval.utils.external_docling_document_loader import (
        ExternalDoclingDocumentLoader,
    )

    eval = LayoutEvaluator()
    gt_path = Path("scratch/DPBench/gt_dataset")

    preds_path = [
        Path("scratch/DPBench/predicted_documents/json"),
        Path("scratch/DPBench/predicted_documents/doctag"),
        Path("scratch/DPBench/predicted_documents/yaml"),
    ]
    for pred_path in preds_path:
        loader = ExternalDoclingDocumentLoader(pred_path)
        v = eval(gt_path, external_document_loader=loader)
        assert v is not None


if __name__ == "__main__":
    #     # test_layout_evaluator()
    #     test_failed_conversions()
    test_layout_evaluator_external_predictions()
