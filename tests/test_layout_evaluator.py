from pathlib import Path

import pytest
from docling.datamodel.base_models import ConversionStatus

from docling_eval.datamodels.types import PredictionFormats
from docling_eval.evaluators.layout_evaluator import LayoutEvaluator
from docling_eval.evaluators.markdown_text_evaluator import MarkdownTextEvaluator


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


# if __name__ == "__main__":
#     # test_layout_evaluator()
#     test_failed_conversions()
