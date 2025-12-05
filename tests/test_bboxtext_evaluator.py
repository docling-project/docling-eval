from pathlib import Path

import pytest

from docling_eval.datamodels.types import PredictionFormats
from docling_eval.evaluators.bbox_text_evaluator import BboxTextEvaluator


@pytest.mark.dependency(
    depends=["tests/test_dataset_builder.py::test_run_dpbench_e2e"],
    scope="session",
)
def test_bboxtext_evaluator():
    r""" """
    test_dataset_dir = Path("scratch/DPBench/eval_dataset_e2e")

    # Default evaluator
    eval1 = BboxTextEvaluator()
    v1 = eval1(test_dataset_dir)
    assert v1 is not None

    # Specify invalid order in prediction_sources
    is_exception = False
    try:
        eval3 = BboxTextEvaluator(prediction_sources=[PredictionFormats.JSON])
        eval3(test_dataset_dir)
    except RuntimeError as ex:
        is_exception = True
    assert is_exception


@pytest.mark.dependency(
    depends=["tests/test_dataset_builder.py::test_run_dpbench_e2e"],
    scope="session",
)
def test_bboxtext_evaluator_external_predictions():
    r"""Testing the evaluator with external predictions"""
    eval = BboxTextEvaluator()
    gt_path = Path("scratch/DPBench/gt_dataset")
    preds_path = Path("scratch/DPBench/predicted_documents/json")

    v = eval(gt_path, external_predictions_path=preds_path)
    assert v is not None


if __name__ == "__main__":
    # test_bboxtext_evaluator()
    test_bboxtext_evaluator_external_predictions()
