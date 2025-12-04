from pathlib import Path

import pytest

from docling_eval.datamodels.types import PredictionFormats
from docling_eval.evaluators.markdown_text_evaluator import MarkdownTextEvaluator


@pytest.mark.dependency(
    depends=["tests/test_dataset_builder.py::test_run_dpbench_e2e"],
    scope="session",
)
def test_markdown_text_evaluator():
    r""" """
    test_dataset_dir = Path("scratch/DPBench/eval_dataset_e2e")

    # Default evaluator
    eval1 = MarkdownTextEvaluator()
    v1 = eval1(test_dataset_dir)
    assert v1 is not None

    # Specify order in prediction_sources
    eval2 = MarkdownTextEvaluator(prediction_sources=[PredictionFormats.MARKDOWN])
    v2 = eval2(test_dataset_dir)
    assert v2 is not None

    # Specify invalid order in prediction_sources
    is_exception = False
    try:
        eval3 = MarkdownTextEvaluator(prediction_sources=[PredictionFormats.JSON])
        eval3(test_dataset_dir)
    except RuntimeError as ex:
        is_exception = True
    assert is_exception


def test_markdown_text_evaluator_external_predictions():
    r"""Testing the evaluator with external predictions"""
    eval = MarkdownTextEvaluator()
    gt_path = Path("scratch/DPBench/gt_dataset")
    preds_path = Path("scratch/DPBench/predicted_documents/json")

    v = eval(gt_path, external_predictions_path=preds_path)
    assert v is not None


if __name__ == "__main__":
    # test_markdown_text_evaluator()
    test_markdown_text_evaluator_external_predictions()
