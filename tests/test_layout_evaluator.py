from pathlib import Path

from docling_eval.datamodels.types import PredictionFormats
from docling_eval.evaluators.layout_evaluator import LayoutEvaluator
from docling_eval.evaluators.markdown_text_evaluator import MarkdownTextEvaluator


def test_layout_evaluator():
    r""" """
    # TODO: Ensure that the test dataset exists
    test_dataset_dir = Path("scratch/DocLayNetV1/eval_dataset")

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


if __name__ == "__main__":
    test_layout_evaluator()
