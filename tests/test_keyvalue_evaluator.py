from pathlib import Path

import pytest

from docling_eval.datamodels.types import PredictionFormats
from docling_eval.evaluators.keyvalue_evaluator import KeyValueEvaluator


@pytest.mark.dependency(
    depends=["tests/test_dataset_builder.py::test_run_funsd"],
    scope="session",
)
def test_keyvalue_evaluator():
    r""" """
    test_dataset_dir = Path("scratch/FUNSD/gt_dataset")

    # Default evaluator
    eval1 = KeyValueEvaluator()
    v1 = eval1(test_dataset_dir)
    assert v1 is not None
    assert v1.evaluated_samples >= 0  # Should return a DatasetKeyValueEvaluation

    # Specify valid prediction_sources (DOCLING_DOCUMENT only)
    eval2 = KeyValueEvaluator(prediction_sources=[PredictionFormats.DOCLING_DOCUMENT])
    v2 = eval2(test_dataset_dir)
    assert v2 is not None

    # Specify invalid prediction_sources
    is_exception = False
    try:
        eval3 = KeyValueEvaluator(prediction_sources=[PredictionFormats.JSON])
        eval3(test_dataset_dir)
    except RuntimeError as ex:
        is_exception = True
    assert is_exception


@pytest.mark.dependency(
    depends=["tests/test_dataset_builder.py::test_run_funsd"],
    scope="session",
)
def test_failed_conversions():
    r"""Test if the evaluator skips invalid data samples"""
    test_dataset_dir = Path("scratch/FUNSD/gt_dataset")

    evaluator = KeyValueEvaluator()
    # Force to only accept failed conversions (should result in no valid evaluations)
    from docling.datamodel.base_models import ConversionStatus

    evaluator._accepted_status = [ConversionStatus.FAILURE]

    v1 = evaluator(test_dataset_dir)
    assert v1 is not None
    assert len(v1.evaluations) == 0
