from pathlib import Path

import pytest

from docling_eval.evaluators.ocr_evaluator import OCREvaluator


@pytest.mark.dependency(
    depends=["tests/test_dataset_builder.py::test_run_dpbench_e2e"],
    scope="session",
)
def test_ocr_evaluator():
    r""" """
    test_dataset_dir = Path("scratch/DPBench/eval_dataset_e2e")

    # Default evaluator
    eval1 = OCREvaluator()
    v1 = eval1(test_dataset_dir)
    assert v1 is not None
