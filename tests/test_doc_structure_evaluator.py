from pathlib import Path

import pytest

from docling_eval.evaluators.doc_structure_evaluator import DocStructureEvaluator


@pytest.mark.dependency(
    depends=["tests/test_dataset_builder.py::test_run_dpbench_e2e"],
    scope="session",
)
def test_doc_structure_evaluator_external_predictions():
    r"""Testing the evaluator with external predictions"""
    eval = DocStructureEvaluator()
    gt_path = Path("scratch/DPBench/gt_dataset")
    preds_path = Path("scratch/DPBench/predicted_documents/json")

    v = eval(gt_path, external_predictions_path=preds_path)
    assert v is not None


if __name__ == "__main__":
    test_doc_structure_evaluator_external_predictions()
