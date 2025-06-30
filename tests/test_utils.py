import json
from pathlib import Path

from docling_eval.evaluators.ocr.evaluation_models import OcrDatasetEvaluationResult


def validate_evaluation_results(
    target_path: Path,
    benchmark: str,
    modality: str,
    evaluation_type: str = "ocr",
) -> OcrDatasetEvaluationResult:
    eval_json_filename = f"evaluation_{benchmark}_{modality}.json"
    eval_json_path = target_path / "evaluations" / evaluation_type / eval_json_filename

    assert eval_json_path.exists(), f"Evaluation JSON file not found: {eval_json_path}"

    with open(eval_json_path, "r") as f:
        result = json.load(f)

    assert result is not None, "Evaluation JSON file is empty or invalid."
    assert result, "Overall metrics not found in evaluation results."

    metrics = OcrDatasetEvaluationResult(**result)

    assert (
        metrics.f1_score > 0
    ), f"F1 score ({metrics.f1_score}) must be greater than 0."
    assert (
        metrics.precision > 0
    ), f"Precision score ({metrics.precision}) must be greater than 0."
    assert (
        metrics.recall > 0
    ), f"Recall score ({metrics.recall}) must be greater than 0."

    return metrics
