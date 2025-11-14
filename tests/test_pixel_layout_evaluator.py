from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pytest
from docling.datamodel.base_models import ConversionStatus
from docling_core.types.doc.labels import DocItemLabel

from docling_eval.datamodels.types import BenchMarkNames, PredictionFormats
from docling_eval.evaluators.markdown_text_evaluator import MarkdownTextEvaluator
from docling_eval.evaluators.pixel.pixel_types import (
    DatasetPixelLayoutEvaluation,
    MultiLabelMatrixEvaluation,
)
from docling_eval.evaluators.pixel_layout_evaluator import PixelLayoutEvaluator


@pytest.mark.dependency(
    depends=["tests/test_dataset_builder.py::test_run_dpbench_e2e"],
    scope="session",
)
def test_layout_evaluator():
    r""" """
    test_dataset_dir = Path("scratch/DPBench/eval_dataset_e2e")

    # Initialize default evaluator
    eval1 = PixelLayoutEvaluator()

    # Peform the evaluation
    evaluation: DatasetPixelLayoutEvaluation = eval1(test_dataset_dir)

    # Generic assertions
    assert evaluation is not None
    for rejection_type, rejection_count in evaluation.rejected_samples.items():
        assert (
            rejection_count == 0
        ), f"Unexpected rejections of type: {rejection_type.value}"

    # Pixel evalution assertions
    assert evaluation.num_pages == len(evaluation.page_evaluations)
    detailed_class_names: dict[str, str] = (
        evaluation.matrix_evaluation.detailed.class_names
    )
    num_classes = len(detailed_class_names)
    confusion_matrix_list = evaluation.matrix_evaluation.detailed.confusion_matrix
    detailed_confusion_matrix = np.asarray(confusion_matrix_list)
    assert detailed_confusion_matrix.shape == (
        num_classes,
        num_classes,
    ), "Wrong detailed confusion matrix dims"
    colapsed_confusion_matrix_list = (
        evaluation.matrix_evaluation.colapsed.confusion_matrix
    )
    colapsed_confusion_matrix = np.asarray(colapsed_confusion_matrix_list)
    assert colapsed_confusion_matrix.shape == (
        2,
        2,
    ), "Wrong colapsed confusion matrix dims"

    # Save the evaluation
    pixel_save_root = Path(
        "scratch/DPBench/evaluations/layout/pixel_layout_evaluations"
    )
    eval1.save_evaluations(
        BenchMarkNames.DPBENCH,
        evaluation,
        pixel_save_root,
    )
    expected_json_fn = pixel_save_root / "evaluation_DPBench_pixel_layout.json"
    expected_excel_fn = pixel_save_root / "evaluation_DPBench_pixel_layout.xlsx"
    assert expected_json_fn.is_file(), "Missing evaluation json file"
    assert expected_excel_fn.is_file(), "Missing evaluation excel file"

    # Initialize with custom label mappings
    label_mapping: Optional[Dict[DocItemLabel, Optional[DocItemLabel]]] = {
        DocItemLabel.CAPTION: DocItemLabel.TITLE,
        DocItemLabel.DOCUMENT_INDEX: None,
    }
    eval2 = PixelLayoutEvaluator(label_mapping=label_mapping)
    assert len(eval2._matrix_doclabelitem_to_id) + 1 == len(eval2._matrix_id_to_name)
    assert (
        eval2._matrix_doclabelitem_to_id[DocItemLabel.CAPTION]
        == eval2._matrix_doclabelitem_to_id[DocItemLabel.TITLE]
    ), "Wrong label mapping in _matrix_doclabelitem_to_id"
    assert (
        DocItemLabel.CAPTION.value not in eval2._matrix_id_to_name.values()
    ), "Wrong label mapping in _matrix_id_to_name"


# if __name__ == "__main__":
#     test_layout_evaluator()
