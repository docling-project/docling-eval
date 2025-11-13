from pathlib import Path
from typing import Dict, Optional

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

    # Custom label mappings
    label_mapping: Optional[Dict[DocItemLabel, Optional[DocItemLabel]]] = {
        DocItemLabel.CAPTION: DocItemLabel.TITLE,
        DocItemLabel.DOCUMENT_INDEX: None,
    }
    eval2 = PixelLayoutEvaluator(label_mapping=label_mapping)

    # Save the evaluations
    pixel_ds_evaluation: DatasetPixelLayoutEvaluation = eval1(test_dataset_dir)
    pixel_save_root: Path = test_dataset_dir / "pixel_layout_evaluations"
    eval1.save_evaluations(
        BenchMarkNames.DPBENCH,
        pixel_ds_evaluation,
        pixel_save_root,
    )


if __name__ == "__main__":
    test_layout_evaluator()
