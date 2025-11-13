from pathlib import Path
from typing import Dict, Optional

import pytest
from docling.datamodel.base_models import ConversionStatus
from docling_core.types.doc.labels import DocItemLabel

from docling_eval.datamodels.types import PredictionFormats
from docling_eval.evaluators.markdown_text_evaluator import MarkdownTextEvaluator
from docling_eval.evaluators.pixel_layout_evaluator import PixelLayoutEvaluator


@pytest.mark.dependency(
    depends=["tests/test_dataset_builder.py::test_run_dpbench_e2e"],
    scope="session",
)
def test_layout_evaluator():
    r""" """
    test_dataset_dir = Path("scratch/DPBench/eval_dataset_e2e")

    # Default evaluator
    eval1 = PixelLayoutEvaluator()
    v1 = eval1(test_dataset_dir)
    assert v1 is not None

    # Custom label mappings
    label_mapping: Optional[Dict[DocItemLabel, Optional[DocItemLabel]]] = {
        DocItemLabel.CAPTION: DocItemLabel.TITLE,
        DocItemLabel.DOCUMENT_INDEX: None,
    }
    eval2 = PixelLayoutEvaluator(label_mapping=label_mapping)
    # v2 = eval2(test_dataset_dir)
    # assert v2 is not None


if __name__ == "__main__":
    test_layout_evaluator()
