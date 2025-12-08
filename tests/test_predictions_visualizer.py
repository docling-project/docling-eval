from pathlib import Path

import pytest
from datasets import load_dataset

from docling_eval.datamodels.dataset_record import DatasetRecordWithPrediction
from docling_eval.utils.external_predictions_visualizer import PredictionsVisualizer


def _first_doc_id(parquet_root: Path) -> str:
    split_files = sorted((parquet_root / "test").glob("*.parquet"))
    ds = load_dataset(
        "parquet", data_files={"test": [str(path) for path in split_files]}
    )
    record = DatasetRecordWithPrediction.model_validate(ds["test"][0])
    return record.doc_id


@pytest.mark.dependency(
    depends=["tests/test_dataset_builder.py::test_run_dpbench_e2e"],
    scope="session",
)
def test_predictions_visualizer_with_embedded_predictions() -> None:
    dataset_dir = Path("scratch/DPBench/eval_dataset_e2e")
    output_dir = Path("scratch/DPBench/visualizer_tests/embedded")
    output_dir.mkdir(parents=True, exist_ok=True)

    visualizer = PredictionsVisualizer(visualizations_dir=output_dir)
    visualizer.create_visualizations(
        dataset_dir=dataset_dir,
        split="test",
        begin_index=0,
        end_index=1,
    )

    doc_id = _first_doc_id(dataset_dir)
    layout_file = output_dir / f"{doc_id}_layout.html"
    assert layout_file.is_file()


@pytest.mark.dependency(
    depends=["tests/test_dataset_builder.py::test_run_dpbench_e2e"],
    scope="session",
)
def test_predictions_visualizer_with_external_predictions() -> None:
    gt_dir = Path("scratch/DPBench/gt_dataset")
    external_predictions_dir = Path("scratch/DPBench/predicted_documents/json")
    output_dir = Path("scratch/DPBench/visualizer_tests/external")
    output_dir.mkdir(parents=True, exist_ok=True)

    visualizer = PredictionsVisualizer(
        visualizations_dir=output_dir,
        external_predictions_dir=external_predictions_dir,
    )
    visualizer.create_visualizations(
        dataset_dir=gt_dir,
        split="test",
        begin_index=0,
        end_index=1,
    )

    doc_id = _first_doc_id(gt_dir)
    layout_file = output_dir / f"{doc_id}_layout.html"
    assert layout_file.is_file()
