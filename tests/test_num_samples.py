from pathlib import Path

import pytest
from datasets import Dataset
from docling.datamodel.base_models import ConversionStatus
from docling_core.types.doc import DocItemLabel, DoclingDocument
from typer.testing import CliRunner

from docling_eval.cli import main as cli_main
from docling_eval.datamodels.dataset_record import DatasetRecordWithPrediction
from docling_eval.datamodels.types import BenchMarkNames, EvaluationModality
from docling_eval.evaluators.base_evaluator import BaseEvaluator
from docling_eval.evaluators.doc_structure_evaluator import DocStructureEvaluator
from docling_eval.utils.utils import save_shard_to_disk

NUM_RECORDS = 5


def _make_doc(name: str) -> DoclingDocument:
    doc = DoclingDocument(name=name)
    doc.add_text(label=DocItemLabel.TEXT, text=f"Some text of {name}")
    return doc


@pytest.fixture
def eval_dataset_dir(tmp_path: Path) -> Path:
    r"""Create a tiny evaluation dataset with NUM_RECORDS records in the test split"""
    records = []
    for i in range(NUM_RECORDS):
        doc = _make_doc(f"doc_{i}")
        record = DatasetRecordWithPrediction(
            doc_id=f"doc_{i}",
            ground_truth_doc=doc,
            predicted_doc=doc,
            status=ConversionStatus.SUCCESS,
        )
        records.append(record.as_record_dict())

    ds_dir = tmp_path / "eval_dataset"
    split_dir = ds_dir / "test"
    split_dir.mkdir(parents=True)
    save_shard_to_disk(
        items=records,
        dataset_path=split_dir,
        schema=DatasetRecordWithPrediction.pyarrow_schema(),
    )
    return ds_dir


def test_select_samples():
    ds = Dataset.from_dict({"x": list(range(NUM_RECORDS))})

    assert BaseEvaluator._select_samples(ds, None) is ds
    assert BaseEvaluator._select_samples(ds, NUM_RECORDS) is ds
    assert BaseEvaluator._select_samples(ds, NUM_RECORDS + 10) is ds
    assert BaseEvaluator._select_samples(ds, 2)["x"] == [0, 1]

    for invalid in [0, -1]:
        with pytest.raises(ValueError):
            BaseEvaluator._select_samples(ds, invalid)


@pytest.mark.parametrize(
    "num_samples, expected",
    [(None, NUM_RECORDS), (2, 2), (NUM_RECORDS + 10, NUM_RECORDS)],
)
def test_evaluator_num_samples(eval_dataset_dir: Path, num_samples, expected):
    evaluator = DocStructureEvaluator()
    evaluation = evaluator(eval_dataset_dir, num_samples=num_samples)
    assert evaluation.evaluated_samples == expected


def test_evaluate_num_samples(eval_dataset_dir: Path, tmp_path: Path):
    evaluations = cli_main.evaluate(
        modality=EvaluationModality.DOCUMENT_STRUCTURE,
        benchmark=BenchMarkNames.DPBENCH,
        idir=eval_dataset_dir,
        odir=tmp_path / "evaluations",
        num_samples=3,
    )
    assert evaluations is not None
    assert len(evaluations) == 1
    assert evaluations[0].evaluated_samples == 3


def test_evaluate_cmd_num_samples(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(cli_main, "evaluate", fake_evaluate)
    runner = CliRunner()
    base_args = [
        "evaluate",
        "--benchmark",
        BenchMarkNames.DPBENCH.value,
        "--modality",
        EvaluationModality.DOCUMENT_STRUCTURE.value,
        "--input-dir",
        str(tmp_path),
        "--output-dir",
        str(tmp_path),
    ]

    result = runner.invoke(cli_main.app, base_args + ["--num-samples", "7"])
    assert result.exit_code == 0, result.output
    assert captured["num_samples"] == 7

    captured.clear()
    result = runner.invoke(cli_main.app, base_args)
    assert result.exit_code == 0, result.output
    assert captured["num_samples"] is None

    result = runner.invoke(cli_main.app, base_args + ["--num-samples", "0"])
    assert result.exit_code != 0
