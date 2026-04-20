import shutil
from pathlib import Path

import pytest
from datasets import load_dataset
from docling_core.types import DoclingDocument
from PIL import Image

from docling_eval.datamodels.dataset_record import DatasetRecordWithBBox
from docling_eval.datamodels.types import BenchMarkNames
from docling_eval.dataset_builders.doclingsdg_builder import DoclingSDGDatasetBuilder


def test_doclingsdg_builder_with_png_and_docling_json(tmp_path: Path):
    dataset_source = tmp_path / "doclingsdg_source"
    target = tmp_path / "doclingsdg_target"
    dataset_source.mkdir(parents=True)

    png_src = Path("tests/data/files/2305.03393v1-pg9-img.png")
    png_dst = dataset_source / "sample.png"
    shutil.copy2(png_src, png_dst)

    document = DoclingDocument(name="sample")
    document.save_as_json(dataset_source / "sample.json")

    builder = DoclingSDGDatasetBuilder(
        dataset_source=dataset_source,
        target=target,
    )
    builder.save_to_disk(chunk_size=4)

    ds = load_dataset(
        "parquet",
        data_files={"test": str(target / "test" / "*.parquet")},
    )

    assert len(ds["test"]) == 1
    row = ds["test"][0]
    assert row["document_id"] == "sample"
    assert row["mime_type"] == "image/png"
    assert row["BinaryDocument"] is not None
    assert len(row["GroundTruthPageImages"]) == 1
    assert "GroundTruthBboxOnPageImages" in row

    restored = DatasetRecordWithBBox.model_validate(row)
    assert restored.ground_truth_bbox_on_page_images == {}


def test_get_dataset_builder_returns_doclingsdg(tmp_path: Path):
    try:
        from docling_eval.cli.main import get_dataset_builder
    except ModuleNotFoundError as exc:
        pytest.skip(f"Optional dependency missing for cli.main import: {exc}")

    dataset_source = tmp_path / "doclingsdg_source"
    dataset_source.mkdir(parents=True)

    builder = get_dataset_builder(
        benchmark=BenchMarkNames.DOCLING_SDG,
        target=tmp_path / "target",
        dataset_source=dataset_source,
    )

    assert isinstance(builder, DoclingSDGDatasetBuilder)


def test_doclingsdg_builder_supports_paged_png_inputs(tmp_path: Path):
    dataset_source = tmp_path / "doclingsdg_paged_source"
    target = tmp_path / "doclingsdg_paged_target"
    dataset_source.mkdir(parents=True)

    png_src = Path("tests/data/files/2305.03393v1-pg9-img.png")
    shutil.copy2(png_src, dataset_source / "multi_page_000001.png")
    shutil.copy2(png_src, dataset_source / "multi_page_000002.png")

    document = DoclingDocument(name="multi")
    document.save_as_json(dataset_source / "multi.json")

    builder = DoclingSDGDatasetBuilder(dataset_source=dataset_source, target=target)
    builder.save_to_disk(chunk_size=4)

    ds = load_dataset(
        "parquet",
        data_files={"test": str(target / "test" / "*.parquet")},
    )
    row = ds["test"][0]

    assert row["document_id"] == "multi"
    assert row["mime_type"] == "application/pdf"
    assert len(row["GroundTruthPageImages"]) == 2
    assert "GroundTruthBboxOnPageImages" in row

    restored = DatasetRecordWithBBox.model_validate(row)
    assert restored.ground_truth_bbox_on_page_images == {}


def test_doclingsdg_builder_populates_top_level_bboxes(tmp_path: Path):
    dataset_source = tmp_path / "doclingsdg_bbox_source"
    target = tmp_path / "doclingsdg_bbox_target"
    dataset_source.mkdir(parents=True)

    sample_json = next(Path("tests/data/test_doclingsdg_docs").glob("*.json"))
    json_dst = dataset_source / sample_json.name
    shutil.copy2(sample_json, json_dst)

    # Use a large enough image so sample bbox coordinates stay in bounds.
    Image.new("RGB", (2000, 2000), "white").save(
        dataset_source / f"{sample_json.stem}.png"
    )

    builder = DoclingSDGDatasetBuilder(dataset_source=dataset_source, target=target)
    builder.save_to_disk(chunk_size=4)

    ds = load_dataset(
        "parquet",
        data_files={"test": str(target / "test" / "*.parquet")},
    )
    row = ds["test"][0]

    restored = DatasetRecordWithBBox.model_validate(row)
    assert 0 in restored.ground_truth_bbox_on_page_images
    assert len(restored.ground_truth_bbox_on_page_images[0]) >= 1
    first_box = restored.ground_truth_bbox_on_page_images[0][0]
    assert first_box["label"] == "table"
    assert "category_id" in first_box
    assert len(first_box["bbox"]) == 4
    assert len(first_box["ltrb"]) == 4
