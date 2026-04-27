import json
import shutil
from pathlib import Path

import pytest
from datasets import load_dataset
from docling_core.types import DoclingDocument
from PIL import Image

from docling_eval.datamodels.dataset_record import DatasetRecordWithBBox
from docling_eval.datamodels.types import BenchMarkNames
from docling_eval.dataset_builders.doclingsdg_builder import DoclingSDGDatasetBuilder


def _copy_json_png_pair(source_json: Path, target_dir: Path) -> None:
    png_path = source_json.with_suffix(".png")
    assert png_path.exists(), f"Missing PNG pair for {source_json}"
    shutil.copy2(source_json, target_dir / source_json.name)
    shutil.copy2(png_path, target_dir / png_path.name)


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


def test_doclingsdg_builder_scans_nested_subfolders(tmp_path: Path):
    dataset_source = tmp_path / "doclingsdg_nested_source"
    target = tmp_path / "doclingsdg_nested_target"
    nested_dir = dataset_source / "nested"
    nested_dir.mkdir(parents=True)

    sample_json = next(Path("tests/data/test_doclingsdg_docs").glob("*.json"))
    _copy_json_png_pair(sample_json, nested_dir)

    builder = DoclingSDGDatasetBuilder(dataset_source=dataset_source, target=target)
    builder.save_to_disk(chunk_size=4)

    ds = load_dataset(
        "parquet",
        data_files={"test": str(target / "test" / "*.parquet")},
    )
    assert len(ds["test"]) == 1


def test_doclingsdg_builder_table_regions_bbox_labels(tmp_path: Path):
    dataset_source = tmp_path / "doclingsdg_table_regions_source"
    target = tmp_path / "doclingsdg_table_regions_target"
    dataset_source.mkdir(parents=True)

    sample_json = Path(
        "tests/data/test_doclingsdg_docs/"
        "data_none__seed_teds_0.940_table_table_dataset_20260310_"
        "tight_margin_ftn_margin_doc20169_t000_row35_col8__0736.json"
    )
    _copy_json_png_pair(sample_json, dataset_source)

    builder = DoclingSDGDatasetBuilder(
        dataset_source=dataset_source,
        target=target,
        modality="table_regions",
    )
    builder.save_to_disk(chunk_size=4)

    ds = load_dataset(
        "parquet",
        data_files={"test": str(target / "test" / "*.parquet")},
    )
    assert len(ds["test"]) == 1
    row = ds["test"][0]
    assert row["modalities"] == ["table_regions"]

    restored = DatasetRecordWithBBox.model_validate(row)
    labels = {box["label"] for box in restored.ground_truth_bbox_on_page_images[0]}
    assert "table" in labels
    assert "row" in labels
    assert "column" in labels
    assert "cell_single" in labels
    assert "cell_merged" in labels


def test_doclingsdg_builder_table_regions_adds_90_degree_tag(tmp_path: Path):
    dataset_source = tmp_path / "doclingsdg_rot_source"
    target = tmp_path / "doclingsdg_rot_target"
    dataset_source.mkdir(parents=True)

    rotated_source = Path("EXAMPLE_DOCLING_SDG_TABLES/rotated_90_deg")
    sample_json = sorted(rotated_source.glob("*.json"))[0]
    _copy_json_png_pair(sample_json, dataset_source)

    builder = DoclingSDGDatasetBuilder(
        dataset_source=dataset_source,
        target=target,
        modality="table_regions",
    )
    builder.save_to_disk(chunk_size=4)

    ds = load_dataset(
        "parquet",
        data_files={"test": str(target / "test" / "*.parquet")},
    )
    assert len(ds["test"]) == 1
    assert "90_degree" in ds["test"][0]["tags"]


def test_doclingsdg_builder_table_regions_skips_malformed_sample(tmp_path: Path):
    dataset_source = tmp_path / "doclingsdg_malformed_source"
    target = tmp_path / "doclingsdg_malformed_target"
    dataset_source.mkdir(parents=True)

    good_sample = next(Path("EXAMPLE_DOCLING_SDG_TABLES/generated").glob("*.json"))
    _copy_json_png_pair(good_sample, dataset_source)

    malformed_sample = Path(
        "EXAMPLE_DOCLING_SDG_TABLES/broken/"
        "data_none_seed_teds_0.094_table_doclaynet_set_a_file_doc2353_t005_"
        "row17_col2_0741__var70885.json"
    )
    _copy_json_png_pair(malformed_sample, dataset_source)

    builder = DoclingSDGDatasetBuilder(
        dataset_source=dataset_source,
        target=target,
        modality="table_regions",
    )
    builder.save_to_disk(chunk_size=4)

    ds = load_dataset(
        "parquet",
        data_files={"test": str(target / "test" / "*.parquet")},
    )
    assert len(ds["test"]) == 1


def test_doclingsdg_builder_table_regions_skips_out_of_bounds_table_bbox(
    tmp_path: Path,
):
    dataset_source = tmp_path / "doclingsdg_oob_source"
    target = tmp_path / "doclingsdg_oob_target"
    dataset_source.mkdir(parents=True)

    sample_json = next(Path("tests/data/test_doclingsdg_docs").glob("*.json"))
    sample_png = sample_json.with_suffix(".png")
    assert sample_png.exists()
    shutil.copy2(sample_png, dataset_source / sample_png.name)

    with sample_json.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    # Force the first table bbox out of image bounds.
    table_prov = payload["tables"][0]["prov"][0]["bbox"]
    table_prov["r"] = float(table_prov["r"]) + 5000.0

    broken_json = dataset_source / sample_json.name
    with broken_json.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle)

    builder = DoclingSDGDatasetBuilder(
        dataset_source=dataset_source,
        target=target,
        modality="table_regions",
    )
    builder.save_to_disk(chunk_size=4)

    assert not list((target / "test").glob("*.parquet"))


def test_doclingsdg_builder_table_regions_skips_duplicate_left_on_same_row_band(
    tmp_path: Path,
):
    dataset_source = tmp_path / "doclingsdg_same_left_source"
    target = tmp_path / "doclingsdg_same_left_target"
    dataset_source.mkdir(parents=True)

    good_sample = next(Path("EXAMPLE_DOCLING_SDG_TABLES/generated").glob("*.json"))
    _copy_json_png_pair(good_sample, dataset_source)

    malformed_sample = Path(
        "EXAMPLE_DOCLING_SDG_TABLES/broken/"
        "data_none_seed_teds_0.079_table_table_dataset_20260310_tight_margin_"
        "wordscape_tight_doc97901_t000_row1_col2_0059__var653882.json"
    )
    _copy_json_png_pair(malformed_sample, dataset_source)

    builder = DoclingSDGDatasetBuilder(
        dataset_source=dataset_source,
        target=target,
        modality="table_regions",
    )
    builder.save_to_disk(chunk_size=4)

    ds = load_dataset(
        "parquet",
        data_files={"test": str(target / "test" / "*.parquet")},
    )
    assert len(ds["test"]) == 1


def test_doclingsdg_builder_table_regions_uses_table_visualizer_for_viz(
    tmp_path: Path,
):
    dataset_source = tmp_path / "doclingsdg_viz_source"
    target = tmp_path / "doclingsdg_viz_target"
    dataset_source.mkdir(parents=True)

    sample_json = Path(
        "tests/data/test_doclingsdg_docs/"
        "data_none__seed_teds_0.940_table_table_dataset_20260310_"
        "tight_margin_ftn_margin_doc20169_t000_row35_col8__0736.json"
    )
    _copy_json_png_pair(sample_json, dataset_source)

    builder = DoclingSDGDatasetBuilder(
        dataset_source=dataset_source,
        target=target,
        modality="table_regions",
    )
    builder.save_to_disk(chunk_size=4, do_visualization=True)

    viz_file = target / "visualizations" / f"{sample_json.stem}_layout.html"
    assert viz_file.exists()
    content = viz_file.read_text(encoding="utf-8")
    assert "Table Regions Visualization" in content
