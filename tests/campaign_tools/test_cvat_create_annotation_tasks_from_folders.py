from pathlib import Path

from docling_eval.campaign_tools.cvat_create_annotation_tasks_from_folders import (
    process_subdirectories,
)


def test_process_subdirectories_resumes_from_absolute_index(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    source_dir = input_root / "dataset_a"
    source_dir.mkdir(parents=True, exist_ok=True)

    for index in range(6):
        (source_dir / f"file_{index:03d}.pdf").write_bytes(b"%PDF-1.4\n")

    created_gt_ranges: list[tuple[str, int, int]] = []
    created_cvat_dirs: list[str] = []

    def fake_count_pages_in_file(_: Path) -> int:
        return 1

    def fake_create_gt(
        *,
        benchmark,
        dataset_source: Path,
        output_dir: Path,
        do_visualization: bool,
        begin_index: int,
        end_index: int,
    ) -> None:
        del benchmark, dataset_source, do_visualization
        created_gt_ranges.append((output_dir.name, begin_index, end_index))
        (output_dir / "gt_dataset" / "test").mkdir(parents=True, exist_ok=True)

    def fake_create_cvat(
        *,
        gt_dir: Path,
        output_dir: Path,
        bucket_size: int,
        use_predictions: bool,
        sliding_window: int,
    ) -> None:
        del gt_dir, bucket_size, use_predictions, sliding_window
        created_cvat_dirs.append(output_dir.parent.name)
        output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "docling_eval.campaign_tools.cvat_create_annotation_tasks_from_folders.count_pages_in_file",
        fake_count_pages_in_file,
    )
    monkeypatch.setattr(
        "docling_eval.campaign_tools.cvat_create_annotation_tasks_from_folders.create_gt",
        fake_create_gt,
    )
    monkeypatch.setattr(
        "docling_eval.campaign_tools.cvat_create_annotation_tasks_from_folders.create_cvat",
        fake_create_cvat,
    )

    (output_root / "dataset_a_chunk_002" / "cvat_dataset_preannotated").mkdir(
        parents=True, exist_ok=True
    )

    process_subdirectories(
        input_directory=input_root,
        output_directory=output_root,
        sliding_window=1,
        use_predictions=False,
        max_files_per_chunk=2,
        start_index=2,
        end_index=-1,
        chunk_number_offset=1,
    )

    assert created_gt_ranges == [("dataset_a_chunk_003", 4, 6)]
    assert created_cvat_dirs == ["dataset_a_chunk_003"]
