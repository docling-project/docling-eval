import json
from pathlib import Path

import pyarrow as pa

from docling_eval.utils import utils


def test_save_shard_to_disk_splits_one_call_into_multiple_shards(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_path = tmp_path / "test"
    dataset_path.mkdir(parents=True, exist_ok=True)

    items = [
        {"document_id": "doc_a", "payload": "a"},
        {"document_id": "doc_b", "payload": "b"},
        {"document_id": "doc_c", "payload": "c"},
    ]
    schema = pa.schema([("document_id", pa.string()), ("payload", pa.string())])

    monkeypatch.setattr(utils, "_ARROW_SHARD_TARGET_BYTES", 100)
    monkeypatch.setattr(utils, "_estimate_prepared_record_size_bytes", lambda _: 70)

    result = utils.save_shard_to_disk(
        items=items,
        dataset_path=dataset_path,
        schema=schema,
    )

    parquet_files = sorted(dataset_path.glob("*.parquet"))

    assert result.written_record_count == 3
    assert result.written_shard_count == 3
    assert result.skipped_record_count == 0
    assert result.next_shard_id == 3
    assert result.skipped_doc_ids == []
    assert [path.name for path in parquet_files] == [
        "shard_000000_000000.parquet",
        "shard_000000_000001.parquet",
        "shard_000000_000002.parquet",
    ]


def test_save_shard_to_disk_skips_failed_shard_and_records_audit(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_path = tmp_path / "test"
    dataset_path.mkdir(parents=True, exist_ok=True)

    items = [
        {"document_id": "good_a", "payload": "a"},
        {"document_id": "bad_doc", "payload": "b"},
        {"document_id": "good_b", "payload": "c"},
    ]
    schema = pa.schema([("document_id", pa.string()), ("payload", pa.string())])

    monkeypatch.setattr(utils, "_ARROW_SHARD_TARGET_BYTES", 100)
    monkeypatch.setattr(utils, "_estimate_prepared_record_size_bytes", lambda _: 60)

    def fake_save_to_parquet_direct(
        items: list[dict[str, str]],
        dataset_path: Path,
        thread_id: int,
        shard_id: int,
        schema: pa.Schema,
    ) -> None:
        del schema
        if items[0]["document_id"] == "bad_doc":
            raise pa.ArrowCapacityError("too large")
        (dataset_path / f"shard_{thread_id:06}_{shard_id:06}.parquet").write_bytes(
            b"ok"
        )

    monkeypatch.setattr(utils, "_save_to_parquet_direct", fake_save_to_parquet_direct)

    result = utils.save_shard_to_disk(
        items=items,
        dataset_path=dataset_path,
        schema=schema,
    )

    parquet_files = sorted(dataset_path.glob("*.parquet"))
    audit_entries = [
        json.loads(line)
        for line in (dataset_path / "skipped_records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert result.written_record_count == 2
    assert result.written_shard_count == 2
    assert result.skipped_record_count == 1
    assert result.next_shard_id == 3
    assert result.skipped_doc_ids == ["bad_doc"]
    assert [path.name for path in parquet_files] == [
        "shard_000000_000000.parquet",
        "shard_000000_000002.parquet",
    ]
    assert audit_entries == [
        {
            "thread_id": 0,
            "shard_id": 1,
            "doc_id": "bad_doc",
            "doc_path": None,
            "estimated_size_bytes": 60,
            "record_count_in_dropped_shard": 1,
            "reason": "ArrowCapacityError: too large",
        }
    ]
