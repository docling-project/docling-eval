#!/usr/bin/env python3
"""
Split paired (name.json, name.png) files from an input directory tree into
train/test/val output directories by moving the files.
Output split directories are kept flat. If multiple pairs would collide on the
same output basename, the script appends numeric suffixes (`_01`, `_02`, ...)
to both files in a pair.

Example:
    python docling_eval/utils/split_input_data.py \
        /path/to/input_root \
        /path/to/output/train \
        /path/to/output/test \
        /path/to/output/val

The input root can contain nested folders, for example:
    /path/to/input_root/batch_a/doc_001.json + doc_001.png
    /path/to/input_root/batch_b/doc_002.json + doc_002.png

If the same basename appears in multiple subfolders, output remains flat and
later collisions are renamed as a pair, for example:
    doc_001.json + doc_001.png
    doc_001_01.json + doc_001_01.png

Dry-run example:
    python docling_eval/utils/split_input_data.py \
        /path/to/input_root \
        /path/to/output/train \
        /path/to/output/test \
        /path/to/output/val \
        --dry-run
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FilePair:
    stem: str
    json_path: Path
    png_path: Path


def _collect_pairs(input_dir: Path) -> tuple[list[FilePair], list[str], list[str]]:
    json_candidates = list(input_dir.rglob("*.json"))
    json_candidates.extend(input_dir.rglob("*.JSON"))

    # Dedupe in case of case-insensitive overlap
    json_files = sorted({p.resolve(): p for p in json_candidates}.values())

    pairs: list[FilePair] = []
    missing_png: list[str] = []

    for json_path in json_files:
        rel_stem = str(json_path.relative_to(input_dir).with_suffix(""))
        png_lower = json_path.with_suffix(".png")
        png_upper = json_path.with_suffix(".PNG")

        if png_lower.exists():
            png_path = png_lower
        elif png_upper.exists():
            png_path = png_upper
        else:
            missing_png.append(rel_stem)
            continue

        pairs.append(
            FilePair(
                stem=rel_stem,
                json_path=json_path,
                png_path=png_path,
            )
        )

    paired_stems = {p.stem for p in pairs}
    png_candidates = list(input_dir.rglob("*.png"))
    png_candidates.extend(input_dir.rglob("*.PNG"))
    orphan_png = sorted(
        {
            str(png_path.relative_to(input_dir).with_suffix(""))
            for png_path in png_candidates
            if str(png_path.relative_to(input_dir).with_suffix("")) not in paired_stems
        }
    )

    return pairs, missing_png, orphan_png


def _split_counts(
    total: int, train_ratio: float, test_ratio: float, val_ratio: float
) -> tuple[int, int, int]:
    if total == 0:
        return 0, 0, 0

    train_count = int(total * train_ratio)
    test_count = int(total * test_ratio)
    # Make sure all items are assigned.
    val_count = total - train_count - test_count
    return train_count, test_count, val_count


def _validate_ratios(
    train_ratio: float, test_ratio: float, val_ratio: float
) -> tuple[float, float, float]:
    ratios = [train_ratio, test_ratio, val_ratio]
    if any(r < 0 for r in ratios):
        raise ValueError("Split ratios must be non-negative.")

    ratio_sum = sum(ratios)
    if ratio_sum <= 0:
        raise ValueError("At least one split ratio must be > 0.")

    if abs(ratio_sum - 1.0) < 1e-9:
        return train_ratio, test_ratio, val_ratio

    # Normalize if the user passed ratios that do not sum to 1 exactly.
    return train_ratio / ratio_sum, test_ratio / ratio_sum, val_ratio / ratio_sum


def _ensure_targets(*dirs: Path) -> None:
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def _init_reserved_names(target_dir: Path) -> set[str]:
    return {p.name for p in target_dir.iterdir() if p.is_file()}


def _resolve_pair_targets(
    pair: FilePair, target_dir: Path, reserved_names: set[str]
) -> tuple[Path, Path]:
    base_stem = pair.json_path.stem
    json_ext = pair.json_path.suffix
    png_ext = pair.png_path.suffix

    index = 0
    while True:
        suffix = "" if index == 0 else f"_{index:02d}"
        candidate_stem = f"{base_stem}{suffix}"
        json_name = f"{candidate_stem}{json_ext}"
        png_name = f"{candidate_stem}{png_ext}"

        json_target = target_dir / json_name
        png_target = target_dir / png_name

        name_taken = json_name in reserved_names or png_name in reserved_names
        file_taken = json_target.exists() or png_target.exists()
        if not name_taken and not file_taken:
            reserved_names.add(json_name)
            reserved_names.add(png_name)
            return json_target, png_target

        index += 1


def _move_pair(
    pair: FilePair, target_dir: Path, dry_run: bool, reserved_names: set[str]
) -> None:
    json_target, png_target = _resolve_pair_targets(pair, target_dir, reserved_names)

    if dry_run:
        print(f"[DRY-RUN] {pair.json_path} -> {json_target}")
        print(f"[DRY-RUN] {pair.png_path} -> {png_target}")
        return

    shutil.move(str(pair.json_path), str(json_target))
    shutil.move(str(pair.png_path), str(png_target))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Move paired (name.json, name.png) files from input directory and its "
            "subdirectories into train/test/val directories according to split ratios."
        )
    )

    parser.add_argument(
        "input_dir", type=Path, help="Directory with source JSON/PNG pairs"
    )
    parser.add_argument(
        "train_output_dir", type=Path, help="Output directory for train split"
    )
    parser.add_argument(
        "test_output_dir", type=Path, help="Output directory for test split"
    )
    parser.add_argument(
        "val_output_dir", type=Path, help="Output directory for val split"
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio (default: 0.8)",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.1, help="Test split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="Val split ratio (default: 0.1)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves without changing files",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    input_dir: Path = args.input_dir
    if not input_dir.exists() or not input_dir.is_dir():
        print(
            f"Error: input directory does not exist or is not a directory: {input_dir}",
            file=sys.stderr,
        )
        return 1

    try:
        train_ratio, test_ratio, val_ratio = _validate_ratios(
            args.train_ratio,
            args.test_ratio,
            args.val_ratio,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    train_out: Path = args.train_output_dir
    test_out: Path = args.test_output_dir
    val_out: Path = args.val_output_dir

    _ensure_targets(train_out, test_out, val_out)

    pairs, missing_png, orphan_png = _collect_pairs(input_dir)
    if not pairs:
        print("No valid (name.json, name.png) pairs found. Nothing to move.")
        if missing_png:
            print(f"JSON without PNG pairs: {len(missing_png)}")
        if orphan_png:
            print(f"PNG without JSON pairs: {len(orphan_png)}")
        return 0

    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    train_count, test_count, val_count = _split_counts(
        len(pairs), train_ratio, test_ratio, val_ratio
    )

    train_pairs = pairs[:train_count]
    test_pairs = pairs[train_count : train_count + test_count]
    val_pairs = pairs[train_count + test_count :]

    assert len(train_pairs) + len(test_pairs) + len(val_pairs) == len(pairs)

    train_reserved = _init_reserved_names(train_out)
    test_reserved = _init_reserved_names(test_out)
    val_reserved = _init_reserved_names(val_out)

    for pair in train_pairs:
        _move_pair(pair, train_out, args.dry_run, train_reserved)
    for pair in test_pairs:
        _move_pair(pair, test_out, args.dry_run, test_reserved)
    for pair in val_pairs:
        _move_pair(pair, val_out, args.dry_run, val_reserved)

    print(
        "Done. "
        f"Moved pairs -> train: {len(train_pairs)}, test: {len(test_pairs)}, val: {len(val_pairs)}"
    )

    if missing_png:
        print(f"Skipped JSON without matching PNG: {len(missing_png)}")
    if orphan_png:
        print(f"Found PNG without matching JSON: {len(orphan_png)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
