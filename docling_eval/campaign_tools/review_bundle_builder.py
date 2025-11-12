"""Utilities to create static review bundles for CVAT submissions."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from pandas.api import types as pd_types


@dataclass
class VisualizationFile:
    """Structured information about a visualization HTML file."""

    label: str
    relative_path: str


@dataclass
class ManifestEntry:
    """Serializable row that backs a single review frame."""

    entry_id: str
    doc_name: str
    image_name: str
    review_value: Any
    metadata: dict[str, Any]
    visualizations: list[VisualizationFile]


def _read_analysis_sheet(analysis_path: Path, review_column: str) -> pd.DataFrame:
    if analysis_path.suffix.lower() != ".csv":
        raise ValueError(
            f"Analysis file must be a CSV export. Found '{analysis_path.name}'."
        )
    dataframe = pd.read_csv(analysis_path)
    if review_column not in dataframe.columns:
        available = ", ".join(str(column) for column in dataframe.columns)
        raise ValueError(
            f"Column '{review_column}' not present in {analysis_path}. Available columns: {available}."
        )
    required_columns = {"doc_name", "image_name"}
    missing = required_columns.difference({str(column) for column in dataframe.columns})
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"Analysis sheet must contain columns {missing_columns}.")
    review_series = dataframe[review_column]
    if not pd_types.is_numeric_dtype(review_series):
        numeric_series = pd.to_numeric(review_series, errors="coerce")
        if numeric_series.notna().any():
            dataframe = dataframe.copy()
            dataframe["__sort_key"] = numeric_series
            sorted_frame = dataframe.sort_values(
                by=["__sort_key", review_column],
                ascending=[False, False],
                kind="stable",
            ).drop(columns="__sort_key")
        else:
            sorted_frame = dataframe.sort_values(
                by=review_column, ascending=False, kind="stable"
            )
    else:
        sorted_frame = dataframe.sort_values(
            by=review_column, ascending=False, kind="stable"
        )
    return sorted_frame.reset_index(drop=True)


def _stringify_value(value: Any) -> Any:
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _collect_visualizations(submission_dir: Path) -> list[Path]:
    visualizations: list[Path] = []
    for html_dir in sorted(submission_dir.glob("*/eval_dataset/visualizations")):
        visualizations.extend(sorted(html_dir.rglob("*.html")))
    return visualizations


def _select_matches(doc_name: str, html_files: Iterable[Path]) -> list[Path]:
    raw_name = doc_name.strip()
    plain_name = Path(raw_name).stem
    matches: list[Path] = []
    for html_path in html_files:
        file_name = html_path.name
        stem_name = html_path.stem
        if file_name.startswith(raw_name) or file_name.startswith(plain_name):
            matches.append(html_path)
        elif stem_name.startswith(raw_name) or stem_name.startswith(plain_name):
            matches.append(html_path)
    return matches


def _stage_visualization(
    html_path: Path,
    submission_dir: Path,
    bundle_dir: Path,
    staged_paths: dict[Path, str],
) -> str:
    if html_path in staged_paths:
        return staged_paths[html_path]
    if not html_path.is_relative_to(submission_dir):
        raise ValueError(
            f"Visualization {html_path} is outside the submission directory {submission_dir}."
        )
    relative_source = html_path.relative_to(submission_dir)
    destination = bundle_dir / "visualizations" / relative_source
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(html_path, destination)
    relative_path = destination.relative_to(bundle_dir).as_posix()
    staged_paths[html_path] = relative_path
    return relative_path


def _build_manifest_entries(
    dataframe: pd.DataFrame,
    html_files: list[Path],
    submission_dir: Path,
    bundle_dir: Path,
    stage_visualizations: bool,
    review_column: str,
) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    staged_paths: dict[Path, str] = {}
    for row_index, (_, row) in enumerate(dataframe.iterrows()):
        doc_value = row["doc_name"]
        image_value = row["image_name"]
        if pd.isna(doc_value) or pd.isna(image_value):
            raise ValueError(
                f"Row {row_index} is missing doc_name or image_name. These fields are required."
            )
        doc_name = str(doc_value).strip()
        image_name = str(image_value).strip()
        if not doc_name or not image_name:
            raise ValueError(
                (
                    "Row "
                    f"{row_index} contains blank doc_name or image_name. Populate these columns "
                    "before running the review bundle builder."
                )
            )
        review_value = _stringify_value(row.get(review_column))
        metadata = {
            str(column): _stringify_value(row[column]) for column in dataframe.columns
        }
        matches = _select_matches(doc_name, html_files)
        relative_matches = []
        for match in matches:
            if stage_visualizations:
                staged_path = _stage_visualization(
                    match, submission_dir, bundle_dir, staged_paths
                )
                relative_matches.append(
                    VisualizationFile(label=match.name, relative_path=staged_path)
                )
            else:
                relative_to_submission = match.relative_to(submission_dir)
                relative_from_bundle = Path("..") / relative_to_submission
                relative_matches.append(
                    VisualizationFile(
                        label=match.name,
                        relative_path=relative_from_bundle.as_posix(),
                    )
                )
        entry = ManifestEntry(
            entry_id=f"entry_{row_index:05d}",
            doc_name=doc_name,
            image_name=image_name,
            review_value=review_value,
            metadata=metadata,
            visualizations=relative_matches,
        )
        entries.append(entry)
    return entries


def _write_json(path: Path, content: Any) -> None:
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


def build_review_bundle(
    submission_dir: Path,
    review_column: str = "need_review",
    analysis_filename: str = "combined_evaluation.csv",
    stage_visualizations: bool = False,
) -> Path:
    submission_dir = submission_dir.resolve()
    if not submission_dir.exists():
        raise FileNotFoundError(
            f"Submission directory {submission_dir} does not exist."
        )
    analysis_path = submission_dir / analysis_filename
    if not analysis_path.exists():
        raise FileNotFoundError(
            f"Analysis file {analysis_path} not found inside {submission_dir}."
        )
    dataframe = _read_analysis_sheet(analysis_path, review_column)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_dir = submission_dir / f"_review_bundle_{timestamp}"
    bundle_dir.mkdir(parents=True, exist_ok=False)
    asset_dir = Path(__file__).with_name("review_bundle_assets")
    shutil.copytree(asset_dir, bundle_dir, dirs_exist_ok=True)
    html_files = _collect_visualizations(submission_dir)
    entries = _build_manifest_entries(
        dataframe=dataframe,
        html_files=html_files,
        submission_dir=submission_dir,
        bundle_dir=bundle_dir,
        stage_visualizations=stage_visualizations,
        review_column=review_column,
    )
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "submission_dir": str(submission_dir),
        "review_column": review_column,
        "analysis_file": str(analysis_path),
        "total_entries": len(entries),
        "entries": [asdict(entry) for entry in entries],
    }
    _write_json(bundle_dir / "manifest.json", manifest)
    _write_json(bundle_dir / "review_state.json", {"decisions": []})
    return bundle_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build a static review bundle.")
    parser.add_argument(
        "submission_dir", type=Path, help="Path to the submission folder"
    )
    parser.add_argument(
        "--review-column",
        default="need_review",
        help="Column to sort by in descending order",
    )
    parser.add_argument(
        "--analysis-filename",
        default="combined_evaluation.csv",
        help="Analysis CSV filename located inside the submission directory",
    )
    parser.add_argument(
        "--stage-visualizations",
        action="store_true",
        help="Copy matching visualization HTML files into the bundle (useful when hosting the bundle alone).",
    )
    args = parser.parse_args()
    bundle_path = build_review_bundle(
        submission_dir=args.submission_dir,
        review_column=args.review_column,
        analysis_filename=args.analysis_filename,
        stage_visualizations=args.stage_visualizations,
    )
    print(f"Review bundle created at {bundle_path}")


if __name__ == "__main__":
    main()
