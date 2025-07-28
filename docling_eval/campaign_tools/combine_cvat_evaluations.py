"""combine_cvat_evaluations.py

We will use this script to combine the output metrics produced by our CVAT evaluation tooling into a single
spread-sheet (CSV or XLSX).

Inputs
------
* evaluation_CVAT_layout.json  - layout-level metrics (`evaluations_per_image`)
* evaluation_CVAT_document_structure.json - document-structure metrics
  (`evaluations`)
* file_name_user_id.csv - staff self-confidence / provenance table

The script matches the three sources by a **document id** that is derived from
an image / doc name **without the file-extension** and we produde single table.

Usage
-----
    python combine_cvat_evaluations.py \
        --layout_json evaluation_results/evaluation_CVAT_layout.json \
        --docstruct_json evaluation_results/evaluation_CVAT_document_structure.json \
        --user_csv file_name_user_id.csv \
        --out combined_evaluation.xlsx

*If ``--out`` ends with ``.csv`` the script will write a CSV; otherwise an
Excel workbook is produced.*
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Final, Optional

import pandas as pd
from xlsxwriter.utility import xl_range


def _to_doc_id(path_like: str) -> str:
    basename = os.path.basename(path_like)
    stem, _ = os.path.splitext(basename)
    # remove -page-1 suffix
    stem = stem.replace("-page-1", "")
    return stem


def load_layout(json_path: Path) -> pd.DataFrame:
    """Load *evaluation_CVAT_layout.json* and return a DataFrame."""
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if "evaluations_per_image" not in data:
        raise KeyError(
            "The supplied layout evaluation JSON does not contain the "
            "'evaluations_per_image' field."
        )

    # Flatten dictionaries into columns
    df = pd.json_normalize(data["evaluations_per_image"])

    # A handful of convenient renames for readability
    df = df.rename(
        columns={
            "name": "image_name",  # original filename / doc identifier
            "true_element_count": "element_count_set_A",
            "pred_element_count": "element_count_set_B",
            "true_table_count": "table_count_set_A",
            "pred_table_count": "table_count_set_B",
            "true_picture_count": "picture_count_set_A",
            "pred_picture_count": "picture_count_set_B",
            "table_count_diff": "table_count_diff",
            "picture_count_diff": "picture_count_diff",
        }
    )

    # Build merge key
    df["doc_id"] = df["image_name"].map(_to_doc_id)

    return df


def load_doc_structure(json_path: Path) -> pd.DataFrame:
    """Load *evaluation_CVAT_document_structure.json* and return a DataFrame."""
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if "evaluations" not in data:
        raise KeyError(
            "The supplied document-structure evaluation JSON does not contain "
            "the 'evaluations' field."
        )

    df = pd.DataFrame(data["evaluations"])
    df = df.rename(
        columns={
            "doc_id": "image_name",  # keep a consistent identifier column
            "edit_distance": "edit_distance_struct",  # be explicit
        }
    )
    df["doc_id"] = df["image_name"].map(_to_doc_id)

    return df


def load_user_table(csv_path: Path) -> pd.DataFrame:
    """Load *file_name_user_id.csv* (staff provenance) and return a DataFrame."""
    df = pd.read_csv(csv_path)

    # Drop pandas' default index column if present ("Unnamed: 0") or any first
    # column that is entirely numeric index-like.
    first_col: Final[str] = df.columns[0]
    if first_col.lower().startswith("unnamed") or df[first_col].is_monotonic_increasing:
        df = df.drop(columns=[first_col])

    # Normalise column names just in case.
    df = df.rename(
        columns={
            "image_name": "image_name",  # present in sample - keep identical
            "user": "annotator_id",
            "grading_scale": "self_confidence",
        }
    )

    df["doc_id"] = df["image_name"].map(_to_doc_id)

    return df


def merge_tables(
    layout_df: pd.DataFrame,
    doc_df: pd.DataFrame,
    user_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge layout, document structure, and optionally user dataframes.
    If user_df is None, skip merging user columns.
    """
    df = layout_df.merge(
        doc_df[["doc_id", "edit_distance_struct"]], on="doc_id", how="outer"
    )
    if user_df is not None:
        df = df.merge(
            user_df[["doc_id", "annotator_id", "self_confidence", "image_name"]],
            on="doc_id",
            how="left",
            suffixes=("", "_user"),
        )
        # the self_confidence column numeric
        df["self_confidence"] = pd.to_numeric(df["self_confidence"], errors="coerce")
        # confidence difference between the annotators
        df["diff_self_confidence"] = df.groupby("doc_id")["self_confidence"].transform(
            lambda x: x.max() - x.min()
        )

        # to check the self-confidence values
        avg_self_confidence = df["self_confidence"].mean()
        std_self_confidence = df["self_confidence"].std()
        quantiles = df["self_confidence"].quantile([0.01, 0.5, 0.99])

        print(f"Average self-confidence: {avg_self_confidence:.4f}")
        print(f"Standard deviation: {std_self_confidence:.4f}")
        print(
            f"Quantiles (1%, 50%, 99%): {quantiles[0.01]:.4f}, {quantiles[0.5]:.4f}, {quantiles[0.99]:.4f}"
        )

    # we can re-order the most relevant columns towards the front.
    preferred_order = [
        "doc_id",
        "image_name",  # from user table (includes extension)
        "avg_weighted_label_matched_iou_50",  # "value" in layout JSON file
        "segmentation_f1",  # may or may not exist, depending on evaluation config
        "edit_distance_struct",
        "map_val",
        "annotator_id",
        "self_confidence",
    ]
    ordered_cols = [c for c in preferred_order if c in df.columns] + [
        c for c in df.columns if c not in preferred_order
    ]
    df = df[ordered_cols]

    # filter columns to be present, and remove the rest
    filter_cols = [
        "doc_id",
        "segmentation_f1",
        "segmentation_f1_no_pictures",
        "avg_weighted_label_matched_iou_50",
        "edit_distance_struct",
        "annotator_id",
        "self_confidence",
        "diff_self_confidence",
        "element_count_diff",
        "element_count_set_A",
        "element_count_set_B",
        "table_count_set_A",
        "table_count_set_B",
        "picture_count_set_A",
        "picture_count_set_B",
        "table_count_diff",
        "picture_count_diff",
    ]
    df = df[[col for col in filter_cols if col in df.columns]]

    return df


def _write_as_excel_table(df: pd.DataFrame, path: Path) -> None:
    """
    Write *df* to *path* as an Excel **Table** and append five derived columns:

    layout_different   → 1 if segmentation_f1_no_pictures < 0.9   else ""
    structure_different→ 1 if edit_distance_struct      ≥ 10      else ""
    tables_different   → 1 if table_count_diff          ≥ 1       else ""
    pictures_different → 1 if picture_count_diff        ≥ 2       else ""
    need_review        → SUM of the four flags above                (0-4)
    """
    df = df.copy()

    # append empty placeholders – the formulas will fill them
    extra_cols = [
        "layout_different",
        "structure_different",
        "tables_different",
        "pictures_different",
        "need_review",
    ]
    for col in extra_cols:
        df[col] = ""

    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        # 1️⃣  dump the data frame
        df.to_excel(writer, sheet_name="Evaluation", index=False)

        # 2️⃣  grab handles we need
        wb = writer.book
        ws = writer.sheets["Evaluation"]
        n_rows, n_cols = df.shape  # includes derived columns

        # 3️⃣  build column specs with formulas for the derived columns
        column_settings = [{"header": h} for h in df.columns]

        # Tune these thresholds to your liking:
        col_formula = {
            "layout_different": '=IF(OR([@[segmentation_f1_no_pictures]]<0.9, [@[avg_weighted_label_matched_iou_50]]<0.9),1,"")',
            "structure_different": '=IF([@[edit_distance_struct]]>=10,1,"")',
            "tables_different": '=IF([@[table_count_diff]]>=1,1,"")',
            "pictures_different": '=IF([@[picture_count_diff]]>=2,1,"")',
            "need_review": "=SUM([@[layout_different]]:[@[pictures_different]])",
        }

        for spec in column_settings:
            hdr = spec["header"]
            if hdr in col_formula:
                spec["formula"] = col_formula[hdr]

        # 4️⃣  add the Excel Table (auto-filter, style, formulas copied down)
        table_range = xl_range(0, 0, n_rows, n_cols - 1)  # zero-based, incl. header
        ws.add_table(
            table_range,
            {
                "name": "EvaluationTbl",
                "header_row": True,
                "columns": column_settings,
                # 👇 style must be a string such as "TableStyleMedium2"
                "style": "TableStyleMedium2",
            },
        )

        # 5️⃣  quality-of-life tweaks
        ws.freeze_panes(1, 0)
        ws.autofit()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Combine CVAT layout & document-structure evaluation JSONs "
        "with the staff provenance CSV into a single spreadsheet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--layout_json",
        type=Path,
        default=Path("evaluation_results/evaluation_CVAT_layout.json"),
        help="Path to evaluation_CVAT_layout.json",
    )
    p.add_argument(
        "--docstruct_json",
        type=Path,
        default=Path("evaluation_results/evaluation_CVAT_document_structure.json"),
        help="Path to evaluation_CVAT_document_structure.json",
    )
    p.add_argument(
        "--user_csv",
        type=Path,
        default=Path("file_name_user_id.csv"),
        help="Path to file_name_user_id.csv",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("combined_evaluation.xlsx"),
        help=(
            "Output file; extension decides format (\n"
            "    - .xlsx  → Excel\n"
            "    - other  → CSV)"
        ),
    )
    return p


def combine_cvat_evaluations(
    layout_json: Path,
    docstruct_json: Path,
    user_csv: Optional[Path] = None,
    out: Path = Path("combined_evaluation.xlsx"),
) -> pd.DataFrame:
    """
    Combine CVAT layout & document-structure evaluation JSONs with the staff provenance CSV into a single spreadsheet.

    Args:
        layout_json: Path to evaluation_CVAT_layout.json
        docstruct_json: Path to evaluation_CVAT_document_structure.json
        user_csv: Optional path to file_name_user_id.csv
        out: Output file path; extension decides format (.xlsx for Excel, otherwise CSV)

    Returns:
        The combined DataFrame that was written to disk.
    """
    layout_df = load_layout(layout_json)
    doc_df = load_doc_structure(docstruct_json)
    user_df: Optional[pd.DataFrame] = None
    if user_csv is not None:
        user_df = load_user_table(user_csv)

    combined_df = merge_tables(layout_df, doc_df, user_df)

    if out.suffix.lower() == ".xlsx":
        # combined_df.to_excel(out, index=False)
        _write_as_excel_table(combined_df, out)
    else:
        combined_df.to_csv(out, index=False)

    print(f"\u2713 Combined evaluation written to {out.resolve()}")
    return combined_df


def main() -> None:
    args = build_arg_parser().parse_args()
    combine_cvat_evaluations(
        layout_json=args.layout_json,
        docstruct_json=args.docstruct_json,
        user_csv=args.user_csv,
        out=args.out,
    )


if __name__ == "__main__":
    main()
