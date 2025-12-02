"""combine_cvat_evaluations.py

We will use this script to combine the output metrics produced by our CVAT evaluation tooling into a single
spread-sheet (CSV or XLSX).

Inputs
------
* evaluation_CVAT_layout.json  - layout-level metrics (`evaluations_per_image`)
* evaluation_CVAT_document_structure.json - document-structure metrics
  (`evaluations`)
* evaluation_CVAT_key_value.json - key-value extraction metrics (`evaluations`)
* file_name_user_id.csv - staff self-confidence / provenance table

The script matches the four sources by a **document id** that is derived from
an image / doc name **without the file-extension** and we produde single table.

Usage
-----
    python combine_cvat_evaluations.py \
        --layout_json evaluation_results/evaluation_CVAT_layout.json \
        --docstruct_json evaluation_results/evaluation_CVAT_document_structure.json \
        --keyvalue_json evaluation_results/evaluation_CVAT_key_value.json \
        --user_csv file_name_user_id.csv \
        --out combined_evaluation.xlsx

*If ``--out`` ends with ``.csv`` the script will write a CSV; otherwise an
Excel workbook is produced.*
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Final, Optional

import pandas as pd
from xlsxwriter.utility import xl_range

from docling_eval.datamodels.cvat_page_mapping import CvatPageMapping


def _to_doc_id(path_like: str) -> str:
    basename = os.path.basename(path_like)
    stem, _ = os.path.splitext(basename)
    # remove -page-1 suffix
    stem = stem.replace("-page-1", "")
    return stem


def _enrich_with_mapping(
    df: pd.DataFrame,
    mapping: Optional[CvatPageMapping],
    *,
    image_column: str = "image_name",
) -> pd.DataFrame:
    if image_column not in df.columns:
        return df

    if {"doc_name", "page_no", image_column}.issubset(df.columns):
        return df

    enriched = df.copy()
    original_values = enriched[image_column].astype(str).tolist()

    pattern = re.compile(
        r"^(?P<doc>.+?)[\s_-]*page[\s_-]*(?P<page>\d+)$", re.IGNORECASE
    )
    hash_pattern = re.compile(
        r"^doc_(?P<hash>[0-9a-f]{64})_page_(?P<page>\d{6})(?:\.png)?$",
        re.IGNORECASE,
    )

    doc_names: list[str] = []
    page_nos: list[Optional[int]] = []
    resolved_image_names: list[str] = []

    for raw_value in original_values:
        candidate_doc: Optional[str] = None
        candidate_page: Optional[int] = None
        candidate_hash: Optional[str] = None

        match = pattern.match(raw_value)
        if match:
            candidate_doc = match.group("doc").strip()
            candidate_page = int(match.group("page"))

        hash_match = hash_pattern.match(raw_value)
        if hash_match:
            candidate_hash = hash_match.group("hash")
            candidate_page = int(hash_match.group("page"))

        ref = None
        if mapping is not None:
            ref = mapping.page_for_image(raw_value)
            if (
                ref is None
                and candidate_hash is not None
                and candidate_page is not None
            ):
                for page_ref in mapping.pages_for_doc_hash(candidate_hash):
                    if page_ref.page_no == candidate_page:
                        ref = page_ref
                        break

            if ref is None and candidate_doc is not None and candidate_page is not None:
                for page_ref in mapping.pages_for_doc_name(candidate_doc):
                    if page_ref.page_no == candidate_page:
                        ref = page_ref
                        break

        if ref is not None:
            doc_names.append(ref.doc_name)
            page_nos.append(ref.page_no)
            resolved_image_names.append(ref.image_name)
        else:
            fallback_doc = candidate_doc or _to_doc_id(raw_value)
            doc_names.append(fallback_doc)
            page_nos.append(candidate_page)
            resolved_image_names.append(raw_value)

    enriched["doc_name"] = doc_names
    enriched["page_no"] = page_nos
    enriched[image_column] = resolved_image_names
    return enriched


def load_tables(
    json_path: Path, mapping: Optional[CvatPageMapping] = None
) -> pd.DataFrame:
    """Load evaluation_CVAT_tables.json and return a DataFrame."""
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if "evaluations" not in data:
        raise KeyError(
            "The supplied tables evaluation JSON does not contain the 'evaluations' field."
        )
    df = pd.DataFrame(data["evaluations"])
    if "image_name" not in df.columns and "doc_id" in df.columns:
        df = df.rename(columns={"doc_id": "image_name"})
    df["image_name"] = df["image_name"].astype(str)
    return _enrich_with_mapping(df, mapping)


def load_layout(
    json_path: Path, mapping: Optional[CvatPageMapping] = None
) -> pd.DataFrame:
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

    if "name" in df.columns:
        if "image_name" in df.columns:
            df = df.drop(columns=["name"])
        else:
            df = df.rename(columns={"name": "image_name"})

    # Build merge key
    df = _enrich_with_mapping(df, mapping)
    return df


def load_doc_structure(
    json_path: Path, mapping: Optional[CvatPageMapping] = None
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
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
            "doc_id": "doc_name",
            "edit_distance": "edit_distance_struct",  # be explicit
        }
    )
    per_page_df: Optional[pd.DataFrame] = None
    if "evaluations_per_page" in data:
        per_page_data = data["evaluations_per_page"]
        if isinstance(per_page_data, list) and per_page_data:
            per_page_df = pd.DataFrame(per_page_data)
            if (
                "image_name" not in per_page_df.columns
                and "doc_id" in per_page_df.columns
            ):
                per_page_df = per_page_df.rename(columns={"doc_id": "doc_name"})
            per_page_df = per_page_df.rename(
                columns={"edit_distance": "edit_distance_struct_page"}
            )
            per_page_df = _enrich_with_mapping(per_page_df, mapping)

    return df, per_page_df


def load_key_value(
    json_path: Path, mapping: Optional[CvatPageMapping] = None
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Load *evaluation_CVAT_key_value.json* and return a DataFrame."""
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if "evaluations" not in data:
        raise KeyError(
            "The supplied key-value evaluation JSON does not contain the "
            "'evaluations' field."
        )

    # Convert the evaluations list to a DataFrame
    # The evaluations is a list of KeyValueEvaluation objects
    evaluations_list = []
    for eval_data in data["evaluations"]:
        evaluations_list.append(eval_data)

    df = pd.DataFrame(evaluations_list)
    df = df.rename(columns={"doc_id": "doc_name"})

    per_page_df: Optional[pd.DataFrame] = None
    if "evaluations_per_page" in data:
        per_page_data = data["evaluations_per_page"]
        if isinstance(per_page_data, list) and per_page_data:
            per_page_df = pd.DataFrame(per_page_data)
            if (
                "image_name" not in per_page_df.columns
                and "doc_id" in per_page_df.columns
            ):
                per_page_df = per_page_df.rename(columns={"doc_id": "doc_name"})
            rename_map = {
                col: f"{col}_page"
                for col in per_page_df.columns
                if col
                not in {
                    "doc_name",
                    "image_name",
                    "page_no",
                }
            }
            per_page_df = per_page_df.rename(columns=rename_map)
            per_page_df = _enrich_with_mapping(per_page_df, mapping)

    return df, per_page_df


def load_user_table(
    csv_path: Path, mapping: Optional[CvatPageMapping] = None
) -> pd.DataFrame:
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

    df = _enrich_with_mapping(df, mapping)
    return df


def merge_tables(
    layout_df: pd.DataFrame,
    doc_df: pd.DataFrame,
    doc_page_df: Optional[pd.DataFrame] = None,
    keyvalue_df: Optional[pd.DataFrame] = None,
    keyvalue_page_df: Optional[pd.DataFrame] = None,
    user_df: Optional[pd.DataFrame] = None,
    tables_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    df = layout_df.copy()

    if doc_df is not None and not doc_df.empty:
        doc_metrics = doc_df.copy()
        doc_metrics = doc_metrics.drop(
            columns=[
                col
                for col in [
                    "image_name",
                    "page_no",
                ]
                if col in doc_metrics.columns
            ],
            errors="ignore",
        )
        df = df.merge(doc_metrics, on="doc_name", how="left")

    if doc_page_df is not None and not doc_page_df.empty:
        doc_page_metrics = doc_page_df.drop(
            columns=[
                col
                for col in [
                    "doc_name",
                    "page_no",
                ]
                if col in doc_page_df.columns
            ],
            errors="ignore",
        )
        doc_page_metrics = doc_page_metrics.drop_duplicates(subset=["image_name"])
        df = df.merge(
            doc_page_metrics,
            on="image_name",
            how="left",
        )

    if keyvalue_df is not None and not keyvalue_df.empty:
        kv_doc_metrics = keyvalue_df.copy()
        kv_doc_metrics = kv_doc_metrics.drop(
            columns=[
                col
                for col in [
                    "image_name",
                    "page_no",
                ]
                if col in kv_doc_metrics.columns
            ],
            errors="ignore",
        )
        df = df.merge(kv_doc_metrics, on="doc_name", how="left")

    if keyvalue_page_df is not None and not keyvalue_page_df.empty:
        kv_page_metrics = keyvalue_page_df.drop(
            columns=[
                col
                for col in [
                    "doc_name",
                    "page_no",
                ]
                if col in keyvalue_page_df.columns
            ],
            errors="ignore",
        )
        kv_page_metrics = kv_page_metrics.drop_duplicates(subset=["image_name"])
        df = df.merge(
            kv_page_metrics,
            on="image_name",
            how="left",
        )

    if tables_df is not None and not tables_df.empty:
        table_metrics = tables_df.drop(
            columns=[
                col
                for col in [
                    "doc_name",
                    "page_no",
                ]
                if col in tables_df.columns
            ],
            errors="ignore",
        )
        df = df.merge(
            table_metrics.drop_duplicates(subset=["image_name"]),
            on="image_name",
            how="left",
        )

    if user_df is not None and not user_df.empty:
        df = df.merge(
            user_df.drop_duplicates(subset=["image_name"])[
                ["image_name", "annotator_id", "self_confidence"]
            ],
            on="image_name",
            how="left",
            suffixes=("", "_user"),
        )
        df["self_confidence"] = pd.to_numeric(df["self_confidence"], errors="coerce")
        df["diff_self_confidence"] = df.groupby("doc_name")[
            "self_confidence"
        ].transform(lambda x: x.max() - x.min())

    preferred_order = [
        "doc_name",
        "image_name",
        "page_no",
        "avg_weighted_label_matched_iou_50",
        "segmentation_f1",
        "edit_distance_struct",
        "edit_distance_struct_page",
        # table metrics (consolidated)
        "row_count_abs_diff_sum",
        "col_count_abs_diff_sum",
        "merge_count_abs_diff_sum",
        "sem_body_f1",
        "sem_row_section_f1",
        "sem_row_header_f1",
        "sem_col_header_f1",
        "tables_unmatched",
        "table_pairs",
        "orphan_table_annotation_A",
        "orphan_table_annotation_B",
        # key-values & misc
        "entity_f1",
        "relation_f1",
        "entity_f1_page",
        "relation_f1_page",
        "map_val",
        "annotator_id",
        "self_confidence",
    ]
    ordered = [c for c in preferred_order if c in df.columns] + [
        c for c in df.columns if c not in preferred_order
    ]
    df = df[ordered]

    filter_cols = [
        "doc_name",
        "image_name",
        "page_no",
        "segmentation_f1",
        "segmentation_f1_no_pictures",
        "avg_weighted_label_matched_iou_50",
        "edit_distance_struct",
        "edit_distance_struct_page",
        # consolidated table metrics
        "row_count_abs_diff_sum",
        "col_count_abs_diff_sum",
        "merge_count_abs_diff_sum",
        "sem_body_f1",
        "sem_row_section_f1",
        "sem_row_header_f1",
        "sem_col_header_f1",
        "tables_unmatched",
        "table_pairs",
        "orphan_table_annotation_A",
        "orphan_table_annotation_B",
        # key-values
        "entity_f1",
        "relation_f1",
        "entity_f1_page",
        "relation_f1_page",
        "num_entity_diff",
        "num_entity_diff_normalized",
        "num_link_diff",
        "num_link_diff_normalized",
        # existing counts
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
    df = df[[c for c in filter_cols if c in df.columns]]
    return df


def _write_as_excel_table(df: pd.DataFrame, path: Path) -> None:
    """
    Write *df* to *path* as an Excel **Table** and append derived columns:

    layout_different            → 1 if segmentation or label-IoU is low
    structure_different         → 1 if edit distance is high
    tables_different            → 1 if table count differs
    pictures_different          → 1 if picture count differs
    key_values_different        → 1 if entity/relation F1 is low (and non-zero)
    table_struct_different      → 1 if any table structure count diffs sum to > 0
    table_semantic_different    → 1 if table_pairs > 0 and any semantic F1 < 0.9
    need_review                 → SUM of selected flags (kept as in your current sheet)
    """
    from xlsxwriter.utility import xl_range

    df = df.copy()

    extra_cols = [
        "layout_different",
        "structure_different",
        "tables_different",
        "pictures_different",
        "key_values_different",
        "table_struct_different",
        "table_semantic_different",
        "need_review",
        "need_task2_review",
    ]
    for col in extra_cols:
        if col not in df.columns:
            df[col] = ""

    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Evaluation", index=False)

        ws = writer.sheets["Evaluation"]
        n_rows, n_cols = df.shape

        column_settings = [{"header": h} for h in df.columns]

        col_formula = {
            "layout_different": '=IF(OR([@[segmentation_f1_no_pictures]]<0.9,[@[avg_weighted_label_matched_iou_50]]<0.9),1,"")',
            "structure_different": '=IF([@[edit_distance_struct]]>=10,1,"")',
            "tables_different": '=IF([@[table_count_diff]]>=1,1,"")',
            "pictures_different": '=IF([@[picture_count_diff]]>=2,1,"")',
            "key_values_different": '=IF(OR([@[num_entity_diff]]>2,[@[num_link_diff]]>2,AND([@[entity_f1]]<>0,[@[entity_f1]]<0.95),AND([@[relation_f1]]<>0,[@[relation_f1]]<0.95)),1,"")',
            # 1) any structure count diffs sum to > 0
            "table_struct_different": (
                "=IF(SUM([@[row_count_abs_diff_sum]],[@[col_count_abs_diff_sum]],"
                '[@[merge_count_abs_diff_sum]])>0,1,"")'
            ),
            # 2) gate semantics on table_pairs > 0, then flag if any F1 < 0.9
            "table_semantic_different": (
                "=IF(AND([@[table_pairs]]>0,OR([@[sem_col_header_f1]]<0.9,"
                '[@[sem_row_header_f1]]<0.9,[@[sem_row_section_f1]]<0.9,[@[sem_body_f1]]<0.9)),1,"")'
            ),
            "need_review": "=SUM([@[layout_different]]:[@[pictures_different]])",
            "need_task2_review": "=SUM([@[key_values_different]]:[@[table_semantic_different]],[@[tables_unmatched]])",
        }

        for spec in column_settings:
            hdr = spec["header"]
            if hdr in col_formula:
                spec["formula"] = col_formula[hdr]

        table_range = xl_range(0, 0, n_rows, n_cols - 1)
        ws.add_table(
            table_range,
            {
                "name": "EvaluationTbl",
                "header_row": True,
                "columns": column_settings,
                "style": "TableStyleMedium2",
            },
        )

        ws.freeze_panes(1, 0)
        ws.autofit()  # type: ignore[attr-defined]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Combine CVAT layout, document-structure, and key-value evaluation JSONs "
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
        "--keyvalue_json",
        type=Path,
        default=Path("evaluation_results/evaluation_CVAT_key_value.json"),
        help="Path to evaluation_CVAT_key_value.json",
    )
    p.add_argument(
        "--tables_json",
        type=Path,
        default=None,
        help="Path to evaluation_CVAT_tables.json (optional)",
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
    p.add_argument(
        "--cvat_overview_path",
        type=Path,
        default=None,
        help="Path to cvat_overview.json for image/page mapping (optional)",
    )
    return p


def combine_cvat_evaluations(
    layout_json: Path,
    docstruct_json: Path,
    keyvalue_json: Optional[Path] = None,
    user_csv: Optional[Path] = None,
    tables_json: Optional[Path] = None,
    out: Path = Path("combined_evaluation.xlsx"),
    cvat_overview_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Combine CVAT layout, document-structure, and key-value evaluation JSONs with the staff provenance CSV into a single spreadsheet.

    Args:
        layout_json: Path to evaluation_CVAT_layout.json
        docstruct_json: Path to evaluation_CVAT_document_structure.json
        keyvalue_json: Optional path to evaluation_CVAT_key_value.json
        user_csv: Optional path to file_name_user_id.csv
        out: Output file path; extension decides format (.xlsx for Excel, otherwise CSV)

    Returns:
        The combined DataFrame that was written to disk.
    """
    page_mapping: Optional[CvatPageMapping] = None
    if cvat_overview_path is not None and cvat_overview_path.exists():
        page_mapping = CvatPageMapping.from_overview_path(cvat_overview_path)

    layout_df = load_layout(layout_json, page_mapping)
    doc_df, doc_page_df = load_doc_structure(docstruct_json, page_mapping)
    tables_df = (
        load_tables(tables_json, page_mapping) if tables_json is not None else None
    )

    keyvalue_df: Optional[pd.DataFrame] = None
    keyvalue_page_df: Optional[pd.DataFrame] = None
    if keyvalue_json is not None:
        keyvalue_df, keyvalue_page_df = load_key_value(keyvalue_json, page_mapping)
    user_df: Optional[pd.DataFrame] = None
    if user_csv is not None:
        user_df = load_user_table(user_csv, page_mapping)

    combined_df = merge_tables(
        layout_df,
        doc_df,
        doc_page_df,
        keyvalue_df,
        keyvalue_page_df,
        user_df,
        tables_df,
    )

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
        keyvalue_json=args.keyvalue_json,
        user_csv=args.user_csv,
        tables_json=args.tables_json,
        out=args.out,
        cvat_overview_path=args.cvat_overview_path,
    )


if __name__ == "__main__":
    main()
