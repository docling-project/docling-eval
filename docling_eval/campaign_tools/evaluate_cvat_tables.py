from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import typer
from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.labels import DocItemLabel
from pydantic import BaseModel, Field

from docling_eval.cvat_tools.document import DocumentStructure
from docling_eval.cvat_tools.models import CVATElement, TableStructLabel
from docling_eval.cvat_tools.parser import ParsedCVATFile, parse_cvat_file

DEFAULT_TABLE_PAIR_IOU: float = 0.20
DEFAULT_CONTAINMENT_THRESH: float = 0.50
DEFAULT_SEM_MATCH_IOU: float = 0.30


def iou(a: BoundingBox, b: BoundingBox) -> float:
    return a.intersection_over_union(b)


def inter_area(a: BoundingBox, b: BoundingBox) -> float:
    return a.intersection_area_with(b)


def area(bb: BoundingBox) -> float:
    return bb.area()


def inside_with_tolerance(
    child: BoundingBox, parent: BoundingBox, thresh: float
) -> bool:
    a = area(child)
    if a <= 0.0:
        return False
    return (inter_area(child, parent) / a) >= thresh


class SemClass(str, Enum):
    COL_HEADER = "col_header"
    ROW_HEADER = "row_header"
    ROW_SECTION = "row_section"
    BODY = "body"


SEM_TO_TABLE_LABEL: dict[SemClass, TableStructLabel] = {
    SemClass.COL_HEADER: TableStructLabel.COL_HEADER,
    SemClass.ROW_HEADER: TableStructLabel.ROW_HEADER,
    SemClass.ROW_SECTION: TableStructLabel.ROW_SECTION,
    SemClass.BODY: TableStructLabel.BODY,
}


@dataclass
class TableStruct:
    table_el: CVATElement
    rows: list[CVATElement]
    cols: list[CVATElement]
    merges: list[CVATElement]
    sem: dict[SemClass, list[CVATElement]]


class TablePairMetrics(BaseModel):
    row_count_diff: int
    col_count_diff: int
    merge_count_diff: int
    sem_f1: dict[SemClass, float]


class ImageTablesEvaluation(BaseModel):
    # identifier used for joining in the combiner
    doc_id: str

    # kept metrics
    row_count_abs_diff_sum: int = 0
    col_count_abs_diff_sum: int = 0
    merge_count_abs_diff_sum: int = 0

    sem_body_f1: float = 0.0
    sem_row_section_f1: float = 0.0
    sem_row_header_f1: float = 0.0
    sem_col_header_f1: float = 0.0

    table_pairs: int = 0
    tables_unmatched: int = 0

    orphan_table_annotation_A: int = 0
    orphan_table_annotation_B: int = 0


class TablesEvaluationRun(BaseModel):
    evaluations: list[ImageTablesEvaluation] = Field(default_factory=list)


def list_images_in_xml(xml_path: Path) -> list[str]:
    import xml.etree.ElementTree as ET

    root = ET.parse(xml_path).getroot()
    result: list[str] = []
    for img in root.findall(".//image"):
        name = img.get("name")
        if name:
            result.append(name)
    return result


def _elements_by_label(
    elements: Sequence[CVATElement], label: object
) -> list[CVATElement]:
    return [e for e in elements if e.label == label]


def _collect_tables(
    doc: DocumentStructure, contain_thresh: float
) -> tuple[list[TableStruct], list[CVATElement]]:
    tables = _elements_by_label(doc.elements, DocItemLabel.TABLE)
    result: list[TableStruct] = []

    pool_rows = _elements_by_label(doc.elements, TableStructLabel.TABLE_ROW)
    pool_cols = _elements_by_label(doc.elements, TableStructLabel.TABLE_COLUMN)
    pool_merges = _elements_by_label(doc.elements, TableStructLabel.TABLE_MERGED_CELL)
    pool_sem: dict[SemClass, list[CVATElement]] = {
        sc: _elements_by_label(doc.elements, lab)
        for sc, lab in SEM_TO_TABLE_LABEL.items()
    }

    for t in tables:
        tb = t.bbox
        rows = [
            e for e in pool_rows if inside_with_tolerance(e.bbox, tb, contain_thresh)
        ]
        cols = [
            e for e in pool_cols if inside_with_tolerance(e.bbox, tb, contain_thresh)
        ]
        merges = [
            e for e in pool_merges if inside_with_tolerance(e.bbox, tb, contain_thresh)
        ]
        sem = {
            sc: [
                e
                for e in pool_sem[sc]
                if inside_with_tolerance(e.bbox, tb, contain_thresh)
            ]
            for sc in SemClass
        }
        result.append(
            TableStruct(table_el=t, rows=rows, cols=cols, merges=merges, sem=sem)
        )

    all_tables_bb = [t.table_el.bbox for t in result]

    def not_in_any_table(el: CVATElement) -> bool:
        return not any(
            inside_with_tolerance(el.bbox, tb, contain_thresh) for tb in all_tables_bb
        )

    orphans = [
        e
        for e in pool_rows + pool_cols + pool_merges + sum(pool_sem.values(), [])
        if not_in_any_table(e)
    ]
    return result, orphans


def _pair_tables(
    a: list[TableStruct],
    b: list[TableStruct],
    iou_thresh: float,
) -> tuple[list[tuple[TableStruct, TableStruct]], list[TableStruct], list[TableStruct]]:
    if not a or not b:
        return [], a[:], b[:]

    candidates: list[tuple[int, int, float]] = []
    for i, ta in enumerate(a):
        for j, tb in enumerate(b):
            candidates.append((i, j, iou(ta.table_el.bbox, tb.table_el.bbox)))
    candidates.sort(key=lambda t: t[2], reverse=True)

    used_a: set[int] = set()
    used_b: set[int] = set()
    matched: list[tuple[TableStruct, TableStruct]] = []
    for i, j, s in candidates:
        if s < iou_thresh:
            break
        if i in used_a or j in used_b:
            continue
        matched.append((a[i], b[j]))
        used_a.add(i)
        used_b.add(j)

    unmatched_a = [a[i] for i in range(len(a)) if i not in used_a]
    unmatched_b = [b[j] for j in range(len(b)) if j not in used_b]
    return matched, unmatched_a, unmatched_b


def _greedy_intersection_sum(
    a: Sequence[BoundingBox], b: Sequence[BoundingBox], iou_thresh: float
) -> float:
    if not a or not b:
        return 0.0
    pairs: list[tuple[int, int, float]] = []
    for i, ba in enumerate(a):
        for j, bb in enumerate(b):
            v = iou(ba, bb)
            if v >= iou_thresh:
                pairs.append((i, j, v))
    pairs.sort(key=lambda t: t[2], reverse=True)
    used_i: set[int] = set()
    used_j: set[int] = set()
    inter_sum = 0.0
    for i_idx, j_idx, _ in pairs:
        if i_idx in used_i or j_idx in used_j:
            continue
        used_i.add(i_idx)
        used_j.add(j_idx)
        inter_sum += inter_area(a[i_idx], b[j_idx])
    return inter_sum


def _sem_f1(
    a_boxes: list[BoundingBox], b_boxes: list[BoundingBox], iou_thresh: float
) -> float:
    if not a_boxes and not b_boxes:
        return 1.0
    if not a_boxes or not b_boxes:
        return 0.0
    inter = _greedy_intersection_sum(a_boxes, b_boxes, iou_thresh=iou_thresh)
    a_area = sum(area(bb) for bb in a_boxes)
    b_area = sum(area(bb) for bb in b_boxes)
    if a_area <= 0.0 or b_area <= 0.0:
        return 0.0
    p = inter / a_area
    r = inter / b_area
    return 0.0 if (p + r) == 0.0 else (2.0 * p * r) / (p + r)


def _pair_metrics(ta: TableStruct, tb: TableStruct, sem_iou: float) -> TablePairMetrics:
    sem_f1: dict[SemClass, float] = {}
    for sc in SemClass:
        a_boxes = [e.bbox for e in ta.sem.get(sc, [])]
        b_boxes = [e.bbox for e in tb.sem.get(sc, [])]
        sem_f1[sc] = _sem_f1(a_boxes, b_boxes, iou_thresh=sem_iou)
    tpm = TablePairMetrics(
        row_count_diff=abs(len(ta.rows) - len(tb.rows)),
        col_count_diff=abs(len(ta.cols) - len(tb.cols)),
        merge_count_diff=abs(len(ta.merges) - len(tb.merges)),
        sem_f1=sem_f1,
    )
    # print(f"Rows: A: {len(ta.rows)}, B: {len(tb.rows)}")
    # print(f"Cols: A: {len(ta.cols)}, B: {len(tb.cols)}")
    # print(f"Merges: A: {len(ta.merges)}, B: {len(tb.merges)}")

    return tpm


def _doc_id_from_image_name(image_name: str) -> str:
    return Path(image_name).stem


def _orphans_count(orphans: list[CVATElement]) -> dict[str, int]:
    out: dict[str, int] = {
        "rows": 0,
        "cols": 0,
        "merges": 0,
        "sem_col_header": 0,
        "sem_row_header": 0,
        "sem_row_section": 0,
        "sem_body": 0,
    }
    for el in orphans:
        if el.label == TableStructLabel.TABLE_ROW:
            out["rows"] += 1
        elif el.label == TableStructLabel.TABLE_COLUMN:
            out["cols"] += 1
        elif el.label == TableStructLabel.TABLE_MERGED_CELL:
            out["merges"] += 1
        elif el.label == TableStructLabel.COL_HEADER:
            out["sem_col_header"] += 1
        elif el.label == TableStructLabel.ROW_HEADER:
            out["sem_row_header"] += 1
        elif el.label == TableStructLabel.ROW_SECTION:
            out["sem_row_section"] += 1
        elif el.label == TableStructLabel.BODY:
            out["sem_body"] += 1
    return out


def evaluate_image(
    set_a_xml: Path,
    set_b_xml: Path,
    image_name: str,
    containment_thresh: float,
    table_pair_iou: float,
    sem_match_iou: float,
    *,
    parsed_set_a: Optional[ParsedCVATFile] = None,
    parsed_set_b: Optional[ParsedCVATFile] = None,
) -> Optional[ImageTablesEvaluation]:
    try:
        parsed_a = (
            parsed_set_a if parsed_set_a is not None else parse_cvat_file(set_a_xml)
        )
        parsed_b = (
            parsed_set_b if parsed_set_b is not None else parse_cvat_file(set_b_xml)
        )
        doc_a = DocumentStructure.from_parsed_image(parsed_a.get_image(image_name))
        doc_b = DocumentStructure.from_parsed_image(parsed_b.get_image(image_name))
    except Exception:
        return None

    tables_a, orphans_a = _collect_tables(doc_a, containment_thresh)
    tables_b, orphans_b = _collect_tables(doc_b, containment_thresh)

    matched, ua, ub = _pair_tables(tables_a, tables_b, iou_thresh=table_pair_iou)
    pair_metrics = [
        _pair_metrics(ta, tb, sem_iou=sem_match_iou) for (ta, tb) in matched
    ]

    # Sums of absolute differences across matched table pairs
    row_diff_sum = int(sum(pm.row_count_diff for pm in pair_metrics))
    col_diff_sum = int(sum(pm.col_count_diff for pm in pair_metrics))
    merge_diff_sum = int(sum(pm.merge_count_diff for pm in pair_metrics))

    # Average semantic F1 over matched pairs (0 when no pairs)
    def mean_f1(key: SemClass) -> float:
        seq = [pm.sem_f1[key] for pm in pair_metrics]
        return float(sum(seq)) / float(len(seq)) if seq else 0.0

    return ImageTablesEvaluation(
        doc_id=_doc_id_from_image_name(image_name),
        row_count_abs_diff_sum=row_diff_sum,
        col_count_abs_diff_sum=col_diff_sum,
        merge_count_abs_diff_sum=merge_diff_sum,
        sem_body_f1=mean_f1(SemClass.BODY),
        sem_row_section_f1=mean_f1(SemClass.ROW_SECTION),
        sem_row_header_f1=mean_f1(SemClass.ROW_HEADER),
        sem_col_header_f1=mean_f1(SemClass.COL_HEADER),
        table_pairs=len(matched),
        tables_unmatched=(len(ua) + len(ub)),
        orphan_table_annotation_A=len(orphans_a),
        orphan_table_annotation_B=len(orphans_b),
    )


app = typer.Typer(help="Compare table structure/semantics between two CVAT XMLs.")


def evaluate_tables(
    set_a: Path,
    set_b: Path,
    containment_thresh: float = DEFAULT_CONTAINMENT_THRESH,
    table_pair_iou: float = DEFAULT_TABLE_PAIR_IOU,
    sem_match_iou: float = DEFAULT_SEM_MATCH_IOU,
) -> "TablesEvaluationRun":
    """
    Library entrypoint: evaluate tables across images present in both CVAT XMLs.
    Returns the full evaluation model (no file I/O, no Typer types).
    """
    imgs = sorted(set(list_images_in_xml(set_a)) & set(list_images_in_xml(set_b)))
    parsed_set_a = parse_cvat_file(set_a)
    parsed_set_b = parse_cvat_file(set_b)
    evals: list[ImageTablesEvaluation] = []
    for name in imgs:
        res = evaluate_image(
            set_a_xml=set_a,
            set_b_xml=set_b,
            image_name=name,
            containment_thresh=containment_thresh,
            table_pair_iou=table_pair_iou,
            sem_match_iou=sem_match_iou,
            parsed_set_a=parsed_set_a,
            parsed_set_b=parsed_set_b,
        )
        if res is not None:
            evals.append(res)
    return TablesEvaluationRun(evaluations=evals)


@app.command()
def run(
    set_a: Path = typer.Option(
        ..., exists=True, readable=True, help="CVAT XML (Set A)"
    ),
    set_b: Path = typer.Option(
        ..., exists=True, readable=True, help="CVAT XML (Set B)"
    ),
    out: Path = typer.Option(
        Path("evaluation_results/evaluation_CVAT_tables.json"), help="Output JSON"
    ),
    containment_thresh: float = typer.Option(
        DEFAULT_CONTAINMENT_THRESH, min=0.0, max=1.0
    ),
    table_pair_iou: float = typer.Option(DEFAULT_TABLE_PAIR_IOU, min=0.0, max=1.0),
    sem_match_iou: float = typer.Option(DEFAULT_SEM_MATCH_IOU, min=0.0, max=1.0),
) -> None:
    result = evaluate_tables(
        set_a=set_a,
        set_b=set_b,
        containment_thresh=containment_thresh,
        table_pair_iou=table_pair_iou,
        sem_match_iou=sem_match_iou,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = result.model_dump(mode="json")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out.resolve()} ({len(result.evaluations)} images)")


if __name__ == "__main__":
    app()
