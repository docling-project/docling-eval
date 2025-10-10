from __future__ import annotations

import argparse
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from docling_eval.campaign_tools.combine_cvat_evaluations import _write_as_excel_table
from docling_eval.campaign_tools.cvat_evaluation_pipeline import CVATEvaluationPipeline

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubmissionSubsetJob:
    """Container describing the artefacts needed to evaluate one submission subset."""

    submission_name: str
    subset_name: str
    tasks_root: Path
    base_cvat_root: Path
    output_dir: Path


def discover_jobs(
    deliveries_root: Path,
    datasets_root: Path,
    output_root: Path,
) -> List[SubmissionSubsetJob]:
    """Enumerate all submission subset combinations that can be evaluated."""
    jobs: List[SubmissionSubsetJob] = []

    if not deliveries_root.exists():
        raise FileNotFoundError(f"Deliveries root does not exist: {deliveries_root}")

    for submission_dir in sorted(deliveries_root.glob("submission-*")):
        if not submission_dir.is_dir():
            continue

        submission_name = submission_dir.name
        delivery_dir = submission_dir / "delivery"
        if not delivery_dir.is_dir():
            _LOGGER.warning("Skipping %s: missing delivery directory", submission_name)
            continue

        for subset_dir in sorted(delivery_dir.iterdir()):
            if not subset_dir.is_dir():
                continue

            subset_name = subset_dir.name
            tasks_root = subset_dir / "cvat_dataset_preannotated"
            if not tasks_root.exists():
                _LOGGER.warning(
                    "Skipping %s/%s: tasks root %s missing",
                    submission_name,
                    subset_name,
                    tasks_root,
                )
                continue

            base_cvat_root = datasets_root / subset_name / "cvat_dataset_preannotated"
            if not base_cvat_root.exists():
                _LOGGER.warning(
                    "Skipping %s/%s: base dataset root %s missing",
                    submission_name,
                    subset_name,
                    base_cvat_root,
                )
                continue

            output_dir = output_root / submission_name / subset_name
            jobs.append(
                SubmissionSubsetJob(
                    submission_name=submission_name,
                    subset_name=subset_name,
                    tasks_root=tasks_root,
                    base_cvat_root=base_cvat_root,
                    output_dir=output_dir,
                )
            )

    return jobs


def run_jobs(
    jobs: Sequence[SubmissionSubsetJob],
    *,
    modalities: Optional[Sequence[str]] = None,
    strict: bool = False,
    dry_run: bool = False,
    user_csv: Optional[Path] = None,
    force: bool = False,
    merge_only: bool = False,
    eval_only: bool = False,
    force_ocr: bool = False,
    ocr_scale: float = 1.0,
) -> None:
    """Execute the CVAT evaluation pipeline for each prepared job."""
    if not jobs:
        _LOGGER.info("No jobs discovered; nothing to do.")
        return

    jobs_by_submission: "OrderedDict[str, list[SubmissionSubsetJob]]" = OrderedDict()
    for job in jobs:
        jobs_by_submission.setdefault(job.submission_name, []).append(job)

    for submission_name, submission_jobs in jobs_by_submission.items():
        if not submission_jobs:
            continue

        submission_dir = submission_jobs[0].output_dir.parent
        submission_dir.mkdir(parents=True, exist_ok=True)
        submission_dfs: List[pd.DataFrame] = []
        failure = False

        _LOGGER.info("=== Processing submission %s ===", submission_name)

        for job in submission_jobs:
            _LOGGER.info(
                "Processing submission=%s subset=%s",
                job.submission_name,
                job.subset_name,
            )

            merged_dir = job.output_dir / "merged_xml"
            merged_gt = merged_dir / "combined_set_A.xml"
            merged_pred = merged_dir / "combined_set_B.xml"

            try:
                if merge_only:
                    if not force and merged_gt.exists() and merged_pred.exists():
                        _LOGGER.info(
                            "Skipping %s/%s: merged XML already present at %s",
                            job.submission_name,
                            job.subset_name,
                            merged_dir,
                        )
                        continue
                else:
                    if job.output_dir.exists() and not force and not eval_only:
                        _LOGGER.info(
                            "Skipping %s/%s: output directory already exists at %s (use --force to re-run)",
                            job.submission_name,
                            job.subset_name,
                            job.output_dir,
                        )
                        continue

                if dry_run:
                    if merge_only:
                        verb = "merge annotations for"
                    elif eval_only:
                        verb = "run evaluation for"
                    else:
                        verb = "evaluate"
                    _LOGGER.info(
                        "Dry-run: would %s %s/%s with base=%s tasks=%s output=%s",
                        verb,
                        job.submission_name,
                        job.subset_name,
                        job.base_cvat_root,
                        job.tasks_root,
                        job.output_dir,
                    )
                    continue

                pipeline = CVATEvaluationPipeline(
                    cvat_root=job.base_cvat_root,
                    output_dir=job.output_dir,
                    strict=strict,
                    tasks_root=job.tasks_root,
                    force_ocr=force_ocr,
                    ocr_scale=ocr_scale,
                )

                job.output_dir.mkdir(parents=True, exist_ok=True)

                if merge_only:
                    pipeline.merge_annotation_xmls(destination_dir=merged_dir)
                    continue

                if eval_only:
                    pipeline.run_table_evaluation()
                    subset_df = pipeline.run_evaluation(
                        modalities=list(modalities) if modalities else None,
                        user_csv=user_csv,
                        subset_label=job.subset_name,
                    )
                else:
                    pipeline.create_ground_truth_dataset()
                    pipeline.create_prediction_dataset()
                    pipeline.run_table_evaluation(reuse_existing=False)
                    subset_df = pipeline.run_evaluation(
                        modalities=list(modalities) if modalities else None,
                        user_csv=user_csv,
                        subset_label=job.subset_name,
                    )

                if subset_df is not None and not subset_df.empty:
                    if "subset" not in subset_df.columns:
                        subset_df = subset_df.copy()
                        subset_df.insert(0, "subset", job.subset_name)
                    submission_dfs.append(subset_df)

            except (
                Exception
            ) as exc:  # noqa: BLE001 - we want to capture all failures per subset
                failure = True
                _LOGGER.error(
                    "Submission %s subset %s failed: %s",
                    submission_name,
                    job.subset_name,
                    exc,
                )
                _LOGGER.debug("Subset failure details", exc_info=True)

        if submission_dfs:
            combined_df = pd.concat(submission_dfs, ignore_index=True)
            combined_out = submission_dir / "combined_evaluation.xlsx"
            status_label = "FAILED" if failure else "SUCCESS"
            _LOGGER.info(
                "Writing submission-level combined evaluation for %s (%s) to %s",
                submission_name,
                status_label,
                combined_out,
            )
            if "subset" not in combined_df.columns:
                combined_df.insert(0, "subset", submission_name)
            _write_as_excel_table(combined_df, combined_out)
        else:
            status_label = "FAILED" if failure else "SKIPPED"
            _LOGGER.warning(
                "Submission %s completed with status %s (no aggregated dataframe)",
                submission_name,
                status_label,
            )

        for job in submission_jobs:
            subset_combined = job.output_dir / "combined_evaluation.xlsx"
            if subset_combined.exists():
                try:
                    subset_combined.unlink()
                    _LOGGER.debug(
                        "Removed subset combined evaluation %s", subset_combined
                    )
                except Exception as cleanup_exc:  # noqa: BLE001
                    _LOGGER.warning(
                        "Failed to remove subset combined evaluation %s: %s",
                        subset_combined,
                        cleanup_exc,
                    )

            intermediate_dir = job.output_dir / "intermediate"
            if intermediate_dir.exists():
                try:
                    for extra in sorted(intermediate_dir.glob("**/*"), reverse=True):
                        if extra.is_file() or extra.is_symlink():
                            extra.unlink(missing_ok=True)
                        else:
                            extra.rmdir()
                    intermediate_dir.rmdir()
                    _LOGGER.debug("Removed intermediate directory %s", intermediate_dir)
                except Exception as cleanup_exc:  # noqa: BLE001
                    _LOGGER.warning(
                        "Failed to remove intermediate directory %s: %s",
                        intermediate_dir,
                        cleanup_exc,
                    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the CVAT evaluation pipeline across all submission deliveries."
        )
    )
    parser.add_argument(
        "--deliveries-root",
        type=Path,
        help="Root directory containing submission-*/delivery/* structures.",
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        help="Root directory containing the canonical base dataset subsets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Directory where evaluation artefacts will be written.",
    )
    parser.add_argument(
        "--modalities",
        nargs="*",
        choices=["layout", "document_structure", "key_value"],
        help="Optional list of evaluation modalities to run.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict mode to stop on conversion errors.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the actions without running the pipeline.",
    )
    parser.add_argument(
        "--user-csv",
        type=Path,
        help="Optional CSV to merge into combined evaluation output.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run evaluations even when the output directory already exists.",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only create combined set A/B XMLs for each submission subset.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip dataset creation and rerun only the evaluation stage.",
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Force OCR on PDF page images instead of using native text layer.",
    )
    parser.add_argument(
        "--ocr-scale",
        type=float,
        default=1.0,
        help="Scale for rendering PDFs for OCR (default: 1.0 = 72 DPI). Higher values may improve OCR accuracy.",
    )

    return parser.parse_args(argv)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    configure_logging()
    args = parse_args(argv)

    if args.merge_only and args.eval_only:
        _LOGGER.error("Cannot use --merge-only and --eval-only at the same time.")
        return

    jobs = discover_jobs(args.deliveries_root, args.datasets_root, args.output_root)
    run_jobs(
        jobs,
        modalities=args.modalities,
        strict=args.strict,
        dry_run=args.dry_run,
        user_csv=args.user_csv,
        force=args.force,
        merge_only=args.merge_only,
        eval_only=args.eval_only,
        force_ocr=args.force_ocr,
        ocr_scale=args.ocr_scale,
    )


if __name__ == "__main__":
    main()
