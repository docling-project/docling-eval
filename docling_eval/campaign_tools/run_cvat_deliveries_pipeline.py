from __future__ import annotations

import argparse
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from docling_eval.campaign_tools.combine_cvat_evaluations import _write_as_excel_table
from docling_eval.campaign_tools.cvat_evaluation_pipeline import (
    GROUND_TRUTH_PATTERN,
    PREDICTION_PATTERN,
    CVATEvaluationPipeline,
)
from docling_eval.cvat_tools.models import CVATValidationRunReport

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionPlan:
    """Encapsulates what stages should run and with what options."""

    run_merge: bool
    run_dataset_creation: bool
    run_evaluation: bool
    run_validation_reports: bool
    force_rerun: bool
    modalities: List[str]

    @classmethod
    def from_args(
        cls,
        merge_only: bool,
        eval_only: bool,
        validation_only: bool,
        force: bool,
        modalities: Optional[Sequence[str]],
    ) -> ExecutionPlan:
        """Create execution plan from CLI arguments.

        Args:
            merge_only: Only merge XML annotations
            eval_only: Only run evaluation (skip dataset creation)
            force: Force rerun even if outputs exist
            modalities: Evaluation modalities to run

        Returns:
            ExecutionPlan describing what should be executed

        Raises:
            ValueError: If incompatible flags are combined
        """
        exclusive_flags = [
            flag for flag in (merge_only, eval_only, validation_only) if flag
        ]
        if len(exclusive_flags) > 1:
            raise ValueError(
                "Cannot combine --merge-only, --eval-only, or --validation-only"
            )

        if merge_only:
            run_merge = True
            run_dataset_creation = False
            run_evaluation = False
            run_validation_reports = False
        elif eval_only:
            run_merge = False
            run_dataset_creation = False
            run_evaluation = True
            run_validation_reports = False
        elif validation_only:
            run_merge = False
            run_dataset_creation = False
            run_evaluation = False
            run_validation_reports = True
        else:
            run_merge = True
            run_dataset_creation = True
            run_evaluation = True
            run_validation_reports = False

        return cls(
            run_merge=run_merge,
            run_dataset_creation=run_dataset_creation,
            run_evaluation=run_evaluation,
            run_validation_reports=run_validation_reports,
            force_rerun=force,
            modalities=(
                list(modalities)
                if modalities
                else ["layout", "document_structure", "key_value"]
            ),
        )

    def should_skip_job(self, job: SubmissionSubsetJob) -> tuple[bool, str]:
        """Determine if a job should be skipped.

        Args:
            job: The submission subset job to check

        Returns:
            Tuple of (should_skip, reason_message)
        """
        if self.run_validation_reports and not (
            self.run_merge or self.run_dataset_creation or self.run_evaluation
        ):
            return False, ""

        if self.force_rerun:
            return False, ""

        merged_dir = job.get_merged_xml_dir()
        merged_gt = merged_dir / "combined_set_A.xml"
        merged_pred = merged_dir / "combined_set_B.xml"

        if self.run_merge and not (self.run_dataset_creation or self.run_evaluation):
            # Merge-only mode: check if merged XMLs exist
            if merged_gt.exists() and merged_pred.exists():
                return True, f"merged XML already present at {merged_dir}"
        elif not self.run_dataset_creation and self.run_evaluation:
            # Eval-only mode: don't skip based on output dir
            return False, ""
        else:
            # Full pipeline: check if output directory exists
            if job.output_dir.exists():
                return (
                    True,
                    f"output directory already exists at {job.output_dir} (use --force to re-run)",
                )

        return False, ""

    def get_description(self) -> str:
        """Get human-readable description of what will be executed."""
        if self.run_merge and not (self.run_dataset_creation or self.run_evaluation):
            return "merge annotations for"
        if self.run_validation_reports and not (
            self.run_merge or self.run_dataset_creation or self.run_evaluation
        ):
            return "regenerate validation reports for"
        elif not self.run_dataset_creation and self.run_evaluation:
            return "run evaluation for"
        else:
            return "evaluate"

    def should_aggregate_validation(self) -> bool:
        """Determine if validation reports should be aggregated."""
        return self.run_evaluation or self.run_validation_reports


@dataclass(frozen=True)
class SubmissionSubsetJob:
    """Container describing the artefacts needed to evaluate one submission subset."""

    submission_name: str
    subset_name: str
    tasks_root: Path
    base_cvat_root: Path
    output_dir: Path

    def get_merged_xml_dir(self) -> Path:
        """Get the directory where merged XML files are stored."""
        return self.output_dir / "merged_xml"

    def format_job_id(self) -> str:
        """Format job identifier for logging."""
        return f"{self.submission_name}/{self.subset_name}"


def _execute_job(
    job: SubmissionSubsetJob,
    plan: ExecutionPlan,
    *,
    strict: bool,
    user_csv: Optional[Path],
    force_ocr: bool,
    ocr_scale: float,
    storage_scale: float,
) -> Optional[pd.DataFrame]:
    """Execute pipeline stages for a single job according to the execution plan.

    Args:
        job: The submission subset job to execute
        plan: Execution plan specifying what stages to run
        strict: Enable strict mode for conversions
        user_csv: Optional user CSV for evaluation
        force_ocr: Force OCR on PDF pages
        ocr_scale: Scale factor for OCR rendering
        storage_scale: Scale for stored page images and coordinates

    Returns:
        DataFrame with evaluation results, or None if no evaluation was run
    """
    pipeline = CVATEvaluationPipeline(
        cvat_root=job.base_cvat_root,
        output_dir=job.output_dir,
        strict=strict,
        tasks_root=job.tasks_root,
        force_ocr=force_ocr,
        ocr_scale=ocr_scale,
        storage_scale=storage_scale,
    )

    job.output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Merge XML annotations
    if plan.run_merge:
        pipeline.merge_annotation_xmls(destination_dir=job.get_merged_xml_dir())

        # If only merging, return early
        if not (plan.run_dataset_creation or plan.run_evaluation):
            return None

    if plan.run_validation_reports:
        pipeline.regenerate_validation_reports_from_merged(
            merged_dir=job.get_merged_xml_dir(),
        )
        return None

    # Stage 2: Create datasets
    if plan.run_dataset_creation:
        pipeline.create_ground_truth_dataset()
        pipeline.create_prediction_dataset()

    # Stage 3: Run evaluations
    if plan.run_evaluation:
        pipeline.run_table_evaluation(reuse_existing=not plan.force_rerun)
        return pipeline.run_evaluation(
            modalities=plan.modalities,
            user_csv=user_csv,
            subset_label=job.subset_name,
        )

    return None


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


def _get_set_label(xml_pattern: str) -> str:
    """Extract set label from XML pattern (e.g., 'set_A' or 'set_B')."""
    return "set_A" if "set_A" in xml_pattern else "set_B"


def _cleanup_path(path: Path, description: str) -> None:
    """Remove a file or directory tree, logging warnings on failure.

    Args:
        path: Path to remove
        description: Human-readable description for logging
    """
    if not path.exists():
        return

    try:
        if path.is_file():
            path.unlink()
        else:
            # Recursively remove directory contents
            for item in sorted(path.glob("**/*"), reverse=True):
                if item.is_file() or item.is_symlink():
                    item.unlink(missing_ok=True)
                else:
                    item.rmdir()
            path.rmdir()
        _LOGGER.debug("Removed %s: %s", description, path)
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("Failed to remove %s %s: %s", description, path, exc)


def aggregate_validation_reports(
    jobs: Sequence[SubmissionSubsetJob],
    xml_pattern: str,
) -> CVATValidationRunReport:
    """Aggregate validation reports across all subsets in a submission.

    Args:
        jobs: Sequence of jobs for a single submission
        xml_pattern: Pattern to match XML files (e.g., "task_{xx}_set_A")

    Returns:
        Aggregated validation report for all subsets
    """
    all_reports = []
    set_label = _get_set_label(xml_pattern)

    for job in jobs:
        subset_report_path = job.output_dir / f"validation_report_{set_label}.json"
        if subset_report_path.exists():
            try:
                subset_run_report = CVATValidationRunReport.model_validate_json(
                    subset_report_path.read_text(encoding="utf-8")
                )
                all_reports.extend(subset_run_report.samples)
            except Exception as exc:
                _LOGGER.warning(
                    "Failed to load validation report from %s: %s",
                    subset_report_path,
                    exc,
                )

    return CVATValidationRunReport(
        samples=all_reports,
        statistics=CVATValidationRunReport.compute_statistics(all_reports),
    )


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
    validation_only: bool = False,
    force_ocr: bool = False,
    ocr_scale: float = 1.0,
    storage_scale: float = 2.0,
) -> None:
    """Execute the CVAT evaluation pipeline for each prepared job."""
    if not jobs:
        _LOGGER.info("No jobs discovered; nothing to do.")
        return

    # Create execution plan from arguments
    plan = ExecutionPlan.from_args(
        merge_only=merge_only,
        eval_only=eval_only,
        validation_only=validation_only,
        force=force,
        modalities=modalities,
    )

    jobs_by_submission: "OrderedDict[str, list[SubmissionSubsetJob]]" = OrderedDict()
    for job in jobs:
        jobs_by_submission.setdefault(job.submission_name, []).append(job)

    # Track overall statistics
    overall_stats = {
        "total_submissions": 0,
        "successful_submissions": 0,
        "partial_submissions": 0,
        "failed_submissions": 0,
        "total_subsets": 0,
        "completed_subsets": 0,
        "failed_subsets": 0,
        "skipped_subsets": 0,
    }

    for submission_name, submission_jobs in jobs_by_submission.items():
        if not submission_jobs:
            continue

        overall_stats["total_submissions"] += 1
        submission_dir = submission_jobs[0].output_dir.parent
        submission_dir.mkdir(parents=True, exist_ok=True)
        submission_dfs: List[pd.DataFrame] = []
        completed_jobs: List[SubmissionSubsetJob] = []
        failed_jobs: List[tuple[str, str]] = []  # (job_id, error_message)
        skipped_jobs: List[tuple[str, str]] = []  # (job_id, reason)

        _LOGGER.info("")
        _LOGGER.info("╔" + "=" * 78 + "╗")
        _LOGGER.info("║" + f" SUBMISSION: {submission_name} ".center(78) + "║")
        _LOGGER.info("╚" + "=" * 78 + "╝")

        for job in submission_jobs:
            job_id = job.format_job_id()
            _LOGGER.info("→ Processing %s", job_id)

            # Check if job should be skipped
            should_skip, skip_reason = plan.should_skip_job(job)
            if should_skip:
                _LOGGER.info("  ⊘ Skipped: %s", skip_reason)
                skipped_jobs.append((job_id, skip_reason))
                continue

            # Handle dry-run mode
            if dry_run:
                _LOGGER.info(
                    "  [DRY-RUN] Would %s with base=%s tasks=%s output=%s",
                    plan.get_description(),
                    job.base_cvat_root,
                    job.tasks_root,
                    job.output_dir,
                )
                continue

            # Execute the pipeline stages
            try:
                subset_df = _execute_job(
                    job,
                    plan,
                    strict=strict,
                    user_csv=user_csv,
                    force_ocr=force_ocr,
                    ocr_scale=ocr_scale,
                    storage_scale=storage_scale,
                )

                if subset_df is not None and not subset_df.empty:
                    if "subset" not in subset_df.columns:
                        subset_df = subset_df.copy()
                        subset_df.insert(0, "subset", job.subset_name)
                    submission_dfs.append(subset_df)

                # Track successfully completed jobs for validation report aggregation
                completed_jobs.append(job)
                _LOGGER.info("  ✓ Completed successfully")

            except (
                Exception
            ) as exc:  # noqa: BLE001 - we want to capture all failures per subset
                error_msg = str(exc)
                failed_jobs.append((job_id, error_msg))
                _LOGGER.error("  ✗ FAILED: %s", error_msg)
                _LOGGER.debug("Subset failure details", exc_info=True)

        # Aggregate validation reports across successfully completed subsets only
        if plan.should_aggregate_validation():
            if completed_jobs:
                _LOGGER.info(
                    "Aggregating validation reports for submission %s (%d/%d subsets completed)",
                    submission_name,
                    len(completed_jobs),
                    len(submission_jobs),
                )
                for set_pattern, set_label in [
                    (GROUND_TRUTH_PATTERN, "set_A"),
                    (PREDICTION_PATTERN, "set_B"),
                ]:
                    aggregated_report = aggregate_validation_reports(
                        completed_jobs, set_pattern
                    )
                    submission_validation_path = (
                        submission_dir / f"validation_report_{set_label}.json"
                    )
                    submission_validation_path.write_text(
                        aggregated_report.model_dump_json(indent=2),
                        encoding="utf-8",
                    )
                    _LOGGER.info(
                        "✓ Submission-level %s validation report: %s (%d samples)",
                        set_label,
                        submission_validation_path,
                        len(aggregated_report.samples),
                    )
            else:
                _LOGGER.warning(
                    "No subsets completed successfully for submission %s - skipping validation report aggregation",
                    submission_name,
                )

        # Print submission summary
        _LOGGER.info("")
        _LOGGER.info("=" * 80)
        _LOGGER.info(f"SUBMISSION SUMMARY: {submission_name}")
        _LOGGER.info("=" * 80)
        _LOGGER.info(f"  Total subsets:     {len(submission_jobs)}")
        _LOGGER.info(f"  ✓ Completed:       {len(completed_jobs)}")
        _LOGGER.info(f"  ✗ Failed:          {len(failed_jobs)}")
        _LOGGER.info(f"  ⊘ Skipped:         {len(skipped_jobs)}")

        if failed_jobs:
            _LOGGER.error("  Failed subsets:")
            for job_id, error_msg in failed_jobs:
                _LOGGER.error(f"    • {job_id}: {error_msg}")

        if submission_dfs:
            combined_df = pd.concat(submission_dfs, ignore_index=True)
            combined_out = submission_dir / "combined_evaluation.xlsx"
            if "subset" not in combined_df.columns:
                combined_df.insert(0, "subset", submission_name)
            _write_as_excel_table(combined_df, combined_out)
            _LOGGER.info(f"  Combined evaluation: {combined_out}")

        status = (
            "✓ SUCCESS"
            if not failed_jobs
            else ("⚠ PARTIAL" if completed_jobs else "✗ FAILED")
        )
        _LOGGER.info(f"  Status: {status}")
        _LOGGER.info("=" * 80)

        # Update overall statistics
        overall_stats["total_subsets"] += len(submission_jobs)
        overall_stats["completed_subsets"] += len(completed_jobs)
        overall_stats["failed_subsets"] += len(failed_jobs)
        overall_stats["skipped_subsets"] += len(skipped_jobs)

        if not failed_jobs:
            overall_stats["successful_submissions"] += 1
        elif completed_jobs:
            overall_stats["partial_submissions"] += 1
        else:
            overall_stats["failed_submissions"] += 1

        # Clean up per-subset artifacts (they're aggregated at submission level)
        for job in submission_jobs:
            _cleanup_path(
                job.output_dir / "combined_evaluation.xlsx",
                "subset combined evaluation",
            )
            _cleanup_path(job.output_dir / "intermediate", "intermediate directory")

    # Print overall summary
    _LOGGER.info("")
    _LOGGER.info("╔" + "=" * 78 + "╗")
    _LOGGER.info("║" + " PIPELINE EXECUTION COMPLETE ".center(78) + "║")
    _LOGGER.info("╚" + "=" * 78 + "╝")
    _LOGGER.info("Overall Statistics:")
    _LOGGER.info(f"  Submissions: {overall_stats['total_submissions']} total")
    _LOGGER.info(f"    ✓ Successful:  {overall_stats['successful_submissions']}")
    _LOGGER.info(f"    ⚠ Partial:     {overall_stats['partial_submissions']}")
    _LOGGER.info(f"    ✗ Failed:      {overall_stats['failed_submissions']}")
    _LOGGER.info("")
    _LOGGER.info(f"  Subsets: {overall_stats['total_subsets']} total")
    _LOGGER.info(f"    ✓ Completed:   {overall_stats['completed_subsets']}")
    _LOGGER.info(f"    ✗ Failed:      {overall_stats['failed_subsets']}")
    _LOGGER.info(f"    ⊘ Skipped:     {overall_stats['skipped_subsets']}")
    _LOGGER.info("=" * 80)

    # Exit with error if any failures occurred
    if overall_stats["failed_subsets"] > 0:
        _LOGGER.error(
            "Pipeline completed with %d failed subset(s). Review error messages above.",
            overall_stats["failed_subsets"],
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
        "--validation-only",
        action="store_true",
        help="Regenerate validation reports using existing merged annotations.",
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
    parser.add_argument(
        "--storage-scale",
        type=float,
        default=2.0,
        help="Scale for stored page images and coordinates (default: 2.0 = 144 DPI).",
    )

    return parser.parse_args(argv)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    configure_logging()
    args = parse_args(argv)

    try:
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
            validation_only=args.validation_only,
            force_ocr=args.force_ocr,
            ocr_scale=args.ocr_scale,
            storage_scale=args.storage_scale,
        )
    except ValueError as exc:
        _LOGGER.error("%s", exc)
        return


if __name__ == "__main__":
    main()
