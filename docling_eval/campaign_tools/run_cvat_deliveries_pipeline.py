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
    force_rerun: bool
    modalities: List[str]

    @classmethod
    def from_args(
        cls,
        merge_only: bool,
        eval_only: bool,
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
        if merge_only and eval_only:
            raise ValueError("Cannot combine --merge-only and --eval-only")

        return cls(
            run_merge=merge_only or not eval_only,
            run_dataset_creation=not merge_only and not eval_only,
            run_evaluation=not merge_only,
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
        elif not self.run_dataset_creation and self.run_evaluation:
            return "run evaluation for"
        else:
            return "evaluate"


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
) -> Optional[pd.DataFrame]:
    """Execute pipeline stages for a single job according to the execution plan.

    Args:
        job: The submission subset job to execute
        plan: Execution plan specifying what stages to run
        strict: Enable strict mode for conversions
        user_csv: Optional user CSV for evaluation
        force_ocr: Force OCR on PDF pages
        ocr_scale: Scale factor for OCR rendering

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
    )

    job.output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Merge XML annotations
    if plan.run_merge:
        pipeline.merge_annotation_xmls(destination_dir=job.get_merged_xml_dir())

        # If only merging, return early
        if not (plan.run_dataset_creation or plan.run_evaluation):
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

    return CVATValidationRunReport(samples=all_reports)


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

    # Create execution plan from arguments
    plan = ExecutionPlan.from_args(
        merge_only=merge_only,
        eval_only=eval_only,
        force=force,
        modalities=modalities,
    )

    jobs_by_submission: "OrderedDict[str, list[SubmissionSubsetJob]]" = OrderedDict()
    for job in jobs:
        jobs_by_submission.setdefault(job.submission_name, []).append(job)

    for submission_name, submission_jobs in jobs_by_submission.items():
        if not submission_jobs:
            continue

        submission_dir = submission_jobs[0].output_dir.parent
        submission_dir.mkdir(parents=True, exist_ok=True)
        submission_dfs: List[pd.DataFrame] = []
        completed_jobs: List[SubmissionSubsetJob] = []
        failure = False

        _LOGGER.info("=== Processing submission %s ===", submission_name)

        for job in submission_jobs:
            job_id = job.format_job_id()
            _LOGGER.info("Processing %s", job_id)

            # Check if job should be skipped
            should_skip, skip_reason = plan.should_skip_job(job)
            if should_skip:
                _LOGGER.info("Skipping %s: %s", job_id, skip_reason)
                continue

            # Handle dry-run mode
            if dry_run:
                _LOGGER.info(
                    "Dry-run: would %s %s with base=%s tasks=%s output=%s",
                    plan.get_description(),
                    job_id,
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
                )

                if subset_df is not None and not subset_df.empty:
                    if "subset" not in subset_df.columns:
                        subset_df = subset_df.copy()
                        subset_df.insert(0, "subset", job.subset_name)
                    submission_dfs.append(subset_df)

                # Track successfully completed jobs for validation report aggregation
                completed_jobs.append(job)

            except (
                Exception
            ) as exc:  # noqa: BLE001 - we want to capture all failures per subset
                failure = True
                _LOGGER.error("%s failed: %s", job_id, exc)
                _LOGGER.debug("Subset failure details", exc_info=True)

        # Aggregate validation reports across successfully completed subsets only
        if plan.run_evaluation:
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

        # Clean up per-subset artifacts (they're aggregated at submission level)
        for job in submission_jobs:
            _cleanup_path(
                job.output_dir / "combined_evaluation.xlsx",
                "subset combined evaluation",
            )
            _cleanup_path(job.output_dir / "intermediate", "intermediate directory")


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
            force_ocr=args.force_ocr,
            ocr_scale=args.ocr_scale,
        )
    except ValueError as exc:
        _LOGGER.error("%s", exc)
        return


if __name__ == "__main__":
    main()
