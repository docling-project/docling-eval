from __future__ import annotations

from pathlib import Path

from docling_cvat_tools.cvat_tools.models import (
    CVATValidationError,
    CVATValidationReport,
    CVATValidationRunReport,
    ValidationSeverity,
)

from docling_eval.campaign_tools.run_cvat_deliveries_pipeline import (
    ExecutionPlan,
    JobExecutionResult,
    SubmissionSubsetJob,
    _detect_subset_partial_reasons,
    _process_submission,
)


def _write_validation_report(
    output_dir: Path,
    set_label: str,
    *,
    sample_name: str,
    severity: ValidationSeverity,
) -> None:
    report = CVATValidationRunReport(
        samples=[
            CVATValidationReport(
                sample_name=sample_name,
                errors=[
                    CVATValidationError(
                        error_type="test_error",
                        message="test message",
                        severity=severity,
                    )
                ],
            )
        ],
        statistics=CVATValidationRunReport.compute_statistics(
            [
                CVATValidationReport(
                    sample_name=sample_name,
                    errors=[
                        CVATValidationError(
                            error_type="test_error",
                            message="test message",
                            severity=severity,
                        )
                    ],
                )
            ]
        ),
    )
    (output_dir / f"validation_report_{set_label}.json").write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )


def test_detect_subset_partial_reasons_uses_fatal_validation_reports(
    tmp_path: Path,
) -> None:
    _write_validation_report(
        tmp_path,
        "set_A",
        sample_name="doc_a_page_000001.png",
        severity=ValidationSeverity.FATAL,
    )
    _write_validation_report(
        tmp_path,
        "set_B",
        sample_name="doc_b_page_000001.png",
        severity=ValidationSeverity.WARNING,
    )

    reasons = _detect_subset_partial_reasons(tmp_path)

    assert reasons == ["set_A: 1 sample(s) with fatal validation/conversion errors"]


def test_process_submission_counts_partial_subsets(monkeypatch, tmp_path: Path) -> None:
    job = SubmissionSubsetJob(
        submission_name="submission-1",
        subset_name="subset-1",
        tasks_root=tmp_path / "tasks",
        base_cvat_root=tmp_path / "base",
        output_dir=tmp_path / "out" / "submission-1" / "subset-1",
    )
    job.output_dir.mkdir(parents=True)

    plan = ExecutionPlan(
        run_merge=False,
        run_dataset_creation=False,
        run_evaluation=False,
        run_validation_reports=False,
        force_rerun=True,
        modalities=[],
    )

    monkeypatch.setattr(
        "docling_eval.campaign_tools.run_cvat_deliveries_pipeline._execute_job",
        lambda *args, **kwargs: JobExecutionResult(
            subset_df=None,
            partial_reasons=[
                "set_A: 1 sample(s) with fatal validation/conversion errors"
            ],
        ),
    )
    monkeypatch.setattr(
        "docling_eval.campaign_tools.run_cvat_deliveries_pipeline._cleanup_path",
        lambda *args, **kwargs: None,
    )

    result = _process_submission(
        "submission-1",
        [job],
        plan,
        strict=False,
        dry_run=False,
        user_csv=None,
        force_ocr=False,
        ocr_scale=1.0,
        storage_scale=2.0,
        stop_after_json=False,
        resume_from_json=False,
        reuse_eval_dataset=False,
        gt_json_dirname="ground_truth_json",
        pred_json_dirname="predictions_json",
        do_visualization=False,
    )

    assert result == {
        "total_subsets": 1,
        "completed_subsets": 0,
        "partial_subsets": 1,
        "failed_subsets": 0,
        "skipped_subsets": 0,
        "successful": False,
        "partial": True,
    }
