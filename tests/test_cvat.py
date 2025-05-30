import os
import shutil
from pathlib import Path

import pytest

from docling_eval.cli.main import (
    PredictionProviderType,
    create,
    create_cvat,
    create_eval,
    create_gt,
    evaluate,
    get_prediction_provider,
    visualize,
)
from docling_eval.datamodels.types import (
    BenchMarkNames,
    EvaluationModality,
    PredictionFormats,
    PredictionProviderType,
)
from docling_eval.dataset_builders.cvat_dataset_builder import CvatDatasetBuilder
from docling_eval.dataset_builders.cvat_preannotation_builder import (
    CvatPreannotationBuilder,
)
from docling_eval.dataset_builders.dpbench_builder import DPBenchDatasetBuilder

IS_CI = bool(os.getenv("CI"))


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_run_cvat_on_gt():
    gt_path = Path(f"./scratch/{BenchMarkNames.DPBENCH.value}-GT/")
    cvat_path = Path(f"./scratch/{BenchMarkNames.DPBENCH.value}-CVAT/")

    # Stage 1: Create and pre-annotate dataset
    dataset_layout = DPBenchDatasetBuilder(
        target=gt_path,
        begin_index=15,
        end_index=20,
    )  # 10-25 is a small range which has samples with tables included.

    dataset_layout.retrieve_input_dataset()
    dataset_layout.save_to_disk()  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    builder = CvatPreannotationBuilder(
        dataset_source=gt_path / "test", target=cvat_path, bucket_size=20
    )
    builder.prepare_for_annotation()

    ## Stage 2: Re-build dataset
    shutil.copy(
        "./tests/data/annotations_cvat.zip",
        str(cvat_path / "cvat_annotations" / "zips"),
    )

    # Create dataset from CVAT annotations
    dataset_builder = CvatDatasetBuilder(
        name="MyCVATAnnotations",
        dataset_source=cvat_path,
        target=cvat_path / "datasets",
        split="test",
    )
    dataset_builder.retrieve_input_dataset()
    dataset_builder.save_to_disk(do_visualization=True)


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_run_cvat_on_pred():
    target_path = Path(f"./scratch/{BenchMarkNames.DPBENCH.value}-GT/")
    cvat_path = Path(f"./scratch/{BenchMarkNames.DPBENCH.value}-CVAT/")

    docling_provider = get_prediction_provider(PredictionProviderType.DOCLING)

    dataset_layout = DPBenchDatasetBuilder(
        target=target_path / "gt_dataset",
        begin_index=15,
        end_index=20,
    )  # 10-25 is a small range which has samples with tables included.

    dataset_layout.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset_layout.save_to_disk()  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    docling_provider.create_prediction_dataset(
        name=dataset_layout.name,
        gt_dataset_dir=target_path / "gt_dataset",
        target_dataset_dir=target_path / "eval_dataset_e2e",
    )

    builder = CvatPreannotationBuilder(
        dataset_source=target_path / "eval_dataset_e2e" / "test",
        target=cvat_path,
        bucket_size=20,
    )
    builder.prepare_for_annotation()

    ## Stage 2: Re-build dataset
    shutil.copy(
        "./tests/data/annotations_cvat.zip",
        str(cvat_path / "cvat_annotations" / "zips"),
    )

    # Create dataset from CVAT annotations
    dataset_builder = CvatDatasetBuilder(
        name="MyCVATAnnotations",
        dataset_source=cvat_path,
        target=cvat_path / "datasets",
        split="test",
    )
    dataset_builder.retrieve_input_dataset()
    dataset_builder.save_to_disk(do_visualization=True)


def run_cvat_e2e(idir: Path, odir: Path, annotation_xmlfile: Path):
    # Stage 1: create a plain-file gt/eval-dataset
    create(
        benchmark=BenchMarkNames.PLAIN_FILES,
        dataset_source=idir,
        output_dir=odir,
        prediction_provider=PredictionProviderType.PDF_DOCLING,
    )
    assert os.path.exists(odir / "gt_dataset")
    assert os.path.exists(odir / "eval_dataset")

    # Stage 2: create the CVAT setup from pdfs
    create_cvat(
        gt_dir=odir / "eval_dataset/test",
        output_dir=odir / "cvat_dataset_preannotated",
        bucket_size=10,
        use_predictions=True,
    )
    assert os.path.exists(odir / "cvat_dataset_preannotated")

    # Stage 3: copy the manual annotations
    shutil.copy(
        annotation_xmlfile, odir / "cvat_dataset_preannotated/cvat_annotations/xmls"
    )

    # Stage 4: Create the dataset
    create_gt(
        benchmark=BenchMarkNames.CVAT,
        dataset_source=odir / "cvat_dataset_preannotated",
        output_dir=odir / "cvat_dataset_annotated",
    )
    assert os.path.exists(odir / "cvat_dataset_annotated")

    # Stage 5.1: create predictions with pdf-docling and evaluate layout
    create_eval(
        benchmark=BenchMarkNames.PLAIN_FILES,
        gt_dir=odir / "cvat_dataset_annotated/gt_dataset",
        output_dir=odir / "cvat_dataset_annotated/eval_pdf_docling",
        prediction_provider=PredictionProviderType.PDF_DOCLING,
    )
    assert os.path.exists(odir / "cvat_dataset_annotated/eval_pdf_docling")
    """
    assert (
        count_files(
            directory=odir / "cvat_dataset_annotated/gt_dataset/visualizations/"
        )
        == 3
    )
    """

    evaluate(
        modality=EvaluationModality.LAYOUT,
        benchmark=BenchMarkNames.PLAIN_FILES,
        idir=odir / "cvat_dataset_annotated/eval_pdf_docling/eval_dataset",
        odir=odir / "cvat_dataset_annotated/eval_pdf_docling",
    )
    assert os.path.exists(
        odir
        / "cvat_dataset_annotated/eval_pdf_docling"
        / "evaluation_PlainFiles_layout.json"
    )

    visualize(
        modality=EvaluationModality.LAYOUT,
        benchmark=BenchMarkNames.PLAIN_FILES,
        idir=odir / "cvat_dataset_annotated/eval_pdf_docling",
        odir=odir / "cvat_dataset_annotated/eval_pdf_docling",
    )
    assert os.path.exists(
        odir
        / "cvat_dataset_annotated/eval_pdf_docling"
        / "evaluation_PlainFiles_layout_f1.txt"
    )

    """
    # Stage 5.2: create predictions with macocr-docling and evaluate layout
    create_eval(
        benchmark=BenchMarkNames.PLAIN_FILES,
        gt_dir=odir / "cvat_dataset_annotated/gt_dataset",
        output_dir=odir / "cvat_dataset_annotated/eval_macocr_docling",
        prediction_provider=PredictionProviderType.MacOCR_DOCLING,
    )
    assert odir / "cvat_dataset_annotated/eval_macocr_docling"

    evaluate(
        modality=EvaluationModality.LAYOUT,
        benchmark=BenchMarkNames.PLAIN_FILES,
        idir=odir / "cvat_dataset_annotated/eval_macocr_docling/eval_dataset",
        odir=odir / "cvat_dataset_annotated/eval_macocr_docling",
    )

    visualize(
        modality=EvaluationModality.LAYOUT,
        benchmark=BenchMarkNames.PLAIN_FILES,
        idir=odir / "cvat_dataset_annotated/eval_macocr_docling",
        odir=odir / "cvat_dataset_annotated/eval_macocr_docling",
    )
    """


def test_run_cvat_e2e():
    run_cvat_e2e(
        idir=Path("./tests/data/cvat_pdfs_dataset_e2e/case_01"),
        odir=Path("./scratch/cvat_pdfs_dataset_e2e/case_01"),
        annotation_xmlfile=Path(
            "./tests/data/cvat_pdfs_dataset_e2e/case_01_annotations.xml"
        ),
    )

    run_cvat_e2e(
        idir=Path("./tests/data/cvat_pdfs_dataset_e2e/case_02"),
        odir=Path("./scratch/cvat_pdfs_dataset_e2e/case_02"),
        annotation_xmlfile=Path(
            "./tests/data/cvat_pdfs_dataset_e2e/case_02_annotations.xml"
        ),
    )
