import os
from pathlib import Path

import pytest

from docling_eval.cli.main import evaluate, get_prediction_provider, visualize
from docling_eval.datamodels.types import (
    BenchMarkNames,
    EvaluationModality,
    PredictionFormats,
    PredictionProviderType,
)
from docling_eval.dataset_builders.doclaynet_v1_builder import DocLayNetV1DatasetBuilder
from docling_eval.dataset_builders.doclaynet_v2_builder import DocLayNetV2DatasetBuilder
from docling_eval.dataset_builders.docvqa_builder import DocVQADatasetBuilder
from docling_eval.dataset_builders.dpbench_builder import DPBenchDatasetBuilder
from docling_eval.dataset_builders.file_dataset_builder import FileDatasetBuilder
from docling_eval.dataset_builders.funsd_builder import FUNSDDatasetBuilder
from docling_eval.dataset_builders.omnidocbench_builder import (
    OmniDocBenchDatasetBuilder,
)
from docling_eval.dataset_builders.otsl_table_dataset_builder import (
    FintabNetDatasetBuilder,
    PubTables1MDatasetBuilder,
    PubTabNetDatasetBuilder,
)
from docling_eval.dataset_builders.pixparse_builder import PixparseDatasetBuilder
from docling_eval.dataset_builders.xfund_builder import XFUNDDatasetBuilder
from docling_eval.prediction_providers.file_provider import FilePredictionProvider
from docling_eval.prediction_providers.tableformer_provider import (
    TableFormerPredictionProvider,
)

IS_CI = bool(os.getenv("CI"))


@pytest.mark.dependency()
def test_run_dpbench_e2e():
    target_path = Path(f"./scratch/{BenchMarkNames.DPBENCH.value}/")
    docling_provider = get_prediction_provider(PredictionProviderType.DOCLING)

    dataset_layout = DPBenchDatasetBuilder(
        target=target_path / "gt_dataset",
        begin_index=10,
        end_index=25,
    )  # 10-25 is a small range which has samples with tables included.

    dataset_layout.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset_layout.save_to_disk()  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    docling_provider.create_prediction_dataset(
        name=dataset_layout.name,
        gt_dataset_dir=target_path / "gt_dataset",
        target_dataset_dir=target_path / "eval_dataset_e2e",
    )

    ## Evaluate Layout
    evaluate(
        modality=EvaluationModality.LAYOUT,
        benchmark=BenchMarkNames.DPBENCH,
        idir=target_path / "eval_dataset_e2e",
        odir=target_path / "evaluations" / EvaluationModality.LAYOUT.value,
    )

    visualize(
        modality=EvaluationModality.LAYOUT,
        benchmark=BenchMarkNames.DPBENCH,
        idir=target_path / "eval_dataset_e2e",
        odir=target_path / "evaluations" / EvaluationModality.LAYOUT.value,
    )

    ## Evaluate Reading order
    evaluate(
        modality=EvaluationModality.READING_ORDER,
        benchmark=BenchMarkNames.DPBENCH,
        idir=target_path / "eval_dataset_e2e",
        odir=target_path / "evaluations" / EvaluationModality.READING_ORDER.value,
    )

    visualize(
        modality=EvaluationModality.READING_ORDER,
        benchmark=BenchMarkNames.DPBENCH,
        idir=target_path / "eval_dataset_e2e",
        odir=target_path / "evaluations" / EvaluationModality.READING_ORDER.value,
    )

    ## Evaluate Markdown text
    evaluate(
        modality=EvaluationModality.MARKDOWN_TEXT,
        benchmark=BenchMarkNames.DPBENCH,
        idir=target_path / "eval_dataset_e2e",
        odir=target_path / "evaluations" / EvaluationModality.MARKDOWN_TEXT.value,
    )

    visualize(
        modality=EvaluationModality.MARKDOWN_TEXT,
        benchmark=BenchMarkNames.DPBENCH,
        idir=target_path / "eval_dataset_e2e",
        odir=target_path / "evaluations" / EvaluationModality.MARKDOWN_TEXT.value,
    )


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_run_doclaynet_with_doctags_fileprovider():
    target_path = Path(f"./scratch/{BenchMarkNames.DOCLAYNETV1.value}-SmolDocling/")
    file_provider = FilePredictionProvider(
        prediction_format=PredictionFormats.DOCTAGS,
        source_path=Path("./tests/data/doclaynet_v1_doctags_sample"),
        do_visualization=True,
        ignore_missing_files=True,
        use_ground_truth_page_images=True,
    )

    dataset_layout = DocLayNetV1DatasetBuilder(
        target=target_path / "gt_dataset",
        end_index=5,
    )

    # dataset_layout.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset_layout.save_to_disk()  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    file_provider.create_prediction_dataset(
        name=dataset_layout.name,
        gt_dataset_dir=target_path / "gt_dataset",
        target_dataset_dir=target_path / "eval_dataset",
    )

    ## Evaluate Layout
    evaluate(
        modality=EvaluationModality.LAYOUT,
        benchmark=BenchMarkNames.DOCLAYNETV1,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.LAYOUT.value,
    )

    visualize(
        modality=EvaluationModality.LAYOUT,
        benchmark=BenchMarkNames.DOCLAYNETV1,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.LAYOUT.value,
    )

    ## Evaluate Markdown text
    evaluate(
        modality=EvaluationModality.MARKDOWN_TEXT,
        benchmark=BenchMarkNames.DOCLAYNETV1,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.MARKDOWN_TEXT.value,
    )

    visualize(
        modality=EvaluationModality.MARKDOWN_TEXT,
        benchmark=BenchMarkNames.DOCLAYNETV1,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.MARKDOWN_TEXT.value,
    )


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_run_omnidocbench_e2e():
    target_path = Path(f"./scratch/{BenchMarkNames.OMNIDOCBENCH.value}/")
    docling_provider = get_prediction_provider(PredictionProviderType.DOCLING)

    dataset_layout = OmniDocBenchDatasetBuilder(
        target=target_path / "gt_dataset",
        end_index=5,
    )

    dataset_layout.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset_layout.save_to_disk()  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    docling_provider.create_prediction_dataset(
        name=dataset_layout.name,
        gt_dataset_dir=target_path / "gt_dataset",
        target_dataset_dir=target_path / "eval_dataset",
    )

    # Evaluate Layout
    evaluate(
        modality=EvaluationModality.LAYOUT,
        benchmark=BenchMarkNames.OMNIDOCBENCH,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.LAYOUT.value,
    )

    visualize(
        modality=EvaluationModality.LAYOUT,
        benchmark=BenchMarkNames.OMNIDOCBENCH,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.LAYOUT.value,
    )

    # Evaluate Reading Order
    evaluate(
        modality=EvaluationModality.READING_ORDER,
        benchmark=BenchMarkNames.OMNIDOCBENCH,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.READING_ORDER.value,
    )

    visualize(
        modality=EvaluationModality.READING_ORDER,
        benchmark=BenchMarkNames.OMNIDOCBENCH,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.READING_ORDER.value,
    )

    # Evaluate Markdown Text
    evaluate(
        modality=EvaluationModality.MARKDOWN_TEXT,
        benchmark=BenchMarkNames.OMNIDOCBENCH,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.MARKDOWN_TEXT.value,
    )

    visualize(
        modality=EvaluationModality.MARKDOWN_TEXT,
        benchmark=BenchMarkNames.OMNIDOCBENCH,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.MARKDOWN_TEXT.value,
    )


@pytest.mark.dependency(
    depends=["tests/test_dataset_builder.py::test_run_dpbench_e2e"],
    scope="session",
)
def test_run_dpbench_tables():
    target_path = Path(f"./scratch/{BenchMarkNames.DPBENCH.value}/")
    tableformer_provider = TableFormerPredictionProvider(do_visualization=True)

    tableformer_provider.create_prediction_dataset(
        name="DPBench tables eval",
        gt_dataset_dir=target_path / "gt_dataset",
        target_dataset_dir=target_path / "eval_dataset_tables",
    )

    evaluate(
        modality=EvaluationModality.TABLE_STRUCTURE,
        benchmark=BenchMarkNames.DPBENCH,
        idir=target_path / "eval_dataset_tables",
        odir=target_path / "evaluations" / EvaluationModality.TABLE_STRUCTURE.value,
    )

    visualize(
        modality=EvaluationModality.TABLE_STRUCTURE,
        benchmark=BenchMarkNames.DPBENCH,
        idir=target_path / "eval_dataset_tables",
        odir=target_path / "evaluations" / EvaluationModality.TABLE_STRUCTURE.value,
    )


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_run_omnidocbench_tables():
    target_path = Path(f"./scratch/{BenchMarkNames.OMNIDOCBENCH.value}/")
    tableformer_provider = TableFormerPredictionProvider()

    dataset_tables = OmniDocBenchDatasetBuilder(
        target=target_path / "gt_dataset",
        end_index=5,
    )

    dataset_tables.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset_tables.save_to_disk()  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    tableformer_provider.create_prediction_dataset(
        name=dataset_tables.name,
        gt_dataset_dir=target_path / "gt_dataset",
        target_dataset_dir=target_path / "eval_dataset",
    )

    evaluate(
        modality=EvaluationModality.TABLE_STRUCTURE,
        benchmark=BenchMarkNames.OMNIDOCBENCH,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.TABLE_STRUCTURE.value,
    )

    visualize(
        modality=EvaluationModality.TABLE_STRUCTURE,
        benchmark=BenchMarkNames.OMNIDOCBENCH,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.TABLE_STRUCTURE.value,
    )


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_run_doclaynet_v1_e2e():
    target_path = Path(f"./scratch/{BenchMarkNames.DOCLAYNETV1.value}/")
    docling_provider = get_prediction_provider(PredictionProviderType.DOCLING)

    dataset_layout = DocLayNetV1DatasetBuilder(
        # prediction_provider=docling_provider,
        target=target_path / "gt_dataset",
        end_index=5,
    )

    # dataset_layout.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset_layout.save_to_disk()  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    docling_provider.create_prediction_dataset(
        name=dataset_layout.name,
        gt_dataset_dir=target_path / "gt_dataset",
        target_dataset_dir=target_path / "eval_dataset",
    )

    # Evaluate Layout
    evaluate(
        modality=EvaluationModality.LAYOUT,
        benchmark=BenchMarkNames.DOCLAYNETV1,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.LAYOUT.value,
    )

    visualize(
        modality=EvaluationModality.LAYOUT,
        benchmark=BenchMarkNames.DOCLAYNETV1,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.LAYOUT.value,
    )

    # Evaluate Markdown Text
    evaluate(
        modality=EvaluationModality.MARKDOWN_TEXT,
        benchmark=BenchMarkNames.DOCLAYNETV1,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.MARKDOWN_TEXT.value,
    )

    visualize(
        modality=EvaluationModality.MARKDOWN_TEXT,
        benchmark=BenchMarkNames.DOCLAYNETV1,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.MARKDOWN_TEXT.value,
    )


@pytest.mark.skip("Test needs local data which is unavailable.")
def test_run_doclaynet_v2_e2e():
    target_path = Path(f"./scratch/{BenchMarkNames.DOCLAYNETV2.value}/")
    docling_provider = get_prediction_provider(PredictionProviderType.DOCLING)

    dataset_layout = DocLayNetV2DatasetBuilder(
        dataset_source=Path("/path/to/doclaynet_v2_benchmark"),
        target=target_path / "gt_dataset",
        end_index=5,
    )

    dataset_layout.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset_layout.save_to_disk()  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    docling_provider.create_prediction_dataset(
        name=dataset_layout.name,
        gt_dataset_dir=target_path / "gt_dataset",
        target_dataset_dir=target_path / "eval_dataset",
    )

    # Evaluate Layout
    evaluate(
        modality=EvaluationModality.LAYOUT,
        benchmark=BenchMarkNames.DOCLAYNETV2,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.LAYOUT.value,
    )

    visualize(
        modality=EvaluationModality.LAYOUT,
        benchmark=BenchMarkNames.DOCLAYNETV2,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.LAYOUT.value,
    )

    # Evaluate Markdown Text
    evaluate(
        modality=EvaluationModality.MARKDOWN_TEXT,
        benchmark=BenchMarkNames.DOCLAYNETV2,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.MARKDOWN_TEXT.value,
    )

    visualize(
        modality=EvaluationModality.MARKDOWN_TEXT,
        benchmark=BenchMarkNames.DOCLAYNETV2,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.MARKDOWN_TEXT.value,
    )


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_run_funsd():
    target_path = Path(f"./scratch/{BenchMarkNames.FUNSD.value}/")

    dataset_layout = FUNSDDatasetBuilder(
        dataset_source=target_path / "input_dataset",
        target=target_path / "gt_dataset",
        end_index=5,
    )

    dataset_layout.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset_layout.save_to_disk()  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_run_xfund():
    target_path = Path(f"./scratch/{BenchMarkNames.XFUND.value}/")

    dataset_layout = XFUNDDatasetBuilder(
        dataset_source=target_path / "input_dataset",
        target=target_path / "gt_dataset",
        end_index=5,
    )

    dataset_layout.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset_layout.save_to_disk()  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_run_fintabnet_builder():
    target_path = Path(f"./scratch/{BenchMarkNames.FINTABNET.value}/")
    tableformer_provider = TableFormerPredictionProvider(do_visualization=True)

    dataset = FintabNetDatasetBuilder(
        target=target_path / "gt_dataset",
        end_index=5,
    )

    # dataset.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset.save_to_disk()  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    tableformer_provider.create_prediction_dataset(
        name=dataset.name,
        gt_dataset_dir=target_path / "gt_dataset",
        target_dataset_dir=target_path / "eval_dataset",
    )

    evaluate(
        modality=EvaluationModality.TABLE_STRUCTURE,
        benchmark=BenchMarkNames.FINTABNET,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.TABLE_STRUCTURE.value,
    )

    visualize(
        modality=EvaluationModality.TABLE_STRUCTURE,
        benchmark=BenchMarkNames.FINTABNET,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.TABLE_STRUCTURE.value,
    )


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_run_p1m_builder():
    target_path = Path(f"./scratch/{BenchMarkNames.PUB1M.value}/")
    tableformer_provider = TableFormerPredictionProvider(do_visualization=True)

    dataset = PubTables1MDatasetBuilder(
        target=target_path / "gt_dataset",
        end_index=5,
    )

    # dataset.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset.save_to_disk()  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    tableformer_provider.create_prediction_dataset(
        name=dataset.name,
        gt_dataset_dir=target_path / "gt_dataset",
        target_dataset_dir=target_path / "eval_dataset",
    )

    evaluate(
        modality=EvaluationModality.TABLE_STRUCTURE,
        benchmark=BenchMarkNames.PUB1M,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.TABLE_STRUCTURE.value,
    )

    visualize(
        modality=EvaluationModality.TABLE_STRUCTURE,
        benchmark=BenchMarkNames.PUB1M,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.TABLE_STRUCTURE.value,
    )


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_run_pubtabnet_builder():
    target_path = Path(f"./scratch/{BenchMarkNames.PUBTABNET.value}/")
    tableformer_provider = TableFormerPredictionProvider(do_visualization=True)

    dataset = PubTabNetDatasetBuilder(
        target=target_path / "gt_dataset",
        end_index=25,
    )

    # dataset.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset.save_to_disk()  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    tableformer_provider.create_prediction_dataset(
        name=dataset.name,
        split="val",
        gt_dataset_dir=target_path / "gt_dataset",
        target_dataset_dir=target_path / "eval_dataset",
        end_index=25,
    )

    evaluate(
        modality=EvaluationModality.TABLE_STRUCTURE,
        benchmark=BenchMarkNames.PUBTABNET,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.TABLE_STRUCTURE.value,
        split="val",
    )

    visualize(
        modality=EvaluationModality.TABLE_STRUCTURE,
        benchmark=BenchMarkNames.PUBTABNET,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.TABLE_STRUCTURE.value,
        split="val",
    )


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_run_docvqa_builder():
    target_path = Path(f"./scratch/{BenchMarkNames.DOCVQA.value}/")

    dataset_layout = DocVQADatasetBuilder(
        target=target_path / "gt_dataset", end_index=25, split="validation"
    )

    dataset_layout.save_to_disk()  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.
    docling_provider = get_prediction_provider(PredictionProviderType.DOCLING)

    docling_provider.create_prediction_dataset(
        name=dataset_layout.name,
        gt_dataset_dir=target_path / "gt_dataset",
        target_dataset_dir=target_path / "eval_dataset_e2e",
        split="validation",
    )


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_run_pixparse_builder():
    target_path = Path(f"./scratch/{BenchMarkNames.PIXPARSEIDL.value}/")

    dataset_pixparse = PixparseDatasetBuilder(target=target_path / "gt_dataset")

    dataset_pixparse.retrieve_input_dataset()
    dataset_pixparse.save_to_disk()  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.
    docling_provider = get_prediction_provider(PredictionProviderType.DOCLING)
    docling_provider.ignore_missing_predictions = False

    docling_provider.create_prediction_dataset(
        name=dataset_pixparse.name,
        gt_dataset_dir=target_path / "gt_dataset",
        target_dataset_dir=target_path / "eval_dataset_e2e",
        end_index=5,
    )

    evaluate(
        modality=EvaluationModality.OCR,
        benchmark=BenchMarkNames.PIXPARSEIDL,
        idir=target_path / "eval_dataset_e2e",
        odir=target_path / "evaluations" / EvaluationModality.OCR.value,
    )

    visualize(
        modality=EvaluationModality.OCR,
        benchmark=BenchMarkNames.PIXPARSEIDL,
        idir=target_path / "eval_dataset_e2e",
        odir=target_path / "evaluations" / EvaluationModality.OCR.value,
    )


def test_file_dataset_builder():
    target_path = Path(f"./scratch/file_dataset/")

    dataset_builder = FileDatasetBuilder(
        name="Test_Files", dataset_source=Path("./tests/data/files"), target=target_path
    )

    dataset_builder.save_to_disk(do_visualization=True)
