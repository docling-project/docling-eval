from pathlib import Path
from typing import List, Optional

import pytest
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrOptions,
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.document_converter import PdfFormatOption
from docling.models.factories import get_ocr_factory

from docling_eval.datamodels.types import (
    BenchMarkNames,
    EvaluationModality,
    PredictionFormats,
)
from docling_eval.dataset_builders.doclaynet_v1_builder import DocLayNetV1DatasetBuilder
from docling_eval.dataset_builders.doclaynet_v2_builder import DocLayNetV2DatasetBuilder
from docling_eval.dataset_builders.dpbench_builder import DPBenchDatasetBuilder
from docling_eval.dataset_builders.funsd_builder import FUNSDDatasetBuilder
from docling_eval.dataset_builders.omnidocbench_builder import (
    OmniDocBenchDatasetBuilder,
)
from docling_eval.dataset_builders.otsl_table_dataset_builder import (
    FintabNetDatasetBuilder,
    PubTables1MDatasetBuilder,
    PubTabNetDatasetBuilder,
)
from docling_eval.dataset_builders.xfund_builder import XFUNDDatasetBuilder
from docling_eval.legacy.main import evaluate, visualise
from docling_eval.prediction_providers.docling_provider import DoclingPredictionProvider
from docling_eval.prediction_providers.file_provider import FilePredictionProvider
from docling_eval.prediction_providers.tableformer_provider import (
    TableFormerPredictionProvider,
)

ocr_factory = get_ocr_factory()


def create_docling_prediction_provider(
    page_image_scale: float = 2.0,
    do_ocr: bool = False,
    ocr_lang: Optional[List[str]] = None,
    ocr_engine: str = EasyOcrOptions.kind,
    artifacts_path: Optional[Path] = None,
):
    ocr_options: OcrOptions = ocr_factory.create_options(  # type: ignore
        kind=ocr_engine,
    )
    if ocr_lang is not None:
        ocr_options.lang = ocr_lang

    pipeline_options = PdfPipelineOptions(
        do_ocr=do_ocr,
        ocr_options=ocr_options,
        do_table_structure=True,
        artifacts_path=artifacts_path,
    )

    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    pipeline_options.images_scale = page_image_scale
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    return DoclingPredictionProvider(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        },
        do_visualization=True,
    )


def test_run_dpbench_e2e():
    target_path = Path(f"./scratch/{BenchMarkNames.DPBENCH.value}/")
    docling_provider = create_docling_prediction_provider(page_image_scale=2.0)

    dataset_layout = DPBenchDatasetBuilder(
        target=target_path / "gt_dataset",
        end_index=5,
    )

    dataset_layout.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset_layout.save_to_disk(
        chunk_size=5
    )  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    docling_provider.create_prediction_dataset(
        name=dataset_layout.name,
        gt_dataset_dir=target_path / "gt_dataset",
        target_dataset_dir=target_path / "eval_dataset",
    )

    ## Evaluate Layout
    evaluate(
        modality=EvaluationModality.LAYOUT,
        benchmark=BenchMarkNames.DPBENCH,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.LAYOUT.value,
    )

    visualise(
        modality=EvaluationModality.LAYOUT,
        benchmark=BenchMarkNames.DPBENCH,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.LAYOUT.value,
    )

    ## Evaluate Reading order
    evaluate(
        modality=EvaluationModality.READING_ORDER,
        benchmark=BenchMarkNames.DPBENCH,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.READING_ORDER.value,
    )

    visualise(
        modality=EvaluationModality.READING_ORDER,
        benchmark=BenchMarkNames.DPBENCH,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.READING_ORDER.value,
    )

    ## Evaluate Markdown text
    evaluate(
        modality=EvaluationModality.MARKDOWN_TEXT,
        benchmark=BenchMarkNames.DPBENCH,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.MARKDOWN_TEXT.value,
    )

    visualise(
        modality=EvaluationModality.MARKDOWN_TEXT,
        benchmark=BenchMarkNames.DPBENCH,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.MARKDOWN_TEXT.value,
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

    dataset_layout.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset_layout.save_to_disk(
        chunk_size=80
    )  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

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

    visualise(
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

    visualise(
        modality=EvaluationModality.MARKDOWN_TEXT,
        benchmark=BenchMarkNames.DOCLAYNETV1,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.MARKDOWN_TEXT.value,
    )


def test_run_omnidocbench_e2e():
    target_path = Path(f"./scratch/{BenchMarkNames.OMNIDOCBENCH.value}/")
    docling_provider = create_docling_prediction_provider(page_image_scale=2.0)

    dataset_layout = OmniDocBenchDatasetBuilder(
        target=target_path / "gt_dataset",
        end_index=5,
    )

    dataset_layout.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset_layout.save_to_disk(
        chunk_size=5
    )  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

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

    visualise(
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

    visualise(
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

    visualise(
        modality=EvaluationModality.MARKDOWN_TEXT,
        benchmark=BenchMarkNames.OMNIDOCBENCH,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.MARKDOWN_TEXT.value,
    )


def test_run_dpbench_tables():
    target_path = Path(f"./scratch/{BenchMarkNames.DPBENCH.value}/")
    tableformer_provider = TableFormerPredictionProvider()

    dataset_tables = DPBenchDatasetBuilder(
        target=target_path / "gt_dataset",
        end_index=5,
    )

    dataset_tables.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset_tables.save_to_disk(
        chunk_size=5
    )  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    tableformer_provider.create_prediction_dataset(
        name=dataset_tables.name,
        gt_dataset_dir=target_path / "gt_dataset",
        target_dataset_dir=target_path / "eval_dataset",
    )

    evaluate(
        modality=EvaluationModality.TABLE_STRUCTURE,
        benchmark=BenchMarkNames.DPBENCH,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.TABLE_STRUCTURE.value,
    )

    visualise(
        modality=EvaluationModality.TABLE_STRUCTURE,
        benchmark=BenchMarkNames.DPBENCH,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.TABLE_STRUCTURE.value,
    )


def test_run_omnidocbench_tables():
    target_path = Path(f"./scratch/{BenchMarkNames.OMNIDOCBENCH.value}/")
    tableformer_provider = TableFormerPredictionProvider()

    dataset_tables = OmniDocBenchDatasetBuilder(
        target=target_path / "gt_dataset",
        end_index=5,
    )

    dataset_tables.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset_tables.save_to_disk(
        chunk_size=5,
        max_num_chunks=1,
    )  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

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

    visualise(
        modality=EvaluationModality.TABLE_STRUCTURE,
        benchmark=BenchMarkNames.OMNIDOCBENCH,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.TABLE_STRUCTURE.value,
    )


def test_run_doclaynet_v1_e2e():
    target_path = Path(f"./scratch/{BenchMarkNames.DOCLAYNETV1.value}/")
    docling_provider = create_docling_prediction_provider(page_image_scale=2.0)

    dataset_layout = DocLayNetV1DatasetBuilder(
        # prediction_provider=docling_provider,
        target=target_path / "gt_dataset",
        end_index=5,
    )

    dataset_layout.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset_layout.save_to_disk(
        chunk_size=5
    )  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

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

    visualise(
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

    visualise(
        modality=EvaluationModality.MARKDOWN_TEXT,
        benchmark=BenchMarkNames.DOCLAYNETV1,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.MARKDOWN_TEXT.value,
    )


@pytest.mark.skip("Test needs local data which is unavailable.")
def test_run_doclaynet_v2_e2e():
    target_path = Path(f"./scratch/{BenchMarkNames.DOCLAYNETV2.value}/")
    docling_provider = create_docling_prediction_provider(page_image_scale=2.0)

    dataset_layout = DocLayNetV2DatasetBuilder(
        dataset_source=Path("/path/to/doclaynet_v2_benchmark"),
        target=target_path / "gt_dataset",
        end_index=5,
    )

    dataset_layout.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset_layout.save_to_disk(
        chunk_size=5
    )  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

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

    visualise(
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

    visualise(
        modality=EvaluationModality.MARKDOWN_TEXT,
        benchmark=BenchMarkNames.DOCLAYNETV2,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.MARKDOWN_TEXT.value,
    )


def test_run_funsd():
    target_path = Path(f"./scratch/{BenchMarkNames.FUNSD.value}/")

    dataset_layout = FUNSDDatasetBuilder(
        dataset_source=target_path / "input_dataset",
        target=target_path / "gt_dataset",
        end_index=5,
    )

    dataset_layout.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset_layout.save_to_disk(
        chunk_size=5
    )  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.


def test_run_xfund():
    target_path = Path(f"./scratch/{BenchMarkNames.XFUND.value}/")

    dataset_layout = XFUNDDatasetBuilder(
        dataset_source=target_path / "input_dataset",
        target=target_path / "gt_dataset",
        end_index=5,
    )

    dataset_layout.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset_layout.save_to_disk(
        chunk_size=5
    )  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.


def test_run_fintabnet_builder():
    target_path = Path(f"./scratch/{BenchMarkNames.FINTABNET.value}/")
    tableformer_provider = TableFormerPredictionProvider(do_visualization=True)

    dataset = FintabNetDatasetBuilder(
        target=target_path / "gt_dataset",
        end_index=5,
    )

    dataset.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset.save_to_disk(
        chunk_size=5
    )  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

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

    visualise(
        modality=EvaluationModality.TABLE_STRUCTURE,
        benchmark=BenchMarkNames.FINTABNET,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.TABLE_STRUCTURE.value,
    )


def test_run_p1m_builder():
    target_path = Path(f"./scratch/{BenchMarkNames.PUB1M.value}/")
    tableformer_provider = TableFormerPredictionProvider(do_visualization=True)

    dataset = PubTables1MDatasetBuilder(
        target=target_path / "gt_dataset",
        end_index=5,
    )

    dataset.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset.save_to_disk(
        chunk_size=5
    )  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

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

    visualise(
        modality=EvaluationModality.TABLE_STRUCTURE,
        benchmark=BenchMarkNames.PUB1M,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.TABLE_STRUCTURE.value,
    )


def test_run_pubtabnet_builder():
    target_path = Path(f"./scratch/{BenchMarkNames.PUBTABNET.value}/")
    tableformer_provider = TableFormerPredictionProvider(do_visualization=True)

    dataset = PubTabNetDatasetBuilder(
        target=target_path / "gt_dataset",
        end_index=25,
    )

    dataset.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset.save_to_disk(
        chunk_size=80
    )  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

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

    visualise(
        modality=EvaluationModality.TABLE_STRUCTURE,
        benchmark=BenchMarkNames.PUBTABNET,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.TABLE_STRUCTURE.value,
        split="val",
    )
