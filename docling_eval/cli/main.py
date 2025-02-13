import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer
from tabulate import tabulate  # type: ignore

from docling_eval.benchmarks.constants import BenchMarkNames, EvaluationModality
from docling_eval.benchmarks.doclaynet_v1.create import create_dlnv1_e2e_dataset
from docling_eval.benchmarks.dpbench.create import (
    create_dpbench_e2e_dataset,
    create_dpbench_tableformer_dataset,
)
from docling_eval.benchmarks.omnidocbench.create import (
    create_omnidocbench_e2e_dataset,
    create_omnidocbench_tableformer_dataset,
)
from docling_eval.benchmarks.tableformer_huggingface_otsl.create import (
    create_fintabnet_tableformer_dataset,
    create_p1m_tableformer_dataset,
    create_pubtabnet_tableformer_dataset,
)
from docling_eval.evaluators.layout_evaluator import (
    DatasetLayoutEvaluation,
    LayoutEvaluator,
)
from docling_eval.evaluators.markdown_text_evaluator import (
    DatasetMarkdownEvaluation,
    MarkdownTextEvaluator,
)
from docling_eval.evaluators.readingorder_evaluator import (
    DatasetReadingOrderEvaluation,
    ReadingOrderEvaluator,
    ReadingOrderVisualizer,
)
from docling_eval.evaluators.stats import DatasetStatistics
from docling_eval.evaluators.table_evaluator import (
    DatasetTableEvaluation,
    TableEvaluator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

app = typer.Typer(
    name="docling-eval",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


class EvaluationTask(str, Enum):
    CREATE = "create"
    EVALUATE = "evaluate"
    VISUALIZE = "visualize"


def log_and_save_stats(
    odir: Path,
    benchmark: BenchMarkNames,
    modality: EvaluationModality,
    metric: str,
    stats: DatasetStatistics,
) -> tuple[Path, Path]:
    r"""
    For the given DatasetStatistics related to the provided benchmark/modality/metric:
    - Generate a textual table. Log it and save it in a file.
    - Generate a plot and save it in a file.

    The filenames of the generated files are derived by the benchmark/modality/metric

    Returns
    -------
    log_filename: Path of the saved log file
    fig_filename: Path of the saved png file
    """
    log_filename = odir / f"evaluation_{benchmark.value}_{modality.value}_{metric}.txt"
    fig_filename = odir / f"evaluation_{benchmark.value}_{modality.value}_{metric}.png"

    data, headers = stats.to_table(metric)
    content = f"{benchmark.value} {modality.value} {metric}:\n\n"
    content += "mean={:.2f} median={:.2f} std={:.2f}\n\n".format(
        stats.mean, stats.median, stats.std
    )
    content += tabulate(data, headers=headers, tablefmt="github")

    log.info(content)
    with open(log_filename, "w") as fd:
        fd.write(content)
        log.info("Saving statistics report to %s", log_filename)
    stats.save_histogram(figname=fig_filename, name=metric)

    return log_filename, fig_filename


def create(
    modality: EvaluationModality,
    benchmark: BenchMarkNames,
    odir: Path,
    idir: Optional[Path] = None,
    image_scale: float = 1.0,
    artifacts_path: Optional[Path] = None,
    split: str = "test",
    max_items: int = 1000,
):
    r""""""
    if odir is None:
        odir = Path("./benchmarks") / benchmark.value / modality.value

    if benchmark == BenchMarkNames.DPBENCH:
        if idir is None:
            log.error("The input dir for %s must be provided", BenchMarkNames.DPBENCH)
        assert idir is not None

        if (
            modality == EvaluationModality.END2END
            or modality == EvaluationModality.LAYOUT
        ):
            # No support for max_items
            create_dpbench_e2e_dataset(
                dpbench_dir=idir,
                output_dir=odir,
                image_scale=image_scale,
                do_viz=True,
            )

        elif modality == EvaluationModality.TABLEFORMER:
            # No support for max_items
            create_dpbench_tableformer_dataset(
                dpbench_dir=idir,
                output_dir=odir,
                image_scale=image_scale,
                artifacts_path=artifacts_path,
            )

        else:
            log.error(f"{modality} is not yet implemented for {benchmark}")

    elif benchmark == BenchMarkNames.OMNIDOCBENCH:
        if idir is None:
            log.error("The input dir for %s must be provided", BenchMarkNames.DPBENCH)
        assert idir is not None

        if (
            modality == EvaluationModality.END2END
            or modality == EvaluationModality.LAYOUT
        ):
            # No support for max_items
            create_omnidocbench_e2e_dataset(
                omnidocbench_dir=idir, output_dir=odir, image_scale=image_scale
            )
        elif modality == EvaluationModality.TABLEFORMER:
            # No support for max_items
            create_omnidocbench_tableformer_dataset(
                omnidocbench_dir=idir,
                output_dir=odir,
                image_scale=image_scale,
                artifacts_path=artifacts_path,
            )
        else:
            log.error(f"{modality} is not yet implemented for {benchmark}")

    elif benchmark == BenchMarkNames.PUBTABNET:
        if modality == EvaluationModality.TABLEFORMER:
            log.info("Create the tableformer converted PubTabNet dataset")
            create_pubtabnet_tableformer_dataset(
                output_dir=odir,
                max_items=max_items,
                do_viz=True,
                artifacts_path=artifacts_path,
            )
        else:
            log.error(f"{modality} is not yet implemented for {benchmark}")

    elif benchmark == BenchMarkNames.FINTABNET:
        if modality == EvaluationModality.TABLEFORMER:
            log.info("Create the tableformer converted FinTabNet dataset")
            create_fintabnet_tableformer_dataset(
                output_dir=odir,
                max_items=max_items,
                do_viz=True,
                artifacts_path=artifacts_path,
            )
        else:
            log.error(f"{modality} is not yet implemented for {benchmark}")

    elif benchmark == BenchMarkNames.PUB1M:
        if modality == EvaluationModality.TABLEFORMER:
            log.info("Create the tableformer converted Pub1M dataset")
            create_p1m_tableformer_dataset(
                output_dir=odir,
                max_items=max_items,
                do_viz=True,
                artifacts_path=artifacts_path,
            )
        else:
            log.error(f"{modality} is not yet implemented for {benchmark}")

    elif benchmark == BenchMarkNames.DOCLAYNETV1:
        if modality == EvaluationModality.LAYOUT:
            create_dlnv1_e2e_dataset(
                name="ds4sd/DocLayNet-v1.2",
                split=split,
                output_dir=odir,
                do_viz=True,
                max_items=max_items,
            )
        else:
            log.error(f"{modality} is not yet implemented for {benchmark}")

    else:
        log.error(f"{benchmark} is not yet implemented")


def evaluate(
    modality: EvaluationModality,
    benchmark: BenchMarkNames,
    idir: Path,
    odir: Path,
    split: str = "test",
):
    r""""""
    if not os.path.exists(idir):
        log.error(f"Benchmark directory not found: {idir}")

    # Save the evaluation
    save_fn = odir / f"evaluation_{benchmark.value}_{modality.value}.json"

    if modality == EvaluationModality.END2END:
        log.error("not supported")

    elif modality == EvaluationModality.LAYOUT:
        layout_evaluator = LayoutEvaluator()
        layout_evaluation = layout_evaluator(idir, split=split)

        with open(save_fn, "w") as fd:
            json.dump(layout_evaluation.model_dump(), fd, indent=2, sort_keys=True)

    elif modality == EvaluationModality.TABLEFORMER:
        table_evaluator = TableEvaluator()
        table_evaluation = table_evaluator(idir, split=split)

        with open(save_fn, "w") as fd:
            json.dump(table_evaluation.model_dump(), fd, indent=2, sort_keys=True)

    elif modality == EvaluationModality.READING_ORDER:
        readingorder_evaluator = ReadingOrderEvaluator()
        readingorder_evaluation = readingorder_evaluator(idir, split=split)

        with open(save_fn, "w") as fd:
            json.dump(
                readingorder_evaluation.model_dump(),
                fd,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )

    elif modality == EvaluationModality.MARKDOWN_TEXT:
        md_evaluator = MarkdownTextEvaluator()
        md_evaluation = md_evaluator(idir, split=split)

        with open(save_fn, "w") as fd:
            json.dump(
                md_evaluation.model_dump(),
                fd,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )

    elif modality == EvaluationModality.CODEFORMER:
        pass

    log.info("The evaluation has been saved in '%s'", save_fn)


def visualise(
    modality: EvaluationModality,
    benchmark: BenchMarkNames,
    idir: Path,
    odir: Path,
    split: str = "test",
):

    metrics_filename = odir / f"evaluation_{benchmark.value}_{modality.value}.json"

    if modality == EvaluationModality.END2END:
        pass

    elif modality == EvaluationModality.LAYOUT:
        with open(metrics_filename, "r") as fd:
            layout_evaluation = DatasetLayoutEvaluation.parse_file(metrics_filename)

        # Save layout statistics for mAP
        log_filename, _ = log_and_save_stats(
            odir, benchmark, modality, "mAP_0.5_0.95", layout_evaluation.mAP_stats
        )

        # Append to layout statistics the mAP classes
        data, headers = layout_evaluation.to_table()
        content = "\n\n\nClass mAP[0.5:0.95] table:\n\n"
        content += tabulate(data, headers=headers, tablefmt="github")
        log.info(content)
        with open(log_filename, "a") as fd:
            fd.write(content)

    elif modality == EvaluationModality.TABLEFORMER:
        with open(metrics_filename, "r") as fd:
            table_evaluation = DatasetTableEvaluation.parse_file(metrics_filename)

        figname = (
            odir / f"evaluation_{benchmark.value}_{modality.value}-delta_row_col.png"
        )
        table_evaluation.save_histogram_delta_row_col(figname=figname)

        # TEDS struct-with-text
        log_and_save_stats(
            odir, benchmark, modality, "TEDS_struct-with-text", table_evaluation.TEDS
        )

        # TEDS struct-only
        log_and_save_stats(
            odir, benchmark, modality, "TEDS_struct-only", table_evaluation.TEDS_struct
        )

    elif modality == EvaluationModality.READING_ORDER:
        with open(metrics_filename, "r") as fd:
            ro_evaluation = DatasetReadingOrderEvaluation.parse_file(metrics_filename)
        # ARD
        log_and_save_stats(
            odir, benchmark, modality, "ARD_norm", ro_evaluation.ard_stats
        )

        # Weighted ARD
        log_and_save_stats(
            odir, benchmark, modality, "weighted_ARD", ro_evaluation.w_ard_stats
        )

        # Generate visualizations of the reading order across the GT and the prediction
        ro_visualizer = ReadingOrderVisualizer()
        ro_visualizer(idir, metrics_filename, odir, split=split)

    elif modality == EvaluationModality.MARKDOWN_TEXT:
        with open(metrics_filename, "r") as fd:
            md_evaluation = DatasetMarkdownEvaluation.parse_file(metrics_filename)
        log_and_save_stats(odir, benchmark, modality, "BLEU", md_evaluation.bleu_stats)

    elif modality == EvaluationModality.CODEFORMER:
        pass


@app.command(no_args_is_help=True)
def main(
    task: Annotated[
        EvaluationTask,
        typer.Option(
            ...,  # EvaluationTask.CREATE,
            "-t",  # Short name
            "--task",  # Long name
            help="Evaluation task",
        ),
    ],
    modality: Annotated[
        EvaluationModality,
        typer.Option(
            ...,  # EvaluationModality.TABLEFORMER,
            "-m",  # Short name
            "--modality",  # Long name
            help="Evaluation modality",
        ),
    ],
    benchmark: Annotated[
        BenchMarkNames,
        typer.Option(
            ...,  # BenchMarkNames.DPBENCH,
            "-b",  # Short name
            "--benchmark",  # Long name
            help="Benchmark name",
        ),
    ],
    odir: Annotated[
        Path,
        typer.Option(
            ...,
            "-o",  # Short name
            "--output-dir",  # Long name
            help="Output directory",
        ),
    ],
    idir: Annotated[
        Optional[Path],
        typer.Option(
            ...,
            "-i",  # Short name
            "--input-dir",  # Long name
            help="Input directory",
        ),
    ] = None,
    split: Annotated[
        str,
        typer.Option(
            ...,
            "-s",  # Short name
            "--split",  # Long name
            help="Dataset split",
        ),
    ] = "test",
    artifacts_path: Annotated[
        Optional[Path],
        typer.Option(
            ...,
            "-a",  # Short name
            "--artifacts-path",  # Long name
            help="Load artifacts from local path",
        ),
    ] = None,
    max_items: Annotated[
        int,
        typer.Option(
            ...,
            "-n",  # Short name
            "--max-items",  # Long name
            help="How many items to load from the original dataset",
        ),
    ] = 1000,
):
    # Dispatch the command
    if task == EvaluationTask.CREATE:
        create(
            modality,
            benchmark,
            odir,
            idir=idir,
            artifacts_path=artifacts_path,
            split=split,
            max_items=max_items,
        )

    elif task == EvaluationTask.EVALUATE:
        assert idir is not None
        evaluate(modality, benchmark, idir, odir, split)

    elif task == EvaluationTask.VISUALIZE:
        assert idir is not None
        visualise(modality, benchmark, idir, odir, split)

    else:
        log.error("Unsupported command: '%s'", task.value)


if __name__ == "__main__":
    app()
