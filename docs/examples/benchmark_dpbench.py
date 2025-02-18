import logging
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from tabulate import tabulate  # type: ignore

from docling_eval.benchmarks.constants import BenchMarkNames, EvaluationModality
from docling_eval.benchmarks.dpbench.create import (
    create_dpbench_e2e_dataset,
    create_dpbench_tableformer_dataset,
)
from docling_eval.cli.main import evaluate, visualise

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
)

log = logging.getLogger(__name__)


def main():

    idir = Path(f"./benchmarks/{BenchMarkNames.DPBENCH.value}-original")

    log.info("Download the DP-Bench dataset")
    snapshot_download(
        repo_id="upstage/dp-bench",
        repo_type="dataset",
        local_dir=idir,
    )

    odir = Path(f"./benchmarks/{BenchMarkNames.DPBENCH.value}-dataset")

    odir_lay = Path(odir) / "layout"
    odir_tab = Path(odir) / "tableformer"

    for _ in [odir, odir_lay, odir_tab]:
        os.makedirs(_, exist_ok=True)

    image_scale = 2.0

    if True:
        log.info("Create the end-to-end converted DP-Bench dataset")
        create_dpbench_e2e_dataset(
            dpbench_dir=idir, output_dir=odir_lay, image_scale=image_scale, do_viz=True
        )

        # Layout
        log.info("Evaluate the layout for the DP-Bench dataset")
        evaluate(
            modality=EvaluationModality.LAYOUT,
            benchmark=BenchMarkNames.DPBENCH,
            idir=odir_lay,
            odir=odir_lay,
        )
        log.info("Visualize the layout for the DP-Bench dataset")
        visualise(
            modality=EvaluationModality.LAYOUT,
            benchmark=BenchMarkNames.DPBENCH,
            idir=odir_lay,
            odir=odir_lay,
        )

        # Reading order
        log.info("Evaluate the reading-order for the DP-Bench dataset")
        evaluate(
            modality=EvaluationModality.READING_ORDER,
            benchmark=BenchMarkNames.DPBENCH,
            idir=odir_lay,
            odir=odir_lay,
        )
        log.info("Visualize the reading-order for the DP-Bench dataset")
        visualise(
            modality=EvaluationModality.READING_ORDER,
            benchmark=BenchMarkNames.DPBENCH,
            idir=odir_lay,
            odir=odir_lay,
        )

        # Markdown text
        log.info("Evaluate the markdown text for the DP-Bench dataset")
        evaluate(
            modality=EvaluationModality.MARKDOWN_TEXT,
            benchmark=BenchMarkNames.DPBENCH,
            idir=odir_lay,
            odir=odir_lay,
        )
        log.info("Visualize the markdown text for the DP-Bench dataset")
        visualise(
            modality=EvaluationModality.MARKDOWN_TEXT,
            benchmark=BenchMarkNames.DPBENCH,
            idir=odir_lay,
            odir=odir_lay,
        )

    if True:
        log.info("Create the tableformer converted DP-Bench dataset")
        create_dpbench_tableformer_dataset(
            dpbench_dir=idir, output_dir=odir_tab, image_scale=image_scale
        )

        log.info("Evaluate the tableformer for the DP-Bench dataset")
        evaluate(
            modality=EvaluationModality.TABLE_STRUCTURE,
            benchmark=BenchMarkNames.DPBENCH,
            idir=odir_tab,
            odir=odir_tab,
        )
        log.info("Visualize the tableformer for the DP-Bench dataset")
        visualise(
            modality=EvaluationModality.TABLE_STRUCTURE,
            benchmark=BenchMarkNames.DPBENCH,
            idir=odir_tab,
            odir=odir_tab,
        )


if __name__ == "__main__":
    main()
