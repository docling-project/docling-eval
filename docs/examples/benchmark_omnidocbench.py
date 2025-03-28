import logging
import os
from pathlib import Path

from huggingface_hub import snapshot_download

from docling_eval.datamodels.types import BenchMarkNames, EvaluationModality
from docling_eval.legacy.main import evaluate, visualise
from docling_eval.legacy.omnidocbench.create import (
    create_omnidocbench_e2e_dataset,
    create_omnidocbench_tableformer_dataset,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
)

log = logging.getLogger(__name__)


def main():

    idir = Path(f"./benchmarks/{BenchMarkNames.OMNIDOCBENCH.value}-original")

    snapshot_download(
        repo_id="opendatalab/OmniDocBench",
        repo_type="dataset",
        local_dir=idir,
    )

    odir = Path(f"./benchmarks/{BenchMarkNames.OMNIDOCBENCH.value}-dataset")

    odir_lay = Path(odir) / "layout"
    odir_tab = Path(odir) / "tableformer"

    for _ in [odir, odir_lay, odir_tab]:
        os.makedirs(_, exist_ok=True)

    image_scale = 2.0

    if True:
        log.info("Create the end-to-end converted OmniDocBench dataset")
        create_omnidocbench_e2e_dataset(
            omnidocbench_dir=idir,
            output_dir=odir_lay,
            image_scale=image_scale,
            do_viz=True,
        )

        # Layout
        log.info("Evaluate the layout for the OmniDocBench dataset")
        evaluate(
            modality=EvaluationModality.LAYOUT,
            benchmark=BenchMarkNames.OMNIDOCBENCH,
            idir=odir_lay,
            odir=odir_lay,
        )
        visualise(
            modality=EvaluationModality.LAYOUT,
            benchmark=BenchMarkNames.OMNIDOCBENCH,
            idir=odir_lay,
            odir=odir_lay,
        )

        # Reading order
        log.info("Evaluate the reading order for the OmniDocBench dataset")
        evaluate(
            modality=EvaluationModality.READING_ORDER,
            benchmark=BenchMarkNames.OMNIDOCBENCH,
            idir=odir_lay,
            odir=odir_lay,
        )
        visualise(
            modality=EvaluationModality.READING_ORDER,
            benchmark=BenchMarkNames.OMNIDOCBENCH,
            idir=odir_lay,
            odir=odir_lay,
        )

        # Markdown text
        log.info("Evaluate the markdown text for the OmniDocBench dataset")
        evaluate(
            modality=EvaluationModality.MARKDOWN_TEXT,
            benchmark=BenchMarkNames.OMNIDOCBENCH,
            idir=odir_lay,
            odir=odir_lay,
        )
        log.info("Visualize the markdown text for the OmniDocBench dataset")
        visualise(
            modality=EvaluationModality.MARKDOWN_TEXT,
            benchmark=BenchMarkNames.OMNIDOCBENCH,
            idir=odir_lay,
            odir=odir_lay,
        )

    if True:
        log.info("Create the tableformer converted OmniDocBench dataset")
        create_omnidocbench_tableformer_dataset(
            omnidocbench_dir=idir, output_dir=odir_tab, image_scale=image_scale
        )

        log.info("Evaluate the tableformer for the OmniDocBench dataset")
        evaluate(
            modality=EvaluationModality.TABLE_STRUCTURE,
            benchmark=BenchMarkNames.OMNIDOCBENCH,
            idir=odir_tab,
            odir=odir_tab,
        )
        log.info("Visualize the tableformer for the OmniDocBench dataset")
        visualise(
            modality=EvaluationModality.TABLE_STRUCTURE,
            benchmark=BenchMarkNames.OMNIDOCBENCH,
            idir=odir_tab,
            odir=odir_tab,
        )


if __name__ == "__main__":
    main()
