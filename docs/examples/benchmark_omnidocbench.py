import logging
import os
from pathlib import Path

from tabulate import tabulate  # type: ignore

from docling_eval.benchmarks.constants import BenchMarkNames, EvaluationModality
from docling_eval.benchmarks.omnidocbench.create import (
    create_omnidocbench_layout_dataset,
    create_omnidocbench_tableformer_dataset,
)
from docling_eval.cli.main import evaluate, visualise
from docling_eval.utils.repository import clone_repository, is_git_lfs_installed

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():

    REPO_URL = "https://huggingface.co/datasets/opendatalab/OmniDocBench"  # Replace with your repo URL

    idir = Path(f"./benchmarks/{BenchMarkNames.OMNIDOCBENCH.value}-original")

    if is_git_lfs_installed():
        clone_repository(REPO_URL, idir)
    else:
        logger.error("Please install Git LFS and try again.")
        exit()

    odir = Path(f"./benchmarks/{BenchMarkNames.OMNIDOCBENCH.value}-dataset")

    odir_lay = Path(odir) / "layout"
    odir_tab = Path(odir) / "tableformer"

    for _ in [odir, odir_lay, odir_tab]:
        os.makedirs(_, exist_ok=True)

    image_scale = 1.0

    if True:
        create_omnidocbench_layout_dataset(
            omnidocbench_dir=idir, output_dir=odir_lay, image_scale=image_scale
        )

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

    if True:
        create_omnidocbench_tableformer_dataset(
            omnidocbench_dir=idir, output_dir=odir_tab, image_scale=image_scale
        )

        evaluate(
            modality=EvaluationModality.TABLEFORMER,
            benchmark=BenchMarkNames.OMNIDOCBENCH,
            idir=odir_tab,
            odir=odir_tab,
        )
        visualise(
            modality=EvaluationModality.TABLEFORMER,
            benchmark=BenchMarkNames.OMNIDOCBENCH,
            idir=odir_tab,
            odir=odir_tab,
        )


if __name__ == "__main__":
    main()
