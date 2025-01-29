import logging
import os
from pathlib import Path

from tabulate import tabulate  # type: ignore

from docling_eval.benchmarks.constants import BenchMarkNames, EvaluationModality
from docling_eval.benchmarks.doclaynet_v1.create import create_dlnv1_e2e_dataset
from docling_eval.cli.main import evaluate, visualise

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
)

log = logging.getLogger(__name__)


def main():
    odir = Path(f"./benchmarks/{BenchMarkNames.DOCLAYNETV1.value}-dataset")
    split = "train"

    os.makedirs(odir, exist_ok=True)

    if True:
        log.info("Create the end-to-end converted DocLayNetV2 dataset")
        create_dlnv1_e2e_dataset(split=split, output_dir=odir)

        # Layout
        log.info("Evaluate the layout for the DocLayNet dataset")
        evaluate(
            modality=EvaluationModality.LAYOUT,
            benchmark=BenchMarkNames.DOCLAYNETV1,
            idir=odir,
            odir=odir,
        )
        log.info("Visualize the layout for the DocLayNet dataset")
        visualise(
            modality=EvaluationModality.LAYOUT,
            benchmark=BenchMarkNames.DOCLAYNETV1,
            idir=odir,
            odir=odir,
        )
        # Reading order
        log.info("Evaluate the reading-order for the DocLayNet dataset")
        evaluate(
            modality=EvaluationModality.READING_ORDER,
            benchmark=BenchMarkNames.DOCLAYNETV1,
            idir=odir,
            odir=odir,
        )
        log.info("Visualize the reading-order for the DocLayNet dataset")
        visualise(
            modality=EvaluationModality.READING_ORDER,
            benchmark=BenchMarkNames.DOCLAYNETV1,
            idir=odir,
            odir=odir,
        )


if __name__ == "__main__":
    main()
