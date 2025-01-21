import logging
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from tabulate import tabulate  # type: ignore

from docling_eval.benchmarks.constants import BenchMarkNames, EvaluationModality

from docling_eval.cli.main import evaluate, visualise

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
)

log = logging.getLogger(__name__)


def main():

    odir_lay = Path("./benchmarks/DPBench-annotations-v03/datasets")
    
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

if __name__ == "__main__":
    main()    
