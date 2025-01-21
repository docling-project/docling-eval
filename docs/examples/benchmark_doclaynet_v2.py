import logging
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from tabulate import tabulate  # type: ignore

from docling_eval.benchmarks.constants import BenchMarkNames, EvaluationModality
from docling_eval.benchmarks.doclaynet_v2.create import create_dlnv2_e2e_dataset
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

logger = logging.getLogger(__name__)


def main():
    idir = Path(f"./benchmarks/{BenchMarkNames.DOCLAYNETV2.value}-original")
    odir = Path(f"./benchmarks/{BenchMarkNames.DOCLAYNETV2.value}-dataset")

    os.makedirs(odir, exist_ok=True)

    if True:
        create_dlnv2_e2e_dataset(input_dir=idir, output_dir=odir)

        evaluate(
            modality=EvaluationModality.LAYOUT,
            benchmark=BenchMarkNames.DOCLAYNETV2,
            idir=odir,
            odir=odir,
        )
        visualise(
            modality=EvaluationModality.LAYOUT,
            benchmark=BenchMarkNames.DOCLAYNETV2,
            idir=odir,
            odir=odir,
        )


if __name__ == "__main__":
    main()
