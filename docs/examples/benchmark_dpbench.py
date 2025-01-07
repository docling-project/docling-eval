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

logger = logging.getLogger(__name__)


def main():

    # REPO_URL = "https://huggingface.co/datasets/upstage/dp-bench.git"  # Replace with your repo URL

    idir = Path(f"./benchmarks/{BenchMarkNames.DPBENCH.value}-original")

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

    image_scale = 1.0

    if True:
        create_dpbench_e2e_dataset(
            dpbench_dir=idir, output_dir=odir_lay, image_scale=image_scale
        )

        evaluate(
            modality=EvaluationModality.LAYOUT,
            benchmark=BenchMarkNames.DPBENCH,
            idir=odir_lay,
            odir=odir_lay,
        )
        visualise(
            modality=EvaluationModality.LAYOUT,
            benchmark=BenchMarkNames.DPBENCH,
            idir=odir_lay,
            odir=odir_lay,
        )

    if True:
        create_dpbench_tableformer_dataset(
            dpbench_dir=idir, output_dir=odir_tab, image_scale=image_scale
        )

        evaluate(
            modality=EvaluationModality.TABLEFORMER,
            benchmark=BenchMarkNames.DPBENCH,
            idir=odir_tab,
            odir=odir_tab,
        )
        visualise(
            modality=EvaluationModality.TABLEFORMER,
            benchmark=BenchMarkNames.DPBENCH,
            idir=odir_tab,
            odir=odir_tab,
        )


if __name__ == "__main__":
    main()
