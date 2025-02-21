import logging
import os
from pathlib import Path

from docling_eval.benchmarks.constants import BenchMarkNames, EvaluationModality
from docling_eval.benchmarks.tableformer_huggingface_otsl.create import (
    create_p1m_tableformer_dataset,
)
from docling_eval.cli.main import evaluate, visualise

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


def main():
    odir = Path(f"./benchmarks/{BenchMarkNames.PUB1M.value}-dataset")
    odir_tab = Path(odir) / "tableformer"

    for _ in [odir, odir_tab]:
        os.makedirs(_, exist_ok=True)

    if True:
        log.info("Create the tableformer converted Pub1M dataset")
        create_p1m_tableformer_dataset(output_dir=odir_tab, max_items=1000, do_viz=True)

        log.info("Evaluate the tableformer converted Pub1M dataset")
        evaluate(
            modality=EvaluationModality.TABLE_STRUCTURE,
            benchmark=BenchMarkNames.PUB1M,
            idir=odir_tab,
            odir=odir_tab,
        )

        log.info("Visualize the tableformer converted Pub1M dataset")
        visualise(
            modality=EvaluationModality.TABLE_STRUCTURE,
            benchmark=BenchMarkNames.PUB1M,
            idir=odir_tab,
            odir=odir_tab,
        )


if __name__ == "__main__":
    main()
