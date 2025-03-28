import logging
import os
from pathlib import Path

from tabulate import tabulate  # type: ignore

from docling_eval.datamodels.types import BenchMarkNames, EvaluationModality
from docling_eval.legacy.doclaynet_v2.create import create_dlnv2_e2e_dataset
from docling_eval.legacy.main import evaluate, visualise

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
)

log = logging.getLogger(__name__)


def main():
    idir = Path(f"./benchmarks/{BenchMarkNames.DOCLAYNETV2.value}-original")
    odir = Path(f"./benchmarks/{BenchMarkNames.DOCLAYNETV2.value}-dataset")
    odir_lay = Path(odir) / "layout"
    split = "test"

    os.makedirs(odir, exist_ok=True)

    if True:
        log.info("Create the end-to-end converted DocLayNetV2 dataset")
        create_dlnv2_e2e_dataset(
            input_dir=idir,
            split=split,
            do_viz=True,
            output_dir=odir_lay,
            max_items=1000,
        )

        # Layout
        log.info("Evaluate the layout for the DocLayNet dataset")
        evaluate(
            modality=EvaluationModality.LAYOUT,
            benchmark=BenchMarkNames.DOCLAYNETV2,
            idir=odir_lay,
            odir=odir_lay,
        )
        log.info("Visualize the layout for the DocLayNet dataset")
        visualise(
            modality=EvaluationModality.LAYOUT,
            benchmark=BenchMarkNames.DOCLAYNETV2,
            idir=odir_lay,
            odir=odir_lay,
        )

        # Markdown text
        log.info("Evaluate the markdown text for the DocLayNet dataset")
        evaluate(
            modality=EvaluationModality.MARKDOWN_TEXT,
            benchmark=BenchMarkNames.DOCLAYNETV2,
            idir=odir_lay,
            odir=odir_lay,
        )
        log.info("Visualize the markdown text for the DocLayNet dataset")
        visualise(
            modality=EvaluationModality.MARKDOWN_TEXT,
            benchmark=BenchMarkNames.DOCLAYNETV2,
            idir=odir_lay,
            odir=odir_lay,
        )


if __name__ == "__main__":
    main()
