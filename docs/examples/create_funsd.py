import logging
import os
from pathlib import Path

from docling_eval.datamodels.types import BenchMarkNames
from docling_eval.legacy.funsd.create import create_funsd_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
)

log = logging.getLogger(__name__)


def main():
    idir = Path(f"./benchmarks/{BenchMarkNames.FUNSD.value}-original")
    odir = Path(f"./benchmarks/{BenchMarkNames.FUNSD.value}-dataset")

    os.makedirs(odir, exist_ok=True)

    log.info("Create the converted FUNSDF dataset")
    create_funsd_dataset(input_dir=idir, output_dir=odir, splits=["train", "test"])


if __name__ == "__main__":
    main()
