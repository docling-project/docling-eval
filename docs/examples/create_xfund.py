import logging
import os
from pathlib import Path

from docling_eval.datamodels.types import BenchMarkNames
from docling_eval.legacy.xfund.create import create_xfund_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
)

log = logging.getLogger(__name__)


def main():
    idir = Path(f"./benchmarks/{BenchMarkNames.XFUND.value}-original")
    odir = Path(f"./benchmarks/{BenchMarkNames.XFUND.value}-dataset")

    os.makedirs(odir, exist_ok=True)

    log.info("Create the converted XFUND dataset")
    create_xfund_dataset(input_dir=idir, output_dir=odir, splits=["train", "val"])


if __name__ == "__main__":
    main()
