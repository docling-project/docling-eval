import argparse
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


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Evaluate dataset using CVAT annotation files."
#     )
#     parser.add_argument(
#         "-i",
#         "--input_dir",
#         required=True,
#         help="Path to the input directory (contains `test` and `train` with parquet files)",
#     )

#     args = parser.parse_args()
#     return Path(args.input_dir)


# def main():

#     odir_lay = parse_args()

#     # Layout
#     log.info("Evaluate the layout for the DP-Bench dataset")
#     evaluate(
#         modality=EvaluationModality.LAYOUT,
#         benchmark=BenchMarkNames.DPBENCH,
#         idir=odir_lay,
#         odir=odir_lay,
#     )
#     log.info("Visualize the layout for the DP-Bench dataset")
#     visualise(
#         modality=EvaluationModality.LAYOUT,
#         benchmark=BenchMarkNames.DPBENCH,
#         idir=odir_lay,
#         odir=odir_lay,
#     )


# if __name__ == "__main__":
#     main()
