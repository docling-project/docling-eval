import argparse
import logging
from pathlib import Path

from docling_eval.cli.main import evaluate
from docling_eval.datamodels.types import BenchMarkNames, EvaluationModality

_log = logging.getLogger(__name__)


def evaluate_external_predictions(
    benchmark: BenchMarkNames,
    modality: EvaluationModality,
    gt_path: Path,
    predictions_dir: Path,
    save_dir: Path,
):
    r""" """
    evaluate(
        modality,
        benchmark,
        gt_path,
        save_dir,
        external_predictions_path=predictions_dir,
    )


def main():
    r""" """
    parser = argparse.ArgumentParser(
        description="Example how to use GT from parquet and predictions from externally provided prediction files",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        required=True,
        type=BenchMarkNames,
        help="Evaluation modality",
    )
    parser.add_argument(
        "-m",
        "--modality",
        required=True,
        type=EvaluationModality,
        help="Evaluation modality",
    )
    parser.add_argument(
        "-g",
        "--gt_parquet_dir",
        required=True,
        type=Path,
        help="Path to the parquet GT dataset",
    )
    parser.add_argument(
        "-p",
        "--predictions_dir",
        required=True,
        type=Path,
        help="Dir with the external prediction files (json, dt, yaml)",
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        required=False,
        type=Path,
        help="Path to save the produced evaluation files",
    )
    args = parser.parse_args()

    # Configure logger
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    evaluate_external_predictions(
        args.benchmark,
        args.modality,
        args.gt_parquet_dir,
        args.predictions_dir,
        args.save_dir,
    )


if __name__ == "__main__":
    main()
