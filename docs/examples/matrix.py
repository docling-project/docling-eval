import argparse
import logging
from pathlib import Path
from typing import List, Optional

from docling_eval.aggregations.consolidator import Consolidator
from docling_eval.aggregations.multi_evalutor import MultiEvaluator
from docling_eval.datamodels.types import (
    BenchMarkNames,
    EvaluationModality,
    PredictionProviderType,
)

# Configure logging
logging.getLogger("docling").setLevel(logging.WARNING)
_log = logging.getLogger(__name__)


def evaluate(
    root_dir: Path,
    benchmarks: List[BenchMarkNames],
    experiments: List[str],
    modalities: List[EvaluationModality],
):
    r""" """
    # Create multi evaluations
    me: MultiEvaluator = MultiEvaluator(root_dir)

    _log.info("Evaluating...")
    m_evals = me(experiments, benchmarks, modalities)
    _log.info("Finish evaluation")


def consolidate(
    working_dir: Path,
):
    r""" """
    multi_evaluation = MultiEvaluator.load_multi_evaluation(working_dir)
    consolidator = Consolidator(working_dir / "consolidation")

    _log.info("Consolidating...")
    dfs, produced_file = consolidator(multi_evaluation)
    _log.info("Finish consolidation")


def main(args):
    r""" """
    task = args.task
    working_dir = Path(args.working_dir)
    benchmarks = (
        [BenchMarkNames(x) for x in args.benchmarks.split(",")]
        if args.benchmarks
        else None
    )
    experiments_or_providers = (
        args.experiments_or_providers.split(",")
        if args.experiments_or_providers
        else None
    )
    modalities = (
        [EvaluationModality(x) for x in args.modalities.split(",")]
        if args.modalities
        else None
    )

    if task == "evaluate":
        if not benchmarks or not experiments_or_providers or not modalities:
            _log.error("Required Benchmarks/Experiments/Modalities")
            return
        evaluate(working_dir, benchmarks, experiments_or_providers, modalities)
    elif task == "consolidate":
        consolidate(working_dir)
    elif task == "both":
        if not benchmarks or not experiments_or_providers or not modalities:
            _log.error("Required Benchmarks/Providers/Modalities")
            return
        evaluate(working_dir, benchmarks, experiments_or_providers, modalities)
        consolidate(working_dir)
    else:
        _log.error("Unsupported task: %s", task)


if __name__ == "__main__":
    desription = """
    Running multi-evaluation and consolidation inside a working directory and generate matrix reports
    
    The working directory must have the structure:
    .
    ├── consolidation
    │   └── consolidation_matrix.xlsx
    └── <benchmark_name>
        ├── gt_dataset [Dir with dataset in parquet format with the ground truth DoclingDocuments]
        ├── <experiment_name1> [It can be the name of a provider or anything else]
        │   ├── eval_dataset
        │   └── evaluations
        │        ├── <modality1>
        │        │   └── evaluation_<benchmark>_<modality1>.json
        │        └── <modality2>
        │            └── evaluation_<benchmark>_<modality2>.json
        └── <experiment_name2> [It can be the name of a provider or anything else]
            ├── eval_dataset
            └── evaluations
                 └── <modality1>
                     └── evaluation_<benchmark>_<modality1>.json
    """
    parser = argparse.ArgumentParser(
        description=desription, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-t",
        "--task",
        required=True,
        help="One of ['evaluate', 'consolidate', 'both']",
    )
    parser.add_argument(
        "-d",
        "--working_dir",
        required=True,
        help="Working directory",
    )
    parser.add_argument(
        "-b",
        "--benchmarks",
        required=False,
        default=None,
        help=f"Evaluate: Comma separated list of {[x.value for x in BenchMarkNames]}",
    )
    parser.add_argument(
        "-e",
        "--experiments_or_providers",
        required=False,
        default=None,
        help=f"Evaluate: Comma separated list of experiments or providers names.",
    )
    parser.add_argument(
        "-m",
        "--modalities",
        required=False,
        default=None,
        help=f"Evaluate: Comma separated list of {[x.value for x in EvaluationModality]}",
    )
    args = parser.parse_args()
    main(args)
