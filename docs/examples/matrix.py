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
    providers: List[PredictionProviderType],
    modalities: List[EvaluationModality],
):
    r""" """
    # Create multi evaluations
    me: MultiEvaluator = MultiEvaluator(root_dir)

    _log.info("Evaluating...")
    m_evals = me(providers, benchmarks, modalities)
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
        [BenchMarkNames(x) for x in args.benchmarks] if args.benchmarks else None
    )
    providers = (
        [PredictionProviderType(x) for x in args.providers] if args.providers else None
    )
    modalities = (
        [EvaluationModality(x) for x in args.modalities] if args.modalities else None
    )

    if task == "evaluate":
        if not benchmarks or not providers or not modalities:
            _log.error("Required Benchmarks/Providers/Modalities")
            return
        evaluate(working_dir, benchmarks, providers, modalities)
    elif task == "consolidate":
        consolidate(working_dir)
    elif task == "both":
        if not benchmarks or not providers or not modalities:
            _log.error("Required Benchmarks/Providers/Modalities")
            return
        evaluate(working_dir, benchmarks, providers, modalities)
        consolidate(working_dir)
    else:
        _log.error("Unsupported task: %s", task)


if __name__ == "__main__":
    desription = """
    Running multi-evaluation and consolidation inside a working directory and generate matrix reports
    
    The working directory must have the structure:

    <working_dir_root>
        |- "consolidation": Output: Generated consolidation reports (e.g. excel files)
        |- [dataset_name]: Input/Output: One of the BenchMarkNames.
              |- "_GT_": Input/Output: Reserved directory for the ground truth data for this dataset.
              |- [experiment_name]: Input/Output: Can be a provider's name or anything else.
                    |- "predictions": Reserved directory with the predictions as parquet files.
                    |- [modality_name]: One of the EvaluationModality.
                         |- "evaluation_xxx.json": Json file with evaluations.
        
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
        nargs="+",
        required=False,
        default=None,
        help=f"Evaluate: Space separated list of {[x.value for x in BenchMarkNames]}",
    )
    parser.add_argument(
        "-p",
        "--providers",
        nargs="+",
        required=False,
        default=None,
        help=f"Evaluate: Space separated list of {[x.value for x in PredictionProviderType]}",
    )
    parser.add_argument(
        "-m",
        "--modalities",
        nargs="+",
        required=False,
        default=None,
        help=f"Evaluate: Space separated list of {[x.value for x in EvaluationModality]}",
    )
    args = parser.parse_args()
    main(args)
