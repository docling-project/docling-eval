import json
import logging
from enum import Enum, auto
from pathlib import Path
from typing import Annotated, Optional

import typer

from docling_eval.evaluators.table_evaluator import TableEvaluator

_log = logging.getLogger(__name__)

app = typer.Typer(
    name="docling-eval",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


class Command(str, Enum):
    CONVERT_GT_DATASET = "convert_gt_dataset"
    PREDICT = "predict"
    EVALUATE = "evaluate"


class EvaluationTask(str, Enum):
    TABLES = "TABLES"
    CHEMISTRY = "CHEMISTRY"
    CODE = "CODE"
    FULLPAGE = "FULLPAGE"


def convert_ds():
    r""""""
    pass


def predict_ds():
    r""""""
    pass


def evaluate_ds(
    ds_path: Path, evaluation_task: EvaluationTask, split: str, save_path: Path
):
    r""""""
    ds_evaluation = None
    if evaluation_task == EvaluationTask.TABLES:
        table_evaluator = TableEvaluator()
        ds_evaluation = table_evaluator(ds_path, split)
    else:
        _log.info("Unsupported evaluation task")

    if ds_evaluation is None:
        _log.error("No evaluation has been produced")
        return

    # Save the evaluation
    save_fn = "evaluation.json"
    with open(save_fn, "w") as fd:
        json.dump(ds_evaluation.model_dump(), fd, indent=2, sort_keys=True)
        _log.info("The evaluation has been saved in '%s'", save_fn)


@app.command(no_args_is_help=True)
def main(
    command: Annotated[
        Command,
        typer.Argument(
            ...,
            help="The command to execute",
        ),
    ],
    ds_path: Annotated[
        Path,
        typer.Argument(
            ...,
            help="The path for the ground truth dataset",
        ),
    ],
    evaluation_task: Annotated[
        EvaluationTask,
        typer.Argument(
            ...,
            help="The evaluation task to perform",
        ),
    ],
    split: Annotated[
        str,
        typer.Option(
            ...,
            help="The ground truth split",
        ),
    ] = "val",
    save_path: Annotated[
        Path,
        typer.Option(
            ...,
            help="The path for the ground truth dataset",
        ),
    ] = Path("viz/"),
):
    logging.basicConfig(level=logging.INFO)

    # Dispatch the command
    if command == Command.CONVERT_GT_DATASET:
        convert_ds()
    elif command == Command.PREDICT:
        predict_ds()
    elif command == Command.EVALUATE:
        evaluate_ds(ds_path, evaluation_task, split, save_path)
    else:
        _log.error("Unsupported command: '%s'", command)


if __name__ == "__main__":
    app()
