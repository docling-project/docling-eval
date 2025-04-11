import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional

from pydantic import BaseModel

from docling_eval.cli.main import (
    PredictionProviderType,
    evaluate,
    get_dataset_builder,
    get_prediction_provider,
)
from docling_eval.datamodels.types import BenchMarkNames, EvaluationModality
from docling_eval.evaluators.base_evaluator import DatasetEvaluationType
from docling_eval.evaluators.bbox_text_evaluator import DatasetBoxesTextEvaluation
from docling_eval.evaluators.layout_evaluator import DatasetLayoutEvaluation
from docling_eval.evaluators.markdown_text_evaluator import DatasetMarkdownEvaluation
from docling_eval.evaluators.readingorder_evaluator import DatasetReadingOrderEvaluation
from docling_eval.evaluators.table_evaluator import DatasetTableEvaluation
from docling_eval.utils.utils import dataset_exists

_log = logging.getLogger(__name__)


# TODO:
# class DatasetBuildResult(str, Enum):
#     PRE_EXISTING = "pre_existing"
#     CREATED = "created"
#     FAILED_TO_LOAD = "failed_to_load"
#     FAILED_TO_CREATE = "failed_to_create"


class MultiEvaluation(BaseModel, Generic[DatasetEvaluationType]):
    evaluations: Dict[
        BenchMarkNames,
        Dict[PredictionProviderType, Dict[EvaluationModality, DatasetEvaluationType]],
    ] = {}


def load_evaluation(
    benchmark: BenchMarkNames,
    modality: EvaluationModality,
    eval_dir: Path,
) -> Optional[DatasetEvaluationType]:
    r"""Load evaluation from file"""

    # TODO: Fix the type of values
    modality_eval_classes: Dict[EvaluationModality, Any] = {
        EvaluationModality.BBOXES_TEXT: DatasetBoxesTextEvaluation,
        EvaluationModality.LAYOUT: DatasetLayoutEvaluation,
        EvaluationModality.TABLE_STRUCTURE: DatasetTableEvaluation,
        EvaluationModality.READING_ORDER: DatasetReadingOrderEvaluation,
        EvaluationModality.MARKDOWN_TEXT: DatasetMarkdownEvaluation,
    }

    eval_fn = eval_dir / f"evaluation_{benchmark.value}_{modality.value}.json"
    if not eval_fn.exists():
        return None

    with open(eval_fn, "r") as fd:
        eval_json = json.load(fd)
        eval_class = modality_eval_classes[modality]
        evaluation = eval_class.model_validate(eval_json)
    return evaluation


class MultiEvaluator(Generic[DatasetEvaluationType]):
    r"""
    Evaluate combinations of multiple Providers, Benchmark, EvaluationModality

    GT dir structure: gt_dir / benchmark_name / parquet files
    Prediction dir structure: output_dir / benchmark / provider / modality / parquet files
    """

    # Leaf dirs for GT, predictions, evaluations
    GT_LEAF_DIR = "_GT_"
    PRED_LEAF_DIR = "predictions"
    EVAL_LEAF_DIR = "evaluations"

    def __init__(
        self,
        gt_dir: Path,
        output_dir: Path,
        default_split: str = "test",
        begin_index: int = 0,
        end_index: int = -1,
    ):
        r""" """
        self._output_dir = output_dir
        self._gt_dir = gt_dir
        self._default_split = default_split
        self._begin_index = begin_index
        self._end_index = end_index

        self._output_dir.mkdir(parents=True, exist_ok=True)

    def __call__(
        self,
        prediction_provider_types: List[PredictionProviderType],
        benchmarks: List[BenchMarkNames],
        modalities: List[EvaluationModality],
        dataset_sources: Optional[Dict[BenchMarkNames, Path]] = None,
        dataset_splits: Optional[Dict[BenchMarkNames, str]] = None,
    ) -> MultiEvaluation:
        r""" """
        # Build any missing dataset
        benchmark_preds = self._build_datasets(
            prediction_provider_types,
            benchmarks,
            modalities,
            dataset_sources,
            dataset_splits,
        )

        # Perform the evaluations
        multi_evaluation = self._run_evaluations(benchmark_preds)
        return multi_evaluation

    def _build_datasets(
        self,
        prediction_provider_types: List[PredictionProviderType],
        benchmarks: List[BenchMarkNames],
        modalities: List[EvaluationModality],
        dataset_sources: Optional[Dict[BenchMarkNames, Path]] = None,
        dataset_splits: Optional[Dict[BenchMarkNames, str]] = None,
    ) -> Dict[
        BenchMarkNames, Dict[PredictionProviderType, Dict[EvaluationModality, Path]]
    ]:
        r"""
        1. Get the predicted datasets
        2. If a predicted dataset is missing, check if the GT for this dataset exists.
        3. If both pred and GT datasets exist, build the GT dataset and the pred dataset.
        4. If GT is present, build the pred dataset.

        Return the paths of the prediction datasets
        """
        # TODO: Report of the GT/pred datasets that have been created or loaded
        # gt_build_report: Dict[BenchMarkNames, DatasetBuildResult] = {}
        # pred_build_report: Dict[BenchMarkNames, Dict[
        #     BenchMarkNames, Dict[PredictionProviderType, Dict[EvaluationModality, DatasetBuildResult]]
        # ]] = {}

        # Dict with benchmark predictions
        benchmark_preds: Dict[
            BenchMarkNames, Dict[PredictionProviderType, Dict[EvaluationModality, Path]]
        ] = {}
        gt_dir = self._gt_dir or self._output_dir

        # Set the benchmark_preds
        for benchmark in benchmarks:
            # Decide how to name the dir for GT dataset
            benchmark_gt_dir = gt_dir / benchmark.value / MultiEvaluator.GT_LEAF_DIR
            split = (
                dataset_splits.get(benchmark, self._default_split)
                if dataset_splits
                else self._default_split
            )

            if benchmark not in benchmark_preds:
                benchmark_preds[benchmark] = {}
            for provider_type in prediction_provider_types:
                if provider_type not in benchmark_preds[benchmark]:
                    benchmark_preds[benchmark][provider_type] = {}
                for modality in modalities:
                    # Decide how to name the dir for pred dataset
                    benchmark_pred_dir = (
                        self._output_dir
                        / benchmark.value
                        / provider_type.value
                        / modality.value
                        / MultiEvaluator.PRED_LEAF_DIR
                    )
                    if dataset_exists(benchmark_pred_dir, split):
                        benchmark_preds[benchmark][provider_type][
                            modality
                        ] = benchmark_pred_dir
                        continue

                    # Create the GT dataset if needed
                    if not dataset_exists(benchmark_gt_dir, split):
                        dataset_source = (
                            dataset_sources.get(benchmark) if dataset_sources else None
                        )

                        _log.info("Creating GT for: %s", benchmark.value)
                        self._create_gt(
                            benchmark, benchmark_gt_dir, split, dataset_source
                        )

                    # Create the pred dataset
                    _log.info(
                        "Creating predictions for: %s / %s / %s",
                        benchmark.value,
                        provider_type.value,
                        modality.value,
                    )
                    self._create_eval(
                        benchmark,
                        provider_type,
                        modality,
                        benchmark_gt_dir,
                        split,
                        benchmark_pred_dir,
                    )

                    benchmark_preds[benchmark][provider_type][
                        modality
                    ] = benchmark_pred_dir

        return benchmark_preds

    def _run_evaluations(
        self,
        dataset_preds: Dict[
            BenchMarkNames, Dict[PredictionProviderType, Dict[EvaluationModality, Path]]
        ],
        dataset_splits: Optional[Dict[BenchMarkNames, str]] = None,
    ) -> MultiEvaluation:
        evaluations: Dict[
            BenchMarkNames,
            Dict[
                PredictionProviderType, Dict[EvaluationModality, DatasetEvaluationType]
            ],
        ] = {}
        for benchmark, prov_mod_paths in dataset_preds.items():
            split = (
                dataset_splits.get(benchmark, self._default_split)
                if dataset_splits
                else self._default_split
            )
            if benchmark not in evaluations:
                evaluations[benchmark] = {}
            for provider_type, mod_paths in prov_mod_paths.items():
                if provider_type not in evaluations[benchmark]:
                    evaluations[benchmark][provider_type] = {}
                for modality, pred_dir in mod_paths.items():
                    eval_dir = (
                        self._output_dir
                        / benchmark.value
                        / provider_type.value
                        / modality.value
                        / MultiEvaluator.EVAL_LEAF_DIR
                    )
                    # Check if the evaluations are already present
                    evaluation = load_evaluation(benchmark, modality, eval_dir)
                    if not evaluation:
                        evaluation = evaluate(
                            modality, benchmark, pred_dir, eval_dir, split
                        )
                    if evaluation:
                        assert evaluation
                        evaluations[benchmark][provider_type][modality] = evaluation

        multi_evaluation: MultiEvaluation = MultiEvaluation(evaluations=evaluations)
        return multi_evaluation

    def _create_gt(
        self,
        benchmark: BenchMarkNames,
        gt_dir: Path,
        split: str,
        dataset_source: Optional[Path],
    ) -> bool:
        r"""
        Create GT dataset at the gt_dir
        """
        try:
            dataset_builder = get_dataset_builder(
                benchmark=benchmark,
                target=gt_dir,
                split=split,
                dataset_source=dataset_source,
                begin_index=self._begin_index,
                end_index=self._end_index,
            )

            # Retrieve and save the dataset
            if dataset_builder.must_retrieve:
                dataset_builder.retrieve_input_dataset()
            dataset_builder.save_to_disk(chunk_size=80)

            _log.info(f"Ground truth dataset created at {gt_dir}")
            return True
        except ValueError as e:
            _log.error(f"Error creating dataset builder: {str(e)}")
            return False

    def _create_eval(
        self,
        benchmark: BenchMarkNames,
        prediction_provider: PredictionProviderType,
        modality: EvaluationModality,
        gt_dir: Path,
        split: str,
        pred_dir: Path,
    ) -> bool:
        r"""
        Create eval dataset at the pred_dir
        """
        # Check if ground truth exists
        if not gt_dir.exists():
            _log.error(f"Ground truth directory not found: {gt_dir}")
            return False
        try:
            # Create the appropriate prediction provider
            provider = get_prediction_provider(
                provider_type=prediction_provider,
                do_visualization=False,
            )

            # Get the dataset name from the benchmark
            dataset_name = f"{benchmark.value}: {modality.value}"

            # Create predictions
            provider.create_prediction_dataset(
                name=dataset_name,
                gt_dataset_dir=gt_dir,
                target_dataset_dir=pred_dir,
                split=split,
                begin_index=self._begin_index,
                end_index=self._end_index,
            )

            _log.info(f"Evaluation dataset created at {pred_dir}")
            return True
        except ValueError as e:
            _log.error(f"Error creating prediction provider: {str(e)}")
            return False

    @staticmethod
    def load_multi_evaluation(multi_evaluation_path: Path) -> MultiEvaluation:
        r"""Load MultiEvaluation from disk files"""
        # benchmark -> provider -> modality -> DatasetEvaluation
        evaluations: Dict[
            BenchMarkNames,
            Dict[
                PredictionProviderType, Dict[EvaluationModality, DatasetEvaluationType]
            ],
        ] = {}

        for benchmark_path in multi_evaluation_path.iterdir():
            try:
                benchmark = BenchMarkNames(benchmark_path.name)
            except ValueError:
                continue
            for provider_path in benchmark_path.iterdir():
                if provider_path.name == "_GT_":
                    continue
                try:
                    provider = PredictionProviderType(provider_path.name)
                except ValueError:
                    continue

                for modality_path in provider_path.iterdir():
                    try:
                        modality = EvaluationModality(modality_path.name)
                    except ValueError:
                        continue
                    evaluations_path = modality_path / MultiEvaluator.EVAL_LEAF_DIR

                    # Load the evaluation
                    evaluation = load_evaluation(benchmark, modality, evaluations_path)
                    if not evaluation:
                        continue

                    if benchmark not in evaluations:
                        evaluations[benchmark] = {}
                    if provider not in evaluations[benchmark]:
                        evaluations[benchmark][provider] = {}
                    evaluations[benchmark][provider][modality] = evaluation

        multi_evalution: MultiEvaluation = MultiEvaluation(evaluations=evaluations)
        return multi_evalution
