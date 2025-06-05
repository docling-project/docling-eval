import json
import logging
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional

from datasets import load_dataset
from datasets.iterable_dataset import IterableDataset
from pydantic import BaseModel

from docling_eval.cli.main import evaluate, get_dataset_builder, get_prediction_provider
from docling_eval.datamodels.types import (
    BenchMarkNames,
    EvaluationModality,
    PredictionProviderType,
)
from docling_eval.evaluators.base_evaluator import DatasetEvaluationType
from docling_eval.evaluators.bbox_text_evaluator import DatasetBoxesTextEvaluation
from docling_eval.evaluators.layout_evaluator import DatasetLayoutEvaluation
from docling_eval.evaluators.markdown_text_evaluator import DatasetMarkdownEvaluation
from docling_eval.evaluators.readingorder_evaluator import DatasetReadingOrderEvaluation
from docling_eval.evaluators.table_evaluator import DatasetTableEvaluation
from docling_eval.utils.utils import dataset_exists, modalities_of_prediction_type

_log = logging.getLogger(__name__)


class SingleEvaluation(BaseModel, Generic[DatasetEvaluationType]):
    evaluation: DatasetEvaluationType
    experiment: str

    # The prediction provider cannot be deteremined if it comes outside of docling-eval.
    prediction_provider_type: Optional[PredictionProviderType] = None


class MultiEvaluation(BaseModel):
    # Benchmark -> experiment -> modality -> SingleEvaluation
    evaluations: Dict[
        BenchMarkNames,
        Dict[str, Dict[EvaluationModality, SingleEvaluation]],
    ] = {}


def load_evaluation(
    benchmark: BenchMarkNames,
    modality: EvaluationModality,
    eval_dir: Path,
) -> Optional[DatasetEvaluationType]:
    r"""Load evaluation from file"""

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


def validate_modality(
    prediction_provider_type: PredictionProviderType,
    modality: EvaluationModality,
) -> bool:
    supported_modalities = modalities_of_prediction_type(prediction_provider_type)
    if not supported_modalities:
        return False
    if modality in supported_modalities:
        return True
    return False


def read_prediction_provider_type(
    pred_dir: Path,
    split: str,
) -> Optional[PredictionProviderType]:
    r"""
    Open the evaluation dataset and read the prediction provider column
    """
    try:
        parquet_files = str(pred_dir / split / "*.parquet")
        ds: IterableDataset = load_dataset(
            "parquet",
            data_files={split: parquet_files},
            split=split,
            streaming=True,
        )
        for data in ds:
            info = data.get("predictor_info")
            if not info:
                return None
            asset = info.get("asset")
            if not asset:
                return None
            try:
                pred_provider_type = PredictionProviderType(asset)
                return pred_provider_type
            except Exception as ex:
                return None
    except Exception as ex:
        pass
    return None


class MultiEvaluator(Generic[DatasetEvaluationType]):
    r"""
    Evaluate combinations of multiple Providers, Benchmark, EvaluationModality

    GT dir structure: gt_dir / benchmark_name / parquet files
    Prediction dir structure: output_dir / benchmark / provider / modality / parquet files
    """

    # Leaf dirs for GT, predictions, evaluations
    # GT_LEAF_DIR = "_GT_"
    # PRED_LEAF_DIR = "predictions"

    GT_LEAF_DIR = "gt_dataset"
    PRED_LEAF_DIR = "eval_dataset"
    EVALUATIONS_DIR = "evaluations"

    def __init__(
        self,
        root_dir: Path,
        default_split: str = "test",
        begin_index: int = 0,
        end_index: int = -1,
    ):
        r""" """
        self._root_dir = root_dir
        self._default_split = default_split
        self._begin_index = begin_index
        self._end_index = end_index

        self._root_dir.mkdir(parents=True, exist_ok=True)

    def __call__(
        self,
        experiment_names: List[str],
        benchmarks: List[BenchMarkNames],
        modalities: List[EvaluationModality],
        dataset_sources: Optional[Dict[BenchMarkNames, Path]] = None,
        dataset_splits: Optional[Dict[BenchMarkNames, str]] = None,
    ) -> MultiEvaluation:
        r""" """
        # Build any missing dataset
        benchmark_experiments = self._build_datasets(
            experiment_names,
            benchmarks,
            dataset_sources,
            dataset_splits,
        )

        # Perform the evaluations
        multi_evaluation = self._run_evaluations(
            modalities, benchmark_experiments, dataset_splits
        )
        return multi_evaluation

    def _build_datasets(
        self,
        experiment_names: List[str],
        benchmarks: List[BenchMarkNames],
        dataset_sources: Optional[Dict[BenchMarkNames, Path]] = None,
        dataset_splits: Optional[Dict[BenchMarkNames, str]] = None,
    ) -> Dict[BenchMarkNames, Dict[str, Path]]:
        r"""
        1. Get the predicted datasets
        2. If a predicted dataset is missing, check if the GT for this dataset exists.
        3. If both pred and GT datasets exist, build the GT dataset and the pred dataset.
        4. If GT is present, build the pred dataset.

        Notice: In case the prediction dataset does not exist, the experiment name MUST match a
        provider name.

        Return the paths of the prediction datasets:
        benchmark_name -> experiment_name -> Path
        """
        # Dict with benchmark predictions
        benchmark_experiments: Dict[BenchMarkNames, Dict[str, Path]] = {}

        # Set the benchmark_preds
        for benchmark in benchmarks:
            benchmark_gt_dir = (
                self._root_dir / benchmark.value / MultiEvaluator.GT_LEAF_DIR
            )
            split = (
                dataset_splits.get(benchmark, self._default_split)
                if dataset_splits
                else self._default_split
            )

            if benchmark not in benchmark_experiments:
                benchmark_experiments[benchmark] = {}
            for experiment_name in experiment_names:
                benchmark_pred_dir = (
                    self._root_dir
                    / benchmark.value
                    / experiment_name
                    / MultiEvaluator.PRED_LEAF_DIR
                )
                if dataset_exists(benchmark_pred_dir, split):
                    benchmark_experiments[benchmark][
                        experiment_name
                    ] = benchmark_pred_dir
                    continue

                # Create the GT dataset if needed
                if not dataset_exists(benchmark_gt_dir, split):
                    dataset_source = (
                        dataset_sources.get(benchmark) if dataset_sources else None
                    )

                    _log.info("Creating GT for: %s", benchmark.value)
                    self._create_gt(benchmark, benchmark_gt_dir, split, dataset_source)

                # Map the experiment_name to a PredictionProviderType
                try:
                    provider_type = PredictionProviderType(experiment_name)
                except ValueError as ex:
                    _log.error(
                        "Prediction dataset is missing and experiment %s does NOT match any provider name",
                        experiment_name,
                    )
                    raise ex

                # Create the prediction dataset
                _log.info(
                    "Creating predictions for: %s / %s",
                    benchmark.value,
                    provider_type.value,
                )
                self._create_eval(
                    benchmark,
                    provider_type,
                    benchmark_gt_dir,
                    split,
                    benchmark_pred_dir,
                )

                benchmark_experiments[benchmark][experiment_name] = benchmark_pred_dir

        return benchmark_experiments

    def _run_evaluations(
        self,
        modalities: List[EvaluationModality],
        benchmark_experiments: Dict[BenchMarkNames, Dict[str, Path]],
        dataset_splits: Optional[Dict[BenchMarkNames, str]] = None,
    ) -> MultiEvaluation:
        evaluations: Dict[
            BenchMarkNames,
            Dict[str, Dict[EvaluationModality, SingleEvaluation]],
        ] = {}
        for benchmark, exp_mod_paths in benchmark_experiments.items():
            split = (
                dataset_splits.get(benchmark, self._default_split)
                if dataset_splits
                else self._default_split
            )
            if benchmark not in evaluations:
                evaluations[benchmark] = {}
            for experiment, pred_dir in exp_mod_paths.items():
                if experiment not in evaluations[benchmark]:
                    evaluations[benchmark][experiment] = {}

                # Try to get the prediction provider
                provider_type = read_prediction_provider_type(pred_dir, split)

                for modality in modalities:
                    # If the provider is available, check if it supports the asked modality
                    if provider_type and not validate_modality(provider_type, modality):
                        _log.error(
                            "Prediction dataset from provider '%s', which does not support modality '%s'",
                            provider_type.value,
                            modality,
                        )
                        continue

                    eval_dir = (
                        self._root_dir
                        / benchmark.value
                        / experiment
                        / MultiEvaluator.EVALUATIONS_DIR
                        / modality.value
                    )
                    # Check if the evaluations are already present
                    evaluation = load_evaluation(benchmark, modality, eval_dir)
                    if not evaluation:
                        _log.info(
                            "Evaluate benchmark: %s, for experiment: %s, for modality: %s",
                            benchmark.value,
                            experiment,
                            modality,
                        )
                        evaluation = evaluate(
                            modality, benchmark, pred_dir, eval_dir, split
                        )
                    if evaluation:
                        assert evaluation
                        evaluations[benchmark][experiment][modality] = SingleEvaluation(
                            evaluation=evaluation,
                            experiment=experiment,
                            prediction_provider_type=provider_type,
                        )

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
        provider_type: PredictionProviderType,
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
                provider_type=provider_type,
                do_visualization=False,
            )

            # Create predictions
            provider.create_prediction_dataset(
                name=benchmark.value,
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
            Dict[Path, Dict[EvaluationModality, DatasetEvaluationType]],
        ] = {}

        # Get the benchmark
        for benchmark_path in multi_evaluation_path.iterdir():
            try:
                benchmark = BenchMarkNames(benchmark_path.name)
            except ValueError:
                continue
            # Get the experiment
            for experiment_path in benchmark_path.iterdir():
                if not experiment_path.is_dir():
                    continue

                experiment = experiment_path.name
                if experiment == MultiEvaluator.GT_LEAF_DIR:
                    continue

                # Load the evaluations for each modality
                evaluations_path = experiment_path / MultiEvaluator.EVALUATIONS_DIR
                if not evaluations_path.is_dir():
                    continue
                for modality_path in evaluations_path.iterdir():
                    try:
                        modality = EvaluationModality(modality_path.name)
                    except ValueError:
                        continue

                    # Load the evaluation
                    evaluation = load_evaluation(benchmark, modality, modality_path)
                    if not evaluation:
                        continue

                    if benchmark not in evaluations:
                        evaluations[benchmark] = {}
                    if experiment not in evaluations[benchmark]:
                        evaluations[benchmark][experiment] = {}

                    evaluations[benchmark][experiment][modality] = SingleEvaluation(
                        evaluation=evaluation,
                        experiment=experiment,
                    )

        multi_evaluation: MultiEvaluation = MultiEvaluation(evaluations=evaluations)
        return multi_evaluation
