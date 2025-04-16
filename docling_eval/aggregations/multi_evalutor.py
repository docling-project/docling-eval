import json
import logging
from enum import Enum
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
    pred_path: Path,
) -> Optional[PredictionProviderType]:
    try:
        # Discover the split
        split = None
        for split_path in pred_path.iterdir():
            split = split_path.name
            break
        if not split:
            return None

        parquet_files = str(pred_path / split / "*.parquet")
        ds: IterableDataset = load_dataset(
            "parquet",
            data_files={split: parquet_files},
            split=split,
            streaming=True,
        )
        for data in ds:
            # TODO: Check why the predictor_info does not exist
            return PredictionProviderType.DOCLING

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
    GT_LEAF_DIR = "_GT_"
    PRED_LEAF_DIR = "predictions"

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
            dataset_sources,
            dataset_splits,
        )

        # Perform the evaluations
        multi_evaluation = self._run_evaluations(modalities, benchmark_preds)
        return multi_evaluation

    def _build_datasets(
        self,
        prediction_provider_types: List[PredictionProviderType],
        benchmarks: List[BenchMarkNames],
        dataset_sources: Optional[Dict[BenchMarkNames, Path]] = None,
        dataset_splits: Optional[Dict[BenchMarkNames, str]] = None,
    ) -> Dict[BenchMarkNames, Dict[PredictionProviderType, Path]]:
        r"""
        1. Get the predicted datasets
        2. If a predicted dataset is missing, check if the GT for this dataset exists.
        3. If both pred and GT datasets exist, build the GT dataset and the pred dataset.
        4. If GT is present, build the pred dataset.

        Return the paths of the prediction datasets
        """
        # Dict with benchmark predictions
        benchmark_preds: Dict[BenchMarkNames, Dict[PredictionProviderType, Path]] = {}

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

            if benchmark not in benchmark_preds:
                benchmark_preds[benchmark] = {}
            for provider_type in prediction_provider_types:
                benchmark_pred_dir = (
                    self._root_dir
                    / benchmark.value
                    / provider_type.value
                    / MultiEvaluator.PRED_LEAF_DIR
                )
                if dataset_exists(benchmark_pred_dir, split):
                    benchmark_preds[benchmark][provider_type] = benchmark_pred_dir
                    continue

                # Create the GT dataset if needed
                if not dataset_exists(benchmark_gt_dir, split):
                    dataset_source = (
                        dataset_sources.get(benchmark) if dataset_sources else None
                    )

                    _log.info("Creating GT for: %s", benchmark.value)
                    self._create_gt(benchmark, benchmark_gt_dir, split, dataset_source)

                # Create the pred dataset
                _log.info(
                    "Creating predictions for: %s / %s / %s",
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

                benchmark_preds[benchmark][provider_type] = benchmark_pred_dir

        return benchmark_preds

    def _run_evaluations(
        self,
        modalities: List[EvaluationModality],
        benchmark_preds: Dict[BenchMarkNames, Dict[PredictionProviderType, Path]],
        dataset_splits: Optional[Dict[BenchMarkNames, str]] = None,
    ) -> MultiEvaluation:
        evaluations: Dict[
            BenchMarkNames,
            Dict[str, Dict[EvaluationModality, SingleEvaluation]],
        ] = {}
        for benchmark, prov_mod_paths in benchmark_preds.items():
            split = (
                dataset_splits.get(benchmark, self._default_split)
                if dataset_splits
                else self._default_split
            )
            if benchmark not in evaluations:
                evaluations[benchmark] = {}
            for provider_type, pred_dir in prov_mod_paths.items():
                experiment = provider_type.value
                if experiment not in evaluations[benchmark]:
                    evaluations[benchmark][experiment] = {}

                for modality in modalities:
                    # Check if the provider supports the asked modality
                    if not validate_modality(provider_type, modality):
                        _log.error(
                            "Provider %s does not support modality: %s",
                            provider_type,
                            modality,
                        )
                        continue

                    eval_dir = (
                        self._root_dir / benchmark.value / experiment / modality.value
                    )
                    # Check if the evaluations are already present
                    evaluation = load_evaluation(benchmark, modality, eval_dir)
                    if not evaluation:
                        evaluation = evaluate(
                            modality, benchmark, pred_dir, eval_dir, split
                        )
                    if evaluation:
                        assert evaluation
                        evaluations[benchmark][experiment][modality] = SingleEvaluation(
                            evaluation=evaluation,
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
        prediction_provider: PredictionProviderType,
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

        for benchmark_path in multi_evaluation_path.iterdir():
            try:
                benchmark = BenchMarkNames(benchmark_path.name)
            except ValueError:
                continue
            for experiment_path in benchmark_path.iterdir():
                experiment = experiment_path.name
                if experiment == "_GT_":
                    continue

                # Get the provider_type from the prediction
                pred_provider_type = read_prediction_provider_type(
                    experiment_path / MultiEvaluator.PRED_LEAF_DIR
                )

                # Get the experiment
                for modality_path in experiment_path.iterdir():
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
                        prediction_provider_type=pred_provider_type,
                    )

        multi_evalution: MultiEvaluation = MultiEvaluation(evaluations=evaluations)
        return multi_evalution
