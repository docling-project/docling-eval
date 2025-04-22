from pathlib import Path

import pytest

from docling_eval.aggregations.multi_evalutor import MultiEvaluator
from docling_eval.datamodels.types import (
    BenchMarkNames,
    EvaluationModality,
    PredictionProviderType,
)


def build_real_multi_evals():
    save_dir = Path("scratch/multi_test")

    benchmarks = [BenchMarkNames.DPBENCH, BenchMarkNames.DOCLAYNETV1]
    prediction_provider_types = [PredictionProviderType.DOCLING]
    modalities = [
        EvaluationModality.LAYOUT,
        EvaluationModality.MARKDOWN_TEXT,
        EvaluationModality.TABLE_STRUCTURE,
    ]

    me = MultiEvaluator(save_dir, begin_index=0)
    m_evals = me(prediction_provider_types, benchmarks, modalities)
    assert m_evals is not None


@pytest.mark.dependency()
def test_multi_evaluator():
    r""" """
    save_dir = Path("scratch/multi_test")

    benchmarks = [BenchMarkNames.DPBENCH]
    prediction_provider_types = [PredictionProviderType.DOCLING]
    modalities = [EvaluationModality.LAYOUT, EvaluationModality.MARKDOWN_TEXT]

    # Create multi evaluator for 2 samples of the dataset
    me = MultiEvaluator(save_dir, begin_index=0, end_index=2)

    # MultiEvaluator for 1 dataset, 1 provider, 1 modality
    m_evals = me(prediction_provider_types, benchmarks, modalities)

    assert m_evals is not None
    assert m_evals.evaluations is not None
    assert BenchMarkNames.DPBENCH in m_evals.evaluations
    assert PredictionProviderType.DOCLING in m_evals.evaluations[BenchMarkNames.DPBENCH]
    assert (
        EvaluationModality.LAYOUT
        in m_evals.evaluations[BenchMarkNames.DPBENCH][PredictionProviderType.DOCLING]
    )

    # MultiEvaluator for 1 dataset, 1 provider, 2 modalities
    modalities.append(EvaluationModality.MARKDOWN_TEXT)
    m_evals2 = me(prediction_provider_types, benchmarks, modalities)

    assert m_evals2 is not None
    assert m_evals2.evaluations is not None
    assert BenchMarkNames.DPBENCH in m_evals2.evaluations
    assert (
        PredictionProviderType.DOCLING in m_evals2.evaluations[BenchMarkNames.DPBENCH]
    )
    assert (
        EvaluationModality.MARKDOWN_TEXT
        in m_evals2.evaluations[BenchMarkNames.DPBENCH][PredictionProviderType.DOCLING]
    )

    # TODO: Test for datasets with multiple providers
    # TODO: Test for datasets with external data sources


def test_loading_from_disk():
    save_dir = Path("scratch/multi_test")
    loaded_m_evals = MultiEvaluator.load_multi_evaluation(save_dir)
    assert loaded_m_evals is not None


# if __name__ == "__main__":
#     import logging

#     logging.getLogger("docling").setLevel(logging.WARNING)
#     logging.getLogger(__name__).setLevel(logging.INFO)

#     # test_multi_evaluator()
#     test_loading_from_disk()
#     # build_real_multi_evals()
