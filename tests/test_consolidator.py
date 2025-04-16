from pathlib import Path

from docling_eval.aggregations.consolidator import Consolidator
from docling_eval.aggregations.multi_evalutor import MultiEvaluator


def test_consolidator():
    r""" """
    save_dir = Path("scratch/multi_test")
    multi_evaluation = MultiEvaluator.load_multi_evaluation(save_dir)

    output_path = Path("scratch/consolidator")
    consolidator = Consolidator(output_path)
    dfs, produced_file = consolidator(multi_evaluation)
    assert dfs is not None
    assert produced_file is not None
    assert produced_file.exists()


# if __name__ == "__main__":
#     test_consolidator()
