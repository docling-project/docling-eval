from __future__ import annotations

import argparse
import json
from multiprocessing import cpu_count
from pathlib import Path
import re
from typing import Dict, List, Tuple

import evaluate
from datasets import load_dataset

from nltk import download as nltk_download
from nltk import edit_distance, word_tokenize
from nltk.metrics import f_measure, precision, recall
from nltk.translate import meteor_score

nltk_download("punkt", quiet=True)
bleu_eval = evaluate.load("bleu")


class MarkdownTextEvaluator:
    """Compute a bundle of string‑similarity metrics for a pair of snippets."""

    def __init__(self):
        self._bleu_eval = bleu_eval

    def _compute_bleu_score(self, true_txt: str, pred_txt: str) -> float:
        try:
            result = self._bleu_eval.compute(
                predictions=[pred_txt], references=[[true_txt]]
            )
            bleu = result["bleu"]
        except Exception:
            bleu = 0.0
        return bleu

    def compute(self, true_txt: str, pred_txt: str) -> Dict[str, float]:
        """Return dict of metrics for *one* (reference, prediction) pair."""
        true_tok = word_tokenize(true_txt)
        pred_tok = word_tokenize(pred_txt)
        true_set, pred_set = set(true_tok), set(pred_tok)

        bleu = self._compute_bleu_score(true_txt, pred_txt)

        try:
            f1_ = f_measure(true_set, pred_set)
        except Exception:
            f1_ = 0.0
        try:
            prec_ = precision(true_set, pred_set)
        except Exception:
            prec_ = 0.0
        try:
            rec_ = recall(true_set, pred_set)
        except Exception:
            rec_ = 0.0
        try:
            edit_dist = edit_distance(pred_tok, true_tok) / max(
                len(pred_tok), len(true_tok)
            )
        except:
            edit_dist = 1.0
        try:
            meteor = meteor_score.meteor_score([true_tok], pred_tok)
        except Exception:
            meteor = 0.0

        return {
            "bleu_score": bleu if bleu else 0.0,
            "f1_score": f1_ if f1_ else 0.0,
            "precision": prec_ if prec_ else 0.0,
            "recall": rec_ if rec_ else 0.0,
            "edit_distance": edit_dist if edit_dist is not None else 1.0,
            "meteor": meteor if meteor else 0.0,
        }


def _eval_pair(args: Tuple[int, str, str]) -> Dict[str, float]:
    """Worker function executed in a child process.

    Each worker instantiates its own evaluator (lightweight) – avoids pickling
    issues with *evaluate* objects.
    """

    idx, true_txt, pred_txt = args
    evaluator = MarkdownTextEvaluator()
    metrics = evaluator.compute(true_txt, pred_txt)
    metrics["index"] = idx
    return metrics

def sanitize(text: str) -> str:
    to_remove = [
        " <code>",
        "</code>",
        " <formula>",
        "</formula>",
        "<doctag>",
        "</doctag>",
        "<loc_0><loc_0><loc_500><loc_500>",
    ]
    for s in to_remove:
        text = text.replace(s, "")

    # pattern = r"^<_([^_>]+)_>\s(.*)"
    # match = re.match(pattern, text, flags=re.DOTALL)
    # if match:
    #     text = str(match.group(2))  # everything after the <_language_>

    # text = text.strip()

    return text


def compute_scores(
    inferred_dir: Path,
    gt_dataset_path: Path,
    output_dir: Path,
    num_workers: int,
    split: str = "test",
) -> None:
    """Evaluate prediction files in *inferred_dir* against the dataset *split*.

    Work is farmed out to a `concurrent.futures.ProcessPoolExecutor` to utilise
    multiple CPU cores for tokenisation & metric computation.
    """

    inferred_dir = inferred_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort numerically by stem so that order matches dataset.
    pred_files: List[Path] = sorted(
        inferred_dir.iterdir(), key=lambda p: (p.stem.split(".")[0])
    )
    if not pred_files:
        raise FileNotFoundError(f"No prediction files found in {inferred_dir}")

    gt_dataset = load_dataset(
        str(gt_dataset_path), split=split, num_proc=num_workers or cpu_count()
    )
    id_to_gt = {
        str(rec["id"]): sanitize(rec["texts"][0]["assistant"]) for rec in gt_dataset
    }
    if len(pred_files) != len(gt_dataset):
        raise ValueError(
            "Length mismatch "
            f"{len(pred_files)} predictions vs {len(gt_dataset)} references"
        )

    jobs = []
    for pred_path in inferred_dir.iterdir():
        pred_id = pred_path.stem
        if pred_id not in id_to_gt:
            print(f"⚠️ Warning: {pred_id} not in ground truth dataset. Skipping.")
            continue

        pred_txt = sanitize(pred_path.read_text(encoding="utf8", errors="ignore"))
        true_txt = id_to_gt[pred_id]
        jobs.append(
            (
                pred_id,
                true_txt,
                pred_txt
            )
        )

    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm

    per_example: List[Dict[str, float]] = []

    if num_workers == 1:
        # Serial path – useful for debugging or when requested.
        for job in tqdm(jobs, desc="Evaluating", unit="sample"):
            per_example.append(_eval_pair(job))
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            for result in tqdm(
                pool.map(_eval_pair, jobs),
                total=len(jobs),
                desc="Evaluating",
                unit="sample",
            ):
                per_example.append(result)

    # ------------------------------------------------------------------
    # Persist results & aggregate means
    # ------------------------------------------------------------------
    per_example_path = output_dir / "per_example_scores.json"
    mean_scores_path = output_dir / "mean_scores.json"

    with per_example_path.open("w", encoding="utf8") as f:
        json.dump(per_example, f, indent=2)

    metric_keys = [
        "bleu_score",
        "f1_score",
        "precision",
        "recall",
        "edit_distance",
        "meteor",
    ]
    mean_scores = {
        k: sum(d[k] for d in per_example) / len(per_example) for k in metric_keys
    }

    with mean_scores_path.open("w", encoding="utf8") as f:
        json.dump(mean_scores, f, indent=2)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute string‑similarity metrics for markdown/code recognition tasks (multiprocess)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--file_source_path", type=Path, help="Directory with model‑generated snippet files."
    )
    parser.add_argument(
        "--gt_dir", type=Path, help="Path to the 🤗 dataset (local script or repo)."
    )
    parser.add_argument(
        "--output_dir", type=Path, help="Output directory for JSON results."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=cpu_count(),
        help="How many processes to spawn (1 ⇒ serial).",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Which dataset split to use."
    )

    args = parser.parse_args()

    compute_scores(
        inferred_dir=args.file_source_path,
        gt_dataset_path=args.gt_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        split=args.split,
    )


if __name__ == "__main__":
    main()
