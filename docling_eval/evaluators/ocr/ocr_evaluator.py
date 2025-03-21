import glob
import json
import logging
import os
import statistics
from pathlib import Path
from typing import Any, Dict, List

import evaluate
import pandas as pd
from docling_core.types.doc.document import DoclingDocument
from pydantic import BaseModel
from tqdm import tqdm  # type: ignore

from docling_eval.benchmarks.constants import BenchMarkColumns  # type: ignore

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
)

log = logging.getLogger(__name__)


class DatasetStatistics(BaseModel):
    mean: float
    median: float
    min: float
    max: float
    std: float
    count: int


class PageEvaluation(BaseModel):
    doc_id: str
    gd_text: str
    extracted_text: str
    cer: float
    source_file: str
    engine: str


class FileEvaluation(BaseModel):
    file_name: str
    engine_name: str
    evaluations: List[PageEvaluation]
    cer_stats: DatasetStatistics


class DatasetTextractEvaluation(BaseModel):
    file_evaluations: List[FileEvaluation]


def compute_stats(cers_list: List[float]) -> DatasetStatistics:
    """Compute statistics for CER scores"""
    result = {
        "mean": statistics.mean(cers_list) if cers_list else 0.0,
        "median": statistics.median(cers_list) if cers_list else 0.0,
        "min": min(cers_list) if cers_list else 0.0,
        "max": max(cers_list) if cers_list else 0.0,
        "std": statistics.stdev(cers_list) if len(cers_list) > 1 else 0.0,
        "count": len(cers_list),
    }

    return DatasetStatistics(**result)


def load_jsonl_dataset(jsonl_path: str) -> Dict[str, List[Dict]]:
    """Load dataset from JSONL file(s) and organize by source file"""
    data_by_file: Dict[str, List[Dict[str, Any]]] = {}
    for file_path in glob.glob(jsonl_path):
        file_name = os.path.basename(file_path)
        data_by_file[file_name] = []

        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    item["_source_file"] = file_path
                    data_by_file[file_name].append(item)

    return data_by_file


class OCREvaluator:
    def __init__(self):
        # https://huggingface.co/spaces/evaluate-metric/cer
        self._cer_eval = evaluate.load("cer")

    def __call__(
        self, input_ds_path: Path, output_path: Path, split: str = "test"
    ) -> DatasetTextractEvaluation:
        search_pattern = os.path.join(input_ds_path, "*", split, "shard_*.parquet")
        parquet_files = glob.glob(search_pattern)
        logging.info(f"Loading Parquet files from pattern: {search_pattern}")
        logging.info(f"Found {len(parquet_files)} matching files")

        dataset_by_file = {}
        engine_mapping = {}

        for file_path in parquet_files:
            file_name = Path(file_path).name
            path_parts = Path(file_path).parts
            engine_name = path_parts[-3]
            composite_key = f"{engine_name}/{file_name}"
            engine_mapping[composite_key] = engine_name

            try:
                df = pd.read_parquet(file_path)
                dataset_by_file[composite_key] = df.to_dict("records")
                logging.info(f"Successfully loaded {file_path} with {len(df)} records")
            except Exception as e:
                logging.error(f"Error loading {file_path}: {str(e)}")

        if not dataset_by_file:
            raise ValueError(f"Failed to load dataset from {search_pattern}")

        file_count = len(dataset_by_file)
        total_records = sum(len(records) for records in dataset_by_file.values())
        logging.info(f"Loaded {total_records} records from {file_count} Parquet files")

        file_evaluations = []

        for composite_key, dataset in dataset_by_file.items():
            engine_name = engine_mapping[composite_key]
            file_name = composite_key.split("/", 1)[1]  # Remove engine prefix

            logging.info(
                f"Processing file: {file_name} (Engine: {engine_name}) with {len(dataset)} records"
            )

            text_evaluations_list = []
            file_cers_list = []

            for i, data in tqdm(
                enumerate(dataset),
                desc=f"Processing {engine_name}/{file_name}",
                ncols=120,
                total=len(dataset),
            ):
                doc_id = data[BenchMarkColumns.DOC_ID]
                true_doc_dict = data[BenchMarkColumns.GROUNDTRUTH]
                pred_doc_dict = data[BenchMarkColumns.PREDICTION]

                true_doc = DoclingDocument.model_validate_json(true_doc_dict)
                pred_doc = DoclingDocument.model_validate_json(pred_doc_dict)

                # Extract text from documents
                gd_text = self._extract_text(true_doc.export_to_dict())
                extracted_text = self._extract_text(pred_doc.export_to_dict())

                if gd_text and extracted_text:
                    cer = self._compute_cer_score(gd_text, extracted_text)
                else:
                    cer = 1.0  # max error when text is missing

                file_cers_list.append(cer)

                text_evaluation = PageEvaluation(
                    doc_id=doc_id,
                    gd_text=gd_text,
                    extracted_text=extracted_text,
                    cer=cer,
                    source_file=file_name,
                    engine=engine_name,
                )

                text_evaluations_list.append(text_evaluation)
                logging.debug(
                    f"Engine: {engine_name}, File: {file_name}, Document {doc_id} CER: {cer:.4f}"
                )

            file_cer_stats = compute_stats(file_cers_list)

            file_evaluation = FileEvaluation(
                file_name=file_name,
                engine_name=engine_name,
                evaluations=text_evaluations_list,
                cer_stats=file_cer_stats,
            )

            file_evaluations.append(file_evaluation)

            logging.info(f"\n{engine_name}/{file_name} CER Statistics:")
            logging.debug(f"  - Mean CER: {file_cer_stats.mean:.4f}")
            logging.debug(f"  - Median CER: {file_cer_stats.median:.4f}")
            logging.debug(f"  - Min CER: {file_cer_stats.min:.4f}")
            logging.debug(f"  - Max CER: {file_cer_stats.max:.4f}")
            logging.debug(f"  - Std Dev: {file_cer_stats.std:.4f}")
            logging.debug(f"  - Count: {file_cer_stats.count}")

        # Only create output after processing all files
        output_path.mkdir(parents=True, exist_ok=True)

        results_path = output_path / "evaluation_results.json"
        with open(results_path, "w") as f:
            results_data = {
                "file_evaluations": [
                    {
                        "file_name": eval_item.file_name,
                        "engine_name": eval_item.engine_name,  # Include engine name in output
                        "statistics": eval_item.cer_stats.model_dump(),
                        "evaluations": [e.model_dump() for e in eval_item.evaluations],
                    }
                    for eval_item in file_evaluations
                ]
            }
            json.dump(results_data, f, indent=2)

        logging.info(f"Saved evaluation results to {results_path}")

        chart_path = output_path / "ocr_comparison_chart.png"
        self._create_comparison_chart(file_evaluations, chart_path)

        return DatasetTextractEvaluation(file_evaluations=file_evaluations)

    def _compute_cer_score(self, true_txt: str, pred_txt: str):
        """Compute CER score with the HF evaluate and the default Tokenizer"""
        result = self._cer_eval.compute(predictions=[pred_txt], references=[true_txt])
        return result

    def _extract_text(self, json_data: Dict[str, Any]) -> str:
        """Extract text from document JSON structure"""
        extracted_text = ""
        if "texts" in json_data:
            for text_item in json_data["texts"]:
                if "text" in text_item:
                    extracted_text += text_item["text"] + " "
        return extracted_text.strip()

    def _create_comparison_chart(
        self, file_evaluations: List[FileEvaluation], output_path: Path
    ):
        """Create a bar chart comparing CER across different OCR engines"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            engine_data = {}  # type: ignore
            for eval_item in file_evaluations:
                engine_name = eval_item.engine_name

                if engine_name not in engine_data:
                    engine_data[engine_name] = {
                        "mean_cers": [],
                        "median_cers": [],
                        "counts": [],
                    }

                engine_data[engine_name]["mean_cers"].append(eval_item.cer_stats.mean)
                engine_data[engine_name]["median_cers"].append(
                    eval_item.cer_stats.median
                )
                engine_data[engine_name]["counts"].append(eval_item.cer_stats.count)

            engines = []
            mean_cers = []
            median_cers = []

            for engine, data in engine_data.items():
                engines.append(engine)

                if len(data["mean_cers"]) > 1:
                    total_count = sum(data["counts"])
                    weighted_mean = (
                        sum(m * c for m, c in zip(data["mean_cers"], data["counts"]))
                        / total_count
                    )
                    weighted_median = (
                        sum(m * c for m, c in zip(data["median_cers"], data["counts"]))
                        / total_count
                    )
                    mean_cers.append(weighted_mean)
                    median_cers.append(weighted_median)
                else:
                    mean_cers.append(data["mean_cers"][0])
                    median_cers.append(data["median_cers"][0])

            x = np.arange(len(engines))
            width = 0.35

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(x - width / 2, mean_cers, width, label="Mean CER")
            ax.bar(x + width / 2, median_cers, width, label="Median CER")

            ax.set_xlabel("OCR Engine")
            ax.set_ylabel("Character Error Rate (CER)")
            ax.set_title("OCR Engine Comparison - Lower is Better")
            ax.set_xticks(x)
            ax.set_xticklabels(engines)
            ax.legend()

            for i, v in enumerate(mean_cers):
                ax.text(i - width / 2, v + 0.01, f"{v:.3f}", ha="center")

            for i, v in enumerate(median_cers):
                ax.text(i + width / 2, v + 0.01, f"{v:.3f}", ha="center")

            plt.tight_layout()
            plt.savefig(output_path)
            logging.info(f"Saved comparison chart to {output_path}")

        except ImportError:
            logging.warning(
                "Could not create comparison chart: matplotlib is not installed"
            )


if __name__ == "__main__":
    evaluator = OCREvaluator()
    dataset_path = Path("docling-eval/custom-dataset/ground-truth")
    output_path = Path("docling-eval/custom-dataset/ground-truth")
    evaluation_results = evaluator(dataset_path, output_path)
    print(f"Completed evaluation. Results saved to {output_path}")
