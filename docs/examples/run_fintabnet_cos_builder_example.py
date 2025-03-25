from pathlib import Path

from docling_eval.benchmarks.constants import BenchMarkNames, EvaluationModality
from docling_eval.cli.main import evaluate
from docling_eval_next.dataset_builders.fintabnet_cos_builder import (
    FintabnetCOSTableStructureDatasetBuilder,
)
from docling_eval_next.prediction_providers.azure_prediction_provider import (
    AzureDocIntelligencePredictionProvider,
)


def main():
    """Main function that will run Fintabnet Table Structure Benchmark.
    Pulls the 'fintabnet' dataset from HF and saves it to disk.
    """
    # Define the place where the temporary output has to be saved
    target_path = Path("./output/tables_quality_fintabnet_crops/")

    # Define the predictor that needs to be run on each item of the dataset
    provider_args = {
        "skip_api_if_prediction_is_present": True,
        "predictions_dir": target_path / "microsoft",
    }
    provider = AzureDocIntelligencePredictionProvider(
        **provider_args
    )  # Microsoft Azure Document Intelligence API Provider

    # 1. Create the dataset builder
    dataset = FintabnetCOSTableStructureDatasetBuilder(
        prediction_provider=provider,
        target=target_path,
    )

    # 2. Download the dataset
    downloaded_path = (
        dataset.retrieve_input_dataset()
    )  # fetches the source dataset from HF
    print(f"Dataset downloaded to {downloaded_path}")

    # 3. Run prediction and save the output in parquet format locally; Note that this saved data will have both ground truth and prediction
    dataset.save_to_disk()  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    # 4. Run evaluation using the saved data in step 3 above
    evaluate(
        modality=EvaluationModality.TABLE_STRUCTURE,
        benchmark=BenchMarkNames.FINTABNET,
        idir=target_path,
        odir=target_path / "teds",
        split="intermediate_files",
    )


if __name__ == "__main__":
    main()
