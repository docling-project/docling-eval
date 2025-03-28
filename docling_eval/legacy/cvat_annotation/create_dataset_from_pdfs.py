import argparse
import glob
import json
import os
from pathlib import Path

from datasets import Features
from datasets import Image as Features_Image
from datasets import Sequence, Value

from docling_eval.datamodels.types import BenchMarkColumns
from docling_eval.legacy.converters.conversion import create_pdf_docling_converter
from docling_eval.utils.utils import (
    docling_version,
    extract_images,
    get_binary,
    save_shard_to_disk,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process input and output directories and a pre-annotation file."
    )

    parser.add_argument(
        "-i",
        "--input_dir",
        required=True,
        help="Path to directory with pdf's",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Path to the output directory containing the dataset.",
    )
    parser.add_argument(
        "-b", "--bucket-size", required=True, help="Numbers of documents in the bucket."
    )

    args = parser.parse_args()

    return (Path(args.input_dir), Path(args.output_dir), int(args.bucket_size))


def _write_datasets_info(
    name: str, output_dir: Path, num_train_rows: int, num_test_rows: int
):
    features = Features(
        {
            BenchMarkColumns.CONVERTER_VERSION: Value("string"),
            BenchMarkColumns.STATUS: Value("string"),
            BenchMarkColumns.DOC_ID: Value("string"),
            # BenchMarkColumns.DOC_PATH: Value("string"),
            # BenchMarkColumns.DOC_HASH: Value("string"),
            # BenchMarkColumns.GROUNDTRUTH: Value("string"),
            # BenchMarkColumns.GROUNDTRUTH_PICTURES: Sequence(Features_Image()),
            # BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES: Sequence(Features_Image()),
            BenchMarkColumns.PREDICTION: Value("string"),
            BenchMarkColumns.PREDICTION_PICTURES: Sequence(Features_Image()),
            BenchMarkColumns.PREDICTION_PAGE_IMAGES: Sequence(Features_Image()),
            BenchMarkColumns.ORIGINAL: Value("string"),
            BenchMarkColumns.MIMETYPE: Value("string"),
        }
    )

    schema = features.to_dict()
    # print(json.dumps(schema, indent=2))

    dataset_infos = {
        "train": {
            "description": f"Training split of {name}",
            "schema": schema,
            "num_rows": num_train_rows,
        },
        "test": {
            "description": f"Test split of {name}",
            "schema": schema,
            "num_rows": num_test_rows,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "dataset_infos.json", "w") as fw:
        json.dump(dataset_infos, fw, indent=2)


def main():

    image_scale = 2
    artifacts_path = None

    source_dir, target_dir, bucket_size = parse_args()

    test_dir = target_dir / "test"
    train_dir = target_dir / "train"

    for _ in [target_dir, test_dir, train_dir]:
        if not os.path.exists(_):
            os.makedirs(_)

    # Create Converter
    doc_converter = create_pdf_docling_converter(
        page_image_scale=image_scale, artifacts_path=artifacts_path
    )

    pdfs = sorted(glob.glob(str(source_dir / "*.pdf")))

    records = []
    for pdf_path in pdfs:
        print(f"processing {pdf_path}")

        # Create the predicted Document
        conv_results = doc_converter.convert(source=pdf_path, raises_on_error=True)
        pred_doc = conv_results.document

        pred_doc, pred_pictures, pred_page_images = extract_images(
            document=pred_doc,
            pictures_column=BenchMarkColumns.PREDICTION_PICTURES.value,  # pictures_column,
            page_images_column=BenchMarkColumns.PREDICTION_PAGE_IMAGES.value,  # page_images_column,
        )

        record = {
            BenchMarkColumns.CONVERTER_VERSION: docling_version(),
            BenchMarkColumns.STATUS: str(conv_results.status),
            BenchMarkColumns.DOC_ID: str(os.path.basename(pdf_path)),
            BenchMarkColumns.PREDICTION: json.dumps(pred_doc.export_to_dict()),
            BenchMarkColumns.PREDICTION_PAGE_IMAGES: pred_page_images,
            BenchMarkColumns.PREDICTION_PICTURES: pred_pictures,
            BenchMarkColumns.ORIGINAL: get_binary(pdf_path),
            BenchMarkColumns.MIMETYPE: "application/pdf",
        }
        records.append(record)

    save_shard_to_disk(items=records, dataset_path=test_dir)

    _write_datasets_info(
        name="PDFBench: end-to-end",
        output_dir=target_dir,
        num_train_rows=0,
        num_test_rows=len(records),
    )


if __name__ == "__main__":
    main()
