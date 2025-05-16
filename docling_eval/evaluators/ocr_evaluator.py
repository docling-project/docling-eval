import copy
import glob
import json
import logging
import traceback
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset
from docling_core.types.doc import CoordOrigin
from docling_core.types.doc.page import SegmentedPage
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from tqdm import tqdm

from docling_eval.datamodels.dataset_record import DatasetRecordWithPrediction
from docling_eval.datamodels.types import BenchMarkColumns, PredictionFormats
from docling_eval.evaluators.base_evaluator import BaseEvaluator
from docling_eval.evaluators.ocr.benchmark_framework import ModelBenchmark
from docling_eval.evaluators.ocr.utils import (
    _create_ocr_dictionary_from_segmented_page,
    _process_segpages_data,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


_log = logging.getLogger(__name__)


class DatasetOcrEvaluation(BaseModel):
    f1_score: float = 0.0
    recall: float = 0.0
    precision: float = 0.0


class OCREvaluator(BaseEvaluator):
    def __init__(
        self,
        intermediate_evaluations_path: Optional[Path] = None,
        prediction_sources: List[PredictionFormats] = [
            PredictionFormats.DOCLING_DOCUMENT
        ],
    ):
        super().__init__(
            intermediate_evaluations_path=intermediate_evaluations_path,
            prediction_sources=prediction_sources,
            supported_prediction_formats=[PredictionFormats.DOCLING_DOCUMENT],
        )

    def __call__(
        self,
        ds_path: Path,
        split: str = "test",
    ) -> DatasetOcrEvaluation:
        ignore_zone_filter_type = "default"
        add_space_prediction = True
        add_space_gt = True
        benchmark_runner = ModelBenchmark(
            model_name="ocr_iocr",
            ignore_zone_filter=ignore_zone_filter_type,
            add_space_between_merged_prediction_words=add_space_prediction,
            add_space_between_merged_gt_words=add_space_gt,
        )

        _log.info("Loading the split '%s' from: '%s'", split, ds_path)
        split_path = str(ds_path / split / "*.parquet")
        split_files = glob.glob(split_path)
        if not split_files:
            _log.warning(
                "No parquet files found for split '%s' in '%s'", split, ds_path
            )
            return DatasetOcrEvaluation()

        _log.info("Found %d files: %s", len(split_files), split_files)
        ds = load_dataset("parquet", data_files={split: split_files})
        _log.info("Overview of dataset: %s", ds)

        ds_selection: Dataset = ds[split]

        # page_evaluations_list: List[PageOcrEvaluation] = []
        num_processed_items = 0

        for i, data_item_row in tqdm(
            enumerate(ds_selection),
            desc="Evaluating OCR",
            ncols=120,
            total=len(ds_selection),
        ):
            # NOTE: Somehow the validation of the data record is not working as expected
            # try:
            #     data_record = DatasetRecordWithPrediction.model_validate(data)
            # except Exception as e:
            #     _log.error("Failed to validate record %d: %s. Data: %s", i, e, data)
            #     continue

            # doc_id = data_record.doc_id

            # if data_record.status not in self._accepted_status:
            #     _log.warning(
            #         "Skipping record %s due to status: %s", doc_id, data_record.status
            #     )
            #     continue

            # true_segpages = data_record.ground_truth_segmented_pages
            # pred_segpages = data_record.predicted_segmented_pages

            if BenchMarkColumns.DOC_ID not in data_item_row:
                print(f"Skipping item {i} due to missing '{BenchMarkColumns.DOC_ID}'.")
                continue

            doc_id_value = data_item_row[BenchMarkColumns.DOC_ID]
            image_path_from_data_item = data_item_row.get(BenchMarkColumns.DOC_ID)

            gt_ocr_data_for_benchmark = None
            pred_ocr_data_for_benchmark = None

            gt_segpages_column_key = "ground_truth_segmented_pages"
            if (
                gt_segpages_column_key in data_item_row
                and data_item_row[gt_segpages_column_key]
            ):
                try:
                    gt_segmented_pages_map_data = _process_segpages_data(
                        data_item_row[gt_segpages_column_key], doc_id_value
                    )
                    if gt_segmented_pages_map_data:
                        page_to_process_idx_gt = sorted(
                            gt_segmented_pages_map_data.keys()
                        )[0]
                        gt_segmented_page_to_process = gt_segmented_pages_map_data[
                            page_to_process_idx_gt
                        ]
                        gt_page_dict_data = _create_ocr_dictionary_from_segmented_page(
                            segmented_page=gt_segmented_page_to_process,
                            doc_id=doc_id_value,
                            page_number=page_to_process_idx_gt,
                            image_path_override=image_path_from_data_item,
                        )
                        gt_ocr_data_for_benchmark = {"images": [gt_page_dict_data]}
                    else:
                        print(f"No valid GT segmented pages for {doc_id_value}")
                except Exception as e:
                    print(f"Error processing GT for {doc_id_value}: {e}")
                    traceback.print_exc()

            if not gt_ocr_data_for_benchmark:
                gt_ocr_data_for_benchmark = {
                    "images": [
                        {
                            "image_path": image_path_from_data_item
                            or f"{doc_id_value}_gt_fallback_page_0.png",
                            "page_dimensions": {"width": 0, "height": 0},
                            "words": [],
                            "category": "",
                            "sub-category": "",
                        }
                    ]
                }

            pred_segpages_column_key = "predicted_segmented_pages"
            if (
                pred_segpages_column_key in data_item_row
                and data_item_row[pred_segpages_column_key]
            ):
                try:
                    pred_segmented_pages_map_data = _process_segpages_data(
                        data_item_row[pred_segpages_column_key], doc_id_value
                    )
                    if pred_segmented_pages_map_data:
                        page_to_process_idx_pred = sorted(
                            pred_segmented_pages_map_data.keys()
                        )[0]
                        pred_segmented_page_to_process = pred_segmented_pages_map_data[
                            page_to_process_idx_pred
                        ]
                        pred_page_dict_data = (
                            _create_ocr_dictionary_from_segmented_page(
                                segmented_page=pred_segmented_page_to_process,
                                doc_id=doc_id_value,
                                page_number=page_to_process_idx_pred,
                                image_path_override=image_path_from_data_item,
                            )
                        )
                        pred_ocr_data_for_benchmark = {"images": [pred_page_dict_data]}
                    else:
                        print(f"No valid Pred segmented pages for {doc_id_value}")
                except Exception as e:
                    print(f"Error processing Pred for {doc_id_value}: {e}")
                    traceback.print_exc()

            if not pred_ocr_data_for_benchmark:
                pred_ocr_data_for_benchmark = {
                    "images": [
                        {
                            "image_path": image_path_from_data_item
                            or f"{doc_id_value}_pred_fallback_page_0.png",
                            "page_dimensions": {"width": 0, "height": 0},
                            "words": [],
                            "category": "",
                            "sub-category": "",
                        }
                    ]
                }

            if (
                gt_ocr_data_for_benchmark["images"][0].get("words") is not None
                and pred_ocr_data_for_benchmark["images"][0].get("words") is not None
            ):
                benchmark_runner.load_res_and_gt_files(
                    gt_words=gt_ocr_data_for_benchmark,
                    pred_words=pred_ocr_data_for_benchmark,
                )
                num_processed_items += 1
            else:
                print(
                    f"Skipping {doc_id_value} due to missing 'words' list in processed OCR data."
                )

        if num_processed_items > 0:
            print(f"Processed {num_processed_items} documents for benchmark.")
            benchmark_runner.run_benchmark()
            final_metrics_results = benchmark_runner.get_metrics_values(
                float_precision=1
            )
            print("\nFinal Metrics:")
            print(json.dumps(final_metrics_results, indent=2))
        else:
            print("No documents were processed for the benchmark.")

        if final_metrics_results and isinstance(final_metrics_results, list):
            metrics = final_metrics_results[0]
            dataset_evaluation_result = DatasetOcrEvaluation(
                f1_score=metrics.get("F1", 0.0),
                recall=metrics.get("Recall", 0.0),
                precision=metrics.get("Precision", 0.0),
            )
        else:
            dataset_evaluation_result = DatasetOcrEvaluation()

        _log.info(f"Final F1 Score: {dataset_evaluation_result.f1_score:.4f}")
        _log.info(f"Final Precision: {dataset_evaluation_result.precision:.4f}")
        _log.info(f"Final Recall: {dataset_evaluation_result.recall:.4f}")

        return dataset_evaluation_result


class OCRVisualizer:
    """
    Generate visualizations comparing ground truth and predicted OCR results
    """

    def __init__(self):
        self._line_width = 2
        self._true_box_color = "green"
        self._pred_box_color = "red"
        self._correct_box_color = "blue"
        self._text_color = "black"
        self._viz_sub_dir = "ocr_viz"

        self._font = ImageFont.load_default()
        try:
            self._font = ImageFont.truetype("arial.ttf", size=10)
        except IOError:
            self._font = ImageFont.load_default()

    def __call__(
        self,
        ds_path: Path,
        ocr_report_fn: Optional[Path] = None,
        save_dir: Path = Path("./"),
        split: str = "test",
    ):
        """
        Visualize the original and predicted OCR results. Generate one visualization per document
        and save it in the output dir.

        Args:
            ds_path: Path to the dataset
            ocr_report_fn: Optional path to pre-generated OCR report
            save_dir: Directory to save visualizations
            split: Dataset split to use
        """
        save_dir /= self._viz_sub_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        ocr_preds_idx: dict[str, Dict] = {}
        if ocr_report_fn and ocr_report_fn.exists():
            with open(ocr_report_fn, "r") as fd:
                ocr_evaluation_dict = json.load(fd)
                for evaluation in ocr_evaluation_dict.get("evaluations", []):
                    doc_id = evaluation["doc_id"]
                    ocr_preds_idx[doc_id] = evaluation

        parquet_files = str(ds_path / split / "*.parquet")
        ds = load_dataset("parquet", data_files={split: parquet_files})
        if ds is not None:
            ds_selection = ds[split]

        viz_fns: list[Path] = []
        for i, data in tqdm(
            enumerate(ds_selection),
            desc="OCR visualizations",
            ncols=120,
            total=len(ds_selection),
        ):
            doc_id = data[BenchMarkColumns.DOC_ID]
            page_images = data[BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES]

            if ocr_report_fn and doc_id not in ocr_preds_idx:
                continue

            true_segpages = {}
            pred_segpages = {}

            if "ground_truth_segmented_pages" in data:
                if isinstance(data["ground_truth_segmented_pages"], bytes):
                    try:
                        true_segpages_dict = json.loads(
                            data["ground_truth_segmented_pages"]
                        )
                        for page_idx, page_data in true_segpages_dict.items():
                            page_idx = (
                                int(page_idx) if isinstance(page_idx, str) else page_idx
                            )
                            true_segpages[page_idx] = SegmentedPage.model_validate(
                                page_data
                            )
                    except Exception as e:
                        print(
                            f"Error deserializing ground truth segmented pages for doc {doc_id}: {e}"
                        )
                elif isinstance(data["ground_truth_segmented_pages"], dict):
                    for page_idx, page_data in data[
                        "ground_truth_segmented_pages"
                    ].items():
                        page_idx = (
                            int(page_idx) if isinstance(page_idx, str) else page_idx
                        )
                        if isinstance(page_data, dict):
                            true_segpages[page_idx] = SegmentedPage.model_validate(
                                page_data
                            )
                        else:
                            try:
                                true_segpages[page_idx] = (
                                    SegmentedPage.model_validate_json(page_data)
                                )
                            except Exception as e:
                                print(
                                    f"Error parsing page data for gt doc {doc_id}: {e}"
                                )

            if "predicted_segmented_pages" in data:
                if isinstance(data["predicted_segmented_pages"], bytes):
                    try:
                        pred_segpages_dict = json.loads(
                            data["predicted_segmented_pages"]
                        )
                        for page_idx, page_data in pred_segpages_dict.items():
                            page_idx = (
                                int(page_idx) if isinstance(page_idx, str) else page_idx
                            )
                            pred_segpages[page_idx] = SegmentedPage.model_validate(
                                page_data
                            )
                    except Exception as e:
                        print(
                            f"Error deserializing predicted segmented pages for doc {doc_id}: {e}"
                        )
                elif isinstance(data["predicted_segmented_pages"], dict):
                    for page_idx, page_data in data[
                        "predicted_segmented_pages"
                    ].items():
                        page_idx = (
                            int(page_idx) if isinstance(page_idx, str) else page_idx
                        )
                        if isinstance(page_data, dict):
                            pred_segpages[page_idx] = SegmentedPage.model_validate(
                                page_data
                            )
                        else:
                            try:
                                pred_segpages[page_idx] = (
                                    SegmentedPage.model_validate_json(page_data)
                                )
                            except Exception as e:
                                print(
                                    f"Error parsing page data for pred doc {doc_id}: {e}"
                                )

            image_bytes = page_images[0]["bytes"]
            image = Image.open(BytesIO(image_bytes))

            viz_image = self._draw_ocr_comparison(
                doc_id, image, true_segpages, pred_segpages
            )
            viz_fn = save_dir / f"{doc_id}_ocr_viz.png"
            viz_fns.append(viz_fn)
            viz_image.save(viz_fn)

        return viz_fns

    def _draw_ocr_comparison(
        self,
        doc_id: str,
        page_image: Image.Image,
        true_segpages: Dict[int, SegmentedPage],
        pred_segpages: Dict[int, SegmentedPage],
    ) -> Image.Image:
        """
        Draw the ground truth and predicted OCR results on the same image
        """
        true_img = copy.deepcopy(page_image)
        pred_img = copy.deepcopy(page_image)

        true_draw = ImageDraw.Draw(true_img)
        pred_draw = ImageDraw.Draw(pred_img)

        if not true_segpages:
            print(f"No ground truth segmented pages found for doc {doc_id}")
            return page_image

        page_idx = list(true_segpages.keys())[0]
        true_page = true_segpages[page_idx]

        if page_idx not in pred_segpages:
            page_idx = list(pred_segpages.keys())[0] if pred_segpages else -1

        pred_page = pred_segpages.get(page_idx, None)

        page_height = true_page.dimension.height
        page_width = true_page.dimension.width

        scale_x = page_image.width / page_width
        scale_y = page_image.height / page_height

        if true_page and true_page.has_words:
            for cell in true_page.word_cells:
                bbox = cell.rect.to_bounding_box()
                if bbox.coord_origin != CoordOrigin.TOPLEFT:
                    bbox = bbox.to_top_left_origin(page_height=page_height)

                l = round(bbox.l * scale_x)
                r = round(bbox.r * scale_x)
                t = round(bbox.t * scale_y)
                b = round(bbox.b * scale_y)

                true_draw.rectangle(
                    [l, t, r, b],
                    outline=self._true_box_color,
                    width=self._line_width,
                )

                text_pos = (l, t - 15) if t > 15 else (l, b + 2)
                true_draw.text(
                    text_pos,
                    cell.text,
                    fill=self._text_color,
                    font=self._font,
                )

        if pred_page and pred_page.has_words:
            for cell in pred_page.word_cells:
                bbox = cell.rect.to_bounding_box()
                if bbox.coord_origin != CoordOrigin.TOPLEFT:
                    bbox = bbox.to_top_left_origin(page_height=page_height)

                l = round(bbox.l * scale_x)
                r = round(bbox.r * scale_x)
                t = round(bbox.t * scale_y)
                b = round(bbox.b * scale_y)

                is_correct = False
                if true_page and true_page.has_words:
                    for gt_cell in true_page.word_cells:
                        if gt_cell.text == cell.text:
                            gt_bbox = gt_cell.rect.to_bounding_box()
                            if gt_bbox.coord_origin != CoordOrigin.TOPLEFT:
                                gt_bbox = gt_bbox.to_top_left_origin(
                                    page_height=page_height
                                )

                            if not (
                                l > round(gt_bbox.r * scale_x)
                                or r < round(gt_bbox.l * scale_x)
                                or t > round(gt_bbox.b * scale_y)
                                or b < round(gt_bbox.t * scale_y)
                            ):
                                is_correct = True
                                break

                box_color = (
                    self._correct_box_color if is_correct else self._pred_box_color
                )

                pred_draw.rectangle(
                    [l, t, r, b],
                    outline=box_color,
                    width=self._line_width,
                )

                text_pos = (l, t - 15) if t > 15 else (l, b + 2)
                pred_draw.text(
                    text_pos,
                    cell.text,
                    fill=self._text_color,
                    font=self._font,
                )

        mode = page_image.mode
        w, h = page_image.size
        combined_img = Image.new(mode, (2 * w, h), "white")
        combined_img.paste(true_img, (0, 0))
        combined_img.paste(pred_img, (w, 0))

        combined_draw = ImageDraw.Draw(combined_img)
        combined_draw.text(
            (10, 10),
            "Ground Truth OCR",
            fill="black",
            font=self._font,
        )
        combined_draw.text(
            (w + 10, 10),
            "Predicted OCR",
            fill="black",
            font=self._font,
        )

        legend_y = 30
        combined_draw.rectangle(
            [10, legend_y, 30, legend_y + 15],
            outline=self._true_box_color,
            fill=self._true_box_color,
        )
        combined_draw.text(
            (35, legend_y), "Ground Truth Word", fill="black", font=self._font
        )

        combined_draw.rectangle(
            [w + 10, legend_y, w + 30, legend_y + 15],
            outline=self._correct_box_color,
            fill=self._correct_box_color,
        )
        combined_draw.text(
            (w + 35, legend_y), "Correct Prediction", fill="black", font=self._font
        )

        combined_draw.rectangle(
            [w + 10, legend_y + 25, w + 30, legend_y + 40],
            outline=self._pred_box_color,
            fill=self._pred_box_color,
        )
        combined_draw.text(
            (w + 35, legend_y + 25),
            "Incorrect Prediction",
            fill="black",
            font=self._font,
        )

        return combined_img
