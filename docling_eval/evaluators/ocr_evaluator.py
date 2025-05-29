import copy
import glob
import json
import logging
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset
from docling_core.types.doc import CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, PageGeometry, SegmentedPage
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from tqdm import tqdm

from docling_eval.datamodels.types import BenchMarkColumns, PredictionFormats
from docling_eval.evaluators.base_evaluator import BaseEvaluator
from docling_eval.evaluators.ocr.benchmark_constants import OcrReportEvaluationEntry
from docling_eval.evaluators.ocr.benchmark_framework import _ModelBenchmark

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


_log = logging.getLogger(__name__)


class DatasetOcrEvaluation(BaseModel):
    f1_score: float = 0.0
    recall: float = 0.0
    precision: float = 0.0


def _parse_segmented_pages_from_raw(
    segpages_data_raw: Any, doc_id: str
) -> Optional[Dict[int, SegmentedPage]]:
    segmented_pages_map: Dict[int, SegmentedPage] = {}
    if isinstance(segpages_data_raw, (bytes, str)):
        try:
            segpages_dict_payload: Any = json.loads(segpages_data_raw)
        except json.JSONDecodeError as e:
            _log.warning(
                f"JSONDecodeError for doc {doc_id}: {e}. Data: {str(segpages_data_raw)[:200]}"
            )
            return None
    elif isinstance(segpages_data_raw, dict):
        segpages_dict_payload = segpages_data_raw
    else:
        _log.warning(
            f"Unrecognized segmented_pages data format for doc {doc_id}: {type(segpages_data_raw)}"
        )
        return None

    if not isinstance(segpages_dict_payload, dict):
        _log.warning(
            f"Expected dict payload for segmented_pages for doc {doc_id}, got {type(segpages_dict_payload)}"
        )
        return None

    for page_idx_str, page_data_item in segpages_dict_payload.items():
        try:
            page_idx: int = int(page_idx_str)
        except ValueError:
            _log.warning(
                f"Invalid page index string '{page_idx_str}' for doc {doc_id}. Skipping page."
            )
            continue

        try:
            if isinstance(page_data_item, dict):
                segmented_pages_map[page_idx] = SegmentedPage.model_validate(
                    page_data_item
                )
            elif isinstance(page_data_item, str):
                segmented_pages_map[page_idx] = SegmentedPage.model_validate_json(
                    page_data_item
                )
            elif isinstance(page_data_item, SegmentedPage):
                segmented_pages_map[page_idx] = page_data_item
            else:
                _log.warning(
                    f"Unrecognized page_data format for doc {doc_id}, page {page_idx}: {type(page_data_item)}"
                )
                continue
        except Exception as e_page_val:
            _log.error(
                f"Error validating page data for doc {doc_id}, page {page_idx}: {e_page_val}"
            )
            traceback.print_exc()
            continue
    return segmented_pages_map if segmented_pages_map else None


class OCREvaluator(BaseEvaluator):
    def __init__(
        self,
        intermediate_evaluations_path: Optional[Path] = None,
        prediction_sources: List[PredictionFormats] = [
            PredictionFormats.DOCLING_DOCUMENT
        ],
    ) -> None:
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
        benchmark_runner = _ModelBenchmark(
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

        num_processed_items = 0
        final_metrics_results: List[Dict[str, Any]] = []

        empty_rect = BoundingRectangle(
            r_x0=0,
            r_y0=0,
            r_x1=0,
            r_y1=0,
            r_x2=0,
            r_y2=0,
            r_x3=0,
            r_y3=0,
            coord_origin=CoordOrigin.TOPLEFT,
        )
        empty_page_geometry = PageGeometry(angle=0, rect=empty_rect)
        fallback_segmented_page = SegmentedPage(dimension=empty_page_geometry)

        for i, data_item_row in tqdm(
            enumerate(ds_selection),
            desc="Evaluating OCR",
            ncols=120,
            total=len(ds_selection),
        ):
            if BenchMarkColumns.DOC_ID not in data_item_row:
                _log.warning(
                    f"Skipping item {i} due to missing '{BenchMarkColumns.DOC_ID}'."
                )
                continue

            doc_id_value: str = data_item_row[BenchMarkColumns.DOC_ID]

            gt_segmented_page: SegmentedPage = copy.deepcopy(fallback_segmented_page)
            pred_segmented_page: SegmentedPage = copy.deepcopy(fallback_segmented_page)

            gt_segpages_column_key = BenchMarkColumns.GROUNDTRUTH_SEGMENTED_PAGES
            if (
                gt_segpages_column_key in data_item_row
                and data_item_row[gt_segpages_column_key]
            ):
                try:
                    gt_segmented_pages_map_data: Optional[Dict[int, SegmentedPage]] = (
                        _parse_segmented_pages_from_raw(
                            data_item_row[gt_segpages_column_key], doc_id_value
                        )
                    )
                    if gt_segmented_pages_map_data:
                        page_to_process_idx_gt: int = sorted(
                            gt_segmented_pages_map_data.keys()
                        )[0]
                        gt_segmented_page = gt_segmented_pages_map_data[
                            page_to_process_idx_gt
                        ]
                    else:
                        _log.warning(
                            f"No valid GT segmented pages for {doc_id_value}, using fallback."
                        )
                except Exception as e:
                    _log.error(
                        f"Error processing GT for {doc_id_value}: {e}, using fallback."
                    )
                    traceback.print_exc()

            pred_segpages_column_key = BenchMarkColumns.PREDICTED_SEGMENTED_PAGES
            if (
                pred_segpages_column_key in data_item_row
                and data_item_row[pred_segpages_column_key]
            ):
                try:
                    pred_segmented_pages_map_data: Optional[
                        Dict[int, SegmentedPage]
                    ] = _parse_segmented_pages_from_raw(
                        data_item_row[pred_segpages_column_key], doc_id_value
                    )
                    if pred_segmented_pages_map_data:
                        page_to_process_idx_pred: int = sorted(
                            pred_segmented_pages_map_data.keys()
                        )[0]
                        pred_segmented_page = pred_segmented_pages_map_data[
                            page_to_process_idx_pred
                        ]
                    else:
                        _log.warning(
                            f"No valid Pred segmented pages for {doc_id_value}, using fallback."
                        )
                except Exception as e:
                    _log.error(
                        f"Error processing Pred for {doc_id_value}: {e}, using fallback."
                    )
                    traceback.print_exc()

            benchmark_runner.load_res_and_gt_files(
                gt_page=gt_segmented_page,
                pred_page=pred_segmented_page,
            )
            num_processed_items += 1

        if num_processed_items > 0:
            _log.info(f"Processed {num_processed_items} documents for benchmark.")
            benchmark_runner.run_benchmark()
            final_metrics_results = benchmark_runner.get_metrics_values(
                float_precision=1
            )
            _log.info("\nFinal Metrics:")
            _log.info(json.dumps(final_metrics_results, indent=2))
        else:
            _log.warning("No documents were processed for the benchmark.")
            final_metrics_results = []

        dataset_evaluation_result = DatasetOcrEvaluation()
        if (
            final_metrics_results
            and isinstance(final_metrics_results, list)
            and len(final_metrics_results) > 0
        ):
            metrics_dict_val: Dict[str, Any] = final_metrics_results[0]
            if isinstance(metrics_dict_val, dict):
                dataset_evaluation_result = DatasetOcrEvaluation(
                    f1_score=metrics_dict_val.get("F1", 0.0),
                    recall=metrics_dict_val.get("Recall", 0.0),
                    precision=metrics_dict_val.get("Precision", 0.0),
                )

        _log.info(f"Final F1 Score: {dataset_evaluation_result.f1_score:.4f}")
        _log.info(f"Final Precision: {dataset_evaluation_result.precision:.4f}")
        _log.info(f"Final Recall: {dataset_evaluation_result.recall:.4f}")

        return dataset_evaluation_result


class OCRVisualizer:
    def __init__(self) -> None:
        self._line_width: int = 2
        self._true_box_color: str = "green"
        self._pred_box_color: str = "red"
        self._correct_box_color: str = "blue"
        self._text_color: str = "black"
        self._viz_sub_dir: str = "ocr_viz"

        self._font: ImageFont.FreeTypeFont = ImageFont.load_default()
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
    ) -> List[Path]:
        save_dir_path: Path = save_dir / self._viz_sub_dir
        save_dir_path.mkdir(parents=True, exist_ok=True)

        ocr_preds_idx: Dict[str, OcrReportEvaluationEntry] = {}
        if ocr_report_fn and ocr_report_fn.exists():
            with open(ocr_report_fn, "r") as fd:
                ocr_evaluation_dict_file_content: Dict[str, Any] = json.load(fd)
                for evaluation_item_dict in ocr_evaluation_dict_file_content.get(
                    "evaluations", []
                ):
                    try:
                        entry = OcrReportEvaluationEntry.model_validate(
                            evaluation_item_dict
                        )
                        ocr_preds_idx[entry.doc_id] = entry
                    except Exception as e:
                        _log.warning(
                            f"Failed to parse evaluation item: {evaluation_item_dict}. Error: {e}"
                        )

        parquet_files: str = str(ds_path / split / "*.parquet")
        ds: Dataset = load_dataset("parquet", data_files={split: parquet_files})

        viz_fns: List[Path] = []
        if ds is not None and split in ds:
            ds_selection: Dataset = ds[split]

            for i, data_item_row in tqdm(
                enumerate(ds_selection),
                desc="OCR visualizations",
                ncols=120,
                total=len(ds_selection),
            ):
                doc_id: str = data_item_row[BenchMarkColumns.DOC_ID]
                page_images_raw: Any = data_item_row.get(
                    BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES
                )

                page_images: List[Dict[str, bytes]] = []
                if isinstance(page_images_raw, list) and page_images_raw:
                    if (
                        isinstance(page_images_raw[0], dict)
                        and "bytes" in page_images_raw[0]
                    ):
                        page_images = page_images_raw

                if ocr_report_fn and doc_id not in ocr_preds_idx:
                    continue

                true_segpages: Dict[int, SegmentedPage] = {}
                pred_segpages: Dict[int, SegmentedPage] = {}

                gt_column: str = BenchMarkColumns.GROUNDTRUTH_SEGMENTED_PAGES
                if gt_column in data_item_row and data_item_row[gt_column]:
                    processed_gt: Optional[Dict[int, SegmentedPage]] = (
                        _parse_segmented_pages_from_raw(
                            data_item_row[gt_column], doc_id
                        )
                    )
                    if processed_gt:
                        true_segpages = processed_gt

                pred_column: str = BenchMarkColumns.PREDICTED_SEGMENTED_PAGES
                if pred_column in data_item_row and data_item_row[pred_column]:
                    processed_pred: Optional[Dict[int, SegmentedPage]] = (
                        _parse_segmented_pages_from_raw(
                            data_item_row[pred_column], doc_id
                        )
                    )
                    if processed_pred:
                        pred_segpages = processed_pred

                if not page_images:
                    _log.warning(
                        f"No page images found for doc {doc_id}. Skipping visualization."
                    )
                    continue

                image_bytes: bytes = page_images[0]["bytes"]
                image: Image.Image = Image.open(BytesIO(image_bytes)).convert("RGB")

                viz_image: Image.Image = self._draw_ocr_comparison(
                    doc_id, image, true_segpages, pred_segpages
                )
                viz_fn: Path = save_dir_path / f"{doc_id}_ocr_viz.png"
                viz_fns.append(viz_fn)
                viz_image.save(viz_fn)
        else:
            _log.warning(
                f"Dataset or split '{split}' not found. No visualizations generated."
            )

        return viz_fns

    def _draw_ocr_comparison(
        self,
        doc_id: str,
        page_image: Image.Image,
        true_segpages: Dict[int, SegmentedPage],
        pred_segpages: Dict[int, SegmentedPage],
    ) -> Image.Image:
        true_img: Image.Image = copy.deepcopy(page_image)
        pred_img: Image.Image = copy.deepcopy(page_image)

        true_draw: ImageDraw.ImageDraw = ImageDraw.Draw(true_img)
        pred_draw: ImageDraw.ImageDraw = ImageDraw.Draw(pred_img)

        if not true_segpages:
            _log.warning(
                f"No ground truth segmented pages found for doc {doc_id} to draw."
            )

        true_page_idx_list: List[int] = list(true_segpages.keys())
        pred_page_idx_list: List[int] = list(pred_segpages.keys())

        page_idx_to_draw: int = -1
        if true_page_idx_list:
            page_idx_to_draw = true_page_idx_list[0]
        elif pred_page_idx_list:
            page_idx_to_draw = pred_page_idx_list[0]

        true_page: Optional[SegmentedPage] = (
            true_segpages.get(page_idx_to_draw) if page_idx_to_draw != -1 else None
        )
        pred_page: Optional[SegmentedPage] = (
            pred_segpages.get(page_idx_to_draw) if page_idx_to_draw != -1 else None
        )

        page_height: float = 0.0
        page_width: float = 0.0

        if true_page:
            page_height = true_page.dimension.height
            page_width = true_page.dimension.width
        elif pred_page:
            page_height = pred_page.dimension.height
            page_width = pred_page.dimension.width

        if page_width == 0 or page_height == 0:
            page_width = float(page_image.width)
            page_height = float(page_image.height)

        scale_x: float
        scale_y: float
        if page_width == 0 or page_height == 0:
            _log.warning(
                f"Page dimensions are zero for doc {doc_id}. Cannot scale drawings. Using 1.0."
            )
            scale_x, scale_y = 1.0, 1.0
        else:
            scale_x = page_image.width / page_width
            scale_y = page_image.height / page_height

        if true_page and true_page.has_words:
            for cell in true_page.word_cells:
                bbox = cell.rect.to_bounding_box()
                if bbox.coord_origin != CoordOrigin.TOPLEFT:
                    if page_height == 0:
                        _log.warning(
                            f"Page height is zero for doc {doc_id} with GT cells needing coordinate conversion. Conversion may be incorrect."
                        )
                    bbox = bbox.to_top_left_origin(page_height=page_height)

                l: int = round(bbox.l * scale_x)
                r: int = round(bbox.r * scale_x)
                t: int = round(bbox.t * scale_y)
                b: int = round(bbox.b * scale_y)

                true_draw.rectangle(
                    [l, t, r, b],
                    outline=self._true_box_color,
                    width=self._line_width,
                )

                text_pos: Tuple[int, int] = (l, t - 15) if t > 15 else (l, b + 2)
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
                    if page_height == 0:
                        _log.warning(
                            f"Page height is zero for doc {doc_id} with Pred cells needing coordinate conversion. Conversion may be incorrect."
                        )
                    bbox = bbox.to_top_left_origin(page_height=page_height)

                l: int = round(bbox.l * scale_x)
                r: int = round(bbox.r * scale_x)
                t: int = round(bbox.t * scale_y)
                b: int = round(bbox.b * scale_y)

                is_correct: bool = False
                if true_page and true_page.has_words:
                    for gt_cell in true_page.word_cells:
                        if gt_cell.text == cell.text:
                            gt_bbox = gt_cell.rect.to_bounding_box()
                            if gt_bbox.coord_origin != CoordOrigin.TOPLEFT:
                                if page_height == 0:
                                    _log.warning(
                                        f"Page height is zero for doc {doc_id} with GT cells (for Pred comparison) needing coord conversion. Conversion may be incorrect."
                                    )
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

                box_color: str = (
                    self._correct_box_color if is_correct else self._pred_box_color
                )

                pred_draw.rectangle(
                    [l, t, r, b],
                    outline=box_color,
                    width=self._line_width,
                )

                text_pos: Tuple[int, int] = (l, t - 15) if t > 15 else (l, b + 2)
                pred_draw.text(
                    text_pos,
                    cell.text,
                    fill=self._text_color,
                    font=self._font,
                )

        mode: str = page_image.mode
        w: int
        h: int
        w, h = page_image.size
        combined_img: Image.Image = Image.new(mode, (2 * w, h), "white")
        combined_img.paste(true_img, (0, 0))
        combined_img.paste(pred_img, (w, 0))

        combined_draw: ImageDraw.ImageDraw = ImageDraw.Draw(combined_img)
        title_font_size: int = max(15, int(h * 0.02))
        legend_font_size: int = max(12, int(h * 0.015))
        title_font: ImageFont.FreeTypeFont
        legend_font: ImageFont.FreeTypeFont
        try:
            title_font = ImageFont.truetype("arial.ttf", size=title_font_size)
            legend_font = ImageFont.truetype("arial.ttf", size=legend_font_size)
        except IOError:
            title_font = self._font
            legend_font = self._font

        combined_draw.text(
            (10, 10),
            "Ground Truth OCR",
            fill="black",
            font=title_font,
        )
        combined_draw.text(
            (w + 10, 10),
            "Predicted OCR",
            fill="black",
            font=title_font,
        )

        legend_y_start: int = title_font_size + 20
        legend_box_size: int = legend_font_size
        legend_spacing: int = int(legend_font_size * 0.5)

        current_legend_y: int = legend_y_start
        combined_draw.rectangle(
            [
                10,
                current_legend_y,
                10 + legend_box_size,
                current_legend_y + legend_box_size,
            ],
            outline=self._true_box_color,
            fill=self._true_box_color,
        )
        combined_draw.text(
            (15 + legend_box_size, current_legend_y),
            "Ground Truth Word",
            fill="black",
            font=legend_font,
        )

        current_legend_y_pred: int = legend_y_start
        combined_draw.rectangle(
            [
                w + 10,
                current_legend_y_pred,
                w + 10 + legend_box_size,
                current_legend_y_pred + legend_box_size,
            ],
            outline=self._correct_box_color,
            fill=self._correct_box_color,
        )
        combined_draw.text(
            (w + 15 + legend_box_size, current_legend_y_pred),
            "Correct Prediction",
            fill="black",
            font=legend_font,
        )

        current_legend_y_pred += legend_box_size + legend_spacing
        combined_draw.rectangle(
            [
                w + 10,
                current_legend_y_pred,
                w + 10 + legend_box_size,
                current_legend_y_pred + legend_box_size,
            ],
            outline=self._pred_box_color,
            fill=self._pred_box_color,
        )
        combined_draw.text(
            (w + 15 + legend_box_size, current_legend_y_pred),
            "Incorrect Prediction",
            fill="black",
            font=legend_font,
        )

        return combined_img
