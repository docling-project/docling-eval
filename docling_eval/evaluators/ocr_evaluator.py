import copy
import glob
import json
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from datasets import Dataset, load_dataset
from docling_core.types.doc import CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, PageGeometry, SegmentedPage
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from docling_eval.datamodels.dataset_record import DatasetRecordWithPrediction
from docling_eval.datamodels.types import BenchMarkColumns, PredictionFormats
from docling_eval.evaluators.base_evaluator import BaseEvaluator
from docling_eval.evaluators.ocr.benchmark_runner import _OcrBenchmark
from docling_eval.evaluators.ocr.evaluation_models import OcrDatasetEvaluationResult
from docling_eval.evaluators.ocr.processing_utils import parse_segmented_pages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
_log = logging.getLogger(__name__)


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
    ) -> OcrDatasetEvaluationResult:
        dataset_path = ds_path
        data_split_name = split
        ignore_zone_filter_config = "default"
        use_space_for_prediction_merge = True
        use_space_for_gt_merge = True

        benchmark_tool = _OcrBenchmark(
            model_identifier="ocr_model",
            ignore_zone_filter_type=ignore_zone_filter_config,
            add_space_for_merged_prediction_words=use_space_for_prediction_merge,
            add_space_for_merged_gt_words=use_space_for_gt_merge,
        )

        _log.info("Loading data split '%s' from: '%s'", data_split_name, dataset_path)
        path_to_split_files = str(dataset_path / data_split_name / "*.parquet")
        dataset_files = glob.glob(path_to_split_files)
        if not dataset_files:
            _log.warning(
                "No parquet files found for split '%s' in '%s'",
                data_split_name,
                dataset_path,
            )
            return OcrDatasetEvaluationResult()

        _log.info(
            "Found %d files for processing: %s", len(dataset_files), dataset_files
        )
        hf_dataset = load_dataset(
            "parquet", data_files={data_split_name: dataset_files}
        )
        _log.info("Dataset overview: %s", hf_dataset)

        selected_dataset_split: Dataset = hf_dataset[data_split_name]
        processed_item_count = 0

        empty_bounding_rect = BoundingRectangle(
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
        empty_page_dims = PageGeometry(angle=0, rect=empty_bounding_rect)

        for i, data_row in tqdm(
            enumerate(selected_dataset_split),
            desc="Evaluating OCR performance",
            ncols=120,
            total=len(selected_dataset_split),
        ):
            if BenchMarkColumns.DOC_ID not in data_row:
                _log.warning(
                    f"Skipping item {i} due to missing '{BenchMarkColumns.DOC_ID}' column."
                )
                continue

            # NOTE: Somehow the validation of the data record is not working as expected
            # try:
            #     data_record = DatasetRecordWithPrediction.model_validate(data_row)
            # except Exception as e:
            #     _log.error("Failed to validate record %d: %s. Data: %s", i, e, data_row)
            #     raise RuntimeError(
            #         f"Failed to validate record {i}: {e}. Data: {data_row}"
            #     )

            # doc_id = data_record.doc_id

            # if data_record.status not in self._accepted_status:
            #     _log.warning(
            #         "Skipping record %s due to status: %s", doc_id, data_record.status
            #     )
            #     continue

            # true_segpages = data_record.ground_truth_segmented_pages
            # pred_segpages = data_record.predicted_segmented_pages

            document_id: str = data_row[BenchMarkColumns.DOC_ID]
            gt_page_data: SegmentedPage = SegmentedPage(dimension=empty_page_dims)
            pred_page_data: SegmentedPage = SegmentedPage(dimension=empty_page_dims)

            page_identifier_for_benchmark: str = document_id

            gt_seg_pages_key = BenchMarkColumns.GROUNDTRUTH_SEGMENTED_PAGES

            if gt_seg_pages_key in data_row and data_row[gt_seg_pages_key]:
                try:
                    gt_pages_map: Optional[Dict[int, SegmentedPage]] = (
                        parse_segmented_pages(data_row[gt_seg_pages_key], document_id)
                    )
                    if gt_pages_map:
                        first_page_idx_gt: int = sorted(gt_pages_map.keys())[0]
                        gt_page_data = gt_pages_map[first_page_idx_gt]
                        page_identifier_for_benchmark = (
                            f"{document_id}_p{first_page_idx_gt}"
                        )
                    else:
                        _log.debug(
                            f"No valid GT segmented pages for {document_id}, using default empty page."
                        )
                except Exception as e:
                    _log.error(
                        f"Error processing GT for {document_id}: {e}, using default. Trace: {traceback.format_exc()}"
                    )

            pred_seg_pages_key = BenchMarkColumns.PREDICTED_SEGMENTED_PAGES
            if pred_seg_pages_key in data_row and data_row[pred_seg_pages_key]:
                try:
                    pred_pages_map: Optional[Dict[int, SegmentedPage]] = (
                        parse_segmented_pages(data_row[pred_seg_pages_key], document_id)
                    )
                    if pred_pages_map:
                        first_page_idx_pred: int = sorted(pred_pages_map.keys())[0]
                        pred_page_data = pred_pages_map[first_page_idx_pred]
                        if page_identifier_for_benchmark == document_id:
                            page_identifier_for_benchmark = (
                                f"{document_id}_p{first_page_idx_pred}"
                            )
                    else:
                        _log.debug(
                            f"No valid Prediction segmented pages for {document_id}, using default empty page."
                        )
                except Exception as e:
                    _log.error(
                        f"Error processing Prediction for {document_id}: {e}, using default. Trace: {traceback.format_exc()}"
                    )

            benchmark_tool.process_single_page_pair(
                ground_truth_page=gt_page_data,
                prediction_page=pred_page_data,
                image_identifier=page_identifier_for_benchmark,
            )
            processed_item_count += 1

        overall_evaluation_results = OcrDatasetEvaluationResult()
        if processed_item_count > 0:
            _log.info(f"Processed {processed_item_count} documents for OCR benchmark.")
            formatted_summary: List[Dict[str, Any]] = (
                benchmark_tool.get_formatted_metrics_summary(float_precision=1)
            )
            _log.info("\nAggregated OCR Metrics:")
            _log.info(json.dumps(formatted_summary, indent=2))

            if (
                formatted_summary
                and isinstance(formatted_summary, list)
                and len(formatted_summary) > 0
            ):
                metrics_from_summary: Dict[str, Any] = formatted_summary[0]
                if isinstance(metrics_from_summary, dict):
                    overall_evaluation_results = OcrDatasetEvaluationResult(
                        f1_score=metrics_from_summary.get("F1", 0.0),
                        recall=metrics_from_summary.get("Recall", 0.0),
                        precision=metrics_from_summary.get("Precision", 0.0),
                    )
        else:
            _log.warning("No documents were processed for the OCR benchmark.")

        _log.info(f"Final Dataset F1 Score: {overall_evaluation_results.f1_score:.4f}")
        _log.info(
            f"Final Dataset Precision: {overall_evaluation_results.precision:.4f}"
        )
        _log.info(f"Final Dataset Recall: {overall_evaluation_results.recall:.4f}")

        return overall_evaluation_results


class OCRVisualizer:
    def __init__(self) -> None:
        self._outline_thickness: int = 2
        self._ground_truth_color: str = "green"
        self._prediction_color: str = "red"
        self._correct_match_color: str = "blue"
        self._text_label_color: str = "black"
        self._visualization_subdir_name: str = "ocr_visualizations"

        self._default_font: Any = ImageFont.load_default()
        try:
            self._rendering_font: ImageFont.FreeTypeFont = ImageFont.truetype(
                "arial.ttf", size=10
            )
        except IOError:
            self._rendering_font = self._default_font  # type: ignore

    def __call__(
        self,
        dataset_path: Path,
        ocr_evaluation_report_path: Optional[Path] = None,
        output_directory: Path = Path("./visual_output"),
        data_split_name: str = "test",
    ) -> List[Path]:
        visualizations_output_path: Path = (
            output_directory / self._visualization_subdir_name
        )
        visualizations_output_path.mkdir(parents=True, exist_ok=True)

        path_to_parquet_files: str = str(dataset_path / data_split_name / "*.parquet")
        hf_dataset: Dataset = load_dataset(
            "parquet", data_files={data_split_name: path_to_parquet_files}
        )

        generated_visualization_paths: List[Path] = []
        if hf_dataset and data_split_name in hf_dataset:
            dataset_for_visualization: Dataset = hf_dataset[data_split_name]

            for i, data_row in tqdm(
                enumerate(dataset_for_visualization),
                desc="Generating OCR visualizations",
                ncols=120,
                total=len(dataset_for_visualization),
            ):
                doc_id_val: str = data_row[BenchMarkColumns.DOC_ID]
                page_images_data: Any = data_row.get(
                    BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES
                )

                ground_truth_segmented_pages: Dict[int, SegmentedPage] = {}
                prediction_segmented_pages: Dict[int, SegmentedPage] = {}

                gt_col_name: str = BenchMarkColumns.GROUNDTRUTH_SEGMENTED_PAGES
                if gt_col_name in data_row and data_row[gt_col_name]:
                    parsed_gt_pages: Optional[Dict[int, SegmentedPage]] = (
                        parse_segmented_pages(data_row[gt_col_name], doc_id_val)
                    )
                    if parsed_gt_pages:
                        ground_truth_segmented_pages = parsed_gt_pages

                pred_col_name: str = BenchMarkColumns.PREDICTED_SEGMENTED_PAGES
                if pred_col_name in data_row and data_row[pred_col_name]:
                    parsed_pred_pages: Optional[Dict[int, SegmentedPage]] = (
                        parse_segmented_pages(data_row[pred_col_name], doc_id_val)
                    )
                    if parsed_pred_pages:
                        prediction_segmented_pages = parsed_pred_pages

                image_item: Union[dict, Image.Image] = page_images_data[0]
                if isinstance(image_item, dict):
                    base_image: Image.Image = image_item["image"]
                else:
                    base_image = image_item
                if base_image.mode != "RGB":
                    base_image = base_image.convert("RGB")

                comparison_image: Image.Image = self._render_ocr_comparison_on_image(
                    doc_id_val,
                    base_image,
                    ground_truth_segmented_pages,
                    prediction_segmented_pages,
                )
                output_image_path: Path = (
                    visualizations_output_path / f"{doc_id_val}_ocr_comparison.png"
                )
                generated_visualization_paths.append(output_image_path)
                comparison_image.save(output_image_path)
        else:
            _log.warning(
                f"Dataset or split '{data_split_name}' not found. No visualizations will be generated."
            )

        return generated_visualization_paths

    def _render_ocr_comparison_on_image(
        self,
        doc_id: str,
        source_page_image: Image.Image,
        ground_truth_pages: Dict[int, SegmentedPage],
        prediction_pages: Dict[int, SegmentedPage],
    ) -> Image.Image:
        gt_image_canvas: Image.Image = copy.deepcopy(source_page_image)
        pred_image_canvas: Image.Image = copy.deepcopy(source_page_image)

        gt_draw_context: ImageDraw.ImageDraw = ImageDraw.Draw(gt_image_canvas)
        pred_draw_context: ImageDraw.ImageDraw = ImageDraw.Draw(pred_image_canvas)

        if not ground_truth_pages:
            _log.debug(
                f"No ground truth segmented pages provided for doc {doc_id} for drawing."
            )

        page_index_for_drawing: int = -1
        if ground_truth_pages:
            page_index_for_drawing = sorted(list(ground_truth_pages.keys()))[0]
        elif prediction_pages:
            page_index_for_drawing = sorted(list(prediction_pages.keys()))[0]

        gt_page_to_draw: Optional[SegmentedPage] = (
            ground_truth_pages.get(page_index_for_drawing)
            if page_index_for_drawing != -1
            else None
        )
        pred_page_to_draw: Optional[SegmentedPage] = (
            prediction_pages.get(page_index_for_drawing)
            if page_index_for_drawing != -1
            else None
        )

        page_h: float = 0.0
        page_w: float = 0.0

        if gt_page_to_draw:
            page_h = gt_page_to_draw.dimension.height
            page_w = gt_page_to_draw.dimension.width
        elif pred_page_to_draw:
            page_h = pred_page_to_draw.dimension.height
            page_w = pred_page_to_draw.dimension.width

        if page_w == 0 or page_h == 0:
            page_w = float(source_page_image.width)
            page_h = float(source_page_image.height)

        scale_factor_x: float = source_page_image.width / page_w if page_w > 0 else 1.0
        scale_factor_y: float = source_page_image.height / page_h if page_h > 0 else 1.0

        if gt_page_to_draw and gt_page_to_draw.has_words:
            for cell_item in gt_page_to_draw.word_cells:
                bbox_obj = cell_item.rect.to_bounding_box()
                if bbox_obj.coord_origin != CoordOrigin.TOPLEFT:
                    bbox_obj = bbox_obj.to_top_left_origin(page_height=page_h)

                l_coord, t_coord = round(bbox_obj.l * scale_factor_x), round(
                    bbox_obj.t * scale_factor_y
                )
                r_coord, b_coord = round(bbox_obj.r * scale_factor_x), round(
                    bbox_obj.b * scale_factor_y
                )

                gt_draw_context.rectangle(
                    [l_coord, t_coord, r_coord, b_coord],
                    outline=self._ground_truth_color,
                    width=self._outline_thickness,
                )
                text_y_pos = t_coord - 15 if t_coord > 15 else b_coord + 2
                gt_draw_context.text(
                    (l_coord, text_y_pos),
                    cell_item.text,
                    fill=self._text_label_color,
                    font=self._rendering_font,
                )

        if pred_page_to_draw and pred_page_to_draw.has_words:
            for cell_item in pred_page_to_draw.word_cells:
                bbox_obj = cell_item.rect.to_bounding_box()
                if bbox_obj.coord_origin != CoordOrigin.TOPLEFT:
                    bbox_obj = bbox_obj.to_top_left_origin(page_height=page_h)

                l_coord, t_coord = round(bbox_obj.l * scale_factor_x), round(
                    bbox_obj.t * scale_factor_y
                )
                r_coord, b_coord = round(bbox_obj.r * scale_factor_x), round(
                    bbox_obj.b * scale_factor_y
                )

                is_match_correct: bool = False
                if gt_page_to_draw and gt_page_to_draw.has_words:
                    for gt_cell_item in gt_page_to_draw.word_cells:
                        if gt_cell_item.text == cell_item.text:
                            gt_bbox_obj = gt_cell_item.rect.to_bounding_box()
                            if gt_bbox_obj.coord_origin != CoordOrigin.TOPLEFT:
                                gt_bbox_obj = gt_bbox_obj.to_top_left_origin(
                                    page_height=page_h
                                )

                            if not (
                                l_coord > round(gt_bbox_obj.r * scale_factor_x)
                                or r_coord < round(gt_bbox_obj.l * scale_factor_x)
                                or t_coord > round(gt_bbox_obj.b * scale_factor_y)
                                or b_coord < round(gt_bbox_obj.t * scale_factor_y)
                            ):
                                is_match_correct = True
                                break

                box_draw_color: str = (
                    self._correct_match_color
                    if is_match_correct
                    else self._prediction_color
                )
                pred_draw_context.rectangle(
                    [l_coord, t_coord, r_coord, b_coord],
                    outline=box_draw_color,
                    width=self._outline_thickness,
                )
                text_y_pos = t_coord - 15 if t_coord > 15 else b_coord + 2
                pred_draw_context.text(
                    (l_coord, text_y_pos),
                    cell_item.text,
                    fill=self._text_label_color,
                    font=self._rendering_font,
                )

        img_mode: str = source_page_image.mode
        img_w, img_h = source_page_image.size
        stitched_image: Image.Image = Image.new(img_mode, (2 * img_w, img_h), "white")
        stitched_image.paste(gt_image_canvas, (0, 0))
        stitched_image.paste(pred_image_canvas, (img_w, 0))

        stitched_draw_context: ImageDraw.ImageDraw = ImageDraw.Draw(stitched_image)
        header_font_size: int = max(15, int(img_h * 0.02))
        sub_header_font_size: int = max(12, int(img_h * 0.015))

        try:
            title_text_font = ImageFont.truetype("arial.ttf", size=header_font_size)
            legend_text_font = ImageFont.truetype(
                "arial.ttf", size=sub_header_font_size
            )
        except IOError:
            title_text_font = self._default_font
            legend_text_font = self._default_font

        stitched_draw_context.text(
            (10, 10), "Ground Truth OCR", fill="black", font=title_text_font
        )
        stitched_draw_context.text(
            (img_w + 10, 10), "Predicted OCR", fill="black", font=title_text_font
        )

        legend_start_y: int = header_font_size + 20
        legend_rect_dim: int = sub_header_font_size
        legend_item_gap: int = int(sub_header_font_size * 0.5)

        stitched_draw_context.rectangle(
            [
                10,
                legend_start_y,
                10 + legend_rect_dim,
                legend_start_y + legend_rect_dim,
            ],
            outline=self._ground_truth_color,
            fill=self._ground_truth_color,
        )
        stitched_draw_context.text(
            (15 + legend_rect_dim, legend_start_y),
            "Ground Truth Word",
            fill="black",
            font=legend_text_font,
        )

        current_pred_legend_y = legend_start_y
        stitched_draw_context.rectangle(
            [
                img_w + 10,
                current_pred_legend_y,
                img_w + 10 + legend_rect_dim,
                current_pred_legend_y + legend_rect_dim,
            ],
            outline=self._correct_match_color,
            fill=self._correct_match_color,
        )
        stitched_draw_context.text(
            (img_w + 15 + legend_rect_dim, current_pred_legend_y),
            "Correct Prediction",
            fill="black",
            font=legend_text_font,
        )

        current_pred_legend_y += legend_rect_dim + legend_item_gap
        stitched_draw_context.rectangle(
            [
                img_w + 10,
                current_pred_legend_y,
                img_w + 10 + legend_rect_dim,
                current_pred_legend_y + legend_rect_dim,
            ],
            outline=self._prediction_color,
            fill=self._prediction_color,
        )
        stitched_draw_context.text(
            (img_w + 15 + legend_rect_dim, current_pred_legend_y),
            "Incorrect Prediction",
            fill="black",
            font=legend_text_font,
        )

        return stitched_image
