import copy
import glob
import io
import json
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from datasets import Dataset, load_dataset
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, PageGeometry, SegmentedPage
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from tqdm import tqdm

from docling_eval.datamodels.dataset_record import DatasetRecordWithPrediction
from docling_eval.datamodels.types import BenchMarkColumns, PredictionFormats
from docling_eval.evaluators.base_evaluator import BaseEvaluator
from docling_eval.evaluators.ocr.benchmark_runner import _OcrBenchmark
from docling_eval.evaluators.ocr.evaluation_models import (
    DocumentEvaluationMetadata,
    OcrDatasetEvaluationResult,
    TextCellUnit,
    TruePositiveMatch,
    Word,
    WordEvaluationMetadata,
    _CalculationConstants,
)
from docling_eval.evaluators.ocr.geometry_utils import box_to_key
from docling_eval.evaluators.ocr.performance_calculator import _OcrPerformanceCalculator
from docling_eval.evaluators.ocr.processing_utils import (
    calculate_edit_distance,
    parse_segmented_pages,
)

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
        text_unit: TextCellUnit = TextCellUnit.WORD,
    ) -> None:
        super().__init__(
            intermediate_evaluations_path=intermediate_evaluations_path,
            prediction_sources=prediction_sources,
            supported_prediction_formats=[PredictionFormats.DOCLING_DOCUMENT],
        )
        self.intermediate_evaluations_path = intermediate_evaluations_path
        self.text_unit: TextCellUnit = text_unit

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
            aggregation_mode="union",
            text_unit=self.text_unit,
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

        if self.intermediate_evaluations_path is not None:
            evaluation_metadata_output_path = (
                self.intermediate_evaluations_path / "evaluation_metadata"
            )
            evaluation_metadata_output_path.mkdir(parents=True, exist_ok=True)

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
            if (
                page_identifier_for_benchmark
                in benchmark_tool.image_to_performance_calculator_map
            ):
                perf_calculator = benchmark_tool.image_to_performance_calculator_map[
                    page_identifier_for_benchmark
                ]
                metadata = self._create_evaluation_metadata(
                    document_id, perf_calculator
                )
                metadata_path = (
                    evaluation_metadata_output_path / f"{document_id}_metadata.json"
                )
                with open(metadata_path, "w") as f:
                    f.write(metadata.model_dump_json(indent=2))
            processed_item_count += 1

        overall_evaluation_results = OcrDatasetEvaluationResult()
        if processed_item_count > 0:
            _log.info(f"Processed {processed_item_count} documents for OCR benchmark.")
            formatted_summary: List[Dict[str, Any]] = (
                benchmark_tool.get_formatted_metrics_summary(float_precision=2)
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
                        f1_score=metrics_from_summary.get("f1", 0.0),
                        recall=metrics_from_summary.get("recall", 0.0),
                        precision=metrics_from_summary.get("precision", 0.0),
                        word_accuracy_sensitive=metrics_from_summary.get(
                            "word_accuracy_sensitive", 0.0
                        ),
                        word_accuracy_insensitive=metrics_from_summary.get(
                            "word_accuracy_insensitive", 0.0
                        ),
                        character_accuracy_sensitive=metrics_from_summary.get(
                            "character_accuracy_sensitive", 0.0
                        ),
                        character_accuracy_insensitive=metrics_from_summary.get(
                            "character_accuracy_insensitive", 0.0
                        ),
                    )
        else:
            _log.warning("No documents were processed for the OCR benchmark.")

        _log.info(f"Final Dataset F1 Score: {overall_evaluation_results.f1_score:.4f}")
        _log.info(
            f"Final Dataset Precision: {overall_evaluation_results.precision:.4f}"
        )
        _log.info(f"Final Dataset Recall: {overall_evaluation_results.recall:.4f}")

        return overall_evaluation_results

    def _create_evaluation_metadata(
        self, doc_id: str, perf_calculator: _OcrPerformanceCalculator
    ) -> DocumentEvaluationMetadata:
        def word_to_metadata(
            word: Word,
            is_tp: bool = False,
            is_fp: bool = False,
            is_fn: bool = False,
            other_word: Optional[Word] = None,
        ) -> WordEvaluationMetadata:
            edit_dist_sensitive = None
            edit_dist_insensitive = None

            if other_word is not None:
                edit_dist_sensitive = calculate_edit_distance(
                    word.text, other_word.text, None
                )
                edit_dist_insensitive = calculate_edit_distance(
                    word.text.upper(),
                    other_word.text.upper(),
                    None,
                )

            return WordEvaluationMetadata(
                text=word.text,
                confidence=word.confidence,
                bounding_box=BoundingBox.model_validate(word.bbox.model_dump()),
                is_true_positive=is_tp,
                is_false_positive=is_fp,
                is_false_negative=is_fn,
                edit_distance_sensitive=edit_dist_sensitive,
                edit_distance_insensitive=edit_dist_insensitive,
            )

        true_positives = []
        for gt_word, pred_word in perf_calculator.confirmed_gt_prediction_matches:
            ed_sensitive = calculate_edit_distance(
                gt_word.text, pred_word.text, _CalculationConstants.CHAR_NORMALIZE_MAP
            )
            is_perfect_match = ed_sensitive == 0

            gt_meta = word_to_metadata(
                gt_word, is_tp=is_perfect_match, other_word=pred_word
            )
            pred_meta = word_to_metadata(
                pred_word, is_tp=is_perfect_match, other_word=gt_word
            )

            tp_match = TruePositiveMatch(pred=pred_meta, gt=gt_meta)
            true_positives.append(tp_match)

        false_positives = [
            word_to_metadata(word, is_fp=True)
            for word in perf_calculator.current_false_positives
        ]

        false_negatives = [
            word_to_metadata(word, is_fn=True)
            for word in perf_calculator.current_false_negatives
        ]

        metrics = perf_calculator.calculate_image_metrics()

        return DocumentEvaluationMetadata(
            doc_id=doc_id,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            metrics=metrics,
        )


class OCRVisualizer:
    def __init__(self) -> None:
        self._outline_thickness: int = 2
        self._ground_truth_color: str = "green"
        self._prediction_color: str = "red"
        self._correct_match_color: str = "blue"
        self._text_label_color: str = "black"
        self._visualization_subdir_name: str = "ocr_visualizations"
        self._error_reports_subdir_name: str = "ocr_error_reports"

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
        visualizations_output_path = output_directory / self._visualization_subdir_name
        visualizations_output_path.mkdir(parents=True, exist_ok=True)

        metadata_path = output_directory / "evaluation_metadata"
        if not metadata_path.exists():
            _log.warning(
                f"No evaluation metadata found at {metadata_path}. Please run evaluation first."
            )
            raise FileNotFoundError(
                f"No evaluation metadata found at {metadata_path}. Please run evaluation first."
            )
            return []

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

                metadata_file = metadata_path / f"{doc_id_val}_metadata.json"
                if not metadata_file.exists():
                    _log.warning(f"No metadata found for document {doc_id_val}")
                    continue

                try:
                    with open(metadata_file, "r") as f:
                        metadata = DocumentEvaluationMetadata.model_validate_json(
                            f.read()
                        )
                except Exception as e:
                    _log.error(f"Failed to load metadata for {doc_id_val}: {e}")
                    continue

                page_images_data: Any = data_row.get(
                    BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES
                )

                if not page_images_data or len(page_images_data) == 0:
                    _log.warning(f"No page images found for document {doc_id_val}")
                    continue

                image_item: Union[dict, Image.Image] = page_images_data[0]
                base_image: Image.Image

                try:
                    if isinstance(image_item, dict):
                        if "image" in image_item and isinstance(
                            image_item["image"], Image.Image
                        ):
                            base_image = image_item["image"]
                        elif "bytes" in image_item and image_item["bytes"]:
                            base_image = Image.open(io.BytesIO(image_item["bytes"]))
                        elif "path" in image_item and image_item["path"]:
                            base_image = Image.open(image_item["path"])
                        else:
                            raise ValueError(
                                f"Unsupported image_item format: {image_item}"
                            )
                    else:
                        base_image = image_item

                    if base_image.mode != "RGB":
                        base_image = base_image.convert("RGB")
                except Exception as e:
                    _log.error(f"Failed to load image for {doc_id_val}: {e}")
                    continue

                gt_seg_pages: Dict[int, SegmentedPage] = {}
                pred_seg_pages: Dict[int, SegmentedPage] = {}

                gt_col_name: str = BenchMarkColumns.GROUNDTRUTH_SEGMENTED_PAGES
                if gt_col_name in data_row and data_row[gt_col_name]:
                    parsed_gt_pages = parse_segmented_pages(
                        data_row[gt_col_name], doc_id_val
                    )
                    if parsed_gt_pages:
                        gt_seg_pages = parsed_gt_pages

                pred_col_name: str = BenchMarkColumns.PREDICTED_SEGMENTED_PAGES
                if pred_col_name in data_row and data_row[pred_col_name]:
                    parsed_pred_pages = parse_segmented_pages(
                        data_row[pred_col_name], doc_id_val
                    )
                    if parsed_pred_pages:
                        pred_seg_pages = parsed_pred_pages

                page_idx = -1
                if gt_seg_pages:
                    page_idx = sorted(list(gt_seg_pages.keys()))[0]
                elif pred_seg_pages:
                    page_idx = sorted(list(pred_seg_pages.keys()))[0]

                gt_page = gt_seg_pages.get(page_idx)
                pred_page = pred_seg_pages.get(page_idx)

                page_h, page_w = 0.0, 0.0
                if gt_page:
                    page_h, page_w = gt_page.dimension.height, gt_page.dimension.width
                elif pred_page:
                    page_h, page_w = (
                        pred_page.dimension.height,
                        pred_page.dimension.width,
                    )

                if page_w == 0 or page_h == 0:
                    page_w, page_h = float(base_image.width), float(base_image.height)

                comparison_image: Image.Image = self._render_comparison_from_metadata(
                    base_image, metadata, page_w, page_h
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

    def _render_comparison_from_metadata(
        self,
        source_page_image: Image.Image,
        metadata: DocumentEvaluationMetadata,
        page_w: float,
        page_h: float,
    ) -> Image.Image:
        gt_image_canvas = copy.deepcopy(source_page_image)
        pred_image_canvas = copy.deepcopy(source_page_image)

        gt_draw_context = ImageDraw.Draw(gt_image_canvas)
        pred_draw_context = ImageDraw.Draw(pred_image_canvas)

        scale_factor_x = source_page_image.width / page_w if page_w > 0 else 1.0
        scale_factor_y = source_page_image.height / page_h if page_h > 0 else 1.0

        # draw ground truth words (TPs + FNs)
        all_gt_words = []
        for tp_match in metadata.true_positives:
            all_gt_words.append(tp_match.gt)
        all_gt_words.extend(metadata.false_negatives)

        for word_meta in all_gt_words:
            bbox_obj = word_meta.bounding_box
            if bbox_obj.coord_origin != CoordOrigin.TOPLEFT:
                bbox_obj = bbox_obj.to_top_left_origin(page_height=page_h)
            l, t = (
                round(bbox_obj.l * scale_factor_x),
                round(bbox_obj.t * scale_factor_y),
            )
            r, b = (
                round(bbox_obj.r * scale_factor_x),
                round(bbox_obj.b * scale_factor_y),
            )
            gt_draw_context.rectangle(
                [l, t, r, b],
                outline=self._ground_truth_color,
                width=self._outline_thickness,
            )
            text_y = t - 15 if t > 15 else b + 2
            gt_draw_context.text(
                (l, text_y),
                word_meta.text,
                fill=self._text_label_color,
                font=self._rendering_font,
            )

        correct_tp_pairs: List[TruePositiveMatch] = []
        incorrect_tp_pairs: List[TruePositiveMatch] = []
        for tp_match in metadata.true_positives:
            pred_meta = tp_match.pred
            ed = pred_meta.edit_distance_sensitive
            if ed is not None and ed == 0:
                correct_tp_pairs.append(tp_match)
            else:
                incorrect_tp_pairs.append(tp_match)

        # draw correct predictions (exact text match)
        for tp in correct_tp_pairs:
            pred_meta = tp.pred
            bbox_obj = pred_meta.bounding_box
            if bbox_obj.coord_origin != CoordOrigin.TOPLEFT:
                bbox_obj = bbox_obj.to_top_left_origin(page_height=page_h)
            l, t = (
                round(bbox_obj.l * scale_factor_x),
                round(bbox_obj.t * scale_factor_y),
            )
            r, b = (
                round(bbox_obj.r * scale_factor_x),
                round(bbox_obj.b * scale_factor_y),
            )
            pred_draw_context.rectangle(
                [l, t, r, b],
                outline=self._correct_match_color,
                width=self._outline_thickness,
            )

        # draw incorrect predictions (TPs with text mismatch) and label "pred | gt"
        for tp in incorrect_tp_pairs:
            pred_meta = tp.pred
            gt_meta = tp.gt
            bbox_obj = pred_meta.bounding_box
            if bbox_obj.coord_origin != CoordOrigin.TOPLEFT:
                bbox_obj = bbox_obj.to_top_left_origin(page_height=page_h)
            l, t = (
                round(bbox_obj.l * scale_factor_x),
                round(bbox_obj.t * scale_factor_y),
            )
            r, b = (
                round(bbox_obj.r * scale_factor_x),
                round(bbox_obj.b * scale_factor_y),
            )
            pred_draw_context.rectangle(
                [l, t, r, b],
                outline=self._prediction_color,
                width=self._outline_thickness,
            )
            # label with "pred | gt"
            text_y = t - 15 if t > 15 else b + 2
            # label_text = f"{pred_meta.text} | {gt_meta.text}"
            label_text = f"{pred_meta.text}"
            pred_draw_context.text(
                (l, text_y),
                label_text,
                fill=self._text_label_color,
                font=self._rendering_font,
            )

        # Draw incorrect predictions (FPs)
        all_incorrect_preds = metadata.false_positives.copy()

        for word_meta in all_incorrect_preds:
            bbox_obj = word_meta.bounding_box
            if bbox_obj.coord_origin != CoordOrigin.TOPLEFT:
                bbox_obj = bbox_obj.to_top_left_origin(page_height=page_h)
            l, t = (
                round(bbox_obj.l * scale_factor_x),
                round(bbox_obj.t * scale_factor_y),
            )
            r, b = (
                round(bbox_obj.r * scale_factor_x),
                round(bbox_obj.b * scale_factor_y),
            )
            pred_draw_context.rectangle(
                [l, t, r, b],
                outline=self._prediction_color,
                width=self._outline_thickness,
            )
            text_y = t - 15 if t > 15 else b + 2
            pred_draw_context.text(
                (l, text_y),
                word_meta.text,
                fill=self._text_label_color,
                font=self._rendering_font,
            )

        img_w, img_h = source_page_image.size
        img_mode = source_page_image.mode

        header_font_size = max(15, int(img_h * 0.02))
        sub_header_font_size = max(12, int(img_h * 0.015))

        try:
            title_text_font = ImageFont.truetype("arial.ttf", size=header_font_size)
            legend_text_font = ImageFont.truetype(
                "arial.ttf", size=sub_header_font_size
            )
        except IOError:
            title_text_font = self._default_font
            legend_text_font = self._default_font

        line_height = sub_header_font_size + 8
        footer_padding = 20
        footer_height = (7 * line_height) + footer_padding

        final_img_h = img_h + footer_height
        stitched_image = Image.new(img_mode, (2 * img_w, final_img_h), "white")

        stitched_image.paste(gt_image_canvas, (0, 0))
        stitched_image.paste(pred_image_canvas, (img_w, 0))

        stitched_draw_context = ImageDraw.Draw(stitched_image)

        stitched_draw_context.text(
            (10, 10), "Ground Truth OCR", fill="black", font=title_text_font
        )
        stitched_draw_context.text(
            (img_w + 10, 10), "Predicted OCR", fill="black", font=title_text_font
        )

        metrics_summary = metadata.metrics

        stats_texts = {
            "TP (True Positives)": f"{metrics_summary.number_of_true_positive_matches}",
            "FP (False Positives)": f"{metrics_summary.number_of_false_positive_detections}",
            "FN (False Negatives)": f"{metrics_summary.number_of_false_negative_detections}",
            "---": "---",
            "Precision": f"{metrics_summary.detection_precision:.2f}% (TP / (TP + FP))",
            "Recall": f"{metrics_summary.detection_recall:.2f}% (TP / (TP + FN))",
            "F1-Score": f"{metrics_summary.detection_f1:.2f}% (2*P*R / (P+R))",
        }

        stats_start_x = 15
        stats_start_y = img_h + (footer_padding // 2)

        for i, (key, val) in enumerate(stats_texts.items()):
            y_pos = stats_start_y + (i * line_height)
            if key != "---":
                stitched_draw_context.text(
                    (stats_start_x, y_pos),
                    f"{key}: {val}",
                    fill=self._text_label_color,
                    font=legend_text_font,
                )
            else:
                stitched_draw_context.line(
                    [
                        (stats_start_x, y_pos + line_height // 2),
                        (img_w * 2 - 20, y_pos + line_height // 2),
                    ],
                    fill="lightgray",
                    width=1,
                )

        legend_start_x = img_w + 15
        legend_start_y = img_h + (footer_padding // 2)
        legend_rect_dim = sub_header_font_size

        legend_items = {
            "Ground Truth Word": self._ground_truth_color,
            "Correct Prediction": self._correct_match_color,
            "Incorrect Prediction": self._prediction_color,
        }

        for i, (label, color) in enumerate(legend_items.items()):
            y_pos = legend_start_y + (i * line_height)
            stitched_draw_context.rectangle(
                [
                    legend_start_x,
                    y_pos,
                    legend_start_x + legend_rect_dim,
                    y_pos + legend_rect_dim,
                ],
                outline=color,
                fill=color,
            )
            stitched_draw_context.text(
                (legend_start_x + legend_rect_dim + 10, y_pos),
                label,
                fill="black",
                font=legend_text_font,
            )

        return stitched_image
