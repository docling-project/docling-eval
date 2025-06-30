import glob
import logging
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset, load_dataset
from docling_core.types.doc.document import (
    DEFAULT_EXPORT_LABELS,
    ContentLayer,
    DocItem,
    DoclingDocument,
)
from docling_core.types.doc.labels import DocItemLabel
from pydantic import BaseModel
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm  # type: ignore

from docling_eval.datamodels.dataset_record import DatasetRecordWithPrediction
from docling_eval.datamodels.types import BenchMarkColumns, PredictionFormats
from docling_eval.evaluators.base_evaluator import (
    BaseEvaluator,
    DatasetEvaluation,
    EvaluationRejectionType,
    UnitEvaluation,
    docling_document_from_doctags,
)
from docling_eval.evaluators.stats import DatasetStatistics, compute_stats
from docling_eval.utils.utils import tensor_to_float

_log = logging.getLogger(__name__)


class MissingPredictionStrategy(Enum):
    """Strategy for handling missing predictions."""

    PENALIZE = "penalize"  # Treat missing predictions as zero score
    IGNORE = "ignore"  # Skip the GT-Pred pair entirely


class LabelFilteringStrategy(Enum):
    """Strategy for determining which labels to evaluate."""

    INTERSECTION = (
        "intersection"  # Only evaluate labels present in both GT and predictions
    )
    UNION = "union"  # Evaluate all labels present in the label mapping


class ClassLayoutEvaluation(BaseModel):
    r"""
    Class based layout evaluation
    """

    name: str
    label: str
    value: float  # AP[0.5:0.05:0.95]


class ImageLayoutEvaluation(UnitEvaluation):
    r"""
    Image based layout evaluation
    """

    name: str
    value: float  # Area weighted average IoU for label-matched GT/pred bboxes for IoU thres = 0.5

    map_val: float  # AP at IoU thres=0.50
    map_50: float  # AP at IoU thres=0.50
    map_75: float  # AP at IoU thres=0.75

    # Weighted average IoU for the page bboxes with matching labels (between GT and pred)
    # The weight is the bbox size and each measurement corresponds to a different IoU threshold
    avg_weighted_label_matched_iou_50: float
    avg_weighted_label_matched_iou_75: float
    avg_weighted_label_matched_iou_90: float
    avg_weighted_label_matched_iou_95: float

    segmentation_precision: float
    segmentation_recall: float
    segmentation_f1: float

    # Area-level metrics excluding PICTURE labels
    segmentation_precision_no_pictures: Optional[float] = None
    segmentation_recall_no_pictures: Optional[float] = None
    segmentation_f1_no_pictures: Optional[float] = None


class DatasetLayoutEvaluation(DatasetEvaluation):
    true_labels: Dict[str, int]
    pred_labels: Dict[str, int]
    mAP: float  # The mean AP[0.5:0.05:0.95] across all classes

    intersecting_labels: List[str]
    evaluations_per_class: List[ClassLayoutEvaluation]
    evaluations_per_image: List[ImageLayoutEvaluation]

    # Statistics
    map_stats: DatasetStatistics  # Stats for the mAP[0.5:0.05:0.95] across all images
    map_50_stats: DatasetStatistics
    map_75_stats: DatasetStatistics
    weighted_map_50_stats: DatasetStatistics
    weighted_map_75_stats: DatasetStatistics
    weighted_map_90_stats: DatasetStatistics
    weighted_map_95_stats: DatasetStatistics

    segmentation_precision_stats: DatasetStatistics
    segmentation_recall_stats: DatasetStatistics
    segmentation_f1_stats: DatasetStatistics

    # Statistics for metrics excluding PICTURE labels
    segmentation_precision_no_pictures_stats: Optional[DatasetStatistics] = None
    segmentation_recall_no_pictures_stats: Optional[DatasetStatistics] = None
    segmentation_f1_no_pictures_stats: Optional[DatasetStatistics] = None

    def to_table(self) -> Tuple[List[List[str]], List[str]]:
        headers = ["label", "Class mAP[0.5:0.95]"]

        self.evaluations_per_class = sorted(
            self.evaluations_per_class, key=lambda x: x.value, reverse=True
        )

        table = []
        for i in range(len(self.evaluations_per_class)):
            table.append(
                [
                    f"{self.evaluations_per_class[i].label}",
                    f"{100.0*self.evaluations_per_class[i].value:.2f}",
                ]
            )

        return table, headers


class LayoutEvaluator(BaseEvaluator):
    def __init__(
        self,
        label_mapping: Optional[Dict[DocItemLabel, Optional[DocItemLabel]]] = None,
        intermediate_evaluations_path: Optional[Path] = None,
        prediction_sources: List[PredictionFormats] = [],
        missing_prediction_strategy: MissingPredictionStrategy = MissingPredictionStrategy.PENALIZE,
        label_filtering_strategy: LabelFilteringStrategy = LabelFilteringStrategy.INTERSECTION,
    ):
        supported_prediction_formats: List[PredictionFormats] = [
            PredictionFormats.DOCLING_DOCUMENT,
            PredictionFormats.DOCTAGS,
            PredictionFormats.JSON,
            PredictionFormats.YAML,
        ]
        if not prediction_sources:
            prediction_sources = supported_prediction_formats
        super().__init__(
            intermediate_evaluations_path=intermediate_evaluations_path,
            prediction_sources=prediction_sources,
            supported_prediction_formats=supported_prediction_formats,
        )

        self.filter_labels = []
        self.label_names = {}
        self.label_mapping = label_mapping or {v: v for v in DocItemLabel}
        self.missing_prediction_strategy = missing_prediction_strategy
        self.label_filtering_strategy = label_filtering_strategy

        for i, _ in enumerate(DEFAULT_EXPORT_LABELS):
            self.filter_labels.append(_)
            self.label_names[i] = _

    def __call__(
        self,
        ds_path: Path,
        split: str = "test",
    ) -> DatasetLayoutEvaluation:
        logging.info("Loading the split '%s' from: '%s'", split, ds_path)

        # Load the dataset
        split_path = str(ds_path / split / "*.parquet")
        split_files = glob.glob(split_path)
        logging.info("#-files: %s", len(split_files))
        ds = load_dataset("parquet", data_files={split: split_files})
        logging.info("Overview of dataset: %s", ds)

        # Select the split
        ds_selection: Dataset = ds[split]

        (
            true_labels,
            pred_labels,
            intersection_labels,
            union_labels,
        ) = self._find_intersecting_labels(ds_selection)
        true_labels_str = ", ".join(sorted(true_labels))
        logging.info(f"True labels: {true_labels_str}")

        pred_labels_str = ", ".join(sorted(pred_labels))
        logging.info(f"Pred labels: {pred_labels_str}")

        intersection_labels_str = ", ".join(sorted(intersection_labels))
        logging.info(f"Intersection labels: {intersection_labels_str}")

        union_labels_str = ", ".join(sorted(union_labels))
        logging.info(f"Union labels: {union_labels_str}")

        logging.info(
            f"Using missing prediction strategy: {self.missing_prediction_strategy.value}"
        )
        logging.info(
            f"Using label filtering strategy: {self.label_filtering_strategy.value}"
        )

        # Determine which labels to use for evaluation based on strategy
        if self.label_filtering_strategy == LabelFilteringStrategy.INTERSECTION:
            filter_labels = intersection_labels
        elif self.label_filtering_strategy == LabelFilteringStrategy.UNION:
            # Use all labels from the mapping that have non-None values
            filter_labels = [
                DocItemLabel(mapped_label)
                for mapped_label in set(self.label_mapping.values())
                if mapped_label is not None
            ]
        else:
            raise ValueError(
                f"Unknown label filtering strategy: {self.label_filtering_strategy}"
            )

        filter_labels_str = ", ".join(sorted([label.value for label in filter_labels]))
        logging.info(f"Filter labels for evaluation: {filter_labels_str}")

        doc_ids = []
        ground_truths = []
        predictions = []
        rejected_samples: Dict[EvaluationRejectionType, int] = {
            EvaluationRejectionType.INVALID_CONVERSION_STATUS: 0,
            EvaluationRejectionType.MISSING_PREDICTION: 0,
            EvaluationRejectionType.MISMATHCED_DOCUMENT: 0,
        }

        for i, data in tqdm(
            enumerate(ds_selection),
            desc="Layout evaluations",
            ncols=120,
            total=len(ds_selection),
        ):
            data_record = DatasetRecordWithPrediction.model_validate(data)
            doc_id = data_record.doc_id
            if data_record.status not in self._accepted_status:
                _log.error(
                    "Skipping record without successfull conversion status: %s", doc_id
                )
                rejected_samples[EvaluationRejectionType.INVALID_CONVERSION_STATUS] += 1
                continue

            true_doc = data_record.ground_truth_doc
            pred_doc = self._get_pred_doc(data_record)
            if not pred_doc:
                _log.error("There is no prediction for doc_id=%s", doc_id)
                rejected_samples[EvaluationRejectionType.MISSING_PREDICTION] += 1
                continue

            gts, preds = self._extract_layout_data(
                true_doc=true_doc,
                pred_doc=pred_doc,
                filter_labels=filter_labels,
            )

            # Track mismatched documents when using PENALIZE strategy and there are missing pages
            true_pages = set()
            for item, level in true_doc.iterate_items(
                included_content_layers={c for c in ContentLayer},
                traverse_pictures=True,
            ):
                if (
                    isinstance(item, DocItem)
                    and self.label_mapping[item.label] in filter_labels
                ):
                    for prov in item.prov:
                        true_pages.add(prov.page_no)

            pred_pages = set()
            for item, level in pred_doc.iterate_items(
                included_content_layers={c for c in ContentLayer},
                traverse_pictures=True,
            ):
                if (
                    isinstance(item, DocItem)
                    and self.label_mapping[item.label] in filter_labels
                ):
                    for prov in item.prov:
                        pred_pages.add(prov.page_no)

            if (
                self.missing_prediction_strategy == MissingPredictionStrategy.PENALIZE
                and len(true_pages - pred_pages) > 0
            ):
                rejected_samples[EvaluationRejectionType.MISMATHCED_DOCUMENT] += 1

            # logging.info(f"gts: {gts}")
            # logging.info(f"preds: {preds}")

            # The new _extract_layout_data method ensures proper alignment
            # gts and preds are guaranteed to have the same length and corresponding indices
            if len(gts) > 0:
                for i, (page_no, _) in enumerate(gts):
                    doc_ids.append(data[BenchMarkColumns.DOC_ID] + f"-page-{page_no}")

                # Extract the tensor dictionaries from tuples
                gt_tensors = [tensor_dict for _, tensor_dict in gts]
                pred_tensors = [tensor_dict for _, tensor_dict in preds]

                ground_truths.extend(gt_tensors)
                predictions.extend(pred_tensors)

        # Note: We no longer need to check for mismatched documents since
        # _extract_layout_data ensures proper alignment based on missing_prediction_strategy

        assert len(doc_ids) == len(ground_truths), "doc_ids==len(ground_truths)"
        assert len(doc_ids) == len(predictions), "doc_ids==len(predictions)"

        # Initialize metric for the bboxes of the entire document
        metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

        # Update metric with predictions and ground truths
        metric.update(predictions, ground_truths)

        # Compute mAP and other metrics per class
        result = metric.compute()

        evaluations_per_class: List[ClassLayoutEvaluation] = []

        total_mAP = result["map"]
        if "map_per_class" in result:
            for label_idx, class_map in enumerate(result["map_per_class"]):
                label = filter_labels[label_idx].value
                evaluations_per_class.append(
                    ClassLayoutEvaluation(
                        name="Class AP[0.5:0.95]",
                        label=label,
                        value=class_map,
                    )
                )

        # Compute mAP for each image individually
        map_values = []
        map_50_values = []
        map_75_values = []
        weighted_map_50_values = []
        weighted_map_75_values = []
        weighted_map_90_values = []
        weighted_map_95_values = []

        evaluations_per_image: List[ImageLayoutEvaluation] = []
        for i, (doc_id, pred, gt) in enumerate(
            zip(doc_ids, predictions, ground_truths)
        ):
            # logging.info(f"gt: {gt}")
            # logging.info(f"pred: {pred}")

            precision, recall, f1 = self._compute_area_level_metrics_for_tensors(
                gt_boxes=gt["boxes"],
                pred_boxes=pred["boxes"],
                page_width=100,
                page_height=100,
                mask_width=512,
                mask_height=512,
            )

            # Compute metrics excluding PICTURE labels
            precision_no_pics, recall_no_pics, f1_no_pics = (
                self._compute_area_level_metrics_excluding_pictures(
                    gt_boxes=gt["boxes"],
                    gt_labels=gt["labels"],
                    pred_boxes=pred["boxes"],
                    pred_labels=pred["labels"],
                    filter_labels=filter_labels,
                    page_width=100,
                    page_height=100,
                    mask_width=512,
                    mask_height=512,
                )
            )

            # Reset the metric for the next image
            metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

            # Update with single image - these are already tensor-only dicts
            metric.update([pred], [gt])

            # Compute metrics
            result = metric.compute()

            # Extract mAP for this image
            map_value = tensor_to_float(result["map_50"])
            map_50 = tensor_to_float(result["map_50"])
            map_75 = tensor_to_float(result["map_75"])

            result = self._compute_average_iou_with_labels_across_iou(
                pred_boxes=pred["boxes"],
                pred_labels=pred["labels"],
                gt_boxes=gt["boxes"],
                gt_labels=gt["labels"],
            )
            average_iou_50 = tensor_to_float(result["average_iou_50"])
            average_iou_75 = tensor_to_float(result["average_iou_75"])
            average_iou_90 = tensor_to_float(result["average_iou_90"])
            average_iou_95 = tensor_to_float(result["average_iou_95"])

            # Set the stats
            map_values.append(map_value)
            map_50_values.append(map_50)
            map_75_values.append(map_75)
            weighted_map_50_values.append(average_iou_50)
            weighted_map_75_values.append(average_iou_75)
            weighted_map_90_values.append(average_iou_90)
            weighted_map_95_values.append(average_iou_95)

            logging.info(
                f"doc: {doc_id}\tprecision: {precision:.2f}, recall: {recall:.2f}, f1: {f1:.2f}, map_50: {map_50:.2f}, "
                f"precision_no_pics: {precision_no_pics:.2f}, recall_no_pics: {recall_no_pics:.2f}, f1_no_pics: {f1_no_pics:.2f}"
            )

            image_evaluation = ImageLayoutEvaluation(
                name=doc_id,
                value=average_iou_50,
                map_val=map_value,
                map_50=map_50,
                map_75=map_75,
                avg_weighted_label_matched_iou_50=average_iou_50,
                avg_weighted_label_matched_iou_75=average_iou_75,
                avg_weighted_label_matched_iou_90=average_iou_90,
                avg_weighted_label_matched_iou_95=average_iou_95,
                segmentation_precision=precision,
                segmentation_recall=recall,
                segmentation_f1=f1,
                segmentation_precision_no_pictures=precision_no_pics,
                segmentation_recall_no_pictures=recall_no_pics,
                segmentation_f1_no_pictures=f1_no_pics,
            )
            evaluations_per_image.append(image_evaluation)
            if self._intermediate_evaluations_path:
                self.save_intermediate_evaluations(
                    "Layout_image", i, doc_id, evaluations_per_image
                )

        evaluations_per_class = sorted(evaluations_per_class, key=lambda x: -x.value)
        evaluations_per_image = sorted(evaluations_per_image, key=lambda x: -x.value)

        dataset_layout_evaluation = DatasetLayoutEvaluation(
            evaluated_samples=len(evaluations_per_image),
            rejected_samples=rejected_samples,
            mAP=total_mAP,
            evaluations_per_class=evaluations_per_class,
            evaluations_per_image=evaluations_per_image,
            map_stats=compute_stats(map_values),
            map_50_stats=compute_stats(map_50_values),
            map_75_stats=compute_stats(map_75_values),
            weighted_map_50_stats=compute_stats(weighted_map_50_values),
            weighted_map_75_stats=compute_stats(weighted_map_75_values),
            weighted_map_90_stats=compute_stats(weighted_map_90_values),
            weighted_map_95_stats=compute_stats(weighted_map_95_values),
            segmentation_precision_stats=compute_stats(
                [_.segmentation_precision for _ in evaluations_per_image]
            ),
            segmentation_recall_stats=compute_stats(
                [_.segmentation_recall for _ in evaluations_per_image]
            ),
            segmentation_f1_stats=compute_stats(
                [_.segmentation_f1 for _ in evaluations_per_image]
            ),
            segmentation_precision_no_pictures_stats=compute_stats(
                [
                    _.segmentation_precision_no_pictures
                    for _ in evaluations_per_image
                    if _.segmentation_precision_no_pictures is not None
                ]
            ),
            segmentation_recall_no_pictures_stats=compute_stats(
                [
                    _.segmentation_recall_no_pictures
                    for _ in evaluations_per_image
                    if _.segmentation_recall_no_pictures is not None
                ]
            ),
            segmentation_f1_no_pictures_stats=compute_stats(
                [
                    _.segmentation_f1_no_pictures
                    for _ in evaluations_per_image
                    if _.segmentation_f1_no_pictures is not None
                ]
            ),
            true_labels=true_labels,
            pred_labels=pred_labels,
            intersecting_labels=[_.value for _ in filter_labels],
        )
        return dataset_layout_evaluation

    def _get_pred_doc(
        self, data_record: DatasetRecordWithPrediction
    ) -> Optional[DoclingDocument]:
        r"""
        Get the predicted DoclingDocument
        """
        pred_doc = None
        for prediction_format in self._prediction_sources:
            if prediction_format == PredictionFormats.DOCLING_DOCUMENT:
                pred_doc = data_record.predicted_doc
            elif prediction_format == PredictionFormats.JSON:
                if data_record.original_prediction:
                    pred_doc = DoclingDocument.load_from_json(
                        data_record.original_prediction
                    )
            elif prediction_format == PredictionFormats.YAML:
                if data_record.original_prediction:
                    pred_doc = DoclingDocument.load_from_yaml(
                        data_record.original_prediction
                    )
            elif prediction_format == PredictionFormats.DOCTAGS:
                pred_doc = docling_document_from_doctags(data_record)
            if pred_doc is not None:
                break

        return pred_doc

    def _compute_iou(self, box1, box2):
        """Compute IoU between two bounding boxes."""
        x1 = torch.max(box1[0], box2[0])
        y1 = torch.max(box1[1], box2[1])
        x2 = torch.min(box1[2], box2[2])
        y2 = torch.min(box1[3], box2[3])

        intersection = torch.max(x2 - x1, torch.tensor(0.0)) * torch.max(
            y2 - y1, torch.tensor(0.0)
        )
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0

    def _compute_average_iou_with_labels(
        self, pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thresh=0.5
    ):
        """
        Compute the average IoU for label-matched detections and weight by bbox area.

        FIXED: Now ensures mathematical consistency between weights and IoU values.
        Only matched predictions contribute to both numerator and denominator.

        Args:
            pred_boxes (torch.Tensor): Predicted bounding boxes (N x 4).
            pred_labels (torch.Tensor): Labels for predicted boxes (N).
            gt_boxes (torch.Tensor): Ground truth bounding boxes (M x 4).
            gt_labels (torch.Tensor): Labels for ground truth boxes (M).
            iou_thresh (float): IoU threshold for a match.

        Returns:
            dict: Average IoU and unmatched ground truth information.
        """
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return {
                "average_iou": 0.0,
                "unmatched_gt": len(gt_boxes),
                "matched_gt": 0,
            }

        matched_gt = set()
        matched_predictions = []  # Store (weight, iou) for matched predictions only

        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            pred_area = abs((pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1]))

            # Try to match this prediction with a GT box
            for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if gt_idx not in matched_gt and pred_label == gt_label:
                    iou = self._compute_iou(pred_box, gt_box)
                    if iou >= iou_thresh:
                        matched_gt.add(gt_idx)
                        matched_predictions.append((pred_area, iou.item()))
                        break

        # Compute weighted average IoU using only matched predictions
        if not matched_predictions:
            avg_iou = 0.0
        else:
            total_weighted_iou = 0.0
            total_weight = 0.0

            for weight, iou in matched_predictions:
                total_weighted_iou += weight * iou
                total_weight += weight

            avg_iou = total_weighted_iou / total_weight

        return {
            "average_iou": avg_iou,
            "unmatched_gt": len(gt_boxes) - len(matched_gt),
            "matched_gt": len(matched_gt),
        }

    def _compute_average_iou_with_labels_across_iou(
        self, pred_boxes, pred_labels, gt_boxes, gt_labels
    ):
        res_50 = self._compute_average_iou_with_labels(
            pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thresh=0.50
        )
        res_75 = self._compute_average_iou_with_labels(
            pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thresh=0.75
        )
        res_90 = self._compute_average_iou_with_labels(
            pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thresh=0.90
        )
        res_95 = self._compute_average_iou_with_labels(
            pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thresh=0.95
        )

        return {
            "average_iou_50": res_50["average_iou"],
            "average_iou_75": res_75["average_iou"],
            "average_iou_90": res_90["average_iou"],
            "average_iou_95": res_95["average_iou"],
        }

    def _find_intersecting_labels(
        self,
        ds: Dataset,
    ) -> tuple[dict[str, int], dict[str, int], list[DocItemLabel], list[DocItemLabel]]:
        r"""
        Compute counters per labels for the groundtruth, prediciton and their intersections

        Returns
        -------
        true_labels: dict[label -> counter]
        pred_labels: dict[label -> counter]
        intersection_labels: list[DocItemLabel]
        """

        true_labels: Dict[str, int] = {}
        pred_labels: Dict[str, int] = {}

        for i, data in tqdm(
            enumerate(ds), desc="Layout evaluations", ncols=120, total=len(ds)
        ):
            data_record = DatasetRecordWithPrediction.model_validate(data)
            true_doc = data_record.ground_truth_doc
            pred_doc = self._get_pred_doc(data_record)

            for item, level in true_doc.iterate_items(
                included_content_layers={c for c in ContentLayer},
                traverse_pictures=True,
            ):
                if isinstance(item, DocItem):
                    mapped_label = self.label_mapping.get(item.label)
                    if mapped_label is not None:
                        for prov in item.prov:
                            if mapped_label in true_labels:
                                true_labels[mapped_label] += 1
                            else:
                                true_labels[mapped_label] = 1

            if pred_doc:
                for item, level in pred_doc.iterate_items(
                    included_content_layers={c for c in ContentLayer},
                    traverse_pictures=True,
                ):
                    if isinstance(item, DocItem):
                        mapped_label = self.label_mapping.get(item.label)
                        if mapped_label is not None:
                            for prov in item.prov:
                                if mapped_label in pred_labels:
                                    pred_labels[mapped_label] += 1
                                else:
                                    pred_labels[mapped_label] = 1

        """
        logging.info(f"True labels:")
        for label, count in true_labels.items():
            logging.info(f" => {label}: {count}")

        logging.info(f"Pred labels:")
        for label, count in pred_labels.items():
            logging.info(f" => {label}: {count}")
        """

        intersection_labels: List[DocItemLabel] = []
        union_labels: List[DocItemLabel] = []
        for label, count in true_labels.items():
            union_labels.append(DocItemLabel(label))

            if label in pred_labels:
                intersection_labels.append(DocItemLabel(label))

        for label, count in pred_labels.items():
            if label not in true_labels:
                union_labels.append(DocItemLabel(label))

        return true_labels, pred_labels, intersection_labels, union_labels

    def _collect_items_by_page(
        self,
        doc: DoclingDocument,
        filter_labels: List[DocItemLabel],
    ) -> Dict[int, List[DocItem]]:
        """
        Collect DocItems by page number for the given document and filter labels.

        Args:
            doc: The DoclingDocument to process
            filter_labels: List of labels to include in the collection

        Returns:
            Dictionary mapping page numbers to lists of DocItems
        """
        pages_to_objects: Dict[int, List[DocItem]] = defaultdict(list)

        for item, level in doc.iterate_items(
            included_content_layers={c for c in ContentLayer},
            traverse_pictures=True,
            with_groups=True,
        ):
            if (
                isinstance(item, DocItem)
                and self.label_mapping[item.label] in filter_labels
            ):
                for prov in item.prov:
                    pages_to_objects[prov.page_no].append(item)

        return pages_to_objects

    def _extract_layout_data(
        self,
        true_doc: DoclingDocument,
        pred_doc: DoclingDocument,
        filter_labels: List[DocItemLabel],
    ) -> Tuple[
        List[Tuple[int, Dict[str, torch.Tensor]]],
        List[Tuple[int, Dict[str, torch.Tensor]]],
    ]:
        r"""
        Filter to keep only bboxes from the given labels
        Convert each bbox to top-left-origin, normalize to page size and scale 100

        This method ensures proper page-wise alignment between GT and predictions.
        Each returned GT tensor at index i corresponds exactly to the prediction tensor at index i.

        Returns
        -------
        ground_truths: List of (page_no, tensor_dict) tuples
        predictions: List of (page_no, tensor_dict) tuples
        """
        # Collect all DocItems by page for both GT and predictions
        true_pages_to_objects = self._collect_items_by_page(true_doc, filter_labels)
        pred_pages_to_objects = self._collect_items_by_page(pred_doc, filter_labels)

        # Get all pages that have GT data (we evaluate based on GT pages)
        gt_pages = set(true_pages_to_objects.keys())
        pred_pages = set(pred_pages_to_objects.keys())

        _log.debug(f"GT pages: {sorted(gt_pages)}, Pred pages: {sorted(pred_pages)}")

        # Process pages in sorted order to ensure consistent alignment
        ground_truths: List[Tuple[int, Dict[str, torch.Tensor]]] = []
        predictions: List[Tuple[int, Dict[str, torch.Tensor]]] = []

        for page_no in sorted(gt_pages):
            # Always process GT for this page
            gt_data = self._extract_page_data(
                page_no=page_no,
                items=true_pages_to_objects[page_no],
                doc=true_doc,
                filter_labels=filter_labels,
                is_prediction=False,
            )

            # Handle prediction for this page based on strategy
            if page_no in pred_pages:
                # We have prediction data for this page
                pred_data = self._extract_page_data(
                    page_no=page_no,
                    items=pred_pages_to_objects[page_no],
                    doc=pred_doc,
                    filter_labels=filter_labels,
                    is_prediction=True,
                )
            else:
                # No prediction data for this page
                if (
                    self.missing_prediction_strategy
                    == MissingPredictionStrategy.PENALIZE
                ):
                    # Create empty prediction tensor (zero score)
                    pred_data = {
                        "boxes": torch.empty(0, 4),
                        "labels": torch.empty(0, dtype=torch.long),
                        "scores": torch.empty(0),
                    }
                elif (
                    self.missing_prediction_strategy == MissingPredictionStrategy.IGNORE
                ):
                    # Skip this page entirely
                    continue
                else:
                    raise ValueError(
                        f"Unknown missing prediction strategy: {self.missing_prediction_strategy}"
                    )

            # Add the aligned GT-Pred pair with page number
            ground_truths.append((page_no, gt_data))
            predictions.append((page_no, pred_data))

        # Verify alignment (this should always be true now)
        assert len(ground_truths) == len(
            predictions
        ), f"Critical error: GT and Pred lists misaligned: {len(ground_truths)} vs {len(predictions)}"

        # Verify corresponding page numbers match
        for i, ((gt_page, _), (pred_page, _)) in enumerate(
            zip(ground_truths, predictions)
        ):
            assert (
                gt_page == pred_page
            ), f"Page number mismatch at index {i}: GT page {gt_page} vs Pred page {pred_page}"

        _log.debug(f"Processed {len(ground_truths)} page pairs successfully")

        return ground_truths, predictions

    def _extract_page_data(
        self,
        page_no: int,
        items: List[DocItem],
        doc: DoclingDocument,
        filter_labels: List[DocItemLabel],
        is_prediction: bool,
    ) -> Dict[str, torch.Tensor]:
        """Extract bbox data for a single page."""
        page_size = doc.pages[page_no].size
        page_height = page_size.height
        page_width = page_size.width

        bboxes = []
        labels = []
        if is_prediction:
            scores = []

        for item in items:
            for prov in item.prov:
                if (
                    prov.page_no == page_no
                ):  # Only process provenances for this specific page
                    bbox = prov.bbox.to_top_left_origin(page_height=page_height)
                    bbox = bbox.normalized(page_size)
                    bbox = bbox.scaled(100.0)

                    bboxes.append([bbox.l, bbox.t, bbox.r, bbox.b])
                    labels.append(filter_labels.index(self.label_mapping[item.label]))  # type: ignore
                    if is_prediction:
                        scores.append(1.0)  # FIXME: Use actual confidence scores

        result: Dict[str, torch.Tensor] = {
            "boxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        if is_prediction:
            result["scores"] = torch.tensor(scores, dtype=torch.float32)

        return result

    def _compute_area_level_metrics_for_tensors(
        self,
        gt_boxes: torch.Tensor,
        pred_boxes: torch.Tensor,
        page_width: int,
        page_height: int,
        mask_width: int = 512,
        mask_height: int = 512,
    ) -> Tuple[float, float, float]:
        """
        Compute area-level precision, recall, and F1 score for tensor format boxes.
        Handles overlapping boxes by using binary masks at the specified resolution.

        Args:
            gt_boxes: Ground truth boxes as tensor of shape (N, 4) with [x1, y1, x2, y2] format
            pred_boxes: Predicted boxes as tensor of shape (M, 4) with [x1, y1, x2, y2] format
            page_width: Width of the original page
            page_height: Height of the original page
            mask_width: Width of the mask to use for computation (default: 512)
            mask_height: Height of the mask to use for computation (default: 512)

        Returns:
            Dictionary containing precision, recall, and F1 scores
        """
        if gt_boxes.shape[0] == 0:
            precision = 1.0 if pred_boxes.shape[0] == 0 else 0.0
            recall = 1.0
            f1 = 1.0 if pred_boxes.shape[0] == 0 else 0.0
            return precision, recall, f1

        if pred_boxes.shape[0] == 0:
            precision = 1.0
            recall = 0.0
            f1 = 0.0
            return precision, recall, f1

        # Calculate scaling factors (ensure float division)
        x_scale = float(mask_width) / float(page_width)
        y_scale = float(mask_height) / float(page_height)

        # Create empty masks
        gt_mask = torch.zeros((mask_height, mask_width), dtype=torch.bool, device="cpu")
        pred_mask = torch.zeros(
            (mask_height, mask_width), dtype=torch.bool, device="cpu"
        )

        # Fill ground truth mask
        for i in range(gt_boxes.shape[0]):
            x1, y1, x2, y2 = gt_boxes[i].tolist()

            # Scale coordinates to mask space
            x1, y1 = max(0, int(x1 * x_scale)), max(0, int(y1 * y_scale))
            x2, y2 = min(mask_width, int(x2 * x_scale)), min(
                mask_height, int(y2 * y_scale)
            )

            if x2 > x1 and y2 > y1:
                gt_mask[y1:y2, x1:x2] = True

        # Fill prediction mask
        for i in range(pred_boxes.shape[0]):
            x1, y1, x2, y2 = pred_boxes[i].tolist()

            # Scale coordinates to mask space
            x1, y1 = max(0, int(x1 * x_scale)), max(0, int(y1 * y_scale))
            x2, y2 = min(mask_width, int(x2 * x_scale)), min(
                mask_height, int(y2 * y_scale)
            )

            if x2 > x1 and y2 > y1:
                pred_mask[y1:y2, x1:x2] = True

        # Calculate areas (accounting for overlaps)
        total_gt_area = torch.sum(gt_mask).item()
        total_pred_area = torch.sum(pred_mask).item()

        # Calculate intersection (logical AND of masks)
        intersection_mask = torch.logical_and(gt_mask, pred_mask)
        total_intersection = torch.sum(intersection_mask).item()

        # Calculate metrics
        precision = total_intersection / total_pred_area if total_pred_area > 0 else 0.0
        recall = total_intersection / total_gt_area if total_gt_area > 0 else 0.0

        # Calculate F1 score
        f1 = 0.0
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)

        return precision, recall, f1

    def _compute_area_level_metrics_excluding_pictures(
        self,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        pred_boxes: torch.Tensor,
        pred_labels: torch.Tensor,
        filter_labels: List[DocItemLabel],
        page_width: int,
        page_height: int,
        mask_width: int = 512,
        mask_height: int = 512,
    ) -> Tuple[float, float, float]:
        """
        Compute area-level precision, recall, and F1 score excluding PICTURE labels.
        Handles overlapping boxes by using binary masks at the specified resolution.

        Args:
            gt_boxes: Ground truth boxes as tensor of shape (N, 4) with [x1, y1, x2, y2] format
            gt_labels: Ground truth labels as tensor of shape (N,)
            pred_boxes: Predicted boxes as tensor of shape (M, 4) with [x1, y1, x2, y2] format
            pred_labels: Predicted labels as tensor of shape (M,)
            filter_labels: List of DocItemLabel used for label indexing
            page_width: Width of the original page
            page_height: Height of the original page
            mask_width: Width of the mask to use for computation (default: 512)
            mask_height: Height of the mask to use for computation (default: 512)

        Returns:
            Tuple containing precision, recall, and F1 scores (excluding PICTURE labels)
        """
        # Find the index of PICTURE label if it exists in filter_labels
        picture_label_idx = None
        try:
            picture_label_idx = filter_labels.index(DocItemLabel.PICTURE)
        except ValueError:
            # PICTURE label not in filter_labels, no filtering needed
            pass

        # Filter out PICTURE labels from ground truth
        if picture_label_idx is not None and len(gt_labels) > 0:
            non_picture_mask = gt_labels != picture_label_idx
            filtered_gt_boxes = gt_boxes[non_picture_mask]
        else:
            filtered_gt_boxes = gt_boxes

        # Filter out PICTURE labels from predictions
        if picture_label_idx is not None and len(pred_labels) > 0:
            non_picture_mask = pred_labels != picture_label_idx
            filtered_pred_boxes = pred_boxes[non_picture_mask]
        else:
            filtered_pred_boxes = pred_boxes

        # Use the existing method with filtered boxes
        return self._compute_area_level_metrics_for_tensors(
            gt_boxes=filtered_gt_boxes,
            pred_boxes=filtered_pred_boxes,
            page_width=page_width,
            page_height=page_height,
            mask_width=mask_width,
            mask_height=mask_height,
        )
