import glob
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from datasets import Dataset, load_dataset
from docling_core.types.doc.document import (
    DEFAULT_EXPORT_LABELS,
    DocItem,
    DoclingDocument,
)
from docling_core.types.doc.labels import DocItemLabel
from pydantic import BaseModel
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm  # type: ignore

from docling_eval.benchmarks.constants import BenchMarkColumns


class LayoutEvaluation(BaseModel):
    name: str
    label: str
    value: float


class DatasetLayoutEvaluation(BaseModel):
    true_labels: Dict[str, int]
    pred_labels: Dict[str, int]

    intersecting_labels: List[str]

    evaluations: List[LayoutEvaluation]

    def to_table(self) -> Tuple[List[List[str]], List[str]]:

        headers = ["label", "Class mAP[0.5:0.95]"]

        self.evaluations = sorted(self.evaluations, key=lambda x: x.value, reverse=True)

        table = []
        for i in range(len(self.evaluations)):
            table.append(
                [
                    f"{self.evaluations[i].label}",
                    f"{100.0*self.evaluations[i].value:.2f}",
                ]
            )

        return table, headers


class LayoutEvaluator:

    def __init__(self) -> None:
        self.filter_labels = []
        self.label_names = {}

        for i, _ in enumerate(DEFAULT_EXPORT_LABELS):
            self.filter_labels.append(_)
            self.label_names[i] = _

    def __call__(self, ds_path: Path, split: str = "test") -> DatasetLayoutEvaluation:

        test_path = str(ds_path / "test" / "*.parquet")
        train_path = str(ds_path / "train" / "*.parquet")

        test_files = glob.glob(test_path)
        train_files = glob.glob(train_path)
        logging.info(f"test-files: {test_files}, train-files: {train_files}")

        # Load all files into the `test`-`train` split
        ds = None
        if len(test_files) > 0 and len(train_files) > 0:
            ds = load_dataset(
                "parquet", data_files={"test": test_files, "train": train_files}
            )
        elif len(test_files) > 0 and len(train_files) == 0:
            ds = load_dataset("parquet", data_files={"test": test_files})

        logging.info(f"oveview of dataset: {ds}")

        if ds is not None:
            ds_selection = ds[split]

        true_labels, pred_labels, intersection_labels = self._find_intersecting_labels(
            ds_selection
        )
        logging.info(f"Intersection labels: {intersection_labels}")

        ground_truths = []
        predictions = []

        for i, data in tqdm(
            enumerate(ds_selection),
            desc="Layout evaluations",
            ncols=120,
            total=len(ds_selection),
        ):
            true_doc_dict = data[BenchMarkColumns.GROUNDTRUTH]
            true_doc = DoclingDocument.model_validate_json(true_doc_dict)
            pred_doc_dict = data[BenchMarkColumns.PREDICTION]
            pred_doc = DoclingDocument.model_validate_json(pred_doc_dict)

            gts, preds = self._evaluate_layouts_in_documents(
                doc_id=data[BenchMarkColumns.DOC_ID],
                true_doc=true_doc,
                pred_doc=pred_doc,
                filter_labels=intersection_labels,
            )

            if len(gts) == len(preds):
                ground_truths.extend(gts)
                predictions.extend(preds)
            else:
                logging.error("Ignoring predictions for document")

        assert len(ground_truths) == len(
            predictions
        ), "len(ground_truths)==len(predictions)"

        # Initialize Mean Average Precision metric
        metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

        # Update metric with predictions and ground truths
        metric.update(predictions, ground_truths)

        # Compute mAP and other metrics
        result = metric.compute()

        evaluations: List[LayoutEvaluation] = []
        for key, value in result.items():
            if isinstance(value, float):
                evaluations.append(LayoutEvaluation(name=key, value=value, label=None))

        if "map_per_class" in result:
            for label_idx, class_map in enumerate(result["map_per_class"]):
                evaluations.append(
                    LayoutEvaluation(
                        name="Class mAP[0.5:0.95]",
                        label=intersection_labels[label_idx].value,
                        value=class_map,
                    )
                )

        """
        # Print results
        print("Results:")
        for key, value in result.items():
            try:
                print(f"{key}: {value:.3f}")
            except:
                print(f"{key}: {value}")

        # Overall mAP
        print(f"Overall mAP[0.5:0.95]: {result['map'].item():.3f}")

        print("\nPer-Class mAP[0.5:0.95]:")
        if "map_per_class" in result:
            for label_idx, class_map in enumerate(result["map_per_class"]):
                # label_name = self.label_names.get(label_idx, f"Class {label_idx}")  # Use label name or default
                print(
                    f" => {label_idx} {intersection_labels[label_idx].value}: {class_map:.3f}"
                )
        """

        return DatasetLayoutEvaluation(
            evaluations=evaluations,
            true_labels=true_labels,
            pred_labels=pred_labels,
            intersecting_labels=[_.value for _ in intersection_labels],
        )

    def _find_intersecting_labels(
        self, ds: Dataset
    ) -> tuple[dict[str, int], dict[str, int], list[DocItemLabel]]:
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
            true_doc_dict = data[BenchMarkColumns.GROUNDTRUTH]
            true_doc = DoclingDocument.model_validate_json(true_doc_dict)

            pred_doc_dict = data[BenchMarkColumns.PREDICTION]
            pred_doc = DoclingDocument.model_validate_json(pred_doc_dict)

            for item, level in true_doc.iterate_items():
                if isinstance(item, DocItem):  # and item.label in filter_labels:
                    for prov in item.prov:
                        if item.label in true_labels:
                            true_labels[item.label] += 1
                        else:
                            true_labels[item.label] = 1

            for item, level in pred_doc.iterate_items():
                if isinstance(item, DocItem):  # and item.label in filter_labels:
                    for prov in item.prov:
                        if item.label in pred_labels:
                            pred_labels[item.label] += 1
                        else:
                            pred_labels[item.label] = 1

        """
        logging.info(f"True labels:")
        for label, count in true_labels.items():
            logging.info(f" => {label}: {count}")

        logging.info(f"Pred labels:")
        for label, count in pred_labels.items():
            logging.info(f" => {label}: {count}")
        """

        intersection_labels: List[DocItemLabel] = []
        for label, count in true_labels.items():
            if label in pred_labels:
                intersection_labels.append(DocItemLabel(label))

        return true_labels, pred_labels, intersection_labels

    def _evaluate_layouts_in_documents(
        self,
        doc_id: str,
        true_doc: DoclingDocument,
        pred_doc: DoclingDocument,
        filter_labels: List[DocItemLabel],
    ) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
        r"""
        Returns
        -------
        ground_truths: List of dict with keys "bboxes", "labels" and values are tensors
        predictions: List of dict with keys "bboxes", "labels", "scores" and values are tensors
        """

        # logging.info(f"#-true-tables: {len(true_tables)}, #-pred-tables: {len(pred_tables)}")
        assert len(true_doc.pages) == len(
            pred_doc.pages
        ), f"len(true_doc.pages)==len(pred_doc.pages) => {len(true_doc.pages)}=={len(pred_doc.pages)}"

        # page_num -> List[DocItem]
        true_pages_to_objects: Dict[int, List[DocItem]] = {}
        pred_pages_to_objects: Dict[int, List[DocItem]] = {}

        for item, level in true_doc.iterate_items():
            if isinstance(item, DocItem) and item.label in filter_labels:
                for prov in item.prov:
                    if prov.page_no not in true_pages_to_objects:
                        true_pages_to_objects[prov.page_no] = [item]
                    else:
                        true_pages_to_objects[prov.page_no].append(item)

        for item, level in pred_doc.iterate_items():
            if isinstance(item, DocItem) and item.label in filter_labels:
                for prov in item.prov:
                    if prov.page_no not in pred_pages_to_objects:
                        pred_pages_to_objects[prov.page_no] = [item]
                    else:
                        pred_pages_to_objects[prov.page_no].append(item)

        ground_truths = []
        predictions = []

        # logging.info("\n\n ============================================ \n\n")

        for page_no, items in true_pages_to_objects.items():

            page_size = true_doc.pages[page_no].size

            page_height = page_size.height
            page_width = page_size.width

            bboxes = []
            labels = []
            for item in items:
                for prov in item.prov:

                    bbox = prov.bbox.to_bottom_left_origin(page_height=page_height)
                    bbox = bbox.normalized(page_size)
                    bbox = bbox.scaled(100.0)

                    # logging.info(f"ground-truth {page_no}: {page_width, page_height} -> {item.label}, {bbox.coord_origin}: [{bbox.l}, {bbox.t}, {bbox.r}, {bbox.b}]")

                    bboxes.append([bbox.l, bbox.t, bbox.r, bbox.b])
                    labels.append(filter_labels.index(item.label))

            ground_truths.append(
                {
                    "boxes": torch.tensor(bboxes),
                    "labels": torch.tensor(labels),
                }
            )

        for page_no, items in pred_pages_to_objects.items():

            page_size = pred_doc.pages[page_no].size

            page_height = page_size.height
            page_width = page_size.width

            bboxes = []
            labels = []
            scores = []
            for item in items:
                for prov in item.prov:

                    bbox = prov.bbox.to_bottom_left_origin(page_height=page_height)
                    bbox = bbox.normalized(page_size)
                    bbox = bbox.scaled(100.0)

                    # logging.info(f"prediction {page_no}: {page_width, page_height} -> {item.label}, {bbox.coord_origin}: [{bbox.l}, {bbox.b}, {bbox.r}, {bbox.t}]")

                    bboxes.append([bbox.l, bbox.b, bbox.r, bbox.t])
                    labels.append(filter_labels.index(item.label))
                    scores.append(1.0)  # FIXME

            predictions.append(
                {
                    "boxes": torch.tensor(bboxes),
                    "labels": torch.tensor(labels),
                    "scores": torch.tensor(scores),
                }
            )

        """
        assert len(ground_truths) == len(
            predictions
        ), f"len(ground_truths)==len(predictions) => {len(ground_truths)}=={len(predictions)}"
        """

        return ground_truths, predictions
