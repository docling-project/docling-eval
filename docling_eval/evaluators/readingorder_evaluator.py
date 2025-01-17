import copy
import json
import logging
import math
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset
from deepsearch_glm.andromeda_nlp import nlp_model  # type: ignore
from docling.datamodel.base_models import BoundingBox
from docling_core.types.doc.document import DocItem, DoclingDocument, TextItem
from docling_core.utils.legacy import docling_document_to_legacy
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from tqdm import tqdm  # type: ignore

from docling_eval.benchmarks.constants import BenchMarkColumns
from docling_eval.benchmarks.utils import draw_arrow
from docling_eval.evaluators.stats import DatasetStatistics, compute_stats

_log = logging.getLogger(__name__)


class PageReadingOrderEvaluation(BaseModel):
    doc_id: str

    # BBoxes are in BOTTOMLEFT origin and in the true order
    bboxes: List[Tuple[float, float, float, float]]
    pred_order: List[int]
    ard_norm: float  # Normalized ARD: 0 is the worst and 1 is the best
    w_ard_norm: (
        float  # Weighted normalized ARD. The weight is the (bbox_area / page_area)
    )


class DatasetReadingOrderEvaluation(BaseModel):
    evaluations: List[PageReadingOrderEvaluation]
    ard_stats: DatasetStatistics
    w_ard_stats: DatasetStatistics


class ReadingOrderEvaluator:
    r"""
    Evaluate the reading order using the Average Relative Distance metric
    """

    def __init__(self):
        self._nlp_model = nlp_model(loglevel="error", text_ordering=True)

    def __call__(
        self, ds_path: Path, split: str = "test"
    ) -> DatasetReadingOrderEvaluation:
        parquet_files = str(ds_path / split / "*.parquet")
        ds = load_dataset("parquet", data_files={split: parquet_files})
        _log.info(f"oveview of dataset: {ds}")
        if ds is not None:
            ds_selection = ds[split]

        evaluations: list[PageReadingOrderEvaluation] = []
        ards = []
        w_ards = []

        broken_inputs = 0
        for i, data in tqdm(
            enumerate(ds_selection),
            desc="Reading order evaluations",
            ncols=120,
            total=len(ds_selection),
        ):
            doc_id = data[BenchMarkColumns.DOC_ID]
            true_doc_dict = data[BenchMarkColumns.GROUNDTRUTH]
            true_doc: DoclingDocument = DoclingDocument.model_validate_json(
                true_doc_dict
            )
            # print(f"\n{i} - doc_id: {doc_id}")
            # self._show_items(true_doc)

            reading_order = self._get_reading_order_preds(doc_id, true_doc)
            if reading_order is None:
                print(f"Broken input: {doc_id}")
                broken_inputs += 1
                continue

            # Compute metrics
            # ard_norm = self._compute_ard_norm(reading_order)
            ard_norm, w_ard_norm = self._compute_ard(reading_order)
            ards.append(ard_norm)
            w_ards.append(w_ard_norm)

            page_evaluation = PageReadingOrderEvaluation(
                doc_id=doc_id,
                bboxes=[b.as_tuple() for b in reading_order["bboxes"]],
                pred_order=reading_order["pred_order"],
                ard_norm=ard_norm,
                w_ard_norm=w_ard_norm,
            )
            # print("pred_reading_order")
            # print(page_evaluation)
            # print(f"ard={ard}")

            evaluations.append(page_evaluation)

        if broken_inputs > 0:
            _log.error(f"broken_inputs={broken_inputs}")

        # Compute statistics for metrics
        ard_stats = compute_stats(ards)
        w_ard_stats = compute_stats(w_ards)

        ds_reading_order_evaluation = DatasetReadingOrderEvaluation(
            evaluations=evaluations, ard_stats=ard_stats, w_ard_stats=w_ard_stats
        )

        return ds_reading_order_evaluation

    def _get_reading_order_preds(self, doc_id: str, true_doc: DoclingDocument):
        r"""

        Returns
        -------
        reading_order: Keys are "bboxes" and "pred_order"
        """
        try:
            page_size = true_doc.pages[1].size

            # Convert the bboxes to bottom-left coords before running the GLM
            bboxes = []
            for item_id, (item, level) in enumerate(true_doc.iterate_items()):
                pred_len = len(item.prov)  # type: ignore
                if pred_len > 1:
                    _log.warning(
                        "Skipping element %s in document %s as it has %s provenances",
                        item_id,
                        doc_id,
                        pred_len,
                    )
                    continue

                # Convert the bbox to BOTTOM-LEFT origin
                bbox = item.prov[0].bbox.to_bottom_left_origin(page_size.height)  # type: ignore
                item.prov[0].bbox = bbox  # type: ignore
                bboxes.append(copy.deepcopy(bbox))

            # Run the reading order model
            legacy_doc = docling_document_to_legacy(true_doc)
            legacy_doc_dict = legacy_doc.model_dump(by_alias=True, exclude_none=True)
            legacy_doc_dict = self._filter_out_bboxes(legacy_doc_dict, bboxes)
            legacy_doc_dict = self._ensure_bboxes_in_legacy_tables(legacy_doc_dict)
            glm_doc = self._nlp_model.apply_on_doc(legacy_doc_dict)

            # original reading order -> predicted reading order
            orig_to_pred_order: Dict[int, int] = {}
            for po, pe in enumerate(glm_doc["page-elements"]):
                orig_to_pred_order[pe["orig-order"]] = po
            pred_order = [orig_to_pred_order[x] for x in range(len(orig_to_pred_order))]

            reading_order = {"bboxes": bboxes, "pred_order": pred_order}
            return reading_order
        except RuntimeError as ex:
            _log.error(str(ex))
            return None

    def _compute_ard(self, reading_order: Dict) -> tuple[float, float]:
        r"""
        Compute the metrics:
        1. Normalized Average Relative Distance (ARD)
        2. Weighted normalized Average Relative Distance.

        ARD = (1/n) * sum(e_k)
        e_k = abs(pred_order_index  - gt_order_index)
        0 is the best and n-1 is the worst where n is the number of bboxes

        ARD_norm = 1 - (ARD / n)
        0 is the worst and 1 is the best

        weighted_ARD = (1/n) * sum(e_k * weight_k)
        weight_k = area(bbox_k) / area(page)
        weighted ARD_norm = 1 - (weighted_ARD / n)

        Returns
        -------
        ard_norm: Normalized average relative distance
        ward_norm: Normalized weighted average to the area of the bbox
        """
        n = len(reading_order["bboxes"])
        if n == 0:
            return 0.0, 0.0

        # Compute bbox weights
        bbox_areas = [b.area() for b in reading_order["bboxes"]]
        total_bboxes = sum(bbox_areas)
        weights = [(a / total_bboxes) for a in bbox_areas]

        # Compute ARD and weighted ARD
        ard = 0.0
        w_ard = 0.0
        for true_ro, pred_ro in enumerate(reading_order["pred_order"]):
            dist = math.fabs(true_ro - pred_ro)
            ard += dist
            w_ard += dist * weights[true_ro]

        n_sq = n * n
        ard_norm = 1 - (ard / n_sq)
        w_ard_norm = 1 - (w_ard / n_sq)
        return ard_norm, w_ard_norm

    def _ensure_bboxes_in_legacy_tables(self, legacy_doc_dict: Dict):
        r"""
        Ensure bboxes for all table cells
        """
        for table in legacy_doc_dict["tables"]:
            for row in table["data"]:
                for cell in row:
                    if "bbox" not in cell:
                        cell["bbox"] = [0, 0, 0, 0]
        return legacy_doc_dict

    def _filter_out_bboxes(
        self, legacy_doc_dict: Dict, bboxes: List[BoundingBox]
    ) -> Dict:
        r"""
        Remove entries from "main-text" with bbox outside of the provided bboxes
        """
        # Make set of existing bboxes as tuples
        existing_bboxes = set([b.as_tuple() for b in bboxes])

        # Identify main ids to be deleted
        main_ids_to_delete = set()
        for main_id, main in enumerate(legacy_doc_dict["main-text"]):
            if "prov" not in main:
                continue
            for prov in main["prov"]:
                bbox = prov["bbox"]
                # Check if bbox is a tuple or a list
                if tuple(bbox) not in existing_bboxes:
                    main_ids_to_delete.add(main_id)

        # Reconstruct the main
        if main_ids_to_delete:
            filtered_mains = []
            for main_id, main in enumerate(legacy_doc_dict["main-text"]):
                if main_id in main_ids_to_delete:
                    continue
                filtered_mains.append(main)
            legacy_doc_dict["main-text"] = filtered_mains

        return legacy_doc_dict

    def _show_items(self, true_doc: DoclingDocument):
        r""" """
        page_size = true_doc.pages[1].size
        for i, (item, level) in enumerate(true_doc.iterate_items()):
            bbox = (
                item.prov[0].bbox.to_bottom_left_origin(page_size.height)
                if isinstance(item, DocItem)
                else None
            )
            text = item.text if isinstance(item, TextItem) else None
            label = item.label  # type: ignore
            print(f"True {i}: {level} - {label}: {bbox} - {text}")


class ReadingOrderVisualizer:
    r"""
    Generate visualizations of the GT and predicted reading order
    """

    def __init__(self):
        self._line_width = 2
        self._true_arrow_color = "green"
        self._pred_arrow_color = "red"
        self._item_color = "blue"
        self._viz_sub_dir = "reading_order_viz"

        # Load a font (adjust the font size and path as needed)
        self._font = ImageFont.load_default()
        try:
            self._font = ImageFont.truetype("arial.ttf", size=15)
        except IOError:
            self._font = ImageFont.load_default()

    def __call__(
        self,
        ds_path: Path,
        reading_order_report_fn: Path,
        save_dir: Path,
        split: str = "test",
    ):
        r"""
        Use a pre-generated reading order report and visualize the original and predicted reading
        order. Generate one html visualization per document and save it in the output dir.
        """
        save_dir /= self._viz_sub_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        # Read the evaluation report and make an index: doc_id -> predicted reading order
        ro_preds_idx: dict[str, list[int]] = {}
        with open(reading_order_report_fn, "r") as fd:
            ro_evaluation_dict = json.load(fd)
            for evaluation in ro_evaluation_dict["evaluations"]:
                doc_id = evaluation["doc_id"]
                ro_preds_idx[doc_id] = evaluation["pred_order"]

        # Open the converted dataset
        parquet_files = str(ds_path / split / "*.parquet")
        ds = load_dataset("parquet", data_files={split: parquet_files})
        if ds is not None:
            ds_selection = ds[split]

        # Visualize the reading order
        viz_fns: list[Path] = []
        for i, data in tqdm(
            enumerate(ds_selection),
            desc="Reading order visualizations",
            ncols=120,
            total=len(ds_selection),
        ):
            doc_id = data[BenchMarkColumns.DOC_ID]
            page_images = data[BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES]
            true_doc_dict = data[BenchMarkColumns.GROUNDTRUTH]
            true_doc: DoclingDocument = DoclingDocument.model_validate_json(
                true_doc_dict
            )
            pred_order = ro_preds_idx[doc_id]

            # Draw and save the visualization
            image_bytes = page_images[0]["bytes"]
            image = Image.open(BytesIO(image_bytes))
            viz_image = self._draw_permuted_reading_order(
                doc_id, image, true_doc, pred_order
            )
            viz_fn = save_dir / f"{doc_id}_reading_order_viz.png"
            viz_fns.append(viz_fn)
            viz_image.save(viz_fn)

        return viz_fns

    def _draw_permuted_reading_order(
        self,
        doc_id: str,
        page_image: Image.Image,
        doc: DoclingDocument,
        pred_order: list[int],
    ) -> Image.Image:
        # TODO: Add the reading order also as labels
        bboxes = []

        true_img = copy.deepcopy(page_image)
        true_draw = ImageDraw.Draw(true_img)

        # Draw the bboxes and true order
        x0, y0 = -1.0, -1.0
        for item_id, (item, level) in enumerate(doc.iterate_items()):
            if not isinstance(item, DocItem):
                continue

            pred_len = len(item.prov)
            if pred_len > 1:
                # _log.warning("Skipping element %s in document %s as it has %s provenances",
                #              item_id, doc_id, pred_len)
                continue

            prov = item.prov[0]

            # Get the item's bbox in top-left origin for the image dimensions
            bbox = prov.bbox.to_top_left_origin(
                page_height=doc.pages[prov.page_no].size.height
            )
            bbox = bbox.normalized(doc.pages[prov.page_no].size)
            bbox.l = round(bbox.l * true_img.width)
            bbox.r = round(bbox.r * true_img.width)
            bbox.t = round(bbox.t * true_img.height)
            bbox.b = round(bbox.b * true_img.height)
            if bbox.b > bbox.t:
                bbox.b, bbox.t = bbox.t, bbox.b

            bboxes.append(bbox)

            # Draw rectangle with only a border
            true_draw.rectangle(
                [bbox.l, bbox.b, bbox.r, bbox.t],
                outline=self._item_color,
                width=self._line_width,
            )

            # Get the arrow coordinates
            if x0 == -1 and y0 == -1:
                x0 = (bbox.l + bbox.r) / 2.0
                y0 = (bbox.b + bbox.t) / 2.0
            else:
                x1 = (bbox.l + bbox.r) / 2.0
                y1 = (bbox.b + bbox.t) / 2.0

                true_draw = draw_arrow(
                    true_draw,
                    (x0, y0, x1, y1),
                    color=self._true_arrow_color,
                    line_width=self._line_width,
                )
                x0, y0 = x1, y1

        # Draw the bboxes and the predicted order
        pred_img = copy.deepcopy(page_image)
        pred_draw = ImageDraw.Draw(pred_img)
        x0, y0 = -1.0, -1.0
        for true_id in range(len(bboxes)):
            pred_id = pred_order[true_id]
            bbox = bboxes[pred_id]

            # Draw rectangle with only a border
            pred_draw.rectangle(
                [bbox.l, bbox.b, bbox.r, bbox.t],
                outline=self._item_color,
                width=self._line_width,
            )

            # Get the arrow coordinates
            if x0 == -1 and y0 == -1:
                x0 = (bbox.l + bbox.r) / 2.0
                y0 = (bbox.b + bbox.t) / 2.0
            else:
                x1 = (bbox.l + bbox.r) / 2.0
                y1 = (bbox.b + bbox.t) / 2.0

                pred_draw = draw_arrow(
                    pred_draw,
                    (x0, y0, x1, y1),
                    color=self._pred_arrow_color,
                    line_width=self._line_width,
                )
                x0, y0 = x1, y1

        # Make combined image
        mode = page_image.mode
        w, h = page_image.size
        combined_img = Image.new(mode, (2 * w, h), "white")
        combined_img.paste(true_img, (0, 0))
        combined_img.paste(pred_img, (w, 0))

        return combined_img
