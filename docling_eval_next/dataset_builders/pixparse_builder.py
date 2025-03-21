import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from docling.cli.main import OcrEngine
from docling_core.types import DoclingDocument
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    PageItem,
    ProvenanceItem,
    Size,
)
from docling_core.types.io import DocumentStream
from PIL import Image
from tqdm import tqdm

from docling_eval.benchmarks.utils import get_binary, get_binhash
from docling_eval.visualisation.visualisations import save_comparison_html_with_clusters
from docling_eval_next.datamodels.dataset_record import DatasetRecord
from docling_eval_next.dataset_builders.dataset_builder import (
    BaseEvaluationDatasetBuilder,
    HFSource,
)
from docling_eval_next.prediction_providers.prediction_provider import (
    BasePredictionProvider,
)
from docling_eval_next.utils.hyperscalers.utils import CustomHyperscaler, Hyperscaler

TRUE_HTML_EXPORT_LABELS = {
    DocItemLabel.TITLE,
    DocItemLabel.DOCUMENT_INDEX,
    DocItemLabel.SECTION_HEADER,
    DocItemLabel.PARAGRAPH,
    DocItemLabel.TABLE,
    DocItemLabel.PICTURE,
    DocItemLabel.FORMULA,
    DocItemLabel.CHECKBOX_UNSELECTED,
    DocItemLabel.CHECKBOX_SELECTED,
    DocItemLabel.TEXT,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.CODE,
    DocItemLabel.REFERENCE,
    DocItemLabel.CAPTION,
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.PAGE_FOOTER,
    DocItemLabel.FOOTNOTE,
}

PRED_HTML_EXPORT_LABELS = {
    DocItemLabel.TITLE,
    DocItemLabel.DOCUMENT_INDEX,
    DocItemLabel.SECTION_HEADER,
    DocItemLabel.PARAGRAPH,
    DocItemLabel.TABLE,
    DocItemLabel.PICTURE,
    DocItemLabel.FORMULA,
    DocItemLabel.CHECKBOX_UNSELECTED,
    DocItemLabel.CHECKBOX_SELECTED,
    DocItemLabel.TEXT,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.CODE,
    DocItemLabel.REFERENCE,
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.PAGE_FOOTER,
    DocItemLabel.FOOTNOTE,
}


class OCRDatasetBuilder(BaseEvaluationDatasetBuilder):
    def __init__(
        self,
        name: str,
        prediction_provider: BasePredictionProvider,
        target: Path,
        do_visualization: bool = True,
        ocr_engine: Optional[OcrEngine] = None,
        ocr_lang: List[str] = ["en"],
    ):
        super().__init__(
            name=name,
            dataset_source=HFSource(repo_id="samiuc/pixparse-idl"),
            prediction_provider=prediction_provider,
            target=target,
        )
        self.do_visualization = do_visualization
        self.ocr_engine = ocr_engine
        self.ocr_lang = ocr_lang


class OCRBenchmarkDatasetBuilder(OCRDatasetBuilder):
    def __init__(
        self,
        name: str,
        prediction_provider: BasePredictionProvider,
        dataset_local_path: Path,
        target: Path,
        do_visualization: bool = True,
        ocr_engine: Optional[OcrEngine] = None,
        hyperscaler: Optional[Union[Hyperscaler, CustomHyperscaler]] = None,
        ocr_lang: List[str] = ["en"],
        reprocess: bool = False,
        max_items: int = -1,
    ):
        super().__init__(
            name=name,
            prediction_provider=prediction_provider,
            target=target,
            do_visualization=do_visualization,
            ocr_engine=ocr_engine,
            ocr_lang=ocr_lang,
        )
        self.name = name
        self.dataset_local_path = dataset_local_path
        self.do_visualization = do_visualization
        self.hyperscaler = hyperscaler
        self.reprocess = reprocess
        self.max_items = max_items

    def _create_ground_truth_doc(
        self, doc_id: str, gt_data: Dict, image_file: Path
    ) -> DoclingDocument:
        """Create a DoclingDocument from ground truth data and image file."""
        w, h = Image.open(image_file).size

        true_doc = DoclingDocument(name=doc_id)
        true_doc.pages[1] = PageItem(
            size=Size(width=float(w), height=float(h)), page_no=1
        )

        for page_idx, page in enumerate(gt_data["pages"], 1):
            for text, bbox, _ in zip(page["text"], page["bbox"], page["score"]):
                bbox_obj = BoundingBox.from_tuple(
                    (
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[0] + bbox[2]),
                        float(bbox[1] + bbox[3]),
                    ),
                    CoordOrigin.TOPLEFT,
                )
                prov = ProvenanceItem(
                    page_no=page_idx, bbox=bbox_obj, charspan=(0, len(text))
                )
                true_doc.add_text(label=DocItemLabel.TEXT, text=text, prov=prov)

        return true_doc

    def iterate(self) -> Iterable[DatasetRecord]:
        # if not self.retrieved:
        #     raise RuntimeError(
        #         "You must first retrieve the source dataset. Call retrieve_input_dataset()."
        #     )

        assert self.dataset_local_path is not None

        # Create output directories
        output_dir = self.target
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create visualization directory if needed
        viz_dir = output_dir / "visualizations"
        if self.do_visualization:
            viz_dir.mkdir(parents=True, exist_ok=True)

        # Find all ground truth files
        ground_truth_files = list(self.dataset_local_path.rglob("ground_truth.json"))
        if self.max_items > 0:
            ground_truth_files = ground_truth_files[: self.max_items]

        # Determine which services to process
        services_to_process: list[Union[Hyperscaler, CustomHyperscaler, OcrEngine]] = []

        if self.hyperscaler:
            services_to_process.append(self.hyperscaler)
        if self.ocr_engine:
            services_to_process.append(self.ocr_engine)

        # Default to all hyperscalers only if no specific service is provided
        if not services_to_process:
            services_to_process.extend([h for h in Hyperscaler])

        for gt_file in tqdm(
            ground_truth_files,
            desc="Processing files for OCR Benchmark",
            total=len(ground_truth_files),
            ncols=128,
        ):
            try:
                # Find the corresponding image file
                image_file = gt_file.parent / "original.tif"
                if not image_file.exists():
                    logging.info(f"Warning: No image file found for {gt_file}")
                    continue

                doc_id = gt_file.parent.name
                input_dir = self.dataset_local_path

                # Process with each requested service
                for service in services_to_process:
                    try:
                        service_name = service.value
                        with open(gt_file, "r") as f:
                            gt_data = json.load(f)

                        true_doc = self._create_ground_truth_doc(
                            doc_id, gt_data, image_file
                        )

                        image_bytes = get_binary(image_file)
                        image_stream = DocumentStream(
                            name=image_file.name, stream=BytesIO(image_bytes)
                        )
                        record = DatasetRecord(
                            doc_id=doc_id,
                            doc_hash=get_binhash(image_bytes),
                            ground_truth_doc=true_doc,
                            original=image_stream,
                            mime_type="image/tiff",
                        )

                        # service_records[service_name].append(record)

                        # Predictions need to happen in the PredictionClass
                        self.update_prediction(
                            record=record,
                            # image_bytes # Pass image bytes instead of image_file?
                            reprocess=self.reprocess,
                            image_file=image_file,
                            doc_id=doc_id,
                            input_dir=input_dir,
                            output_dir=output_dir,
                            service=service,
                        )

                        # Generate visualization if requested
                        # if self.do_visualization and record.predicted_doc is not None:
                        #     save_comparison_html_with_clusters(
                        #         filename=viz_dir
                        #         / f"{os.path.basename(image_file)}-clusters.html",
                        #         true_doc=true_doc,
                        #         pred_doc=record.predicted_doc,
                        #         page_image=image_bytes,
                        #         true_labels=TRUE_HTML_EXPORT_LABELS,
                        #         pred_labels=PRED_HTML_EXPORT_LABELS,
                        #     )
                        yield record

                    except Exception as e:
                        logging.error(
                            f"Error processing {doc_id} with {service.value}: {str(e)}"
                        )
                        raise

            except Exception as e:
                logging.error(f"Error processing {gt_file}: {str(e)}")
                raise
