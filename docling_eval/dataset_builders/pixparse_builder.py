import base64
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

from datasets import Features, Sequence, Value, load_dataset
from docling_core.types import DoclingDocument
from docling_core.types.doc import BoundingBox, CoordOrigin, ImageRef, PageItem, Size
from docling_core.types.doc.page import (
    BoundingRectangle,
    PageGeometry,
    SegmentedPage,
    TextCell,
)
from docling_core.types.io import DocumentStream
from PIL import Image
from tqdm import tqdm

from docling_eval.datamodels.dataset_record import DatasetRecord
from docling_eval.datamodels.types import BenchMarkColumns, EvaluationModality
from docling_eval.dataset_builders.dataset_builder import (
    BaseEvaluationDatasetBuilder,
    HFSource,
)
from docling_eval.utils.utils import extract_images, from_pil_to_base64uri, get_binhash


class PixparseDatasetBuilder(BaseEvaluationDatasetBuilder):
    DEFAULT_REPO_ID = "samiuc/pixparse-idl"

    def __init__(
        self,
        target: Path,
        split: str = "test",
        begin_index: int = 0,
        end_index: int = -1,
        dataset_source: Optional[Path] = None,
    ):
        if dataset_source is not None:
            repo_id = str(dataset_source)
        else:
            repo_id = self.DEFAULT_REPO_ID
        source = HFSource(repo_id=repo_id)
        super().__init__(
            name="pixparse-idl",
            dataset_source=source,
            target=target,
            split=split,
            begin_index=begin_index,
            end_index=end_index,
        )
        self.split = split
        self.must_retrieve = True

    def _create_ground_truth_doc(
        self, doc_id: str, gt_data: Dict, image: Image.Image
    ) -> Tuple[DoclingDocument, Dict[int, SegmentedPage]]:
        true_doc = DoclingDocument(name=doc_id)
        img_width, img_height = image.width, image.height

        image_ref = ImageRef(
            mimetype="image/png",
            dpi=72,
            size=Size(width=float(img_width), height=float(img_height)),
            uri=from_pil_to_base64uri(image),
        )
        page_item = PageItem(
            page_no=1,
            size=Size(width=float(img_width), height=float(img_height)),
            image=image_ref,
        )
        true_doc.pages[1] = page_item

        segmented_pages: Dict[int, SegmentedPage] = {}
        pages_data = gt_data.get("pages", [])
        if not pages_data:
            logging.warning(f"No pages found in ground truth for doc_id: {doc_id}")
            return true_doc, segmented_pages

        for page_idx, page in enumerate(pages_data, 1):
            seg_page = SegmentedPage(
                dimension=PageGeometry(
                    angle=0,
                    rect=BoundingRectangle.from_bounding_box(
                        BoundingBox(l=0, t=0, r=img_width, b=img_height)
                    ),
                )
            )

            texts = page.get("text", [])
            bboxes = page.get("bbox", [])
            scores = page.get("score", [])

            for text, bbox, score in zip(texts, bboxes, scores):
                if not text or not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                try:
                    rel_x, rel_y, rel_w, rel_h = map(float, bbox)
                    abs_l = rel_x * img_width
                    abs_t = rel_y * img_height
                    abs_r = (rel_x + rel_w) * img_width
                    abs_b = (rel_y + rel_h) * img_height

                    bbox_obj = BoundingBox.from_tuple(
                        (abs_l, abs_t, abs_r, abs_b),
                        CoordOrigin.TOPLEFT,
                    )

                    cell = TextCell(
                        from_ocr=True,
                        rect=BoundingRectangle.from_bounding_box(bbox_obj),
                        text=text.strip(),
                        orig=text,
                        confidence=float(score) if score is not None else None,
                    )
                    seg_page.textline_cells.append(cell)
                except (ValueError, TypeError) as e:
                    logging.warning(
                        f"Error processing bbox {bbox} for doc_id {doc_id}: {e}"
                    )
                    continue

            segmented_pages[page_idx] = seg_page

        return true_doc, segmented_pages

    def iterate(self) -> Iterable[DatasetRecord]:
        if not self.retrieved and self.must_retrieve:
            raise RuntimeError(
                "You must first retrieve the source dataset. Call retrieve_input_dataset()."
            )

        assert self.dataset_local_path is not None

        self.target.mkdir(parents=True, exist_ok=True)
        features = Features(
            {
                "key": Value("string"),
                "image": {
                    "bytes": Value("binary"),
                    "path": Value("string"),
                },
                "json_data": Value("string"),
                "text": Sequence(Value("string")),
                "bbox": Value("string"),
                "poly": Value("string"),
                "score": Sequence(Value("float32")),
            }
        )

        local_parquet_path = self.dataset_local_path / "idl_ocr_dataset.parquet"

        dataset = load_dataset(
            "parquet",
            data_files=str(local_parquet_path),
            split="train",
            features=features,
        )

        begin, end = self.get_effective_indices(len(dataset))
        dataset = dataset.select(range(begin, end))

        for sample in tqdm(
            dataset,
            desc="Processing files for PixParse IDL dataset from HF",
            total=len(dataset),
            ncols=128,
        ):
            try:
                doc_id = sample["key"]
                image_data = sample["image"]

                if image_data is None or image_data["bytes"] is None:
                    logging.warning(
                        f"Skipping sample because of missing image bytes: {doc_id}"
                    )
                    continue

                image_bytes = image_data["bytes"]
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                gt_data = json.loads(sample["json_data"])

                true_doc, seg_pages = self._create_ground_truth_doc(
                    doc_id, gt_data, image
                )

                true_doc, true_pictures, true_page_images = extract_images(
                    document=true_doc,
                    pictures_column=BenchMarkColumns.GROUNDTRUTH_PICTURES.value,
                    page_images_column=BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES.value,
                )

                with BytesIO() as img_byte_stream:
                    image.save(img_byte_stream, format="PNG")
                    img_byte_stream.seek(0)
                    img_bytes = img_byte_stream.getvalue()

                image_stream = DocumentStream(
                    name=f"{doc_id}.png", stream=BytesIO(img_bytes)
                )

                yield DatasetRecord(
                    doc_id=doc_id,
                    doc_hash=get_binhash(img_bytes),
                    ground_truth_doc=true_doc,
                    ground_truth_segmented_pages=seg_pages,
                    original=image_stream,
                    mime_type="image/png",
                    modalities=[EvaluationModality.OCR],
                    ground_truth_pictures=true_pictures,
                    ground_truth_page_images=true_page_images,
                )

            except Exception as e:
                logging.error(
                    f"Error processing sample {sample.get('key', 'unknown')}: {e}"
                )
                raise
