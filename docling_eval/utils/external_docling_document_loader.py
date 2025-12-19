import logging
from pathlib import Path
from typing import Optional

from docling_core.types.doc.document import (
    DoclingDocument,
    DocTagsDocument,
    DocTagsPage,
)
from PIL import Image

from docling_eval.datamodels.dataset_record import DatasetRecord

_log = logging.getLogger(__name__)


class ExternalDoclingDocumentLoader:
    r""" """

    def __init__(
        self,
        external_predictions_dir: Path,
        enable_cache: bool = True,
    ):
        r""" """
        self._external_predictions_dir = external_predictions_dir
        self._enable_cache = enable_cache

        # doc-id -> DoclingDocument
        self._cache: dict[str, DoclingDocument] = {}

    def predictions_path(self) -> Path:
        r""" """
        return self._external_predictions_dir

    def get(self, record: DatasetRecord) -> Optional[DoclingDocument]:
        r"""
        Cache enabled loading of DoclingDocuments

        Logic:
        1. If the cache is enabled and the doc_id is in cache return it.
        2. Otherwise try to load the document and return it.
        3. If the document cannot be loaded return None
        """
        doc_id = record.doc_id
        doc: Optional[DoclingDocument] = None
        if self._enable_cache and doc_id in self._cache:
            doc = self._cache[doc_id]
        else:
            doc = self.load(record)
        return doc

    def load(
        self,
        record: DatasetRecord,
    ) -> Optional[DoclingDocument]:
        r"""
        Load the DoclingDocument from the external predictions path

        The following fields are used from the `record` parameter:
        - record.doc_id
        - record.ground_truth_page_images[0]
        """
        doc_id = record.doc_id

        json_fn = self._external_predictions_dir / f"{doc_id}.json"
        doctags_fn = ExternalDoclingDocumentLoader.build_doctags_path(
            self._external_predictions_dir, doc_id
        )
        yaml_fn = self._external_predictions_dir / f"{doc_id}.yaml"
        yml_fn = self._external_predictions_dir / f"{doc_id}.yml"

        doc: Optional[DoclingDocument] = None
        if json_fn.is_file():
            doc = DoclingDocument.load_from_json(json_fn)
        elif doctags_fn.is_file():
            gt_page_images = record.ground_truth_page_images
            gt_page_image = gt_page_images[0] if len(gt_page_images) > 0 else None

            doc = ExternalDoclingDocumentLoader.load_doctags(
                doc_id,
                self._external_predictions_dir,
                gt_page_image=gt_page_image,
            )
        elif yaml_fn.is_file():
            doc = DoclingDocument.load_from_yaml(yaml_fn)
        elif yml_fn.is_file():
            doc = DoclingDocument.load_from_yaml(yml_fn)

        if doc is None:
            _log.error("Failed to load document: %s", doc_id)

        # Check if to update the cache
        if self._enable_cache and doc is not None:
            _log.debug("Caching externally loaded document: %s", doc_id)
            self._cache[doc_id] = doc

        return doc

    @staticmethod
    def build_doctags_path(doctags_root: Path, doc_id: str) -> Path:
        r"""Get the full path of the doctags file"""
        dt_path = doctags_root / f"{doc_id}.dt"
        return dt_path

    @staticmethod
    def load_doctags(
        doc_id: str,
        doctags_root: Path,
        page_images_root: Optional[Path] = None,
        gt_page_image: Optional[Image.Image] = None,
        image_filename_extension: str = "png",
    ) -> Optional[DoclingDocument]:
        r"""
        Load a single page DoclingDocument object from a doctags file and a page image.

        The page image is supplied from these sources in the specific order:
        1. The page_images_root: An image with filename <doc_id>.<image_filename_extension> is used
        2. gt_page_image: An explicit Image object is used.
        3. Search for the image with filename <doc_id>.<image_filename_extension> in the doctags root

        Parameters
        ----------
        doctags_root: Root path to load doctags as files with name <doc_id>.dt
        doc_id: The document id of the file to be loaded
        page_images_root: If provided, search for the page images here first.
        gt_page_image: If provided, search use that object for the page image.
        image_filename_extension: The file extension for the page image.

        Returns
        -------
        DoclingDocument object or None if the document cannot be reconstructed
        """
        # Read the doctags file
        doctags_fn = ExternalDoclingDocumentLoader.build_doctags_path(
            doctags_root, doc_id
        )

        try:
            with open(doctags_fn, "r") as fd:
                doctags = fd.read()

            page_image: Optional[Image.Image] = None

            if page_images_root:
                page_image_fn = (
                    page_images_root / f"{doc_id}.{image_filename_extension}"
                )
                if page_image_fn.is_file():
                    page_image = Image.open(page_image_fn)
                else:
                    _log.warning("Failed to load page image: %s", page_image_fn)
            elif gt_page_image is not None:
                page_image = gt_page_image
            else:
                page_image_fn = doctags_root / f"{doc_id}.{image_filename_extension}"
                if page_image_fn.is_file():
                    page_image = Image.open(page_image_fn)
                else:
                    _log.warning(
                        "Missing page image file: %s. Reconstruct doctags without page image",
                        page_image_fn,
                    )

            # Build DoclingDocument
            doctags_page = DocTagsPage(tokens=doctags, image=page_image)
            doctags_doc = DocTagsDocument(pages=[doctags_page])
            doc = DoclingDocument.load_from_doctags(doctags_doc, document_name=doc_id)
            return doc
        except Exception as e:
            _log.error(f"Error loading doctags document {doc_id}: {str(e)}")
            return None
