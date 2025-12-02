"""Utilities to map CVAT page-level identifiers to document metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from docling_eval.datamodels.cvat_types import AnnotatedImage, AnnotationOverview


def _canonical_page_id(doc_hash: str, page_no: int) -> str:
    return f"{doc_hash}__page_{page_no:05d}"


@dataclass(frozen=True)
class PageRef:
    """Reference metadata for a single annotated page."""

    doc_hash: str
    doc_name: str
    page_no: int
    image_name: str

    @property
    def canonical_page_id(self) -> str:
        return _canonical_page_id(self.doc_hash, self.page_no)


class CvatPageMapping:
    """Lookup helper backed by ``cvat_overview.json``."""

    def __init__(
        self,
        pages_by_doc_hash: Dict[str, List[PageRef]],
        pages_by_doc_name: Dict[str, List[PageRef]],
        pages_by_image_name: Dict[str, PageRef],
    ) -> None:
        self._pages_by_doc_hash = pages_by_doc_hash
        self._pages_by_doc_name = pages_by_doc_name
        self._pages_by_image_name = pages_by_image_name

    @classmethod
    def from_overview(cls, overview: AnnotationOverview) -> "CvatPageMapping":
        pages_by_doc_hash: Dict[str, List[PageRef]] = {}
        pages_by_doc_name: Dict[str, List[PageRef]] = {}
        pages_by_image_name: Dict[str, PageRef] = {}

        doc_lookup = {doc.doc_hash: doc.doc_name for doc in overview.doc_annotations}

        def build_page_refs(
            image_name: str, image: AnnotatedImage
        ) -> Iterable[PageRef]:
            if not image.page_nos:
                return []
            doc_name = doc_lookup.get(image.doc_hash, image.doc_name)
            return [
                PageRef(
                    doc_hash=image.doc_hash,
                    doc_name=doc_name,
                    page_no=page_no,
                    image_name=image_name,
                )
                for page_no in image.page_nos
            ]

        for image_name, image in overview.img_annotations.items():
            for ref in build_page_refs(image_name, image):
                pages_by_doc_hash.setdefault(ref.doc_hash, []).append(ref)
                pages_by_doc_name.setdefault(ref.doc_name, []).append(ref)
                pages_by_image_name.setdefault(ref.image_name, ref)

        def sort_refs(refs: List[PageRef]) -> None:
            refs.sort(key=lambda ref: ref.page_no)

        for refs in pages_by_doc_hash.values():
            sort_refs(refs)
        for refs in pages_by_doc_name.values():
            sort_refs(refs)

        return cls(
            pages_by_doc_hash=pages_by_doc_hash,
            pages_by_doc_name=pages_by_doc_name,
            pages_by_image_name=pages_by_image_name,
        )

    @classmethod
    def from_overview_path(cls, overview_path: Path) -> "CvatPageMapping":
        overview = AnnotationOverview.load_from_json(overview_path)
        return cls.from_overview(overview)

    def pages_for_doc_hash(self, doc_hash: str) -> List[PageRef]:
        return self._pages_by_doc_hash.get(doc_hash, [])

    def pages_for_doc_name(self, doc_name: str) -> List[PageRef]:
        return self._pages_by_doc_name.get(doc_name, [])

    def page_for_image(self, image_name: str) -> Optional[PageRef]:
        return self._pages_by_image_name.get(image_name)

    def iter_pages(self) -> Iterable[PageRef]:
        for refs in self._pages_by_doc_hash.values():
            yield from refs
