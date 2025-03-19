import json
from pathlib import Path
from typing import Any, List, Optional, Union

import PIL
from datasets import Features
from datasets import Image as Features_Image
from datasets import Sequence, Value
from docling.datamodel.base_models import ConversionStatus
from docling_core.types import DoclingDocument
from docling_core.types.io import DocumentStream
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.fields import FieldInfo

from docling_eval.benchmarks.constants import EvaluationModality


class DatasetRecord(BaseModel):
    predictor_info: dict = Field(alias="predictor_info", default={})
    status: ConversionStatus = Field(alias="status", default=ConversionStatus.PENDING)

    doc_id: str = Field(alias="document_id")
    doc_path: Optional[Path] = Field(alias="document_filepath", default=None)
    doc_hash: Optional[str] = Field(alias="document_filehash")

    ground_truth_doc: DoclingDocument = Field(alias="GroundTruthDocument")
    predicted_doc: Optional[DoclingDocument] = Field(
        alias="PredictedDocument", default=None
    )
    original: Optional[Union[DocumentStream | Path]] = Field(
        alias="BinaryDocument", default=None
    )

    ground_truth_page_images: List[PIL.Image.Image] = Field(
        alias="GroundTruthPageImages", default=[]
    )
    ground_truth_pictures: List[PIL.Image.Image] = Field(
        alias="GroundTruthPictures", default=[]
    )

    predicted_page_images: List[PIL.Image.Image] = Field(
        alias="GroundTruthPageImages", default=[]
    )
    predicted_pictures: List[PIL.Image.Image] = Field(
        alias="GroundTruthPageImages", default=[]
    )

    mime_type: str = Field(default="")
    modalities: List[EvaluationModality] = Field(default=[])

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    @classmethod
    def get_field_alias(cls, field_name: str) -> str:
        return cls.model_fields[field_name].alias or field_name

    @classmethod
    def features(cls):
        return Features(
            {
                cls.get_field_alias("predictor_info"): Value("string"),
                cls.get_field_alias("status"): Value("string"),
                cls.get_field_alias("doc_id"): Value("string"),
                cls.get_field_alias("doc_path"): Value("string"),
                cls.get_field_alias("doc_hash"): Value("string"),
                cls.get_field_alias("ground_truth_doc"): Value("string"),
                cls.get_field_alias("ground_truth_pictures"): Sequence(
                    Features_Image()
                ),
                cls.get_field_alias("ground_truth_page_images"): Sequence(
                    Features_Image()
                ),
                cls.get_field_alias("predicted_doc"): Value("string"),
                cls.get_field_alias("predicted_pictures"): Sequence(Features_Image()),
                cls.get_field_alias("predicted_page_images"): Sequence(
                    Features_Image()
                ),
                cls.get_field_alias("original"): Value("string"),
                cls.get_field_alias("mime_type"): Value("string"),
                cls.get_field_alias("modalities"): Sequence(Value("string")),
            }
        )

    def _extract_images(self, document: DoclingDocument):

        pictures = []
        page_images = []

        # Save page images
        for img_no, picture in enumerate(document.pictures):
            if picture.image is not None:
                # img = picture.image.pil_image
                # pictures.append(to_pil(picture.image.uri))
                pictures.append(picture.image.pil_image)
                picture.image.uri = Path(
                    f"{self.get_field_alias('predicted_pictures')}/{img_no}"
                )

        # Save page images
        for page_no, page in document.pages.items():
            if page.image is not None:
                # img = page.image.pil_image
                # img.show()
                page_images.append(page.image.pil_image)
                page.image.uri = Path(
                    f"{self.get_field_alias('predicted_page_images')}/{page_no}"
                )

        return pictures, page_images

    def as_record_dict(self):
        record = {
            self.get_field_alias("predictor_info"): json.dumps(self.predictor_info),
            self.get_field_alias("status"): str(self.status),
            self.get_field_alias("doc_id"): self.doc_id,
            self.get_field_alias("doc_path"): str(self.doc_path),
            self.get_field_alias("doc_hash"): self.doc_hash,
            self.get_field_alias("ground_truth_doc"): json.dumps(
                self.ground_truth_doc.export_to_dict()
            ),
            self.get_field_alias("ground_truth_pictures"): self.ground_truth_pictures,
            self.get_field_alias(
                "ground_truth_page_images"
            ): self.ground_truth_page_images,
            self.get_field_alias("mime_type"): self.mime_type,
            self.get_field_alias("modalities"): list(self.modalities),
        }
        if isinstance(self.original, Path):
            with self.original.open("rb") as f:
                record.update({self.get_field_alias("original"): f.read()})
        elif isinstance(self.original, DocumentStream):
            record.update(
                {
                    self.get_field_alias("original"): None
                }  # FIXME: reading from closed I/O in self.original.stream.read()}
            )
        else:
            record.update({self.get_field_alias("original"): None})

        if self.predicted_doc is not None:
            record.update(
                {
                    self.get_field_alias("predicted_doc"): json.dumps(
                        self.predicted_doc.export_to_dict()
                    ),
                }
            )

        return record

    @model_validator(mode="after")
    def validate_model(self) -> "DatasetRecord":
        if self.predicted_doc is not None:
            if not len(self.predicted_pictures) and not len(self.predicted_page_images):
                pictures, page_images = self._extract_images(self.predicted_doc)

                self.predicted_page_images = page_images
                self.predicted_pictures = pictures

        if not len(self.ground_truth_pictures) and not len(
            self.ground_truth_page_images
        ):
            pictures, page_images = self._extract_images(self.ground_truth_doc)

            self.ground_truth_page_images = page_images
            self.ground_truth_pictures = pictures

        return self
