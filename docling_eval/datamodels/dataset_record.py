import json
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Union

import PIL
from datasets import Features
from datasets import Image as Features_Image
from datasets import Sequence, Value
from docling.datamodel.base_models import ConversionStatus
from docling_core.types import DoclingDocument
from docling_core.types.doc.page import SegmentedPage
from docling_core.types.io import DocumentStream
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from docling_eval.datamodels.types import EvaluationModality, PredictionFormats
from docling_eval.utils.utils import extract_images

seg_adapter = TypeAdapter(Dict[int, SegmentedPage])


class FieldType(Enum):
    STRING = "string"
    BINARY = "binary"
    IMAGE_LIST = "image_list"
    STRING_LIST = "string_list"


class SchemaGenerator:
    """Generates both HuggingFace Features and PyArrow schemas from a field definition."""

    @staticmethod
    def _get_features_type(field_type: FieldType):
        mapping = {
            FieldType.STRING: Value("string"),
            FieldType.BINARY: Value("binary"),
            FieldType.IMAGE_LIST: Sequence(Features_Image()),
            FieldType.STRING_LIST: Sequence(Value("string")),
        }
        return mapping[field_type]

    @staticmethod
    def _get_pyarrow_type(field_type: FieldType):
        import pyarrow as pa

        image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])

        mapping = {
            FieldType.STRING: pa.string(),
            FieldType.BINARY: pa.binary(),
            FieldType.IMAGE_LIST: pa.list_(image_type),
            FieldType.STRING_LIST: pa.list_(pa.string()),
        }
        return mapping[field_type]

    @classmethod
    def generate_features(cls, field_definitions: Dict[str, FieldType]) -> Features:
        return Features(
            {
                field_name: cls._get_features_type(field_type)
                for field_name, field_type in field_definitions.items()
            }
        )

    @classmethod
    def generate_pyarrow_schema(cls, field_definitions: Dict[str, FieldType]):
        import pyarrow as pa

        return pa.schema(
            [
                (field_name, cls._get_pyarrow_type(field_type))
                for field_name, field_type in field_definitions.items()
            ]
        )


class DatasetRecord(
    BaseModel
):  # TODO make predictionrecord class, factor prediction-related fields there.
    doc_id: str = Field(alias="document_id")
    doc_path: Optional[Path] = Field(alias="document_filepath", default=None)
    doc_hash: Optional[str] = Field(alias="document_filehash", default=None)

    ground_truth_doc: DoclingDocument = Field(alias="GroundTruthDocument")
    ground_truth_segmented_pages: Dict[int, SegmentedPage] = Field(
        alias="ground_truth_segmented_pages", default={}
    )
    original: Optional[Union[DocumentStream, Path]] = Field(
        alias="BinaryDocument", default=None
    )
    # TODO add optional columns to store the SegmentedPage, both for GT and prediction

    ground_truth_page_images: List[PIL.Image.Image] = Field(
        alias="GroundTruthPageImages", default=[]
    )
    ground_truth_pictures: List[PIL.Image.Image] = Field(
        alias="GroundTruthPictures", default=[]
    )

    mime_type: str = Field(default="application/pdf")
    modalities: List[EvaluationModality] = Field(default=[])

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    @classmethod
    def get_field_alias(cls, field_name: str) -> str:
        return cls.model_fields[field_name].alias or field_name

    @classmethod
    def _get_field_definitions(cls) -> Dict[str, FieldType]:
        """Define the schema for this class. Override in subclasses to extend."""
        return {
            cls.get_field_alias("doc_id"): FieldType.STRING,
            cls.get_field_alias("doc_path"): FieldType.STRING,
            cls.get_field_alias("doc_hash"): FieldType.STRING,
            cls.get_field_alias("ground_truth_doc"): FieldType.STRING,
            cls.get_field_alias("ground_truth_segmented_pages"): FieldType.STRING,
            cls.get_field_alias("ground_truth_pictures"): FieldType.IMAGE_LIST,
            cls.get_field_alias("ground_truth_page_images"): FieldType.IMAGE_LIST,
            cls.get_field_alias("original"): FieldType.BINARY,
            cls.get_field_alias("mime_type"): FieldType.STRING,
            cls.get_field_alias("modalities"): FieldType.STRING_LIST,
        }

    @classmethod
    def features(cls):
        return SchemaGenerator.generate_features(cls._get_field_definitions())

    @classmethod
    def pyarrow_schema(cls):
        """Generate PyArrow schema that matches the HuggingFace datasets image format."""
        return SchemaGenerator.generate_pyarrow_schema(cls._get_field_definitions())

    def _extract_images(
        self,
        document: DoclingDocument,
        pictures_field_prefix: str,
        pages_field_prefix: str,
    ):
        """
        Extract images using the global utility implementation.
        """
        _, pictures, page_images = extract_images(
            document=document,
            pictures_column=pictures_field_prefix,
            page_images_column=pages_field_prefix,
        )
        return pictures, page_images

    def as_record_dict(self):
        record = {
            self.get_field_alias("doc_id"): self.doc_id,
            self.get_field_alias("doc_path"): str(self.doc_path),
            self.get_field_alias("doc_hash"): self.doc_hash,
            self.get_field_alias("ground_truth_doc"): json.dumps(
                self.ground_truth_doc.export_to_dict()
            ),
            self.get_field_alias("ground_truth_pictures"): self.ground_truth_pictures,
            self.get_field_alias("ground_truth_segmented_pages"): seg_adapter.dump_json(
                self.ground_truth_segmented_pages
            ).decode("utf-8"),
            self.get_field_alias(
                "ground_truth_page_images"
            ): self.ground_truth_page_images,
            self.get_field_alias("mime_type"): self.mime_type,
            self.get_field_alias("modalities"): list(
                [m.value for m in self.modalities]
            ),
        }
        if isinstance(self.original, Path):
            with self.original.open("rb") as f:
                record.update({self.get_field_alias("original"): f.read()})
        elif isinstance(self.original, DocumentStream):
            record.update(
                {self.get_field_alias("original"): self.original.stream.read()}
            )
        else:
            record.update({self.get_field_alias("original"): None})

        return record

    @model_validator(mode="after")
    def validate_images(self) -> "DatasetRecord":
        if not len(self.ground_truth_pictures) and not len(
            self.ground_truth_page_images
        ):
            pictures, page_images = self._extract_images(
                self.ground_truth_doc,
                pictures_field_prefix=self.get_field_alias("ground_truth_pictures"),
                pages_field_prefix=self.get_field_alias("ground_truth_page_images"),
            )

            self.ground_truth_page_images = page_images
            self.ground_truth_pictures = pictures

        return self

    @model_validator(mode="before")
    @classmethod
    def validate_record_dict(cls, data: dict):
        gt_doc_alias = cls.get_field_alias("ground_truth_doc")
        if gt_doc_alias in data and isinstance(data[gt_doc_alias], str):
            data[gt_doc_alias] = json.loads(data[gt_doc_alias])

        gt_seg_pages_alias = cls.get_field_alias("ground_truth_segmented_pages")
        if gt_seg_pages_alias in data and isinstance(
            data[gt_seg_pages_alias], (str, bytes)
        ):
            seg_pages_data = data[gt_seg_pages_alias]
            if isinstance(seg_pages_data, bytes):
                seg_pages_data = seg_pages_data.decode("utf-8")
            data[gt_seg_pages_alias] = seg_adapter.validate_json(seg_pages_data)

        gt_page_img_alias = cls.get_field_alias("ground_truth_page_images")
        if gt_page_img_alias in data:
            for ix, item in enumerate(data[gt_page_img_alias]):
                if isinstance(item, dict):
                    data[gt_page_img_alias][ix] = Features_Image().decode_example(item)

        gt_pic_img_alias = cls.get_field_alias("ground_truth_pictures")
        if gt_pic_img_alias in data:
            for ix, item in enumerate(data[gt_pic_img_alias]):
                if isinstance(item, dict):
                    data[gt_pic_img_alias][ix] = Features_Image().decode_example(item)

        gt_binary = cls.get_field_alias("original")
        if gt_binary in data:
            if isinstance(data[gt_binary], bytes):
                data[gt_binary] = DocumentStream(
                    name="file", stream=BytesIO(data[gt_binary])
                )
            elif isinstance(data[gt_binary], PIL.Image.Image):
                # Handle PIL Images by converting to bytes
                img_buffer = BytesIO()
                data[gt_binary].save(img_buffer, format="PNG")
                img_buffer.seek(0)
                data[gt_binary] = DocumentStream(name="image.png", stream=img_buffer)

        return data


class DatasetRecordWithPrediction(DatasetRecord):
    predictor_info: Dict = Field(alias="predictor_info", default={})
    status: ConversionStatus = Field(alias="status", default=ConversionStatus.PENDING)

    predicted_doc: Optional[DoclingDocument] = Field(
        alias="PredictedDocument", default=None
    )

    predicted_segmented_pages: Dict[int, SegmentedPage] = Field(
        alias="predicted_segmented_pages", default={}
    )

    original_prediction: Optional[str] = None
    prediction_format: PredictionFormats = (
        PredictionFormats.DOCLING_DOCUMENT
    )  # default for old files
    prediction_timings: Optional[Dict] = Field(alias="prediction_timings", default=None)

    predicted_page_images: List[PIL.Image.Image] = Field(
        alias="PredictionPageImages", default=[]
    )
    predicted_pictures: List[PIL.Image.Image] = Field(
        alias="PredictionPictures", default=[]
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    @classmethod
    def _get_field_definitions(cls) -> Dict[str, FieldType]:
        """Extend the parent schema with prediction-specific fields."""
        base_definitions = super()._get_field_definitions()
        prediction_definitions = {
            cls.get_field_alias("predictor_info"): FieldType.STRING,
            cls.get_field_alias("status"): FieldType.STRING,
            cls.get_field_alias("predicted_doc"): FieldType.STRING,
            cls.get_field_alias("predicted_segmented_pages"): FieldType.STRING,
            cls.get_field_alias("predicted_pictures"): FieldType.IMAGE_LIST,
            cls.get_field_alias("predicted_page_images"): FieldType.IMAGE_LIST,
            cls.get_field_alias("prediction_format"): FieldType.STRING,
            cls.get_field_alias("prediction_timings"): FieldType.STRING,
            cls.get_field_alias("original_prediction"): FieldType.STRING,
        }
        return {**base_definitions, **prediction_definitions}

    @classmethod
    def features(cls):
        return SchemaGenerator.generate_features(cls._get_field_definitions())

    @classmethod
    def pyarrow_schema(cls):
        """Generate PyArrow schema that matches the HuggingFace datasets image format."""
        return SchemaGenerator.generate_pyarrow_schema(cls._get_field_definitions())

    def as_record_dict(self):
        record = super().as_record_dict()
        record.update(
            {
                self.get_field_alias("prediction_format"): self.prediction_format.value,
                self.get_field_alias("prediction_timings"): (
                    json.dumps(self.prediction_timings)
                    if self.prediction_timings is not None
                    else None
                ),
                self.get_field_alias("predictor_info"): json.dumps(self.predictor_info),
                self.get_field_alias("status"): self.status.value,
            }
        )

        if self.predicted_doc is not None:
            record.update(
                {
                    self.get_field_alias("predicted_doc"): json.dumps(
                        self.predicted_doc.export_to_dict()
                    ),
                    self.get_field_alias(
                        "predicted_segmented_pages"
                    ): seg_adapter.dump_json(self.predicted_segmented_pages).decode(
                        "utf-8"
                    ),
                    self.get_field_alias("predicted_pictures"): self.predicted_pictures,
                    self.get_field_alias(
                        "predicted_page_images"
                    ): self.predicted_page_images,
                    self.get_field_alias("original_prediction"): (
                        self.original_prediction
                    ),
                }
            )

        return record

    @model_validator(mode="after")
    def validate_images(self) -> "DatasetRecordWithPrediction":
        # super().validate_images()

        if self.predicted_doc is not None:
            if not len(self.predicted_pictures) and not len(self.predicted_page_images):
                pictures, page_images = self._extract_images(
                    self.predicted_doc,
                    pictures_field_prefix=self.get_field_alias("predicted_pictures"),
                    pages_field_prefix=self.get_field_alias("predicted_page_images"),
                )

                self.predicted_page_images = page_images
                self.predicted_pictures = pictures

        return self

    @model_validator(mode="before")
    @classmethod
    def validate_prediction_record_dict(cls, data: dict):
        info_alias = cls.get_field_alias("predictor_info")
        if info_alias in data and isinstance(data[info_alias], str):
            data[info_alias] = json.loads(data[info_alias])

        timings_alias = cls.get_field_alias("prediction_timings")
        if timings_alias in data and isinstance(data[timings_alias], str):
            # try:
            data[timings_alias] = json.loads(data[timings_alias])
            # except json.JSONDecodeError:
            #    data[timings_alias] = None

        pred_doc_alias = cls.get_field_alias("predicted_doc")
        if pred_doc_alias in data and isinstance(data[pred_doc_alias], str):
            data[pred_doc_alias] = json.loads(data[pred_doc_alias])

        pred_seg_pages_alias = cls.get_field_alias("predicted_segmented_pages")
        if pred_seg_pages_alias in data and isinstance(
            data[pred_seg_pages_alias], (str, bytes)
        ):
            seg_pages_data = data[pred_seg_pages_alias]
            if isinstance(seg_pages_data, bytes):
                seg_pages_data = seg_pages_data.decode("utf-8")
            data[pred_seg_pages_alias] = seg_adapter.validate_json(seg_pages_data)

        pred_page_img_alias = cls.get_field_alias("predicted_page_images")
        if pred_page_img_alias in data:
            for ix, item in enumerate(data[pred_page_img_alias]):
                if isinstance(item, dict):
                    data[pred_page_img_alias][ix] = Features_Image().decode_example(
                        item
                    )

        pred_pic_img_alias = cls.get_field_alias("predicted_pictures")
        if pred_pic_img_alias in data:
            for ix, item in enumerate(data[pred_pic_img_alias]):
                if isinstance(item, dict):
                    data[pred_pic_img_alias][ix] = Features_Image().decode_example(item)

        return data
