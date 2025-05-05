import copy
import logging
import platform
from typing import Dict, List, Optional, Set

from docling.datamodel.base_models import InputFormat
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, FormatOption
from docling_core.types.doc import DocItemLabel
from pydantic import TypeAdapter

from docling_eval.datamodels.dataset_record import (
    DatasetRecord,
    DatasetRecordWithPrediction,
)
from docling_eval.datamodels.types import (
    EvaluationModality,
    PredictionFormats,
    PredictionProviderType,
)
from docling_eval.prediction_providers.base_prediction_provider import (
    BasePredictionProvider,
)
from docling_eval.utils.utils import docling_version, get_package_version

_log = logging.getLogger(__name__)


class DoclingPredictionProvider(BasePredictionProvider):
    """
    Prediction provider that uses Docling document converter.

    This provider converts documents using the Docling document converter
    with specified format options.
    """

    prediction_provider_type: PredictionProviderType = PredictionProviderType.DOCLING

    prediction_modalities: List[EvaluationModality] = [
        EvaluationModality.LAYOUT,
        EvaluationModality.TABLE_STRUCTURE,
        EvaluationModality.READING_ORDER,
        EvaluationModality.MARKDOWN_TEXT,
        EvaluationModality.BBOXES_TEXT,
    ]

    def __init__(
        self,
        format_options: Optional[Dict[InputFormat, FormatOption]] = None,
        do_visualization: bool = False,
        ignore_missing_predictions: bool = True,
        true_labels: Optional[Set[DocItemLabel]] = None,
        pred_labels: Optional[Set[DocItemLabel]] = None,
        profile_pipeline_timings: bool = True,
    ):
        """
        Initialize the Docling prediction provider.

        Args:
            format_options: Dictionary mapping input formats to format options
            do_visualization: Whether to generate visualizations
            ignore_missing_predictions: Whether to ignore missing predictions
            true_labels: Set of DocItemLabel to use for ground truth visualization
            pred_labels: Set of DocItemLabel to use for prediction visualization
        """

        super().__init__(
            do_visualization=do_visualization,
            ignore_missing_predictions=ignore_missing_predictions,
            true_labels=true_labels,
            pred_labels=pred_labels,
        )

        # Enable the profiling to measure the time spent
        settings.debug.profile_pipeline_timings = profile_pipeline_timings
        _log.info(f"profile_pipeline_timings: {profile_pipeline_timings}")

        self.doc_converter = DocumentConverter(format_options=format_options)

    @property
    def prediction_format(self) -> PredictionFormats:
        """Get the prediction format."""
        return PredictionFormats.DOCLING_DOCUMENT

    def predict(self, record: DatasetRecord) -> DatasetRecordWithPrediction:
        """
        Generate a prediction by converting the document.

        Args:
            record: Input dataset record

        Returns:
            Dataset record with prediction

        Raises:
            RuntimeError: If original document stream is not available
        """
        if record.original is None:
            raise RuntimeError(
                "Stream must be given for docling prediction provider to work."
            )

        # Convert the document
        res = self.doc_converter.convert(copy.deepcopy(record.original))

        # Create prediction record
        pred_record = self.create_dataset_record_with_prediction(
            record, res.document, None, res.timings
        )
        pred_record.predicted_segmented_pages = {
            p.page_no: p.parsed_page for p in res.pages if p.parsed_page is not None
        }
        pred_record.status = res.status

        return pred_record

    def info(self) -> Dict:
        """Get information about the prediction provider."""

        result = {
            "asset": PredictionProviderType.DOCLING,
            "version": docling_version(),
            "package_versions": {
                "docling": get_package_version("docling"),
                "docling-ibm-models": get_package_version("docling-ibm-models"),
                "docling-core": get_package_version("docling-core"),
            },
            "environment": {
                "system": platform.system(),
                "node_name": platform.node(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "options": {
                k: {
                    "pipeline_class": v.pipeline_cls.__name__,
                    "pipeline_options": (
                        v.pipeline_options.model_dump(
                            mode="json", exclude_defaults=True
                        )
                        if v.pipeline_options is not None
                        else None  # Parquet might not like empty dicts!
                    ),
                }
                for k, v in self.doc_converter.format_to_options.items()
                if k in [InputFormat.PDF, InputFormat.IMAGE]
            },
        }
        return result
