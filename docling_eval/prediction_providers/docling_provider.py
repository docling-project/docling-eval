import copy
from typing import Dict, Optional, Set

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, FormatOption
from docling_core.types.doc import DocItemLabel

from docling_eval.datamodels.dataset_record import (
    DatasetRecord,
    DatasetRecordWithPrediction,
)
from docling_eval.datamodels.types import PredictionFormats
from docling_eval.prediction_providers.base_prediction_provider import (
    BasePredictionProvider,
)
from docling_eval.utils.utils import docling_version


class DoclingPredictionProvider(BasePredictionProvider):
    """
    Prediction provider that uses Docling document converter.

    This provider converts documents using the Docling document converter
    with specified format options.
    """

    def __init__(
        self,
        format_options: Optional[Dict[InputFormat, FormatOption]] = None,
        do_visualization: bool = False,
        ignore_missing_predictions: bool = True,
        true_labels: Optional[Set[DocItemLabel]] = None,
        pred_labels: Optional[Set[DocItemLabel]] = None,
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
            record,
            res.document,
            None,
        )
        pred_record.status = res.status

        return pred_record

    def info(self) -> Dict:
        """Get information about the prediction provider."""
        return {"asset": "Docling", "version": docling_version()}
