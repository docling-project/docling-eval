import logging
import math
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, model_serializer

_log = logging.getLogger(__name__)


class LayoutResolution(BaseModel):
    r"""Single bbox resolution"""

    category_id: int

    # bbox coords: (x1, y1, x2, y2) with the origin(0, 0) at the top, left corner, no normalization
    bbox: list[float]


class MultiLabelMatrixAggMetrics(BaseModel):
    classes_precision: dict[str, float]
    classes_recall: dict[str, float]
    classes_f1: dict[str, float]

    classes_precision_mean: float
    classes_recall_mean: float
    classes_f1_mean: float


class MultiLabelMatrixMetrics(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    confusion_matrix: np.ndarray
    precision_matrix: np.ndarray
    recall_matrix: np.ndarray
    f1_matrix: np.ndarray

    agg_metrics: MultiLabelMatrixAggMetrics

    @model_serializer(mode="wrap")
    def serialize_model(self, serializer: Any) -> dict:
        data = serializer(self)
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, np.ndarray):
                data[field_name] = field_value.tolist()
        return data


class MultiLabelMatrixEvaluation(BaseModel):
    detailed_metrics: MultiLabelMatrixMetrics
    colapsed_metrics: Optional[MultiLabelMatrixMetrics] = None


def unpackbits(x: np.ndarray, num_bits: int):
    r"""
    Unpack num_bits bits of each element of the numpy array x
    The number of bits defines how many bits we will take from x to unpack.
    """
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2 ** np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
    return (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])


class MultiLabelConfusionMatrix:
    r""" """

    DETAILED_METRICS_KEY = "detailed_classes"
    COLAPSED_METRICS_KEY = "colapsed_classes"
    ALL_COLAPSED_CLASSES_NAME = "all_classes"

    def __init__(
        self,
        validation_mode: str = "disabled",
    ):
        r"""
        The validation mode can be one of: ["disabled", "log", "raise"]
        """
        self._validation_mode = validation_mode

    def make_binary_representation(
        self,
        image_width: int,
        image_height: int,
        resolutions: list[LayoutResolution],
        set_background: bool = True,
    ) -> np.ndarray:
        r"""
        Create a numpy matrix with the binary representation of the layout resolutions
        Each pixel is represented as one uint64 integer, where the 1-bit flags presence of a class.

        Parameters
        ----------
        set_background: Assign the value 1 to all pixels that still have a zero value in the end

        Returns
        -------
        np.ndarray with the binary representation of the resolutions. Dims are equal to the image size
        """
        # Initialize the representation matrix with 0
        matrix = np.zeros(
            (
                image_height,
                image_width,
            ),
            dtype=np.uint64,
        )

        for res in resolutions:
            x1 = res.bbox[0]
            y1 = res.bbox[1]
            x2 = res.bbox[2]
            y2 = res.bbox[3]
            x_begin = math.floor(x1)
            x_end = math.ceil(x2)
            y_begin = math.floor(y1)
            y_end = math.ceil(y2)

            cat_id = res.category_id
            bit_index = np.uint64(1 << cat_id)
            matrix[y_begin:y_end, x_begin:x_end] |= bit_index

        # Set the background class (binary 1) if there is no other class set
        if set_background:
            matrix[matrix == 0] = 1

        return matrix

    def generate_confusion_matrix(
        self,
        gt: np.ndarray,
        preds: np.ndarray,
        canonical_categories: list[int],
    ) -> np.ndarray:
        r"""
        Create the confusion matrix for multi-label predictions.
        The returned matrix can be used to compute precision, recall.

        In a perfect prediction the matrix should be diagonal. All elements outside of the main
        diagonal indicate prediction errors and contribute to penalties in the calculation of
        precision, recall.

        Inspired by "Multi-label classifier performance evaluation with confusion matrix"
        https://csitcp.org/paper/10/108csit01.pdf

        Returns
        -------
        np.ndarray [num_categories + 1, num_categories + 1]. The +1 is for the background class
        """
        img_height, img_width = gt.shape
        num_categories = len(canonical_categories)

        # confusion_matrix: [num_categories, num_categories]
        confusion_matrix = np.zeros((num_categories, num_categories), dtype=float)
        eye = np.eye(num_categories)

        ############################################################################################
        # Case 1: Perfect prediction
        #

        # [img_height, img_width]
        selections_case1 = gt == preds

        # 1. I = np.eye(num_categories): [num_categories, num_categories]
        # 2. U = unpackbits(gt[selections_case1], num_categories)]: [k, num_categories]
        # 3. C = U[:, None, :] * I[None, :, :]
        # U = unpackbits(gt[selections_case1], num_categories)
        # C = U[:, None, :] * eye[None, :, :]

        # case1_contributions: [num_pixels_with_perfect_preds, num_categories, num_categories]
        case1_contributions = (
            unpackbits(gt[selections_case1], num_categories)[:, None, :]
            * eye[None, :, :]
        )

        # print("Case1 contributions:")
        # print(case1_contributions)

        # Validate the contributions
        self._validate_contributions(gt[selections_case1], case1_contributions, "Case1")

        confusion_matrix += np.sum(case1_contributions, axis=0)

        ############################################################################################
        # Case 2: Prediction has all GT plus extra mistakes
        #

        # Filter out the non-perfect predictions to take the ones where preds contain all GT bits
        # [img_height, img_width]
        selections_case2 = ~selections_case1
        selections_case2[selections_case2 == True] = (
            gt[selections_case2] & preds[selections_case2] == gt[selections_case2]
        )

        # [num_pixels_with_extra_preds, num_categories]
        case2_preds_gt_intersection = preds[selections_case2] & gt[selections_case2]
        case2_preds_gt_intersection = unpackbits(
            case2_preds_gt_intersection, num_categories
        )

        # [num_pixels_with_extra_preds,]
        case2_preds_pixels = preds[selections_case2]
        if len(case2_preds_pixels) > 0:
            # [num_pixels_with_extra_preds, num_categories]
            case2_preds_gt_diff = (
                case2_preds_pixels ^ gt[selections_case2]
            ) & case2_preds_pixels
            case2_preds_gt_diff = unpackbits(case2_preds_gt_diff, num_categories)

            # [num_pixels_with_extra_preds, num_categories, num_categories]
            case2_penalty = (
                case2_preds_gt_intersection[:, :, None]
                * case2_preds_gt_diff[:, None, :]
            )

            # [num_pixels_with_extra_preds, num_categories, num_categories]
            case2_gt_diagonals = (
                unpackbits(gt[selections_case2], num_categories)[:, None, :]
                * eye[None, :, :]
            )

            # [num_pixels_with_extra_preds,]
            case2_gt_multiplier = np.bitwise_count(gt[selections_case2])

            # [num_pixels_with_extra_preds,]
            case2_preds_divider = np.bitwise_count(case2_preds_pixels)

            # [num_pixels_with_extra_preds, num_categories, num_categories]
            case2_gain = case2_gt_multiplier[:, None, None] * case2_gt_diagonals

            # [num_pixels_with_extra_preds, num_categories, num_categories]
            case2_contributions = (case2_penalty + case2_gain) / case2_preds_divider[
                :, None, None
            ]
            # print("Case2 contributions:")
            # print(case2_contributions)

            # Validate the contributions
            self._validate_contributions(
                gt[selections_case2], case2_contributions, "Case2"
            )

            confusion_matrix += np.sum(case2_contributions, axis=0)

        ############################################################################################
        # Case 3: GT has more labels than preds
        # NOTICE: This case NEVER happens for us because our GT has only 1 label
        #

        # [img_height, img_width]
        selections_case3 = ~selections_case1
        selections_case3[selections_case3 == True] = (
            gt[selections_case3] | preds[selections_case3] == gt[selections_case3]
        )

        # [num_pixels_with_additional_gt_labels,]
        case3_preds_pixels = preds[selections_case3]
        if len(case3_preds_pixels) > 0:
            # [num_pixels_with_additional_gt_labels, num_categories]
            case3_gt_preds_diff = (case3_preds_pixels ^ gt[selections_case3]) & gt[
                selections_case3
            ]
            case3_gt_preds_diff = unpackbits(case3_gt_preds_diff, num_categories)

            # [num_pixels_with_additional_gt_labels,]
            case3_preds_divider = np.bitwise_count(case3_preds_pixels)

            # [num_pixels_with_additional_gt_labels, num_categories, num_categories]
            case3_preds_diagonals = (
                unpackbits(case3_preds_pixels, num_categories)[:, None, :]
                * eye[None, :, :]
            )

            # [num_pixels_with_additional_gt_labels, num_categories]
            case3_preds = unpackbits(case3_preds_pixels, num_categories)

            # [num_pixels_with_additional_gt_labels, num_categories, num_categories]
            case3_penalty = (
                case3_gt_preds_diff[:, :, None] * case3_preds[:, None, :]
            ) / case3_preds_divider[:, None, None]

            # [num_pixels_with_additional_gt_labels, num_categories, num_categories]
            case3_contributions = case3_penalty + case3_preds_diagonals
            # print("Case3 contributions:")
            # print(case3_contributions)

            # Validate the contributions
            self._validate_contributions(
                gt[selections_case3], case3_contributions, "Case3"
            )

            confusion_matrix += np.sum(case3_contributions, axis=0)

        ############################################################################################
        # Case 4: Both GT and preds contain labels that are missing from the other one
        #

        # [img_height, img_width]
        general_diff = gt ^ preds
        selections_case4 = np.logical_and(
            (general_diff & gt) > 0, (general_diff & preds) > 0
        )

        # [num_pixels_with_mutual_gt_pred_deltas, num_categories]
        case4_gt_preds_diff = (preds[selections_case4] ^ gt[selections_case4]) & gt[
            selections_case4
        ]
        case4_gt_preds_diff = unpackbits(case4_gt_preds_diff, num_categories)

        # [num_pixels_with_mutual_gt_pred_deltas, num_categories]
        case4_preds_gt_diff = (preds[selections_case4] ^ gt[selections_case4]) & preds[
            selections_case4
        ]
        if len(case4_preds_gt_diff) > 0:
            case4_divider = np.bitwise_count(case4_preds_gt_diff)
            case4_preds_gt_diff = unpackbits(case4_preds_gt_diff, num_categories)

            # [num_pixels_with_mutual_gt_pred_deltas, num_categories]
            case4_preds_gt_intersection = preds[selections_case4] & gt[selections_case4]

            # [num_pixels_with_mutual_gt_pred_deltas, num_categories, num_categories]
            case4_preds_gt_intersection_diagonals = (
                unpackbits(case4_preds_gt_intersection, num_categories)[:, None, :]
                * eye[None, :, :]
            )

            # [num_pixels_with_mutual_gt_pred_deltas, num_categories, num_categories]
            case4_penalty = (
                case4_gt_preds_diff[:, :, None] * case4_preds_gt_diff[:, None, :]
            ) / case4_divider[:, None, None]
            case4_contributions = case4_penalty + case4_preds_gt_intersection_diagonals
            # print("Case4 contributions:")
            # print(case4_contributions)

            # Validate the contributions
            self._validate_contributions(
                gt[selections_case4], case4_contributions, "Case4"
            )

            confusion_matrix += np.sum(case4_contributions, axis=0)

        return confusion_matrix

    def compute_metrics(
        self,
        confusion_matrix: np.ndarray,
        class_names: dict[int, str],
        colapse_non_bg: bool = False,
    ) -> MultiLabelMatrixEvaluation:
        r"""
        Parameters:
        -----------
        confusion_matrix: np.ndarray[num_categories + 1, num_categories + 1]
        class_names: Mapping from class_id to class_names
        colapse_non_bg: Colapse all classes except of the first one that is assumed to be the BG

        Returns
        --------

        """
        # Compute metrics on the full confusion matrix
        detailed_metrics = self._compute_matrix_metrics(confusion_matrix, class_names)
        evaluation = MultiLabelMatrixEvaluation(detailed_metrics=detailed_metrics)

        if colapse_non_bg:
            # Colapse the classes except the background and compute metrics again
            colapsed_confusion_matrix = np.asarray(
                [
                    [confusion_matrix[0, 0], np.sum(confusion_matrix[0, 1:])],
                    [np.sum(confusion_matrix[1:, 0]), np.sum(confusion_matrix[1:, 1:])],
                ]
            )
            colapsed_class_names = {
                0: class_names[0],
                1: MultiLabelConfusionMatrix.ALL_COLAPSED_CLASSES_NAME,
            }
            colapsed_metrics = self._compute_matrix_metrics(
                colapsed_confusion_matrix,
                colapsed_class_names,
            )
            evaluation.colapsed_metrics = colapsed_metrics

        return evaluation

    def _compute_matrix_metrics(
        self,
        confusion_matrix: np.ndarray,
        class_names: dict[int, str],
    ) -> MultiLabelMatrixMetrics:
        r""" """
        col_sums = np.sum(confusion_matrix, axis=0)
        row_sums = np.sum(confusion_matrix, axis=1)

        # Compute precision_matrix and recall_matrix
        precision_matrix = np.divide(
            confusion_matrix,
            col_sums[None, :],
            out=np.zeros(confusion_matrix.shape),
            where=col_sums[None, :] != 0,
        )
        recall_matrix = np.divide(
            confusion_matrix,
            row_sums[:, None],
            out=np.zeros(confusion_matrix.shape),
            where=row_sums[:, None] != 0,
        )
        # Compute the f1 matrix element-wise
        f1_matrix_nom = 2 * precision_matrix * recall_matrix
        f1_matrix_denom = precision_matrix + recall_matrix
        f1_matrix = np.divide(
            f1_matrix_nom,
            f1_matrix_denom,
            out=np.zeros(confusion_matrix.shape),
            where=f1_matrix_denom != 0,
        )

        # Extract diagonal vectors
        precision = np.diag(precision_matrix)
        recall = np.diag(recall_matrix)
        f1 = np.diag(f1_matrix)
        precision_mean = np.average(precision)
        recall_mean = np.average(recall)
        f1_mean = np.average(f1)

        # Generate dicts with metrics per class name
        def get_class_name(class_id: int) -> str:
            return class_names[class_id]

        def array_to_dict(a: np.ndarray) -> dict[str, float]:
            a_dict = {get_class_name(i): float(x) for i, x in enumerate(a)}
            return a_dict

        precision_dict = array_to_dict(precision)
        recall_dict = array_to_dict(recall)
        f1_dict = array_to_dict(f1)

        agg_metrics = MultiLabelMatrixAggMetrics(
            classes_precision=precision_dict,
            classes_recall=recall_dict,
            classes_f1=f1_dict,
            classes_precision_mean=float(precision_mean),
            classes_recall_mean=float(recall_mean),
            classes_f1_mean=float(f1_mean),
        )

        metrics = MultiLabelMatrixMetrics(
            confusion_matrix=confusion_matrix,
            precision_matrix=precision_matrix,
            recall_matrix=recall_matrix,
            f1_matrix=f1_matrix,
            agg_metrics=agg_metrics,
        )
        return metrics

    def _validate_contributions(
        self,
        selected_gt: np.ndarray,
        contributions: np.ndarray,
        info: str,
    ):
        r"""
        Each contribution has the properties:
        1. The sum of each row corresponding to labels in GT is equal to one.
        2. The sum of all elements is equal to cardinality of GT.

        The validation is controled by self._validation_mode:
        - "disabled": No validation
        - "raise": Raise a ValueError
        - "log": Write an error log message

        Parameters:
        -----------
        selected_gt: np.ndarray  1D array with size=selected_pixels, each pixel is a uint64 encoding
        contributions: np.ndarray  [selected_pixels, num_classes, num_classes]
        """
        if self._validation_mode == "disabled":
            return

        contributions_shape = contributions.shape
        if len(contributions_shape) != 3:
            return

        num_categories = contributions_shape[1]

        selected_pixels = np.prod(selected_gt.shape)
        if selected_pixels != contributions_shape[0]:
            self._handle_error(f"{info}: Wrong contributions dimension")

        # Row sum check
        row_sum = np.sum(contributions, axis=2)
        expected_row_sum = unpackbits(selected_gt, num_categories)
        if not np.all(row_sum == expected_row_sum):
            self._handle_error(f"{info}: Wrong contributions row sums")

        # Full sum check
        full_sum = np.sum(row_sum)
        expected_full_sum = np.sum(np.bitwise_count(selected_gt))
        if full_sum != expected_full_sum:
            self._handle_error(f"{info}: Wrong contributions full sums")

    def _handle_error(self, msg: str):
        if self._validation_mode == "raise":
            raise ValueError(msg)
        else:
            _log.error(msg)
