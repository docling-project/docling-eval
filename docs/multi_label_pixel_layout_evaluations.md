# Multi-label pixel layout evaluations

## Objectives

We want to evaluate the multi-label document layout analysis task.
The layout resolution for each document page consists of the bounding boxes of each detected item and one or many classes.
The ground truth contains the bounding box and one class, although in a generalized version of the ground truth can also assign multiple classes to each item.
Everything which is not classified is considered to be the *Background*.

We want to evaluate two sets of layout resolutions against each other.
This can be either the ground truth versus a model prediction or the evaluation across two model predictions.
We name those layout resolutions as LR1 and LR2.

We also want to solve this evaluation task under the following conditions:

- The evaluations take place at the pixel level.
- The evaluation of each document page produces a square confusion matrix [n, n], which is the basis to compute:
  - Document-level confusion matrix.
  - Recall/Precision/F1 matrices per page and document.
  - Recall/Precision/F1 vectors per class.
  - Collapsed recall/precision/F1 matrices which contain only the background and the non-background classes.

Additionally we have the following freedoms:

- We do not require the predictions to contain any confidence scores but only bounding boxes and object classes.
- The two evaluated layout resolutions are free to use any classification taxonomies.


## Confusion matrix structure

The rows of the matrix correspond to the first layout resolution (ground truth or prediction A) and the columns to the second layout resolution.

Each cell (i, j) is the number of pixels that have been assigned to class i according to the first layout resolution (e.g. ground truth)
and to class j according to the second layout resolution.

The structure of the confusion matrix depends on the classification taxonomies used by the two layout resolutions. 
More specifically we distinguish two cases:
- Case A: Both layout resolutions use the same classification taxonomy.
- Case B: The taxonomies differ across the layout resolutions.

<!--------------------------------------------------------------------------------------------->
TODO: Make an illustration to show the differences in the confusion matrix structures

The following table provides some insight on the properties of the confusion matrix and the derived metrics for each case:


|                                           | Same class taxonomy    | Different class taxonomies        |
|-------------------------------------------|------------------------|-----------------------------------|
|Confusion matrix rows represent            | LR1 (e.g. GT)          | LR1 (e.g. GT, predictions A)      |
|Confusion matrix columns represent         | LR2 (e.g. predictions) | LR2 (e.g. predictions B)          |
|Row/column index of the Background class   | (0, 0)                 | (0, 0)                            |
|Rows/Columns after the Background class    | common taxonomy        | taxonomy of LR1 - taxonomy of LR2 |
|Matrix structure when perfect match        | diagonal               | block                             |
|Location of mis-predictions/mis-matches    | off-diagonal           |                                   |
|Recall/Precision/F1 matrices               | yes                    | yes                               |
|Background/class-collapsed R/P/F1 matrices | yes                    | yes                               |
|Recall/Precision/F1 detailed class vectors | yes                    | no                                |
|Recall/Precision/F1 collapsed class vectors| yes                    | yes                               |
|                                           |                        |                                   |


Table 1: Properties of the confusion matrix and its derivatives across different taxonomy schemes


<!--------------------------------------------------------------------------------------------->


## Computation of the confusion matrix and its derivatives

The computation of the multi-label classification matrix is based on the papers:

- [Multi-Label Classifier Performance Evaluation with Confusion Matrix.](https://csitcp.org/paper/10/108csit01.pdf)
- [Comments on "MLCM: Multi-Label Confusion Matrix".](https://www.academia.edu/121504684/Comments_on_MLCM_Multi_Label_Confusion_Matrix)

The papers describe how to build the confusion matrix for the multi-label classification problem under the assumptions:

- The rows represent the ground truth and the columns the predictions.
- Both ground-truth and predictions use the same classes.
- The ground truth may assign more than one classes to the same object.

A _contribution matrix_ is computed for each pair of ground-truth / prediction samples and the sum of them is the _confusion matrix_ of the entire dataset.

Each contribution matrix is computed according to an algorithm that distinguishes 4 cases:

- Case 1: Prediction and GT are a perfect match.
- Case 2: Prediction is a superset of the GT classes (over-prediction).
- Case 3: Prediction is a subset of the GT classes (under-prediction).
- Case 4: Prediction and GT have some partial overlap and some diff (diff-prediction).

For each of those cases the contributions to the confusion matrix can be seen as "gains" that go to the diagonal cells and "penalties" that go to the off-diagonal cells.
In case 1 the contributions are only gains and their value equals to the number of page items.
For the other cases the gains have been penalized by the mis-predictions and both gains and penalties have fractional values.
For example in case of "over-prediction", if the classifier has predicted 3 classes (a, b, c) and the ground truth is (a, b),
the contribution is a gain of 2/3 for the diagonal cells (a, a), (b, b) because 2 out of 3 predictions are correct
and a penalty of 1/3 for the off-diagonal cells (a, c) and (b, c) because the prediction c is wrong.

The contribution matrix for each dataset sample has the following properties:
- All rows without ground truth and all columns without predictions are zero.
- The sum of each non-zero row is 1.
- The sum of all cells equals to the number of GT classes for that sample.

Dividing the dataset-wide confusion matrix by each row-sum gives us the _recall matrix_
and dividing by each column-sum provides the _precision matrix_.
The diagonal of the recall/precision matrices are the recall/precision vectors for the classification classes.

The _F1 matrix_ is the harmonic mean of the precision (P) and recall (R) matrices and is computed as (2 * P * R) / (P + R).


## Pixel-level multi-label confusion matrix

We consider each page pixel as a dataset sample and we compute a contribution matrix according to the previous algorithm.
Summing up the pixel-level contributions provides the confusion matrix for each page
and the sum of all page-level confusion matrices provides the confusion matrix for the entire dataset.

Additionally we compute 2x2 "abstractions" of the confusion matrices that contain only the
"Background" and the non-Background classes collapsed as one:

TODO: Make an illustration to show how the confusion matrix is collapsed


|                | Background | non-Background |
|----------------|------------|----------------|
| Background     | cell(0,0)  | sum(0, 1:)     |
| non-Background | sum(1:, 0) | sum(1:, 1:)    |


Table 2: Collapsed matrix computed for Background and non-Background classes

The collapsed confusion matrix and its derivatives - collapsed recall/precision/F1 -,
allow the evaluation across layout resolutions with different class taxonomies.


## Implementation

TODO: Make an illustration how the bit-packed encoding works.

We use a bit‑packed encoding to represent multi‑label layout resolutions for up to 63 classes plus the Background class.
Each pixel is stored as a single 64‑bit unsigned integer; the i‑th class is encoded by setting bit i.
The background occupies bit 0.

This compact representation enables a vectorized implementation using numpy bitwise and linear algebra operations.
Thanks to instruction-level parallelism, we can compute multiple pixel-level contribution matrices at once.

Each pair of binary page layout representations is then compressed by counting the distinct pixel-pairs.
Only the contribution matrices of the unique pixel-pairs need to be computed.
The page-level confusion matrix is obtained as the weighted sum of the computed contribution matrices
multiplied by the number of appearances of each unique pixel-pair.
Because the number of unique pixel‑pairs is significantly less than the total number of pixels,
 this approach dramatically reduces the computational overhead.

Finally, since pages are independent, the computation of each page‑level confusion matrix can be
also parallelized.


## Discussion

TODO 

