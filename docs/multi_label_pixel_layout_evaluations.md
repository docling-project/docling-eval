# Multi-label pixel layout evaluations

## Objectives

We want to compute metrices for the multi-label document layout analysis task.
Each document page undergoes a layout resolution, where each detected object is assigned a bounding box and one or many classes.
The ground truth contains the bounding box and one object class, although in a generalized version the ground truth can also assign multiple classes for the same object.
Everything which is not classified is considered to be the *Background*.

We want to evaluate 2 sets of layout resolutions against each other.
This can be either the ground truth layout resolutions against the prediction layout resolutions, or 2 predictions against each other.
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
- The two evaluated layout resolutions are free to use any classification labels.


## Confusion matrix

The rows of the matrix correspond to the first layout resolution (ground truth or prediction A) and the columns to the second layout resolution.

Each cell (i, j) is the number of pixels that correspond to class i according to the first layout resolution (e.g. ground truth) and to class j according to the second layout resolution.

The exact structure of the confusion matrix and the evaluation metrics that can be derived from it depend on the number of classes in the two layout resolutions.
More specifically we distinguish two cases:
- Case A: Both layout resolutions use the same classification classes.
- Case B: When the classes differ across the layout resolutions.

|                                           | Same classes in LR1/LR2| Different classes in LR1/LR2           |
|---------------------------------------- --|------------------------|----------------------------------------|
|Rows represent                             | LR1 (e.g. GT)          | LR1 (e.g. GT, predictions A)           |
|Columns represent                          | LR2 (e.g. predictions) | LR2 (e.g. predictions B)               |
|Rows/Columns indices                       | background - classes   | background - classes LR1 - classes LR2 |
|Matrix structure when perfect match        | diagonal               | block                                  |
|Location of mis-predictions/mis-matches    | off-diagonal           |                                        |
|Recall/Precision/F1 matrices               | yes                    | yes                                    |
|Background/class-collapsed R/P/F1 matrices | yes                    | yes                                    |
|Recall/Precision/F1 detailed class vectors | yes                    | no                                     |
|Recall/Precision/F1 collapsed class vectors| yes                    | yes                                    |
|

The background is always in index 0.


## Binary representation of the Layout Resolution


## Multi-label classification confusion matrix


## Computation Optimizations

