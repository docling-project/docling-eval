# Evaluate DocTag files using the FileProvider

Assumptions:

1. We have the ground truth data as an HF parquet dataset.
2. We have a directory with doctag files with the naming convention: `<doc_id>.dt`, where `<doc_id>` must match the `document_id` column in the GT dataset.
3. The same images from the GT will be used together with the `dt` files to construct the prediction documents.


Overview:

1. Run `create-eval` to package the file predictons as a dataset in HF parquet format.
   - Pass the directory of the `dt` files in `--file-source-path`.
   - Give an `--output-dir` to save the produced dataset.
2. Run evaluations for each modality.
   - Pass exactly the same path for `--output-dir` as in step 1.
3. Run visualizations for each evaluation.
   - Pass exactly the same path for `--output-dir` as in step 1.


## Example

Assuming:

1. The GT in HF parquet format is in: `/Users/nli/data/DocLayNetV2/gt_dataset`.
2. The `dt` files are in: `/Users/nli/data/SmolDocling_eval_data/SmolDocling_250M_DT_v2.1_gd_checkpoint_2200/dt`
3. We want to run evaluations for the `layout` modality with visualizations.


Samples of the doctag filenames:

```
05e45d1c6fd3c28a056038b8495f32d76ac8b13a8f917c0771b37fae8e5b62b9.dt
541db2953d542412c66e6bcdad9aed8a1fc0ab4f5ca3981d013a999bc050b998.dt
c7fc21b6389b304c60df5c5d8d354fc9d94b3a462cc37216fabd186b8491df55.dt
6210509094e90aeb5c7f1f5dca17e89185b821740c2805fa087778bcc22fe84c.dt
```

How to evaluate:

1. Generate the HF parquet dataset out of the `dt` files.

```bash
uv run docling-eval \
    create-eval \
    --benchmark DocLayNetV2 \
    --gt-dir "/Users/nli/data/DocLayNetV2/gt_dataset" \
    --prediction-provider File \
    --file-prediction-format doctags \
    --file-source-path "/Users/nli/data/SmolDocling_eval_data/SmolDocling_250M_DT_v2.1_gd_checkpoint_2200/dt" \
    --output-dir "/Users/nli/data/SmolDocling_eval_data/evals"
```

The generated HF parquet dataset of the predictions is in: `/Users/nli/data/SmolDocling_eval_data/evals/eval_dataset`

Also it generates visualizations of the predictions inside: `/Users/nli/data/SmolDocling_eval_data/evals/eval_dataset/visualizations`


2. Use the predictions dataset to make evaluations for the `layout` modality:

```bash
uv run docling-eval \
    evaluate \
    --modality layout \
    --benchmark DocLayNetV2 \
    --output-dir "/Users/nli/data/SmolDocling_eval_data/evals"
```
The generated evaluations will be placed in: `/Users/nli/data/SmolDocling_eval_data/evals/evaluations/layout`


3. Generate visualizations for the evaluations:

```bash
uv run docling-eval \
    visualize \
    --modality layout \
    --benchmark DocLayNetV2 \
    --output-dir "/Users/nli/data/SmolDocling_eval_data/evals"
```

The evaluation visualizations will be in: `/Users/nli/data/SmolDocling_eval_data/evals/evaluations/layout`
