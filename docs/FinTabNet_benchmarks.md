# FinTabNet Benchmarks

Create FinTabNet evaluation datasets. This downloads from Huggingface the [FinTabNet_OTSL dataset](https://huggingface.co/datasets/ds4sd/FinTabNet_OTSL)) and runs the evaluations for tableformer using the first 1000 samples.

```sh
# Make the ground-truth
docling_eval create-gt --benchmark FinTabNet --output-dir ./benchmarks/FinTabNet/ 

# Make predictions for tables.
docling_eval create-eval \
  --benchmark FinTabNet \
  --output-dir ./benchmarks/FinTabNet/ \
  --end-index 1000 \
  --prediction-provider tableformer # use tableformer predictions only
```

## Tableformer Evaluation

Create the evaluation report:

```sh
docling_eval evaluate \
  --modality table_structure \
  --benchmark FinTabNet \
  --output-dir ./benchmarks/FinTabNet/ 
```

[Tableformer evaluation json](evaluations/FinTabNet/evaluation_FinTabNet_tableformer.json)

Visualize the report:

```sh
docling_eval visualize \
  --modality table_structure \
  --benchmark FinTabNet \
  --output-dir ./benchmarks/FinTabNet/ 
```

![TEDS plot](evaluations/FinTabNet/evaluation_FinTabNet_tableformer-delta_row_col.png)

![TEDS struct only plot](evaluations/FinTabNet/evaluation_FinTabNet_tableformer_TEDS_struct-only.png)

[TEDS struct only report](evaluations/FinTabNet/evaluation_FinTabNet_tableformer_TEDS_struct-only.txt)

![TEDS struct with text plot](evaluations/FinTabNet/evaluation_FinTabNet_tableformer_TEDS_struct-with-text.png)

[TEDS struct with text report](evaluations/FinTabNet/evaluation_FinTabNet_tableformer_TEDS_struct-with-text.txt)
