# Pub1M Benchmarks

Create Pub1M evaluation datasets. This downloads from Huggingface the [Pub1M_OTSL](https://huggingface.co/datasets/ds4sd/Pub1M_OTSL) and runs the evaluations for TableFormer using the first 1000 samples. 

```sh
# Make the ground-truth
docling-eval create-gt --benchmark Pub1M --output-dir ./benchmarks/Pub1M/ 

# Make predictions for tables.
docling-eval create-eval \
  --benchmark DPBench \
  --output-dir ./benchmarks/Pub1M/ \
  --end-index 1000 \
  --prediction-provider TableFormer # use tableformer predictions only
```

## Tableformer Evaluation

Create the evaluation report:

```sh
docling-eval evaluate \
  --modality table_structure \
  --benchmark Pub1M \
  --output-dir ./benchmarks/Pub1M/ 
```

[Tableformer evaluation json](evaluations/Pub1M/evaluation_Pub1M_tableformer.json)

Visualize the report:

```sh
docling-eval visualize \
  --modality table_structure \
  --benchmark Pub1M \
  --output-dir ./benchmarks/Pub1M/ 
```

![TEDS plot](evaluations/Pub1M/evaluation_Pub1M_tableformer-delta_row_col.png)

![TEDS struct only plot](evaluations/Pub1M/evaluation_Pub1M_tableformer_TEDS_struct-only.png)

[TEDS struct only report](evaluations/Pub1M/evaluation_Pub1M_tableformer_TEDS_struct-only.txt)

![TEDS struct with text plot](evaluations/Pub1M/evaluation_Pub1M_tableformer_TEDS_struct-with-text.png)

[TEDS struct with text report](evaluations/Pub1M/evaluation_Pub1M_tableformer_TEDS_struct-with-text.txt)
