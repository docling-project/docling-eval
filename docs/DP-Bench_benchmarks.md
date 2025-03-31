# DP-Bench Benchmarks

[DP-Bench on HuggingFace](https://huggingface.co/datasets/upstage/dp-bench)

Create DPBench evaluation datasets:

```sh
# Make the ground-truth
docling_eval create-gt --benchmark DPBench --output-dir ./benchmarks/DPBench/ 

# Make predictions for different modalities.
docling_eval create-eval \
  --modality end-to-end \
  --benchmark DPBench \
  --output-dir ./benchmarks/DPBench/ \
  --prediction-provider docling # use full-document predictions from docling
  
docling_eval create-eval \
  --modality table_structure \
  --benchmark DPBench \
  --output-dir ./benchmarks/DPBench/ \
  --prediction-provider tableformer # use tableformer predictions only
```

## Layout Evaluation

Create the evaluation report:

```sh
docling_eval evaluate \
  --modality layout \
  --benchmark DPBench \
  --output-dir ./scratch/DPBench/ 

```

[Layout evaluation json](evaluations/DPBench/evaluation_DPBench_layout.json)

Visualize the report:

```sh
docling_eval visualize \
  --modality layout \
  --benchmark DPBench \
  --output-dir ./scratch/DPBench/ 
```

[mAP[0.5:0.95] report](evaluations/DPBench/evaluation_DPBench_layout_mAP_0.5_0.95.txt)

![mAP[0.5:0.95] plot](evaluations/DPBench/evaluation_DPBench_layout_mAP_0.5_0.95.png)


## TableFormer Evaluation

Create the evaluation report:

```sh
docling_eval evaluate \
  --modality table_structure \
  --benchmark DPBench \
  --output-dir ./scratch/DPBench/ 
```


Visualize the report:

[Tableformer evaluation json](evaluations/DPBench/evaluation_DPBench_tableformer.json)

Visualize the report:

```sh
docling_eval visualize \
  --modality table_structure \
  --benchmark DPBench \
  --output-dir ./scratch/DPBench/ 
```

![TEDS plot](evaluations/DPBench/evaluation_DPBench_tableformer-delta_row_col.png)

![TEDS struct only plot](evaluations/DPBench/evaluation_DPBench_tableformer_TEDS_struct-only.png)

[TEDS struct only report](evaluations/DPBench/evaluation_DPBench_tableformer_TEDS_struct-only.txt)

![TEDS struct with text plot](evaluations/DPBench/evaluation_DPBench_tableformer_TEDS_struct-with-text.png)

[TEDS struct with text report](evaluations/DPBench/evaluation_DPBench_tableformer_TEDS_struct-with-text.txt)


## Reading order Evaluation

Create the evaluation report:

```sh
docling_eval evaluate \
  --modality reading_order \
  --benchmark DPBench \
  --output-dir ./scratch/DPBench/ 
```

[Reading order json](evaluations/DPBench/evaluation_DPBench_reading_order.json)

Visualize the report:

```sh
docling_eval visualize \
  --modality reading_order \
  --benchmark DPBench \
  --output-dir ./scratch/DPBench/ 
```

![ARD plot](evaluations/DPBench/evaluation_DPBench_reading_order_ARD_norm.png)

[ARD report](evaluations/DPBench/evaluation_DPBench_reading_order_ARD_norm.txt)

![Weighted ARD plot](evaluations/DPBench/evaluation_DPBench_reading_order_weighted_ARD.png)

[Weighted ARD report](evaluations/DPBench/evaluation_DPBench_reading_order_weighted_ARD.txt)


## Markdown text Evaluation

Create the evaluation report:

```sh
docling_eval evaluate \
  --modality markdown_text \
  --benchmark DPBench \
  --output-dir ./scratch/DPBench/ 
```

[Markdown text json](evaluations/DPBench/evaluation_DPBench_markdown_text.json)


Visualize the report:

```sh
docling_eval visualize \
  --modality markdown_text \
  --benchmark DPBench \
  --output-dir ./scratch/DPBench/ 
```


[Markdown text report](evaluations/DPBench/evaluation_DPBench_markdown_text.txt)


![BLEU plot](evaluations/DPBench/evaluation_DPBench_markdown_text_BLEU.png)

![Edit distance plot](evaluations/DPBench/evaluation_DPBench_markdown_text_edit_distance.png)

![F1 plot](evaluations/DPBench/evaluation_DPBench_markdown_text_F1.png)

![Meteor plot](evaluations/DPBench/evaluation_DPBench_markdown_text_meteor.png)

![Precision plot](evaluations/DPBench/evaluation_DPBench_markdown_text_precision.png)

![Recall plot](evaluations/DPBench/evaluation_DPBench_markdown_text_recall.png)
