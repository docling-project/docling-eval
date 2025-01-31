# DP-Bench Benchmarks

Create and evaluate DP-Bench using a single command:

```sh
poetry run python ./docs/examples/benchmark_dpbench.py
```

This command downloads the DP-Bench dataset, runs the evaluations and produces the following files:

## Layout Evaluation

<!--
<details>
<summary><b>Layout evaluation</b></summary>
<br>
-->

Create the report:

```sh
poetry run evaluate \
    -t evaluate \
    -m layout \
    -b DPBench \
    -i benchmarks/DPBench-dataset/layout \
    -o benchmarks/DPBench-dataset/layout
```

[Layout evaluation json](evaluations/DPBench/evaluation_DPBench_layout.json)

Visualize the report:

```sh
poetry run evaluate \
    -t visualize \
    -m layout \
    -b DPBench \
    -i benchmarks/DPBench-dataset/layout \
    -o benchmarks/DPBench-dataset/layout
```

[mAP[0.5:0.95] report](evaluations/DPBench/evaluation_DPBench_layout_mAP[0.5_0.95].txt)

![mAP[0.5:0.95] plot](evaluations/DPBench/evaluation_DPBench_layout_mAP[0.5_0.95].png)

<!--
</details>
-->


## TableFormer Evaluation

<!--
<details>
<summary><b>Tableformer evaluation</b></summary>
<br>
-->

Create the report:

```sh
poetry run evaluate \
    -t evaluate \
    -m tableformer \
    -b DPBench \
    -i benchmarks/DPBench-dataset/tableformer \
    -o benchmarks/DPBench-dataset/tableformer
```

Visualize the report:

[Tableformer evaluation json](evaluations/DPBench/evaluation_DPBench_tableformer.json)

```sh
poetry run evaluate \
    -t visualize \
    -m tableformer \
    -b DPBench \
    -i benchmarks/DPBench-dataset/tableformer \
    -o benchmarks/DPBench-dataset/tableformer
```

![TEDS plot](evaluations/DPBench/evaluation_DPBench_tableformer-delta_row_col.png)

![TEDS struct only plot](evaluations/DPBench/evaluation_DPBench_tableformer_TEDS_struct-only.png)

[TEDS struct only report](evaluations/DPBench/evaluation_DPBench_tableformer_TEDS_struct-only.txt)

![TEDS struct with text plot](evaluations/DPBench/evaluation_DPBench_tableformer_TEDS_struct-with-text.png)

[TEDS struct with text report](evaluations/DPBench/evaluation_DPBench_tableformer_TEDS_struct-with-text.txt)

<!--
</details>
-->


## Reading order Evaluation

<!--
<details>
<summary><b>Reading order evaluation</b></summary>
<br>
-->

Create the report:

```sh
poetry run evaluate \
    -t evaluate \
    -m reading_order \
    -b DPBench \
    -i benchmarks/DPBench-dataset/layout \
    -o benchmarks/DPBench-dataset/layout
```

[Reading order json](evaluations/DPBench/evaluation_DPBench_reading_order.json)

Visualize the report:

```sh
poetry run evaluate \
    -t visulize \
    -m reading_order \
    -b DPBench \
    -i benchmarks/DPBench-dataset/layout \
    -o benchmarks/DPBench-dataset/layout
```

![ARD plot](evaluations/DPBench/evaluation_DPBench_reading_order_ARD_norm.png)

[ARD report](evaluations/DPBench/evaluation_DPBench_reading_order_ARD_norm.txt)

![Weighted ARD plot](evaluations/DPBench/evaluation_DPBench_reading_order_weighted_ARD.png)

[Weighted ARD report](evaluations/DPBench/evaluation_DPBench_reading_order_weighted_ARD.txt)

<!--
</details>
-->

## Markdown text Evaluation

<!--
<details>
<summary><b>Markdown text evaluation</b></summary>
<br>
-->

Create the report:

```sh
poetry run evaluate \
    -t evaluate \
    -m markdown_text \
    -b DPBench \
    -i benchmarks/DPBench-dataset/layout \
    -o benchmarks/DPBench-dataset/layout
```

[Markdown text json](evaluations/DPBench/evaluation_DPBench_markdown_text.json)

Visualize the report:

```sh
poetry run evaluate \
    -t visualize \
    -m markdown_text \
    -b DPBench \
    -i benchmarks/DPBench-dataset/layout \
    -o benchmarks/DPBench-dataset/layout
```


![BLEU plot](evaluations/DPBench/evaluation_DPBench_markdown_text_BLEU.png)

[BLEU report](evaluations/DPBench/evaluation_DPBench_markdown_text_BLEU.txt)

<!--
</details>
-->
