# OmniDocBench

Create and evaluate OmniDocBench using a single command:

```sh
poetry run python ./docs/examples/benchmark_omnidocbench.py
```

This command downloads runs the evaluations and produces the following files:


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
    -b OmniDocBench \
    -i benchmarks/OmniDocBench-dataset/layout \
    -o benchmarks/OmniDocBench-dataset/layout
```

[Layout evaluation json](evaluations/OmniDocBench/evaluation_OmniDocBench_layout.json)

Visualize the report:

```sh
poetry run evaluate \
    -t visualize \
    -m layout \
    -b OmniDocBench \
    -i benchmarks/OmniDocBench-dataset/layout \
    -o benchmarks/OmniDocBench-dataset/layout
```

[mAP[0.5:0.95] report](evaluations/OmniDocBench/evaluation_OmniDocBench_layout_mAP[0.5_0.95].txt)

![mAP[0.5:0.95] plot](evaluations/OmniDocBench/evaluation_OmniDocBench_layout_mAP[0.5_0.95].png)

<!--
</details>
-->


## Tableformer Evaluation

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
    -b OmniDocBench \
    -i benchmarks/OmniDocBench-dataset/tableformer \
    -o benchmarks/OmniDocBench-dataset/tableformer
```

[Tableformer evaluation json](evaluations/OmniDocBench/evaluation_OmniDocBench_tableformer.json)


Visualize the report:

```sh
poetry run evaluate \
    -t visualize \
    -m tableformer \
    -b OmniDocBench \
    -i benchmarks/OmniDocBench-dataset/tableformer \
    -o benchmarks/OmniDocBench-dataset/tableformer
```

![TEDS plot](evaluations/OmniDocBench/evaluation_OmniDocBench_tableformer-delta_row_col.png)

![TEDS struct only plot](evaluations/OmniDocBench/evaluation_OmniDocBench_tableformer_TEDS_struct-only.png)

[TEDS struct only report](evaluations/OmniDocBench/evaluation_OmniDocBench_tableformer_TEDS_struct-only.txt)

![TEDS struct with text plot](evaluations/OmniDocBench/evaluation_OmniDocBench_tableformer_TEDS_struct-with-text.png)

[TEDS struct with text report](evaluations/OmniDocBench/evaluation_OmniDocBench_tableformer_TEDS_struct-with-text.txt)

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
    -b OmniDocBench \
    -i benchmarks/OmniDocBench-dataset/layout \
    -o benchmarks/OmniDocBench-dataset/layout
```

[Reading order json](evaluations/OmniDocBench/evaluation_OmniDocBench_reading_order.json)


Visualize the report:

```sh
poetry run evaluate \
    -t visualize \
    -m reading_order \
    -b OmniDocBench \
    -i benchmarks/OmniDocBench-dataset/layout \
    -o benchmarks/OmniDocBench-dataset/layout
```

[ARD report](evaluations/OmniDocBench/evaluation_OmniDocBench_reading_order_ARD_norm.txt)

[Weighted ARD report](evaluations/OmniDocBench/evaluation_OmniDocBench_reading_order_weighted_ARD.txt)

![ARD plot](evaluations/OmniDocBench/evaluation_OmniDocBench_reading_order_ARD_norm.png)

![Weighted ARD plot](evaluations/OmniDocBench/evaluation_OmniDocBench_reading_order_weighted_ARD.png)


<!--
</details>
-->

## Markdown text evaluation
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
    -b OmniDocBench \
    -i benchmarks/OmniDocBench-dataset/layout \
    -o benchmarks/OmniDocBench-dataset/layout
```

[Markdown text json](evaluations/OmniDocBench/evaluation_OmniDocBench_markdown_text.json)


Visualize the report:

```sh
poetry run evaluate \
    -t visualize \
    -m markdown_text \
    -b OmniDocBench \
    -i benchmarks/OmniDocBench-dataset/layout \
    -o benchmarks/OmniDocBench-dataset/layout
```

![BLEU plot](evaluations/OmniDocBench/evaluation_OmniDocBench_markdown_text_BLEU.png)

[BLEU report](evaluations/OmniDocBench/evaluation_OmniDocBench_markdown_text_BLEU.txt)

<!--
</details>
-->
