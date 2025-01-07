# Docling-eval


[![arXiv](https://img.shields.io/badge/arXiv-2408.09869-b31b1b.svg)](https://arxiv.org/abs/2408.09869)
[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://ds4sd.github.io/docling/)
[![PyPI version](https://img.shields.io/pypi/v/docling)](https://pypi.org/project/docling/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/docling)](https://pypi.org/project/docling/)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![License MIT](https://img.shields.io/github/license/DS4SD/docling)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://static.pepy.tech/badge/docling/month)](https://pepy.tech/projects/docling)

Evaluate [Docling](https://github.com/DS4SD/docling) on various datasets.

## Features

Evaluate docling on various datasets. You can use the cli

```sh
docling-eval % poetry run evaluate --help
2024-12-20 10:51:57,593 - INFO - PyTorch version 2.5.1 available.

 Usage: evaluate [OPTIONS]

╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --task        -t      [create|evaluate|visualize]                                                                Evaluation task [default: None] [required]                                                                              │
│ *  --modality    -m      [end-to-end|layout|tableformer|codeformer]                                                 Evaluation modality [default: None] [required]                                                                          │
│ *  --benchmark   -b      [DPBench|OmniDcoBench|WordScape|PubLayNet|DocLayNet|Pub1M|PubTabNet|FinTabNet|WikiTabNet]  Benchmark name [default: None] [required]                                                                               │
│ *  --input-dir   -i      PATH                                                                                       Input directory [default: None] [required]                                                                              │
│ *  --output-dir  -o      PATH                                                                                       Output directory [default: None] [required]                                                                             │
│    --help                                                                                                           Show this message and exit.                                                                                             │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## End to End examples

### DP-Bench

Using a single command,

```sh
poetry run python ./docs/examples/benchmark_dpbench.py
```

<details>
<summary><b>Layout evaluation for DP-Bench</b></summary>
<br>

👉 Create the dataset,

```sh
poetry run evaluate -t create -m layout -b DPBench -i <location-of-dpbench> -o ./benchmarks/dpbench-layout
```

👉 Evaluate the dataset,

```sh
poetry run evaluate -t evaluate -m layout -b DPBench -i ./benchmarks/dpbench-layout -o ./benchmarks/dpbench-layout
```

| label          |   Class mAP[0.5:0.95] |
|----------------|-----------------------|
| table          |                 89.08 |
| picture        |                 76.1  |
| document_index |                 75.52 |
| text           |                 67.8  |
| caption        |                 45.8  |
| section_header |                 44.26 |
| page_footer    |                 34.42 |
| list_item      |                 29.04 |
| footnote       |                 22.08 |
| page_header    |                 15.11 |
| formula        |                  6.62 |
</details>

<details>
<summary><b>Table evaluations for DP-Bench</b></summary>
<br>

👉 Create the dataset,

```sh
poetry run evaluate -t create -m tableformer -b DPBench -i ./benchmarks/dpbench-original -o ./benchmarks/dpbench-dataset/tableformer
```

👉 Evaluate the dataset,

```sh
poetry run evaluate -t evaluate -m tableformer -b DPBench -i ./benchmarks/dpbench-dataset/tableformer -o ./benchmarks/dpbench-dataset/tableformer
```

👉 Visualise the dataset,

```sh
poetry run evaluate -t visualize -m tableformer -b DPBench -i ./benchmarks/dpbench-dataset/tableformer -o ./benchmarks/dpbench-dataset/tableformer
```

The final result can be visualised as,

![DPBench_TEDS](./docs/evaluations/evaluation_DPBench_tableformer.png)
</details>

### OmniDocBench

Using a single command,

```sh
poetry run python ./docs/examples/benchmark_omnidocbench.py
```

<details>
<summary><b>Layout evaluation for OmniDocBench</b></summary>
<br>

👉 Create the dataset,

```sh
poetry run evaluate -t create -m layout -b OmniDocBench -i ./benchmarks/omnidocbench-original -o ./benchmarks/omnidocbench-dataset/layout
```

👉 Evaluate the dataset,

```sh
poetry run evaluate -t evaluate -m layout -b OmniDocBench -i ./benchmarks/omnidocbench-dataset/layout -o ./benchmarks/omnidocbench-dataset/layout
```

👉 Visualise the dataset,

```sh
poetry run evaluate -t visualize -m tableformer -b OmniDocBench -i ./benchmarks/OmniDocBench-dataset/layout -o ./benchmarks/OmniDocBench-dataset/layout
```

| label          |   Class mAP[0.5:0.95] |
|----------------|-----------------------|
| table          |                 69.32 |
| picture        |                 29.29 |
| text           |                 23.99 |
| page_footer    |                 16.14 |
| section_header |                 13.09 |
| caption        |                 10.74 |
| page_header    |                 10.02 |
| formula        |                  3.83 |
| footnote       |                  2.48 |
</details>

<details>
<summary><b>Table evaluations for OmniDocBench</b></summary>
<br>

👉 Create the dataset,

```sh
poetry run evaluate -t create -m tableformer -b OmniDocBench -i ./benchmarks/omnidocbench-original -o ./benchmarks/omnidocbench-dataset/tableformer
```

👉 Evaluate the dataset,

```sh
poetry run evaluate -t evaluate -m tableformer -b OmniDocBench -i ./benchmarks/omnidocbench-dataset/tableformer -o ./benchmarks/omnidocbench-dataset/tableformer
```

👉 Visualise the dataset,

```sh
poetry run evaluate -t visualize -m tableformer -b OmniDocBench -i ./benchmarks/OmniDocBench-dataset/tableformer -o ./benchmarks/OmniDocBench-dataset/tableformer
```

The final result can be visualised as,

|   x0<=TEDS |   TEDS<=x1 |   prob [%] |   acc [%] |   1-acc [%] |   total |
|------------|------------|------------|-----------|-------------|---------|
|       0    |       0.05 |       0.61 |      0    |      100    |       2 |
|       0.05 |       0.1  |       0    |      0.61 |       99.39 |       0 |
|       0.1  |       0.15 |       0.61 |      0.61 |       99.39 |       2 |
|       0.15 |       0.2  |       0    |      1.21 |       98.79 |       0 |
|       0.2  |       0.25 |       0.3  |      1.21 |       98.79 |       1 |
|       0.25 |       0.3  |       1.21 |      1.52 |       98.48 |       4 |
|       0.3  |       0.35 |       2.12 |      2.73 |       97.27 |       7 |
|       0.35 |       0.4  |       0.91 |      4.85 |       95.15 |       3 |
|       0.4  |       0.45 |       0.91 |      5.76 |       94.24 |       3 |
|       0.45 |       0.5  |       0.91 |      6.67 |       93.33 |       3 |
|       0.5  |       0.55 |       2.12 |      7.58 |       92.42 |       7 |
|       0.55 |       0.6  |       3.03 |      9.7  |       90.3  |      10 |
|       0.6  |       0.65 |       3.33 |     12.73 |       87.27 |      11 |
|       0.65 |       0.7  |       3.94 |     16.06 |       83.94 |      13 |
|       0.7  |       0.75 |       7.27 |     20    |       80    |      24 |
|       0.75 |       0.8  |       6.97 |     27.27 |       72.73 |      23 |
|       0.8  |       0.85 |      13.33 |     34.24 |       65.76 |      44 |
|       0.85 |       0.9  |      13.33 |     47.58 |       52.42 |      44 |
|       0.9  |       0.95 |      22.12 |     60.91 |       39.09 |      73 |
|       0.95 |       1    |      16.97 |     83.03 |       16.97 |      56 |
</details>

### FinTabNet

Using a single command (loading the dataset from Huggingface: [FinTabNet_OTSL](https://huggingface.co/datasets/ds4sd/FinTabNet_OTSL)),

```sh
poetry run python ./docs/examples/benchmark_fintabnet.py
```

<details>
<summary><b>Table evaluations for FinTabNet</b></summary>
<br>

👉 Evaluate the dataset,

```sh
poetry run evaluate -t evaluate -m tableformer -b FinTabNet -i ./benchmarks/fintabnet-dataset/tableformer -o ./benchmarks/fintabnet-dataset/tableformer
```

👉 Visualise the dataset,

```sh
poetry run evaluate -t visualize -m tableformer -b FinTabNet -i ./benchmarks/fintabnet-dataset/tableformer -o ./benchmarks/fintabnet-dataset/tableformer
```

The final result (struct only here) can be visualised as,

|   x0<=TEDS |   TEDS<=x1 |   prob [%] |   acc [%] |   1-acc [%] |   total |
|------------|------------|------------|-----------|-------------|---------|
|       0    |       0.05 |        0   |       0   |       100   |       0 |
|       0.05 |       0.1  |        0   |       0   |       100   |       0 |
|       0.1  |       0.15 |        0   |       0   |       100   |       0 |
|       0.15 |       0.2  |        0.2 |       0   |       100   |       2 |
|       0.2  |       0.25 |        0   |       0.2 |        99.8 |       0 |
|       0.25 |       0.3  |        0   |       0.2 |        99.8 |       0 |
|       0.3  |       0.35 |        0   |       0.2 |        99.8 |       0 |
|       0.35 |       0.4  |        0   |       0.2 |        99.8 |       0 |
|       0.4  |       0.45 |        0   |       0.2 |        99.8 |       0 |
|       0.45 |       0.5  |        0   |       0.2 |        99.8 |       0 |
|       0.5  |       0.55 |        0.3 |       0.2 |        99.8 |       3 |
|       0.55 |       0.6  |        0.5 |       0.5 |        99.5 |       5 |
|       0.6  |       0.65 |        0.7 |       1   |        99   |       7 |
|       0.65 |       0.7  |        0.6 |       1.7 |        98.3 |       6 |
|       0.7  |       0.75 |        1.5 |       2.3 |        97.7 |      15 |
|       0.75 |       0.8  |        3.3 |       3.8 |        96.2 |      33 |
|       0.8  |       0.85 |       15.3 |       7.1 |        92.9 |     153 |
|       0.85 |       0.9  |       19   |      22.4 |        77.6 |     190 |
|       0.9  |       0.95 |       30.7 |      41.4 |        58.6 |     307 |
|       0.95 |       1    |       27.9 |      72.1 |        27.9 |     279 |
</details>

### Pub1M

Using a single command (loading the dataset from Huggingface: [Pub1M_OTSL](https://huggingface.co/datasets/ds4sd/Pub1M_OTSL)),

```sh
poetry run python ./docs/examples/benchmark_p1m.py
```

<details>
<summary><b>Table evaluations for Pub1M</b></summary>
<br>

👉 Evaluate the dataset,

```sh
poetry run evaluate -t evaluate -m tableformer -b Pub1M -i ./benchmarks/Pub1M-dataset/tableformer -o ./benchmarks/Pub1M-dataset/tableformer
```

👉 Visualise the dataset,

```sh
poetry run evaluate -t visualize -m tableformer -b Pub1M -i ./benchmarks/Pub1M-dataset/tableformer -o ./benchmarks/Pub1M-dataset/tableformer
```

|   x0<=TEDS |   TEDS<=x1 |   prob [%] |   acc [%] |   1-acc [%] |   total |
|------------|------------|------------|-----------|-------------|---------|
|       0    |       0.05 |        1.3 |       0   |       100   |      13 |
|       0.05 |       0.1  |        0.8 |       1.3 |        98.7 |       8 |
|       0.1  |       0.15 |        0.2 |       2.1 |        97.9 |       2 |
|       0.15 |       0.2  |        0.2 |       2.3 |        97.7 |       2 |
|       0.2  |       0.25 |        0   |       2.5 |        97.5 |       0 |
|       0.25 |       0.3  |        0   |       2.5 |        97.5 |       0 |
|       0.3  |       0.35 |        0.3 |       2.5 |        97.5 |       3 |
|       0.35 |       0.4  |        0   |       2.8 |        97.2 |       0 |
|       0.4  |       0.45 |        0.1 |       2.8 |        97.2 |       1 |
|       0.45 |       0.5  |        0.3 |       2.9 |        97.1 |       3 |
|       0.5  |       0.55 |        0.8 |       3.2 |        96.8 |       8 |
|       0.55 |       0.6  |        1.6 |       4   |        96   |      16 |
|       0.6  |       0.65 |        1.6 |       5.6 |        94.4 |      16 |
|       0.65 |       0.7  |        2.3 |       7.2 |        92.8 |      23 |
|       0.7  |       0.75 |        4.6 |       9.5 |        90.5 |      46 |
|       0.75 |       0.8  |       10.8 |      14.1 |        85.9 |     108 |
|       0.8  |       0.85 |       15.3 |      24.9 |        75.1 |     153 |
|       0.85 |       0.9  |       21.6 |      40.2 |        59.8 |     216 |
|       0.9  |       0.95 |       22.9 |      61.8 |        38.2 |     229 |
|       0.95 |       1    |       15.3 |      84.7 |        15.3 |     153 |
</details>

### PubTabNet

Using a single command (loading the dataset from Huggingface: [Pubtabnet_OTSL](https://huggingface.co/datasets/ds4sd/Pubtabnet_OTSL)),

```sh
poetry run python ./docs/examples/benchmark_pubtabnet.py
```

<details>
<summary><b>Table evaluations for Pubtabnet</b></summary>
<br>

👉 Evaluate the dataset,

```sh
poetry run evaluate -t evaluate -m tableformer -b Pubtabnet -i ./benchmarks/pubtabnet-dataset/tableformer -o ./benchmarks/pubtabnet-dataset/tableformer
```

👉 Visualise the dataset,

```sh
poetry run evaluate -t visualize -m tableformer -b Pubtabnet -i ./benchmarks/pubtabnet-dataset/tableformer -o ./benchmarks/pubtabnet-dataset/tableformer
```

The final result (struct only here) can be visualised as,

|   x0<=TEDS |   TEDS<=x1 |   prob [%] |   acc [%] |   1-acc [%] |   total |
|------------|------------|------------|-----------|-------------|---------|
|       0    |       0.05 |       0    |      0    |      100    |       0 |
|       0.05 |       0.1  |       0.01 |      0    |      100    |       1 |
|       0.1  |       0.15 |       0.01 |      0.01 |       99.99 |       1 |
|       0.15 |       0.2  |       0.02 |      0.02 |       99.98 |       2 |
|       0.2  |       0.25 |       0    |      0.04 |       99.96 |       0 |
|       0.25 |       0.3  |       0    |      0.04 |       99.96 |       0 |
|       0.3  |       0.35 |       0    |      0.04 |       99.96 |       0 |
|       0.35 |       0.4  |       0    |      0.04 |       99.96 |       0 |
|       0.4  |       0.45 |       0.02 |      0.04 |       99.96 |       2 |
|       0.45 |       0.5  |       0.1  |      0.06 |       99.94 |      10 |
|       0.5  |       0.55 |       0.1  |      0.15 |       99.85 |      10 |
|       0.55 |       0.6  |       0.24 |      0.25 |       99.75 |      25 |
|       0.6  |       0.65 |       0.47 |      0.49 |       99.51 |      49 |
|       0.65 |       0.7  |       1.04 |      0.96 |       99.04 |     108 |
|       0.7  |       0.75 |       2.44 |      2    |       98    |     254 |
|       0.75 |       0.8  |       4.65 |      4.44 |       95.56 |     483 |
|       0.8  |       0.85 |      13.71 |      9.09 |       90.91 |    1425 |
|       0.85 |       0.9  |      21.2  |     22.8  |       77.2  |    2204 |
|       0.9  |       0.95 |      28.48 |     43.99 |       56.01 |    2961 |
|       0.95 |       1    |      27.53 |     72.47 |       27.53 |    2862 |
</details>

## Contributing

Please read [Contributing to Docling](https://github.com/DS4SD/docling/blob/main/CONTRIBUTING.md) for details.

## License

The Docling codebase is under MIT license.
For individual model usage, please refer to the model licenses found in the original packages.

## IBM ❤️ Open Source AI

Docling-eval has been brought to you by IBM.
