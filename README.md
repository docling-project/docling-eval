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
│ *  --benchmark   -b      [DPBench|OmniDcoBench|WordScape|PubLayNet|DocLayNetV1|Pub1M|PubTabNet|FinTabNet|WikiTabNet]  Benchmark name [default: None] [required]                                                                               │
│ *  --input-dir   -i      PATH                                                                                       Input directory [default: None] [required]                                                                              │
│ *  --output-dir  -o      PATH                                                                                       Output directory [default: None] [required]                                                                             │
│    --help                                                                                                           Show this message and exit.                                                                                             │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## End to End examples

### FinTabNet

Using a single command (loading the dataset from Huggingface: [FinTabNet_OTSL](https://huggingface.co/datasets/ds4sd/FinTabNet_OTSL)),

```sh
poetry run python docs/examples/benchmark_fintabnet.py
```

<details>
<summary><b>Table evaluations for FinTabNet</b></summary>
<br>

👉 Evaluate the dataset:

```sh
poetry run evaluate \
    -t evaluate \
    -m tableformer \
    -b FinTabNet \
    -i benchmarks/FinTabNet-dataset/tableformer \
    -o benchmarks/FinTabNet-dataset/tableformer
```

[Tableformer evaluation json](docs/evaluations/FinTabNet/evaluation_FinTabNet_tableformer.json)

👉 Visualize the dataset:

```sh
poetry run evaluate \
    -t visualize \
    -m tableformer \
    -b FinTabNet \
    -i benchmarks/FinTabNet-dataset/tableformer \
    -o benchmarks/FinTabNet-dataset/tableformer
```

![TEDS plot](docs/evaluations/FinTabNet/evaluation_FinTabNet_tableformer-delta_row_col.png)

![TEDS struct only plot](docs/evaluations/FinTabNet/evaluation_FinTabNet_tableformer_TEDS_struct-only.png)

[TEDS struct only report](docs/evaluations/FinTabNet/evaluation_FinTabNet_tableformer_TEDS_struct-only.txt)

![TEDS struct with text plot](docs/evaluations/FinTabNet/evaluation_FinTabNet_tableformer_TEDS_struct-with-text.png)

[TEDS struct with text report](docs/evaluations/FinTabNet/evaluation_FinTabNet_tableformer_TEDS_struct-with-text.txt)

</details>

### DocLayNet v1

Using a single command,

```sh
poetry run python ./docs/examples/benchmark_doclaynet_v1.py
```

This command downloads the DocLayNet v1.1 dataset, runs the evaluations and produces the following files:

<details>
<summary><b>Layout evaluation</b></summary>
<br>

- [Layout evaluation json](docs/evaluations/DocLayNetV1/evaluation_DocLayNetV1_layout.json)
- [mAP[0.5:0.95] report](docs/evaluations/DocLayNetV1/evaluation_DocLayNetV1_layout_mAP[0.5_0.95].txt)
- [mAP[0.5:0.95] plot](docs/evaluations/DocLayNetV1/evaluation_DocLayNetV1_layout_mAP[0.5_0.95].png)

</details>
=======


### Pub1M

Using a single command (loading the dataset from Huggingface: [Pub1M_OTSL](https://huggingface.co/datasets/ds4sd/Pub1M_OTSL)),

```sh
poetry run python docs/examples/benchmark_p1m.py
```

<details>
<summary><b>Table evaluations for Pub1M</b></summary>
<br>

👉 Evaluate the dataset:

```sh
poetry run evaluate \
    -t evaluate \
    -m tableformer \
    -b Pub1M \
    -i benchmarks/Pub1M-dataset/tableformer \
    -o benchmarks/Pub1M-dataset/tableformer
```

[Tableformer evaluation json](docs/evaluations/Pub1M/evaluation_Pub1M_tableformer.json)

👉 Visualize the dataset:

```sh
poetry run evaluate \
    -t visualize \
    -m tableformer \
    -b Pub1M \
    -i benchmarks/Pub1M-dataset/tableformer \
    -o benchmarks/Pub1M-dataset/tableformer
```

![TEDS plot](docs/evaluations/Pub1M/evaluation_Pub1M_tableformer-delta_row_col.png)

![TEDS struct only plot](docs/evaluations/Pub1M/evaluation_Pub1M_tableformer_TEDS_struct-only.png)

[TEDS struct only report](docs/evaluations/Pub1M/evaluation_Pub1M_tableformer_TEDS_struct-only.txt)

![TEDS struct with text plot](docs/evaluations/Pub1M/evaluation_Pub1M_tableformer_TEDS_struct-with-text.png)

[TEDS struct with text report](docs/evaluations/Pub1M/evaluation_Pub1M_tableformer_TEDS_struct-with-text.txt)

</details>


### PubTabNet

Using a single command (loading the dataset from Huggingface: [Pubtabnet_OTSL](https://huggingface.co/datasets/ds4sd/Pubtabnet_OTSL)),

```sh
poetry run python ./docs/examples/benchmark_pubtabnet.py
```

<details>
<summary><b>Table evaluations for PubTabNet</b></summary>
<br>

👉 Evaluate the dataset:

```sh
poetry run evaluate \
    -t evaluate \
    -m tableformer \
    -b PubTabNet \
    -i benchmarks/PubTabNet-dataset/tableformer \
    -o benchmarks/PubTabNet-dataset/tableformer
```

[Tableformer evaluation json](docs/evaluations/PubTabNet/evaluation_PubTabNet_tableformer.json)

👉 Visualize the dataset:

```sh
poetry run evaluate \
    -t visualize \
    -m tableformer \
    -b PubTabNet \
    -i benchmarks/PubTabNet-dataset/tableformer \
    -o benchmarks/PubTabNet-dataset/tableformer
```

![TEDS plot](docs/evaluations/PubTabNet/evaluation_PubTabNet_tableformer-delta_row_col.png)

![TEDS struct only plot](docs/evaluations/PubTabNet/evaluation_PubTabNet_tableformer_TEDS_struct-only.png)

[TEDS struct only report](docs/evaluations/PubTabNet/evaluation_PubTabNet_tableformer_TEDS_struct-only.txt)

![TEDS struct with text plot](docs/evaluations/PubTabNet/evaluation_PubTabNet_tableformer_TEDS_struct-with-text.png)

[TEDS struct with text report](docs/evaluations/PubTabNet/evaluation_PubTabNet_tableformer_TEDS_struct-with-text.txt)


</details>


## DP-Bench

[See DP-Bench benchmarks](docs/DP-Bench_benchmarks.md)


## OmniDocBench

[See OmniDocBench benchmarks](docs/OmniDocBench_benchmarks.md)


## Contributing

Please read [Contributing to Docling](https://github.com/DS4SD/docling/blob/main/CONTRIBUTING.md) for details.


## License

The Docling codebase is under MIT license.
For individual model usage, please refer to the model licenses found in the original packages.


## IBM ❤️ Open Source AI

Docling-eval has been brought to you by IBM.
