<p align="center">
  <a href="https://github.com/docling-project/docling-eval">
    <img loading="lazy" alt="Docling" src="docs/assets/docling-eval-pic.png" width="40%"/>
  </a>
</p>

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

```shell
terminal %> poetry run docling_eval --help
                                                                                                                                                                                                                                                
 Usage: docling_eval [OPTIONS] COMMAND [ARGS]...                                                                                                                                                                                                
                                                                                                                                                                                                                                                
 Docling Evaluation CLI for benchmarking document processing tasks.                                                                                                                                                                             
                                                                                                                                                                                                                                                
╭─ Options ────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                          │
╰──────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────────────╮
│ create        Create both ground truth and evaluation datasets in one step.          │
│ create-eval   Create evaluation dataset from existing ground truth.                  │
│ create-gt     Create ground truth dataset only.                                      │
│ evaluate      Evaluate predictions against ground truth.                             │
│ visualize     Visualize evaluation results.                                          │
╰──────────────────────────────────────────────────────────────────────────────────────╯


```

## Benchmarks

- General
    - [DP-Bench benchmarks](docs/DP-Bench_benchmarks.md): Text, layout, reading order and table structure evaluation on the DP-Bench dataset.
    - [OmniDocBench benchmarks](docs/OmniDocBench_benchmarks.md): Text, layout, reading order and table structure evaluation on the OmniDocBench dataset.
- Layout
    - [DocLayNetV1 Benchmarks](docs/DocLayNetv1_benchmarks.md): Text and layout evaluation on the DocLayNet v1.2 dataset.
- Table-Structure
    - [FinTabnet Benchmarks](docs/FinTabNet_benchmarks.md): Table structure evaluation on the FinTabNet dataset.
    - [PubTabNet benchmarks](docs/PubTabNet_benchmarks.md): Table structure evaluation on the PubTabNet dataset.
    - [Pub1M benchmarks](docs/P1M_benchmarks.md): Table structure evaluation on the Pub1M dataset.

On our list for next benchmarks:

- [OmniOCR](getomni-ai/ocr-benchmark)
- Hyperscalers
- [CoMix](https://github.com/emanuelevivoli/CoMix/tree/main/docs/datasets)
- [DocVQA](https://huggingface.co/datasets/lmms-lab/DocVQA)
- [rd-tablebench](https://huggingface.co/datasets/reducto/rd-tablebench)
  
## Contributing

Please read [Contributing to Docling](https://github.com/DS4SD/docling/blob/main/CONTRIBUTING.md) for details.


## License

The Docling codebase is under MIT license.
For individual model usage, please refer to the model licenses found in the original packages.


## IBM ❤️ Open Source AI

Docling-eval has been brought to you by IBM.
