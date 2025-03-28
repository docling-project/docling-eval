# Overview of Benchmark datasets

This part of the code contains all the scripts to create the groundtruth datasets for evaluation

## DP-Bench

The (DP-Bench](https://huggingface.co/datasets/upstage/dp-bench/tree/main) is a small (200 doc's, single page) dataset with combined layout and table-structure.

The dataset has the following layout,

```
../dp-bench
├── dataset
│   ├── pdfs
│   └── sample_results
├── scripts
└── src
```

To create the dataset,

```sh
poetry run python ./docling_eval/legacy/dpbench/create.py -h

2024-12-19 12:47:08,933 - INFO - PyTorch version 2.5.1 available.
usage: create.py [-h] -i DPBENCH_DIRECTORY [-o OUTPUT_DIRECTORY] [-s IMAGE_SCALE] [-m {end-2-end,table,formula,all}]

Process DP-Bench benchmark from directory into HF dataset.

options:
  -h, --help            show this help message and exit
  -i DPBENCH_DIRECTORY, --dpbench-directory DPBENCH_DIRECTORY
                        input directory with documents
  -o OUTPUT_DIRECTORY, --output-directory OUTPUT_DIRECTORY
                        output directory with shards
  -s IMAGE_SCALE, --image-scale IMAGE_SCALE
                        image-scale of the pages
  -m {end-2-end,table,formula,all}, --mode {end-2-end,table,formula,all}
                        mode of dataset
```

To inspect the tables from the groundtruth (assuming the shard is located at `./benchmarks/dpbench/tables`) The output will be stored in `./benchmarks/dpbench/tables/tables_gt`. Simply run,

```
poetry run python ./docling_eval/benchmarks/iterators.py -i ./benchmarks/dpbench/tables -o ./benchmarks/dpbench/tables/tables_gt -m table -c GroundTruthDocument
```

To inspect the tables from the prediction (assuming the shard is located at `./benchmarks/dpbench/tables`) The output will be stored in `./benchmarks/dpbench/tables/tables_pred`. Simply run,

```
poetry run python ./docling_eval/benchmarks/iterators.py -i ./benchmarks/dpbench/tables -o ./benchmarks/dpbench/tables/tables_pred -m table -c PredictedDocument
```