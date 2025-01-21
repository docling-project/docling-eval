# Creating Groundtruth dataset using the CVAT annotation tool

## Prerequisites

```sh
poetry run python ./docs/examples/benchmark_dpbench.py
```

## Set up the necessary files

```sh
poetry run python docling_eval/benchmarks/cvat_annotation/preannotate.py -i ./benchmarks/DPBench-dataset/layout/test/ -o ./benchmarks/docling-DPBench
```

## Online annotation


## Packaging the annotations into new dataset

```sh
poetry run python docling_eval/benchmarks/cvat_annotation/create.py -i ./benchmarks/docling-DPBench
```

## Running evaluation


```sh
poetry run evaluate -t evaluate -m layout -b DPBench -i ./benchmarks/docling-DPBench/layout -o ./benchmarks/docling-DPBench/layout
```


```sh
poetry run evaluate -t visualize -m layout -b DPBench -i ./benchmarks/docling-DPBench/layout -o ./benchmarks/docling-DPBench/layout
```