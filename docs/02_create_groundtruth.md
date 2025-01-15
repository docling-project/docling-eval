# Creating Groundtruth dataset using the CVAT annotation tool

## Prerequisites

```sh
poetry run python ./docs/examples/benchmark_dpbench.py
```

## Set up the necessary files

```sh
poetry run python ./docling_eval/benchmarks/annotation_formats/preannotate.py -i ./benchmarks/DPBench-dataset/layout/test/ -o ./benchmarks/DPBench-annotations
```

## Online annotation


## Packaging the annotations into new dataset

```sh
poetry run python ./docling_eval/benchmarks/annotation_formats/create.py -i ./benchmarks/DPBench-annotations
```

## Running evaluation

sh
```
poetry run evaluate -t evaluate -m layout -b DPBench -i ./benchmarks/DPBench-annotations/layout -o ./benchmarks/DPBench-annotations/layout
```

sh
```
poetry run evaluate -t visualize -m layout -b DPBench -i ./benchmarks/DPBench-annotations/layout -o ./benchmarks/DPBench-annotations/layout
```