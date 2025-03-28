# Pub1M Benchmarks

Create and evaluate PubTabNet dataset using a single command. This command downloads from Huggingface the [PubTabNet_OTSL dataset](https://huggingface.co/datasets/ds4sd/Pubtabnet_OTSL) and runs the evaluations for TableFormer using the first 1000 samples.

```sh
poetry run python docs/examples/benchmark_pubtabnet.py
```

## Layout Evaluation

Create the evaluation report:

```sh
poetry run evaluate \
    -t evaluate \
    -m tableformer \
    -b PubTabNet \
    -i legacy/PubTabNet-dataset/tableformer \
    -o legacy/PubTabNet-dataset/tableformer
```

[Tableformer evaluation json](evaluations/PubTabNet/evaluation_PubTabNet_tableformer.json)

Visualize the report:

```sh
poetry run evaluate \
    -t visualize \
    -m tableformer \
    -b PubTabNet \
    -i legacy/PubTabNet-dataset/tableformer \
    -o legacy/PubTabNet-dataset/tableformer
```

![TEDS plot](evaluations/PubTabNet/evaluation_PubTabNet_tableformer-delta_row_col.png)

![TEDS struct only plot](evaluations/PubTabNet/evaluation_PubTabNet_tableformer_TEDS_struct-only.png)

[TEDS struct only report](evaluations/PubTabNet/evaluation_PubTabNet_tableformer_TEDS_struct-only.txt)

![TEDS struct with text plot](evaluations/PubTabNet/evaluation_PubTabNet_tableformer_TEDS_struct-with-text.png)

[TEDS struct with text report](evaluations/PubTabNet/evaluation_PubTabNet_tableformer_TEDS_struct-with-text.txt)

