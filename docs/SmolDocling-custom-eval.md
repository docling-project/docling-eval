# Evaluate SmolDocling with docling-eval

Below are instructions to evaluate custom weights for SmolDocling with docling-eval.

## Prepare SmolDocling weights for docling

Docling can run SmolDocling out of the box. By default, it will download the model weights from Huggingface and keep them in the user `~/.cache` dir.
If you want to inject custom weights and config, you need to prepare a directory like this:

```shell
models/ # the dir you will point docling-eval to (see below)
├─ ds4sd--SmolDocling-256M-preview/ # the dir you place custom weights in. The name _must_ match the SmolDocling HF repo id, but using -- for /.
```

## Run docling-eval

You can now run `docling-eval` as shown below. Example given for the Docling-DocLayNetV1 dataset:

```shell
# Create GT dataset for DocLayNet v1 test set (only once)
mkdir benchmarks

huggingface-cli login --token your_hf_token_123 # token-type: read is good, get it here: https://huggingface.co/settings/tokens
huggingface-cli download --repo-type dataset --local-dir ./benchmarks/DLN_GT/gt_dataset ds4sd/Docling-DocLayNetV1
# alternatively, create the GT dataset yourself: docling-eval create-gt --benchmark DocLayNetV1 --output-dir ./benchmarks/DLN_GT/ 

## --- Do benchnmarks ---
export HF_HUB_OFFLINE=1 # no communication with huggingface from now!

# Make predictions for smoldocling
docling-eval create-eval \
  --benchmark DocLayNetV1 \
  --gt-dir ./benchmarks/DLN_GT/gt_dataset/ \
  --output-dir ./benchmarks/DLN_smoldocling_experiment1/ \
  --prediction-provider SmolDocling \
  --artifacts-path /path/to/your/models/ # see above. Must include the ds4sd--SmolDocling-256M-preview dir.

# Layout metrics eval
docling-eval evaluate \
  --modality layout \
  --benchmark DocLayNetV1 \
  --output-dir ./benchmarks/DLN_smoldocling_experiment1/ 

docling-eval visualize \
  --modality layout \
  --benchmark DocLayNetV1 \
  --output-dir ./benchmarks/DLN_smoldocling_experiment1/ 

# Text metrics eval
docling-eval evaluate \
  --modality markdown_text \
  --benchmark DocLayNetV1 \
  --output-dir ./benchmarks/DLN_smoldocling_experiment1/ 

# Text metrics eval
docling-eval visualize \
  --modality markdown_text \
  --benchmark DocLayNetV1 \
  --output-dir ./benchmarks/DLN_smoldocling_experiment1/ 
  
```
To repeat this with another set of weights, please replace the content of your `models/ds4sd--SmolDocling-256M-preview` directory, and adjust the
experiment name used in your `--output-dir` arguments.

**Note**: MacOS users should use weights converted with mlx-vlm. 
Install `mlx-vlm`, convert the weights, and place them in a `ds4sd--SmolDocling-256M-preview-mlx-bf16` subdirectory instead.


