# Frequently Asked Questions

## docling-eval seem stuck

Add the environment variable (in case HF is not responding), 

```sh
caffeinate HF_HUB_OFFLINE=1 uv run docling_eval create-eval --benchmark DocLayNetV1 --gt-dir ./benchmarks/DocLayNetV1/gt_dataset --output-dir ./benchmarks/DocLayNetV1/smoldocling_v4 --prediction-provider SmolDocling --end-index 256
```