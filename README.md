# FactoryLens

Configurable industrial visual anomaly detection experiments on MVTec AD, using a PyTorch
implementation of PatchCore. Each run records detection quality, execution time,
and the configuration behind the result.

## What it provides

- **Normal-only fitting.** Build a memory bank from pretrained image features,
  then score new images by their distance from that bank. No backbone fine-tuning.
- **Controlled experiments.** JSON configurations, command-line overrides, and
  seeded data splitting and coreset selection.
- **Inspectable results.** Per-image predictions alongside AUROC, memory bank
  size, configuration, device information, and Git revision/dirty state.
- **Performance tracing.** OpenTelemetry spans and CUDA event timings for feature
  extraction, coreset selection, and evaluation. Compare runtime as well as accuracy.

## Quick start

Requires Python 3.12+ and `uv`. The default configuration uses an NVIDIA GPU with
CUDA support; set `device` to `"cpu"` in the JSON to run on CPU.

From the repository root:

```bash
uv sync --locked
```

Download MVTec AD separately and extract it under
`data/mvtec_anomaly_detection/`, keeping its category folders intact. For example:

```text
data/mvtec_anomaly_detection/grid/
├── train/good/
├── test/
└── ground_truth/
```

Run an experiment:

```bash
uv run --locked python -m visual_inspection.experiment --category grid --coreset-ratio 0.01
```

The first run downloads pretrained Wide ResNet-50-2 weights if they are not cached.

## Configuration

The default config is
[`visual_inspection/configs/patchcore__baseline.json`](visual_inspection/configs/patchcore__baseline.json):

```json
{
  "category": "bottle",
  "seed": 42,
  "backbone": "wide_resnet50_2",
  "batch_size": 8,
  "image_size": 224,
  "coreset_ratio": 0.001,
  "device": "cuda"
}
```

`coreset_ratio` controls the fraction of training patch embeddings retained in the
memory bank. Larger banks increase storage and scoring cost; larger images produce
more patches. Only `wide_resnet50_2` is currently supported.

Use `--config` to select another JSON by filename in the config directory or by
absolute path. `--category` and `--coreset-ratio` override it without changing the file.

## Results and traces

Each successful run writes to `runs/<run_id>/`:

- `result.json`: image AUROC, threshold, bank size, split sizes, elapsed time,
  CUDA stage timings, effective configuration, and environment/code metadata.
- `predictions.json`: test image paths, labels, defect types, and anomaly scores.
  Higher scores indicate greater anomaly.

OpenTelemetry traces are saved in `visual_inspection/traces/` with the same run ID.
Together, scores and stage timings help compare detection quality against runtime
and identify which stages become more expensive as configurations change.

## Evaluation

Each category gets its own memory bank. Normal training images are split 80/20:
80% build the bank, and 20% calibrate a threshold at the 99th percentile of normal
validation scores. Image AUROC is measured on the category's test set and does not
depend on that threshold.

Results across all 15 MVTec AD categories with seed 42, image size 224,
batch size 8, and coreset ratio 0.01:

| Category | Image AUROC | Memory bank vectors |
| --- | ---: | ---: |
| Bottle | 1.0000 | 1,317 |
| Cable | 0.9796 | 1,411 |
| Capsule | 0.9836 | 1,379 |
| Carpet | 0.9831 | 1,756 |
| Grid | 0.9950 | 1,662 |
| Hazelnut | 1.0000 | 2,453 |
| Leather | 1.0000 | 1,536 |
| Metal nut | 1.0000 | 1,379 |
| Pill | 0.9722 | 1,677 |
| Screw | 0.9555 | 2,007 |
| Tile | 1.0000 | 1,442 |
| Toothbrush | 0.9000 | 376 |
| Transistor | 0.9946 | 1,340 |
| Wood | 0.9912 | 1,552 |
| Zipper | 0.9958 | 1,505 |
| **Mean across categories** | **0.9834** | — |

These are single-seed, image-level development results. The mean gives each
category equal weight. Seeds
support repeat runs in the same environment; they do not guarantee identical
results across hardware or library versions. If test scores guide configuration
choices, that test set is no longer an untouched final evaluation.

Pixel-level evaluation and saved model checkpoints are not implemented yet.
Agent-driven experiment search is planned as a separate project using this engine.

## License

FactoryLens code is licensed under the [MIT License](LICENSE).
Datasets, pretrained weights, and dependencies remain subject to their own terms.
