# Anomaly Detection

A small agent-controlled experiment harness for MVTec AD.

The first version only defines the main boundaries:

- `data.py` owns the dataset.
- `model.py` owns PyTorch models.
- `evaluation.py` owns metrics.
- `experiment.py` connects the ML pieces.
- `storage.py` owns experiment artifacts.
- `tools.py` exposes safe operations to Codex.
- `agent.py` connects those tools to Theseus.

The experiment currently fits a normal memory bank, calibrates a threshold, and
evaluates image AUROC. Pixel metrics and agent tools remain placeholders.

Run an experiment (the baseline JSON selects CUDA):

```bash
.venv/bin/python -m visual_inspection.experiment --category grid --coreset-ratio 0.01
```

`--config` selects another JSON from `visual_inspection/configs/` (or an absolute
path). Command-line overrides do not modify the JSON. Only `wide_resnet50_2` is
supported; other backbone names raise an error.

Each successful run returns a summary and saves two files under `runs/<run_id>/`:

- `result.json`: effective config, AUROC, elapsed seconds, bank size, threshold,
  split sizes, Git revision/dirty flag, PyTorch version and device name.
- `predictions.json`: each test image's dataset-relative path, defect type, true
  label and raw anomaly score. Larger scores mean more anomalous.

The command-line run's trace uses the same ID in `visual_inspection/traces/`.
Generated runs and traces are ignored by Git. Failed runs raise an exception;
they do not currently write a result summary. Model weights are not saved.

The seed controls the data split, Python's coreset choice and PyTorch randomness.
cuDNN uses deterministic convolution selection. This is intended for repeated
runs in the same environment, not a promise of identical results across hardware
or library versions. A dirty Git flag means the commit alone does not identify
the exact code used.

AUROC still uses your pairwise ranking rule, with half credit for ties. The
comparisons now run as tensors on CPU, avoiding a GPU synchronization for every
pair. The pair matrix is suitable for these per-category MVTec test sets, not
arbitrarily large evaluation datasets.

Keep results per category. Once test results guide model/config changes, treat
them as development feedback rather than an untouched final evaluation.

Focused metric and storage checks:

```bash
.venv/bin/python -m unittest discover -s tests -p test_evaluation_metrics.py -v
```

GPU validation on 2026-09-09 (GTX 1660 SUPER, seed 42, image size 224,
batch size 8, coreset ratio 0.01):

| Category | Image AUROC | Bank rows | Test images | Run seconds |
| --- | ---: | ---: | ---: | ---: |
| grid | 0.994987 | 1662 | 78 | 52.7 |
| grid, repeat | 0.994987 | 1662 | 78 | 52.4 |
| cable | 0.979573 | 1411 | 150 | 49.8 |

Both grid runs produced exactly equal saved image scores and thresholds.
Saved scores for both categories also reproduced AUROC using the original
Python pairwise rule. Four focused tests passed, and an unsupported backbone
was confirmed to raise an error. These are development results, not a claim
about performance across all categories.
