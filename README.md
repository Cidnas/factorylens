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

Training and evaluation logic will be added one small step at a time.
