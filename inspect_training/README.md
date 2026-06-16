# inspect_training

Tools to inspect agent training runs (currently `basic_cnn`).

## Training history file

Each training run writes a JSON history next to the checkpoint:

```
agents/basic_cnn/checkpoints/basic_cnn_history.json
```

One record is appended per completed epoch with train/val loss and accuracy.

## Commands

### Summary

```bash
python run_inspect_training.py summary
python run_inspect_training.py summary --checkpoint agents/basic_cnn/checkpoints/basic_cnn.pt
```

### Plot loss and accuracy

```bash
python run_inspect_training.py plot
python run_inspect_training.py plot --output reports/basic_cnn_metrics.png
python run_inspect_training.py plot --show
```

Or directly:

```bash
python -m inspect_training.plot_metrics --checkpoint agents/basic_cnn/checkpoints/basic_cnn.pt
```

## Note on older runs

Runs completed before history logging was added only have the latest epoch in
`*_training_state.pt`. Re-run training (or continue with `--resume`) to build a
full history file.
