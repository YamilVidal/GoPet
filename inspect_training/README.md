# inspect_training

Tools to inspect agent training runs (`basic_cnn`, `basic_cnn_5x5`, …).

## Training history file

Each training run writes a JSON history next to the checkpoint:

```
agents/{agent}/checkpoints/{agent}_history.json
```

One record is appended per completed epoch with train/val loss and accuracy.

## Commands

Use `--agent` to pick which agent to inspect (defaults to `basic_cnn`). You can still
pass `--checkpoint` to override the path.

### Summary

```bash
python run_inspect_training.py summary
python run_inspect_training.py summary --agent basic_cnn_5x5
python run_inspect_training.py summary --checkpoint agents/basic_cnn/checkpoints/basic_cnn.pt
```

### Plot loss and accuracy

```bash
python run_inspect_training.py plot
python run_inspect_training.py plot --agent basic_cnn_5x5
python run_inspect_training.py plot --agent basic_cnn_5x5 --output reports/basic_cnn_5x5_metrics.png
python run_inspect_training.py plot --show
```

Or directly:

```bash
python -m inspect_training.summary --agent basic_cnn_5x5
python -m inspect_training.plot_metrics --agent basic_cnn_5x5
```

## Note on older runs

Runs completed before history logging was added only have the latest epoch in
`*_training_state.pt`. Re-run training (or continue with `--resume`) to build a
full history file.
