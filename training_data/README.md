# Shared training data

Preprocessed `.npy` shards live here so multiple agents can train on the same
data without re-running SGF replay.

## Layout

```
training_data/
  datasets/
    {dataset_id}/
      train_features.npy
      train_labels.npy
      val_features.npy          # optional
      val_labels.npy            # optional
      manifest.json             # build params + agent usage history
  usage/
    {agent_id}.jsonl            # append-only log of training runs per agent
```

`dataset_id` is derived from the training SGF source and build parameters, for
example `sgf_train_g4000_s0_bs19_p5`. Override with `--dataset-id` on
`run_train.py`.

## Usage

```bash
# Build or reuse cached data, then train basic_cnn
python run_train.py --max-games 5000 --epochs 10

# Preprocess only (writes under training_data/datasets/...)
python run_train.py --max-games 5000 --preprocess-only --force-rebuild

# Reuse an existing dataset explicitly
python run_train.py --dataset-id sgf_train_g4000_s0_bs19_p5 --epochs 20
```

Raw SGF files remain under `Games/` (gitignored). This folder is also
gitignored.

## Tracking

Each dataset's `manifest.json` records how it was built (SGF directory, game
cap, seed, board size, planes) and which agents trained on it.

Each agent also gets an append-only log at `usage/{agent_id}.jsonl`. Training
checkpoints store `dataset_id` and `dataset_dir` in `training_config`.
