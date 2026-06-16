# BasicPolicyCNN5x5 (GoPet)

Same layout as [`basic_cnn`](../basic_cnn/README.md), but trunk convolutions use **5×5** kernels
(`padding=2`) instead of 3×3. More parameters and receptive field per layer; training reuses
shared data under `training_data/`.

## Train

```bash
# From repo root
python run_train.py --agent basic_cnn_5x5 --max-games 5000 --epochs 10

# Resume
python run_train.py --agent basic_cnn_5x5 --epochs 20 \
  --resume agents/basic_cnn_5x5/checkpoints/basic_cnn_5x5_training_state.pt
```

Checkpoints: `agents/basic_cnn_5x5/checkpoints/` (gitignored).

## Play

```bash
python run_play.py --model agents/basic_cnn_5x5/checkpoints/basic_cnn_5x5_latest.pt
```
