# BasicPolicyCNN

A small convolutional policy network for **19×19 Go**, trained with supervised learning on professional SGF games. The design prioritizes fast CPU training and compatibility with the GoPet play server.

Implementation: [`model.py`](model.py)

---

## Overview

`BasicPolicyCNN` predicts the **next move** (including pass) from the current board state. It is a **policy-only** network — there is no value head and no MCTS. At inference time, illegal moves are masked before selecting an action.

| Property | Value |
|----------|-------|
| Board size | 19×19 |
| Input channels | 5 |
| Output logits | 362 (361 intersections + 1 pass) |
| Parameters | ~66k |
| Target device | CPU |

---

## Input: feature planes

The network expects tensors shaped **`[batch, 5, 19, 19]`**, produced by `gopet.encoding.encode_state()` with `NUM_BASIC_PLANES = 5`.

| Plane | Content |
|-------|---------|
| 0 | Current player's stones |
| 1 | Opponent's stones |
| 2 | Empty intersections |
| 3 | Constant 1 if Black to play, else 0 |
| 4 | Constant 1 if White to play, else 0 |

Each plane is binary (0.0 or 1.0) except the side-to-move planes, which are uniform across the board.

---

## Output: action space

The model returns **`[batch, 362]`** unnormalized logits (one per legal action index):

| Index range | Meaning |
|-------------|---------|
| `0 … 360` | Intersection moves, row-major: `index = row × 19 + col` |
| `361` | Pass |

During training, the label is the expert move from SGF replay. During play, logits for illegal moves are set to `-inf` via `gopet.encoding.mask_policy_logits()` before argmax.

---

## Architecture

```
Input [B, 5, 19, 19]
        │
        ▼
┌───────────────────────────────────┐
│  Trunk (same spatial size 19×19)  │
│  Conv 5→32, 3×3, pad 1  + ReLU    │
│  Conv 32→32, 3×3, pad 1 + ReLU    │
│  Conv 32→64, 3×3, pad 1 + ReLU    │
│  Conv 64→64, 3×3, pad 1 + ReLU    │
└───────────────────────────────────┘
        │
        ├──────────────────────────────┐
        ▼                              ▼
  Conv 64→1, 1×1                  Global avg pool
  → [B, 1, 19, 19]                over 19×19
  → flatten → [B, 361]            → [B, 64]
        │                              │
        │                         Linear 64→1
        │                              │
        └────────── concat ────────────┘
                      │
                      ▼
              [B, 362] logits
```

### Trunk

Four **3×3 convolutional layers** with `padding=1` keep the spatial resolution at 19×19 throughout. Channel progression: **5 → 32 → 32 → 64 → 64**. Each conv is followed by ReLU.

This stack extracts local stone patterns (captures, connections, liberties) without downsampling, which is important for move prediction on a full-size board.

### Policy head (board moves)

A **1×1 convolution** maps the 64-channel feature map to a single channel, producing one logit per intersection. The result is flattened to 361 values. This is cheaper and more natural than flattening the full feature map into a large dense layer.

### Pass head

**Global average pooling** over the 64-channel feature map summarizes the whole-board context into a 64-vector, then a **linear layer** outputs a single pass logit. Board and pass logits are concatenated.

---

## Design choices

| Choice | Rationale |
|--------|-----------|
| 5 input planes | Fast to encode; enough context for a baseline policy |
| No max-pooling | Preserves one-to-one spatial alignment for move logits |
| 1×1 spatial head | ~66k params; trains in hours on CPU |
| Separate pass head | Pass is a global decision, not tied to one intersection |
| No batch norm | Keeps the network small and training simple on CPU |

---

## Training

Supervised learning with **cross-entropy loss** against expert moves from SGF replay.

```bash
python run_train.py --max-games 5000 --epochs 9 --force-rebuild
```

Related modules:

| File | Role |
|------|------|
| [`train.py`](train.py) | Training loop, checkpoints, resume |
| [`data.py`](data.py) | SGF subsampling and `.npy` cache |
| [`../../estimate_training_time.py`](../../estimate_training_time.py) | Time estimates for games/epoch settings |

Checkpoints are saved after every epoch (`basic_cnn.pt`, `basic_cnn_latest.pt`, per-epoch snapshots, and a resume state file).

---

## Inference

Load a checkpoint in the browser server:

```bash
python run_play.py --model agents/basic_cnn/checkpoints/basic_cnn_latest.pt
```

`PolicyAgent` in `gopet/web/agents.py` loads the saved `nn.Module`, encodes the board, runs a forward pass, masks illegal moves, and picks the highest-probability legal action.

---

## Limitations

This is a **baseline supervised policy**. It learns to imitate training games but does not search ahead. Expect play that is better than random but far below strong human or engine level. Future improvements could add more input planes (liberties, ko), a value head, or MCTS.
