# BasicPolicyCNN (GoPet)

A small convolutional Go policy network for **[GoPet](https://github.com/)** — a personal *pet project* to train and play **19×19 Go** on a laptop CPU, without GPUs or large-scale infrastructure.

This agent learns from professional SGF games (supervised imitation) and plays through the GoPet browser UI. It is a **baseline policy net**, not a strong engine: no MCTS, no value head, no distributed training.

---

## What this is

GoPet is an experiment in building a minimal Go stack end-to-end:

- Fast board engine and feature encoding (`gopet/`)
- SGF replay → NumPy training data
- A tiny CNN that predicts the next move
- Browser play against the trained model
- Optional territory scoring (goscorer) for resignation in human games

`basic_cnn` is the first and simplest agent in that lineup — deliberately small (~66k parameters) so it can be trained in a few hours on CPU.

---

## Quick start

### Train

```bash
# From repo root
pip install -r requirements.txt

# Full run (example: ~5k games, several epochs)
python run_train.py --max-games 5000 --epochs 10 --force-rebuild

# Resume more epochs on cached data
python run_train.py --epochs 20 --resume agents/basic_cnn/checkpoints/basic_cnn_training_state.pt
```

Training data lives under `Games/` (not in git). Cached shards and checkpoints go to `agents/basic_cnn/data/` and `agents/basic_cnn/checkpoints/` (also gitignored).

Estimate wall-clock time before a long run:

```bash
python estimate_training_time.py --games 5000 --epochs 10 --budget-hours 6
```

Inspect metrics after training:

```bash
python run_inspect_training.py summary
python run_inspect_training.py plot
```

### Play

```bash
python run_play.py --model agents/basic_cnn/checkpoints/basic_cnn_latest.pt
```

Open http://127.0.0.1:5000/ and select the **policy** agent.

---

## Model at a glance

| Property | Value |
|----------|-------|
| Board | 19×19 |
| Input | 5 feature planes (`gopet.encoding`) |
| Output | 362 logits (361 points + pass) |
| Parameters | ~66k |
| Training | Supervised CE loss on expert moves |
| Device | CPU |

Implementation: [`model.py`](model.py)

---

## Architecture

```
Input [B, 5, 19, 19]
        │
        ▼
┌───────────────────────────────────┐
│  Trunk (spatial size stays 19×19) │
│  Conv 5→32→32→64→64  (3×3, ReLU)  │
└───────────────────────────────────┘
        │
        ├──────────────────────────────┐
        ▼                              ▼
  Conv 1×1 → 361 board logits    GAP + Linear → pass logit
        └────────── concat ────────────┘
                      ▼
              [B, 362] logits
```

### Input planes

| Plane | Content |
|-------|---------|
| 0 | Current player's stones |
| 1 | Opponent's stones |
| 2 | Empty intersections |
| 3 | 1 if Black to move |
| 4 | 1 if White to move |

### Action index

| Index | Meaning |
|-------|---------|
| `0 … 360` | `row × 19 + col` |
| `361` | Pass |

Illegal moves are masked at inference (`gopet.encoding.mask_policy_logits`).

---

## Project layout

| File | Role |
|------|------|
| [`model.py`](model.py) | `BasicPolicyCNN` definition |
| [`train.py`](train.py) | Training loop, checkpoints, resume |
| [`data.py`](data.py) | SGF subsampling and `.npy` cache |
| [`../../run_train.py`](../../run_train.py) | CLI entry point |
| [`../../run_play.py`](../../run_play.py) | Browser play server |
| [`../../inspect_training/`](../../inspect_training/) | Loss/accuracy plots |
| [`../../score_estimation/`](../../score_estimation/) | Territory + seki scoring |

---

## Behaviour in human games

When playing a human via the web UI, the policy agent can **resign** if:

- move count ≥ 120, and
- estimated score (goscorer) shows a deficit ≥ 60 points

Resignation is disabled for bot-vs-bot / self-play runs.

---

## Limitations

This is a learning project, not a production Go engine:

- Imitates training data; does not search ahead
- No capture tracking in score estimation yet
- Mid-game score estimates can be wrong in unsettled fights
- Strength is “better than random”, not dan-level

Possible next steps: ownership head, more input planes, MCTS, self-play.

---

## License & data

- GoPet code: see repository root
- SGF training data (e.g. JGDB): download separately into `Games/` — not shipped with the repo
- Vendored [goscorer](https://github.com/lightvector/goscorer) (MIT) in `score_estimation/`
