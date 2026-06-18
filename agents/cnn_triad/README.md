# cnn_triad

Three regional policy networks (same 5×5 conv architecture as `basic_cnn_5x5`) trained on canonical crops of a 19×19 board:

| Head | Region | Crop | Symmetric copies at inference |
|------|--------|------|-------------------------------|
| Corner | 5×5 corners | 5×5 | 4 (rotated to align) |
| Side | edge strips | 9×5 | 4 (rotated to align) |
| Center | middle | 9×9 | 1 (no rotation) |

At play time, all crops are evaluated, logits are mapped to global board coordinates, illegal moves are masked, and the move with the highest probability wins (including pass).

## Canonical rotation

**Corners and sides** are both rotated so every symmetric copy of the board maps to the same canonical orientation before entering the net. The **center** head uses a single fixed crop with no rotation.

| Head | Physical copies | Rotation |
|------|-----------------|----------|
| Corner | NW, NE, SE, SW | 0°, 90°, 180°, 270° — board corner always at the same local corner of the 5×5 input |
| Side | top, bottom, left, right | rotated so the outer edge is always the same row of the 9×5 input |
| Center | one 9×9 block | none |

- **Training**: each move is assigned to the matching copy (e.g. a NE corner move uses the NE slice, rotated to canonical form). Labels are in local crop coordinates.
- **Inference**: all four corner crops and all four side crops are run through the same corner/side net respectively; logits are un-rotated back to global vertices and merged. The center net runs once.

See `agents/cnn_triad/regions.py` for the exact `CropSpec` definitions.

## Board layout (19×19)

```
+-------+-----------+-------+
| 5×5   | 9×5 side  | 5×5   |
| corner| (top)     | corner|
+-------+-----------+-------+
| 5×9   |           | 5×9   |
| side  |  9×9      | side  |
| (left)|  center   |(right)|
+-------+-----------+-------+
| 5×5   | 9×5 side  | 5×5   |
| corner| (bottom)  | corner|
+-------+-----------+-------+
```

## Training from existing caches

```bash
# Preprocess three regional datasets
python run_train.py --agent cnn_triad --preprocess-only --board-size 19

# Train all heads and save bundle
python run_train.py --agent cnn_triad --board-size 19 --epochs 5

# Train a single head
python run_train.py --agent cnn_triad --head corner --epochs 3
```

Quick smoke test on fixture SGFs:

```bash
python run_train.py --agent cnn_triad --test
```

If preprocessing is already done, point at the cache and omit `--force-rebuild`:

```bash
python run_train.py --agent cnn_triad --data-dir training_data/datasets/<your_dataset_id> --board-size 19 --epochs 5
```

Use the same `--sgf-dir`, `--max-games`, and `--seed` as when the cache was built so the manifest validates.

## Checkpoints

Under `agents/cnn_triad/checkpoints/`:

| File | Purpose |
|------|---------|
| `cnn_triad_corner.pt` | Corner head (playable single-head checkpoint) |
| `cnn_triad_side.pt` | Side head |
| `cnn_triad_center.pt` | Center head |
| `cnn_triad.pt` | **Playable bundle** used by web UI and `run_match.py` |
| `cnn_triad_latest.pt` | Latest bundle |

Per-head `*_training_state.pt` files are for resume only.

Cached training arrays live under `training_data/datasets/cnn_triad_*` with prefixes `corner_train`, `side_train`, `center_train`.

## Evaluation

```bash
python run_match.py --agent-a cnn_triad --agent-b basic_cnn_5x5 --games 100 --board-size 19
```

`cnn_triad` only supports 19×19 boards.
