"""Tests for cnn_triad history aggregation into bundle history."""

from __future__ import annotations

from pathlib import Path

from agents.cnn_triad.train import aggregate_triad_history
from inspect_training.history import EpochRecord, TrainingHistory, history_path_for_checkpoint, save_history


def test_aggregate_triad_history_writes_epochs(tmp_path: Path) -> None:
    corner = tmp_path / "cnn_triad_corner.pt"
    side = tmp_path / "cnn_triad_side.pt"
    center = tmp_path / "cnn_triad_center.pt"
    bundle = tmp_path / "cnn_triad.pt"
    corner.write_bytes(b"")
    side.write_bytes(b"")
    center.write_bytes(b"")
    bundle.write_bytes(b"")

    def make_history(path: Path, bias: float) -> TrainingHistory:
        h = TrainingHistory(checkpoint=str(path), training_config={}, epochs=[])
        h.append(EpochRecord(epoch=1, train_loss=1.0 + bias, train_acc=0.1 + bias))
        h.append(EpochRecord(epoch=2, train_loss=2.0 + bias, train_acc=0.2 + bias))
        return h

    save_history(make_history(corner, 0.0), history_path_for_checkpoint(corner))
    save_history(make_history(side, 1.0), history_path_for_checkpoint(side))
    save_history(make_history(center, 2.0), history_path_for_checkpoint(center))

    agg = aggregate_triad_history(
        bundle_checkpoint=bundle,
        corner_checkpoint=corner,
        side_checkpoint=side,
        center_checkpoint=center,
        training_config={"agent_id": "cnn_triad"},
    )
    assert [e.epoch for e in agg.epochs] == [1, 2]
    # mean of (1.0,2.0,3.0) = 2.0 at epoch 1
    assert abs(agg.epochs[0].train_loss - 2.0) < 1e-6

