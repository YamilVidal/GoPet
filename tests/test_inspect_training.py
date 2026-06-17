from __future__ import annotations

import json
from pathlib import Path

from inspect_training.agents import default_checkpoint, resolve_checkpoint
from inspect_training.history import (
    EpochRecord,
    TrainingHistory,
    history_path_for_checkpoint,
    load_history,
    load_or_create_history,
    save_history,
)


def test_history_roundtrip(tmp_path: Path) -> None:
    checkpoint = tmp_path / "basic_cnn.pt"
    history = TrainingHistory(
        checkpoint=str(checkpoint),
        training_config={"epochs": 3},
        epochs=[
            EpochRecord(epoch=1, train_loss=2.0, train_acc=0.1, val_loss=1.8, val_acc=0.15),
            EpochRecord(epoch=2, train_loss=1.5, train_acc=0.2, val_loss=1.4, val_acc=0.22),
        ],
    )
    path = history_path_for_checkpoint(checkpoint)
    save_history(history, path)

    loaded = load_history(path)
    assert loaded.checkpoint == str(checkpoint)
    assert len(loaded.epochs) == 2
    assert loaded.epochs[-1].train_loss == 1.5


def test_append_replaces_same_epoch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    history = load_or_create_history(checkpoint, {"lr": 1e-3}, reset=True)
    history.append(EpochRecord(epoch=1, train_loss=1.0, train_acc=0.5))
    history.append(EpochRecord(epoch=1, train_loss=0.9, train_acc=0.55))
    assert len(history.epochs) == 1
    assert history.epochs[0].train_loss == 0.9


def test_history_path_for_checkpoint() -> None:
    path = history_path_for_checkpoint(Path("agents/basic_cnn/checkpoints/basic_cnn.pt"))
    assert path.name == "basic_cnn_history.json"


def test_default_checkpoint_for_agents() -> None:
    assert default_checkpoint("basic_cnn").name == "basic_cnn.pt"
    assert default_checkpoint("basic_cnn_5x5").name == "basic_cnn_5x5.pt"


def test_resolve_checkpoint_prefers_explicit_path() -> None:
    explicit = resolve_checkpoint("basic_cnn", "custom/model.pt")
    assert explicit == Path("custom/model.pt")
    resolved = resolve_checkpoint("basic_cnn_5x5", None)
    assert resolved.name in {"basic_cnn_5x5.pt", "basic_cnn_5x5_latest.pt"}
