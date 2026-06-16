"""Load and save per-epoch training metrics as JSON."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: Optional[float] = None
    val_acc: Optional[float] = None
    time_sec: Optional[float] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpochRecord:
        return cls(
            epoch=int(data["epoch"]),
            train_loss=float(data["train_loss"]),
            train_acc=float(data["train_acc"]),
            val_loss=None if data.get("val_loss") is None else float(data["val_loss"]),
            val_acc=None if data.get("val_acc") is None else float(data["val_acc"]),
            time_sec=None if data.get("time_sec") is None else float(data["time_sec"]),
        )


@dataclass
class TrainingHistory:
    checkpoint: str
    training_config: dict[str, Any] = field(default_factory=dict)
    epochs: list[EpochRecord] = field(default_factory=list)

    def append(self, record: EpochRecord) -> None:
        self.epochs = [e for e in self.epochs if e.epoch != record.epoch]
        self.epochs.append(record)
        self.epochs.sort(key=lambda e: e.epoch)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "training_config": self.training_config,
            "epochs": [asdict(e) for e in self.epochs],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingHistory:
        return cls(
            checkpoint=str(data.get("checkpoint", "")),
            training_config=dict(data.get("training_config", {})),
            epochs=[EpochRecord.from_dict(e) for e in data.get("epochs", [])],
        )


def history_path_for_checkpoint(checkpoint_path: Path) -> Path:
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.parent / f"{checkpoint_path.stem}_history.json"


def load_history(path: Path) -> TrainingHistory:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        return TrainingHistory.from_dict(json.load(handle))


def save_history(history: TrainingHistory, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(history.to_dict(), handle, indent=2)
        handle.write("\n")


def load_or_create_history(
    checkpoint_path: Path,
    training_config: dict[str, Any],
    *,
    reset: bool = False,
) -> TrainingHistory:
    path = history_path_for_checkpoint(checkpoint_path)
    if reset or not path.exists():
        return TrainingHistory(
            checkpoint=str(checkpoint_path),
            training_config=training_config,
            epochs=[],
        )
    history = load_history(path)
    history.training_config.update(training_config)
    return history
