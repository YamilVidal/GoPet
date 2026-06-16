#!/usr/bin/env python
"""Plot training and validation loss/accuracy from a training history file."""

from __future__ import annotations

import argparse
from pathlib import Path

from inspect_training.history import TrainingHistory, history_path_for_checkpoint, load_history


def plot_history(
    history: TrainingHistory,
    *,
    output: Path,
    show: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    if not history.epochs:
        raise ValueError("History has no epoch records to plot.")

    epochs = [e.epoch for e in history.epochs]
    train_loss = [e.train_loss for e in history.epochs]
    train_acc = [e.train_acc for e in history.epochs]
    has_val = any(e.val_loss is not None for e in history.epochs)
    val_loss = [e.val_loss for e in history.epochs]
    val_acc = [e.val_acc for e in history.epochs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    title = Path(history.checkpoint).stem
    fig.suptitle(f"Training metrics: {title}")

    axes[0].plot(epochs, train_loss, marker="o", label="train")
    if has_val:
        axes[0].plot(epochs, val_loss, marker="o", label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Cross-entropy loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, marker="o", label="train")
    if has_val:
        axes[1].plot(epochs, val_acc, marker="o", label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Move prediction accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Saved plot to {output}")
    if show:
        plt.show()
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot training metrics from history JSON")
    parser.add_argument(
        "--history",
        default=None,
        help="Path to *_history.json (default: derived from --checkpoint)",
    )
    parser.add_argument(
        "--checkpoint",
        default="agents/basic_cnn/checkpoints/basic_cnn.pt",
        help="Checkpoint path used to locate *_history.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path (default: next to history file, .png)",
    )
    parser.add_argument("--show", action="store_true", help="Display plot interactively")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    history_path = Path(args.history) if args.history else history_path_for_checkpoint(Path(args.checkpoint))
    if not history_path.exists():
        print(f"History file not found: {history_path}")
        print("Run training after this update, or pass --history explicitly.")
        return 1

    history = load_history(history_path)
    output = Path(args.output) if args.output else history_path.with_suffix(".png")
    plot_history(history, output=output, show=args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
