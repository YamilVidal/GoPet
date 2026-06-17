#!/usr/bin/env python
"""Print a summary of a training checkpoint and metrics history."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from inspect_training.agents import add_agent_arguments, resolve_checkpoint
from inspect_training.history import history_path_for_checkpoint, load_history


def summarize_training_state(state_path: Path) -> dict:
    payload = torch.load(state_path, map_location="cpu")
    return {
        "epoch": payload.get("epoch"),
        "interrupted": payload.get("interrupted", False),
        "metrics": payload.get("metrics"),
        "training_config": payload.get("training_config", {}),
    }


def print_summary(*, checkpoint: Path, history_path: Path, state_path: Path) -> None:
    print(f"Checkpoint: {checkpoint}")
    print(f"History:    {history_path} ({'found' if history_path.exists() else 'missing'})")
    print(f"State:      {state_path} ({'found' if state_path.exists() else 'missing'})")
    print()

    if state_path.exists():
        info = summarize_training_state(state_path)
        print("Latest training state")
        print(f"  epoch:       {info['epoch']}")
        print(f"  interrupted: {info['interrupted']}")
        metrics = info.get("metrics") or {}
        if metrics:
            print(f"  train_loss:  {metrics.get('train_loss')}")
            print(f"  train_acc:   {metrics.get('train_acc')}")
            print(f"  val_loss:    {metrics.get('val_loss')}")
            print(f"  val_acc:     {metrics.get('val_acc')}")
        config = info.get("training_config") or {}
        if config:
            print("  config:")
            for key in sorted(config):
                print(f"    {key}: {config[key]}")
        print()

    if history_path.exists():
        history = load_history(history_path)
        print(f"History epochs: {len(history.epochs)}")
        if history.epochs:
            first = history.epochs[0]
            last = history.epochs[-1]
            print(
                f"  first epoch {first.epoch}: "
                f"loss={first.train_loss:.4f} acc={first.train_acc:.4f}"
            )
            print(
                f"  last epoch  {last.epoch}: "
                f"loss={last.train_loss:.4f} acc={last.train_acc:.4f}"
            )
            if last.val_loss is not None:
                print(f"  last val: loss={last.val_loss:.4f} acc={last.val_acc:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize agent training checkpoint")
    add_agent_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = resolve_checkpoint(args.agent, args.checkpoint)
    stem = checkpoint.stem
    directory = checkpoint.parent
    history_path = history_path_for_checkpoint(checkpoint)
    state_path = directory / f"{stem}_training_state.pt"
    print_summary(checkpoint=checkpoint, history_path=history_path, state_path=state_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
