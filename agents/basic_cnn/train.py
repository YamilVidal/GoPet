#!/usr/bin/env python
"""Train the basic CNN policy network on SGF replay data."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from agents.basic_cnn.data import load_or_build_dataset
from agents.basic_cnn.model import BasicPolicyCNN
from gopet.encoding import NUM_BASIC_PLANES
from inspect_training.history import (
    EpochRecord,
    history_path_for_checkpoint,
    load_or_create_history,
    save_history,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SGF_DIR = ROOT / "Games" / "jgdb" / "sgf" / "train"
DEFAULT_VAL_DIR = ROOT / "Games" / "jgdb" / "sgf" / "val"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
FIXTURE_SGF_DIR = ROOT / "tests" / "fixtures" / "sgf"


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: Optional[float] = None
    val_acc: Optional[float] = None


def checkpoint_paths(checkpoint_path: Path) -> dict[str, Path]:
    """Derive related checkpoint file paths from the main checkpoint path."""
    checkpoint_path = Path(checkpoint_path)
    stem = checkpoint_path.stem
    suffix = checkpoint_path.suffix or ".pt"
    directory = checkpoint_path.parent
    return {
        "final": checkpoint_path,
        "latest": directory / f"{stem}_latest{suffix}",
        "training_state": directory / f"{stem}_training_state{suffix}",
    }


def epoch_checkpoint_path(checkpoint_path: Path, epoch: int) -> Path:
    checkpoint_path = Path(checkpoint_path)
    stem = checkpoint_path.stem
    suffix = checkpoint_path.suffix or ".pt"
    return checkpoint_path.parent / f"{stem}_epoch_{epoch:03d}{suffix}"


def save_playable_model(model: nn.Module, path: Path) -> None:
    """Save a PolicyAgent-compatible model checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, path)


def save_training_state(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: EpochMetrics,
    checkpoint_path: Path,
    training_config: dict[str, Any],
    interrupted: bool = False,
) -> None:
    paths = checkpoint_paths(checkpoint_path)
    payload = {
        "epoch": metrics.epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": asdict(metrics),
        "training_config": training_config,
        "interrupted": interrupted,
    }
    torch.save(payload, paths["training_state"])


def save_epoch_checkpoints(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: EpochMetrics,
    checkpoint_path: Path,
    training_config: dict[str, Any],
    interrupted: bool = False,
) -> None:
    paths = checkpoint_paths(checkpoint_path)
    epoch_path = epoch_checkpoint_path(checkpoint_path, metrics.epoch)
    save_playable_model(model, paths["final"])
    save_playable_model(model, paths["latest"])
    save_playable_model(model, epoch_path)
    save_training_state(
        model=model,
        optimizer=optimizer,
        metrics=metrics,
        checkpoint_path=checkpoint_path,
        training_config=training_config,
        interrupted=interrupted,
    )

    status = "interrupted" if interrupted else "completed"
    print(
        f"Saved {status} checkpoints: "
        f"{paths['final'].name}, {paths['latest'].name}, "
        f"{epoch_path.name}, {paths['training_state'].name}"
    )


def load_training_state(
    resume_path: Path,
    *,
    board_size: int,
    learning_rate: float,
    device: torch.device,
) -> Tuple[BasicPolicyCNN, torch.optim.Optimizer, int, Optional[EpochMetrics]]:
    payload = torch.load(resume_path, map_location=device)
    model = BasicPolicyCNN(in_planes=NUM_BASIC_PLANES, board_size=board_size).to(device)
    model.load_state_dict(payload["model_state_dict"])

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(payload["optimizer_state_dict"])

    completed_epoch = int(payload["epoch"])
    metrics_data = payload.get("metrics")
    metrics = EpochMetrics(**metrics_data) if metrics_data else None
    return model, optimizer, completed_epoch, metrics


def resolve_sgf_directory(explicit: Optional[str], fallback: Path) -> Path:
    if explicit:
        return Path(explicit)
    if fallback.exists():
        return fallback
    if FIXTURE_SGF_DIR.exists():
        return FIXTURE_SGF_DIR
    return fallback


def configure_cpu_threads(num_threads: Optional[int]) -> None:
    if num_threads is not None:
        torch.set_num_threads(num_threads)


def make_dataloader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    val_fraction: float = 0.0,
    seed: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    x = torch.from_numpy(features)
    y = torch.from_numpy(labels).long()
    dataset = TensorDataset(x, y)

    if val_fraction <= 0.0 or len(dataset) < 2:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader, None

    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def format_duration_hms(seconds: float) -> str:
    total = int(round(max(0.0, seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _estimate_remaining_seconds(elapsed: float, percent: int) -> float:
    if percent <= 0:
        return 0.0
    return elapsed * (100 - percent) / percent


def _report_batch_progress(
    batch_index: int,
    num_batches: int,
    *,
    phase: str,
    next_progress: int,
    phase_start: float,
) -> int:
    if num_batches <= 0:
        return next_progress
    percent = int(100 * batch_index / num_batches)
    elapsed = time.time() - phase_start
    while next_progress <= percent and next_progress <= 100:
        remaining = _estimate_remaining_seconds(elapsed, next_progress)
        line = f"  {phase}: {next_progress:3d}%  ETA {format_duration_hms(remaining)}"
        print(f"{line:<48}", end="\r", flush=True)
        next_progress += 5
    return next_progress


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    *,
    phase: str = "train",
    show_progress: bool = True,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_batches = len(loader)
    next_progress = 5
    phase_start = time.time()

    for batch_index, (features, labels) in enumerate(loader, start=1):
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += batch_size

        if show_progress:
            next_progress = _report_batch_progress(
                batch_index,
                num_batches,
                phase=phase,
                next_progress=next_progress,
                phase_start=phase_start,
            )

    if show_progress and num_batches > 0:
        print()

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def train_model(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: Optional[np.ndarray],
    val_labels: Optional[np.ndarray],
    board_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    checkpoint_path: Path,
    resume_path: Optional[Path] = None,
    training_config: Optional[dict[str, Any]] = None,
) -> BasicPolicyCNN:
    train_loader, holdout_loader = make_dataloader(
        train_features,
        train_labels,
        batch_size=batch_size,
        shuffle=True,
        val_fraction=0.1 if val_features is None else 0.0,
    )

    val_loader = None
    if val_features is not None and len(val_features):
        val_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(val_features),
                torch.from_numpy(val_labels).long(),
            ),
            batch_size=batch_size,
            shuffle=False,
        )
    elif holdout_loader is not None:
        val_loader = holdout_loader

    if training_config is None:
        training_config = {
            "board_size": board_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_samples": len(train_features),
        }

    start_epoch = 1
    if resume_path is not None:
        model, optimizer, completed_epoch, resume_metrics = load_training_state(
            resume_path,
            board_size=board_size,
            learning_rate=learning_rate,
            device=device,
        )
        start_epoch = completed_epoch + 1
        metrics_history = load_or_create_history(
            checkpoint_path,
            training_config,
            reset=False,
        )
        print(f"Resuming from {resume_path} after epoch {completed_epoch}")
        if resume_metrics is not None:
            print(
                f"Previous epoch metrics: "
                f"loss={resume_metrics.train_loss:.4f} acc={resume_metrics.train_acc:.4f}"
            )
        if start_epoch > epochs:
            print(f"Training already complete ({completed_epoch}/{epochs} epochs).")
            save_playable_model(model, checkpoint_path)
            return model
    else:
        model = BasicPolicyCNN(in_planes=NUM_BASIC_PLANES, board_size=board_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        metrics_history = load_or_create_history(
            checkpoint_path,
            training_config,
            reset=True,
        )

    criterion = nn.CrossEntropyLoss()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Training samples: {len(train_features):,}")
    if val_loader is not None:
        print(f"Validation samples: {len(val_loader.dataset):,}")

    last_metrics: Optional[EpochMetrics] = None
    current_epoch = start_epoch
    try:
        for epoch in range(start_epoch, epochs + 1):
            current_epoch = epoch
            epoch_start = time.time()
            print(f"Epoch {epoch}/{epochs} train")
            train_loss, train_acc = run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                phase="train",
            )

            val_loss: Optional[float] = None
            val_acc: Optional[float] = None
            if val_loader is not None:
                print(f"Epoch {epoch}/{epochs} val")
                with torch.no_grad():
                    val_loss, val_acc = run_epoch(
                        model,
                        val_loader,
                        criterion,
                        None,
                        device,
                        phase="val",
                    )

            elapsed = time.time() - epoch_start
            message = (
                f"Epoch {epoch}/{epochs} "
                f"loss={train_loss:.4f} acc={train_acc:.4f} "
                f"time={format_duration_hms(elapsed)}"
            )
            if val_loader is not None:
                message += f" val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            print(message)

            last_metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
            )
            metrics_history.append(
                EpochRecord(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    time_sec=elapsed,
                )
            )
            save_history(metrics_history, history_path_for_checkpoint(checkpoint_path))
            save_epoch_checkpoints(
                model=model,
                optimizer=optimizer,
                metrics=last_metrics,
                checkpoint_path=checkpoint_path,
                training_config=training_config,
            )
    except KeyboardInterrupt:
        paths = checkpoint_paths(checkpoint_path)
        print(f"\nTraining interrupted during epoch {current_epoch} — saving checkpoint...")
        save_playable_model(model, paths["final"])
        save_playable_model(model, paths["latest"])
        if last_metrics is not None:
            save_training_state(
                model=model,
                optimizer=optimizer,
                metrics=last_metrics,
                checkpoint_path=checkpoint_path,
                training_config=training_config,
                interrupted=True,
            )
            print(
                f"Saved playable model to {paths['final']} and {paths['latest']}. "
                f"Resume from epoch {last_metrics.epoch + 1} with "
                f"--resume {paths['training_state']}"
            )
        else:
            print(
                f"Saved playable model to {paths['final']} and {paths['latest']}. "
                "No completed epoch to resume from yet."
            )
        raise SystemExit(130)

    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train BasicPolicyCNN on SGF data")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick pipeline smoke test on a tiny SGF subset",
    )
    parser.add_argument("--sgf-dir", default=None, help="Training SGF directory")
    parser.add_argument("--val-dir", default=None, help="Validation SGF directory")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Cached .npy directory")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Output checkpoint path (default depends on --test)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from a *_training_state.pt checkpoint",
    )
    parser.add_argument("--max-games", type=int, default=None, help="Cap SGF files processed")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--board-size", type=int, default=19)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=None, help="PyTorch CPU thread count")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild cached .npy files")
    parser.add_argument("--preprocess-only", action="store_true", help="Build datasets and exit")
    return parser


def apply_test_defaults(args: argparse.Namespace) -> None:
    if not args.test:
        return
    args.max_games = args.max_games or 25
    args.epochs = args.epochs or 2
    args.batch_size = args.batch_size or 32
    args.data_dir = str(Path(args.data_dir) / "test")
    if args.checkpoint is None:
        args.checkpoint = str(DEFAULT_CHECKPOINT_DIR / "test.pt")


def apply_full_defaults(args: argparse.Namespace) -> None:
    args.epochs = args.epochs or 5
    args.batch_size = args.batch_size or 64
    args.max_games = args.max_games or 4000
    if args.checkpoint is None:
        args.checkpoint = str(DEFAULT_CHECKPOINT_DIR / "basic_cnn.pt")


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.test:
        apply_test_defaults(args)
    else:
        apply_full_defaults(args)

    configure_cpu_threads(args.threads)
    device = torch.device("cpu")

    train_sgf_dir = resolve_sgf_directory(args.sgf_dir, DEFAULT_SGF_DIR)
    val_sgf_dir = resolve_sgf_directory(args.val_dir, DEFAULT_VAL_DIR)

    data_dir = Path(args.data_dir)
    train_prefix = "train"
    val_prefix = "val"

    print(f"Training SGF directory: {train_sgf_dir}")
    if not train_sgf_dir.exists():
        print(f"Training directory not found: {train_sgf_dir}", file=sys.stderr)
        return 1

    train_features, train_labels = load_or_build_dataset(
        train_sgf_dir,
        data_directory=data_dir,
        prefix=train_prefix,
        max_games=args.max_games,
        seed=args.seed,
        force_rebuild=args.force_rebuild,
    )
    if len(train_features) == 0:
        print("No training positions were generated.", file=sys.stderr)
        return 1

    print(
        f"Loaded training data: {len(train_features):,} positions "
        f"from up to {args.max_games} games"
    )

    val_features = None
    val_labels = None
    if not args.test or args.val_dir is not None:
        if val_sgf_dir.exists():
            val_max_games = max(1, min(200, (args.max_games or 200) // 10))
            val_features, val_labels = load_or_build_dataset(
                val_sgf_dir,
                data_directory=data_dir,
                prefix=val_prefix,
                max_games=val_max_games,
                seed=args.seed + 1,
                force_rebuild=args.force_rebuild,
            )
            print(f"Loaded validation data: {len(val_features):,} positions")
    else:
        print("Validation: using 10% holdout from training data")

    if args.preprocess_only:
        print("Preprocessing complete.")
        return 0

    train_model(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        board_size=args.board_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        checkpoint_path=Path(args.checkpoint),
        resume_path=Path(args.resume) if args.resume else None,
        training_config={
            "board_size": args.board_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "max_games": args.max_games,
            "train_samples": len(train_features),
            "seed": args.seed,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
