#!/usr/bin/env python
"""Train the cnn_triad regional policy agent on SGF replay data."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from agents.basic_cnn.train import (
    configure_cpu_threads,
    make_dataloader,
    resolve_sgf_directory,
    run_epoch,
    save_epoch_checkpoints,
    save_playable_model,
    load_training_state,
    checkpoint_paths,
    EpochMetrics,
    DEFAULT_SGF_DIR,
    DEFAULT_VAL_DIR,
    FIXTURE_SGF_DIR,
)
from agents.cnn_triad.data import (
    build_all_heads,
    head_cache_prefix,
)
from agents.cnn_triad.model import center_net, corner_net, side_net
from agents.cnn_triad.regions import TRIAD_BOARD_SIZE, Region, require_triad_board_size
from agents.cnn_triad.triad import assemble_triad_module
from gopet.data.training_cache import cache_is_valid, dataset_directory, make_dataset_id, record_agent_usage
from gopet.encoding import NUM_BASIC_PLANES
from inspect_training.history import (
    EpochRecord,
    TrainingHistory,
    history_path_for_checkpoint,
    load_history,
    load_or_create_history,
    save_history,
)

AGENT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = AGENT_DIR / "checkpoints"

HEAD_CHOICES = ("corner", "side", "center", "all")


@dataclass(frozen=True)
class HeadConfig:
    region: Region
    stem: str
    model_factory: Callable[[], nn.Module]


HEADS = {
    "corner": HeadConfig(Region.CORNER, "cnn_triad_corner", corner_net),
    "side": HeadConfig(Region.SIDE, "cnn_triad_side", side_net),
    "center": HeadConfig(Region.CENTER, "cnn_triad_center", center_net),
}


def train_head(
    *,
    head: HeadConfig,
    train_features,
    train_labels,
    val_features,
    val_labels,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    checkpoint_path: Path,
    resume_path: Optional[Path],
    training_config: dict,
) -> nn.Module:
    train_loader, holdout = make_dataloader(
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
    elif holdout is not None:
        val_loader = holdout

    start_epoch = 1
    if resume_path is not None:
        model, optimizer, completed_epoch, _ = load_training_state(
            resume_path,
            board_size=TRIAD_BOARD_SIZE,
            learning_rate=learning_rate,
            device=device,
            model_factory=lambda _: head.model_factory(),
            training_config=training_config,
        )
        start_epoch = completed_epoch + 1
        history = load_or_create_history(checkpoint_path, training_config, reset=False)
        print(f"Resuming {head.stem} from {resume_path} after epoch {completed_epoch}")
    else:
        model = head.model_factory().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        history = load_or_create_history(checkpoint_path, training_config, reset=True)

    criterion = nn.CrossEntropyLoss()
    print(f"Training {head.stem}: {sum(p.numel() for p in model.parameters()):,} parameters, {len(train_features):,} samples")

    for epoch in range(start_epoch, epochs + 1):
        print(f"{head.stem} epoch {epoch}/{epochs} train")
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, phase="train")
        val_loss = val_acc = None
        if val_loader is not None:
            print(f"{head.stem} epoch {epoch}/{epochs} val")
            with torch.no_grad():
                val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device, phase="val")

        msg = f"{head.stem} epoch {epoch}/{epochs} loss={train_loss:.4f} acc={train_acc:.4f}"
        if val_loader is not None:
            msg += f" val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        print(msg)

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
        )
        from inspect_training.history import EpochRecord

        history.append(
            EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                time_sec=0.0,
            )
        )
        save_history(history, history_path_for_checkpoint(checkpoint_path))
        save_epoch_checkpoints(
            model=model,
            optimizer=optimizer,
            metrics=metrics,
            checkpoint_path=checkpoint_path,
            training_config=training_config,
        )

    return model


def save_triad_bundle(
    corner: nn.Module,
    side: nn.Module,
    center: nn.Module,
    bundle_path: Path,
) -> None:
    module = assemble_triad_module(corner, side, center)
    save_playable_model(module, bundle_path)
    paths = checkpoint_paths(bundle_path)
    save_playable_model(module, paths["latest"])


def aggregate_triad_history(
    *,
    bundle_checkpoint: Path,
    corner_checkpoint: Path,
    side_checkpoint: Path,
    center_checkpoint: Path,
    training_config: dict,
) -> TrainingHistory:
    """Aggregate per-head histories into a bundle history for plotting.

    The plotting tools expect a single `*_history.json` derived from the bundle
    checkpoint path. We write one by averaging train/val metrics across heads
    per epoch (intersection of epoch numbers).
    """

    corner_history = load_history(history_path_for_checkpoint(corner_checkpoint))
    side_history = load_history(history_path_for_checkpoint(side_checkpoint))
    center_history = load_history(history_path_for_checkpoint(center_checkpoint))

    corner_epochs = {e.epoch: e for e in corner_history.epochs}
    side_epochs = {e.epoch: e for e in side_history.epochs}
    center_epochs = {e.epoch: e for e in center_history.epochs}

    shared_epochs = sorted(set(corner_epochs) & set(side_epochs) & set(center_epochs))
    history = TrainingHistory(checkpoint=str(bundle_checkpoint), training_config=dict(training_config), epochs=[])

    for epoch in shared_epochs:
        c = corner_epochs[epoch]
        s = side_epochs[epoch]
        m = center_epochs[epoch]

        def mean3(a: float, b: float, d: float) -> float:
            return (a + b + d) / 3.0

        val_loss = None
        val_acc = None
        if c.val_loss is not None and s.val_loss is not None and m.val_loss is not None:
            val_loss = mean3(c.val_loss, s.val_loss, m.val_loss)
        if c.val_acc is not None and s.val_acc is not None and m.val_acc is not None:
            val_acc = mean3(c.val_acc, s.val_acc, m.val_acc)

        time_sec = None
        if c.time_sec is not None and s.time_sec is not None and m.time_sec is not None:
            time_sec = mean3(c.time_sec, s.time_sec, m.time_sec)

        history.append(
            EpochRecord(
                epoch=epoch,
                train_loss=mean3(c.train_loss, s.train_loss, m.train_loss),
                train_acc=mean3(c.train_acc, s.train_acc, m.train_acc),
                val_loss=val_loss,
                val_acc=val_acc,
                time_sec=time_sec,
            )
        )
    return history


def ensure_head_caches(
    sgf_directory: Path,
    *,
    data_dir: Path,
    split: str,
    max_games: Optional[int],
    seed: int,
    board_size: int,
    force_rebuild: bool,
    dataset_id: str,
) -> None:
    all_valid = True
    for region in (Region.CORNER, Region.SIDE, Region.CENTER):
        prefix = head_cache_prefix(region, split)
        if not cache_is_valid(
            data_dir,
            prefix=prefix,
            sgf_directory=sgf_directory,
            max_games=max_games,
            seed=seed,
            board_size=board_size,
            num_planes=NUM_BASIC_PLANES,
        ):
            all_valid = False
            break
    if all_valid and not force_rebuild:
        return
    build_all_heads(
        sgf_directory,
        data_directory=data_dir,
        split=split,
        max_games=max_games,
        seed=seed,
        board_size=board_size,
        num_planes=NUM_BASIC_PLANES,
        force_rebuild=force_rebuild,
        dataset_id=dataset_id,
    )


def load_head_from_checkpoint(path: Path) -> nn.Module:
    if not path.exists():
        raise FileNotFoundError(f"Head checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, nn.Module):
        return payload
    raise ValueError(f"Unsupported checkpoint format: {path}")


def resolve_head_checkpoint(stem: str) -> Path:
    base = CHECKPOINT_DIR / f"{stem}.pt"
    latest = checkpoint_paths(base)["latest"]
    return latest if latest.exists() else base


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train cnn_triad regional policy agent")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--sgf-dir", default=None)
    parser.add_argument("--val-dir", default=None)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--checkpoint", default=None, help="Bundle output path (cnn_triad.pt)")
    parser.add_argument("--head", choices=HEAD_CHOICES, default="all")
    parser.add_argument("--resume", default=None, help="Resume a single head training_state")
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--board-size", type=int, default=TRIAD_BOARD_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--preprocess-only", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    require_triad_board_size(args.board_size)

    if args.test:
        args.max_games = args.max_games or 25
        args.epochs = args.epochs or 2
        args.batch_size = args.batch_size or 32
        if args.checkpoint is None:
            args.checkpoint = str(CHECKPOINT_DIR / "test_triad.pt")
    else:
        args.epochs = args.epochs or 5
        args.batch_size = args.batch_size or 64
        args.max_games = args.max_games or 4000
        if args.checkpoint is None:
            args.checkpoint = str(CHECKPOINT_DIR / "cnn_triad.pt")

    configure_cpu_threads(args.threads)
    device = torch.device("cpu")

    train_sgf_dir = resolve_sgf_directory(args.sgf_dir, DEFAULT_SGF_DIR)
    val_sgf_dir = resolve_sgf_directory(args.val_dir, DEFAULT_VAL_DIR)
    if not train_sgf_dir.exists():
        if args.test and FIXTURE_SGF_DIR.exists():
            train_sgf_dir = FIXTURE_SGF_DIR
        else:
            print(f"Training directory not found: {train_sgf_dir}", file=sys.stderr)
            return 1

    dataset_id = args.dataset_id or make_dataset_id(
        train_sgf_directory=train_sgf_dir,
        max_games=args.max_games,
        seed=args.seed,
        board_size=args.board_size,
        num_planes=NUM_BASIC_PLANES,
        test_mode=args.test,
    )
    if not dataset_id.startswith("cnn_triad_") and args.head == "all":
        dataset_id = f"cnn_triad_{dataset_id}"
    data_dir = Path(args.data_dir) if args.data_dir else dataset_directory(dataset_id)

    print(f"Agent: cnn_triad")
    print(f"Dataset id: {dataset_id}")
    print(f"Cached data directory: {data_dir}")

    ensure_head_caches(
        train_sgf_dir,
        data_dir=data_dir,
        split="train",
        max_games=args.max_games,
        seed=args.seed,
        board_size=args.board_size,
        force_rebuild=args.force_rebuild,
        dataset_id=dataset_id,
    )

    val_heads = None
    if val_sgf_dir.exists():
        val_max = max(1, min(200, (args.max_games or 200) // 10))
        try:
            ensure_head_caches(
                val_sgf_dir,
                data_dir=data_dir,
                split="val",
                max_games=val_max,
                seed=args.seed + 1,
                board_size=args.board_size,
                force_rebuild=args.force_rebuild,
                dataset_id=dataset_id,
            )
            val_heads = {
                region: (
                    np.load(data_dir / f"{head_cache_prefix(region, 'val')}_features.npy"),
                    np.load(data_dir / f"{head_cache_prefix(region, 'val')}_labels.npy"),
                )
                for region in (Region.CORNER, Region.SIDE, Region.CENTER)
            }
        except Exception:
            val_heads = None

    if args.preprocess_only:
        print("Preprocessing complete.")
        return 0

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    heads_to_train = list(HEADS.keys()) if args.head == "all" else [args.head]
    trained: dict[str, nn.Module] = {}
    bundle_path = Path(args.checkpoint)

    for head_name in heads_to_train:
        head = HEADS[head_name]
        prefix = head_cache_prefix(head.region, "train")
        train_features = np.load(data_dir / f"{prefix}_features.npy")
        train_labels = np.load(data_dir / f"{prefix}_labels.npy")

        val_features = val_labels = None
        if val_heads is not None:
            val_features, val_labels = val_heads[head.region]

        head_ckpt = CHECKPOINT_DIR / f"{head.stem}.pt"
        training_config = {
            "agent_id": "cnn_triad",
            "head": head_name,
            "dataset_id": dataset_id,
            "board_size": args.board_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "num_planes": NUM_BASIC_PLANES,
        }
        resume = Path(args.resume) if args.resume and args.head != "all" else None
        trained[head_name] = train_head(
            head=head,
            train_features=train_features,
            train_labels=train_labels,
            val_features=val_features,
            val_labels=val_labels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            checkpoint_path=head_ckpt,
            resume_path=resume,
            training_config=training_config,
        )

    # Load any heads not trained this run from latest checkpoints
    if args.head == "all":
        try:
            corner = trained.get("corner") or load_head_from_checkpoint(resolve_head_checkpoint("cnn_triad_corner"))
            side = trained.get("side") or load_head_from_checkpoint(resolve_head_checkpoint("cnn_triad_side"))
            center = trained.get("center") or load_head_from_checkpoint(resolve_head_checkpoint("cnn_triad_center"))
            save_triad_bundle(corner, side, center, bundle_path)

            # Write an aggregated history for the bundle checkpoint so
            # inspect_training/plot_metrics.py can find and plot it.
            bundle_history = aggregate_triad_history(
                bundle_checkpoint=bundle_path,
                corner_checkpoint=resolve_head_checkpoint("cnn_triad_corner"),
                side_checkpoint=resolve_head_checkpoint("cnn_triad_side"),
                center_checkpoint=resolve_head_checkpoint("cnn_triad_center"),
                training_config={
                    "agent_id": "cnn_triad",
                    "dataset_id": dataset_id,
                    "board_size": args.board_size,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "num_planes": NUM_BASIC_PLANES,
                },
            )
            save_history(bundle_history, history_path_for_checkpoint(bundle_path))

            record_agent_usage(
                agent_id="cnn_triad",
                dataset_id=dataset_id,
                dataset_dir=data_dir,
                usage_info={
                    "checkpoint": str(bundle_path.resolve()),
                    "epochs": args.epochs,
                    "max_games": args.max_games,
                },
            )
            print(f"Saved triad bundle to {bundle_path}")
        except FileNotFoundError as exc:
            print(f"Could not assemble triad bundle: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
