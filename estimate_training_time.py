#!/usr/bin/env python
"""Estimate total BasicPolicyCNN training time (preprocessing + training).

Calibrated from a local benchmark on this project:
  200 games, 1 epoch -> 40,673 train positions, 72.3 s training epoch,
  ~111 s preprocessing (with --force-rebuild).

Usage:
  python estimate_training_time.py --games 5000 --epochs 9
  python estimate_training_time.py --games 4000 5000 5500 --epochs 8 9 10
  python estimate_training_time.py --games 5000 --epochs 9 --cached-data
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

# Benchmark: 200 requested games, 192 usable after replay skips.
BENCHMARK_GAMES = 200
BENCHMARK_USABLE_GAMES = 192
BENCHMARK_TRAIN_POSITIONS = 40_673
BENCHMARK_TRAIN_EPOCH_SEC = 72.3
BENCHMARK_PREPROCESS_SEC = 110.8
BENCHMARK_VAL_GAMES = 20  # min(200, 200 // 10) from train.py
BENCHMARK_VAL_POSITIONS = 3_636
VAL_GAMES_CAP = 200


@dataclass(frozen=True)
class TimingEstimate:
    games: int
    epochs: int
    include_preprocess: bool
    usable_games: float
    train_positions: int
    val_games: int
    val_positions: int
    preprocess_sec: float
    train_epoch_sec: float
    val_epoch_sec: float

    @property
    def training_sec(self) -> float:
        return self.epochs * (self.train_epoch_sec + self.val_epoch_sec)

    @property
    def total_sec(self) -> float:
        return self.preprocess_sec + self.training_sec


def val_games_for(max_games: int) -> int:
    return max(1, min(VAL_GAMES_CAP, max_games // 10))


def estimate(
    games: int,
    epochs: int,
    *,
    include_preprocess: bool = True,
) -> TimingEstimate:
    usable_rate = BENCHMARK_USABLE_GAMES / BENCHMARK_GAMES
    positions_per_usable_game = BENCHMARK_TRAIN_POSITIONS / BENCHMARK_USABLE_GAMES
    sec_per_position = BENCHMARK_TRAIN_EPOCH_SEC / BENCHMARK_TRAIN_POSITIONS
    preprocess_per_game = BENCHMARK_PREPROCESS_SEC / BENCHMARK_GAMES
    val_positions_per_game = BENCHMARK_VAL_POSITIONS / BENCHMARK_VAL_GAMES

    usable_games = games * usable_rate
    train_positions = int(round(usable_games * positions_per_usable_game))

    val_game_count = val_games_for(games)
    val_positions = int(round(val_game_count * val_positions_per_game))

    preprocess_sec = preprocess_per_game * games if include_preprocess else 0.0
    train_epoch_sec = train_positions * sec_per_position
    val_epoch_sec = val_positions * sec_per_position

    return TimingEstimate(
        games=games,
        epochs=epochs,
        include_preprocess=include_preprocess,
        usable_games=usable_games,
        train_positions=train_positions,
        val_games=val_game_count,
        val_positions=val_positions,
        preprocess_sec=preprocess_sec,
        train_epoch_sec=train_epoch_sec,
        val_epoch_sec=val_epoch_sec,
    )


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, remainder = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def print_estimate(est: TimingEstimate) -> None:
    preprocess_label = "preprocessing" if est.include_preprocess else "preprocessing (skipped, cached data)"
    print(f"\nGames: {est.games:,}   Epochs: {est.epochs}")
    print(f"  Usable games (est.):     {est.usable_games:,.0f}")
    print(f"  Train positions (est.): {est.train_positions:,}")
    print(f"  Val games (est.):        {est.val_games:,}  ({est.val_positions:,} positions)")
    print(f"  {preprocess_label}:      {format_duration(est.preprocess_sec)}")
    print(
        f"  Per epoch (train + val): {format_duration(est.train_epoch_sec + est.val_epoch_sec)} "
        f"({format_duration(est.train_epoch_sec)} train, {format_duration(est.val_epoch_sec)} val)"
    )
    print(f"  All epochs:              {format_duration(est.training_sec)}")
    print(f"  TOTAL:                   {format_duration(est.total_sec)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate BasicPolicyCNN training time for games/epoch settings.",
    )
    parser.add_argument(
        "--games",
        type=int,
        nargs="+",
        default=[5000],
        help="Number of SGF games to use (default: 5000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=[9],
        help="Number of training epochs (default: 9)",
    )
    parser.add_argument(
        "--cached-data",
        action="store_true",
        help="Assume .npy cache exists and skip preprocessing time",
    )
    parser.add_argument(
        "--budget-hours",
        type=float,
        default=None,
        help="If set, mark configs that fit within this total-time budget",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    include_preprocess = not args.cached_data
    budget_sec = args.budget_hours * 3600 if args.budget_hours is not None else None

    print("BasicPolicyCNN training time estimate")
    print(
        f"Calibration: {BENCHMARK_GAMES} games / 1 epoch -> "
        f"{BENCHMARK_TRAIN_EPOCH_SEC:.1f}s train, "
        f"{BENCHMARK_PREPROCESS_SEC:.1f}s preprocess"
    )
    if args.cached_data:
        print("Mode: cached data (no preprocessing)")

    estimates = [
        estimate(games, epochs, include_preprocess=include_preprocess)
        for games in args.games
        for epochs in args.epochs
    ]

    for est in estimates:
        print_estimate(est)
        if budget_sec is not None:
            status = "within budget" if est.total_sec <= budget_sec else "over budget"
            print(f"  Budget {args.budget_hours:g}h: {status}")

    if len(estimates) > 1:
        print("\nSummary")
        print(f"{'Games':>8} {'Epochs':>7} {'Total':>12} {'Preprocess':>12} {'Training':>12}")
        for est in estimates:
            print(
                f"{est.games:8,d} {est.epochs:7d} "
                f"{format_duration(est.total_sec):>12} "
                f"{format_duration(est.preprocess_sec):>12} "
                f"{format_duration(est.training_sec):>12}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
