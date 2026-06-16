"""Terminal progress reporting helpers."""

from __future__ import annotations

import time


def format_duration_hms(seconds: float) -> str:
    total = int(round(max(0.0, seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def estimate_remaining_seconds(elapsed: float, percent: int) -> float:
    if percent <= 0:
        return 0.0
    return elapsed * (100 - percent) / percent


def report_item_progress(
    item_index: int,
    total_items: int,
    *,
    phase: str,
    next_progress: int,
    phase_start: float,
    step_percent: int = 5,
) -> int:
    """Print progress at fixed percent steps. Returns updated next_progress."""

    if total_items <= 0:
        return next_progress

    percent = int(100 * item_index / total_items)
    elapsed = time.time() - phase_start
    while next_progress <= percent and next_progress <= 100:
        remaining = estimate_remaining_seconds(elapsed, next_progress)
        line = f"  {phase}: {next_progress:3d}%  ETA {format_duration_hms(remaining)}"
        print(f"{line:<48}", end="\r", flush=True)
        next_progress += step_percent
    return next_progress


def finish_progress_line(total_items: int, *, show_progress: bool) -> None:
    if show_progress and total_items > 0:
        print()
