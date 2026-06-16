"""Throughput benchmarks comparing gopet and dlgo goboard_fast."""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "DLGO - code"))

from dlgo.goboard_fast import GameState as DlgoGameState
from dlgo.goboard_fast import Move as DlgoMove
from dlgo.gotypes import Point as DlgoPoint

from gopet.encoding import encode_state
from gopet.game_state import GameState
from gopet.types import Move


def play_random_game_gopet(size: int, moves: int, seed: int) -> None:
    rng = random.Random(seed)
    state = GameState.new_game(size)
    for _ in range(moves):
        legal = [m for m in state.legal_moves() if m.is_play]
        move = rng.choice(legal) if legal else Move.pass_turn()
        state = state.apply_move(move)
        if state.is_over():
            state = GameState.new_game(size)


def play_random_game_dlgo(size: int, moves: int, seed: int) -> None:
    rng = random.Random(seed)
    state = DlgoGameState.new_game(size)
    for _ in range(moves):
        legal = [m for m in state.legal_moves() if m.is_play]
        if legal:
            move = rng.choice(legal)
        else:
            move = DlgoMove.pass_turn()
        state = state.apply_move(move)
        if state.is_over():
            state = DlgoGameState.new_game(size)


def bench(fn, repeats: int = 20) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    elapsed = time.perf_counter() - start
    return repeats / elapsed


def play_random_game_gopet_mut(size: int, moves: int, seed: int) -> None:
    rng = random.Random(seed)
    state = GameState.new_game(size)
    for _ in range(moves):
        legal = [m for m in state.legal_moves() if m.is_play]
        move = rng.choice(legal) if legal else Move.pass_turn()
        state.apply_move_mut(move)
        if state.is_over():
            state = GameState.new_game(size)


def run_benchmarks() -> None:
    configs = [(9, 120), (19, 120)]
    repeats = 10

    print("Board play throughput (games/sec, higher is better)")
    for size, moves in configs:
        gopet_rate = bench(lambda s=size, m=moves: play_random_game_gopet(s, m, 0), repeats)
        dlgo_rate = bench(lambda s=size, m=moves: play_random_game_dlgo(s, m, 0), repeats)
        print(f"  {size}x{size}: gopet={gopet_rate:.2f}  dlgo={dlgo_rate:.2f}  speedup={gopet_rate/dlgo_rate:.2f}x")

    print("\nIn-place play with undo stack (games/sec)")
    for size, moves in configs:
        rate = bench(lambda s=size, m=moves: play_random_game_gopet_mut(s, m, 0), repeats)
        print(f"  {size}x{size}: gopet_mut={rate:.2f}")

    def gopet_play_encode(size: int = 9, moves: int = 60) -> None:
        rng = random.Random(1)
        state = GameState.new_game(size)
        for _ in range(moves):
            encode_state(state)
            legal = [m for m in state.legal_moves() if m.is_play]
            move = rng.choice(legal) if legal else Move.pass_turn()
            state = state.apply_move(move)

    print("\nPlay + encode throughput (loops/sec)")
    rate = bench(lambda: gopet_play_encode(9, 60), repeats=5)
    print(f"  9x9: {rate:.2f}")


if __name__ == "__main__":
    run_benchmarks()
