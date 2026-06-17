"""Tests for capture tracking in game state and scoring."""

from __future__ import annotations

from gopet.game_state import GameState
from gopet.scoring import compute_game_result, compute_territory_score
from gopet.types import Color, Move

# Alternating moves that surround and capture the black stone at (4, 4).
_CAPTURE_MOVES = [
    (4, 4),
    (3, 4),
    (6, 6),
    (5, 4),
    (6, 7),
    (4, 3),
    (6, 8),
    (4, 5),
]


def _play_capture_sequence(*, mutable: bool = False) -> GameState:
    state = GameState.new_game(9)
    for row, col in _CAPTURE_MOVES:
        move = Move.play(row, col)
        if mutable:
            state.apply_move_mut(move)
        else:
            state = state.apply_move(move)
    return state


def test_game_state_tracks_capture_on_apply_move() -> None:
    state = _play_capture_sequence()
    assert state.black_captures == 0
    assert state.white_captures == 1
    assert state.board.get(4, 4) == 0


def test_undo_restores_capture_counts() -> None:
    state = _play_capture_sequence(mutable=True)
    assert state.white_captures == 1

    state.undo()
    assert state.white_captures == 0
    assert state.board.get(4, 4) == Color.black


def test_compute_game_result_reports_captures() -> None:
    state = _play_capture_sequence()
    result = compute_game_result(state, komi=7.5)
    assert result.black_captures == 0
    assert result.white_captures == 1


def test_compute_territory_score_includes_prisoners() -> None:
    state = _play_capture_sequence()
    result = compute_territory_score(state, komi=0.0)
    assert result.white_captures == 1
    assert result.w >= 1.0
