from __future__ import annotations

import pytest

from gopet.game_state import GameState
from gopet.types import Color, Move
from score_estimation.territory_seki import (
    estimate_score_during_play,
    estimate_territory_score_with_seki,
)


def test_goscorer_wrapper_runs_on_empty_board() -> None:
    state = GameState.new_game(19)
    est = estimate_territory_score_with_seki(state, komi=7.5)
    # Empty board: both have 0 territory and 0 dead stones.
    assert est.black_points == 0.0
    assert est.white_points == 0.0
    assert est.komi == 7.5
    assert est.winner == Color.white


def test_area_score_counts_stones_during_play() -> None:
    state = GameState.new_game(19)
    state = state.apply_move(Move.play(3, 3))  # black
    state = state.apply_move(Move.play(3, 4))  # white
    state = state.apply_move(Move.play(4, 3))  # black

    est = estimate_score_during_play(state, komi=7.5)
    assert est.black_points == 2.0
    assert est.white_points == 1.0
    assert est.score_diff_black_minus_white == pytest.approx(-6.5)
    assert str(est) == "W+6.5"


def test_territory_score_stays_at_komi_midgame() -> None:
    """Territory scoring ignores living stones, so mid-game it shows komi only."""
    state = GameState.new_game(19)
    state = state.apply_move(Move.play(3, 3))
    state = state.apply_move(Move.play(3, 4))
    state = state.apply_move(Move.play(4, 3))

    est = estimate_territory_score_with_seki(state, komi=7.5)
    assert est.black_points == 0.0
    assert est.white_points == 0.0
    assert str(est) == "W+7.5"

