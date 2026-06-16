from __future__ import annotations

from gopet.game_state import GameState
from gopet.types import Color, Move
from score_estimation.territory_seki import estimate_territory_score_with_seki


def test_goscorer_wrapper_runs_on_empty_board() -> None:
    state = GameState.new_game(19)
    est = estimate_territory_score_with_seki(state, komi=7.5)
    # Empty board: both have 0 territory and 0 dead stones.
    assert est.black_points == 0.0
    assert est.white_points == 0.0
    assert est.komi == 7.5
    assert est.winner == Color.white

