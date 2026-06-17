"""Tests for head-to-head match simulation."""

from __future__ import annotations

from gopet.eval.match import play_game, resolve_winner, run_match_series
from gopet.game_state import GameState
from gopet.types import Color, Move
from gopet.web.agents import Agent


class PassAgent(Agent):
    def select_move(self, state: GameState) -> Move:
        return Move.pass_turn()


class ResignAgent(Agent):
    def select_move(self, state: GameState) -> Move:
        return Move.resign()


def test_resign_agent_loses() -> None:
    outcome = play_game(
        PassAgent(),
        ResignAgent(),
        black_name="pass",
        white_name="resign",
        board_size=9,
    )
    assert outcome.end_reason == "resign"
    assert outcome.winner == Color.black


def test_double_pass_scores_game() -> None:
    outcome = play_game(
        PassAgent(),
        PassAgent(),
        black_name="pass_a",
        white_name="pass_b",
        board_size=9,
    )
    assert outcome.end_reason == "score"
    assert outcome.move_count == 2
    assert outcome.winner == Color.white


def test_max_moves_does_not_apply_komi() -> None:
    state = GameState.new_game(9)
    state = state.apply_move(Move.play(3, 3))
    winner = resolve_winner(state, komi=7.5, end_reason="max_moves")
    assert winner == Color.black


def test_score_end_applies_komi_on_empty_board() -> None:
    state = GameState.new_game(9)
    state = state.apply_move(Move.pass_turn()).apply_move(Move.pass_turn())
    winner = resolve_winner(state, komi=7.5, end_reason="score")
    assert winner == Color.white


def test_run_match_series_counts_all_games() -> None:
    stats = run_match_series(
        PassAgent(),
        PassAgent(),
        agent_a_name="pass_a",
        agent_b_name="pass_b",
        games=10,
        board_size=9,
        show_progress=False,
    )
    assert stats.games == 10
    assert stats.wins_a + stats.wins_b == 10
    assert stats.wins_black + stats.wins_white == 10
