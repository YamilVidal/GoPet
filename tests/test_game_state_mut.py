"""Tests for in-place game state updates (apply_move_mut, step, GoEnv)."""

from __future__ import annotations

from gopet.env import GoEnv
from gopet.game_state import GameState, _PriorState
from gopet.types import Color, Move
from gopet.web.agents import PolicyAgent


def test_apply_move_mut_double_pass_ends_game() -> None:
    state = GameState.new_game(9)
    state.apply_move_mut(Move.pass_turn())
    assert not state.is_over()
    state.apply_move_mut(Move.pass_turn())
    assert state.is_over()


def test_step_double_pass_ends_game() -> None:
    state = GameState.new_game(9)
    pass_action = state.board.height * state.board.width
    state, _, done, _ = state.step(pass_action)
    assert not done
    state, _, done, _ = state.step(pass_action)
    assert done
    assert state.is_over()


def test_go_env_ko_blocks_recapture() -> None:
    """Ko must be enforced when stepping through GoEnv."""
    state = GameState.new_game(9)
    moves = [
        Move.play(4, 4),
        Move.play(3, 4),
        Move.play(4, 5),
        Move.play(3, 5),
        Move.play(3, 3),
        Move.play(2, 4),
        Move.play(2, 5),
    ]
    for move in moves:
        state = state.apply_move(move)

    env = GoEnv(board_size=9)
    env.state = state
    ko_action = 3 * 9 + 4
    assert env.legal_mask()[ko_action] == 0.0


def test_apply_move_mut_ko_blocks_recapture() -> None:
    state = GameState.new_game(9)
    moves = [
        Move.play(4, 4),
        Move.play(3, 4),
        Move.play(4, 5),
        Move.play(3, 5),
        Move.play(3, 3),
        Move.play(2, 4),
        Move.play(2, 5),
    ]
    for move in moves:
        state.apply_move_mut(move)

    ko_recapture = Move.play(3, 4)
    assert state.is_valid_move(ko_recapture) is False


def test_policy_agent_move_count_no_infinite_loop() -> None:
    state = GameState.new_game(9)
    for move in (Move.play(2, 2), Move.play(2, 3), Move.play(3, 2)):
        state.apply_move_mut(move)

    assert PolicyAgent._move_count(state) == 3


def test_prior_state_chain_for_is_over() -> None:
    state = GameState.new_game(9)
    state.apply_move_mut(Move.pass_turn())
    assert isinstance(state.previous_state, _PriorState)
    assert state.previous_state.last_move is None
