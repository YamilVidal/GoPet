"""Tests comparing gopet board behavior against dlgo goboard_fast."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DLGO_ROOT = ROOT / "DLGO - code"
sys.path.insert(0, str(DLGO_ROOT))

from dlgo.goboard_fast import Board as DlgoBoard
from dlgo.goboard_fast import GameState as DlgoGameState
from dlgo.goboard_fast import Move as DlgoMove
from dlgo.gotypes import Player, Point as DlgoPoint

from gopet.board import FastBoard
from gopet.game_state import GameState
from gopet.types import Color, Move


def dlgo_stones_grid(board: DlgoBoard) -> np.ndarray:
    grid = np.zeros((board.num_rows, board.num_cols), dtype=np.int8)
    for row in range(1, board.num_rows + 1):
        for col in range(1, board.num_cols + 1):
            stone = board.get(DlgoPoint(row, col))
            if stone is not None:
                grid[row - 1, col - 1] = int(stone.value)
    return grid


def gopet_stones_grid(board: FastBoard) -> np.ndarray:
    return board.stones.copy()


def assert_boards_equal(gopet_board: FastBoard, dlgo_board: DlgoBoard) -> None:
    np.testing.assert_array_equal(gopet_stones_grid(gopet_board), dlgo_stones_grid(dlgo_board))


def dlgo_move_from_gopet(move: Move) -> DlgoMove:
    if move.is_pass:
        return DlgoMove.pass_turn()
    if move.is_resign:
        return DlgoMove.resign()
    assert move.point is not None
    return DlgoMove.play(DlgoPoint(move.point.row + 1, move.point.col + 1))


def gopet_move_from_dlgo(move: DlgoMove) -> Move:
    if move.is_pass:
        return Move.pass_turn()
    if move.is_resign:
        return Move.resign()
    assert move.point is not None
    return Move.play(move.point.row - 1, move.point.col - 1)


@pytest.mark.parametrize("size", [9, 19])
def test_random_game_parity(size: int) -> None:
    rng = random.Random(42)
    gopet = GameState.new_game(size)
    dlgo = DlgoGameState.new_game(size)

    for _ in range(500):
        assert_boards_equal(gopet.board, dlgo.board)
        assert gopet.next_player.value == dlgo.next_player.value

        gopet_legal = {
            (m.point.row, m.point.col)
            for m in gopet.legal_moves()
            if m.is_play
        }
        dlgo_legal = {
            (m.point.row - 1, m.point.col - 1)
            for m in dlgo.legal_moves()
            if m.is_play
        }
        assert gopet_legal == dlgo_legal

        playable = [m for m in gopet.legal_moves() if m.is_play]
        if playable:
            move = rng.choice(playable)
        else:
            move = Move.pass_turn()

        gopet = gopet.apply_move(move)
        dlgo = dlgo.apply_move(dlgo_move_from_gopet(move))

        if gopet.is_over():
            break


def test_capture_and_undo() -> None:
    board = FastBoard(9)
    assert board.place_stone(Color.black, 4, 4) == (True, 0)
    assert board.place_stone(Color.white, 3, 4) == (True, 0)
    assert board.place_stone(Color.white, 5, 4) == (True, 0)
    assert board.place_stone(Color.white, 4, 3) == (True, 0)

    snapshot = board.snapshot()
    assert board.get(4, 4) == Color.black.value
    assert board.place_stone(Color.white, 4, 5) == (True, 1)
    assert board.get(4, 4) == 0

    board.restore(snapshot)
    assert board.get(4, 4) == Color.black.value
    assert board.get(4, 5) == 0


def test_game_state_undo() -> None:
    state = GameState.new_game(9)
    state.apply_move_mut(Move.play(2, 2))
    state.apply_move_mut(Move.play(2, 3))
    state.undo()
    assert state.board.get(2, 3) == 0
    assert state.next_player == Color.white
    state.undo()
    assert state.board.get(2, 2) == 0
    assert state.next_player == Color.black


def test_ko_is_blocked() -> None:
    """Simple ko: recapture of single stone is forbidden."""
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
        assert state.is_valid_move(move), f"Expected legal: {move}"
        state = state.apply_move(move)

    ko_recapture = Move.play(3, 4)
    assert state.is_valid_move(ko_recapture) is False


def test_sgf_final_position() -> None:
    sgf_path = ROOT / "Games" / "D2" / "1989" / "4" / "KimIn-YiPong-keun14752.sgf"
    if not sgf_path.exists():
        pytest.skip("SGF file not available")

    sys.path.insert(0, str(DLGO_ROOT))
    from dlgo.gosgf import Sgf_game

    content = sgf_path.read_bytes()
    sgf = Sgf_game.from_string(content)
    board = FastBoard(19)
    next_player = Color.black

    for item in sgf.main_sequence_iter():
        color, move_tuple = item.get_move()
        if color is None:
            continue
        player = Color.black if color == "b" else Color.white
        assert player == next_player
        if move_tuple is not None:
            row, col = move_tuple
            assert board.place_stone(player, row, col)[0]
        next_player = player.other

    dlgo_board = DlgoBoard(19, 19)
    sgf2 = Sgf_game.from_string(content)
    dlgo_player = Player.black
    for item in sgf2.main_sequence_iter():
        color, move_tuple = item.get_move()
        if color is None:
            continue
        player = Player.black if color == "b" else Player.white
        if move_tuple is not None:
            row, col = move_tuple
            dlgo_board.place_stone(player, DlgoPoint(row + 1, col + 1))
        dlgo_player = player.other

    assert_boards_equal(board, dlgo_board)
