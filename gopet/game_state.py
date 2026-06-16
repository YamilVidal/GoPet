"""Game state with ko tracking, undo stack, and legal move generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Tuple

import numpy as np

from gopet.board import BoardSnapshot, FastBoard
from gopet.types import Color, Move


def board_signature(board: FastBoard) -> bytes:
    return board.stones.tobytes()


@dataclass
class HistoryEntry:
    board_snapshot: BoardSnapshot
    next_player: Color
    previous_signatures: FrozenSet[Tuple[Color, bytes]]


class GameState:
    def __init__(
        self,
        board: FastBoard,
        next_player: Color,
        previous: Optional[GameState] = None,
        move: Optional[Move] = None,
    ) -> None:
        self.board = board
        self.next_player = next_player
        self.previous_state = previous
        self.last_move = move

        if previous is None:
            self.previous_signatures: FrozenSet[Tuple[Color, bytes]] = frozenset()
        else:
            self.previous_signatures = frozenset(
                previous.previous_signatures
                | {(previous.next_player, board_signature(previous.board))}
            )

        self._undo_stack: List[HistoryEntry] = field(default_factory=list) if False else []
        self._undo_stack = []

    @classmethod
    def new_game(cls, board_size: int | Tuple[int, int]) -> GameState:
        if isinstance(board_size, int):
            height = width = board_size
        else:
            height, width = board_size
        board = FastBoard(height, width)
        return cls(board, Color.black, None, None)

    def clone(self) -> GameState:
        cloned = GameState(self.board.clone(), self.next_player, None, self.last_move)
        cloned.previous_signatures = self.previous_signatures
        cloned.previous_state = self.previous_state
        return cloned

    def apply_move(self, move: Move) -> GameState:
        next_board = self.board.clone()
        if move.is_play:
            assert move.point is not None
            row, col = move.point.row, move.point.col
            if not next_board.place_stone(self.next_player, row, col):
                raise ValueError(f"Illegal move: {move}")
        return GameState(next_board, self.next_player.other, self, move)

    def apply_move_mut(self, move: Move) -> None:
        """Apply a move in place, recording undo state."""
        entry = HistoryEntry(
            board_snapshot=self.board.snapshot(),
            next_player=self.next_player,
            previous_signatures=self.previous_signatures,
        )
        self._undo_stack.append(entry)

        if move.is_play:
            assert move.point is not None
            row, col = move.point.row, move.point.col
            if not self.board.place_stone(self.next_player, row, col):
                self._undo_stack.pop()
                raise ValueError(f"Illegal move: {move}")

        self.previous_signatures = frozenset(
            self.previous_signatures | {(self.next_player, board_signature(self.board))}
        )
        self.previous_state = self
        self.next_player = self.next_player.other
        self.last_move = move

    def undo(self) -> None:
        if not self._undo_stack:
            raise RuntimeError("No moves to undo")
        entry = self._undo_stack.pop()
        self.board.restore(entry.board_snapshot)
        self.next_player = entry.next_player
        self.previous_signatures = entry.previous_signatures
        self.last_move = None
        self.previous_state = None

    def is_move_self_capture(self, player: Color, move: Move) -> bool:
        if not move.is_play:
            return False
        assert move.point is not None
        return self.board.is_self_capture(player, move.point.row, move.point.col)

    def does_move_violate_ko(self, player: Color, move: Move) -> bool:
        if not move.is_play:
            return False
        assert move.point is not None
        row, col = move.point.row, move.point.col
        if not self.board.will_capture(player, row, col):
            return False

        snapshot = self.board.snapshot()
        if not self.board.place_stone(player, row, col):
            self.board.restore(snapshot)
            return False
        next_situation = (player.other, board_signature(self.board))
        self.board.restore(snapshot)
        return next_situation in self.previous_signatures

    def is_valid_move(self, move: Move) -> bool:
        if self.is_over():
            return False
        if move.is_pass or move.is_resign:
            return True
        assert move.point is not None
        row, col = move.point.row, move.point.col
        if self.board.get(row, col) != 0:
            return False
        if self.is_move_self_capture(self.next_player, move):
            return False
        if self.does_move_violate_ko(self.next_player, move):
            return False
        return True

    def is_over(self) -> bool:
        if self.last_move is None:
            return False
        if self.last_move.is_resign:
            return True
        if self.previous_state is None:
            return False
        second_last = self.previous_state.last_move
        if second_last is None:
            return False
        return self.last_move.is_pass and second_last.is_pass

    def legal_mask(self) -> np.ndarray:
        size = self.board.size
        mask = np.zeros(size + 1, dtype=np.bool_)
        if self.is_over():
            return mask

        stones = self.board.stones.ravel()
        for idx in range(size):
            if stones[idx] != 0:
                continue
            row, col = self.board.coord(idx)
            move = Move.play(row, col)
            if self.is_valid_move(move):
                mask[idx] = True
        mask[size] = True
        return mask

    def legal_moves(self) -> List[Move]:
        if self.is_over():
            return []
        moves: List[Move] = []
        for row in range(self.board.height):
            for col in range(self.board.width):
                move = Move.play(row, col)
                if self.is_valid_move(move):
                    moves.append(move)
        moves.append(Move.pass_turn())
        moves.append(Move.resign())
        return moves

    def step(self, action: int) -> Tuple[GameState, float, bool, dict]:
        """Apply action index (0..size-1 play, size=pass). Returns immutable successor."""
        size = self.board.size
        if action == size:
            move = Move.pass_turn()
        else:
            row, col = self.board.coord(action)
            move = Move.play(row, col)
        if not self.is_valid_move(move):
            raise ValueError(f"Illegal action: {action}")
        next_state = self.clone()
        next_state.apply_move_mut(move)
        done = next_state.is_over()
        reward = 0.0
        if done:
            winner = next_state.winner()
            if winner is not None and winner == self.next_player:
                reward = 1.0
            elif winner is not None:
                reward = -1.0
        return next_state, reward, done, {}

    def winner(self):
        from gopet.scoring import compute_game_result

        if not self.is_over():
            return None
        if self.last_move is not None and self.last_move.is_resign:
            return self.next_player
        return compute_game_result(self).winner
