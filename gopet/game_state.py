"""Game state with ko tracking, undo stack, and legal move generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Tuple

import numpy as np

from gopet.board import BoardSnapshot, FastBoard
from gopet.types import Color, Move


def board_signature(board: FastBoard) -> bytes:
    return board.stones.tobytes()


@dataclass(frozen=True)
class _PriorState:
    """Lightweight move history for in-place play (undo / step)."""

    last_move: Optional[Move]
    previous: Optional["_PriorState"]


def _as_prior_chain(previous: Optional[object]) -> Optional[_PriorState]:
    if previous is None:
        return None
    if isinstance(previous, _PriorState):
        return previous
    if isinstance(previous, GameState):
        if previous.last_move is None and previous.previous_state is None:
            return None
        return _PriorState(
            previous.last_move,
            _as_prior_chain(previous.previous_state),
        )
    return None


@dataclass
class HistoryEntry:
    board_snapshot: BoardSnapshot
    next_player: Color
    previous_signatures: FrozenSet[Tuple[Color, bytes]]
    black_captures: int
    white_captures: int
    last_move: Optional[Move]
    previous_state: Optional[object]


class GameState:
    def __init__(
        self,
        board: FastBoard,
        next_player: Color,
        previous: Optional[GameState] = None,
        move: Optional[Move] = None,
        *,
        black_captures: int = 0,
        white_captures: int = 0,
    ) -> None:
        self.board = board
        self.next_player = next_player
        self.previous_state = previous
        self.last_move = move
        self.black_captures = black_captures
        self.white_captures = white_captures

        if previous is None:
            self.previous_signatures: FrozenSet[Tuple[Color, bytes]] = frozenset()
        else:
            self.previous_signatures = frozenset(
                previous.previous_signatures
                | {(previous.next_player, board_signature(previous.board))}
            )

        self._undo_stack: List[HistoryEntry] = []

    @classmethod
    def new_game(cls, board_size: int | Tuple[int, int]) -> GameState:
        if isinstance(board_size, int):
            height = width = board_size
        else:
            height, width = board_size
        board = FastBoard(height, width)
        return cls(board, Color.black, None, None)

    def clone(self) -> GameState:
        cloned = GameState(
            self.board.clone(),
            self.next_player,
            None,
            self.last_move,
            black_captures=self.black_captures,
            white_captures=self.white_captures,
        )
        cloned.previous_signatures = self.previous_signatures
        cloned.previous_state = self.previous_state
        return cloned

    def apply_move(self, move: Move) -> GameState:
        next_board = self.board.clone()
        black_captures = self.black_captures
        white_captures = self.white_captures
        if move.is_play:
            assert move.point is not None
            row, col = move.point.row, move.point.col
            placed, captured = next_board.place_stone(self.next_player, row, col)
            if not placed:
                raise ValueError(f"Illegal move: {move}")
            if self.next_player == Color.black:
                black_captures += captured
            else:
                white_captures += captured
        return GameState(
            next_board,
            self.next_player.other,
            self,
            move,
            black_captures=black_captures,
            white_captures=white_captures,
        )

    def apply_move_mut(self, move: Move) -> None:
        """Apply a move in place, recording undo state."""
        entry = HistoryEntry(
            board_snapshot=self.board.snapshot(),
            next_player=self.next_player,
            previous_signatures=self.previous_signatures,
            black_captures=self.black_captures,
            white_captures=self.white_captures,
            last_move=self.last_move,
            previous_state=self.previous_state,
        )
        self._undo_stack.append(entry)

        # Record board position before this ply (same semantics as immutable apply_move).
        self.previous_signatures = frozenset(
            self.previous_signatures
            | {(self.next_player, board_signature(self.board))}
        )

        if move.is_play:
            assert move.point is not None
            row, col = move.point.row, move.point.col
            placed, captured = self.board.place_stone(self.next_player, row, col)
            if not placed:
                self._undo_stack.pop()
                self.previous_signatures = entry.previous_signatures
                raise ValueError(f"Illegal move: {move}")
            if self.next_player == Color.black:
                self.black_captures += captured
            else:
                self.white_captures += captured

        self.previous_state = _PriorState(self.last_move, _as_prior_chain(self.previous_state))
        self.next_player = self.next_player.other
        self.last_move = move

    def _second_last_move(self) -> Optional[Move]:
        previous = self.previous_state
        if previous is None:
            return None
        if isinstance(previous, _PriorState):
            return previous.last_move
        if isinstance(previous, GameState):
            return previous.last_move
        return None

    def undo(self) -> None:
        if not self._undo_stack:
            raise RuntimeError("No moves to undo")
        entry = self._undo_stack.pop()
        self.board.restore(entry.board_snapshot)
        self.next_player = entry.next_player
        self.previous_signatures = entry.previous_signatures
        self.black_captures = entry.black_captures
        self.white_captures = entry.white_captures
        self.last_move = entry.last_move
        self.previous_state = entry.previous_state

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
        placed, _ = self.board.place_stone(player, row, col)
        if not placed:
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
        second_last = self._second_last_move()
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
        next_state = self.apply_move(move)
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
