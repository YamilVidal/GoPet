"""NumPy and PyTorch feature encodings for policy networks."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch

from gopet.game_state import GameState
from gopet.types import BLACK, EMPTY, Color, Move, WHITE


NUM_BASIC_PLANES = 5
NUM_SIMPLE_PLANES = 11


def encode_planes_numpy(state: GameState, num_planes: int = NUM_BASIC_PLANES) -> np.ndarray:
    """Build feature planes as float32 array with shape [C, H, W]."""
    board = state.board
    stones = board.stones
    height, width = board.height, board.width
    planes = np.zeros((num_planes, height, width), dtype=np.float32)

    if num_planes == 1:
        player = int(state.next_player)
        for row in range(height):
            for col in range(width):
                stone = stones[row, col]
                if stone == EMPTY:
                    continue
                if stone == player:
                    planes[0, row, col] = 1.0
                else:
                    planes[0, row, col] = -1.0
        return planes

    player = int(state.next_player)
    opponent = WHITE if player == BLACK else BLACK

    planes[0] = stones == player
    planes[1] = stones == opponent
    planes[2] = stones == EMPTY
    if player == BLACK:
        planes[3][:] = 1.0
    else:
        planes[4][:] = 1.0

    if num_planes >= NUM_SIMPLE_PLANES:
        for row in range(height):
            for col in range(width):
                liberties = board.get_go_string_liberties(row, col)
                if liberties is None:
                    move = Move.play(row, col)
                    if state.does_move_violate_ko(state.next_player, move):
                        planes[10, row, col] = 1.0
                    continue
                bucket = min(4, liberties) - 1
                stone = stones[row, col]
                if stone == WHITE:
                    bucket += 4
                planes[5 + bucket, row, col] = 1.0

    return planes


def encode_state(state: GameState, num_planes: int = NUM_BASIC_PLANES, device: str = "cpu") -> torch.Tensor:
    planes = encode_planes_numpy(state, num_planes=num_planes)
    return torch.from_numpy(planes).to(device=device, dtype=torch.float32)


def encode_batch(
    states: Sequence[GameState],
    num_planes: int = NUM_BASIC_PLANES,
    device: str = "cpu",
) -> torch.Tensor:
    batch = np.stack([encode_planes_numpy(state, num_planes=num_planes) for state in states])
    return torch.from_numpy(batch).to(device=device, dtype=torch.float32)


def legal_mask_numpy(state: GameState) -> np.ndarray:
    return state.legal_mask()


def legal_mask_tensor(state: GameState, device: str = "cpu") -> torch.Tensor:
    mask = legal_mask_numpy(state)
    return torch.from_numpy(mask).to(device=device)


def mask_policy_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Set illegal move logits to -inf. logits shape [B, A] or [A]."""
    masked = logits.clone()
    masked[..., ~legal_mask] = float("-inf")
    return masked


def action_to_move(state: GameState, action: int) -> Move:
    if action == state.board.size:
        return Move.pass_turn()
    row, col = state.board.coord(action)
    return Move.play(row, col)


def move_to_action(state: GameState, move: Move) -> int:
    if move.is_pass:
        return state.board.size
    if move.is_resign:
        raise ValueError("Resign moves have no board action index")
    assert move.point is not None
    return state.board.index(move.point.row, move.point.col)
