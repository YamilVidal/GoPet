"""Encoder output sanity checks."""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "DLGO - code"))

from dlgo.encoders.oneplane import OnePlaneEncoder
from dlgo.goboard_fast import GameState as DlgoGameState
from dlgo.goboard_fast import Move as DlgoMove
from dlgo.gotypes import Point as DlgoPoint

from gopet.encoding import NUM_BASIC_PLANES, encode_planes_numpy, encode_state, legal_mask_tensor
from gopet.game_state import GameState
from gopet.types import Color, Move


def test_basic_planes_shape() -> None:
    state = GameState.new_game(9)
    planes = encode_planes_numpy(state, num_planes=NUM_BASIC_PLANES)
    assert planes.shape == (NUM_BASIC_PLANES, 9, 9)
    assert planes[0].sum() == 0
    assert planes[3].sum() == 81


def test_torch_encode() -> None:
    state = GameState.new_game(9)
    state = state.apply_move(Move.play(4, 4))
    tensor = encode_state(state)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (NUM_BASIC_PLANES, 9, 9)
    assert tensor.dtype == torch.float32


def test_oneplane_parity_after_moves() -> None:
    dlgo = DlgoGameState.new_game(9)
    gopet = GameState.new_game(9)
    encoder = OnePlaneEncoder((9, 9))

    moves = [(4, 4), (3, 4), (5, 4), (4, 3)]
    for row, col in moves:
        dlgo = dlgo.apply_move(DlgoMove.play(DlgoPoint(row + 1, col + 1)))
        gopet = gopet.apply_move(Move.play(row, col))

        dlgo_planes = encoder.encode(dlgo)[0]
        gopet_planes = encode_planes_numpy(gopet, num_planes=1)[0]
        np.testing.assert_allclose(gopet_planes, dlgo_planes, atol=1e-6)


def test_legal_mask_includes_pass() -> None:
    state = GameState.new_game(9)
    mask = legal_mask_tensor(state)
    assert mask.shape == (82,)
    assert mask[-1].item() is True
