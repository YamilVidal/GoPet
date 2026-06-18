"""Tests for cnn_triad merged inference."""

from __future__ import annotations

import torch

from agents.cnn_triad.triad import TriadPolicyModule
from gopet.game_state import GameState
from gopet.types import Move


def test_global_logits_length() -> None:
    module = TriadPolicyModule()
    state = GameState.new_game(19)
    logits, _ = module.global_logits(state)
    assert logits.shape == (362,)


def test_illegal_moves_masked() -> None:
    module = TriadPolicyModule()
    state = GameState.new_game(19)
    state = state.apply_move(Move.play(3, 3))
    logits, _ = module.global_logits(state)
    occupied = state.board.index(3, 3)
    assert logits[occupied].item() == float("-inf")


def test_select_move_returns_legal_move() -> None:
    module = TriadPolicyModule()
    module.eval()
    state = GameState.new_game(19)
    move, _, _ = module.select_move(state)
    assert state.is_valid_move(move)


def test_bundle_is_torch_serializable(tmp_path) -> None:
    module = TriadPolicyModule()
    path = tmp_path / "triad.pt"
    torch.save(module, path)
    loaded = torch.load(path, map_location="cpu")
    assert isinstance(loaded, TriadPolicyModule)
