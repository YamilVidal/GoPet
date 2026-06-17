"""Tests for checkpoint resolution and web server validation."""

from __future__ import annotations

import json

import pytest

from agents.registry import resolve_checkpoint
from gopet.eval.match import resolve_winner
from gopet.game_state import GameState
from gopet.types import Color, Move
from gopet.web.agents import Agent, RandomAgent
from gopet.web.server import create_app


def test_resolve_checkpoint_prefers_latest(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agents import registry

    base = tmp_path / "basic_cnn.pt"
    latest = tmp_path / "basic_cnn_latest.pt"
    base.write_bytes(b"base")
    latest.write_bytes(b"latest")
    monkeypatch.setitem(registry.AGENT_CHECKPOINTS, "basic_cnn", base)

    assert resolve_checkpoint("basic_cnn") == latest
    assert resolve_checkpoint("basic_cnn", str(base)) == base


def test_max_moves_tie_counts_as_draw() -> None:
    state = GameState.new_game(9)
    winner = resolve_winner(state, komi=7.5, end_reason="max_moves")
    assert winner is None


class IllegalMoveAgent(Agent):
    def select_move(self, state: GameState) -> Move:
        return Move.play(4, 4)


def test_server_rejects_invalid_json_shape() -> None:
    app = create_app(agents={"random": RandomAgent()})
    client = app.test_client()

    response = client.post(
        "/api/select-move/random",
        data=json.dumps(["not", "an", "object"]),
        content_type="application/json",
    )
    assert response.status_code == 400


def test_server_validates_bot_move() -> None:
    app = create_app(agents={"illegal": IllegalMoveAgent()})
    client = app.test_client()

    response = client.post(
        "/api/select-move/illegal",
        data=json.dumps({"board_size": 9, "moves": ["E5"], "against_human": False}),
        content_type="application/json",
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["bot_move"] == "pass"
