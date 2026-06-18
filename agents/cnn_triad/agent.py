"""Play agent for the cnn_triad regional policy bundle."""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import torch

from agents.cnn_triad.triad import TriadPolicyModule
from gopet.game_state import GameState
from gopet.types import Move
from gopet.web.agents import Agent, MoveSelection
from gopet.web.coords import point_to_vertex


class TriadPolicyAgent(Agent):
    """Three-head regional policy agent merged by global argmax."""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.device = device
        payload = torch.load(model_path, map_location=device)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            raise ValueError(
                f"Checkpoint '{model_path}' is a training-state file, not a playable triad bundle."
            )
        if not isinstance(payload, TriadPolicyModule):
            raise ValueError(f"Expected TriadPolicyModule checkpoint at '{model_path}'")
        self.model = payload
        self.model.eval()
        self._lock = threading.Lock()
        self._last_diagnostics: dict = {}

    def select_move_with_diagnostics(self, state: GameState) -> MoveSelection:
        with self._lock:
            move, masked, sources = self.model.select_move(state, device=self.device)
            probs = torch.softmax(masked, dim=-1).cpu().numpy()
            board_size = state.board.height
            pass_action = board_size * board_size
            top_actions = np.argsort(-probs)[:3]
            top_moves = []
            for rank, action in enumerate(top_actions, start=1):
                action = int(action)
                if action == pass_action:
                    label = "pass"
                else:
                    row, col = state.board.coord(action)
                    label = point_to_vertex(row, col, board_size)
                top_moves.append(
                    {
                        "rank": rank,
                        "move": label,
                        "prob": float(probs[action]),
                        "action": action,
                    }
                )
            winner = max(sources, key=lambda item: item["logit"], default=None)
            diagnostics = {
                "top_moves": top_moves,
                "pass_prob": float(probs[pass_action]),
                "winning_head": winner["head"] if winner else None,
                "winning_crop": winner["crop"] if winner else None,
            }
            self._last_diagnostics = diagnostics
            return MoveSelection(move=move, diagnostics=diagnostics)

    def select_move(self, state: GameState) -> Move:
        return self.select_move_with_diagnostics(state).move

    def diagnostics(self) -> dict:
        with self._lock:
            return dict(self._last_diagnostics)

    def set_game_context(self, *, against_human: bool) -> None:
        # Triad agent does not resign by score estimate.
        return
