"""Play agents exposed to the browser UI."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np

from gopet.encoding import legal_mask_numpy, mask_policy_logits
from gopet.game_state import GameState
from gopet.types import BLACK, Color, Move
from gopet.web.coords import move_from_vertex, point_to_vertex, vertex_from_move


class Agent(ABC):
    @abstractmethod
    def select_move(self, state: GameState) -> Move:
        raise NotImplementedError

    def diagnostics(self) -> dict:
        return {}


class RandomAgent(Agent):
    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def select_move(self, state: GameState) -> Move:
        playable = [m for m in state.legal_moves() if m.is_play]
        if playable:
            return self._rng.choice(playable)
        return Move.pass_turn()


class PolicyAgent(Agent):
    """PyTorch policy network agent."""

    RESIGN_AFTER_MOVE = 120
    RESIGN_POINT_MARGIN = 60.0

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        import torch

        self.device = device
        self.model = torch.load(model_path, map_location=device)
        self.model.eval()
        self._last_probs: Optional[np.ndarray] = None
        self._last_board_size: Optional[int] = None
        self._last_score_estimate: Optional[dict] = None
        self._against_human: bool = False

    def set_game_context(self, *, against_human: bool) -> None:
        """Set context for the next move selection.

        Resignation by score estimation is only enabled when playing against humans.
        """
        self._against_human = bool(against_human)

    @staticmethod
    def _move_count(state: GameState) -> int:
        """Count ply moves played so far by walking history."""
        count = 0
        cursor: Optional[GameState] = state
        while cursor is not None and cursor.last_move is not None:
            count += 1
            cursor = cursor.previous_state
        return count

    def _update_score_estimate(self, state: GameState) -> None:
        if not self._against_human:
            self._last_score_estimate = {"disabled": True, "reason": "not_against_human"}
            return

        move_count = self._move_count(state)
        try:
            from score_estimation.territory_seki import estimate_score_during_play

            est = estimate_score_during_play(state, komi=7.5)
            diff = float(est.score_diff_black_minus_white)
            bot_is_black = int(state.next_player) == BLACK
            bot_margin = diff if bot_is_black else -diff
            self._last_score_estimate = {
                "move_count": move_count,
                "estimate": str(est),
                "diff_black_minus_white": diff,
                "bot_margin": bot_margin,
            }
        except Exception as exc:
            self._last_score_estimate = {"move_count": move_count, "error": str(exc)}

    def _should_resign(self, state: GameState) -> bool:
        if not self._against_human:
            return False

        info = self._last_score_estimate or {}
        move_count = int(info.get("move_count", 0))
        if move_count < self.RESIGN_AFTER_MOVE:
            return False

        diff = info.get("diff_black_minus_white")
        if diff is None:
            return False

        resign = (int(state.next_player) == BLACK and diff <= -self.RESIGN_POINT_MARGIN) or (
            int(state.next_player) != BLACK and diff >= self.RESIGN_POINT_MARGIN
        )
        info["resign"] = resign
        self._last_score_estimate = info
        return resign

    def select_move(self, state: GameState) -> Move:
        import torch

        from gopet.encoding import encode_state

        self._update_score_estimate(state)
        if self._should_resign(state):
            return Move.resign()

        with torch.no_grad():
            features = encode_state(state, device=self.device).unsqueeze(0)
            logits = self.model(features).squeeze(0)
            mask = torch.from_numpy(legal_mask_numpy(state)).to(self.device)
            masked = mask_policy_logits(logits, mask)
            action = int(torch.argmax(masked).item())
            self._last_probs = torch.softmax(masked, dim=-1).cpu().numpy()
            self._last_board_size = state.board.height

        from gopet.encoding import action_to_move

        return action_to_move(state, action)

    def diagnostics(self) -> dict:
        if self._last_probs is None or self._last_board_size is None:
            return {} if self._last_score_estimate is None else {"score_estimate": self._last_score_estimate}

        board_size = self._last_board_size
        pass_action = board_size * board_size
        pass_prob = float(self._last_probs[pass_action])

        top_actions = np.argsort(-self._last_probs)[:3]
        top_moves = []
        for rank, action in enumerate(top_actions, start=1):
            action = int(action)
            if action == pass_action:
                move_label = "pass"
            else:
                row, col = divmod(action, board_size)
                move_label = point_to_vertex(row, col, board_size)
            top_moves.append(
                {
                    "rank": rank,
                    "move": move_label,
                    "prob": float(self._last_probs[action]),
                    "action": action,
                }
            )

        out = {
            "top_moves": top_moves,
            "pass_prob": pass_prob,
            "top_action": top_moves[0]["action"],
            "top_prob": top_moves[0]["prob"],
        }
        if self._last_score_estimate is not None:
            out["score_estimate"] = self._last_score_estimate
        return out


def replay_state(board_size: int, moves: List[str]) -> GameState:
    state = GameState.new_game(board_size)
    for vertex in moves:
        move = move_from_vertex(vertex, board_size)
        if not state.is_valid_move(move):
            raise ValueError(f"Illegal replay move: {vertex}")
        state = state.apply_move(move)
    return state


def build_default_agents(model_path: Optional[str] = None) -> Dict[str, Agent]:
    agents: Dict[str, Agent] = {"random": RandomAgent()}
    if model_path:
        agents["policy"] = PolicyAgent(model_path)
    return agents
