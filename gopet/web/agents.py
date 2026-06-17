"""Play agents exposed to the browser UI."""

from __future__ import annotations

import random
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from gopet.encoding import legal_mask_numpy, mask_policy_logits
from gopet.game_state import GameState, _PriorState
from gopet.types import BLACK, Color, Move
from gopet.web.coords import move_from_vertex, point_to_vertex, vertex_from_move


@dataclass(frozen=True)
class MoveSelection:
    move: Move
    diagnostics: dict


class Agent(ABC):
    @abstractmethod
    def select_move(self, state: GameState) -> Move:
        raise NotImplementedError

    def select_move_with_diagnostics(self, state: GameState) -> MoveSelection:
        move = self.select_move(state)
        return MoveSelection(move=move, diagnostics=self.diagnostics())

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
        self.device = device
        payload = torch.load(model_path, map_location=device)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            raise ValueError(
                f"Checkpoint '{model_path}' is a training-state file, not a playable model. "
                "Use the main .pt or *_latest.pt checkpoint."
            )
        self.model = payload
        self.model.eval()
        self._lock = threading.Lock()
        self._last_diagnostics: dict = {}
        self._against_human: bool = False

    def set_game_context(self, *, against_human: bool) -> None:
        """Set context for the next move selection.

        Resignation by score estimation is only enabled when playing against humans.
        """
        with self._lock:
            self._against_human = against_human is True

    @staticmethod
    def _move_count(state: GameState) -> int:
        """Count ply moves played so far by walking history."""
        count = 0
        cursor: Optional[object] = state
        while cursor is not None:
            if isinstance(cursor, GameState):
                if cursor.last_move is None:
                    break
                count += 1
                cursor = cursor.previous_state
            elif isinstance(cursor, _PriorState):
                if cursor.last_move is None:
                    break
                count += 1
                cursor = cursor.previous
            else:
                break
        return count

    def _update_score_estimate(self, state: GameState, against_human: bool) -> Optional[dict]:
        if not against_human:
            return {"disabled": True, "reason": "not_against_human"}

        move_count = self._move_count(state)
        try:
            from score_estimation.territory_seki import estimate_score_during_play

            est = estimate_score_during_play(state, komi=7.5)
            diff = float(est.score_diff_black_minus_white)
            bot_is_black = int(state.next_player) == BLACK
            bot_margin = diff if bot_is_black else -diff
            return {
                "move_count": move_count,
                "estimate": str(est),
                "diff_black_minus_white": diff,
                "bot_margin": bot_margin,
            }
        except Exception as exc:
            return {"move_count": move_count, "error": str(exc)}

    def _should_resign(self, state: GameState, score_estimate: Optional[dict]) -> bool:
        if score_estimate is None or score_estimate.get("disabled"):
            return False

        move_count = int(score_estimate.get("move_count", 0))
        if move_count < self.RESIGN_AFTER_MOVE:
            return False

        diff = score_estimate.get("diff_black_minus_white")
        if diff is None:
            return False

        return (int(state.next_player) == BLACK and diff <= -self.RESIGN_POINT_MARGIN) or (
            int(state.next_player) != BLACK and diff >= self.RESIGN_POINT_MARGIN
        )

    def _build_diagnostics(
        self,
        *,
        probs: Optional[np.ndarray],
        board_size: Optional[int],
        score_estimate: Optional[dict],
        resign: bool,
    ) -> dict:
        out: dict = {}
        if score_estimate is not None:
            score_info = dict(score_estimate)
            if resign:
                score_info["resign"] = True
            out["score_estimate"] = score_info

        if probs is None or board_size is None:
            return out

        pass_action = board_size * board_size
        pass_prob = float(probs[pass_action])

        top_actions = np.argsort(-probs)[:3]
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
                    "prob": float(probs[action]),
                    "action": action,
                }
            )

        out.update(
            {
                "top_moves": top_moves,
                "pass_prob": pass_prob,
                "top_action": top_moves[0]["action"],
                "top_prob": top_moves[0]["prob"],
            }
        )
        return out

    def select_move_with_diagnostics(self, state: GameState) -> MoveSelection:
        with self._lock:
            return self._select_move_locked(state)

    def select_move(self, state: GameState) -> Move:
        return self.select_move_with_diagnostics(state).move

    def _select_move_locked(self, state: GameState) -> MoveSelection:
        from gopet.encoding import action_to_move, encode_state

        against_human = self._against_human
        score_estimate = self._update_score_estimate(state, against_human)
        if self._should_resign(state, score_estimate):
            diagnostics = self._build_diagnostics(
                probs=None,
                board_size=None,
                score_estimate=score_estimate,
                resign=True,
            )
            self._last_diagnostics = diagnostics
            return MoveSelection(move=Move.resign(), diagnostics=diagnostics)

        with torch.no_grad():
            features = encode_state(state, device=self.device).unsqueeze(0)
            logits = self.model(features).squeeze(0)
            mask = torch.from_numpy(legal_mask_numpy(state)).to(self.device)
            masked = mask_policy_logits(logits, mask)
            action = int(torch.argmax(masked).item())
            probs = torch.softmax(masked, dim=-1).cpu().numpy()
            board_size = state.board.height

        move = action_to_move(state, action)
        diagnostics = self._build_diagnostics(
            probs=probs,
            board_size=board_size,
            score_estimate=score_estimate,
            resign=False,
        )
        self._last_diagnostics = diagnostics
        return MoveSelection(move=move, diagnostics=diagnostics)

    def diagnostics(self) -> dict:
        with self._lock:
            return dict(self._last_diagnostics)


def replay_state(board_size: int, moves: List[str]) -> GameState:
    state = GameState.new_game(board_size)
    for vertex in moves:
        move = move_from_vertex(vertex, board_size)
        if not state.is_valid_move(move):
            raise ValueError(f"Illegal replay move: {vertex}")
        state = state.apply_move(move)
    return state


def build_default_agents(model_path: Optional[str] = None) -> Dict[str, Agent]:
    """Build agent map exposed by the web server.

    - Always includes: ``random``
    - Includes any trained policy agents found in ``agents/registry.py`` whose
      checkpoints exist on disk (prefers ``*_latest.pt`` if present).
    - If ``model_path`` is provided, also exposes it as ``policy``.
    """

    agents: Dict[str, Agent] = {"random": RandomAgent()}

    try:
        from agents.registry import list_policy_agents, resolve_checkpoint

        for agent_id in list_policy_agents():
            chosen = resolve_checkpoint(agent_id)
            if chosen.exists():
                agents[agent_id] = PolicyAgent(str(chosen))
    except Exception:
        # If registry import fails, keep server usable with at least random / policy.
        pass

    if model_path:
        agents["policy"] = PolicyAgent(str(Path(model_path)))
    return agents
