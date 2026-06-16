"""Lightweight environment wrapper for self-play loops."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from gopet.encoding import encode_planes_numpy, legal_mask_numpy
from gopet.game_state import GameState
from gopet.types import Move


class GoEnv:
    def __init__(self, board_size: int = 9, num_feature_planes: int = 5) -> None:
        self.board_size = board_size
        self.num_feature_planes = num_feature_planes
        self.state: Optional[GameState] = None
        self.reset()

    def reset(self) -> np.ndarray:
        self.state = GameState.new_game(self.board_size)
        assert self.state is not None
        return encode_planes_numpy(self.state, num_planes=self.num_feature_planes)

    def legal_mask(self) -> np.ndarray:
        assert self.state is not None
        return legal_mask_numpy(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        assert self.state is not None
        self.state, reward, done, info = self.state.step(action)
        obs = encode_planes_numpy(self.state, num_planes=self.num_feature_planes)
        return obs, reward, done, info

    def step_move(self, move: Move) -> Tuple[np.ndarray, float, bool, dict]:
        from gopet.encoding import move_to_action

        assert self.state is not None
        action = move_to_action(self.state, move)
        return self.step(action)
