"""Merged inference for the three regional policy heads."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from agents.cnn_triad.model import center_net, corner_net, side_net
from agents.cnn_triad.regions import (
    TRIAD_BOARD_SIZE,
    Region,
    all_inference_crops,
    extract_crop,
    list_crops,
    local_action_to_global,
    pass_action_index,
    require_triad_board_size,
)
from gopet.encoding import encode_planes_numpy, legal_mask_numpy, mask_policy_logits
from gopet.game_state import GameState
from gopet.types import Move


class TriadPolicyModule(nn.Module):
    """Three regional 5x5 CNNs merged by global argmax at play time."""

    def __init__(self) -> None:
        super().__init__()
        self.corner_net = corner_net()
        self.side_net = side_net()
        self.center_net = center_net()
        self.board_size = TRIAD_BOARD_SIZE
        self.num_global_actions = self.board_size * self.board_size + 1

    def net_for_region(self, region: Region) -> nn.Module:
        if region == Region.CORNER:
            return self.corner_net
        if region == Region.SIDE:
            return self.side_net
        if region == Region.CENTER:
            return self.center_net
        raise ValueError(f"No network for region {region}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Use global_logits(state) for play")

    @torch.no_grad()
    def global_logits(
        self,
        state: GameState,
        *,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, List[dict]]:
        require_triad_board_size(state.board.height)
        planes = encode_planes_numpy(state)
        legal = legal_mask_numpy(state)
        logits = np.full(self.num_global_actions, float("-inf"), dtype=np.float32)
        sources: List[dict] = []

        self.eval()
        for spec in all_inference_crops():
            crop = extract_crop(planes, spec)
            tensor = torch.from_numpy(crop).unsqueeze(0).to(device=device, dtype=torch.float32)
            head_logits = self.net_for_region(spec.region)(tensor).squeeze(0).cpu().numpy()
            spatial = spec.out_height * spec.out_width
            pass_idx = pass_action_index(spec.out_height, spec.out_width)

            for action in range(spatial):
                move = local_action_to_global(spec, action)
                assert move.point is not None
                global_action = state.board.index(move.point.row, move.point.col)
                value = float(head_logits[action])
                if value > logits[global_action]:
                    logits[global_action] = value
                    sources.append(
                        {
                            "action": int(global_action),
                            "logit": value,
                            "head": spec.region.value,
                            "crop": spec.name,
                        }
                    )

            pass_logit = float(head_logits[pass_idx])
            pass_action = self.board_size * self.board_size
            if pass_logit > logits[pass_action]:
                logits[pass_action] = pass_logit
                sources.append(
                    {
                        "action": pass_action,
                        "logit": pass_logit,
                        "head": spec.region.value,
                        "crop": spec.name,
                    }
                )

        logits[~legal] = float("-inf")
        return torch.from_numpy(logits), sources

    @torch.no_grad()
    def select_move(
        self,
        state: GameState,
        *,
        device: str = "cpu",
    ) -> Tuple[Move, torch.Tensor, List[dict]]:
        logits, sources = self.global_logits(state, device=device)
        mask = torch.from_numpy(legal_mask_numpy(state))
        masked = mask_policy_logits(logits, mask)
        action = int(torch.argmax(masked).item())
        if action == self.board_size * self.board_size:
            move = Move.pass_turn()
        else:
            row, col = state.board.coord(action)
            move = Move.play(row, col)
        return move, masked, sources


def assemble_triad_module(
    corner: Optional[nn.Module] = None,
    side: Optional[nn.Module] = None,
    center: Optional[nn.Module] = None,
) -> TriadPolicyModule:
    module = TriadPolicyModule()
    if corner is not None:
        module.corner_net = corner
    if side is not None:
        module.side_net = side
    if center is not None:
        module.center_net = center
    return module
