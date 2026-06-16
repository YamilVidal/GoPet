"""Small spatial policy CNN for 19x19 Go."""

from __future__ import annotations

import torch
import torch.nn as nn

from gopet.encoding import NUM_BASIC_PLANES


class BasicPolicyCNN(nn.Module):
    """Policy network with a conv trunk and 1x1 spatial move head."""

    def __init__(
        self,
        in_planes: int = NUM_BASIC_PLANES,
        board_size: int = 19,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.num_actions = board_size * board_size + 1

        self.trunk = nn.Sequential(
            nn.Conv2d(in_planes, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Conv2d(64, 1, kernel_size=1)
        self.pass_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)
        board_logits = self.policy_head(features).flatten(1)
        pass_logit = self.pass_head(features.mean(dim=(2, 3)))
        return torch.cat([board_logits, pass_logit], dim=1)
