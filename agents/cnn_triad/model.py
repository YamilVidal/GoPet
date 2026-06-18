"""Regional policy CNNs with 5x5 convolutions (rectangular crops)."""

from __future__ import annotations

import torch
import torch.nn as nn

from gopet.encoding import NUM_BASIC_PLANES

CONV_KERNEL = 5
CONV_PADDING = CONV_KERNEL // 2


class RegionalPolicyCNN5x5(nn.Module):
    """Policy network with a 5x5 conv trunk and 1x1 spatial move head."""

    def __init__(
        self,
        height: int,
        width: int,
        in_planes: int = NUM_BASIC_PLANES,
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.num_actions = height * width + 1

        self.trunk = nn.Sequential(
            nn.Conv2d(in_planes, 32, kernel_size=CONV_KERNEL, padding=CONV_PADDING),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=CONV_KERNEL, padding=CONV_PADDING),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=CONV_KERNEL, padding=CONV_PADDING),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=CONV_KERNEL, padding=CONV_PADDING),
            nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Conv2d(64, 1, kernel_size=1)
        self.pass_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)
        board_logits = self.policy_head(features).flatten(1)
        pass_logit = self.pass_head(features.mean(dim=(2, 3)))
        return torch.cat([board_logits, pass_logit], dim=1)


def corner_net(in_planes: int = NUM_BASIC_PLANES) -> RegionalPolicyCNN5x5:
    return RegionalPolicyCNN5x5(5, 5, in_planes=in_planes)


def side_net(in_planes: int = NUM_BASIC_PLANES) -> RegionalPolicyCNN5x5:
    return RegionalPolicyCNN5x5(9, 5, in_planes=in_planes)


def center_net(in_planes: int = NUM_BASIC_PLANES) -> RegionalPolicyCNN5x5:
    return RegionalPolicyCNN5x5(9, 9, in_planes=in_planes)
