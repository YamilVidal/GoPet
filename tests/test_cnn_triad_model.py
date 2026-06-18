"""Tests for regional policy CNN shapes."""

import torch

from agents.cnn_triad.model import (
    CONV_KERNEL,
    RegionalPolicyCNN5x5,
    center_net,
    corner_net,
    side_net,
)
from gopet.encoding import NUM_BASIC_PLANES


def test_uses_5x5_convolutions() -> None:
    model = corner_net()
    conv_layers = [layer for layer in model.trunk if isinstance(layer, torch.nn.Conv2d)]
    assert len(conv_layers) == 4
    assert all(layer.kernel_size == (CONV_KERNEL, CONV_KERNEL) for layer in conv_layers)


def test_forward_shapes() -> None:
    corner = corner_net()
    side = side_net()
    center = center_net()

    corner_logits = corner(torch.randn(2, NUM_BASIC_PLANES, 5, 5))
    side_logits = side(torch.randn(2, NUM_BASIC_PLANES, 9, 5))
    center_logits = center(torch.randn(2, NUM_BASIC_PLANES, 9, 9))

    assert corner_logits.shape == (2, 26)
    assert side_logits.shape == (2, 46)
    assert center_logits.shape == (2, 82)


def test_rectangular_module_fields() -> None:
    model = RegionalPolicyCNN5x5(9, 5)
    assert model.height == 9
    assert model.width == 5
    assert model.num_actions == 46
