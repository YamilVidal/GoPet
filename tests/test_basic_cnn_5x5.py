"""Tests for BasicPolicyCNN5x5."""

import numpy as np
import torch

from agents.basic_cnn_5x5.model import BasicPolicyCNN5x5, CONV_KERNEL
from gopet.encoding import NUM_BASIC_PLANES


def test_uses_5x5_convolutions() -> None:
    model = BasicPolicyCNN5x5(board_size=19)
    conv_layers = [layer for layer in model.trunk if isinstance(layer, torch.nn.Conv2d)]
    assert len(conv_layers) == 4
    assert all(layer.kernel_size == (CONV_KERNEL, CONV_KERNEL) for layer in conv_layers)


def test_forward_shape() -> None:
    model = BasicPolicyCNN5x5(in_planes=NUM_BASIC_PLANES, board_size=19)
    batch = torch.randn(4, NUM_BASIC_PLANES, 19, 19)
    logits = model(batch)
    assert logits.shape == (4, 19 * 19 + 1)


def test_parameter_count_larger_than_3x3() -> None:
    from agents.basic_cnn.model import BasicPolicyCNN

    small = sum(p.numel() for p in BasicPolicyCNN(board_size=19).parameters())
    large = sum(p.numel() for p in BasicPolicyCNN5x5(board_size=19).parameters())
    assert large > small


def test_policy_agent_compatible_output_size() -> None:
    model = BasicPolicyCNN5x5(board_size=19)
    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros(1, NUM_BASIC_PLANES, 19, 19)).squeeze(0)
    assert logits.shape[0] == 362
    assert np.isfinite(logits.numpy()).all()
