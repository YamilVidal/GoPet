"""Tests for BasicPolicyCNN."""

from pathlib import Path

import numpy as np
import pytest
import torch

from agents.basic_cnn.model import BasicPolicyCNN
from agents.basic_cnn.train import (
    EpochMetrics,
    checkpoint_paths,
    epoch_checkpoint_path,
    save_epoch_checkpoints,
)
from gopet.encoding import NUM_BASIC_PLANES


def test_forward_shape() -> None:
    model = BasicPolicyCNN(in_planes=NUM_BASIC_PLANES, board_size=19)
    batch = torch.randn(4, NUM_BASIC_PLANES, 19, 19)
    logits = model(batch)
    assert logits.shape == (4, 19 * 19 + 1)


def test_train_one_batch() -> None:
    model = BasicPolicyCNN(board_size=19)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    features = torch.randn(8, NUM_BASIC_PLANES, 19, 19)
    labels = torch.randint(0, 19 * 19 + 1, (8,))

    logits = model(features)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)


def test_policy_agent_compatible_output_size() -> None:
    model = BasicPolicyCNN(board_size=19)
    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros(1, NUM_BASIC_PLANES, 19, 19)).squeeze(0)
    assert logits.shape[0] == 362
    assert np.isfinite(logits.numpy()).all()


def test_epoch_checkpoint_files(tmp_path: Path) -> None:
    model = BasicPolicyCNN(board_size=19)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "basic_cnn.pt"
    metrics = EpochMetrics(epoch=1, train_loss=1.0, train_acc=0.1)

    save_epoch_checkpoints(
        model=model,
        optimizer=optimizer,
        metrics=metrics,
        checkpoint_path=checkpoint_path,
        training_config={"epochs": 2},
    )

    paths = checkpoint_paths(checkpoint_path)
    assert paths["final"].exists()
    assert paths["latest"].exists()
    assert epoch_checkpoint_path(checkpoint_path, 1).exists()
    assert paths["training_state"].exists()

    loaded = torch.load(paths["final"], map_location="cpu")
    assert isinstance(loaded, BasicPolicyCNN)
