"""Tests for SGF replay data generation."""

from pathlib import Path

import pytest

from gopet.data.sgf_processor import GoDatasetBuilder

ROOT = Path(__file__).resolve().parents[1]


def test_replay_kim_in_game() -> None:
    sgf_path = ROOT / "Games" / "D2" / "1989" / "4" / "KimIn-YiPong-keun14752.sgf"
    if not sgf_path.exists():
        pytest.skip("SGF file not available")

    builder = GoDatasetBuilder(board_size=19, num_planes=5)
    features, labels = builder.replay_game(sgf_path.read_bytes())

    assert features.shape[1:] == (5, 19, 19)
    assert features.shape[0] == labels.shape[0]
    assert features.shape[0] > 100
    assert labels.min() >= 0
    assert labels.max() < 19 * 19 + 1
