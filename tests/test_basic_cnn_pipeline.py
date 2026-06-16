"""Integration tests for basic_cnn training data pipeline."""

from pathlib import Path

import pytest

from agents.basic_cnn.data import build_training_arrays, sample_sgf_paths
from gopet.data.sgf_processor import GoDatasetBuilder

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "sgf"


def test_sample_sgf_paths_limits_count() -> None:
    if not FIXTURE_DIR.exists():
        pytest.skip("fixture SGF directory not available")

    paths = sample_sgf_paths(FIXTURE_DIR, max_games=1, seed=0)
    assert len(paths) == 1


def test_build_training_arrays_from_fixture(tmp_path: Path) -> None:
    sgf_path = FIXTURE_DIR / "sample.sgf"
    if not sgf_path.exists():
        pytest.skip("fixture SGF not available")

    features, labels = build_training_arrays(
        [sgf_path],
        data_directory=tmp_path,
        output_prefix="fixture",
        skip_errors=True,
        save=True,
        dataset_id="fixture_test",
        sgf_directory=FIXTURE_DIR,
        max_games=1,
        seed=0,
    )

    assert features.shape[1:] == (5, 19, 19)
    assert features.shape[0] == labels.shape[0]
    assert features.shape[0] > 10
    assert (tmp_path / "fixture_features.npy").exists()
    assert (tmp_path / "fixture_labels.npy").exists()
    assert (tmp_path / "manifest.json").exists()


def test_replay_fixture_game() -> None:
    sgf_path = FIXTURE_DIR / "sample.sgf"
    if not sgf_path.exists():
        pytest.skip("fixture SGF not available")

    builder = GoDatasetBuilder(board_size=19, num_planes=5)
    features, labels = builder.replay_game(sgf_path.read_bytes())
    assert features.shape[0] == labels.shape[0]
    assert labels.max() < 362
