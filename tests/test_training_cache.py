"""Tests for shared training data cache and usage tracking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gopet.data.training_cache import (
    SplitRecord,
    cache_is_valid,
    make_dataset_id,
    manifest_path,
    record_agent_usage,
    record_split,
)


def test_make_dataset_id_from_sgf_path() -> None:
    dataset_id = make_dataset_id(
        train_sgf_directory=Path("Games/jgdb/sgf/train"),
        max_games=4000,
        seed=0,
        board_size=19,
        num_planes=5,
    )
    assert dataset_id == "sgf_train_g4000_s0_bs19_p5"


def test_manifest_written_and_validates_cache(tmp_path: Path) -> None:
    dataset_id = "fixture_g1_s0_bs19_p5"
    record_split(
        tmp_path,
        dataset_id=dataset_id,
        board_size=19,
        num_planes=5,
        split=SplitRecord(
            prefix="train",
            sgf_directory=str((tmp_path / "sgf").resolve()),
            max_games=1,
            seed=0,
            position_count=42,
            sgf_file_count=1,
            features_file="train_features.npy",
            labels_file="train_labels.npy",
        ),
    )
    (tmp_path / "train_features.npy").write_bytes(b"")
    (tmp_path / "train_labels.npy").write_bytes(b"")

    assert manifest_path(tmp_path).exists()
    assert cache_is_valid(
        tmp_path,
        prefix="train",
        sgf_directory=tmp_path / "sgf",
        max_games=1,
        seed=0,
        board_size=19,
        num_planes=5,
    )
    assert not cache_is_valid(
        tmp_path,
        prefix="train",
        sgf_directory=tmp_path / "sgf",
        max_games=2,
        seed=0,
        board_size=19,
        num_planes=5,
    )


def test_cache_without_manifest_is_invalid(tmp_path: Path) -> None:
    (tmp_path / "train_features.npy").write_bytes(b"")
    (tmp_path / "train_labels.npy").write_bytes(b"")
    assert not cache_is_valid(
        tmp_path,
        prefix="train",
        sgf_directory=tmp_path / "sgf",
        max_games=1,
        seed=0,
        board_size=19,
        num_planes=5,
    )


def test_record_agent_usage_updates_manifest_and_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from gopet.data import training_cache

    monkeypatch.setattr(training_cache, "USAGE_ROOT", tmp_path / "usage")

    record_split(
        tmp_path,
        dataset_id="fixture",
        board_size=19,
        num_planes=5,
        split=SplitRecord(
            prefix="train",
            sgf_directory=str(tmp_path.resolve()),
            max_games=1,
            seed=0,
            position_count=10,
            sgf_file_count=1,
            features_file="train_features.npy",
            labels_file="train_labels.npy",
        ),
    )

    record_agent_usage(
        agent_id="basic_cnn",
        dataset_id="fixture",
        dataset_dir=tmp_path,
        usage_info={"epochs": 3, "checkpoint": "/tmp/model.pt"},
    )

    manifest = json.loads(manifest_path(tmp_path).read_text(encoding="utf-8"))
    assert len(manifest["used_by"]) == 1
    assert manifest["used_by"][0]["agent_id"] == "basic_cnn"
    assert manifest["used_by"][0]["epochs"] == 3

    usage_log = tmp_path / "usage" / "basic_cnn.jsonl"
    assert usage_log.exists()
    assert "basic_cnn" in usage_log.read_text(encoding="utf-8")
