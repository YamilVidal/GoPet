"""Tests for cnn_triad regional dataset building."""

from __future__ import annotations

from agents.cnn_triad.data import head_cache_prefix
from agents.cnn_triad.regions import Region, classify_move, find_crop_for_move
from gopet.types import Move


def test_head_cache_prefix_names() -> None:
    assert head_cache_prefix(Region.CORNER, "train") == "corner_train"
    assert head_cache_prefix(Region.SIDE, "val") == "side_val"


def test_corner_move_maps_to_corner_crop() -> None:
    move = Move.play(0, 0)
    assert classify_move(0, 0) == Region.CORNER
    spec = find_crop_for_move(move)
    assert spec is not None
    assert spec.region == Region.CORNER


def test_center_move_maps_to_center_crop() -> None:
    move = Move.play(10, 10)
    assert classify_move(10, 10) == Region.CENTER
    spec = find_crop_for_move(move)
    assert spec is not None
    assert spec.name == "center"


def test_side_move_maps_to_side_crop() -> None:
    move = Move.play(0, 10)
    assert classify_move(0, 10) == Region.SIDE
    spec = find_crop_for_move(move)
    assert spec is not None
    assert spec.region == Region.SIDE
