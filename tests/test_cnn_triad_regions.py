"""Tests for cnn_triad board regions and crops."""

from __future__ import annotations

import numpy as np
import pytest

from agents.cnn_triad.regions import (
    TRIAD_BOARD_SIZE,
    Region,
    all_inference_crops,
    classify_move,
    extract_crop,
    find_crop_for_move,
    global_move_to_local_action,
    global_to_local,
    list_crops,
    local_action_to_global,
    local_to_global,
    pass_action_index,
    require_triad_board_size,
)
from gopet.types import Move


def test_requires_19_board() -> None:
    with pytest.raises(ValueError):
        require_triad_board_size(9)


def test_partition_covers_every_intersection_once() -> None:
    counts = {Region.CORNER: 0, Region.SIDE: 0, Region.CENTER: 0}
    for row in range(TRIAD_BOARD_SIZE):
        for col in range(TRIAD_BOARD_SIZE):
            counts[classify_move(row, col)] += 1
    assert counts[Region.CORNER] == 4 * 5 * 5
    assert counts[Region.CENTER] == 9 * 9
    assert counts[Region.SIDE] == TRIAD_BOARD_SIZE * TRIAD_BOARD_SIZE - counts[Region.CORNER] - counts[Region.CENTER]


@pytest.mark.parametrize(
    ("row", "col", "region"),
    [
        (0, 0, Region.CORNER),
        (0, 10, Region.SIDE),
        (10, 10, Region.CENTER),
        (18, 18, Region.CORNER),
        (4, 9, Region.SIDE),
    ],
)
def test_classify_sample_points(row: int, col: int, region: Region) -> None:
    assert classify_move(row, col) == region


def test_global_local_round_trip_corners() -> None:
    for spec in list_crops(Region.CORNER):
        for row in range(spec.row0, spec.row0 + spec.height):
            for col in range(spec.col0, spec.col0 + spec.width):
                local = global_to_local(spec, row, col)
                back_row, back_col = local_to_global(spec, *local)
                assert (back_row, back_col) == (row, col)


def test_global_local_round_trip_sides() -> None:
    for spec in list_crops(Region.SIDE):
        for row in range(spec.row0, spec.row0 + spec.height):
            for col in range(spec.col0, spec.col0 + spec.width):
                local = global_to_local(spec, row, col)
                back_row, back_col = local_to_global(spec, *local)
                assert (back_row, back_col) == (row, col)


def test_action_round_trip_center() -> None:
    spec = list_crops(Region.CENTER)[0]
    move = Move.play(10, 10)
    action = global_move_to_local_action(move, spec)
    back = local_action_to_global(spec, action)
    assert back == move


def test_find_crop_for_corner_move() -> None:
    move = Move.play(0, 0)
    spec = find_crop_for_move(move)
    assert spec is not None
    assert spec.name == "corner_nw"


def test_extract_crop_shape() -> None:
    planes = np.arange(5 * 19 * 19, dtype=np.float32).reshape(5, 19, 19)
    for spec in all_inference_crops():
        crop = extract_crop(planes, spec)
        assert crop.shape == (5, spec.out_height, spec.out_width)


def test_pass_action_index_per_head() -> None:
    corner = list_crops(Region.CORNER)[0]
    side = list_crops(Region.SIDE)[0]
    center = list_crops(Region.CENTER)[0]
    assert pass_action_index(corner.out_height, corner.out_width) == 25
    assert pass_action_index(side.out_height, side.out_width) == 45
    assert pass_action_index(center.out_height, center.out_width) == 81
