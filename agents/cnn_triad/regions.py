"""Board region partition and canonical crops for cnn_triad (19x19)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from gopet.types import Move

TRIAD_BOARD_SIZE = 19
CORNER_BAND = 5
SIDE_BAND = 9
CENTER_BAND = 9

# Inclusive index bounds on a 19x19 board (0-indexed).
_CORNER_LO = 0
_CORNER_HI = CORNER_BAND - 1
_CORNER_FAR_LO = TRIAD_BOARD_SIZE - CORNER_BAND
_CENTER_LO = CORNER_BAND
_CENTER_HI = TRIAD_BOARD_SIZE - CORNER_BAND - 1


class Region(Enum):
    CORNER = "corner"
    SIDE = "side"
    CENTER = "center"
    PASS = "pass"


@dataclass(frozen=True)
class CropSpec:
    """One symmetric copy of a regional crop on the full board."""

    name: str
    region: Region
    row0: int
    col0: int
    height: int
    width: int
    rotation_k: int

    @property
    def out_height(self) -> int:
        if self.rotation_k % 2 == 0:
            return self.height
        return self.width

    @property
    def out_width(self) -> int:
        if self.rotation_k % 2 == 0:
            return self.width
        return self.height


def require_triad_board_size(board_size: int) -> None:
    if board_size != TRIAD_BOARD_SIZE:
        raise ValueError(
            f"cnn_triad only supports {TRIAD_BOARD_SIZE}x{TRIAD_BOARD_SIZE} boards, got {board_size}"
        )


def classify_move(row: int, col: int, board_size: int = TRIAD_BOARD_SIZE) -> Region:
    require_triad_board_size(board_size)
    in_corner_row = row <= _CORNER_HI or row >= _CORNER_FAR_LO
    in_corner_col = col <= _CORNER_HI or col >= _CORNER_FAR_LO
    if in_corner_row and in_corner_col:
        return Region.CORNER
    if _CENTER_LO <= row <= _CENTER_HI and _CENTER_LO <= col <= _CENTER_HI:
        return Region.CENTER
    return Region.SIDE


def classify_game_move(move: Move, board_size: int = TRIAD_BOARD_SIZE) -> Region:
    if move.is_pass or move.is_resign:
        return Region.PASS
    assert move.point is not None
    return classify_move(move.point.row, move.point.col, board_size)


def list_crops(region: Region) -> List[CropSpec]:
    if region == Region.CORNER:
        return [
            CropSpec("corner_nw", Region.CORNER, 0, 0, CORNER_BAND, CORNER_BAND, 0),
            CropSpec("corner_ne", Region.CORNER, 0, _CORNER_FAR_LO, CORNER_BAND, CORNER_BAND, 1),
            CropSpec("corner_se", Region.CORNER, _CORNER_FAR_LO, _CORNER_FAR_LO, CORNER_BAND, CORNER_BAND, 2),
            CropSpec("corner_sw", Region.CORNER, _CORNER_FAR_LO, 0, CORNER_BAND, CORNER_BAND, 3),
        ]
    if region == Region.SIDE:
        return [
            CropSpec("side_top", Region.SIDE, 0, _CENTER_LO, CORNER_BAND, SIDE_BAND, 1),
            CropSpec("side_bottom", Region.SIDE, _CORNER_FAR_LO, _CENTER_LO, CORNER_BAND, SIDE_BAND, 3),
            CropSpec("side_left", Region.SIDE, _CENTER_LO, 0, SIDE_BAND, CORNER_BAND, 0),
            CropSpec("side_right", Region.SIDE, _CENTER_LO, _CORNER_FAR_LO, SIDE_BAND, CORNER_BAND, 2),
        ]
    if region == Region.CENTER:
        return [
            CropSpec(
                "center",
                Region.CENTER,
                _CENTER_LO,
                _CENTER_LO,
                CENTER_BAND,
                CENTER_BAND,
                0,
            )
        ]
    return []


def all_inference_crops() -> List[CropSpec]:
    crops: List[CropSpec] = []
    for region in (Region.CORNER, Region.SIDE, Region.CENTER):
        crops.extend(list_crops(region))
    return crops


def contains_global(spec: CropSpec, row: int, col: int) -> bool:
    return (
        spec.row0 <= row < spec.row0 + spec.height
        and spec.col0 <= col < spec.col0 + spec.width
    )


def find_crop_for_move(move: Move, board_size: int = TRIAD_BOARD_SIZE) -> Optional[CropSpec]:
    if not move.is_play:
        return None
    assert move.point is not None
    region = classify_move(move.point.row, move.point.col, board_size)
    for spec in list_crops(region):
        if contains_global(spec, move.point.row, move.point.col):
            return spec
    return None


def _rotate_coords(
    row: int,
    col: int,
    height: int,
    width: int,
    rotation_k: int,
) -> Tuple[int, int, int, int]:
    k = rotation_k % 4
    for _ in range(k):
        row, col = col, height - 1 - row
        height, width = width, height
    return row, col, height, width


def _unrotate_coords(
    row: int,
    col: int,
    out_height: int,
    out_width: int,
    rotation_k: int,
) -> Tuple[int, int, int, int]:
    k = (-rotation_k) % 4
    height, width = out_height, out_width
    for _ in range(k):
        row, col = col, height - 1 - row
        height, width = width, height
    return row, col, height, width


def global_to_local(spec: CropSpec, row: int, col: int) -> Tuple[int, int]:
    slice_row = row - spec.row0
    slice_col = col - spec.col0
    local_row, local_col, _, _ = _rotate_coords(
        slice_row,
        slice_col,
        spec.height,
        spec.width,
        spec.rotation_k,
    )
    return local_row, local_col


def local_to_global(spec: CropSpec, local_row: int, local_col: int) -> Tuple[int, int]:
    slice_row, slice_col, _, _ = _unrotate_coords(
        local_row,
        local_col,
        spec.out_height,
        spec.out_width,
        spec.rotation_k,
    )
    return spec.row0 + slice_row, spec.col0 + slice_col


def extract_crop(planes: np.ndarray, spec: CropSpec) -> np.ndarray:
    """Slice and rotate feature planes to canonical orientation."""
    sliced = planes[
        :,
        spec.row0 : spec.row0 + spec.height,
        spec.col0 : spec.col0 + spec.width,
    ]
    if spec.rotation_k % 4 == 0:
        return sliced.copy()
    return np.rot90(sliced, spec.rotation_k % 4, axes=(-2, -1)).copy()


def local_action_index(local_row: int, local_col: int, width: int) -> int:
    return local_row * width + local_col


def pass_action_index(height: int, width: int) -> int:
    return height * width


def global_move_to_local_action(move: Move, spec: CropSpec) -> int:
    if move.is_pass:
        return pass_action_index(spec.out_height, spec.out_width)
    assert move.point is not None
    local_row, local_col = global_to_local(spec, move.point.row, move.point.col)
    return local_action_index(local_row, local_col, spec.out_width)


def local_action_to_global(spec: CropSpec, action: int) -> Move:
    spatial = spec.out_height * spec.out_width
    if action >= spatial:
        return Move.pass_turn()
    local_row, local_col = divmod(action, spec.out_width)
    row, col = local_to_global(spec, local_row, local_col)
    return Move.play(row, col)


def head_shapes() -> dict[Region, Tuple[int, int]]:
    corner = list_crops(Region.CORNER)[0]
    side = list_crops(Region.SIDE)[0]
    center = list_crops(Region.CENTER)[0]
    return {
        Region.CORNER: (corner.out_height, corner.out_width),
        Region.SIDE: (side.out_height, side.out_width),
        Region.CENTER: (center.out_height, center.out_width),
    }
