"""Size-parameterized Zobrist hashing for board positions."""

from __future__ import annotations

import numpy as np

from gopet.types import BLACK, EMPTY, WHITE


class ZobristTable:
    """Precomputed XOR codes for each (intersection, color) pair."""

    __slots__ = ("height", "width", "size", "codes")

    def __init__(self, height: int, width: int, seed: int = 0) -> None:
        self.height = height
        self.width = width
        self.size = height * width
        rng = np.random.default_rng(seed)
        raw = rng.integers(
            0,
            np.iinfo(np.uint64).max,
            size=(3, self.size),
            dtype=np.uint64,
        )
        self.codes = raw

    def code(self, color: int, idx: int) -> np.uint64:
        return self.codes[color, idx]

    def xor_stone(self, current: np.uint64, color: int, idx: int) -> np.uint64:
        return np.uint64(current ^ self.codes[color, idx])

    def xor_empty(self, current: np.uint64, idx: int) -> np.uint64:
        return np.uint64(current ^ self.codes[EMPTY, idx])
