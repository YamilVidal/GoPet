"""Shared types for the gopet Go board package."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import NamedTuple, Optional


EMPTY = 0
BLACK = 1
WHITE = 2


class Color(enum.IntEnum):
    black = BLACK
    white = WHITE

    @property
    def other(self) -> Color:
        return Color.white if self == Color.black else Color.black


class Point(NamedTuple):
    """0-indexed board coordinate."""

    row: int
    col: int


@dataclass(frozen=True)
class Move:
    """A pass, resign, or stone placement."""

    point: Optional[Point] = None
    is_pass: bool = False
    is_resign: bool = False

    def __post_init__(self) -> None:
        assert (self.point is not None) ^ self.is_pass ^ self.is_resign

    @property
    def is_play(self) -> bool:
        return self.point is not None

    @classmethod
    def play(cls, row: int, col: int) -> Move:
        return cls(point=Point(row, col))

    @classmethod
    def pass_turn(cls) -> Move:
        return cls(is_pass=True)

    @classmethod
    def resign(cls) -> Move:
        return cls(is_resign=True)

    def __str__(self) -> str:
        if self.is_pass:
            return "pass"
        if self.is_resign:
            return "resign"
        assert self.point is not None
        return f"({self.point.row}, {self.point.col})"
