"""Territory scoring adapted from DLGO."""

from __future__ import annotations

from collections import namedtuple
from typing import Dict, List, Optional, Set, Tuple

from gopet.board import FastBoard
from gopet.game_state import GameState
from gopet.types import Color, EMPTY


class Territory:
    def __init__(self, territory_map: Dict[Tuple[int, int], object]) -> None:
        self.num_black_territory = 0
        self.num_white_territory = 0
        self.num_black_stones = 0
        self.num_white_stones = 0
        self.num_dame = 0
        self.dame_points: List[Tuple[int, int]] = []

        for status in territory_map.values():
            if status == Color.black:
                self.num_black_stones += 1
            elif status == Color.white:
                self.num_white_stones += 1
            elif status == "territory_b":
                self.num_black_territory += 1
            elif status == "territory_w":
                self.num_white_territory += 1
            elif status == "dame":
                self.num_dame += 1


class GameResult(namedtuple("GameResult", "b w komi")):
    @property
    def winner(self) -> Color:
        if self.b > self.w + self.komi:
            return Color.black
        return Color.white

    @property
    def winning_margin(self) -> float:
        w = self.w + self.komi
        return abs(self.b - w)

    def __str__(self) -> str:
        w = self.w + self.komi
        if self.b > w:
            return f"B+{self.b - w:.1f}"
        return f"W+{w - self.b:.1f}"


def evaluate_territory(board: FastBoard) -> Territory:
    status: Dict[Tuple[int, int], object] = {}
    for row in range(board.height):
        for col in range(board.width):
            if (row, col) in status:
                continue
            stone = board.get(row, col)
            if stone != EMPTY:
                status[(row, col)] = Color(stone)
            else:
                group, neighbors = _collect_region(board, row, col, status)
                if len(neighbors) == 1:
                    neighbor_stone = neighbors.pop()
                    fill_with = (
                        "territory_b" if neighbor_stone == Color.black else "territory_w"
                    )
                else:
                    fill_with = "dame"
                for pos in group:
                    status[pos] = fill_with
    return Territory(status)


def _collect_region(
    board: FastBoard,
    row: int,
    col: int,
    visited: Dict[Tuple[int, int], bool],
) -> Tuple[List[Tuple[int, int]], Set[Optional[Color]]]:
    start = (row, col)
    if start in visited:
        return [], set()

    all_points = [start]
    all_borders: Set[Optional[Color]] = set()
    visited[start] = True
    here = board.get(row, col)

    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not board.is_on_grid(nr, nc):
                continue
            neighbor = board.get(nr, nc)
            if neighbor == here:
                if (nr, nc) not in visited:
                    visited[(nr, nc)] = True
                    all_points.append((nr, nc))
                    stack.append((nr, nc))
            else:
                border = None if neighbor == EMPTY else Color(neighbor)
                all_borders.add(border)

    return all_points, all_borders


def compute_game_result(game_state: GameState, komi: float = 7.5) -> GameResult:
    territory = evaluate_territory(game_state.board)
    return GameResult(
        territory.num_black_territory + territory.num_black_stones,
        territory.num_white_territory + territory.num_white_stones,
        komi=komi,
    )
