"""Fast NumPy board with union-find chain tracking and snapshot undo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from gopet.types import BLACK, EMPTY, Color, WHITE
from gopet.zobrist import ZobristTable


@dataclass
class BoardSnapshot:
    stones: np.ndarray
    chain_head: np.ndarray
    chain_liberties: np.ndarray
    chain_color: np.ndarray
    hash: np.uint64


class FastBoard:
    """Array-backed Go board with incremental Zobrist hashing."""

    __slots__ = (
        "height",
        "width",
        "size",
        "stones",
        "chain_head",
        "chain_liberties",
        "chain_color",
        "neighbors",
        "zobrist",
        "hash",
    )

    def __init__(self, height: int, width: Optional[int] = None, seed: int = 0) -> None:
        if width is None:
            width = height
        self.height = height
        self.width = width
        self.size = height * width

        self.stones = np.zeros((height, width), dtype=np.int8)
        self.chain_head = np.full(self.size, -1, dtype=np.int32)
        self.chain_liberties = np.zeros(self.size, dtype=np.int16)
        self.chain_color = np.zeros(self.size, dtype=np.int8)

        self.neighbors = self._build_neighbors()
        self.zobrist = ZobristTable(height, width, seed=seed)
        self.hash = np.uint64(0)

    @property
    def num_rows(self) -> int:
        return self.height

    @property
    def num_cols(self) -> int:
        return self.width

    def index(self, row: int, col: int) -> int:
        return row * self.width + col

    def coord(self, idx: int) -> Tuple[int, int]:
        return divmod(idx, self.width)

    def is_on_grid(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def get(self, row: int, col: int) -> int:
        return int(self.stones[row, col])

    def get_color(self, row: int, col: int) -> Optional[Color]:
        stone = self.stones[row, col]
        if stone == EMPTY:
            return None
        return Color(stone)

    def zobrist_hash(self) -> np.uint64:
        return self.hash

    def clone(self) -> FastBoard:
        copied = object.__new__(FastBoard)
        copied.height = self.height
        copied.width = self.width
        copied.size = self.size
        copied.stones = self.stones.copy()
        copied.chain_head = self.chain_head.copy()
        copied.chain_liberties = self.chain_liberties.copy()
        copied.chain_color = self.chain_color.copy()
        copied.neighbors = self.neighbors
        copied.zobrist = self.zobrist
        copied.hash = self.hash
        return copied

    def snapshot(self) -> BoardSnapshot:
        return BoardSnapshot(
            stones=self.stones.copy(),
            chain_head=self.chain_head.copy(),
            chain_liberties=self.chain_liberties.copy(),
            chain_color=self.chain_color.copy(),
            hash=self.hash,
        )

    def restore(self, snapshot: BoardSnapshot) -> None:
        self.stones[:] = snapshot.stones
        self.chain_head[:] = snapshot.chain_head
        self.chain_liberties[:] = snapshot.chain_liberties
        self.chain_color[:] = snapshot.chain_color
        self.hash = snapshot.hash

    def place_stone(self, color: Color | int, row: int, col: int) -> bool:
        """Place a stone. Returns False if occupied or suicide."""
        if not self.is_on_grid(row, col):
            raise ValueError(f"Point ({row}, {col}) is off the board")
        if self.stones[row, col] != EMPTY:
            return False

        color_val = int(color)
        idx = self.index(row, col)
        snapshot = self.snapshot()

        captured = self._play_stone(idx, color_val)
        root = self._find(idx)
        if self.chain_liberties[root] == 0 and not captured:
            self.restore(snapshot)
            return False
        return True

    def is_self_capture(self, color: Color | int, row: int, col: int) -> bool:
        if not self.is_on_grid(row, col) or self.stones[row, col] != EMPTY:
            return False

        color_val = int(color)
        idx = self.index(row, col)
        friendly_roots: List[int] = []

        for neighbor in self.neighbors[idx]:
            neighbor_color = self.stones.flat[neighbor]
            if neighbor_color == EMPTY:
                return False
            if neighbor_color == color_val:
                friendly_roots.append(self._find(neighbor))
            elif self.chain_liberties[self._find(neighbor)] == 1:
                return False

        if not friendly_roots:
            return True

        return all(self.chain_liberties[root] == 1 for root in set(friendly_roots))

    def will_capture(self, color: Color | int, row: int, col: int) -> bool:
        if not self.is_on_grid(row, col) or self.stones[row, col] != EMPTY:
            return False

        color_val = int(color)
        opp = WHITE if color_val == BLACK else BLACK
        idx = self.index(row, col)
        seen: set[int] = set()

        for neighbor in self.neighbors[idx]:
            if self.stones.flat[neighbor] != opp:
                continue
            root = self._find(neighbor)
            if root in seen:
                continue
            seen.add(root)
            if self.chain_liberties[root] == 1:
                return True
        return False

    def get_go_string_liberties(self, row: int, col: int) -> Optional[int]:
        if self.stones[row, col] == EMPTY:
            return None
        root = self._find(self.index(row, col))
        return int(self.chain_liberties[root])

    def _build_neighbors(self) -> List[List[int]]:
        neighbors: List[List[int]] = [[] for _ in range(self.size)]
        for row in range(self.height):
            for col in range(self.width):
                idx = self.index(row, col)
                if row > 0:
                    neighbors[idx].append(self.index(row - 1, col))
                if row + 1 < self.height:
                    neighbors[idx].append(self.index(row + 1, col))
                if col > 0:
                    neighbors[idx].append(self.index(row, col - 1))
                if col + 1 < self.width:
                    neighbors[idx].append(self.index(row, col + 1))
        return neighbors

    def _find(self, idx: int) -> int:
        root = idx
        while self.chain_head[root] != root:
            root = int(self.chain_head[root])
        while idx != root:
            parent = int(self.chain_head[idx])
            self.chain_head[idx] = root
            idx = parent
        return root

    def _merge(self, root_a: int, root_b: int) -> int:
        if root_a == root_b:
            return root_a
        if self.chain_liberties[root_a] < self.chain_liberties[root_b]:
            root_a, root_b = root_b, root_a
        self.chain_head[root_b] = root_a
        self.chain_liberties[root_a] = self._count_liberties_for_chain(root_a)
        return root_a

    def _count_liberties_for_chain(self, root: int) -> int:
        liberties: set[int] = set()
        for idx in range(self.size):
            if self.stones.flat[idx] == EMPTY:
                continue
            if self._find(idx) != root:
                continue
            for neighbor in self.neighbors[idx]:
                if self.stones.flat[neighbor] == EMPTY:
                    liberties.add(neighbor)
        return len(liberties)

    def _collect_chain_indices(self, root: int) -> List[int]:
        return [
            idx
            for idx in range(self.size)
            if self.stones.flat[idx] != EMPTY and self._find(idx) == root
        ]

    def _remove_chain(self, root: int) -> List[int]:
        captured: List[int] = []
        chain_indices = self._collect_chain_indices(root)
        for idx in chain_indices:
            color = int(self.stones.flat[idx])
            self.stones.flat[idx] = EMPTY
            self.chain_head[idx] = -1
            self.chain_liberties[idx] = 0
            self.hash = self.zobrist.xor_stone(self.hash, color, idx)
            captured.append(idx)

        affected_roots: set[int] = set()
        for idx in captured:
            for neighbor in self.neighbors[idx]:
                if self.stones.flat[neighbor] != EMPTY:
                    affected_roots.add(self._find(neighbor))

        for affected in affected_roots:
            self.chain_liberties[affected] = self._count_liberties_for_chain(affected)

        return captured

    def _play_stone(self, idx: int, color: int) -> List[int]:
        opp = WHITE if color == BLACK else BLACK
        captured: List[int] = []
        opponent_roots: set[int] = set()

        for neighbor in self.neighbors[idx]:
            if self.stones.flat[neighbor] == opp:
                opponent_roots.add(self._find(neighbor))

        for root in opponent_roots:
            self.chain_liberties[root] -= 1
            if self.chain_liberties[root] == 0:
                captured.extend(self._remove_chain(root))

        self.stones.flat[idx] = color
        self.hash = self.zobrist.xor_stone(self.hash, color, idx)
        self.chain_head[idx] = idx
        self.chain_color[idx] = color
        self.chain_liberties[idx] = self._count_liberties_for_chain(idx)

        root = idx
        for neighbor in self.neighbors[idx]:
            if self.stones.flat[neighbor] == color and neighbor != idx:
                other_root = self._find(neighbor)
                if other_root != root:
                    root = self._merge(root, other_root)

        return captured
