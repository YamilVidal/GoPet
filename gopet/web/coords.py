"""Coordinate conversion between GTP-style vertices and gopet moves."""

from __future__ import annotations

from gopet.types import Move

COLS = "ABCDEFGHJKLMNOPQRST"


def vertex_to_point(vertex: str, board_size: int) -> tuple[int, int]:
    """Convert GTP vertex (e.g. 'D4') to 0-indexed (row, col)."""
    vertex = vertex.strip().upper()
    if not vertex:
        raise ValueError("Empty vertex")
    col_char = vertex[0]
    row_str = vertex[1:]
    if col_char not in COLS[:board_size]:
        raise ValueError(f"Invalid column {col_char} for board size {board_size}")
    col = COLS.index(col_char)
    row = board_size - int(row_str)
    if row < 0 or row >= board_size:
        raise ValueError(f"Invalid row in vertex {vertex}")
    return row, col


def point_to_vertex(row: int, col: int, board_size: int) -> str:
    """Convert 0-indexed (row, col) to GTP vertex."""
    display_row = board_size - row
    return f"{COLS[col]}{display_row}"


def move_from_vertex(vertex: str, board_size: int) -> Move:
    vertex = vertex.strip().lower()
    if vertex == "pass":
        return Move.pass_turn()
    if vertex == "resign":
        return Move.resign()
    row, col = vertex_to_point(vertex, board_size)
    return Move.play(row, col)


def vertex_from_move(move: Move, board_size: int) -> str:
    if move.is_pass:
        return "pass"
    if move.is_resign:
        return "resign"
    assert move.point is not None
    return point_to_vertex(move.point.row, move.point.col, board_size)
