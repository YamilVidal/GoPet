"""Tests for web coordinate conversion."""

from gopet.types import Move
from gopet.web.coords import move_from_vertex, point_to_vertex, vertex_from_move, vertex_to_point


def test_vertex_round_trip_9x9() -> None:
    for row in range(9):
        for col in range(9):
            vertex = point_to_vertex(row, col, 9)
            assert vertex_to_point(vertex, 9) == (row, col)


def test_move_from_vertex_pass() -> None:
    assert move_from_vertex("pass", 9).is_pass


def test_vertex_from_move() -> None:
    move = Move.play(4, 4)
    assert vertex_from_move(move, 9) == "E5"
