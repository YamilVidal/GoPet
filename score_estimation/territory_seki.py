"""Score estimation helpers: area scoring for live play, territory+seki for endgame.

This module wraps the vendored `score_estimation.goscorer` implementation for
Japanese-style territory scoring, and GoPet's area scorer for mid-game estimates.

Notes:
- `estimate_score_during_play` counts stones on the board plus surrounded empty
  points (area style). Use this for live display and resignation during a game.
- `estimate_territory_score_with_seki` counts only empty territory, dead stones,
  and prisoners — not living stones — so mid-game it stays near komi only.
- We currently pass capture counts as 0 because GoPet does not track them yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from gopet.game_state import GameState
from gopet.types import BLACK, EMPTY, WHITE, Color

from score_estimation import goscorer as _goscorer


@dataclass(frozen=True)
class ScoreEstimate:
    """Score estimate in points, from Black's perspective."""

    black_points: float
    white_points: float
    komi: float

    @property
    def score_diff_black_minus_white(self) -> float:
        return float(self.black_points - (self.white_points + self.komi))

    @property
    def winner(self) -> Color:
        return Color.black if self.score_diff_black_minus_white > 0 else Color.white

    def __str__(self) -> str:
        diff = self.score_diff_black_minus_white
        if diff > 0:
            return f"B+{diff:.1f}"
        return f"W+{abs(diff):.1f}"


def _board_to_stones_matrix(state: GameState) -> List[List[int]]:
    """Convert GoPet board to goscorer stones matrix (0 empty, 1 black, 2 white)."""

    stones = state.board.stones
    # board.stones is a NumPy array of ints in {0,1,2} matching gopet.types constants.
    # Convert to a nested Python list in [row][col] order.
    return stones.astype(np.int8).tolist()


def _default_marked_dead(state: GameState) -> List[List[bool]]:
    """Create a marked-dead matrix. GoPet does not support manual dead marking yet."""

    height, width = state.board.height, state.board.width
    return [[False for _ in range(width)] for _ in range(height)]


def estimate_area_score(state: GameState, *, komi: float = 7.5) -> ScoreEstimate:
    """Estimate score using area counting (stones + surrounded empty points)."""

    from gopet.scoring import compute_game_result

    result = compute_game_result(state, komi=komi)
    return ScoreEstimate(
        black_points=float(result.b),
        white_points=float(result.w),
        komi=komi,
    )


def estimate_score_during_play(state: GameState, *, komi: float = 7.5) -> ScoreEstimate:
    """Mid-game score estimate suitable for live display and resignation."""

    return estimate_area_score(state, komi=komi)


def estimate_territory_score_with_seki(
    state: GameState,
    *,
    komi: float = 7.5,
    marked_dead: Optional[Sequence[Sequence[bool]]] = None,
    score_false_eyes: bool = False,
) -> ScoreEstimate:
    """Estimate final score using goscorer territory scoring with seki detection.

    Returns a ScoreEstimate with Black/White points and komi.

    Args:
        state: current GameState
        komi: komi added to White
        marked_dead: optional bool matrix of stones marked dead by the user (same size as board)
        score_false_eyes: forwarded to goscorer to decide whether to count false eyes as territory
    """

    stones = _board_to_stones_matrix(state)
    if marked_dead is None:
        marked_dead_matrix = _default_marked_dead(state)
    else:
        marked_dead_matrix = [list(row) for row in marked_dead]

    # GoPet currently doesn't track captured stones explicitly, so we pass captures=0.
    # goscorer returns a dict {BLACK: black_points, WHITE: white_points_including_komi}.
    final_scores = _goscorer.final_territory_score(
        stones,
        marked_dead_matrix,
        black_points_from_captures=0,
        white_points_from_captures=0,
        komi=komi,
        score_false_eyes=score_false_eyes,
    )
    black_points = float(final_scores[_goscorer.BLACK])
    white_points_including_komi = float(final_scores[_goscorer.WHITE])
    white_points = white_points_including_komi - float(komi)

    return ScoreEstimate(black_points=black_points, white_points=white_points, komi=komi)


def estimate_winner_with_seki(
    state: GameState,
    *,
    komi: float = 7.5,
    marked_dead: Optional[Sequence[Sequence[bool]]] = None,
    score_false_eyes: bool = False,
) -> Color:
    return estimate_territory_score_with_seki(
        state,
        komi=komi,
        marked_dead=marked_dead,
        score_false_eyes=score_false_eyes,
    ).winner

