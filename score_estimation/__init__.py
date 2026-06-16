"""Score estimation helpers for GoPet.

Area scoring for live play; vendored `goscorer` territory + seki for endgame.
"""

from score_estimation.territory_seki import (
    ScoreEstimate,
    estimate_area_score,
    estimate_score_during_play,
    estimate_territory_score_with_seki,
    estimate_winner_with_seki,
)

__all__ = [
    "ScoreEstimate",
    "estimate_area_score",
    "estimate_score_during_play",
    "estimate_territory_score_with_seki",
    "estimate_winner_with_seki",
]

