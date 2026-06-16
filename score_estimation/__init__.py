"""Score estimation helpers for GoPet.

This package currently vendors and wraps the `goscorer` territory + seki scoring
algorithm by David J Wu ("lightvector").
"""

from score_estimation.territory_seki import (
    estimate_territory_score_with_seki,
    estimate_winner_with_seki,
)

__all__ = [
    "estimate_territory_score_with_seki",
    "estimate_winner_with_seki",
]

