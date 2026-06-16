# `score_estimation`

This folder contains score estimation utilities for GoPet.

## goscorer (territory + seki)

We vendor **`goscorer`** by David J Wu (“lightvector”) to provide **territory scoring with automated seki detection**:

- Upstream: `https://github.com/lightvector/goscorer`
- Vendored file: `score_estimation/goscorer.py` (from upstream `python/goscorer.py`)
- License: MIT, copied to `score_estimation/LICENSE.goscorer.txt`

### Wrapper API

Use `score_estimation.territory_seki`:

- `estimate_territory_score_with_seki(state, komi=7.5, ...)` → `ScoreEstimate`
- `estimate_winner_with_seki(state, komi=7.5, ...)` → winner color

### Intended use

This is most reliable for **late-game / scoring-like positions**, and is suitable for:

- estimating the final score to decide **whether to resign**
- computing a final result without manual seki handling

It can be wrong in unsettled fights (as with any endgame-oriented territory scorer).

