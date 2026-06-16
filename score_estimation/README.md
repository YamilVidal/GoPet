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

**Live play (display + resignation):** use `estimate_score_during_play` / `estimate_area_score`.
These count stones on the board plus surrounded empty points (area style), so the
estimate moves as the game progresses.

**Endgame / scoring positions:** use `estimate_territory_score_with_seki`.
This is Japanese-style territory scoring with seki detection. It counts empty
territory, dead stones, and prisoners — **not** living stones on the board — so
mid-game it stays near komi only (e.g. stuck at W+7.5 on an empty-looking board).

Territory scoring can still be wrong in unsettled fights (as with any endgame-oriented scorer).

## Area scoring (live play)

`estimate_score_during_play` wraps GoPet's `gopet.scoring.compute_game_result`:

- `estimate_area_score(state, komi=7.5)` → `ScoreEstimate`
- `estimate_score_during_play(state, komi=7.5)` → same (alias for clarity)

Used by `PolicyAgent` diagnostics and resignation when playing humans.

