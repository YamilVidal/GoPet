"""Head-to-head match simulation between play agents."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from gopet.game_state import GameState
from gopet.progress import finish_progress_line, format_duration_hms, report_item_progress
from gopet.scoring import compute_game_result
from gopet.types import Color, Move
from gopet.web.agents import Agent


@dataclass(frozen=True)
class GameOutcome:
    winner: Optional[Color]
    end_reason: str
    move_count: int
    black_agent: str
    white_agent: str


@dataclass
class MatchStats:
    agent_a: str
    agent_b: str
    games: int
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0
    wins_black: int = 0
    wins_white: int = 0
    resignations: int = 0
    max_move_stops: int = 0
    by_end_reason: Dict[str, int] = field(default_factory=dict)

    @property
    def win_rate_a(self) -> float:
        if self.games == 0:
            return 0.0
        return self.wins_a / self.games

    @property
    def win_rate_b(self) -> float:
        if self.games == 0:
            return 0.0
        return self.wins_b / self.games


def default_max_moves(board_size: int) -> int:
    """Enough plies to fill the board twice over before a forced stop."""

    return 2 * board_size * board_size


def resolve_winner(state: GameState, *, komi: float, end_reason: str) -> Optional[Color]:
    """Determine the winner for a finished or force-stopped game.

    Komi is applied only when the game ends by scoring (two consecutive passes).
    Forced ``max_moves`` stops use raw area counts so an unfinished game is not
    automatically awarded to White via komi. Ties on ``max_moves`` count as draws.
    """

    if state.last_move is not None and state.last_move.is_resign:
        # Player to move resigned; opponent (next_player) wins.
        return state.next_player

    apply_komi = end_reason == "score"
    result = compute_game_result(state, komi=komi if apply_komi else 0.0)
    black_score = result.b
    white_score = result.w + (komi if apply_komi else 0.0)
    if black_score > white_score:
        return Color.black
    if white_score > black_score:
        return Color.white
    if end_reason == "max_moves":
        return None
    return Color.white


def play_game(
    black: Agent,
    white: Agent,
    *,
    black_name: str,
    white_name: str,
    board_size: int = 19,
    komi: float = 7.5,
    max_moves: Optional[int] = None,
) -> GameOutcome:
    if max_moves is None:
        max_moves = default_max_moves(board_size)
    if max_moves <= 0:
        raise ValueError("max_moves must be positive")
    state = GameState.new_game(board_size)
    move_count = 0

    while not state.is_over() and move_count < max_moves:
        agent = black if state.next_player == Color.black else white
        move = agent.select_move(state)
        if not state.is_valid_move(move):
            move = Move.resign()
        state = state.apply_move(move)
        move_count += 1

    if state.is_over() and state.last_move is not None and state.last_move.is_resign:
        end_reason = "resign"
    elif state.is_over():
        end_reason = "score"
    else:
        end_reason = "max_moves"

    winner = resolve_winner(state, komi=komi, end_reason=end_reason)
    return GameOutcome(
        winner=winner,
        end_reason=end_reason,
        move_count=move_count,
        black_agent=black_name,
        white_agent=white_name,
    )


def run_match_series(
    agent_a: Agent,
    agent_b: Agent,
    *,
    agent_a_name: str,
    agent_b_name: str,
    games: int = 1000,
    board_size: int = 19,
    komi: float = 7.5,
    max_moves: Optional[int] = None,
    show_progress: bool = True,
) -> MatchStats:
    if max_moves is None:
        max_moves = default_max_moves(board_size)
    stats = MatchStats(agent_a=agent_a_name, agent_b=agent_b_name, games=games)
    next_progress = 5
    phase_start = time.time()

    if show_progress:
        print(f"Match: {agent_a_name} vs {agent_b_name} ({games:,} games, colors alternate)")

    for game_index in range(1, games + 1):
        if game_index % 2 == 1:
            black_agent, white_agent = agent_a, agent_b
            black_name, white_name = agent_a_name, agent_b_name
        else:
            black_agent, white_agent = agent_b, agent_a
            black_name, white_name = agent_b_name, agent_a_name

        outcome = play_game(
            black_agent,
            white_agent,
            black_name=black_name,
            white_name=white_name,
            board_size=board_size,
            komi=komi,
            max_moves=max_moves,
        )

        if outcome.winner == Color.black:
            stats.wins_black += 1
        elif outcome.winner == Color.white:
            stats.wins_white += 1
        else:
            stats.draws += 1

        black_is_a = game_index % 2 == 1
        if outcome.winner == Color.black:
            if black_is_a:
                stats.wins_a += 1
            else:
                stats.wins_b += 1
        elif outcome.winner == Color.white:
            if black_is_a:
                stats.wins_b += 1
            else:
                stats.wins_a += 1

        stats.by_end_reason[outcome.end_reason] = stats.by_end_reason.get(outcome.end_reason, 0) + 1
        if outcome.end_reason == "resign":
            stats.resignations += 1
        if outcome.end_reason == "max_moves":
            stats.max_move_stops += 1

        if show_progress:
            next_progress = report_item_progress(
                game_index,
                games,
                phase="match",
                next_progress=next_progress,
                phase_start=phase_start,
            )

    finish_progress_line(games, show_progress=show_progress)
    if show_progress:
        elapsed = time.time() - phase_start
        print(f"Finished {games:,} games in {format_duration_hms(elapsed)}")

    return stats


def format_match_report(stats: MatchStats) -> str:
    lines = [
        f"Results ({stats.games:,} games, colors alternated)",
        f"  {stats.agent_a}: {stats.wins_a:,} wins ({stats.win_rate_a * 100:.1f}%)",
        f"  {stats.agent_b}: {stats.wins_b:,} wins ({stats.win_rate_b * 100:.1f}%)",
        f"  Black won: {stats.wins_black:,}  White won: {stats.wins_white:,}",
    ]
    if stats.draws:
        lines.append(f"  Draws: {stats.draws:,}")
    if stats.resignations:
        lines.append(f"  Resignations: {stats.resignations:,}")
    if stats.max_move_stops:
        lines.append(f"  Max-move stops: {stats.max_move_stops:,}")
    if stats.by_end_reason:
        reasons = ", ".join(f"{key}={value}" for key, value in sorted(stats.by_end_reason.items()))
        lines.append(f"  End reasons: {reasons}")
    return "\n".join(lines)
