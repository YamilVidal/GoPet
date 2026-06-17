#!/usr/bin/env python
"""Simulate games between two agents and report win rates."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from gopet.eval.load_agent import list_match_agents, load_match_agent
from gopet.eval.match import format_match_report, run_match_series


def build_parser() -> argparse.ArgumentParser:
    choices = list_match_agents()
    parser = argparse.ArgumentParser(description="Simulate agent vs agent matches")
    parser.add_argument(
        "--agent-a",
        required=True,
        choices=choices,
        help="First agent (plays Black on odd-numbered games)",
    )
    parser.add_argument(
        "--agent-b",
        required=True,
        choices=choices,
        help="Second agent (plays Black on even-numbered games)",
    )
    parser.add_argument("--games", type=int, default=1000, help="Number of games to play")
    parser.add_argument("--board-size", type=int, default=19)
    parser.add_argument("--komi", type=float, default=7.5)
    parser.add_argument(
        "--max-moves",
        type=int,
        default=None,
        help="Stop and score after this many plies (default: 2 * board_size^2)",
    )
    parser.add_argument("--checkpoint-a", default=None, help="Override checkpoint for --agent-a")
    parser.add_argument("--checkpoint-b", default=None, help="Override checkpoint for --agent-b")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed when using random agent")
    parser.add_argument("--quiet", action="store_true", help="Disable progress output")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.agent_a == args.agent_b and args.checkpoint_a == args.checkpoint_b:
        print("Warning: both sides use the same agent and checkpoint.", file=sys.stderr)

    try:
        agent_a = load_match_agent(
            args.agent_a,
            checkpoint=args.checkpoint_a,
            seed=args.seed,
        )
        agent_b = load_match_agent(
            args.agent_b,
            checkpoint=args.checkpoint_b,
            seed=args.seed + 1,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    name_a = args.agent_a if args.checkpoint_a is None else f"{args.agent_a}@{args.checkpoint_a}"
    name_b = args.agent_b if args.checkpoint_b is None else f"{args.agent_b}@{args.checkpoint_b}"

    stats = run_match_series(
        agent_a,
        agent_b,
        agent_a_name=name_a,
        agent_b_name=name_b,
        games=args.games,
        board_size=args.board_size,
        komi=args.komi,
        max_moves=args.max_moves,
        show_progress=not args.quiet,
    )
    print(format_match_report(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
