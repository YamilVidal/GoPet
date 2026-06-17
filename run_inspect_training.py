#!/usr/bin/env python
"""Inspect agent training runs (summary and plots)."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from inspect_training.plot_metrics import main as plot_main
from inspect_training.summary import main as summary_main


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    parser = argparse.ArgumentParser(description="Inspect GoPet agent training")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("summary", help="Print checkpoint and history summary")
    sub.add_parser("plot", help="Plot loss and accuracy curves")

    args, remainder = parser.parse_known_args(argv)

    if args.command == "summary":
        return summary_main(remainder if remainder else None)
    if args.command == "plot":
        return plot_main(remainder if remainder else None)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
