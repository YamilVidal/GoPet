#!/usr/bin/env python
"""Inspect agent training runs (summary and plots)."""

import argparse
import sys

from inspect_training.plot_metrics import main as plot_main
from inspect_training.summary import main as summary_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect GoPet agent training")
    sub = parser.add_subparsers(dest="command", required=True)

    summary_parser = sub.add_parser("summary", help="Print checkpoint and history summary")
    summary_parser.add_argument("--checkpoint", default="agents/basic_cnn/checkpoints/basic_cnn.pt")

    plot_parser = sub.add_parser("plot", help="Plot loss and accuracy curves")
    plot_parser.add_argument("--history", default=None)
    plot_parser.add_argument("--checkpoint", default="agents/basic_cnn/checkpoints/basic_cnn.pt")
    plot_parser.add_argument("--output", default=None)
    plot_parser.add_argument("--show", action="store_true")

    args, remainder = parser.parse_known_args()
    if args.command == "summary":
        return summary_main(["--checkpoint", args.checkpoint])
    if args.command == "plot":
        plot_argv = ["--checkpoint", args.checkpoint]
        if args.history:
            plot_argv.extend(["--history", args.history])
        if args.output:
            plot_argv.extend(["--output", args.output])
        if args.show:
            plot_argv.append("--show")
        return plot_main(plot_argv)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
