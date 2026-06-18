#!/usr/bin/env python
"""Train a policy CNN agent on SGF replay data."""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Dict, List, Optional

TRAIN_MODULES: Dict[str, str] = {
    "basic_cnn": "agents.basic_cnn.train",
    "basic_cnn_5x5": "agents.basic_cnn_5x5.train",
    "cnn_triad": "agents.cnn_triad.train",
}


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--agent",
        default="basic_cnn",
        choices=sorted(TRAIN_MODULES.keys()),
        help="Agent to train (default: basic_cnn)",
    )
    pre_args, remaining = pre_parser.parse_known_args(argv)

    module = importlib.import_module(TRAIN_MODULES[pre_args.agent])
    return int(module.main(remaining))


if __name__ == "__main__":
    raise SystemExit(main())
