"""Default checkpoint paths for inspectable agents."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from agents.registry import (
    AGENT_CHECKPOINTS,
    default_checkpoint,
    list_policy_agents,
    resolve_checkpoint,
)

__all__ = [
    "AGENT_CHECKPOINTS",
    "add_agent_arguments",
    "default_checkpoint",
    "list_agents",
    "resolve_checkpoint",
]


def list_agents() -> list[str]:
    return list_policy_agents()


def add_agent_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--agent",
        default="basic_cnn",
        choices=list_agents(),
        help="Agent to inspect (default: basic_cnn)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override checkpoint path (default: from --agent)",
    )
