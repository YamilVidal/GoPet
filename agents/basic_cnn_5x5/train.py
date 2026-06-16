#!/usr/bin/env python
"""Train the basic_cnn_5x5 policy network on SGF replay data."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from agents.basic_cnn.train import AgentTrainConfig, run_training
from agents.basic_cnn_5x5.model import BasicPolicyCNN5x5
from gopet.encoding import NUM_BASIC_PLANES

AGENT_DIR = Path(__file__).resolve().parent

AGENT_CONFIG = AgentTrainConfig(
    agent_id="basic_cnn_5x5",
    checkpoint_stem="basic_cnn_5x5",
    checkpoint_dir=AGENT_DIR / "checkpoints",
    model_factory=lambda board_size: BasicPolicyCNN5x5(
        in_planes=NUM_BASIC_PLANES,
        board_size=board_size,
    ),
    description="Train BasicPolicyCNN5x5 on SGF data",
)


def main(argv: Optional[list[str]] = None) -> int:
    return run_training(AGENT_CONFIG, argv)


if __name__ == "__main__":
    raise SystemExit(main())
