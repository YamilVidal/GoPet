"""Registered policy agents and default checkpoint paths."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[1]

AGENT_CHECKPOINTS: Dict[str, Path] = {
    "basic_cnn": ROOT / "agents" / "basic_cnn" / "checkpoints" / "basic_cnn.pt",
    "basic_cnn_5x5": ROOT / "agents" / "basic_cnn_5x5" / "checkpoints" / "basic_cnn_5x5.pt",
}


def list_policy_agents() -> list[str]:
    return sorted(AGENT_CHECKPOINTS.keys())


def default_checkpoint(agent_id: str) -> Path:
    try:
        return AGENT_CHECKPOINTS[agent_id]
    except KeyError as exc:
        known = ", ".join(list_policy_agents())
        raise ValueError(f"Unknown agent '{agent_id}'. Known agents: {known}") from exc


def resolve_checkpoint(agent_id: str, checkpoint: Optional[str]) -> Path:
    if checkpoint:
        return Path(checkpoint)
    return default_checkpoint(agent_id)
