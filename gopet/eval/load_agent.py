"""Load play agents for evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from agents.registry import list_policy_agents, resolve_checkpoint
from gopet.web.agents import Agent, PolicyAgent, RandomAgent

MATCH_AGENT_CHOICES = ["random", *list_policy_agents()]


def list_match_agents() -> List[str]:
    return list(MATCH_AGENT_CHOICES)


def load_match_agent(
    agent_id: str,
    *,
    checkpoint: Optional[str] = None,
    seed: Optional[int] = None,
    device: str = "cpu",
) -> Agent:
    if agent_id == "random":
        return RandomAgent(seed=seed)

    path = resolve_checkpoint(agent_id, checkpoint)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found for agent '{agent_id}': {path}. Train the agent first."
        )

    agent = PolicyAgent(str(path), device=device)
    agent.set_game_context(against_human=False)
    return agent
