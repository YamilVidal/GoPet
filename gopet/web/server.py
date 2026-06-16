"""Flask server for browser play against GoPet agents."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from flask import Flask, jsonify, request, send_from_directory

from gopet.web.agents import Agent, build_default_agents, replay_state
from gopet.web.coords import vertex_from_move

WEB_ROOT = Path(__file__).resolve().parents[2] / "web"
STATIC_ROOT = WEB_ROOT / "static"


def create_app(agents: Optional[Dict[str, Agent]] = None, model_path: Optional[str] = None) -> Flask:
    agent_map = agents or build_default_agents(model_path=model_path)
    app = Flask(__name__, static_folder=str(STATIC_ROOT), static_url_path="/static")

    @app.route("/")
    def index() -> str:
        return send_from_directory(STATIC_ROOT, "index.html")

    @app.route("/api/agents")
    def list_agents():
        return jsonify({"agents": sorted(agent_map.keys())})

    @app.route("/api/select-move/<agent_name>", methods=["POST"])
    def select_move(agent_name: str):
        if agent_name not in agent_map:
            return jsonify({"error": f"Unknown agent: {agent_name}"}), 404

        payload = request.get_json(force=True)
        board_size = int(payload.get("board_size", 9))
        moves = payload.get("moves", [])

        try:
            state = replay_state(board_size, moves)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        if state.is_over():
            return jsonify({"error": "Game is already over"}), 400

        agent = agent_map[agent_name]
        against_human = bool(payload.get("against_human", False))
        if hasattr(agent, "set_game_context"):
            try:
                agent.set_game_context(against_human=against_human)  # type: ignore[attr-defined]
            except TypeError:
                # Backwards compatibility if agent has a different signature.
                pass
        bot_move = agent.select_move(state)
        return jsonify(
            {
                "bot_move": vertex_from_move(bot_move, board_size),
                "diagnostics": agent.diagnostics(),
            }
        )

    return app


def run_server(
    host: str = "127.0.0.1",
    port: int = 5000,
    debug: bool = False,
    model_path: Optional[str] = None,
) -> None:
    app = create_app(model_path=model_path)
    print(f"GoPet play server: http://{host}:{port}/")
    print(f"Static assets: {STATIC_ROOT}")
    if not (STATIC_ROOT / "vendor" / "jgoboard" / "dist" / "jgoboard.js").exists():
        print("Warning: jGoBoard not installed. Run: cd web && npm install")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server(debug=True)
