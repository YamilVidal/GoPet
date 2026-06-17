"""Flask server for browser play against GoPet agents."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from flask import Flask, jsonify, request, send_from_directory

from gopet.types import Move
from gopet.web.agents import (
    Agent,
    PolicyAgentCache,
    build_default_agents,
    list_policy_checkpoints,
    replay_state,
)
from gopet.web.coords import vertex_from_move

WEB_ROOT = Path(__file__).resolve().parents[2] / "web"
STATIC_ROOT = WEB_ROOT / "static"


def _parse_select_move_payload(payload: object) -> tuple[int, list[str], bool, Optional[str]]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")

    board_size_raw = payload.get("board_size", 9)
    if isinstance(board_size_raw, bool) or not isinstance(board_size_raw, int):
        raise ValueError("board_size must be an integer")
    if board_size_raw <= 0:
        raise ValueError("board_size must be positive")

    moves_raw = payload.get("moves", [])
    if not isinstance(moves_raw, list) or not all(isinstance(item, str) for item in moves_raw):
        raise ValueError("moves must be a list of vertex strings")

    against_human_raw = payload.get("against_human", False)
    if not isinstance(against_human_raw, bool):
        raise ValueError("against_human must be a boolean")

    checkpoint_raw = payload.get("checkpoint", None)
    if checkpoint_raw is not None and not isinstance(checkpoint_raw, str):
        raise ValueError("checkpoint must be a string or null")

    return board_size_raw, moves_raw, against_human_raw, checkpoint_raw


def create_app(agents: Optional[Dict[str, Agent]] = None, model_path: Optional[str] = None) -> Flask:
    agent_map = agents or build_default_agents(model_path=model_path)
    policy_cache = PolicyAgentCache()
    app = Flask(__name__, static_folder=str(STATIC_ROOT), static_url_path="/static")

    @app.route("/")
    def index() -> str:
        return send_from_directory(STATIC_ROOT, "index.html")

    @app.route("/api/agents")
    def list_agents():
        checkpoints: Dict[str, list[str]] = {}
        for name in sorted(agent_map.keys()):
            if name == "random" or name == "policy":
                checkpoints[name] = []
                continue
            try:
                checkpoints[name] = list_policy_checkpoints(name)
            except Exception:
                checkpoints[name] = []
        return jsonify({"agents": sorted(agent_map.keys()), "checkpoints": checkpoints})

    @app.route("/api/select-move/<agent_name>", methods=["POST"])
    def select_move(agent_name: str):
        if agent_name not in agent_map:
            return jsonify({"error": f"Unknown agent: {agent_name}"}), 404

        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400

        payload = request.get_json(silent=True)
        try:
            board_size, moves, against_human, checkpoint = _parse_select_move_payload(payload)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        try:
            state = replay_state(board_size, moves)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        if state.is_over():
            return jsonify({"error": "Game is already over"}), 400

        agent = agent_map[agent_name]
        if checkpoint:
            if agent_name == "random":
                return jsonify({"error": "random agent does not support checkpoints"}), 400
            if agent_name == "policy":
                return jsonify({"error": "policy agent does not support selectable checkpoints"}), 400
            try:
                agent = policy_cache.get(agent_name, checkpoint)
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
        if hasattr(agent, "set_game_context"):
            try:
                agent.set_game_context(against_human=against_human)  # type: ignore[attr-defined]
            except TypeError:
                pass

        selection = agent.select_move_with_diagnostics(state)
        bot_move = selection.move
        if not state.is_valid_move(bot_move):
            bot_move = Move.pass_turn()
            if not state.is_valid_move(bot_move):
                bot_move = Move.resign()

        return jsonify(
            {
                "bot_move": vertex_from_move(bot_move, board_size),
                "diagnostics": selection.diagnostics,
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
