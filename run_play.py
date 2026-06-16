#!/usr/bin/env python
"""Start the GoPet browser play server."""

import argparse

from gopet.web.server import run_server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GoPet browser play server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--model",
        default=None,
        help="Optional PyTorch model path for the policy agent",
    )
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, debug=args.debug, model_path=args.model)


if __name__ == "__main__":
    main()
