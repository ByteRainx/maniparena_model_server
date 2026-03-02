"""CLI entry point — do NOT modify this file."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from websocket_server import WebSocketModelServer
from my_policy import (
    MyPolicy,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CONTROL_MODE,
    DEFAULT_ACTION_HORIZON,
    DEFAULT_DEVICE,
)


def main():
    parser = argparse.ArgumentParser(
        description="ManipArena Model Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH,
                        help="Model checkpoint path")
    parser.add_argument("--control-mode", default=DEFAULT_CONTROL_MODE,
                        choices=["joints", "end_pose"])
    parser.add_argument("--action-horizon", type=int,
                        default=DEFAULT_ACTION_HORIZON)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log = logging.getLogger(__name__)

    log.info("Checkpoint : %s", args.checkpoint)
    log.info("Control    : %s", args.control_mode)
    log.info("Horizon    : %d", args.action_horizon)
    log.info("Device     : %s", args.device)
    log.info("Server     : %s:%d", args.host, args.port)

    policy = MyPolicy(
        checkpoint_path=args.checkpoint,
        control_mode=args.control_mode,
        action_horizon=args.action_horizon,
        device=args.device,
    )

    server = WebSocketModelServer(
        policy=policy, host=args.host, port=args.port,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Server stopped.")


if __name__ == "__main__":
    main()
