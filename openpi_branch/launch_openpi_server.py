#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _add_paths() -> None:
    repo_root = Path(__file__).resolve().parents[3]  # /.../rain
    openpi_root = repo_root / "vla" / "openpi"
    openpi_src = openpi_root / "src"
    openpi_client_src = openpi_root / "packages" / "openpi-client" / "src"
    maniparena_root = repo_root / "vla" / "maniparena_model_server"

    # Insert in a safe order (avoid shadowing OpenPI with this repo's own folders).
    for p in (maniparena_root, openpi_client_src, openpi_src, openpi_root):
        ps = str(p)
        if ps not in sys.path:
            sys.path.insert(0, ps)


def main() -> None:
    _add_paths()

    from websocket_server import WebSocketModelServer
    from openpi_policy import OpenPIX2RobotPolicy

    parser = argparse.ArgumentParser(
        description="OpenPI model server (x2robot_client websocket protocol)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="OpenPI training config name")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Checkpoint step directory")
    parser.add_argument("--default-prompt", type=str, default=None)
    parser.add_argument("--model-device", type=str, default="cuda")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--warmup", action="store_true", help="Run one dummy inference after loading")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    logger = logging.getLogger(__name__)

    logger.info(
        "Loading OpenPI policy: config=%s checkpoint=%s device=%s",
        args.config,
        args.checkpoint_dir,
        args.model_device,
    )
    policy = OpenPIX2RobotPolicy(
        config_name=args.config,
        checkpoint_dir=args.checkpoint_dir,
        default_prompt=args.default_prompt,
        pytorch_device=args.model_device,
    )
    logger.info("Policy metadata: %s", getattr(policy, "metadata", {}))

    if args.warmup:
        import numpy as np
        dummy = {
            "state": {
                "follow1_pos": np.zeros(7, dtype=np.float32),
                "follow2_pos": np.zeros(7, dtype=np.float32),
            },
            "views": {},
            "instruction": np.array([args.default_prompt or "warmup"], dtype=object),
        }
        _ = policy.infer(dummy)
        logger.info("Warmup completed")

    server = WebSocketModelServer(policy=policy, host=args.host, port=args.port)
    logger.info("Serving websocket on ws://%s:%s", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
