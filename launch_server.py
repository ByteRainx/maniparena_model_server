"""
Server entrypoint.

Keeps a simple infer_new.py-style startup flow and clear CLI configuration.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from websocket_server import WebSocketModelServer
from my_policy import MyPolicy, DEFAULT_CHECKPOINT_PATH, DEFAULT_CONTROL_MODE, DEFAULT_ACTION_HORIZON, DEFAULT_DEVICE


def main():
    """Main function: start the WebSocket server."""
    
    parser = argparse.ArgumentParser(
        description="Universal model server - compatible with x2robot_client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Checkpoint path"
    )
    parser.add_argument(
        "--control-mode",
        type=str,
        default=DEFAULT_CONTROL_MODE,
        choices=["joints", "end_pose"],
        help="Control mode: joints or end_pose"
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=DEFAULT_ACTION_HORIZON,
        help="Action sequence length (model output horizon)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Device (cuda:0, cuda:1, cpu, etc.)"
    )
    
    # Server configuration
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (0.0.0.0 means listen on all interfaces)"
    )
    
    # Logging configuration
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Print startup configuration
    logger.info("=" * 60)
    logger.info("Starting universal model server")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Control Mode: {args.control_mode}")
    logger.info(f"Action Horizon: {args.action_horizon}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Server: {args.host}:{args.port}")
    logger.info("-" * 60)
    
    # Create policy
    try:
        policy = MyPolicy(
            checkpoint_path=args.checkpoint,
            control_mode=args.control_mode,
            action_horizon=args.action_horizon,
            device=args.device,
        )
    except Exception as e:
        logger.error(f"Failed to load policy: {e}", exc_info=True)
        sys.exit(1)
    
    # Create and run server
    server = WebSocketModelServer(
        policy=policy,
        host=args.host,
        port=args.port,
    )
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("\nServer stopped (Ctrl+C)")
    except Exception as e:
        logger.error(f"Server runtime error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

