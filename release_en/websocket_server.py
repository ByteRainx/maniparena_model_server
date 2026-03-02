"""WebSocket server — do NOT modify this file.

Protocol:
  1. Client connects -> server sends metadata (msgpack)
  2. Client sends observation (msgpack) -> server returns actions (msgpack)
"""

import logging
import threading
from typing import Any, Dict

try:
    import msgpack
except ImportError:
    raise ImportError("Install: pip install msgpack")

try:
    import msgpack_numpy as m  # type: ignore
    m.patch()
except ImportError:
    m = None

try:
    import websockets
    import websockets.sync.server
    from websockets.exceptions import ConnectionClosed
except ImportError:
    raise ImportError("Install: pip install websockets")

logger = logging.getLogger(__name__)


class WebSocketModelServer:
    def __init__(self, policy: Any, host: str = "0.0.0.0", port: int = 8000):
        self.policy = policy
        self.host = host
        self.port = port
        self._lock = threading.Lock()

    def _handle_client(self, conn: websockets.sync.server.ServerConnection) -> None:
        addr = conn.remote_address
        logger.info("Client connected: %s", addr)
        try:
            metadata = getattr(self.policy, "metadata", {}) or {}
            conn.send(msgpack.packb(metadata, use_bin_type=True))
            logger.info("Sent metadata to %s: %s", addr, metadata)

            while True:
                try:
                    msg = conn.recv()
                    if isinstance(msg, str):
                        logger.warning("Text message from %s: %s", addr, msg)
                        continue

                    obs = msgpack.unpackb(msg, raw=False)
                    logger.debug("Observation keys: %s", list(obs.keys()))

                    try:
                        with self._lock:
                            result = self.policy.infer(obs)
                    except Exception as exc:
                        logger.exception("Inference error for %s", addr)
                        conn.send(f"Error: {exc}", text=True)
                        continue

                    conn.send(msgpack.packb(result, use_bin_type=True))
                    logger.debug("Sent result to %s", addr)

                except ConnectionClosed:
                    logger.info("Client disconnected: %s", addr)
                    break
        except Exception:
            logger.exception("Unhandled error for %s", addr)
        finally:
            logger.info("Handler finished: %s", addr)

    def serve_forever(self) -> None:
        uri = f"ws://{self.host}:{self.port}"
        logger.info("Starting server on %s", uri)
        with websockets.sync.server.serve(
            self._handle_client,
            host=self.host,
            port=self.port,
            max_size=None,
            compression=None,
        ) as server:
            logger.info("Server ready: %s", uri)
            server.serve_forever()
