"""
WebSocket服务器核心实现

处理与x2robot_client的通信协议：
1. 客户端连接后，服务器立即发送metadata（msgpack编码）
2. 客户端发送observation（msgpack编码）
3. 服务器调用policy.infer(obs)并返回结果（msgpack编码）
4. 如果出错，服务器发送文本错误消息
"""

import logging
import threading
from typing import Any, Dict

try:
    import msgpack
except ImportError:
    raise ImportError("请安装: pip install msgpack")

# msgpack-numpy is recommended (client may send numpy arrays), but keep it optional:
# if not installed, numpy arrays will arrive as a dict (data/shape) and can be handled in convert_input/utils.
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
    raise ImportError("请安装: pip install websockets")

logger = logging.getLogger(__name__)


class WebSocketModelServer:
    """WebSocket服务器，处理x2robot_client连接"""
    
    def __init__(
        self,
        policy: Any,
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        """
        初始化服务器
        
        Args:
            policy: 实现了infer()方法和metadata属性的Policy对象
            host: 服务器地址
            port: 服务器端口
        """
        self.policy = policy
        self.host = host
        self.port = port
        self._infer_lock = threading.Lock()
    
    def _handle_client(self, conn: websockets.sync.server.ServerConnection) -> None:
        """处理单个客户端连接"""
        client_addr = conn.remote_address
        logger.info(f"Client connected: {client_addr}")
        
        try:
            # 1. 连接建立后立即发送metadata
            metadata = getattr(self.policy, "metadata", {}) or {}
            # Use explicit msgpack options to avoid bytes keys/values surprises across environments.
            metadata_bytes = msgpack.packb(metadata, use_bin_type=True)
            conn.send(metadata_bytes)
            logger.info(f"Sent metadata to {client_addr}: {metadata}")
            
            # 2. 循环处理推理请求
            while True:
                try:
                    # 接收observation
                    message = conn.recv()
                    
                    # 如果收到文本消息（可能是错误或控制消息），记录并继续
                    if isinstance(message, str):
                        logger.warning(f"Received text message from {client_addr}: {message}")
                        continue
                    
                    # 解码observation
                    # raw=False ensures str keys (compatible with x2robot_client's dict access patterns).
                    obs = msgpack.unpackb(message, raw=False)
                    logger.debug(f"Received observation from {client_addr}, keys: {list(obs.keys())}")
                    
                    # 调用policy进行推理
                    try:
                        with self._infer_lock:
                            result = self.policy.infer(obs)
                    except Exception as exc:
                        logger.exception(f"Policy inference error for {client_addr}")
                        # 发送错误消息（文本格式）
                        conn.send(f"Error in policy inference: {exc}", text=True)
                        continue
                    
                    # 编码并发送结果
                    result_bytes = msgpack.packb(result, use_bin_type=True)
                    conn.send(result_bytes)
                    logger.debug(f"Sent result to {client_addr}")
                    
                except ConnectionClosed:
                    logger.info(f"Client disconnected: {client_addr}")
                    break
                    
        except Exception as e:
            logger.exception(f"Unhandled error in client handler for {client_addr}")
        finally:
            logger.info(f"Client handler finished: {client_addr}")
    
    def serve_forever(self) -> None:
        """启动服务器并持续运行"""
        uri = f"ws://{self.host}:{self.port}"
        logger.info(f"Starting WebSocket server on {uri}")
        
        with websockets.sync.server.serve(
            self._handle_client,
            host=self.host,
            port=self.port,
            max_size=None,  # 不限制消息大小
            compression=None,  # 禁用压缩以提高性能
        ) as server:
            logger.info("=" * 60)
            logger.info(f"WebSocket server is running on {uri}")
            logger.info("Server is ready and waiting for connections...")
            logger.info("=" * 60)
            server.serve_forever()
