"""
服务器启动文件

参考infer_new.py的简洁风格，配置清晰，使用简单。
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from websocket_server import WebSocketModelServer
from my_policy import MyPolicy, DEFAULT_CHECKPOINT_PATH, DEFAULT_CONTROL_MODE, DEFAULT_ACTION_HORIZON, DEFAULT_DEVICE


def main():
    """主函数：启动WebSocket服务器"""
    
    parser = argparse.ArgumentParser(
        description="通用模型服务器 - 适配x2robot_client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 模型配置参数
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Checkpoint路径"
    )
    parser.add_argument(
        "--control-mode",
        type=str,
        default=DEFAULT_CONTROL_MODE,
        choices=["joints", "end_pose"],
        help="控制模式: joints（关节控制）或 end_pose（末端位姿控制）"
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=DEFAULT_ACTION_HORIZON,
        help="Action序列长度（模型输出的action步数）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="设备 (cuda:0, cuda:1, cpu等)"
    )
    
    # 服务器配置参数
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务器端口"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务器地址（0.0.0.0表示监听所有网络接口）"
    )
    
    # 日志配置
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # 打印配置信息
    logger.info("=" * 60)
    logger.info("启动通用模型服务器")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Control Mode: {args.control_mode}")
    logger.info(f"Action Horizon: {args.action_horizon}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Server: {args.host}:{args.port}")
    logger.info("-" * 60)
    
    # 创建Policy
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
    
    # 创建并启动服务器
    server = WebSocketModelServer(
        policy=policy,
        host=args.host,
        port=args.port,
    )
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("\n服务器已停止（Ctrl+C）")
    except Exception as e:
        logger.error(f"服务器运行出错: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

