"""
工具函数模块

提供常用的输入输出转换工具函数，参赛者可以直接使用或参考。
"""

import base64
import numpy as np
from typing import Optional, Union, Dict, Any


def to_numpy_1d(x: Any, *, name: str = "unknown") -> np.ndarray:
    """
    将各种格式转换为1D numpy数组
    
    支持格式：
    - None -> 空数组
    - numpy.ndarray -> 转换为float32
    - list/tuple -> 转换为numpy数组
    - dict (msgpack_numpy格式) -> 重构为numpy数组
    
    Args:
        x: 输入数据
        name: 数据名称（用于错误提示）
    
    Returns:
        1D numpy数组 (float32)
    
    Raises:
        TypeError: 如果输入格式不支持
    """
    if x is None:
        return np.zeros((0,), dtype=np.float32)
    
    if isinstance(x, np.ndarray):
        arr = x.astype(np.float32, copy=False)
    elif isinstance(x, dict) and "data" in x and "shape" in x:
        # 兼容 msgpack_numpy 的 array 表示（有些环境不会自动还原为 np.ndarray）
        arr = np.array(x["data"], dtype=np.float32).reshape(tuple(x["shape"]))
    elif isinstance(x, (list, tuple)):
        arr = np.array(x, dtype=np.float32)
    else:
        raise TypeError(f"{name}: unsupported type {type(x)}, value: {x}")
    
    return arr.reshape(-1)


def decode_jpeg_any(v: Union[str, bytes, None], *, name: str = "unknown") -> Optional[np.ndarray]:
    """
    解码JPEG图像（支持多种输入格式）
    
    支持格式：
    - None -> None
    - base64字符串 -> 解码为RGB numpy数组
    - bytes/bytearray -> 解码为RGB numpy数组
    
    Args:
        v: 输入图像数据
        name: 数据名称（用于错误提示）
    
    Returns:
        RGB numpy数组 (H, W, 3) 或 None
    
    Raises:
        ValueError: 如果解码失败
    """
    if v is None:
        return None
    
    # 如果是字符串，尝试base64解码
    if isinstance(v, str):
        try:
            v = base64.b64decode(v)
        except Exception:
            # 如果不是base64，返回None
            return None
    
    # 必须是bytes类型
    if not isinstance(v, (bytes, bytearray, memoryview)):
        return None
    
    # 解码JPEG
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise ImportError(
            "缺少依赖 opencv-python：如需解码 CAMERA_* 图像，请先安装 `pip install opencv-python`"
        ) from e

    img_array = np.frombuffer(v, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        raise ValueError(f"Failed to decode JPEG for {name}")
    
    # BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def normalize_joints_to_7d(joints: np.ndarray, control_mode: str) -> np.ndarray:
    """
    将关节数据标准化为7D格式（6关节 + 1夹爪）
    
    如果输入是8D（7关节 + 1夹爪），会裁剪为7D（取前6关节 + 夹爪）
    
    Args:
        joints: 关节数据（7D或8D）
        control_mode: 控制模式（"joints"或"end_pose"）
    
    Returns:
        7D数组 [j1-j6, gripper]
    """
    if control_mode != "joints":
        return joints
    
    if joints.size == 8:
        # 8D -> 7D: 取前6个关节 + 最后一个（夹爪）
        return np.concatenate([joints[:6], joints[-1:]], axis=0)
    elif joints.size == 7:
        return joints
    else:
        raise ValueError(
            f"Expected 7D or 8D joints, got {joints.size}D. "
            f"Shape: {joints.shape}, Value: {joints}"
        )


def convert_observation_to_model_input(
    obs: Dict[str, Any],
    control_mode: str,
    decode_images: bool = True,
) -> Dict[str, Any]:
    """
    将x2robot_client格式的observation转换为模型输入格式
    
    这是一个完整的转换函数，参赛者可以直接使用或参考。
    
    Args:
        obs: x2robot_client格式的observation
        control_mode: 控制模式（"joints"或"end_pose"）
        decode_images: 是否解码图像（如果模型需要numpy数组）
    
    Returns:
        模型输入格式的字典，包含：
        - left/front/right: RGB numpy数组或None
        - state: 14D状态向量 [left(7), right(7)]
        - instruction: 文本指令
    """
    # 1. 解码图像（如果需要）
    images = {}
    if decode_images:
        for key in ("CAMERA_LEFT", "CAMERA_FRONT", "CAMERA_RIGHT"):
            if key in obs:
                images[key.lower().replace("camera_", "")] = decode_jpeg_any(
                    obs.get(key), name=key
                )
    else:
        # 保持原始格式
        for key in ("CAMERA_LEFT", "CAMERA_FRONT", "CAMERA_RIGHT"):
            if key in obs:
                images[key.lower().replace("camera_", "")] = obs.get(key)
    
    # 2. 读取机器人状态
    follow1 = to_numpy_1d(obs.get("ACTION_FOLLOW1_POS"), name="ACTION_FOLLOW1_POS")
    follow2 = to_numpy_1d(obs.get("ACTION_FOLLOW2_POS"), name="ACTION_FOLLOW2_POS")
    
    # 3. 标准化关节数据（joints模式）
    if control_mode == "joints":
        follow1 = normalize_joints_to_7d(follow1, control_mode)
        follow2 = normalize_joints_to_7d(follow2, control_mode)
    
    # 4. 组装14D状态向量
    if follow1.size < 7 or follow2.size < 7:
        raise ValueError(
            f"Expected at least 7D per arm, got left={follow1.size}, right={follow2.size}"
        )
    
    state14 = np.concatenate([
        follow1[:7].astype(np.float32),
        follow2[:7].astype(np.float32)
    ], axis=0)
    
    # 5. 获取指令
    instruction = obs.get("instruction", "") or obs.get("prompt", "")
    
    # 6. 组装模型输入
    model_input = {
        **images,
        "state": state14,
        "instruction": instruction,
    }
    
    return model_input


def convert_model_output_to_x2robot_format(
    actions: np.ndarray,
    control_mode: str,
    action_horizon: int,
) -> Dict[str, Any]:
    """
    将模型输出转换为x2robot_client期望的格式
    
    假设模型输出是 (action_horizon, 14) 的numpy数组
    14D格式: [left(6), left_gripper(1), right(6), right_gripper(1)]
    
    Args:
        actions: 模型输出，shape (action_horizon, 14)
        control_mode: 控制模式（"joints"或"end_pose"）
        action_horizon: action序列长度
    
    Returns:
        x2robot_client格式的输出字典
    """
    # 确保是numpy数组
    if not isinstance(actions, np.ndarray):
        actions = np.array(actions)
    
    # 验证shape
    if actions.ndim != 2:
        raise ValueError(f"Expected 2D array (horizon, 14), got {actions.ndim}D, shape: {actions.shape}")
    
    if actions.shape[1] != 14:
        raise ValueError(f"Expected 14D per action, got {actions.shape[1]}D")
    
    if actions.shape[0] != action_horizon:
        # 允许不同的horizon，但记录警告
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Action horizon mismatch: expected {action_horizon}, got {actions.shape[0]}. "
            f"Using actual horizon."
        )
    
    # 分离左右臂
    left_actions = np.concatenate([
        actions[:, :6],   # left joints/ee (6)
        actions[:, 6:7],  # left gripper (1)
    ], axis=1)  # shape: (horizon, 7)
    
    right_actions = np.concatenate([
        actions[:, 7:13],  # right joints/ee (6)
        actions[:, 13:14], # right gripper (1)
    ], axis=1)  # shape: (horizon, 7)
    
    # 根据控制模式返回不同格式
    if control_mode == "joints":
        return {
            "FOLLOW1_JOINTS": left_actions.tolist(),
            "FOLLOW2_JOINTS": right_actions.tolist(),
            "FOLLOW1_POS": [],
            "FOLLOW2_POS": [],
        }
    else:  # end_pose
        return {
            "FOLLOW1_POS": left_actions.tolist(),
            "FOLLOW2_POS": right_actions.tolist(),
            "FOLLOW1_JOINTS": [],
            "FOLLOW2_JOINTS": [],
        }
