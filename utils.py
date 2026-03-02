"""
工具函数模块

提供常用的输入输出转换工具函数。
兼容新格式（嵌套 state/views/instruction）和旧格式（扁平大写 key）。
"""

import base64
import logging
import numpy as np
from typing import Optional, Union, Dict, Any

logger = logging.getLogger(__name__)


def to_numpy_1d(x: Any, *, name: str = "unknown") -> np.ndarray:
    """
    将各种格式转换为1D numpy数组

    支持格式：
    - None -> 空数组
    - numpy.ndarray -> 转换为float32
    - list/tuple -> 转换为numpy数组
    - dict (msgpack_numpy格式) -> 重构为numpy数组
    """
    if x is None:
        return np.zeros((0,), dtype=np.float32)

    if isinstance(x, np.ndarray):
        arr = x.astype(np.float32, copy=False)
    elif isinstance(x, dict) and "data" in x and "shape" in x:
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
    """
    if v is None:
        return None

    if isinstance(v, np.ndarray):
        return v

    if isinstance(v, str):
        try:
            v = base64.b64decode(v)
        except Exception:
            return None

    if not isinstance(v, (bytes, bytearray, memoryview)):
        return None

    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise ImportError(
            "缺少依赖 opencv-python：请先安装 `pip install opencv-python`"
        ) from e

    img_array = np.frombuffer(v, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError(f"Failed to decode JPEG for {name}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def normalize_joints_to_7d(joints: np.ndarray, control_mode: str) -> np.ndarray:
    """
    将关节数据标准化为7D格式（6关节 + 1夹爪）

    如果输入是8D（7关节 + 1夹爪），会裁剪为7D（取前6关节 + 夹爪）
    """
    if control_mode != "joints":
        return joints

    if joints.size == 8:
        return np.concatenate([joints[:6], joints[-1:]], axis=0)
    elif joints.size == 7:
        return joints
    else:
        raise ValueError(
            f"Expected 7D or 8D joints, got {joints.size}D. "
            f"Shape: {joints.shape}, Value: {joints}"
        )


# ---------------------------------------------------------------------------
# Format detection & instruction extraction
# ---------------------------------------------------------------------------

def _is_new_format(obs: Dict[str, Any]) -> bool:
    """检测 observation 是否使用新的嵌套格式 (state/views/instruction)。"""
    return "state" in obs and isinstance(obs["state"], dict)


def _extract_instruction(obs: Dict[str, Any]) -> str:
    """从 observation 中提取文本指令，兼容新旧格式。"""
    raw = obs.get("instruction", None)
    if raw is None:
        raw = obs.get("prompt", "")
    if isinstance(raw, np.ndarray):
        if raw.size > 0:
            return str(raw.flat[0])
        return ""
    return str(raw) if raw else ""


# ---------------------------------------------------------------------------
# Observation -> Model Input
# ---------------------------------------------------------------------------

def convert_observation_to_model_input(
    obs: Dict[str, Any],
    control_mode: str,
    decode_images: bool = True,
) -> Dict[str, Any]:
    """
    将x2robot_client格式的observation转换为模型输入格式。

    同时兼容新格式（嵌套 state/views/instruction）和旧格式（扁平大写 key）。

    Returns:
        模型输入格式的字典，包含：
        - left/front/right: RGB numpy数组或原始数据
        - state: 14D状态向量 [left(7), right(7)]
        - instruction: 文本指令
        - state/* : 额外状态字段（仅新格式，如 head_pos, lift 等）
    """
    new_fmt = _is_new_format(obs)

    # --- 图像 ---
    images: Dict[str, Any] = {}
    if new_fmt:
        views = obs.get("views", {})
        cam_mapping = {"camera_left": "left", "camera_front": "front", "camera_right": "right"}
        for src_key, dst_key in cam_mapping.items():
            raw = views.get(src_key)
            if raw is not None:
                images[dst_key] = decode_jpeg_any(raw, name=src_key) if decode_images else raw
    else:
        for key in ("CAMERA_LEFT", "CAMERA_FRONT", "CAMERA_RIGHT"):
            raw = obs.get(key)
            if raw is not None:
                dst_key = key.lower().replace("camera_", "")
                images[dst_key] = decode_jpeg_any(raw, name=key) if decode_images else raw

    # --- 双臂状态 ---
    if new_fmt:
        state_dict = obs["state"]
        if control_mode == "joints":
            follow1 = to_numpy_1d(
                state_dict.get("follow1_joints", state_dict.get("follow1_pos")),
                name="follow1",
            )
            follow2 = to_numpy_1d(
                state_dict.get("follow2_joints", state_dict.get("follow2_pos")),
                name="follow2",
            )
        else:
            follow1 = to_numpy_1d(state_dict.get("follow1_pos"), name="follow1_pos")
            follow2 = to_numpy_1d(state_dict.get("follow2_pos"), name="follow2_pos")
    else:
        follow1 = to_numpy_1d(
            obs.get("ACTION_FOLLOW1_POS") or obs.get("follow1_pos"),
            name="follow1",
        )
        follow2 = to_numpy_1d(
            obs.get("ACTION_FOLLOW2_POS") or obs.get("follow2_pos"),
            name="follow2",
        )

    if control_mode == "joints":
        follow1 = normalize_joints_to_7d(follow1, control_mode)
        follow2 = normalize_joints_to_7d(follow2, control_mode)

    if follow1.size < 7 or follow2.size < 7:
        raise ValueError(
            f"Expected at least 7D per arm, got left={follow1.size}, right={follow2.size}"
        )

    state14 = np.concatenate([
        follow1[:7].astype(np.float32),
        follow2[:7].astype(np.float32),
    ], axis=0)

    instruction = _extract_instruction(obs)

    model_input: Dict[str, Any] = {
        **images,
        "state": state14,
        "instruction": instruction,
    }

    # --- 额外状态字段（透传，供模型按需使用）---
    _EXTRA_STATE_KEYS = [
        "follow1_pos", "follow2_pos",
        "follow1_joints", "follow2_joints",
        "follow1_joints_cur", "follow2_joints_cur",
        "follow1_joints_dev", "follow2_joints_dev",
        "head_pos", "lift", "car_pose",
        "velocity_decomposed", "velocity_decomposed_odom",
    ]
    if new_fmt:
        state_dict = obs["state"]
        for k in _EXTRA_STATE_KEYS:
            v = state_dict.get(k)
            if v is not None:
                model_input[f"state/{k}"] = to_numpy_1d(v, name=k)
    else:
        for k in _EXTRA_STATE_KEYS:
            v = obs.get(k) or obs.get(k.upper()) or obs.get(f"ACTION_{k.upper()}")
            if v is not None:
                model_input[f"state/{k}"] = to_numpy_1d(v, name=k)

    return model_input


# ---------------------------------------------------------------------------
# Model Output -> x2robot_client Response
# ---------------------------------------------------------------------------

def convert_model_output_to_x2robot_format(
    actions: np.ndarray,
    control_mode: str,
    action_horizon: int,
    *,
    head_actions: Optional[np.ndarray] = None,
    lift_actions: Optional[np.ndarray] = None,
    velocity_actions: Optional[np.ndarray] = None,
    car_pose_odom_actions: Optional[np.ndarray] = None,
    model_output_text: str = "",
) -> Dict[str, Any]:
    """
    将模型输出转换为x2robot_client期望的新格式（小写 key）。

    Args:
        actions: 双臂动作，shape (T, 14)。14D格式: [left(6), left_gripper(1), right(6), right_gripper(1)]
        control_mode: 控制模式
        action_horizon: action序列长度（仅用于校验提醒）
        head_actions: 头部动作 (T, 2)  [yaw, pitch]，可选
        lift_actions: 升降机动作 (T, 1)，可选
        velocity_actions: 底盘速度 (T, 3) [vx, vy, omega]，可选
        car_pose_odom_actions: 底盘位姿 (T, 3)，可选
        model_output_text: 模型文本输出，可选
    """
    if not isinstance(actions, np.ndarray):
        actions = np.array(actions)

    if actions.ndim != 2:
        raise ValueError(f"Expected 2D array (horizon, 14), got {actions.ndim}D, shape: {actions.shape}")

    if actions.shape[1] != 14:
        raise ValueError(f"Expected 14D per action, got {actions.shape[1]}D")

    if actions.shape[0] != action_horizon:
        logger.warning(
            f"Action horizon mismatch: expected {action_horizon}, got {actions.shape[0]}. "
            f"Using actual horizon."
        )

    left_actions = np.concatenate([actions[:, :6], actions[:, 6:7]], axis=1)
    right_actions = np.concatenate([actions[:, 7:13], actions[:, 13:14]], axis=1)

    if control_mode == "joints":
        result: Dict[str, Any] = {
            "follow1_joints": left_actions.tolist(),
            "follow2_joints": right_actions.tolist(),
            "follow1_pos": left_actions.tolist(),
            "follow2_pos": right_actions.tolist(),
        }
    else:
        result = {
            "follow1_pos": left_actions.tolist(),
            "follow2_pos": right_actions.tolist(),
        }

    # --- 额外输出字段 ---
    if head_actions is not None:
        result["head_pos"] = np.asarray(head_actions).tolist()
    if lift_actions is not None:
        result["lift"] = np.asarray(lift_actions).tolist()
    if velocity_actions is not None:
        result["velocity_decomposed"] = np.asarray(velocity_actions).tolist()
    if car_pose_odom_actions is not None:
        result["car_pose_odom"] = np.asarray(car_pose_odom_actions).tolist()
    if model_output_text:
        result["model_output_text"] = model_output_text

    return result


# ---------------------------------------------------------------------------
# Legacy format converter (旧格式兼容)
# ---------------------------------------------------------------------------

def convert_model_output_to_legacy_format(
    actions: np.ndarray,
    control_mode: str,
    action_horizon: int,
    *,
    head_actions: Optional[np.ndarray] = None,
    lift_actions: Optional[np.ndarray] = None,
    car_pose_actions: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    将模型输出转换为 CX001 客户端格式（大写 key）。

    CX001 客户端期望的 key：
      FOLLOW1_POS / FOLLOW2_POS  — 双臂轨迹 (T, 7)
      HEAD_POS                   — 头部 (T, 2)  [yaw, pitch]
      LIFT_OUT                   — 升降 (T, 1)
      CAR_POSE_OUT               — 底盘 (T, 3)  [x, y, theta]

    所有值必须是 List[List[float]]（内部已调用 .tolist()）。
    """
    if not isinstance(actions, np.ndarray):
        actions = np.array(actions)

    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(f"Expected shape (T, 14), got {actions.shape}")

    left_actions = np.concatenate([actions[:, :6], actions[:, 6:7]], axis=1)
    right_actions = np.concatenate([actions[:, 7:13], actions[:, 13:14]], axis=1)

    if control_mode == "joints":
        result: Dict[str, Any] = {
            "FOLLOW1_JOINTS": left_actions.tolist(),
            "FOLLOW2_JOINTS": right_actions.tolist(),
            "FOLLOW1_POS": [],
            "FOLLOW2_POS": [],
        }
    else:
        result = {
            "FOLLOW1_POS": left_actions.tolist(),
            "FOLLOW2_POS": right_actions.tolist(),
            "FOLLOW1_JOINTS": [],
            "FOLLOW2_JOINTS": [],
        }

    if head_actions is not None:
        result["HEAD_POS"] = np.asarray(head_actions).tolist()
    if lift_actions is not None:
        result["LIFT_OUT"] = np.asarray(lift_actions).tolist()
    if car_pose_actions is not None:
        result["CAR_POSE_OUT"] = np.asarray(car_pose_actions).tolist()

    return result
