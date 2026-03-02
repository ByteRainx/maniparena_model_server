"""I/O conversion utilities for ManipArena model server.

Output helpers (all values returned as Python lists via .tolist()):
  convert_output_desktop  — 14D Desktop format (lowercase keys)
  convert_output_cx001    — 20D CX001 format (UPPERCASE keys)

Input helper:
  convert_observation_to_model_input — parses Desktop and CX001 observations
"""

import base64
import logging
from typing import Any, Dict, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# ── Low-level helpers ──────────────────────────────────────────


def to_numpy_1d(x: Any, *, name: str = "unknown") -> np.ndarray:
    """Convert various formats to a flat float32 numpy array.

    Supports: None, ndarray, list/tuple, msgpack-numpy dict {data, shape}.
    """
    if x is None:
        return np.zeros((0,), dtype=np.float32)
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False).reshape(-1)
    if isinstance(x, dict) and "data" in x and "shape" in x:
        return np.array(x["data"], dtype=np.float32).reshape(tuple(x["shape"])).reshape(-1)
    if isinstance(x, (list, tuple)):
        return np.array(x, dtype=np.float32).reshape(-1)
    raise TypeError(f"{name}: unsupported type {type(x)}")


def decode_jpeg(
    v: Union[str, bytes, None], *, name: str = "image",
) -> Optional[np.ndarray]:
    """Decode base64-JPEG or raw-bytes JPEG into an RGB numpy array."""
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
        import cv2
    except ImportError as e:
        raise ImportError("opencv-python required: pip install opencv-python") from e

    buf = np.frombuffer(v, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to decode JPEG: {name}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def normalize_joints_7d(joints: np.ndarray, control_mode: str) -> np.ndarray:
    """Normalize joint data to 7D (6 joints + 1 gripper).

    8D input (7 joints + 1 gripper) is trimmed to first 6 joints + gripper.
    """
    if control_mode != "joints":
        return joints
    if joints.size == 8:
        return np.concatenate([joints[:6], joints[-1:]])
    if joints.size == 7:
        return joints
    raise ValueError(f"Expected 7D or 8D joints, got {joints.size}D")


# ── Format detection ───────────────────────────────────────────


def _is_nested(obs: Dict[str, Any]) -> bool:
    return "state" in obs and isinstance(obs["state"], dict)


def _extract_instruction(obs: Dict[str, Any]) -> str:
    raw = obs.get("instruction") or obs.get("prompt", "")
    if isinstance(raw, np.ndarray):
        return str(raw.flat[0]) if raw.size > 0 else ""
    return str(raw) if raw else ""


# ── Observation -> Model Input ─────────────────────────────────


def convert_observation_to_model_input(
    obs: Dict[str, Any],
    control_mode: str,
    decode_images: bool = True,
) -> Dict[str, Any]:
    """Parse raw x2robot_client observation into a model-friendly dict.

    Auto-detects format:
      - Desktop (nested state/views/instruction)
      - CX001 ROS1 (flat, lowercase state keys, UPPERCASE camera keys)
      - CX001 ROS2 (flat, ACTION_* prefix, UPPERCASE keys)

    Returns:
        {
            "left": RGB ndarray | raw,   "front": ...,   "right": ...,
            "state": ndarray (14,),      # [left_7d, right_7d]
            "instruction": str,
            "state/head_pos": ...,       # extra fields when available
            "state/lift": ...,
            "state/car_pose": ...,
        }
    """
    nested = _is_nested(obs)

    # ── Images ──
    images: Dict[str, Any] = {}
    if nested:
        views = obs.get("views", {})
        for src, dst in [("camera_left", "left"), ("camera_front", "front"), ("camera_right", "right")]:
            raw = views.get(src)
            if raw is not None:
                images[dst] = decode_jpeg(raw, name=src) if decode_images else raw
    else:
        for key in ("CAMERA_LEFT", "CAMERA_FRONT", "CAMERA_RIGHT"):
            raw = obs.get(key)
            if raw is not None:
                dst = key.lower().replace("camera_", "")
                images[dst] = decode_jpeg(raw, name=key) if decode_images else raw

    # ── Arm state -> 14D ──
    if nested:
        sd = obs["state"]
        if control_mode == "joints":
            f1 = to_numpy_1d(sd.get("follow1_joints", sd.get("follow1_pos")), name="follow1")
            f2 = to_numpy_1d(sd.get("follow2_joints", sd.get("follow2_pos")), name="follow2")
        else:
            f1 = to_numpy_1d(sd.get("follow1_pos"), name="follow1_pos")
            f2 = to_numpy_1d(sd.get("follow2_pos"), name="follow2_pos")
    else:
        f1 = to_numpy_1d(
            obs.get("ACTION_FOLLOW1_POS") or obs.get("follow1_pos"),
            name="follow1",
        )
        f2 = to_numpy_1d(
            obs.get("ACTION_FOLLOW2_POS") or obs.get("follow2_pos"),
            name="follow2",
        )

    if control_mode == "joints":
        f1 = normalize_joints_7d(f1, control_mode)
        f2 = normalize_joints_7d(f2, control_mode)

    if f1.size < 7 or f2.size < 7:
        raise ValueError(f"Expected >=7D per arm, got left={f1.size}, right={f2.size}")

    state14 = np.concatenate([f1[:7], f2[:7]]).astype(np.float32)
    instruction = _extract_instruction(obs)

    result: Dict[str, Any] = {**images, "state": state14, "instruction": instruction}

    # ── Extra state fields (pass-through) ──
    _EXTRA_KEYS = [
        "follow1_pos", "follow2_pos",
        "follow1_joints", "follow2_joints",
        "follow1_joints_cur", "follow2_joints_cur",
        "follow1_joints_dev", "follow2_joints_dev",
        "head_pos", "lift", "car_pose",
        "velocity_decomposed", "velocity_decomposed_odom",
    ]
    if nested:
        sd = obs["state"]
        for k in _EXTRA_KEYS:
            v = sd.get(k)
            if v is not None:
                result[f"state/{k}"] = to_numpy_1d(v, name=k)
    else:
        for k in _EXTRA_KEYS:
            v = obs.get(k) or obs.get(k.upper()) or obs.get(f"ACTION_{k.upper()}")
            if v is not None:
                result[f"state/{k}"] = to_numpy_1d(v, name=k)

    return result


# ── Model Output -> Desktop Response (14D, lowercase) ─────────


def convert_output_desktop(
    actions: np.ndarray,
    control_mode: str,
    action_horizon: int,
    *,
    model_output_text: str = "",
) -> Dict[str, Any]:
    """Convert (T, 14) actions to Desktop client format.

    Returns lowercase keys. All values are Python lists via .tolist().
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(f"Expected (T, 14), got {actions.shape}")

    if actions.shape[0] != action_horizon:
        logger.warning("Horizon mismatch: expected %d, got %d", action_horizon, actions.shape[0])

    left = np.concatenate([actions[:, :6], actions[:, 6:7]], axis=1)
    right = np.concatenate([actions[:, 7:13], actions[:, 13:14]], axis=1)

    if control_mode == "joints":
        result: Dict[str, Any] = {
            "follow1_joints": left.tolist(),
            "follow2_joints": right.tolist(),
            "follow1_pos": left.tolist(),
            "follow2_pos": right.tolist(),
        }
    else:
        result = {
            "follow1_pos": left.tolist(),
            "follow2_pos": right.tolist(),
        }

    if model_output_text:
        result["model_output_text"] = model_output_text
    return result


# ── Model Output -> CX001 Response (20D, UPPERCASE) ───────────


def convert_output_cx001(
    actions: np.ndarray,
    control_mode: str,
    action_horizon: int,
    *,
    head: Optional[np.ndarray] = None,
    lift: Optional[np.ndarray] = None,
    car_pose: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Convert (T, 14) arm actions + optional extras to CX001 format.

    Returns UPPERCASE keys. All values are Python lists via .tolist().

    CX001 client expects:
      FOLLOW1_POS / FOLLOW2_POS  — arm trajectories (T, 7)
      HEAD_POS                   — head (T, 2) [yaw, pitch]
      LIFT_OUT                   — lift (T, 1)
      CAR_POSE_OUT               — chassis (T, 3) [x, y, theta]
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(f"Expected (T, 14), got {actions.shape}")

    left = np.concatenate([actions[:, :6], actions[:, 6:7]], axis=1)
    right = np.concatenate([actions[:, 7:13], actions[:, 13:14]], axis=1)

    if control_mode == "joints":
        result: Dict[str, Any] = {
            "FOLLOW1_JOINTS": left.tolist(),
            "FOLLOW2_JOINTS": right.tolist(),
            "FOLLOW1_POS": [],
            "FOLLOW2_POS": [],
        }
    else:
        result = {
            "FOLLOW1_POS": left.tolist(),
            "FOLLOW2_POS": right.tolist(),
        }

    if head is not None:
        result["HEAD_POS"] = np.asarray(head).tolist()
    if lift is not None:
        result["LIFT_OUT"] = np.asarray(lift).tolist()
    if car_pose is not None:
        result["CAR_POSE_OUT"] = np.asarray(car_pose).tolist()

    return result
