from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class OpenPIAdapterMetadata:
    model_type: str
    control_mode: str
    action_horizon: int
    state_dim: int = 14
    state_dim_per_arm: int = 7
    protocol_version: str = "2.0"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "model_type": self.model_type,
            "control_mode": self.control_mode,
            "action_horizon": self.action_horizon,
            "state_dim": self.state_dim,
            "state_dim_per_arm": self.state_dim_per_arm,
        }


class OpenPIX2RobotPolicy:
    """Adapter for OpenPI-trained models.

    Accepts both new nested format (state/views/instruction) and legacy flat
    format (CAMERA_LEFT / ACTION_FOLLOW1_POS / ...).
    Returns new lowercase format (follow1_pos, head_pos, lift, etc.).
    """

    def __init__(
        self,
        *,
        config_name: str,
        checkpoint_dir: str,
        default_prompt: Optional[str] = None,
        pytorch_device: str = "cuda",
    ) -> None:
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config
        from openpi.training.config import LeRobotX2RobotJointsDataConfig

        self._default_prompt = default_prompt

        train_config = _config.get_config(config_name)
        self._is_joints_model = isinstance(train_config.data, LeRobotX2RobotJointsDataConfig)

        self._policy = _policy_config.create_trained_policy(
            train_config,
            checkpoint_dir,
            default_prompt=default_prompt,
            pytorch_device=pytorch_device,
        )

        self._action_horizon = int(getattr(train_config.model, "action_horizon", 50))
        self._control_mode = "joints" if self._is_joints_model else "end_pose"
        self._metadata = OpenPIAdapterMetadata(
            model_type="openpi_x2robot_joints" if self._is_joints_model else "openpi_x2robot_ee",
            control_mode=self._control_mode,
            action_horizon=self._action_horizon,
        )

    @property
    def metadata(self) -> Dict[str, Any]:
        base = {}
        try:
            base_meta = getattr(self._policy, "metadata", None)
            if isinstance(base_meta, dict):
                base = dict(base_meta)
        except Exception:
            base = {}
        return {**base, **self._metadata.as_dict()}

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    @staticmethod
    def _to_1d_float32(x: Any) -> np.ndarray:
        if x is None:
            return np.array([], dtype=np.float32)
        if isinstance(x, np.ndarray):
            return x.astype(np.float32, copy=False).reshape(-1)
        if isinstance(x, dict):
            if "data" in x and "shape" in x:
                try:
                    return np.array(x["data"], dtype=np.float32).reshape(x["shape"]).reshape(-1)
                except Exception:
                    pass
            try:
                return np.array(list(x.values()), dtype=np.float32).reshape(-1)
            except Exception:
                return np.array([], dtype=np.float32)
        if isinstance(x, (list, tuple)):
            try:
                return np.array(x, dtype=np.float32).reshape(-1)
            except Exception:
                return np.array([], dtype=np.float32)
        return np.array([x], dtype=np.float32).reshape(-1)

    @staticmethod
    def _normalize_7d(pos: np.ndarray, *, joints: bool) -> np.ndarray:
        if pos.size == 7:
            return pos
        if joints and pos.size == 8:
            return np.concatenate([pos[:6], pos[-1:]], axis=0)
        raise ValueError(f"Expected 7D{' or 8D' if joints else ''}, got {pos.size}D")

    @staticmethod
    def _is_new_format(obs: Dict[str, Any]) -> bool:
        return "state" in obs and isinstance(obs["state"], dict)

    @staticmethod
    def _extract_instruction(obs: Dict[str, Any]) -> Optional[str]:
        raw = obs.get("instruction", None)
        if raw is None:
            raw = obs.get("prompt", None)
        if isinstance(raw, np.ndarray):
            return str(raw.flat[0]) if raw.size > 0 else None
        return str(raw) if raw else None

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        from utils import decode_jpeg_any

        new_fmt = self._is_new_format(obs)

        # --- 提取双臂状态 ---
        if new_fmt:
            state = obs["state"]
            if self._is_joints_model:
                follow1 = self._to_1d_float32(
                    state.get("follow1_joints", state.get("follow1_pos"))
                )
                follow2 = self._to_1d_float32(
                    state.get("follow2_joints", state.get("follow2_pos"))
                )
            else:
                follow1 = self._to_1d_float32(state.get("follow1_pos"))
                follow2 = self._to_1d_float32(state.get("follow2_pos"))
        else:
            follow1 = self._to_1d_float32(obs.get("ACTION_FOLLOW1_POS"))
            follow2 = self._to_1d_float32(obs.get("ACTION_FOLLOW2_POS"))

        follow1 = self._normalize_7d(follow1, joints=self._is_joints_model)
        follow2 = self._normalize_7d(follow2, joints=self._is_joints_model)

        state14 = np.concatenate(
            [follow1[:6], follow1[6:7], follow2[:6], follow2[6:7]], axis=0
        ).astype(np.float32)

        openpi_obs: Dict[str, Any] = {"observation/state": state14}

        # --- 提取图像 ---
        if new_fmt:
            views = obs.get("views", {})
            left = decode_jpeg_any(views.get("camera_left"), name="camera_left")
            front = decode_jpeg_any(views.get("camera_front"), name="camera_front")
            right = decode_jpeg_any(views.get("camera_right"), name="camera_right")
        else:
            left = decode_jpeg_any(obs.get("CAMERA_LEFT"), name="CAMERA_LEFT")
            front = decode_jpeg_any(obs.get("CAMERA_FRONT"), name="CAMERA_FRONT")
            right = decode_jpeg_any(obs.get("CAMERA_RIGHT"), name="CAMERA_RIGHT")

        if left is not None:
            openpi_obs["observation/left_arm_image"] = left
        if front is not None:
            openpi_obs["observation/head_image"] = front
        if right is not None:
            openpi_obs["observation/right_arm_image"] = right

        # --- 提取指令 ---
        prompt = self._extract_instruction(obs) or self._default_prompt
        if prompt:
            openpi_obs["prompt"] = prompt

        # --- 运行推理 ---
        result = self._policy.infer(openpi_obs)
        actions = np.asarray(result["actions"], dtype=np.float32)

        left_actions = np.concatenate([actions[:, :6], actions[:, 6:7]], axis=1)
        right_actions = np.concatenate([actions[:, 7:13], actions[:, 13:14]], axis=1)

        # --- 构造输出（新格式：小写 key）---
        if self._control_mode == "joints":
            output = {
                "follow1_joints": left_actions.tolist(),
                "follow2_joints": right_actions.tolist(),
                "follow1_pos": left_actions.tolist(),
                "follow2_pos": right_actions.tolist(),
            }
        else:
            output = {
                "follow1_pos": left_actions.tolist(),
                "follow2_pos": right_actions.tolist(),
            }

        return output
