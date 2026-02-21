"""Model policy entrypoint (edit this file).

Keep this file small. Put runnable examples in `examples/`.
"""

from __future__ import annotations

from typing import Any, Dict

from policy_base import ModelPolicy


# ============ 配置部分 - 参赛者需要修改 ============

DEFAULT_CHECKPOINT_PATH = "/path/to/your/checkpoint"
DEFAULT_CONTROL_MODE = "joints"  # 或 "end_pose"
DEFAULT_ACTION_HORIZON = 50
DEFAULT_DEVICE = "cuda:0"


# ============ Policy实现 - 参赛者需要实现以下方法 ============

class MyPolicy(ModelPolicy):
    """Implement `load_model()` + `run_inference()` for your model."""

    def load_model(self, checkpoint_path: str, device: str) -> Any:
        raise NotImplementedError

    def convert_input(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert x2robot_client obs -> model input.

        Automatically handles both new nested format (state/views/instruction)
        and legacy flat format (CAMERA_LEFT / ACTION_FOLLOW1_POS / ...).
        """
        from utils import convert_observation_to_model_input

        return convert_observation_to_model_input(obs, self.control_mode, decode_images=False)

    def convert_output(self, model_output: Any) -> Dict[str, Any]:
        """Convert model output -> x2robot_client response (new lowercase format).

        model_output is expected to be np.ndarray of shape (action_horizon, 14).
        Override this method if your model outputs additional fields like
        head_pos, lift, velocity_decomposed, etc.
        """
        from utils import convert_model_output_to_x2robot_format

        return convert_model_output_to_x2robot_format(
            model_output, self.control_mode, self.action_horizon
        )

    def run_inference(self, model_input: Dict[str, Any]) -> Any:
        raise NotImplementedError
