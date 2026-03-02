"""Your model implementation — EDIT THIS FILE.

Implement the methods below. See README.md for the full protocol spec.
"""

from __future__ import annotations

from typing import Any, Dict

from policy_base import ModelPolicy

# ── Configuration ──────────────────────────────────────────────

DEFAULT_CHECKPOINT_PATH = "/path/to/your/checkpoint"
DEFAULT_CONTROL_MODE = "end_pose"  # "end_pose" or "joints"
DEFAULT_ACTION_HORIZON = 50
DEFAULT_DEVICE = "cuda:0"


# ── Policy ─────────────────────────────────────────────────────

class MyPolicy(ModelPolicy):

    def load_model(self, checkpoint_path: str, device: str) -> Any:
        """Load your model checkpoint. Return a callable model object."""
        raise NotImplementedError("Implement load_model()")

    def convert_input(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess raw observation -> model input.

        The helper handles both Desktop (nested) and CX001 (flat) formats:
        >>> from utils import convert_observation_to_model_input
        >>> return convert_observation_to_model_input(obs, self.control_mode)
        """
        from utils import convert_observation_to_model_input

        return convert_observation_to_model_input(
            obs, self.control_mode, decode_images=False,
        )

    def run_inference(self, model_input: Dict[str, Any]) -> Any:
        """Run your model and return raw output (e.g. numpy array)."""
        raise NotImplementedError("Implement run_inference()")

    def convert_output(self, model_output: Any) -> Dict[str, Any]:
        """Convert model output -> action dict for the robot client.

        CRITICAL: values must be Python lists, NOT numpy arrays.
        Use .tolist() on every numpy array before returning.

        Desktop (14D) — lowercase keys:
            {"follow1_pos": actions[:, :7].tolist(),
             "follow2_pos": actions[:, 7:14].tolist()}

        CX001 (20D) — UPPERCASE keys:
            {"FOLLOW1_POS": ..., "FOLLOW2_POS": ...,
             "HEAD_POS": ..., "LIFT_OUT": ..., "CAR_POSE_OUT": ...}

        Or use the provided helpers:
            from utils import convert_output_desktop   # 14D
            from utils import convert_output_cx001     # 20D
        """
        raise NotImplementedError("Implement convert_output()")
