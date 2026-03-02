"""Example: PyTorch policy for CX001 Mobile Manipulation (20D) tasks.

Shows how to implement MyPolicy for the CX001 mobile manipulator
which requires head, lift, and chassis outputs in addition to arm actions.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from policy_base import ModelPolicy
from utils import (
    convert_observation_to_model_input,
    convert_output_cx001,
)

DEFAULT_CHECKPOINT_PATH = "/path/to/your/checkpoint.pt"
DEFAULT_CONTROL_MODE = "end_pose"
DEFAULT_ACTION_HORIZON = 50
DEFAULT_DEVICE = "cuda:0"


def build_model() -> Any:
    """Replace with your model architecture."""
    raise NotImplementedError


class TorchCX001Policy(ModelPolicy):
    """CX001 Mobile Manipulation (20D) policy using PyTorch."""

    def load_model(self, checkpoint_path: str, device: str) -> Any:
        import torch

        model = build_model()
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device).eval()
        return model

    def convert_input(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        model_input = convert_observation_to_model_input(
            obs, self.control_mode, decode_images=False,
        )
        # CX001 extra state fields are available as:
        #   model_input["state/head_pos"]  -> ndarray (2,)
        #   model_input["state/lift"]      -> ndarray (1,)
        #   model_input["state/car_pose"]  -> ndarray (3,)
        return model_input

    def run_inference(self, model_input: Dict[str, Any]) -> Any:
        import torch

        with torch.no_grad():
            out = self.model(model_input)
        # Expected: dict with "arms" (T, 14), "head" (T, 2),
        #           "lift" (T, 1), "car_pose" (T, 3)
        return {k: v.detach().cpu().numpy() for k, v in out.items()}

    def convert_output(self, model_output: Any) -> Dict[str, Any]:
        arm_actions = np.asarray(model_output["arms"], dtype=np.float32)
        head = np.asarray(model_output["head"], dtype=np.float32)
        lift = np.asarray(model_output["lift"], dtype=np.float32)
        car = np.asarray(model_output["car_pose"], dtype=np.float32)

        return convert_output_cx001(
            arm_actions, self.control_mode, self.action_horizon,
            head=head, lift=lift, car_pose=car,
        )
