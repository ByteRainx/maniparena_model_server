"""Example: PyTorch policy for Desktop (14D) tasks.

Shows how to implement MyPolicy with a PyTorch model.
Adapt this to your own architecture.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from policy_base import ModelPolicy
from utils import (
    convert_observation_to_model_input,
    convert_output_desktop,
)

DEFAULT_CHECKPOINT_PATH = "/path/to/your/checkpoint.pt"
DEFAULT_CONTROL_MODE = "end_pose"
DEFAULT_ACTION_HORIZON = 50
DEFAULT_DEVICE = "cuda:0"


def build_model() -> Any:
    """Replace with your model architecture."""
    raise NotImplementedError


class TorchDesktopPolicy(ModelPolicy):
    """Desktop (14D) policy using PyTorch."""

    def load_model(self, checkpoint_path: str, device: str) -> Any:
        import torch

        model = build_model()
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device).eval()
        return model

    def convert_input(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return convert_observation_to_model_input(
            obs, self.control_mode, decode_images=False,
        )

    def run_inference(self, model_input: Dict[str, Any]) -> Any:
        import torch

        with torch.no_grad():
            out = self.model(model_input)
        return out.detach().cpu().numpy()

    def convert_output(self, model_output: Any) -> Dict[str, Any]:
        actions = np.asarray(model_output, dtype=np.float32)  # (T, 14)
        return convert_output_desktop(
            actions, self.control_mode, self.action_horizon,
        )
