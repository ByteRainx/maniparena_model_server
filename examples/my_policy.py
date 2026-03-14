"""Policy template — edit this file to plug in your model.

Start the server with:
    python serve.py --checkpoint /path/to/ckpt --port 8000
"""

from __future__ import annotations

from typing import Any, Dict

from maniparena.policy import ModelPolicy
from maniparena.utils import convert_model_output_to_action, convert_observation_to_model_input


class MyPolicy(ModelPolicy):

    def load_model(self, checkpoint_path: str, device: str) -> Any:
        # TODO: load your model here and return it
        raise NotImplementedError("Implement load_model()")

    def run_inference(self, model_input: Dict[str, Any]) -> Any:
        # TODO: run forward pass, return np.ndarray of shape (action_horizon, 14)
        raise NotImplementedError("Implement run_inference()")

    def convert_input(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return convert_observation_to_model_input(obs, self.control_mode, decode_images=False)

    def convert_output(self, model_output: Any) -> Dict[str, Any]:
        # WARNING: output values MUST be Python lists, NOT numpy arrays!
        # The robot client does `[current_pos] + actions` which silently
        # broadcasts if actions is numpy, corrupting the trajectory.
        # convert_model_output_to_action() handles .tolist() internally.
        # If you write custom logic, always call .tolist() on every value.
        return convert_model_output_to_action(model_output, self.control_mode, self.action_horizon)
