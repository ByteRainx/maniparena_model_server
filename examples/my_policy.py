"""Policy template — edit this file to plug in your model.

Start the server with:
    PYTHONPATH=examples python -m maniparena.launch --checkpoint /path/to/ckpt
"""

from __future__ import annotations

from typing import Any, Dict

from maniparena.policy import ModelPolicy
from maniparena.utils import convert_model_output_to_action, convert_observation_to_model_input


class MyPolicy(ModelPolicy):

    def load_model(self, checkpoint_path: str, device: str) -> Any:
        # TODO: load your model here
        raise NotImplementedError("Implement load_model()")

    def run_inference(self, model_input: Dict[str, Any]) -> Any:
        # TODO: run forward pass, return (action_horizon, 14) array
        raise NotImplementedError("Implement run_inference()")

    def convert_input(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return convert_observation_to_model_input(obs, self.control_mode, decode_images=False)

    def convert_output(self, model_output: Any) -> Dict[str, Any]:
        return convert_model_output_to_action(model_output, self.control_mode, self.action_horizon)
