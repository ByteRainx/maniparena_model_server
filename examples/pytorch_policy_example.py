from __future__ import annotations

from typing import Any, Dict

import numpy as np

from policy_base import ModelPolicy
from utils import convert_model_output_to_x2robot_format, convert_observation_to_model_input


def build_model() -> Any:
    raise NotImplementedError


class TorchPolicy(ModelPolicy):
    def load_model(self, checkpoint_path: str, device: str) -> Any:
        import torch

        model = build_model()
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model.to(device).eval()
        return model

    def convert_input(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return convert_observation_to_model_input(obs, self.control_mode, decode_images=False)

    def run_inference(self, model_input: Dict[str, Any]) -> Any:
        import torch

        with torch.no_grad():
            out = self.model(model_input)
        return out.detach().cpu().numpy()

    def convert_output(self, model_output: Any) -> Dict[str, Any]:
        actions = np.asarray(model_output, dtype=np.float32)
        return convert_model_output_to_x2robot_format(actions, self.control_mode, self.action_horizon)
