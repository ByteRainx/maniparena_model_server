"""Base policy class for the x2robot_client websocket server."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ModelPolicy(ABC):
    """Implements `infer()` as: convert_input -> run_inference -> convert_output."""

    def __init__(
        self,
        checkpoint_path: str,
        control_mode: str,
        action_horizon: int,
        device: str = "cuda:0",
    ):
        """Initialize and call `load_model()` once."""
        self.checkpoint_path = checkpoint_path
        self.control_mode = control_mode
        self.action_horizon = action_horizon
        self.device = device

        logger.info(f"Loading model from {checkpoint_path}...")
        self.model = self.load_model(checkpoint_path, device)
        logger.info("Model loaded successfully")

    @abstractmethod
    def load_model(self, checkpoint_path: str, device: str) -> Any:
        """Load your model/checkpoint and return a callable object."""
        raise NotImplementedError

    def convert_input(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert x2robot_client obs into your model input.

        The obs may be in new nested format (state/views/instruction) or
        legacy flat format (CAMERA_LEFT / ACTION_FOLLOW1_POS / ...).
        """
        return obs

    @abstractmethod
    def convert_output(self, model_output: Any) -> Dict[str, Any]:
        """Convert model outputs into x2robot_client response dict.

        Should return lowercase keys matching the new API:
        follow1_pos, follow2_pos, head_pos, lift, velocity_decomposed, etc.
        """
        raise NotImplementedError

    @abstractmethod
    def run_inference(self, model_input: Dict[str, Any]) -> Any:
        """Run your model on `model_input` and return any raw output."""
        raise NotImplementedError

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Standard inference pipeline."""
        try:
            model_input = self.convert_input(obs)
            model_output = self.run_inference(model_input)
            return self.convert_output(model_output)
        except Exception as e:
            logger.error(f"Error in inference: {e}", exc_info=True)
            raise

    def reset(self):
        """Reset policy state if needed"""
        if hasattr(self.model, "reset"):
            self.model.reset()

    @property
    def metadata(self) -> Dict[str, Any]:
        """Metadata sent to client after connection."""
        return {
            "control_mode": self.control_mode,
            "action_horizon": self.action_horizon,
            "state_dim": 14,
            "state_dim_per_arm": 7,
            "protocol_version": "2.0",
        }
