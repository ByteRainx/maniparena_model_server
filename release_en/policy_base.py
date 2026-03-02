"""Abstract base class for model policies — do NOT modify this file."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ModelPolicy(ABC):
    """Pipeline: convert_input -> run_inference -> convert_output."""

    def __init__(
        self,
        checkpoint_path: str,
        control_mode: str,
        action_horizon: int,
        device: str = "cuda:0",
    ):
        self.checkpoint_path = checkpoint_path
        self.control_mode = control_mode
        self.action_horizon = action_horizon
        self.device = device

        logger.info("Loading model from %s ...", checkpoint_path)
        self.model = self.load_model(checkpoint_path, device)
        logger.info("Model loaded.")

    # ── Required overrides ────────────────────────────────────

    @abstractmethod
    def load_model(self, checkpoint_path: str, device: str) -> Any:
        """Load checkpoint and return a callable model object."""

    @abstractmethod
    def run_inference(self, model_input: Dict[str, Any]) -> Any:
        """Run your model on preprocessed input and return raw output."""

    @abstractmethod
    def convert_output(self, model_output: Any) -> Dict[str, Any]:
        """Convert raw model output to the robot client action dict.

        CRITICAL:
        - All trajectory values MUST be Python lists (.tolist()), NOT numpy.
        - Desktop (14D): lowercase keys  — follow1_pos, follow2_pos
        - CX001  (20D): UPPERCASE keys — FOLLOW1_POS, FOLLOW2_POS,
                         HEAD_POS, LIFT_OUT, CAR_POSE_OUT
        """

    # ── Optional overrides ────────────────────────────────────

    def convert_input(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess raw observation. Default: pass-through."""
        return obs

    def reset(self):
        """Reset internal state between episodes."""
        if hasattr(self.model, "reset"):
            self.model.reset()

    @property
    def metadata(self) -> Dict[str, Any]:
        """Metadata sent to the client on connect."""
        return {
            "control_mode": self.control_mode,
            "action_horizon": self.action_horizon,
            "state_dim": 14,
            "state_dim_per_arm": 7,
            "protocol_version": "2.0",
        }

    # ── Internal pipeline (do not override) ───────────────────

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Full inference pipeline — called by the server."""
        try:
            model_input = self.convert_input(obs)
            model_output = self.run_inference(model_input)
            return self.convert_output(model_output)
        except Exception as e:
            logger.error("Inference error: %s", e, exc_info=True)
            raise
