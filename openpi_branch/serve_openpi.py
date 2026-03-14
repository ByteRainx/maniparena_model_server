#!/usr/bin/env python3
"""OpenPI X2Robot inference server — fully compatible with x2robot_client.

Supports:
  - x2robot EE configs (14D: 6+1+6+1)
  - x2robot Joints configs (14D: 6+1+6+1, delta joints)
  - MobileManipulation configs (20D: 14D arm + 2 head + 1 lift + 3 chassis)
  - New nested client format (state/views/instruction)
  - Legacy flat client format (CAMERA_LEFT / ACTION_FOLLOW1_POS)

Usage:
    # EE mode
    python openpi_branch/serve_openpi.py \
        --config pi0_x2robot_pick_banana_merged_ee \
        --checkpoint-dir checkpoints/pi0_x2robot_pick_banana_merged_ee/exp/10000

    # Joints mode
    python openpi_branch/serve_openpi.py \
        --config pi05_pick_banana_joints \
        --checkpoint-dir checkpoints/pi05_pick_banana_joints/exp/10000

    # Mobile Manipulation mode (20D)
    python openpi_branch/serve_openpi.py \
        --config pi05_put_clothes_in_hamper_mm \
        --checkpoint-dir checkpoints/pi05_put_clothes_in_hamper_mm/exp/10000
"""

from __future__ import annotations

import base64
import dataclasses
import logging
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import msgpack
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — make openpi importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_MANIPARENA_ROOT = _SCRIPT_DIR.parent
_OPENPI_ROOT = _MANIPARENA_ROOT.parent / "openpi"

for _p in (_MANIPARENA_ROOT, _OPENPI_ROOT / "src", _OPENPI_ROOT):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

logger = logging.getLogger(__name__)


# =====================================================================
# Cross-compatible msgpack + numpy serialization
# =====================================================================
# x2robot_client uses the *msgpack-numpy* pip package (m.patch()), which
# encodes numpy arrays with a ``{b'nd': True, ...}`` dict.
# openpi_client.msgpack_numpy uses ``{b'__ndarray__': True, ...}``.
# The helpers below handle BOTH formats transparently.
# =====================================================================

def _has_key(d: dict, key_str: str) -> bool:
    return key_str.encode() in d or key_str in d


def _get_val(d: dict, key_str: str):
    return d.get(key_str.encode(), d.get(key_str))


def _pack_default(obj):
    """msgpack default hook — encode numpy types."""
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            return obj.tolist()
        dtype_str = obj.dtype.str
        return {
            b"nd": True,
            b"type": dtype_str.encode() if isinstance(dtype_str, str) else dtype_str,
            b"kind": b"",
            b"shape": obj.shape,
            b"data": obj.tobytes(),
        }
    if isinstance(obj, np.generic):
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj.item()
    return obj


def _unpack_hook(obj: dict):
    """msgpack object_hook — decode numpy arrays from either format."""
    # --- msgpack-numpy (pip) format ---
    if _has_key(obj, "nd"):
        type_val = _get_val(obj, "type")
        if isinstance(type_val, bytes):
            type_val = type_val.decode()

        if isinstance(type_val, list):
            is_object = any(
                (isinstance(item, (list, tuple)) and len(item) >= 2 and "|O" in str(item[1]))
                for item in type_val
            ) if type_val else False
            if is_object:
                shape = tuple(_get_val(obj, "shape"))
                data = _get_val(obj, "data")
                arr = np.empty(shape, dtype=object)
                flat_data = data if isinstance(data, list) else [data]
                for i, item in enumerate(flat_data):
                    if isinstance(item, bytes):
                        item = item.decode("utf-8", errors="replace")
                    arr.flat[i] = item
                return arr
            else:
                dtype = np.dtype([(str(f), str(t)) for f, t in type_val])
                shape = tuple(_get_val(obj, "shape"))
                data = _get_val(obj, "data")
                return np.ndarray(buffer=data, dtype=dtype, shape=shape).copy()

        dtype = np.dtype(type_val)
        shape = tuple(_get_val(obj, "shape"))
        data = _get_val(obj, "data")
        return np.ndarray(buffer=data, dtype=dtype, shape=shape).copy()

    # --- openpi_client format ---
    if _has_key(obj, "__ndarray__"):
        dtype_val = _get_val(obj, "dtype")
        if isinstance(dtype_val, bytes):
            dtype_val = dtype_val.decode()
        dtype = np.dtype(dtype_val)
        shape = tuple(_get_val(obj, "shape"))
        data = _get_val(obj, "data")
        return np.ndarray(buffer=data, dtype=dtype, shape=shape).copy()

    if _has_key(obj, "__npgeneric__"):
        dtype_val = _get_val(obj, "dtype")
        if isinstance(dtype_val, bytes):
            dtype_val = dtype_val.decode()
        return np.dtype(dtype_val).type(_get_val(obj, "data"))

    return obj


def packb(obj) -> bytes:
    return msgpack.packb(obj, default=_pack_default)


def unpackb(data: bytes):
    return msgpack.unpackb(data, object_hook=_unpack_hook, raw=False, strict_map_key=False)


# =====================================================================
# Image utilities
# =====================================================================

def decode_base64_image(v, key_name: str = "unknown") -> np.ndarray:
    """Decode a base64-encoded JPEG to an RGB uint8 numpy array."""
    if isinstance(v, str):
        img_bytes = base64.b64decode(v)
    elif isinstance(v, bytes):
        img_bytes = v
    else:
        raise ValueError(f"Unexpected image type {type(v).__name__} for key '{key_name}'")

    buf = np.frombuffer(img_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"cv2.imdecode returned None for key '{key_name}' (len={len(img_bytes)})")

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# =====================================================================
# Inference Logger
# =====================================================================

_EE_DIM_NAMES = [
    "left_x", "left_y", "left_z", "left_roll", "left_pitch", "left_yaw", "left_gripper",
    "right_x", "right_y", "right_z", "right_roll", "right_pitch", "right_yaw", "right_gripper",
]
_MM_DIM_NAMES = _EE_DIM_NAMES + [
    "head_yaw", "head_pitch", "lift", "chassis_vx", "chassis_vy", "chassis_omega",
]


class InferenceLogger:
    """Record state & action at every inference step and produce plots on demand."""

    def __init__(self, log_dir: str, mode: str = "ee"):
        self.log_dir = log_dir
        self.mode = mode
        os.makedirs(log_dir, exist_ok=True)

        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.action_chunks: list[np.ndarray] = []
        self.prompts: list[str] = []
        self.timestamps: list[float] = []
        self._step = 0

    def record(self, state: np.ndarray, action_chunk: np.ndarray, prompt: str = ""):
        self.states.append(state.copy())
        self.action_chunks.append(action_chunk.copy())
        self.actions.append(action_chunk[0].copy())
        self.prompts.append(prompt)
        self.timestamps.append(time.time())
        self._step += 1
        if self._step % 50 == 0:
            logger.info("[InferenceLogger] recorded %d steps", self._step)

    def flush_plot(self, tag: str = "inference") -> Optional[str]:
        if len(self.states) < 2:
            logger.warning("[InferenceLogger] < 2 steps, skip plot")
            return None

        states = np.array(self.states)
        actions = np.array(self.actions)
        dim = states.shape[1]
        dim_names = _MM_DIM_NAMES[:dim] if dim > 14 else _EE_DIM_NAMES[:dim]

        npz_path = os.path.join(self.log_dir, f"{tag}.npz")
        np.savez(
            npz_path,
            states=states,
            actions=actions,
            action_chunks=np.array(self.action_chunks, dtype=object),
            dim_names=dim_names,
            timestamps=np.array(self.timestamps),
            prompts=np.array(self.prompts, dtype=object),
        )
        logger.info("[InferenceLogger] Saved data -> %s", npz_path)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            n_rows = dim
            fig, axes = plt.subplots(n_rows, 2, figsize=(20, 2.8 * n_rows), squeeze=False)
            fig.suptitle(f"Inference Log: {tag}  |  steps={len(states)}  |  mode={self.mode}", fontsize=14)

            for i in range(n_rows):
                name = dim_names[i] if i < len(dim_names) else f"dim_{i}"
                axes[i, 0].plot(states[:, i], color="blue", linewidth=1, alpha=0.8)
                axes[i, 0].set_ylabel(name, fontsize=8)
                axes[i, 0].set_title("State" if i == 0 else "", fontsize=9)
                axes[i, 0].grid(True, alpha=0.3)
                axes[i, 1].plot(actions[:, i], color="orange", linewidth=1, alpha=0.8)
                axes[i, 1].set_ylabel(name, fontsize=8)
                axes[i, 1].set_title("Action (step 0)" if i == 0 else "", fontsize=9)
                axes[i, 1].grid(True, alpha=0.3)

            axes[-1, 0].set_xlabel("Time Step")
            axes[-1, 1].set_xlabel("Time Step")
            plt.tight_layout()

            png_path = os.path.join(self.log_dir, f"{tag}.png")
            plt.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("[InferenceLogger] Saved plot -> %s", png_path)
            return png_path
        except Exception as exc:
            logger.exception("[InferenceLogger] Plot failed, npz saved: %s", exc)
            return None

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.action_chunks.clear()
        self.prompts.clear()
        self.timestamps.clear()
        self._step = 0


# =====================================================================
# OpenPI X2Robot Serving Policy
# =====================================================================

class OpenPIServingPolicy:
    """Translates between x2robot_client wire format and OpenPI policy.

    Supports three model types:
      - EE (14D):     [left_ee(6), left_gripper(1), right_ee(6), right_gripper(1)]
      - Joints (14D): [left_joints(6), left_gripper(1), right_joints(6), right_gripper(1)]
      - MM (20D):     14D arm + [head_yaw, head_pitch, lift, vx, vy, omega]

    Observation input is the LeRobot dot-separated format expected by OpenPI's
    repack transforms, with correct key names per config type.
    """

    def __init__(
        self,
        config_name: str,
        checkpoint_dir: str,
        default_prompt: Optional[str] = None,
        pytorch_device: str = "cuda",
        inference_logger: Optional[InferenceLogger] = None,
        action_start_ratio: float = 0.0,
        action_end_ratio: float = 1.0,
    ):
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config
        from openpi.training.config import LeRobotX2RobotJointsDataConfig

        self.default_prompt = default_prompt
        self.inference_logger = inference_logger
        self.action_start_ratio = action_start_ratio
        self.action_end_ratio = action_end_ratio

        print(f"[1/3] Loading config: {config_name} ...", flush=True)
        train_config = _config.get_config(config_name)

        self.is_joints_model = isinstance(train_config.data, LeRobotX2RobotJointsDataConfig)
        self.is_mm_model = False
        try:
            from openpi.training.config import LeRobotMobileManipulationDataConfig
            self.is_mm_model = isinstance(train_config.data, LeRobotMobileManipulationDataConfig)
        except ImportError:
            pass

        self.action_horizon = train_config.model.action_horizon

        # Determine image key mapping based on config type
        # x2robot configs use: observation.images.{left_arm, head, right_arm}
        # MM configs use:      observation.images.{leftImg, faceImg, rightImg}
        if self.is_mm_model:
            self._image_key_map = {
                "camera_front": "observation.images.faceImg",
                "camera_left": "observation.images.leftImg",
                "camera_right": "observation.images.rightImg",
            }
            mode_str = "mm(20D)"
        else:
            self._image_key_map = {
                "camera_front": "observation.images.head",
                "camera_left": "observation.images.left_arm",
                "camera_right": "observation.images.right_arm",
            }
            mode_str = "joints(14D)" if self.is_joints_model else "ee(14D)"

        print(f"[2/3] Loading checkpoint into {pytorch_device} (mode={mode_str}) ...", flush=True)
        print(f"      ckpt: {checkpoint_dir}", flush=True)

        t0 = time.time()
        self.policy = _policy_config.create_trained_policy(
            train_config,
            checkpoint_dir,
            default_prompt=default_prompt,
            pytorch_device=pytorch_device,
        )
        elapsed = time.time() - t0

        print(f"[3/3] Model loaded successfully! ({elapsed:.1f}s)", flush=True)
        print(f"      mode={mode_str}  action_horizon={self.action_horizon}", flush=True)
        logger.info("Policy ready — mode=%s, action_horizon=%d, load_time=%.1fs",
                     mode_str, self.action_horizon, elapsed)

    # ------------------------------------------------------------------
    # Format detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_new_format(obs: dict) -> bool:
        return "state" in obs and isinstance(obs["state"], dict)

    # ------------------------------------------------------------------
    # Input conversion
    # ------------------------------------------------------------------

    def _extract_state(self, obs: dict) -> np.ndarray:
        """Build state vector from the client's observation.

        14-D EE/Joints = [left(6), left_gripper(1), right(6), right_gripper(1)]
        20-D MM        = above 14-D + [head_yaw, head_pitch, lift, vx, vy, omega]
        """
        new_fmt = self._is_new_format(obs)

        if new_fmt:
            state_dict = obs["state"]
            if self.is_joints_model:
                follow1 = state_dict.get("follow1_joints", state_dict.get("follow1_pos"))
                follow2 = state_dict.get("follow2_joints", state_dict.get("follow2_pos"))
            else:
                follow1 = state_dict.get("follow1_pos")
                follow2 = state_dict.get("follow2_pos")
        else:
            follow1 = obs.get("ACTION_FOLLOW1_POS")
            follow2 = obs.get("ACTION_FOLLOW2_POS")

        if follow1 is None or follow2 is None:
            available = list(obs.get("state", obs).keys()) if new_fmt else list(obs.keys())
            raise ValueError(
                f"Missing arm state. Available keys: {available}. "
                f"Mode: {'mm' if self.is_mm_model else 'joints' if self.is_joints_model else 'ee'}"
            )

        follow1 = np.asarray(follow1, dtype=np.float32).flatten()
        follow2 = np.asarray(follow2, dtype=np.float32).flatten()

        parts = [follow1[:6], follow1[6:7], follow2[:6], follow2[6:7]]

        if self.is_mm_model and new_fmt:
            state_dict = obs["state"]
            head = np.asarray(state_dict.get("head_pos", [0.0, 0.0]), dtype=np.float32).flatten()[:2]
            lift = np.asarray(state_dict.get("lift", [0.0]), dtype=np.float32).flatten()[:1]
            chassis = np.asarray(
                state_dict.get("velocity_decomposed_odom",
                               state_dict.get("velocity_decomposed", [0.0, 0.0, 0.0])),
                dtype=np.float32,
            ).flatten()[:3]
            parts.extend([head, lift, chassis])

        return np.concatenate(parts).astype(np.float32)

    def _extract_images(self, obs: dict) -> Dict[str, np.ndarray]:
        """Decode camera images and map to OpenPI repack key names."""
        new_fmt = self._is_new_format(obs)
        images: Dict[str, np.ndarray] = {}

        if new_fmt:
            views = obs.get("views", {})
            raw_mapping = {
                "camera_front": views.get("camera_front"),
                "camera_left": views.get("camera_left"),
                "camera_right": views.get("camera_right"),
            }
        else:
            raw_mapping = {
                "camera_front": obs.get("CAMERA_FRONT"),
                "camera_left": obs.get("CAMERA_LEFT"),
                "camera_right": obs.get("CAMERA_RIGHT"),
            }

        for client_key, raw in raw_mapping.items():
            if raw is not None:
                openpi_key = self._image_key_map[client_key]
                images[openpi_key] = decode_base64_image(raw, client_key)

        return images

    @staticmethod
    def _recover_instruction_value(raw: Any) -> str:
        """Recover instruction string from various wire formats.

        Handles: np.ndarray, list/tuple, msgpack_numpy dict (pickled
        np.object_ array), bytes, and plain str.
        """
        if raw is None:
            return ""
        if isinstance(raw, np.ndarray):
            return str(raw.flat[0]) if raw.size > 0 else ""
        if isinstance(raw, dict):
            import pickle
            data = raw.get("data", raw.get(b"data"))
            if data is not None:
                try:
                    arr = pickle.loads(data)
                    if isinstance(arr, np.ndarray) and arr.size > 0:
                        return str(arr.flat[0])
                    return str(arr)
                except Exception:
                    pass
            return ""
        if isinstance(raw, (bytes, bytearray)):
            return raw.decode("utf-8", errors="replace")
        if isinstance(raw, (list, tuple)):
            return str(raw[0]) if raw else ""
        return str(raw) if raw else ""

    def _extract_prompt(self, obs: dict) -> str:
        for key in ("instruction", "INSTRUCTION", "prompt", "PROMPT"):
            raw = obs.get(key)
            if raw is not None:
                result = self._recover_instruction_value(raw)
                if result:
                    return result

        return self.default_prompt or ""

    # ------------------------------------------------------------------
    # Output conversion
    # ------------------------------------------------------------------

    def _convert_output(self, result: dict, state: np.ndarray) -> Dict[str, Any]:
        """Convert OpenPI output to x2robot_client format.

        1. Prepend current state as action[0] for smooth trajectory start.
        2. Apply action slicing (trim noisy tail).
        3. Split into per-component arrays.

        14-D -> follow1_pos(7) + follow2_pos(7)
        20-D -> follow1_pos(7) + follow2_pos(7) + head_pos(2) + lift(1) + velocity_decomposed(3)
        """
        actions = np.asarray(result["actions"])  # (T, D)
        D = actions.shape[1]

        state_row = state[:D].reshape(1, D).astype(actions.dtype)
        actions = np.concatenate([state_row, actions], axis=0)  # (1+T, D)

        total = actions.shape[0]
        start_idx = int(self.action_start_ratio * total)
        end_idx = int(self.action_end_ratio * total)
        end_idx = max(end_idx, start_idx + 1)
        actions = actions[start_idx:end_idx]

        left_actions = actions[:, :7].copy()
        right_actions = actions[:, 7:14].copy()

        if self.is_joints_model:
            output: Dict[str, Any] = {
                "follow1_joints": left_actions,
                "follow2_joints": right_actions,
                "follow1_pos": left_actions,
                "follow2_pos": right_actions,
            }
        else:
            output = {
                "follow1_pos": left_actions,
                "follow2_pos": right_actions,
            }

        if self.is_mm_model and D >= 20:
            output["head_pos"] = actions[:, 14:16].copy()
            output["lift"] = actions[:, 16:17].copy()
            output["velocity_decomposed"] = actions[:, 17:20].copy()

        return output

    # ------------------------------------------------------------------
    # Main inference entry point
    # ------------------------------------------------------------------

    def infer(self, obs: dict) -> Dict[str, Any]:
        """client format -> OpenPI format -> inference -> client format."""
        state = self._extract_state(obs)
        openpi_obs: dict = {"observation.state": state}
        openpi_obs.update(self._extract_images(obs))

        prompt = self._extract_prompt(obs)
        if prompt:
            openpi_obs["prompt"] = prompt

        logger.debug("OpenPI obs keys: %s | state shape: %s",
                      list(openpi_obs.keys()), state.shape)

        result = self.policy.infer(openpi_obs)

        if self.inference_logger is not None:
            raw_actions = np.asarray(result["actions"])
            self.inference_logger.record(state, raw_actions, prompt)

        return self._convert_output(result, state)

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    @property
    def metadata(self) -> dict:
        if self.is_mm_model:
            control_mode = "mobile_manipulation"
            state_dim = 20
        elif self.is_joints_model:
            control_mode = "joints"
            state_dim = 14
        else:
            control_mode = "end_pose"
            state_dim = 14
        return {
            "model_type": "openpi_x2robot",
            "control_mode": control_mode,
            "action_horizon": self.action_horizon,
            "state_dim": state_dim,
            "state_dim_per_arm": 7,
            "protocol_version": "2.0",
            "action_prepend_state": True,
            "action_start_ratio": self.action_start_ratio,
            "action_end_ratio": self.action_end_ratio,
        }


# =====================================================================
# WebSocket server
# =====================================================================

def _create_ws_handler(policy: OpenPIServingPolicy, metadata: dict):
    """Return a per-connection handler closure."""
    import websockets.sync.server
    from websockets.exceptions import ConnectionClosed

    infer_lock = threading.Lock()

    def _handle(conn: websockets.sync.server.ServerConnection) -> None:
        client = conn.remote_address
        logger.info("Client connected: %s", client)
        print(f"[CONNECT] client={client}", flush=True)

        try:
            conn.send(packb(metadata))

            while True:
                try:
                    message = conn.recv()
                    if isinstance(message, str):
                        logger.warning("Unexpected text from %s: %.100s", client, message)
                        continue

                    obs = unpackb(message)

                    try:
                        with infer_lock:
                            result = policy.infer(obs)
                    except Exception as exc:
                        logger.exception("Inference error for %s", client)
                        print(f"[INFER ERROR] client={client} err={exc}", flush=True)
                        conn.send(packb({"error": str(exc)}))
                        continue

                    conn.send(packb(result))

                except ConnectionClosed:
                    logger.info("Client disconnected: %s", client)
                    print(f"[DISCONNECT] client={client}", flush=True)
                    break
        except Exception:
            logger.exception("Unhandled error for %s", client)
        finally:
            logger.info("Handler finished for %s", client)

    return _handle


# =====================================================================
# CLI
# =====================================================================

@dataclasses.dataclass
class Args:
    """OpenPI X2Robot inference server — fully compatible with x2robot_client."""

    config: str
    """OpenPI training config name (e.g. pi0_x2robot_pick_banana_merged_ee)"""

    checkpoint_dir: str
    """Path to model checkpoint directory"""

    default_prompt: Optional[str] = None
    """Default prompt when client doesn't provide one"""

    host: str = "0.0.0.0"
    """Server bind address"""

    port: int = 8000
    """Server bind port"""

    model_device: str = "cuda"
    """PyTorch device for model inference"""

    log_dir: str = ""
    """Directory to save inference logs and plots. Empty = disabled."""

    heartbeat_sec: int = 30
    """Print heartbeat every N seconds. 0 = disable."""

    action_start_ratio: float = 0.0
    """Fraction of trajectory start to keep (after prepending state)"""

    action_end_ratio: float = 0.8
    """Fraction of trajectory end to keep (drop noisy tail)"""

    warmup: bool = False
    """Run one dummy inference after loading to warm up JIT/XLA compilation"""


def main(args: Args) -> None:
    import websockets.sync.server

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "unknown"

    print("=" * 60, flush=True)
    print("  OpenPI X2Robot Inference Server", flush=True)
    print("=" * 60, flush=True)
    print(f"  Config      : {args.config}", flush=True)
    print(f"  Checkpoint  : {args.checkpoint_dir}", flush=True)
    print(f"  Prompt      : {args.default_prompt or '(none)'}", flush=True)
    print(f"  Device      : {args.model_device}", flush=True)
    print(f"  Log dir     : {args.log_dir or '(disabled)'}", flush=True)
    print(f"  Action slice: [{args.action_start_ratio:.2f}, {args.action_end_ratio:.2f})", flush=True)
    print(f"  Bind        : {args.host}:{args.port}", flush=True)
    print("-" * 60, flush=True)

    infer_logger: Optional[InferenceLogger] = None
    if args.log_dir:
        infer_logger = InferenceLogger(log_dir=args.log_dir, mode="unknown")

    policy = OpenPIServingPolicy(
        config_name=args.config,
        checkpoint_dir=args.checkpoint_dir,
        default_prompt=args.default_prompt,
        pytorch_device=args.model_device,
        inference_logger=infer_logger,
        action_start_ratio=args.action_start_ratio,
        action_end_ratio=args.action_end_ratio,
    )

    if infer_logger is not None:
        if policy.is_mm_model:
            infer_logger.mode = "mm"
        elif policy.is_joints_model:
            infer_logger.mode = "joints"
        else:
            infer_logger.mode = "ee"

    if args.warmup:
        print("[WARMUP] Running dummy inference ...", flush=True)
        t0 = time.time()
        dummy = {
            "state": {
                "follow1_pos": np.zeros(7, dtype=np.float32),
                "follow2_pos": np.zeros(7, dtype=np.float32),
            },
            "views": {},
            "instruction": np.array([args.default_prompt or "warmup"], dtype=object),
        }
        if policy.is_mm_model:
            dummy["state"]["head_pos"] = np.zeros(2, dtype=np.float32)
            dummy["state"]["lift"] = np.zeros(1, dtype=np.float32)
            dummy["state"]["velocity_decomposed_odom"] = np.zeros(3, dtype=np.float32)
        try:
            _ = policy.infer(dummy)
            print(f"[WARMUP] Done ({time.time() - t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"[WARMUP] Failed (non-fatal): {e}", flush=True)

    metadata = policy.metadata
    print(f"  Metadata: {metadata}", flush=True)

    handler = _create_ws_handler(policy, metadata)

    print("-" * 60, flush=True)
    print(f"  WebSocket server starting ...", flush=True)
    print(f"    Local  : ws://localhost:{args.port}", flush=True)
    print(f"    Host   : ws://{local_ip}:{args.port}", flush=True)
    print("=" * 60, flush=True)

    with websockets.sync.server.serve(
        handler,
        host=args.host,
        port=args.port,
        max_size=None,
        compression=None,
    ) as server:
        print("", flush=True)
        print("=" * 60, flush=True)
        print("  SERVER IS READY — waiting for connections ...", flush=True)
        print("=" * 60, flush=True)
        print("", flush=True)

        stop_event = threading.Event()

        def _heartbeat_loop():
            if args.heartbeat_sec <= 0:
                return
            while not stop_event.wait(args.heartbeat_sec):
                steps = len(infer_logger.states) if infer_logger is not None else 0
                print(
                    f"[HEARTBEAT] server alive | port={args.port} | logged_steps={steps}",
                    flush=True,
                )

        hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
        hb_thread.start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped by user (Ctrl+C).", flush=True)
        finally:
            stop_event.set()
            if infer_logger is not None and len(infer_logger.states) > 0:
                tag = args.config
                print(f"Flushing inference log ({len(infer_logger.states)} steps) ...", flush=True)
                old_sigint = signal.getsignal(signal.SIGINT)
                try:
                    signal.signal(signal.SIGINT, signal.SIG_IGN)
                    infer_logger.flush_plot(tag=tag)
                finally:
                    signal.signal(signal.SIGINT, old_sigint)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
    )
    try:
        import tyro
        main(tyro.cli(Args))
    except ImportError:
        import argparse
        parser = argparse.ArgumentParser(description="OpenPI X2Robot Inference Server")
        parser.add_argument("--config", required=True, help="OpenPI training config name")
        parser.add_argument("--checkpoint-dir", required=True, help="Checkpoint directory")
        parser.add_argument("--default-prompt", default=None)
        parser.add_argument("--host", default="0.0.0.0")
        parser.add_argument("--port", type=int, default=8000)
        parser.add_argument("--model-device", default="cuda")
        parser.add_argument("--log-dir", default="")
        parser.add_argument("--heartbeat-sec", type=int, default=30)
        parser.add_argument("--action-start-ratio", type=float, default=0.0)
        parser.add_argument("--action-end-ratio", type=float, default=0.8)
        parser.add_argument("--warmup", action="store_true")
        parsed = parser.parse_args()
        main(Args(
            config=parsed.config,
            checkpoint_dir=parsed.checkpoint_dir,
            default_prompt=parsed.default_prompt,
            host=parsed.host,
            port=parsed.port,
            model_device=parsed.model_device,
            log_dir=parsed.log_dir,
            heartbeat_sec=parsed.heartbeat_sec,
            action_start_ratio=parsed.action_start_ratio,
            action_end_ratio=parsed.action_end_ratio,
            warmup=parsed.warmup,
        ))
