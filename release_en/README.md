# ManipArena Model Server

WebSocket model server template for the **ManipArena CVPR Benchmark**.
Participants implement their model in `my_policy.py`. The server handles all WebSocket communication with the evaluation robot client.

## Quick Start

```bash
pip install -r requirements.txt
# Edit my_policy.py, then:
python launch_server.py --checkpoint /path/to/ckpt --port 8000
```

## Task Types

|  | Desktop (14D) | Mobile Manipulation (20D) |
|---|---|---|
| Robot | EX001 dual-arm tabletop | CX001 mobile manipulator |
| Action dim | 14 (2 × 7D arms) | 20 (14D arms + 2D head + 1D lift + 3D chassis) |
| Output key casing | **lowercase** | **UPPERCASE** |

### Output Key Reference

| Field | Desktop Key | CX001 Key | Shape per step |
|-------|------------|-----------|----------------|
| Left arm | `follow1_pos` | `FOLLOW1_POS` | `(7,)` |
| Right arm | `follow2_pos` | `FOLLOW2_POS` | `(7,)` |
| Head | — | `HEAD_POS` | `(2,)` |
| Lift | — | `LIFT_OUT` | `(1,)` |
| Chassis | — | `CAR_POSE_OUT` | `(3,)` |

## IMPORTANT: `.tolist()` Required

The robot client concatenates trajectories using Python `+`:

```python
actions = [current_position] + actions   # expects list + list
```

If your server returns a **numpy array**, `+` triggers element-wise broadcasting instead of list concatenation, **silently corrupting the trajectory**.

Always convert:

```python
def convert_output(self, raw_output):
    actions = np.array(raw_output)             # (T, 14)
    return {
        "follow1_pos": actions[:, :7].tolist(),   # Python list, NOT numpy
        "follow2_pos": actions[:, 7:14].tolist(),
    }
```

Returning the wrong key casing causes the client's `.get()` to return `None` or `[]`, and the robot will not move.

## Protocol

```
Client connects → Server sends metadata (msgpack)
Loop: Client sends observation (msgpack) → Server returns actions (msgpack)
```

### Metadata (sent on connect)

```json
{"control_mode": "end_pose", "action_horizon": 50, "state_dim": 14}
```

### Observation Format

**Desktop** — nested dict:

```python
{
    "state": {
        "follow1_pos": [x, y, z, r, p, y, gripper],   # 7D left arm
        "follow2_pos": [x, y, z, r, p, y, gripper],   # 7D right arm
        "follow1_joints_cur": [...],
        "follow2_joints_cur": [...],
    },
    "views": {
        "camera_left": "<base64 JPEG>",
        "camera_front": "<base64 JPEG>",
        "camera_right": "<base64 JPEG>",
    },
    "instruction": np.array(["Pick up the cup."], dtype=np.object_),
}
```

**CX001 (CX001ClientROS2)** — flat dict, UPPERCASE keys, numpy images:

```python
{
    "CAMERA_LEFT": np.ndarray (H, W, 3),     # numpy RGB (NOT base64)
    "CAMERA_FRONT": np.ndarray (H, W, 3),
    "CAMERA_RIGHT": np.ndarray (H, W, 3),
    "ACTION_FOLLOW1_POS": np.array([7D], dtype=np.float32),
    "ACTION_FOLLOW2_POS": np.array([7D], dtype=np.float32),
    "ACTION_FOLLOW1_JOINTS_CUR": np.array([...], dtype=np.float32),
    "ACTION_FOLLOW2_JOINTS_CUR": np.array([...], dtype=np.float32),
    "CAR_POSE": np.array([x, y, theta], dtype=np.float32),
    "LIFT": np.array([height], dtype=np.float32),
    "HEAD_POS": np.array([yaw, pitch], dtype=np.float32),
    "INSTRUCTION": np.array(["..."], dtype=np.object_),
}
```

> Desktop sends base64 JPEG strings for images; CX001 sends numpy RGB arrays.
> The `convert_observation_to_model_input()` helper handles both transparently.

### Action Response

**Desktop (14D)** — lowercase keys:

```python
{
    "follow1_pos": [[x,y,z,r,p,y,grip], ...],   # List[List[float]], T steps
    "follow2_pos": [[x,y,z,r,p,y,grip], ...],
}
```

**CX001 (20D)** — UPPERCASE keys:

```python
{
    "FOLLOW1_POS": [[...], ...],
    "FOLLOW2_POS": [[...], ...],
    "HEAD_POS":    [[yaw, pitch], ...],
    "LIFT_OUT":    [[height], ...],
    "CAR_POSE_OUT":[[x, y, theta], ...],
}
```

## Files

| File | Edit? | Description |
|------|-------|-------------|
| `my_policy.py` | **YES** | Your model implementation |
| `utils.py` | optional | I/O conversion helpers |
| `policy_base.py` | no | Abstract base class |
| `websocket_server.py` | no | WebSocket handler |
| `launch_server.py` | no | CLI entry point |

## Utility Functions

```python
from utils import (
    convert_observation_to_model_input,
    convert_output_desktop,
    convert_output_cx001,
)

# Parse observation (auto-detects Desktop vs CX001 format)
model_in = convert_observation_to_model_input(obs, "end_pose")
# → {"left": ndarray, "front": ndarray, "right": ndarray,
#    "state": ndarray(14,), "instruction": str}

# Desktop output (14D, lowercase keys)
result = convert_output_desktop(actions_14d, "end_pose", 50)

# CX001 output (20D, UPPERCASE keys)
result = convert_output_cx001(
    actions_14d, "end_pose", 50,
    head=head_2d, lift=lift_1d, car_pose=car_3d,
)
```

## CLI

```bash
python launch_server.py \
    --checkpoint /path/to/ckpt \
    --control-mode end_pose \
    --action-horizon 50 \
    --device cuda:0 \
    --port 8000
```
