# Universal Model Server Template

Universal model server template compatible with `x2robot_client` (Desktop / CX001).  
Participants only need to implement model-specific logic. WebSocket transport and protocol handling are already included.

## Architecture

```text
maniparena_model_server/
├── websocket_server.py    # WebSocket server core (do not modify)
├── policy_base.py         # Policy base class (do not modify)
├── my_policy.py           # Your policy implementation (edit this file)
├── utils.py               # Utility helpers (optional)
├── launch_server.py       # Server launcher (do not modify)
├── requirements.txt       # Dependencies
└── examples/              # Reference examples
    └── pytorch_policy_example.py
```

### Data Flow

```text
x2robot_client
    ↓ WebSocket connection
    ↓ server sends metadata
websocket_server.py
    ↓ receives observation (msgpack)
    ↓ calls policy.infer(obs)
my_policy.py
    ↓ convert_input()
    ↓ run_inference()
    ↓ convert_output()
    ↓ returns result
websocket_server.py
    ↓ sends result (msgpack)
x2robot_client
```

## Two Task Types

| | Desktop (14D) | CX001 Mobile Manipulation (20D) |
|---|---|---|
| Robot | EX001 dual-arm desktop | CX001 mobile manipulation |
| Action dimension | 14 (left arm 7D + right arm 7D) | 20 (14D arms + 2D head + 1D lift + 3D chassis) |
| Output keys | **lowercase** | **UPPERCASE** |
| Helper function | `convert_model_output_to_x2robot_format()` | `convert_model_output_to_legacy_format()` |

### Output Key Mapping

| Field | Desktop Key | CX001 Key | Per-step shape |
|---|---|---|---|
| Left arm | `follow1_pos` | `FOLLOW1_POS` | `[7]` |
| Right arm | `follow2_pos` | `FOLLOW2_POS` | `[7]` |
| Head | — | `HEAD_POS` | `[2]` |
| Lift | — | `LIFT_OUT` | `[1]` |
| Chassis | — | `CAR_POSE_OUT` | `[3]` |

## Critical Notes

### 1) Always return Python lists (`.tolist()`)

Desktop client interpolation prepends the current position:

```python
arm1_actions = [self.last_arm_l_pos] + arm1_actions
```

If `arm1_actions` is a numpy array, `+` triggers element-wise broadcasting instead of list concatenation, silently corrupting the trajectory.

Correct pattern:

```python
def convert_output(self, model_output):
    actions = np.array(model_output)  # (T, 14)
    return {
        "follow1_pos": actions[:, :7].tolist(),
        "follow2_pos": actions[:, 7:14].tolist(),
    }
```

### 2) Key casing differs between Desktop and CX001

| Client | Required key style | If key style is wrong |
|---|---|---|
| Desktop | lowercase (`follow1_pos`) | `.get()` returns `None`, action is skipped |
| CX001 | UPPERCASE (`FOLLOW1_POS`) | `.get()` returns `[]`, robot does not move |

### 3) Value type must be `List[List[float]]`

```python
# Correct
"follow1_pos": [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], ...]

# Wrong: numpy array
"follow1_pos": np.array([[0.1, 0.2, ...], ...])

# Wrong: missing time dimension
"follow1_pos": [0.1, 0.2, 0.3, ...]
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Implement `my_policy.py`

Implement:
1. `load_model()` — required
2. `convert_input()` — optional (recommended to use helper)
3. `run_inference()` — required
4. `convert_output()` — required (`.tolist()` is mandatory)

### 3. Start server

```bash
python launch_server.py \
    --checkpoint /path/to/your/checkpoint \
    --control-mode end_pose \
    --port 8000
```

## Input Format

### Desktop client (nested format)

```python
{
    "state": {
        "follow1_pos": [x, y, z, roll, pitch, yaw, gripper],
        "follow2_pos": [x, y, z, roll, pitch, yaw, gripper],
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

### CX001ClientROS2 client (flat format, uppercase keys)

> Mobile manipulation track uses `CX001ClientROS2`.  
> Images are **numpy RGB arrays** (not base64), and state uses uppercase `ACTION_*` keys.

```python
{
    "CAMERA_LEFT": np.ndarray(H, W, 3),
    "CAMERA_FRONT": np.ndarray(H, W, 3),
    "CAMERA_RIGHT": np.ndarray(H, W, 3),
    "ACTION_FOLLOW1_POS": np.array([7D], dtype=np.float32),
    "ACTION_FOLLOW2_POS": np.array([7D], dtype=np.float32),
    "ACTION_FOLLOW1_JOINTS_CUR": np.array([...], dtype=np.float32),
    "ACTION_FOLLOW2_JOINTS_CUR": np.array([...], dtype=np.float32),
    "CAR_POSE": np.array([x, y, theta], dtype=np.float32),
    "LIFT": np.array([height], dtype=np.float32),
    "HEAD_POS": np.array([yaw, pitch], dtype=np.float32),
    "INSTRUCTION": np.array(["task description"], dtype=np.object_),
}
```

`convert_observation_to_model_input()` supports both Desktop nested format and CX001 flat format.
With `decode_images=True`, both base64 JPEG (Desktop) and numpy RGB arrays (CX001) are handled correctly.

## Output Format

### Desktop output (14D, lowercase keys)

```python
{
    "follow1_pos": [[x, y, z, r, p, y, gripper], ...],  # (T, 7)
    "follow2_pos": [[x, y, z, r, p, y, gripper], ...],
}
```

### CX001 output (20D, UPPERCASE keys)

```python
{
    "FOLLOW1_POS": [[x, y, z, r, p, y, gripper], ...],  # (T, 7)
    "FOLLOW2_POS": [[x, y, z, r, p, y, gripper], ...],
    "HEAD_POS": [[yaw, pitch], ...],                    # (T, 2)
    "LIFT_OUT": [[height], ...],                        # (T, 1)
    "CAR_POSE_OUT": [[x, y, theta], ...],               # (T, 3)
}
```

## Utilities (`utils.py`)

### Input conversion

```python
from utils import convert_observation_to_model_input

model_input = convert_observation_to_model_input(obs, control_mode)
```

### Desktop output conversion (14D)

```python
from utils import convert_model_output_to_x2robot_format

result = convert_model_output_to_x2robot_format(
    actions,  # (T, 14)
    control_mode,
    action_horizon,
)
```

### CX001 output conversion (20D)

```python
from utils import convert_model_output_to_legacy_format

result = convert_model_output_to_legacy_format(
    actions,  # (T, 14)
    control_mode,
    action_horizon,
    head_actions=...,      # optional (T, 2)
    lift_actions=...,      # optional (T, 1)
    car_pose_actions=...,  # optional (T, 3)
)
```

## Protocol

1. After connect, server sends metadata (msgpack).
2. Client sends observation and server returns action (both msgpack).

### Metadata

```python
{
    "control_mode": "end_pose",  # "joints" or "end_pose"
    "action_horizon": 50,
    "state_dim": 14,
    "state_dim_per_arm": 7,
    "protocol_version": "2.0",
}
```

## CLI Args

```bash
python launch_server.py --help
```

- `--checkpoint`: checkpoint path
- `--control-mode`: `joints` / `end_pose`
- `--action-horizon`: action sequence length
- `--device`: device (`cuda:0` / `cpu`)
- `--port`: server port (default `8000`)
- `--host`: server host (default `0.0.0.0`)
