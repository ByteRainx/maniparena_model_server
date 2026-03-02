# OpenPI X2Robot Server

Complete OpenPI inference server directly compatible with current `x2robot_client` (EX001 / Desktop / CX001).

## Supported Model Types

| Type | State Dim | Description |
|---|---|---|
| **EE** | 14D | End-effector control `[x,y,z,r,p,y,gripper] × 2` |
| **Joints** | 14D | Joint control `[j1-j6,gripper] × 2` (delta actions) |
| **MobileManipulation** | 20D | 14D arms + head(2) + lift(1) + chassis(3) |

## Available Config Names

```text
# EE mode
pi0_x2robot_pick_banana_ee
pi05_x2robot_pick_banana_ee
pi0_x2robot_pick_banana_0104_ee
pi05_x2robot_pick_banana_0104_ee
pi0_x2robot_pick_banana_merged_ee
pi05_x2robot_pick_banana_merged_ee

# Joints mode
pi05_pick_banana_joints

# Mobile Manipulation mode (20D)
pi0_put_clothes_in_hamper_mm
pi05_put_clothes_in_hamper_mm
pi0_take_and_set_tableware_mm
pi05_take_and_set_tableware_mm
```

## Quick Start

### Option A: standalone server (recommended)

```bash
# EE mode
python openpi_branch/serve_openpi.py \
    --config pi0_x2robot_pick_banana_merged_ee \
    --checkpoint-dir /path/to/checkpoints/step_10000 \
    --default-prompt "pick up the banana" \
    --port 8000 \
    --warmup

# Joints mode
python openpi_branch/serve_openpi.py \
    --config pi05_pick_banana_joints \
    --checkpoint-dir /path/to/checkpoints/step_10000 \
    --port 8000

# Mobile Manipulation (20D)
python openpi_branch/serve_openpi.py \
    --config pi05_put_clothes_in_hamper_mm \
    --checkpoint-dir /path/to/checkpoints/step_10000 \
    --port 8000
```

### Option B: launch through universal WebSocket framework

```bash
python openpi_branch/launch_openpi_server.py \
    --config pi0_x2robot_pick_banana_merged_ee \
    --checkpoint-dir /path/to/checkpoints/step_10000 \
    --port 8000
```

## Full Arguments

```text
python openpi_branch/serve_openpi.py --help

Required:
  --config              OpenPI training config name
  --checkpoint-dir      model checkpoint directory

Optional:
  --default-prompt      fallback prompt if client provides no instruction
  --host                server host (default 0.0.0.0)
  --port                server port (default 8000)
  --model-device        inference device (default cuda)
  --log-dir             directory to save inference logs (default off)
  --heartbeat-sec       heartbeat interval seconds (default 30, 0=off)
  --action-start-ratio  action slicing start ratio (default 0.0)
  --action-end-ratio    action slicing end ratio (default 0.8)
  --warmup              run one dummy inference after load for JIT warmup
```

## Communication Protocol

### Input (Client -> Server)

Server supports both new nested format and legacy flat format:

```python
# New format (EX001 / Desktop)
{
    "state": {
        "follow1_pos": np.float32[7],
        "follow2_pos": np.float32[7],
        "follow1_joints": np.float32[N],   # optional
        "follow2_joints": np.float32[N],   # optional
        "head_pos": np.float32[2],         # MM mode
        "lift": np.float32[1],             # MM mode
        "velocity_decomposed_odom": np.float32[3],  # MM mode
    },
    "views": {
        "camera_left": "base64_jpeg",
        "camera_front": "base64_jpeg",
        "camera_right": "base64_jpeg",
    },
    "instruction": np.array(["pick up the banana"]),
}

# Legacy format (e.g. CX001)
{
    "CAMERA_LEFT": "base64_jpeg",
    "CAMERA_FRONT": "base64_jpeg",
    "CAMERA_RIGHT": "base64_jpeg",
    "ACTION_FOLLOW1_POS": [7D],
    "ACTION_FOLLOW2_POS": [7D],
    "instruction": "pick up the banana",
}
```

### Output (Server -> Client)

```python
# EE mode
{"follow1_pos": np.float32[T, 7], "follow2_pos": np.float32[T, 7]}

# Joints mode
{"follow1_joints": np.float32[T, 7], "follow2_joints": np.float32[T, 7],
 "follow1_pos": np.float32[T, 7], "follow2_pos": np.float32[T, 7]}

# MM mode (20D)
{"follow1_pos": ..., "follow2_pos": ...,
 "head_pos": np.float32[T, 2], "lift": np.float32[T, 1],
 "velocity_decomposed": np.float32[T, 3]}
```

## Architecture

```text
serve_openpi.py
├── Cross-compatible msgpack+numpy
├── OpenPIServingPolicy
│   ├── _extract_state()
│   ├── _extract_images()
│   ├── _extract_prompt()
│   ├── _convert_output()
│   └── infer()
├── InferenceLogger
└── WebSocket Server
```

### Output processing strategy

1. **Prepend State**: prepend current state to ensure smooth trajectory start.
2. **Action Slicing**: keep a valid interval using `action_start_ratio` / `action_end_ratio`.
3. **Per-Component Split**: split unified action vector into arm/head/lift/chassis components.

## Inference Logs

When `--log-dir` is enabled, each inference writes state/action chunks and saves on shutdown:

- `{config_name}.npz` — raw arrays (`states`, `actions`, `prompts`, `timestamps`)
- `{config_name}.png` — visualization (state/action curves per dimension)
