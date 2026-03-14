# ManipArena Model Server

<!-- TODO: ![ManipArena](docs/teaser.jpg) -->

[ManipArena](https://maniparena.github.io/) is a real-robot benchmark for bimanual manipulation, featuring 20 tasks across execution reasoning, semantic reasoning, and mobile manipulation. Participants serve their model remotely — no robot hardware needed.

This repo is the **model server template**. Implement your policy in [`examples/my_policy.py`](examples/my_policy.py), and the framework handles WebSocket transport, observation parsing, and action serialization.

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Edit examples/my_policy.py (implement load_model / run_inference / convert_output)

# 3. Launch
python serve.py --checkpoint /path/to/ckpt --control-mode end_pose --port 8000
```

## Implement Your Policy

Subclass `ModelPolicy` and implement three methods:

```python
from maniparena.policy import ModelPolicy
from maniparena.utils import convert_observation_to_model_input, convert_model_output_to_action

class MyPolicy(ModelPolicy):
    def load_model(self, checkpoint_path, device):
        ...                             # return your model

    def run_inference(self, model_input):
        ...                             # return (T, 14) numpy array

    def convert_output(self, model_output):
        return convert_model_output_to_action(model_output, self.control_mode, self.action_horizon)
```

See [`examples/my_policy.py`](examples/my_policy.py) for the full template,
[`examples/pytorch_example.py`](examples/pytorch_example.py) for a PyTorch reference,
and [`examples/openpi_example.py`](examples/openpi_example.py) for a ready-to-run
[OpenPI](https://github.com/Physical-Intelligence/openpi) example.

## Self-Check Before Submission

We provide three scripts in `scripts/` to validate your server **before** you submit.
Run them against your running server in a separate terminal.

### Step 1 — Ping & handshake

```bash
python scripts/mock_ping.py --uri ws://127.0.0.1:8000
```

Checks: WebSocket reachable, first frame is valid msgpack metadata, required fields
(`control_mode`, `action_horizon`, `state_dim`) present.

### Step 2 — Request/response schema

```bash
python scripts/mock_schema_check.py --uri ws://127.0.0.1:8000
```

Sends a dummy Desktop observation and validates the response:
lowercase keys, `List[List[float]]` trajectories, correct dimensions.

### Step 3 — Open-loop evaluation (recommended)

We recommend running open-loop evaluation before submission to visually check whether your model's predictions align with the ground truth trajectories.

**Quick check** (no video needed, parquet only):

```bash
python scripts/mock_openloop_eval.py \
    --uri ws://127.0.0.1:8000 \
    --data-dir /path/to/lerobot_dataset \
    --save-dir /path/to/output \
    --sample-limit 3 --enable-plots
```

**Full evaluation** (parquet + video, generates pred vs gt plots):

```bash
python scripts/eval_openloop.py \
    --server ws://127.0.0.1:8000 \
    --dataset /path/to/lerobot_dataset \
    --episode 0 \
    --save-dir openloop_plots \
    --action-chunk 32
```

> Both expect LeRobot layout: `data/chunk-*/episode_*.parquet`
> (+ `videos/chunk-*/observation.images.*/episode_*.mp4` for full eval).

---

## Observation & Action Format

### Observation (client → server)

```python
{
    "state": {
        "follow1_pos": [x, y, z, r, p, y, gripper],   # left arm 7D
        "follow2_pos": [x, y, z, r, p, y, gripper],   # right arm 7D
    },
    "views": {
        "camera_left":  "<base64 JPEG>",
        "camera_front": "<base64 JPEG>",
        "camera_right": "<base64 JPEG>",
    },
    "instruction": "Pick up the cup.",
}
```

### Action (server → client)

```python
{
    "follow1_pos": [[x, y, z, r, p, y, grip], ...],   # List[List[float]], T steps
    "follow2_pos": [[x, y, z, r, p, y, grip], ...],
}
```

> [!CAUTION]
> Values **must** be Python lists (`.tolist()`), not numpy arrays. The client does `[current_pos] + actions` — numpy `+` silently broadcasts instead of concatenating, corrupting the trajectory.

## Protocol

```
Client connects → Server sends metadata (msgpack)
Loop:  Client sends observation (msgpack) → Server returns actions (msgpack)
```

Metadata example: `{"control_mode": "end_pose", "action_horizon": 50, "state_dim": 14}`

## CLI Arguments

```bash
python serve.py --help
```

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | (required) | Model checkpoint path |
| `--control-mode` | `end_pose` | `end_pose` or `joints` |
| `--action-horizon` | `50` | Action sequence length (T) |
| `--device` | `cuda:0` | `cuda:0`, `cpu`, etc. |
| `--port` | `8000` | Server port |
| `--host` | `0.0.0.0` | Server host |

## Project Structure

```
serve.py                   # ← start here: python serve.py --checkpoint ...
maniparena/                # core framework (do not modify)
    policy.py              #   ModelPolicy base class
    server.py              #   WebSocket server
    utils.py               #   observation/action conversion helpers
    launch.py              #   CLI entry point
examples/                  # participant code
    my_policy.py           #   ← edit this file
    pytorch_example.py     #   PyTorch reference
    openpi_example.py      #   OpenPI ready-to-run example
scripts/                   # self-check & evaluation tools
    mock_ping.py           #   Step 1: handshake check
    mock_schema_check.py   #   Step 2: schema validation
    mock_openloop_eval.py  #   Step 3: quick open-loop check
    eval_openloop.py       #   Full open-loop eval with plots
```
