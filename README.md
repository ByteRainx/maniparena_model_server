# ManipArena Model Server

<!-- TODO: ![ManipArena](docs/teaser.jpg) -->

Model server template for the [ManipArena](https://maniparena.github.io/) benchmark (CVPR 2025).

Participants implement a policy in [`examples/my_policy.py`](examples/my_policy.py).
The framework handles WebSocket transport, observation parsing, and action serialization —
you only write model-specific code.

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

See [`examples/my_policy.py`](examples/my_policy.py) for the full template and
[`examples/pytorch_example.py`](examples/pytorch_example.py) for a PyTorch reference.

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

### Step 3 — Open-loop evaluation (optional)

```bash
python scripts/mock_openloop_eval.py \
    --uri ws://127.0.0.1:8000 \
    --data-dir /path/to/lerobot_dataset \
    --save-dir /path/to/output \
    --sample-limit 3
```

Replays LeRobot episodes through your server and saves `pred`/`gt` arrays to `.npz`.
Add `--enable-plots` to generate comparison plots.

> `--data-dir` expects LeRobot layout: `data/chunk-*/episode_*.parquet`
> (+ optional `videos/` sibling directory).

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
> **Action values MUST be Python lists (`.tolist()`), NOT numpy arrays.**
>
> The robot client prepends the current position before executing:
> ```python
> arm1_actions = [self.last_arm_l_pos] + arm1_actions   # expects list + list
> ```
> If your server returns a **numpy array**, Python `+` triggers **element-wise broadcasting**
> instead of list concatenation — the trajectory is **silently corrupted** with no error.
> The robot will move to wrong positions and you will have no indication why.
>
> Always use `.tolist()`:
> ```python
> def convert_output(self, model_output):
>     actions = np.array(model_output)              # (T, 14)
>     return {
>         "follow1_pos": actions[:, :7].tolist(),    # Python list, NOT numpy
>         "follow2_pos": actions[:, 7:14].tolist(),
>     }
> ```
>
> The built-in helper `convert_model_output_to_action()` handles this correctly.
> If you write custom output conversion, always verify with `mock_schema_check.py`.

## Protocol

```
Client connects → Server sends metadata (msgpack)
Loop:  Client sends observation (msgpack) → Server returns actions (msgpack)
```

Metadata example: `{"control_mode": "end_pose", "action_horizon": 50, "state_dim": 14}`

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
scripts/                   # self-check tools
    mock_ping.py           #   Step 1: handshake check
    mock_schema_check.py   #   Step 2: schema validation
    mock_openloop_eval.py  #   Step 3: open-loop eval
```
