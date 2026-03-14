# ManipArena Model Server

<!-- TODO: ![ManipArena teaser](docs/teaser.jpg) -->

ManipArena is a benchmark for bimanual manipulation policies on the EX001 desktop robot.
For more details, please check our [website](https://maniparena.github.io/).

This repo contains the **model server template** for ManipArena evaluation.
Participants implement their policy in [`examples/my_policy.py`](examples/my_policy.py);
the server handles all WebSocket communication with the evaluation client.


## Getting Started

```bash
pip install -r requirements.txt

# Edit examples/my_policy.py with your model, then:
PYTHONPATH=examples python -m maniparena.launch \
    --checkpoint /path/to/ckpt \
    --control-mode end_pose \
    --port 8000
```

To verify your server is working, in a separate shell:

```bash
python scripts/test_server.py
```


## Serving Your Policy

Subclass `ModelPolicy` and implement three methods:

```python
from maniparena.policy import ModelPolicy

class MyPolicy(ModelPolicy):
    def load_model(self, checkpoint_path, device):
        ...                             # return your model

    def run_inference(self, model_input):
        ...                             # return (T, 14) action array

    def convert_output(self, model_output):
        ...                             # return {"follow1_pos": ..., "follow2_pos": ...}
```

See [`examples/my_policy.py`](examples/my_policy.py) for the full template and
[`examples/pytorch_example.py`](examples/pytorch_example.py) for a PyTorch reference.


## Observation & Action Format

### Input (observation from client)

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

### Output (action from server)

```python
{
    "follow1_pos": [[x, y, z, r, p, y, grip], ...],   # List[List[float]], T steps
    "follow2_pos": [[x, y, z, r, p, y, grip], ...],
}
```

> **Important:** Values must be Python lists (`.tolist()`), **not** numpy arrays.
> The client does `[current_pos] + actions` — if `actions` is numpy, `+` silently
> broadcasts instead of concatenating, corrupting the trajectory.


## Protocol

```
Client connects → Server sends metadata (msgpack)
Loop: Client sends observation (msgpack) → Server returns actions (msgpack)
```

Metadata sent on connect:

```json
{"control_mode": "end_pose", "action_horizon": 50, "state_dim": 14}
```


## Project Structure

```
maniparena/
  policy.py       # ModelPolicy base class (do not modify)
  server.py       # WebSocket server (do not modify)
  utils.py        # I/O conversion helpers
  launch.py       # CLI entry point
examples/
  my_policy.py    # ← your implementation goes here
  pytorch_example.py
scripts/
  test_server.py  # self-check tool
```
