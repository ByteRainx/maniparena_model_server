# Quick Start Guide

> This is a minimal quick-start guide. For full details, see `README.md`.

## Get Running in 5 Minutes

### 1) Install dependencies

```bash
cd universal_model_server
pip install -r requirements.txt
```

If your `convert_input()`/`utils.py` decodes `CAMERA_*` to numpy RGB (common), install:

```bash
pip install opencv-python
```

### 2) Update config

Edit the top of `my_policy.py`:

```python
DEFAULT_CHECKPOINT_PATH = "/path/to/your/checkpoint"
DEFAULT_CONTROL_MODE = "joints"  # or "end_pose"
DEFAULT_ACTION_HORIZON = 50
DEFAULT_DEVICE = "cuda:0"
```

### 3) Implement 4 methods

#### Method 1: load model

```python
def load_model(self, checkpoint_path: str, device: str):
    import torch
    model = YourModelClass()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model
```

#### Method 2: input conversion

```python
def convert_input(self, obs):
    from utils import convert_observation_to_model_input
    return convert_observation_to_model_input(obs, self.control_mode)
```

#### Method 3: output conversion

```python
def convert_output(self, model_output):
    from utils import convert_model_output_to_x2robot_format
    return convert_model_output_to_x2robot_format(
        model_output, self.control_mode, self.action_horizon
    )
```

#### Method 4: inference

```python
def run_inference(self, model_input):
    import torch
    with torch.no_grad():
        output = self.model(model_input)
    return output.cpu().numpy()
```

### 4) Start server

```bash
python launch_server.py \
    --checkpoint /path/to/your/checkpoint \
    --control-mode joints \
    --port 8000
```

### 5) Test connection

In another terminal:

```python
from x2robot_client.inference_client import RobotClient

client = RobotClient(uri="ws://localhost:8000")
client.connect_sync()

obs = {
    "CAMERA_LEFT": "...",
    "CAMERA_FRONT": "...",
    "CAMERA_RIGHT": "...",
    "ACTION_FOLLOW1_POS": [0.0] * 7,
    "ACTION_FOLLOW2_POS": [0.0] * 7,
    "instruction": "Pick up the cup."
}

result = client.predict_sync(obs)
print(result)
```

## Need More?

- Full docs: `README.md`
- Architecture notes: `ARCHITECTURE.md`
- Utility helper details: `utils.py`
