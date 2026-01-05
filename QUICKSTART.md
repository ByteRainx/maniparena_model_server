# 快速开始指南

> 这是最简化的快速开始指南。如需详细文档，请查看 [README.md](README.md)。

## 5分钟上手

### 1. 安装依赖

```bash
cd universal_model_server
pip install -r requirements.txt
```

如果你需要在 `convert_input()`/`utils.py` 中把 `CAMERA_*` 解码成 numpy RGB（大多数模型需要），再安装：

```bash
pip install opencv-python
```

### 2. 修改配置

编辑 `my_policy.py`，修改顶部配置：

```python
DEFAULT_CHECKPOINT_PATH = "/path/to/your/checkpoint"
DEFAULT_CONTROL_MODE = "joints"  # 或 "end_pose"
DEFAULT_ACTION_HORIZON = 50
DEFAULT_DEVICE = "cuda:0"
```

### 3. 实现4个方法

#### 方法1：加载模型

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

#### 方法2：输入转换（使用工具函数）

```python
def convert_input(self, obs):
    from utils import convert_observation_to_model_input
    return convert_observation_to_model_input(obs, self.control_mode)
```

#### 方法3：输出转换（使用工具函数）

```python
def convert_output(self, model_output):
    from utils import convert_model_output_to_x2robot_format
    return convert_model_output_to_x2robot_format(
        model_output, self.control_mode, self.action_horizon
    )
```

#### 方法4：模型推理

```python
def run_inference(self, model_input):
    import torch
    with torch.no_grad():
        output = self.model(model_input)
    return output.cpu().numpy()
```

### 4. 启动服务器

```bash
python launch_server.py \
    --checkpoint /path/to/your/checkpoint \
    --control-mode joints \
    --port 8000
```

### 5. 测试连接

在另一个终端：

```python
from x2robot_client.inference_client import RobotClient

client = RobotClient(uri="ws://localhost:8000")
client.connect_sync()

obs = {
    "CAMERA_LEFT": "...",  # base64图像
    "CAMERA_FRONT": "...",
    "CAMERA_RIGHT": "...",
    "ACTION_FOLLOW1_POS": [0.0] * 7,
    "ACTION_FOLLOW2_POS": [0.0] * 7,
    "instruction": "Pick up the cup."
}

result = client.predict_sync(obs)
print(result)
```

## 完整示例

查看 `my_policy.py` 中的注释示例，或参考 `README.md` 中的详细说明和更多示例。

## 需要帮助？

- **详细文档**：查看 [README.md](README.md) 了解完整的API文档、输入输出格式、通信协议等
- **架构设计**：查看 [ARCHITECTURE.md](ARCHITECTURE.md) 了解设计原理和扩展点
- **工具函数**：查看 `utils.py` 源码了解所有可用的工具函数及其用法
