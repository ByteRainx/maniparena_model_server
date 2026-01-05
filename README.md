# 通用模型服务器模板

这是一个适配 `x2robot_client` 的通用模型服务器模板。参赛者只需要实现模型相关的部分，其他（WebSocket通信、协议处理等）已经实现好。

## 整体架构

```
universal_model_server/
├── websocket_server.py    # WebSocket服务器核心（不需要修改）
├── policy_base.py         # Policy基类定义（不需要修改）
├── my_policy.py           # 参赛者的Policy实现（需要修改）
├── utils.py               # 工具函数（可选使用）
├── launch_server.py       # 服务器启动文件（不需要修改）
├── requirements.txt       # 依赖列表
├── README.md              # 本文档
└── ARCHITECTURE.md        # 架构设计说明
```

### 数据流

```
x2robot_client (客户端)
    ↓ WebSocket连接
    ↓ 发送metadata
WebSocket服务器 (websocket_server.py)
    ↓ 接收observation (msgpack)
    ↓ 调用 policy.infer(obs)
Policy (my_policy.py)
    ↓ convert_input() - 输入格式转换
    ↓ run_inference() - 模型推理
    ↓ convert_output() - 输出格式转换
    ↓ 返回结果
WebSocket服务器
    ↓ 发送结果 (msgpack)
x2robot_client
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

如果你需要在 `convert_input()` 中把 `CAMERA_*` 解码成 numpy RGB（大多数模型需要），再安装：

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

编辑 `my_policy.py`，实现以下方法（详细示例见下方）：

1. **`load_model()`** - 加载你的模型（必须）
2. **`convert_input()`** - 输入格式转换（可选，推荐使用工具函数）
3. **`convert_output()`** - 输出格式转换（必须，推荐使用工具函数）
4. **`run_inference()`** - 模型推理（必须）

**推荐**：如果模型输入输出格式标准，可以直接使用 `utils.py` 中的工具函数，只需几行代码。

### 4. 启动服务器

```bash
python launch_server.py \
    --checkpoint /path/to/your/checkpoint \
    --control-mode joints \
    --port 8000
```

## 输入输出格式

### 输入格式（来自x2robot_client）

```python
{
    "CAMERA_LEFT": "base64_encoded_jpeg_string",      # Base64编码的JPEG图像
    "CAMERA_FRONT": "base64_encoded_jpeg_string",
    "CAMERA_RIGHT": "base64_encoded_jpeg_string",
    "ACTION_FOLLOW1_POS": [j1, j2, j3, j4, j5, j6, gripper],  # 7D数组
    "ACTION_FOLLOW2_POS": [j1, j2, j3, j4, j5, j6, gripper],  # 7D数组
    "instruction": "Pick up the cup."  # 可选的文本指令
}
```

**注意**：
- `ACTION_FOLLOW1_POS` 和 `ACTION_FOLLOW2_POS` 在joints模式下是 `[j1-j6, gripper]`
- 在end_pose模式下是 `[x, y, z, roll, pitch, yaw, gripper]`

### 输出格式（返回给x2robot_client）

#### Joints模式

```python
{
    "FOLLOW1_JOINTS": [
        [j1, j2, j3, j4, j5, j6, gripper],  # 第1步
        [j1, j2, j3, j4, j5, j6, gripper],  # 第2步
        ...  # 共action_horizon步
    ],
    "FOLLOW2_JOINTS": [
        [j1, j2, j3, j4, j5, j6, gripper],
        [j1, j2, j3, j4, j5, j6, gripper],
        ...
    ],
    "FOLLOW1_POS": [],
    "FOLLOW2_POS": []
}
```

#### End-pose模式

```python
{
    "FOLLOW1_POS": [
        [x, y, z, roll, pitch, yaw, gripper],  # 第1步
        [x, y, z, roll, pitch, yaw, gripper],  # 第2步
        ...  # 共action_horizon步
    ],
    "FOLLOW2_POS": [
        [x, y, z, roll, pitch, yaw, gripper],
        [x, y, z, roll, pitch, yaw, gripper],
        ...
    ],
    "FOLLOW1_JOINTS": [],
    "FOLLOW2_JOINTS": []
}
```

## 工具函数（utils.py）

为了简化实现，我们提供了常用的工具函数：

### 输入转换工具

```python
from utils import convert_observation_to_model_input

# 一键转换x2robot格式到模型输入格式
model_input = convert_observation_to_model_input(obs, self.control_mode)
# 返回: {"left": RGB_array, "front": RGB_array, "right": RGB_array, 
#        "state": 14D_array, "instruction": str}
```

### 输出转换工具

```python
from utils import convert_model_output_to_x2robot_format

# 一键转换模型输出到x2robot格式
result = convert_model_output_to_x2robot_format(
    actions,  # (action_horizon, 14) numpy array
    self.control_mode,
    self.action_horizon
)
```

### 子工具函数

- `to_numpy_1d()`: 将各种格式转换为1D numpy数组
- `decode_jpeg_any()`: 解码JPEG图像（支持base64字符串和bytes）
- `normalize_joints_to_7d()`: 标准化关节数据为7D格式

## 实现示例

### 示例1：使用工具函数（推荐，最简单）

```python
import torch
from utils import (
    convert_observation_to_model_input,
    convert_model_output_to_x2robot_format,
)

class MyPolicy(ModelPolicy):
    def load_model(self, checkpoint_path: str, device: str):
        model = YourModelClass()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model
    
    def convert_input(self, obs):
        # 使用工具函数，一行搞定
        return convert_observation_to_model_input(obs, self.control_mode)
    
    def convert_output(self, model_output):
        # 使用工具函数，一行搞定
        return convert_model_output_to_x2robot_format(
            model_output, self.control_mode, self.action_horizon
        )
    
    def run_inference(self, model_input):
        with torch.no_grad():
            output = self.model(model_input)
        return output.cpu().numpy()
```

### 示例2：自定义转换（高级用法）

```python
import torch
import numpy as np
from utils import to_numpy_1d, decode_jpeg_any

class MyPolicy(ModelPolicy):
    def load_model(self, checkpoint_path: str, device: str):
        # ... 同上 ...
        pass
    
    def convert_input(self, obs):
        # 自定义转换逻辑
        left_img = decode_jpeg_any(obs.get("CAMERA_LEFT"), name="CAMERA_LEFT")
        state = to_numpy_1d(obs.get("ACTION_FOLLOW1_POS"), name="FOLLOW1")
        
        # 根据你的模型需求组装输入
        return {
            "image": left_img,
            "state": state,
            "instruction": obs.get("instruction", "")
        }
    
    def convert_output(self, model_output):
        # 自定义输出转换
        actions = np.array(model_output)
        # ... 自定义处理逻辑 ...
        return result
    
    def run_inference(self, model_input):
        # ... 同上 ...
        pass
```

## 通信协议

### WebSocket协议

1. **连接建立**：客户端连接后，服务器立即发送metadata（msgpack编码）
2. **推理请求**：客户端发送observation（msgpack编码）
3. **推理响应**：服务器返回action（msgpack编码）
4. **错误处理**：如果出错，服务器发送文本错误消息

### Metadata格式

```python
{
    "control_mode": "joints" 或 "end_pose",
    "action_horizon": 50,
    "state_dim": 14,
    "state_dim_per_arm": 7,
    "protocol_version": "1.0"
}
```

## 命令行参数

```bash
python launch_server.py --help
```

主要参数：
- `--checkpoint`: Checkpoint路径
- `--control-mode`: 控制模式（joints/end_pose）
- `--action-horizon`: Action序列长度
- `--device`: 设备（cuda:0/cpu等）
- `--port`: 服务器端口（默认8000）
- `--host`: 服务器地址（默认0.0.0.0）
- `--log-level`: 日志级别（DEBUG/INFO/WARNING/ERROR）

## 测试连接

使用 `x2robot_client` 连接测试：

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

## 注意事项

1. **线程安全**：服务器使用锁保证推理的线程安全
2. **错误处理**：如果推理出错，服务器会发送错误消息给客户端
3. **消息大小**：服务器不限制消息大小，适合传输图像数据
4. **设备管理**：使用 `CUDA_VISIBLE_DEVICES` 环境变量控制GPU设备
5. **msgpack兼容性**：服务器使用 `use_bin_type=True` 和 `raw=False` 确保跨环境兼容性
6. **输入格式兼容**：工具函数自动处理多种输入格式（base64字符串、bytes、numpy数组等）
7. **opencv依赖**：只有在需要把 `CAMERA_*` 解码成 numpy RGB 时才需要安装 `opencv-python`

## 常见问题

### Q: 模型输出shape不匹配怎么办？

A: `convert_model_output_to_x2robot_format()` 会自动处理shape验证，如果horizon不匹配会给出警告但继续处理。

### Q: 如何调试输入输出？

A: 在 `convert_input()` 和 `convert_output()` 中添加日志：

```python
import logging
logger = logging.getLogger(__name__)

def convert_input(self, obs):
    logger.debug(f"Input keys: {list(obs.keys())}")
    logger.debug(f"FOLLOW1_POS shape: {np.array(obs['ACTION_FOLLOW1_POS']).shape}")
    # ...
```

### Q: 支持单臂机器人吗？

A: 当前设计针对双臂机器人（14D），单臂需要修改输出格式转换逻辑。

## 更多信息

- **快速上手**：查看 `QUICKSTART.md` 获取5分钟快速开始指南
- **架构设计**：查看 `ARCHITECTURE.md` 了解详细的设计原理和扩展点
- **工具函数**：查看 `utils.py` 了解所有可用的工具函数

## 参考

- `infer_new.py`: 简洁的配置风格
- `RoboChallengeInference/demo.py`: 模块化的Policy设计
- `x2robot_client`: 客户端实现和协议定义
