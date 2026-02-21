# 通用模型服务器模板

适配最新 `x2robot_client`（EX001 / Desktop 等）的通用模型服务器模板。参赛者只需实现模型相关部分，WebSocket 通信与协议处理已封装完毕。

## 整体架构

```
maniparena_model_server/
├── websocket_server.py    # WebSocket服务器核心（不需要修改）
├── policy_base.py         # Policy基类定义（不需要修改）
├── my_policy.py           # 参赛者的Policy实现（需要修改）
├── utils.py               # 工具函数（可选使用）
├── launch_server.py       # 服务器启动文件（不需要修改）
├── requirements.txt       # 依赖列表
├── README.md              # 本文档
├── ARCHITECTURE.md        # 架构设计说明
└── openpi_branch/         # OpenPI模型适配
    ├── openpi_policy.py
    └── launch_openpi_server.py
```

### 数据流

```
x2robot_client (客户端)
    ↓ WebSocket连接
    ↓ 服务器发送metadata
WebSocket服务器 (websocket_server.py)
    ↓ 接收observation (msgpack)
    ↓ 调用 policy.infer(obs)
Policy (my_policy.py)
    ↓ convert_input()  — 输入格式转换
    ↓ run_inference()  — 模型推理
    ↓ convert_output() — 输出格式转换
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

如果需要解码图像为 numpy RGB：

```bash
pip install opencv-python
```

### 2. 修改配置

编辑 `my_policy.py`，修改顶部默认值：

```python
DEFAULT_CHECKPOINT_PATH = "/path/to/your/checkpoint"
DEFAULT_CONTROL_MODE = "joints"  # 或 "end_pose"
DEFAULT_ACTION_HORIZON = 50
DEFAULT_DEVICE = "cuda:0"
```

### 3. 实现方法

编辑 `my_policy.py`，实现：

1. **`load_model()`** — 加载模型（必须）
2. **`convert_input()`** — 输入格式转换（可选，推荐使用工具函数）
3. **`convert_output()`** — 输出格式转换（必须，推荐使用工具函数）
4. **`run_inference()`** — 模型推理（必须）

### 4. 启动服务器

```bash
python launch_server.py \
    --checkpoint /path/to/your/checkpoint \
    --control-mode joints \
    --port 8000
```

## 输入输出格式（v2 — 与最新 x2robot_client 对齐）

### 输入格式（来自 x2robot_client）

最新的 EX001 / Desktop 客户端使用**嵌套字典**格式：

```python
{
    "state": {
        # --- 双臂末端位姿 (7D: x, y, z, roll, pitch, yaw, gripper) ---
        "follow1_pos": np.array([...], dtype=np.float32),   # (7,) 左臂
        "follow2_pos": np.array([...], dtype=np.float32),   # (7,) 右臂
        # --- 关节角度（可选） ---
        "follow1_joints": np.array([...], dtype=np.float32),
        "follow2_joints": np.array([...], dtype=np.float32),
        # --- 电流/力矩反馈 ---
        "follow1_joints_cur": np.array([...], dtype=np.float32),
        "follow2_joints_cur": np.array([...], dtype=np.float32),
        # --- 关节速度（可选） ---
        "follow1_joints_dev": np.array([...], dtype=np.float32),
        "follow2_joints_dev": np.array([...], dtype=np.float32),
        # --- 头部 / 升降 / 底盘（全身机器人） ---
        "head_pos": np.array([yaw, pitch], dtype=np.float32),         # (2,)
        "lift": np.array([height], dtype=np.float32),                 # (1,)
        "car_pose": np.array([x, y, theta], dtype=np.float32),       # (3,)
        "velocity_decomposed": np.array([vx, vy, omega], dtype=np.float32),      # (3,)
        "velocity_decomposed_odom": np.array([vx, vy, omega], dtype=np.float32), # (3,)
    },
    "views": {
        "camera_left": "base64_jpeg_string",
        "camera_front": "base64_jpeg_string",
        "camera_right": "base64_jpeg_string",
    },
    "instruction": np.array(["Pick up the cup."], dtype=np.object_),
}
```

> 工具函数 `convert_observation_to_model_input()` 同时兼容旧的扁平大写格式（CAMERA_LEFT / ACTION_FOLLOW1_POS 等）。

### 输出格式（返回给 x2robot_client）

使用**小写 key**，与 `state` 输入字段名称对应：

```python
{
    # --- 双臂轨迹 (T, 7) ---
    "follow1_pos": [[x,y,z,r,p,y,gripper], ...],   # end_pose模式
    "follow2_pos": [[x,y,z,r,p,y,gripper], ...],
    # --- 或 joints 模式 ---
    "follow1_joints": [[j1,...,j6,gripper], ...],
    "follow2_joints": [[j1,...,j6,gripper], ...],

    # --- 全身控制（可选） ---
    "head_pos": [[yaw, pitch], ...],                # (T, 2)
    "lift": [[height], ...],                        # (T, 1)
    "velocity_decomposed": [[vx, vy, omega], ...],  # (T, 3)
    "car_pose_odom": [[x, y, theta], ...],          # (T, 3)

    # --- 文本输出（可选） ---
    "model_output_text": "optional model text response",
}
```

## 工具函数（utils.py）

### 输入转换

```python
from utils import convert_observation_to_model_input

model_input = convert_observation_to_model_input(obs, control_mode)
# 返回: {
#   "left": RGB_array, "front": RGB_array, "right": RGB_array,
#   "state": 14D_array,
#   "instruction": str,
#   "state/head_pos": ..., "state/lift": ..., ...  (新格式额外字段)
# }
```

### 输出转换

```python
from utils import convert_model_output_to_x2robot_format

result = convert_model_output_to_x2robot_format(
    actions,           # (T, 14) numpy array
    control_mode,
    action_horizon,
    head_actions=...,  # (T, 2) 可选
    lift_actions=...,  # (T, 1) 可选
    velocity_actions=...,  # (T, 3) 可选
)
```

### 旧格式兼容

如需为 CX001 等旧客户端输出大写 key 格式：

```python
from utils import convert_model_output_to_legacy_format

result = convert_model_output_to_legacy_format(actions, control_mode, action_horizon)
# 返回: {"FOLLOW1_POS": ..., "FOLLOW2_POS": ..., ...}
```

## 通信协议

### WebSocket协议

1. **连接建立**：客户端连接后，服务器立即发送 metadata（msgpack 编码）
2. **推理请求**：客户端发送 observation（msgpack 编码）
3. **推理响应**：服务器返回 action（msgpack 编码）
4. **错误处理**：如果出错，服务器发送文本错误消息

### Metadata格式

```python
{
    "control_mode": "joints" or "end_pose",
    "action_horizon": 50,
    "state_dim": 14,
    "state_dim_per_arm": 7,
    "protocol_version": "2.0"
}
```

## 命令行参数

```bash
python launch_server.py --help
```

主要参数：
- `--checkpoint`: Checkpoint路径
- `--control-mode`: 控制模式（joints / end_pose）
- `--action-horizon`: Action序列长度
- `--device`: 设备（cuda:0 / cpu 等）
- `--port`: 服务器端口（默认 8000）
- `--host`: 服务器地址（默认 0.0.0.0）
- `--log-level`: 日志级别

## 测试连接

```python
import numpy as np
from x2robot_client.inference_client import RobotClient

client = RobotClient(uri="ws://localhost:8000")
client.connect_sync()

obs = {
    "state": {
        "follow1_pos": np.zeros(7, dtype=np.float32),
        "follow2_pos": np.zeros(7, dtype=np.float32),
    },
    "views": {
        "camera_left": None,
        "camera_front": None,
        "camera_right": None,
    },
    "instruction": np.array(["Pick up the cup."], dtype=np.object_),
}

result = client.predict_sync(obs)
print(result.keys())
# dict_keys(['follow1_pos', 'follow2_pos', ...])
```

## 注意事项

1. **向后兼容**：`convert_observation_to_model_input()` 自动检测新旧格式
2. **线程安全**：服务器使用锁保证推理的线程安全
3. **消息大小**：不限制，适合传输图像数据
4. **设备管理**：使用 `CUDA_VISIBLE_DEVICES` 环境变量控制 GPU
5. **opencv 依赖**：仅在解码图像为 numpy RGB 时需要

## 更多信息

- **快速上手**：查看 `QUICKSTART.md`
- **架构设计**：查看 `ARCHITECTURE.md`
- **API 定义**：查看 `x2robot_client/API.md`
