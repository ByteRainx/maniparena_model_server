# 通用模型服务器模板

适配 `x2robot_client`（Desktop / CX001）的通用模型服务器模板。参赛者只需实现模型相关部分，WebSocket 通信与协议处理已封装完毕。

## 整体架构

```
maniparena_model_server/
├── websocket_server.py    # WebSocket服务器核心（不需要修改）
├── policy_base.py         # Policy基类定义（不需要修改）
├── my_policy.py           # 参赛者的Policy实现（需要修改）
├── utils.py               # 工具函数（可选使用）
├── launch_server.py       # 服务器启动文件（不需要修改）
├── requirements.txt       # 依赖列表
└── examples/              # 参考实现
    └── pytorch_policy_example.py
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

## 两种任务类型

| | Desktop (14D) | CX001 Mobile Manipulation (20D) |
|---|---|---|
| 机器人 | EX001 桌面双臂 | CX001 移动操作 |
| 动作维度 | 14（左臂7D + 右臂7D） | 20（14D双臂 + 2D头部 + 1D升降 + 3D底盘） |
| 输出 Key | **小写** | **大写** |
| 工具函数 | `convert_model_output_to_x2robot_format()` | `convert_model_output_to_legacy_format()` |

### 输出 Key 对照表

| 字段 | Desktop Key | CX001 Key | 每步维度 |
|------|-----------|-----------|---------|
| 左臂 | `follow1_pos` | `FOLLOW1_POS` | `[7]` |
| 右臂 | `follow2_pos` | `FOLLOW2_POS` | `[7]` |
| 头部 | — | `HEAD_POS` | `[2]` |
| 升降 | — | `LIFT_OUT` | `[1]` |
| 底盘 | — | `CAR_POSE_OUT` | `[3]` |

## ⚠ 关键注意事项

### 1. 所有输出值必须调用 `.tolist()`

Desktop 客户端在插值时会执行：

```python
arm1_actions = [self.last_arm_l_pos] + arm1_actions
```

如果 `arm1_actions` 是 **numpy 数组**，Python 的 `+` 会触发元素级广播（broadcasting）而非列表拼接，**轨迹数据被静默破坏**。

**正确做法：**

```python
def convert_output(self, model_output):
    actions = np.array(model_output)        # shape (T, 14)
    return {
        "follow1_pos": actions[:, :7].tolist(),   # .tolist() 是必须的
        "follow2_pos": actions[:, 7:14].tolist(),
    }
```

### 2. Desktop 与 CX001 输出 Key 大小写不同

| 客户端 | Key 格式 | 使用错误大小写的后果 |
|--------|---------|---------|
| Desktop | 小写 `follow1_pos` | 大写 → `.get()` 返回 `None` → 不执行动作 |
| CX001 | 大写 `FOLLOW1_POS` | 小写 → `.get()` 返回 `[]` → 机器人不动 |

### 3. 值格式必须是 `List[List[float]]`

```python
# ✅ 正确 — Python 嵌套列表
"follow1_pos": [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], ...]

# ❌ 错误 — numpy 数组（msgpack 序列化后客户端可能无法正确解析）
"follow1_pos": np.array([[0.1, 0.2, ...], ...])

# ❌ 错误 — 一维列表（缺少时间维度）
"follow1_pos": [0.1, 0.2, 0.3, ...]
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 实现 `my_policy.py`

编辑 `my_policy.py`，实现：

1. **`load_model()`** — 加载模型（必须）
2. **`convert_input()`** — 输入格式转换（可选，推荐使用工具函数）
3. **`run_inference()`** — 模型推理（必须）
4. **`convert_output()`** — 输出格式转换（必须，**注意 `.tolist()`**）

### 3. 启动服务器

```bash
python launch_server.py \
    --checkpoint /path/to/your/checkpoint \
    --control-mode end_pose \
    --port 8000
```

<<<<<<< HEAD
## 输入格式
=======
## Self-check Before Submission

Use the built-in mock tools in `tools/` to validate your server before submission.

### 1) Ping / handshake check

Checks:
- WebSocket connection is reachable
- First server frame is msgpack metadata
- Required metadata fields exist (`control_mode`, `action_horizon`, `state_dim`)

```bash
python tools/mock_ping.py --uri ws://127.0.0.1:8000
```

### 2) Request/response schema check

Checks:
- Desktop schema (lowercase keys, list-of-lists trajectories)
- CX001 schema (UPPERCASE keys, optional MM extras)
- Common mistakes: wrong key casing, numpy return (missing `.tolist()`), missing time dimension

```bash
# Check both Desktop and CX001
python tools/mock_schema_check.py --uri ws://127.0.0.1:8000 --mode both

# If you want strict CX001 MM fields (HEAD_POS/LIFT_OUT/CAR_POSE_OUT)
python tools/mock_schema_check.py --uri ws://127.0.0.1:8000 --mode cx001 --require-mm-extras
```

### 3) Open-loop evaluation

Runs offline samples through your server, then saves:
- `*.npz` with `pred`/`gt` arrays (default)
- `*.jpg` plots only when explicitly enabled

```bash
# Default: save npz only (plotting disabled)
python tools/mock_openloop_eval.py \
    --uri ws://127.0.0.1:8000 \
    --data-dir /path/to/offline_dataset \
    --save-dir /path/to/openloop_outputs \
    --sample-limit 3

# Optional: enable plotting
python tools/mock_openloop_eval.py \
    --uri ws://127.0.0.1:8000 \
    --data-dir /path/to/offline_dataset \
    --save-dir /path/to/openloop_outputs \
    --sample-limit 3 \
    --enable-plots
```

## Input Format
>>>>>>> cf00520 (feat: add submission self-check mock tools)

### Desktop 客户端（嵌套字典格式）

```python
{
    "state": {
        "follow1_pos": [x, y, z, roll, pitch, yaw, gripper],   # 7D 左臂
        "follow2_pos": [x, y, z, roll, pitch, yaw, gripper],   # 7D 右臂
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

### CX001ClientROS2 客户端（扁平字典，大写 key）

> 比赛移动操作赛道使用 `CX001ClientROS2`。注意：图像是 **numpy RGB 数组**（非 base64），状态是带 `ACTION_` 前缀的大写 key。

```python
{
    "CAMERA_LEFT": np.ndarray (H, W, 3),                              # numpy RGB
    "CAMERA_FRONT": np.ndarray (H, W, 3),
    "CAMERA_RIGHT": np.ndarray (H, W, 3),
    "ACTION_FOLLOW1_POS": np.array([7D], dtype=np.float32),           # 左臂 7D
    "ACTION_FOLLOW2_POS": np.array([7D], dtype=np.float32),           # 右臂 7D
    "ACTION_FOLLOW1_JOINTS_CUR": np.array([...], dtype=np.float32),   # 电流
    "ACTION_FOLLOW2_JOINTS_CUR": np.array([...], dtype=np.float32),
    "CAR_POSE": np.array([x, y, theta], dtype=np.float32),            # 底盘 (3,)
    "LIFT": np.array([height], dtype=np.float32),                      # 升降 (1,)
    "HEAD_POS": np.array([yaw, pitch], dtype=np.float32),              # 头部 (2,)
    "INSTRUCTION": np.array(["task description"], dtype=np.object_),
}
```

> 工具函数 `convert_observation_to_model_input()` 自动兼容 Desktop（嵌套）和 CX001（扁平）两种格式。
> 图像字段：Desktop 传 base64 JPEG 字符串，CX001 传 numpy RGB 数组，`decode_images=True` 时均能正确处理。

## 输出格式

### Desktop 输出（14D，小写 key）

```python
{
    "follow1_pos": [[x,y,z,r,p,y,gripper], ...],   # List[List[float]], (T, 7)
    "follow2_pos": [[x,y,z,r,p,y,gripper], ...],
}
```

### CX001 输出（20D，大写 key）

```python
{
    "FOLLOW1_POS": [[x,y,z,r,p,y,gripper], ...],   # List[List[float]], (T, 7)
    "FOLLOW2_POS": [[x,y,z,r,p,y,gripper], ...],
    "HEAD_POS": [[yaw, pitch], ...],                 # (T, 2)
    "LIFT_OUT": [[height], ...],                      # (T, 1)
    "CAR_POSE_OUT": [[x, y, theta], ...],            # (T, 3)
}
```

## 工具函数（utils.py）

### 输入转换

```python
from utils import convert_observation_to_model_input

model_input = convert_observation_to_model_input(obs, control_mode)
# 返回: {
#   "left": RGB_array, "front": RGB_array, "right": RGB_array,
#   "state": np.array(14,),
#   "instruction": str,
#   "state/head_pos": ..., "state/lift": ..., ...  (额外字段)
# }
```

### Desktop 输出转换（14D）

```python
from utils import convert_model_output_to_x2robot_format

result = convert_model_output_to_x2robot_format(
    actions,           # (T, 14) numpy array
    control_mode,
    action_horizon,
)
# 返回: {"follow1_pos": [[...], ...], "follow2_pos": [[...], ...]}
```

### CX001 输出转换（20D）

```python
from utils import convert_model_output_to_legacy_format

result = convert_model_output_to_legacy_format(
    actions,           # (T, 14) numpy array
    control_mode,
    action_horizon,
    head_actions=...,       # (T, 2) 可选
    lift_actions=...,       # (T, 1) 可选
    car_pose_actions=...,   # (T, 3) 可选
)
# 返回: {"FOLLOW1_POS": ..., "FOLLOW2_POS": ..., "HEAD_POS": ..., "LIFT_OUT": ..., "CAR_POSE_OUT": ...}
```

## 通信协议

1. **连接建立**：客户端连接后，服务器立即发送 metadata（msgpack 编码）
2. **推理循环**：客户端发送 observation → 服务器返回 action（均为 msgpack 编码）

### Metadata 格式

```python
{
    "control_mode": "end_pose",   # "joints" 或 "end_pose"
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

- `--checkpoint`: Checkpoint 路径
- `--control-mode`: 控制模式（joints / end_pose）
- `--action-horizon`: Action 序列长度
- `--device`: 设备（cuda:0 / cpu）
- `--port`: 服务器端口（默认 8000）
- `--host`: 服务器地址（默认 0.0.0.0）
