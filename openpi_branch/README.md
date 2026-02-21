# OpenPI X2Robot Server

完整的 OpenPI 模型推理服务器，直接兼容最新 `x2robot_client`（EX001 / Desktop / CX001 等）。

## 支持的模型类型

| 类型 | State 维度 | 说明 |
|------|-----------|------|
| **EE** | 14D | End-Effector 末端位姿控制 [x,y,z,r,p,y,gripper] × 2 |
| **Joints** | 14D | 关节角控制 [j1-j6,gripper] × 2 (delta actions) |
| **MobileManipulation** | 20D | 14D arm + head(2) + lift(1) + chassis(3) |

## 可用的 Config 名称

```
# EE 模式
pi0_x2robot_pick_banana_ee
pi05_x2robot_pick_banana_ee
pi0_x2robot_pick_banana_0104_ee
pi05_x2robot_pick_banana_0104_ee
pi0_x2robot_pick_banana_merged_ee
pi05_x2robot_pick_banana_merged_ee

# Joints 模式
pi05_pick_banana_joints

# Mobile Manipulation 模式 (20D)
pi0_put_clothes_in_hamper_mm
pi05_put_clothes_in_hamper_mm
pi0_take_and_set_tableware_mm
pi05_take_and_set_tableware_mm
```

## 快速启动

### 方式一：完整独立服务器（推荐）

```bash
# EE 模式
python openpi_branch/serve_openpi.py \
    --config pi0_x2robot_pick_banana_merged_ee \
    --checkpoint-dir /path/to/checkpoints/step_10000 \
    --default-prompt "pick up the banana" \
    --port 8000 \
    --warmup

# Joints 模式
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

### 方式二：通过 launch_openpi_server.py（使用通用 WebSocket 框架）

```bash
python openpi_branch/launch_openpi_server.py \
    --config pi0_x2robot_pick_banana_merged_ee \
    --checkpoint-dir /path/to/checkpoints/step_10000 \
    --port 8000
```

## 完整参数说明

```
python openpi_branch/serve_openpi.py --help

必需参数:
  --config              OpenPI training config 名称
  --checkpoint-dir      模型 checkpoint 目录路径

可选参数:
  --default-prompt      客户端未提供 instruction 时的默认 prompt
  --host                服务器地址 (默认 0.0.0.0)
  --port                服务器端口 (默认 8000)
  --model-device        推理设备 (默认 cuda)
  --log-dir             推理日志保存目录 (默认不保存)
  --heartbeat-sec       心跳打印间隔秒数 (默认 30, 0=禁用)
  --action-start-ratio  轨迹起始截取比例 (默认 0.0)
  --action-end-ratio    轨迹结束截取比例 (默认 0.8, 丢弃后 20% 噪声)
  --warmup              加载后运行一次 dummy inference 预热 JIT
```

## 通信协议

### 输入 (Client -> Server)

服务器同时支持新格式（嵌套）和旧格式（扁平大写 key）：

```python
# 新格式 (EX001 / Desktop)
{
    "state": {
        "follow1_pos": np.float32[7],    # [x,y,z,r,p,y,gripper]
        "follow2_pos": np.float32[7],
        "follow1_joints": np.float32[N],  # 可选
        "follow2_joints": np.float32[N],  # 可选
        "head_pos": np.float32[2],        # MM 模式
        "lift": np.float32[1],            # MM 模式
        "velocity_decomposed_odom": np.float32[3],  # MM 模式
    },
    "views": {
        "camera_left": "base64_jpeg",
        "camera_front": "base64_jpeg",
        "camera_right": "base64_jpeg",
    },
    "instruction": np.array(["pick up the banana"]),
}

# 旧格式 (CX001 等)
{
    "CAMERA_LEFT": "base64_jpeg",
    "CAMERA_FRONT": "base64_jpeg",
    "CAMERA_RIGHT": "base64_jpeg",
    "ACTION_FOLLOW1_POS": [7D],
    "ACTION_FOLLOW2_POS": [7D],
    "instruction": "pick up the banana",
}
```

### 输出 (Server -> Client)

```python
# EE 模式
{"follow1_pos": np.float32[T, 7], "follow2_pos": np.float32[T, 7]}

# Joints 模式
{"follow1_joints": np.float32[T, 7], "follow2_joints": np.float32[T, 7],
 "follow1_pos": np.float32[T, 7], "follow2_pos": np.float32[T, 7]}

# MM 模式 (20D)
{"follow1_pos": ..., "follow2_pos": ...,
 "head_pos": np.float32[T, 2], "lift": np.float32[T, 1],
 "velocity_decomposed": np.float32[T, 3]}
```

## 架构说明

```
serve_openpi.py
├── Cross-compatible msgpack+numpy   # 兼容 msgpack-numpy 和 openpi_client 两种编码格式
├── OpenPIServingPolicy              # 核心适配层
│   ├── _extract_state()             # 从 client obs 提取 state (14D/20D)
│   ├── _extract_images()            # 解码图像 + 映射到 OpenPI key
│   ├── _extract_prompt()            # 提取文本指令
│   ├── _convert_output()            # Prepend state + 截取 + 拆分
│   └── infer()                      # 完整推理流程
├── InferenceLogger                  # 记录 state/action 并生成可视化
└── WebSocket Server                 # 异步处理客户端连接
```

### 输出处理策略

1. **Prepend State**: 在 action 序列前插入当前 state，确保轨迹从机器人当前位置平滑起始
2. **Action Slicing**: 根据 `action_start_ratio` / `action_end_ratio` 截取有效区间（默认保留前 80%，丢弃尾部噪声预测）
3. **Per-Component Split**: 将统一的 action 向量拆分为各组件（双臂 + 头 + 升降 + 底盘）

## 推理日志

启用 `--log-dir` 后，服务器会在每次推理时记录 state 和 action chunk，关闭时自动保存：

- `{config_name}.npz` — 原始数据（states, actions, prompts, timestamps）
- `{config_name}.png` — 可视化图表（每维度 state vs action 时序曲线）
