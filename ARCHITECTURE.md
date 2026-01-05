# 架构设计说明

> 本文档面向希望深入了解系统设计的开发者。普通用户可以直接查看 [README.md](README.md) 和 [QUICKSTART.md](QUICKSTART.md)。

## 整体逻辑

### 1. 模块划分

```
┌─────────────────────────────────────────┐
│         x2robot_client (客户端)          │
│  - 收集传感器数据（图像、状态）           │
│  - 通过WebSocket发送observation         │
│  - 接收action并执行                     │
└──────────────┬──────────────────────────┘
               │ WebSocket (msgpack)
               ↓
┌─────────────────────────────────────────┐
│    websocket_server.py (服务器核心)     │
│  - 处理WebSocket连接                    │
│  - 发送metadata                         │
│  - 接收observation，调用policy.infer() │
│  - 发送action结果                       │
└──────────────┬──────────────────────────┘
               │ 调用
               ↓
┌─────────────────────────────────────────┐
│      policy_base.py (Policy基类)        │
│  - 定义接口：load_model, convert_*     │
│  - 实现标准推理流程：                    │
│    convert_input → run_inference →      │
│    convert_output                       │
└──────────────┬──────────────────────────┘
               │ 继承
               ↓
┌─────────────────────────────────────────┐
│      my_policy.py (参赛者实现)           │
│  - load_model(): 加载模型               │
│  - convert_input(): 输入转换（可选）     │
│  - convert_output(): 输出转换           │
│  - run_inference(): 模型推理            │
└─────────────────────────────────────────┘
```

### 2. 数据流

```
客户端发送:
{
    "CAMERA_LEFT": "base64_jpeg",
    "CAMERA_FRONT": "base64_jpeg",
    "CAMERA_RIGHT": "base64_jpeg",
    "ACTION_FOLLOW1_POS": [7D数组],
    "ACTION_FOLLOW2_POS": [7D数组],
    "instruction": "文本指令"
}
         ↓
    convert_input()
         ↓
    模型输入格式
         ↓
    run_inference()
         ↓
    模型输出（任意格式）
         ↓
    convert_output()
         ↓
服务器返回:
{
    "FOLLOW1_JOINTS": [[7D], [7D], ...],  # joints模式
    "FOLLOW2_JOINTS": [[7D], [7D], ...],
    "FOLLOW1_POS": [],
    "FOLLOW2_POS": []
}
或
{
    "FOLLOW1_POS": [[7D], [7D], ...],     # end_pose模式
    "FOLLOW2_POS": [[7D], [7D], ...],
    "FOLLOW1_JOINTS": [],
    "FOLLOW2_JOINTS": []
}
```

### 3. 设计原则

#### 3.1 模块化
- **websocket_server.py**: 只负责WebSocket通信，不关心模型细节
- **policy_base.py**: 定义标准接口，实现通用流程
- **my_policy.py**: 参赛者只需实现模型相关部分

#### 3.2 简洁性
- 参考 `infer_new.py` 的风格：配置在顶部，逻辑清晰
- 单文件实现：每个模块职责单一，易于理解

#### 3.3 可扩展性
- Policy基类可以扩展（如添加reset、metadata等方法）
- 输入输出转换可以自定义
- 支持不同的控制模式（joints/end_pose）

#### 3.4 独立性
- 不依赖任何特定的模型框架（PyTorch、TensorFlow等）
- 只依赖标准库和WebSocket库
- 可以适配任何模型

### 4. 关键设计点

#### 4.1 为什么使用基类？

```python
class ModelPolicy(ABC):
    def infer(self, obs):
        # 标准流程已经实现
        model_input = self.convert_input(obs)
        model_output = self.run_inference(model_input)
        return self.convert_output(model_output)
```

**好处**：
- 参赛者不需要关心推理流程
- 只需要实现4个方法
- 流程统一，易于调试

#### 4.2 为什么分离输入输出转换？

```python
def convert_input(self, obs):
    # 将x2robot格式转换为模型格式
    pass

def convert_output(self, model_output):
    # 将模型格式转换为x2robot格式
    pass
```

**好处**：
- 模型可以有自己的输入输出格式
- 转换逻辑独立，易于测试
- 可以复用转换函数

#### 4.3 为什么使用WebSocket？

- **实时性**：适合机器人控制的实时通信
- **双向通信**：可以发送metadata
- **二进制支持**：msgpack编码，高效传输图像数据

#### 4.4 为什么使用msgpack？

- **高效**：比JSON更紧凑，比pickle更安全
- **跨语言**：支持多种编程语言
- **numpy支持**：msgpack-numpy可以直接序列化numpy数组

### 5. 扩展点

如果需要扩展功能，可以在以下位置修改：

1. **添加新的控制模式**：
   - 在 `policy_base.py` 的 `metadata` 中添加
   - 在 `convert_output()` 中处理

2. **添加新的输入字段**：
   - 在 `convert_input()` 中处理
   - 在 `my_policy.py` 中实现

3. **添加模型状态管理**：
   - 在 `policy_base.py` 中添加 `reset()` 方法
   - 在 `my_policy.py` 中实现

4. **添加错误恢复**：
   - 在 `websocket_server.py` 的 `_handle_client()` 中处理

### 6. 与参考实现的对比

#### 6.1 与 `infer_new.py` 的相似点
- 配置在顶部，清晰明了
- 主逻辑简洁，易于理解
- 使用配置对象管理参数

#### 6.2 与 `RoboChallengeInference/demo.py` 的相似点
- Policy类设计
- 模块化结构
- 清晰的接口定义

#### 6.3 改进点
- **更简洁**：单文件实现，减少文件数量
- **更通用**：不依赖特定框架
- **更清晰**：数据流明确，易于调试

## 总结

这个设计实现了：
1. ✅ **模块化**：职责分离，易于维护
2. ✅ **简洁性**：参赛者只需实现4个方法
3. ✅ **独立性**：不依赖特定框架
4. ✅ **可扩展性**：易于添加新功能
5. ✅ **清晰性**：整体逻辑一目了然

## 相关文档

- [README.md](README.md) - 完整的使用文档和API说明
- [QUICKSTART.md](QUICKSTART.md) - 5分钟快速开始指南

