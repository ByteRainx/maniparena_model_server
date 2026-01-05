"""
参赛者的Policy实现示例

这是参赛者需要修改的主要文件。
参考infer_new.py的简洁风格，配置在顶部，实现部分清晰明了。

使用工具函数：
- 可以直接使用 utils.convert_observation_to_model_input() 进行输入转换
- 可以直接使用 utils.convert_model_output_to_x2robot_format() 进行输出转换
- 也可以参考这些函数的实现，自己定制转换逻辑
"""

import numpy as np
from typing import Dict, Any
import logging

from policy_base import ModelPolicy
# 导入工具函数（可选，如果不需要可以直接删除这行）
from utils import (
    convert_observation_to_model_input,
    convert_model_output_to_x2robot_format,
)

logger = logging.getLogger(__name__)


# ============ 配置部分 - 参赛者需要修改 ============

# 这些配置可以通过命令行参数覆盖，但也可以在这里设置默认值
DEFAULT_CHECKPOINT_PATH = "/path/to/your/checkpoint"
DEFAULT_CONTROL_MODE = "joints"  # 或 "end_pose"
DEFAULT_ACTION_HORIZON = 50
DEFAULT_DEVICE = "cuda:0"


# ============ Policy实现 - 参赛者需要实现以下方法 ============

class MyPolicy(ModelPolicy):
    """
    参赛者的Policy实现
    
    只需要实现4个方法：
    1. load_model() - 加载模型
    2. convert_input() - 输入格式转换（可选，如果需要）
    3. convert_output() - 输出格式转换
    4. run_inference() - 模型推理
    """
    
    def load_model(self, checkpoint_path: str, device: str) -> Any:
        """
        加载你的模型
        
        Args:
            checkpoint_path: checkpoint路径
            device: 设备 ("cuda:0" 或 "cpu")
        
        Returns:
            你的模型对象（可以是任何类型，只要能调用推理方法）
        
        注意：
            - 这个方法可以处理任何复杂的初始化逻辑
            - 对于简单模型，直接加载即可
            - 对于复杂模型（如wallx），可以在这里创建配置对象、加载processor等
            - 可以使用self.checkpoint_path、self.device等属性
        
        示例1：简单PyTorch模型
        ```python
        import torch
        model = YourModelClass()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model
        ```
        
        示例2：复杂模型（wallx，需要配置对象）
        ```python
        from wall_x.infer.infer_config import InferConfig
        from wall_x.infer.model_wrapper import WallxModelWrapper
        
        # 在方法内部创建配置对象
        config = InferConfig(
            checkpoint_path=checkpoint_path,
            model_device=device,
            # ... 其他配置
        )
        wrapper = WallxModelWrapper(config)
        return wrapper
        ```
        
        示例3：TensorFlow模型
        ```python
        import tensorflow as tf
        model = tf.keras.models.load_model(checkpoint_path)
        return model
        ```
        """
        # TODO: 在这里加载你的模型
        # 可以处理任何复杂的初始化逻辑，包括：
        # - 创建配置对象
        # - 加载processor、tokenizer等
        # - 初始化模型wrapper
        # - 等等
        
        raise NotImplementedError("请实现 load_model() 方法")
    
    def convert_input(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        输入格式转换（可选）
        
        如果你的模型需要特定的输入格式，在这里转换。
        如果不需要转换，可以删除这个方法（会使用基类的默认实现）。
        
        Args:
            obs: x2robot_client格式的observation
                - CAMERA_LEFT/FRONT/RIGHT: Base64编码的JPEG图像字符串或bytes
                - ACTION_FOLLOW1_POS/FOLLOW2_POS: 7D或8D数组（numpy array、list或dict）
                - instruction: 可选的文本指令
        
        Returns:
            你的模型期望格式的observation字典
        
        推荐用法：
        1. 如果模型输入格式标准（图像+状态+指令），可以直接使用工具函数：
           return convert_observation_to_model_input(obs, self.control_mode)
        
        2. 如果需要自定义格式，可以参考工具函数的实现，或使用其中的子函数：
           from utils import to_numpy_1d, decode_jpeg_any, normalize_joints_to_7d
        """
        # ========== 方式1：使用工具函数（推荐，最简单）==========
        # 如果你的模型输入格式是标准的（图像+状态+指令），可以直接使用：
        # return convert_observation_to_model_input(obs, self.control_mode)
        
        # ========== 方式2：自定义转换（如果需要特殊格式）==========
        # 参考工具函数的实现，或使用其中的子函数：
        # from utils import to_numpy_1d, decode_jpeg_any, normalize_joints_to_7d
        #
        # # 解码图像
        # left_img = decode_jpeg_any(obs.get("CAMERA_LEFT"), name="CAMERA_LEFT")
        # front_img = decode_jpeg_any(obs.get("CAMERA_FRONT"), name="CAMERA_FRONT")
        # right_img = decode_jpeg_any(obs.get("CAMERA_RIGHT"), name="CAMERA_RIGHT")
        #
        # # 读取状态
        # follow1 = to_numpy_1d(obs.get("ACTION_FOLLOW1_POS"), name="ACTION_FOLLOW1_POS")
        # follow2 = to_numpy_1d(obs.get("ACTION_FOLLOW2_POS"), name="ACTION_FOLLOW2_POS")
        #
        # # 标准化（joints模式）
        # if self.control_mode == "joints":
        #     follow1 = normalize_joints_to_7d(follow1, self.control_mode)
        #     follow2 = normalize_joints_to_7d(follow2, self.control_mode)
        #
        # # 组装模型输入（根据你的模型需求调整）
        # model_input = {
        #     "image": left_img,  # 或组合多张图像
        #     "state": np.concatenate([follow1[:7], follow2[:7]]),
        #     "instruction": obs.get("instruction", ""),
        # }
        # return model_input
        
        # ========== 方式3：不需要转换（直接返回）==========
        # 如果你的模型可以直接接受x2robot格式，直接返回：
        return obs
    
    def convert_output(self, model_output: Any) -> Dict[str, Any]:
        """
        输出格式转换（必须实现）
        
        将你的模型输出转换为x2robot_client期望的格式
        
        Args:
            model_output: 你的模型推理结果
                通常是 (action_horizon, 14) 的numpy数组
                14D格式: [left(6), left_gripper(1), right(6), right_gripper(1)]
        
        Returns:
            x2robot_client格式的输出字典
        
        推荐用法：
        如果模型输出是标准的 (action_horizon, 14) 格式，可以直接使用工具函数：
        return convert_model_output_to_x2robot_format(
            model_output, self.control_mode, self.action_horizon
        )
        """
        # ========== 方式1：使用工具函数（推荐，最简单）==========
        # 如果模型输出是标准的 (action_horizon, 14) numpy数组：
        # return convert_model_output_to_x2robot_format(
        #     model_output, self.control_mode, self.action_horizon
        # )
        
        # ========== 方式2：自定义转换（如果需要特殊处理）==========
        # 确保是numpy数组
        # actions = np.array(model_output) if not isinstance(model_output, np.ndarray) else model_output
        #
        # # 验证shape
        # if actions.ndim != 2 or actions.shape[1] != 14:
        #     raise ValueError(f"Expected (horizon, 14), got {actions.shape}")
        #
        # # 分离左右臂
        # left_actions = np.concatenate([actions[:, :6], actions[:, 6:7]], axis=1)
        # right_actions = np.concatenate([actions[:, 7:13], actions[:, 13:14]], axis=1)
        #
        # if self.control_mode == "joints":
        #     return {
        #         "FOLLOW1_JOINTS": left_actions.tolist(),
        #         "FOLLOW2_JOINTS": right_actions.tolist(),
        #         "FOLLOW1_POS": [],
        #         "FOLLOW2_POS": [],
        #     }
        # else:  # end_pose
        #     return {
        #         "FOLLOW1_POS": left_actions.tolist(),
        #         "FOLLOW2_POS": right_actions.tolist(),
        #         "FOLLOW1_JOINTS": [],
        #         "FOLLOW2_JOINTS": [],
        #     }
        
        raise NotImplementedError("请实现 convert_output() 方法")
    
    def run_inference(self, model_input: Dict[str, Any]) -> Any:
        """
        模型推理（必须实现）
        
        Args:
            model_input: 转换后的模型输入格式
        
        Returns:
            你的模型输出（可以是任何格式，会在convert_output中转换）
        """
        # TODO: 在这里调用你的模型推理
        # 示例（PyTorch）：
        # import torch
        # with torch.no_grad():
        #     output = self.model(model_input)
        # return output.cpu().numpy()
        
        # 示例（TensorFlow）：
        # output = self.model.predict(model_input)
        # return output
        
        raise NotImplementedError("请实现 run_inference() 方法")
