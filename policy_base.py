"""
Policy基类定义

定义了参赛者需要实现的接口，以及通用的适配逻辑。
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ModelPolicy(ABC):
    """
    Policy基类
    
    参赛者需要继承这个类并实现以下方法：
    1. load_model() - 加载模型
    2. convert_input() - 输入格式转换（可选）
    3. convert_output() - 输出格式转换
    4. run_inference() - 模型推理
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        control_mode: str,
        action_horizon: int,
        device: str = "cuda:0",
    ):
        """
        初始化Policy
        
        Args:
            checkpoint_path: checkpoint路径
            control_mode: "joints" 或 "end_pose"
            action_horizon: action序列长度
            device: 设备 ("cuda:0" 或 "cpu")
        """
        self.checkpoint_path = checkpoint_path
        self.control_mode = control_mode
        self.action_horizon = action_horizon
        self.device = device
        
        logger.info(f"Loading model from {checkpoint_path}...")
        self.model = self.load_model(checkpoint_path, device)
        logger.info("Model loaded successfully")
    
    @abstractmethod
    def load_model(self, checkpoint_path: str, device: str) -> Any:
        """
        加载模型（参赛者必须实现）
        
        Args:
            checkpoint_path: checkpoint路径
            device: 设备
        
        Returns:
            模型对象（可以是任何类型，只要能调用推理方法）
        
        Note:
            这个方法可以处理任何复杂的初始化逻辑。
            对于简单模型，直接加载即可。
            对于复杂模型（如wallx），可以在这里创建配置对象、加载processor等。
            可以使用self.checkpoint_path、self.device、self.control_mode等属性。
        """
        raise NotImplementedError("请实现 load_model() 方法")
    
    def convert_input(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        输入格式转换（可选，参赛者可以覆盖）
        
        将x2robot_client格式转换为模型期望的格式
        
        Args:
            obs: x2robot_client格式的observation
                - CAMERA_LEFT/FRONT/RIGHT: Base64编码的JPEG图像字符串或bytes
                - ACTION_FOLLOW1_POS/FOLLOW2_POS: 7D或8D数组（numpy array、list或dict）
                - instruction: 可选的文本指令
        
        Returns:
            模型期望格式的observation字典
        
        Note:
            参赛者可以覆盖这个方法来实现自定义转换。
            也可以使用 utils.convert_observation_to_model_input() 工具函数。
        """
        # 默认不做转换，直接返回
        return obs
    
    @abstractmethod
    def convert_output(self, model_output: Any) -> Dict[str, Any]:
        """
        输出格式转换（参赛者必须实现）
        
        将模型输出转换为x2robot_client期望的格式
        
        Args:
            model_output: 模型推理结果
        
        Returns:
            x2robot_client格式的输出字典
            - joints模式: {
                "FOLLOW1_JOINTS": [[j1-j6, gripper], ...],  # action_horizon个7D数组
                "FOLLOW2_JOINTS": [[j1-j6, gripper], ...],
                "FOLLOW1_POS": [],
                "FOLLOW2_POS": [],
              }
            - end_pose模式: {
                "FOLLOW1_POS": [[x,y,z,rpy,gripper], ...],
                "FOLLOW2_POS": [[x,y,z,rpy,gripper], ...],
                "FOLLOW1_JOINTS": [],
                "FOLLOW2_JOINTS": [],
              }
        """
        raise NotImplementedError("请实现 convert_output() 方法")
    
    @abstractmethod
    def run_inference(self, model_input: Dict[str, Any]) -> Any:
        """
        模型推理（参赛者必须实现）
        
        Args:
            model_input: 转换后的模型输入格式
        
        Returns:
            模型输出（可以是任何格式，会在convert_output中转换）
        """
        raise NotImplementedError("请实现 run_inference() 方法")
    
    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        推理接口 - 标准化的推理流程
        
        这个方法已经实现好，参赛者不需要修改。
        它会自动调用convert_input -> run_inference -> convert_output
        
        Args:
            obs: x2robot_client格式的observation
        
        Returns:
            x2robot_client格式的输出
        """
        try:
            # 1. 转换输入格式
            model_input = self.convert_input(obs)
            
            # 2. 模型推理
            model_output = self.run_inference(model_input)
            
            # 3. 转换输出格式
            return self.convert_output(model_output)
            
        except Exception as e:
            logger.error(f"Error in inference: {e}", exc_info=True)
            raise
    
    def reset(self):
        """Reset policy state if needed"""
        if hasattr(self.model, 'reset'):
            self.model.reset()
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """
        返回metadata，会在WebSocket连接时发送给client
        
        参赛者可以覆盖这个方法来自定义metadata
        """
        return {
            "control_mode": self.control_mode,
            "action_horizon": self.action_horizon,
            "state_dim": 14,  # 双臂总共14D
            "state_dim_per_arm": 7,  # 每臂7D
            "protocol_version": "1.0",
        }

