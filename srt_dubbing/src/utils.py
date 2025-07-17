"""
通用工具模块

提供项目中常用的工具函数，避免代码重复。
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import logging


def setup_project_path():
    """
    设置项目路径，确保可以正确导入模块
    
    这个函数应该在每个需要导入项目模块的文件开头调用一次。
    """
    # 获取项目根目录 (index-tts)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent  # srt_dubbing/src/utils.py -> index-tts
    
    # 添加到 sys.path（如果还没有的话）
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    return project_root





def validate_file_exists(file_path: str, file_type: str = "文件") -> bool:
    """
    验证文件是否存在
    
    Args:
        file_path: 文件路径
        file_type: 文件类型描述（用于错误信息）
    
    Returns:
        bool: 文件是否存在
        
    Raises:
        FileNotFoundError: 文件不存在时抛出
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_type}不存在: {file_path}")
    return True


def create_directory_if_needed(file_path: str) -> Path:
    """
    如果目录不存在则创建
    
    Args:
        file_path: 文件路径（将创建其父目录）
    
    Returns:
        Path: 父目录路径
    """
    output_dir = Path(file_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def format_duration(seconds: float) -> str:
    """
    格式化时长显示
    
    Args:
        seconds: 秒数
    
    Returns:
        str: 格式化的时长字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {secs:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"


def format_progress_text(text: str, max_length: int = 30) -> str:
    """
    格式化进度显示的文本
    
    Args:
        text: 原始文本
        max_length: 最大长度
    
    Returns:
        str: 截断并格式化的文本
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def get_audio_info_summary(audio_data) -> Dict[str, Any]:
    """
    获取音频数据的基本信息摘要
    
    Args:
        audio_data: 音频数据 (numpy array)
    
    Returns:
        dict: 音频信息字典
    """
    if not hasattr(audio_data, '__len__') or len(audio_data) == 0:
        return {
            'duration': 0.0,
            'sample_count': 0,
            'is_valid': False
        }
    
    from .config import AUDIO
    
    return {
        'duration': len(audio_data) / AUDIO.DEFAULT_SAMPLE_RATE,
        'sample_count': len(audio_data),
        'is_valid': True
    }


class ProgressLogger:
    """简单的进度日志记录器"""
    
    def __init__(self, total_items: int, description: str = "处理"):
        self.total_items = total_items
        self.current_item = 0
        self.description = description
    
    def update(self, item_index: int, item_description: str = "") -> None:
        """更新进度"""
        self.current_item = item_index + 1
        progress = (self.current_item / self.total_items) * 100
        
        if item_description:
            item_desc = format_progress_text(item_description)
            print(f"{self.description} {self.current_item}/{self.total_items} ({progress:.1f}%): {item_desc}")
        else:
            print(f"{self.description} {self.current_item}/{self.total_items} ({progress:.1f}%)")
    
    def complete(self) -> None:
        """完成进度"""
        print(f"✓ {self.description}完成，共处理 {self.total_items} 项")


def normalize_audio_data(audio_data_int16, normalization_factor: Optional[float] = None):
    """
    规范化音频数据
    
    Args:
        audio_data_int16: int16格式的音频数据
        normalization_factor: 规范化因子
    
    Returns:
        numpy.ndarray: 规范化后的float32音频数据
    """
    import numpy as np
    from .config import AUDIO
    
    if normalization_factor is None:
        normalization_factor = AUDIO.AUDIO_NORMALIZATION_FACTOR
    
    return audio_data_int16.flatten().astype(np.float32) / normalization_factor


def handle_exception_with_fallback(operation_name: str, fallback_value: Any = None):
    """
    异常处理装饰器，提供回退值
    
    Args:
        operation_name: 操作名称（用于日志）
        fallback_value: 回退值
    
    Returns:
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"警告: {operation_name} 失败: {e}")
                return fallback_value
        return wrapper
    return decorator


# 常用的验证函数
def validate_kwargs_required(kwargs: Dict[str, Any], required_keys: List[str]):
    """
    验证必需的关键字参数
    
    Args:
        kwargs: 关键字参数字典
        required_keys: 必需的键列表
    
    Raises:
        ValueError: 缺少必需参数时抛出
    """
    missing_keys = [key for key in required_keys if key not in kwargs or kwargs[key] is None]
    if missing_keys:
        raise ValueError(f"缺少必需参数: {', '.join(missing_keys)}")


# 项目初始化函数
def initialize_project():
    """
    初始化项目环境
    
    Returns:
        Path: project_root_path
    """
    project_root = setup_project_path()
    
    return project_root 