"""
配置管理模块

统一管理SRT配音工具的所有配置常量和默认值。
"""

from pathlib import Path
from typing import Dict, Any, Optional
import os


class AudioConfig:
    """音频相关配置"""
    # 音频处理常量
    DEFAULT_SAMPLE_RATE = 22050
    DEFAULT_CHANNELS = 1
    AUDIO_NORMALIZATION_FACTOR = 32768.0  # int16 to float32 conversion
    
    # 音频合并配置
    DYNAMIC_BUFFER_SIZE = 1024
    MAX_AMPLITUDE = 1.0
    
    # 音频效果配置
    DEFAULT_FADE_DURATION = 0.1
    DEFAULT_GAP_DURATION = 0.1


class StrategyConfig:
    """策略相关配置"""
    # 时间拉伸策略 - 优化音质保护
    TIME_STRETCH_THRESHOLD = 0.05  # 变速阈值 (5%)
    TIME_DURATION_TOLERANCE = 0.1   # 时间偏差容忍度 (0.1秒)
    
    # 保守的变速范围 - 优先保证音质
    MAX_SPEED_RATIO = 1.5    # 减少从2.0到1.5，减少音质损失
    MIN_SPEED_RATIO = 0.7    # 减少从0.5到0.7，减少音质损失
    
    # 高质量模式的变速范围 - 可选的更保守设置
    HIGH_QUALITY_MAX_SPEED = 1.3
    HIGH_QUALITY_MIN_SPEED = 0.8
    
    # 基础策略 - 保持不变
    SILENCE_THRESHOLD = 0.5
    BASIC_MAX_SPEED_RATIO = 1.2
    BASIC_MIN_SPEED_RATIO = 0.8


class ModelConfig:
    """模型相关配置"""
    # IndexTTS模型默认路径
    DEFAULT_MODEL_DIR = "model-dir/index_tts"
    DEFAULT_CONFIG_FILE = "config.yaml"
    
    @classmethod
    def get_default_config_path(cls, model_dir: Optional[str] = None) -> str:
        """获取默认配置文件路径"""
        model_dir = model_dir or cls.DEFAULT_MODEL_DIR
        return os.path.join(model_dir, cls.DEFAULT_CONFIG_FILE)
    
    # TTS推理配置
    DEFAULT_FP16 = True


class F5TTSConfig:
    """F5TTS引擎相关配置"""
    # 参数源自 F5TTS_infer.md
    MODEL = "F5TTS_v1_Base"
    CKPT_FILE = None
    VOCAB_FILE = None
    ODE_METHOD = "euler"
    USE_EMA = True
    VOCODER_LOCAL_PATH = None
    DEVICE = None
    HF_CACHE_DIR = "model-dir/"

    @classmethod
    def get_init_kwargs(cls) -> Dict[str, Any]:
        """获取用于F5TTS初始化的字典"""
        return {
            "model": cls.MODEL,
            "ckpt_file": cls.CKPT_FILE,
            "vocab_file": cls.VOCAB_FILE,
            "ode_method": cls.ODE_METHOD,
            "use_ema": cls.USE_EMA,
            "vocoder_local_path": cls.VOCODER_LOCAL_PATH,
            "device": cls.DEVICE,
            "hf_cache_dir": cls.HF_CACHE_DIR,
        }


class PathConfig:
    """路径相关配置"""
    # 默认输出配置
    DEFAULT_OUTPUT_DIR = "output"
    DEFAULT_OUTPUT_FILE = "output.wav"
    
    @classmethod
    def get_default_output_path(cls) -> str:
        """获取默认输出路径"""
        return os.path.join(cls.DEFAULT_OUTPUT_DIR, cls.DEFAULT_OUTPUT_FILE)


class ValidationConfig:
    """验证相关配置"""
    # SRT验证配置
    TIME_MATCH_TOLERANCE = 0.1  # 时间匹配容忍度
    MIN_TEXT_LENGTH = 1
    
    # 音频验证配置
    MIN_AUDIO_DURATION = 0.01


class LogConfig:
    """日志相关配置"""
    # 进度显示配置
    PROGRESS_TEXT_PREVIEW_LENGTH = 30
    
    # 日志格式
    ERROR_PREFIX = "错误"
    WARNING_PREFIX = "警告"
    INFO_PREFIX = "信息"


# 全局配置实例
CONFIG = {
    'audio': AudioConfig,
    'strategy': StrategyConfig,
    'model': ModelConfig,
    'f5_tts': F5TTSConfig,
    'path': PathConfig,
    'validation': ValidationConfig,
    'log': LogConfig,
}


def get_config(category: str) -> Any:
    """
    获取指定类别的配置
    
    Args:
        category: 配置类别 ('audio', 'strategy', 'model', 'path', 'validation', 'log')
    
    Returns:
        对应的配置类
    """
    return CONFIG.get(category)


# 常用配置的快捷访问
AUDIO = AudioConfig
STRATEGY = StrategyConfig  
MODEL = ModelConfig
F5TTS = F5TTSConfig
PATH = PathConfig
VALIDATION = ValidationConfig
LOG = LogConfig 