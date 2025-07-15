"""
SRT配音项目 - 将SRT字幕文件转换为配音音频

本包提供了完整的SRT字幕到音频转换功能，包括：
- SRT文件解析
- 多种时间同步策略
- 音频处理和合成
- 命令行接口

主要模块：
- srt_parser: SRT文件解析功能
- strategies: 时间同步策略实现
- audio_processor: 音频处理和合成
- cli: 命令行接口
"""

__version__ = "0.1.0"
__author__ = "SRT Dubbing Team"

# 导出主要类和函数
from .srt_parser import SRTParser
from .audio_processor import AudioProcessor
from .cli import main

__all__ = [
    "SRTParser",
    "AudioProcessor", 
    "main"
] 