"""
基础自然合成策略

采用自然语音合成 + 静音填充的方式处理SRT字幕，
优先保证语音质量，使用静音来匹配时间间隔。
"""
import numpy as np
from typing import List, Dict, Any, Optional
import os

# 使用绝对导入，更清晰明确
from srt_dubbing.src.utils import setup_project_path, normalize_audio_data, validate_file_exists
from srt_dubbing.src.config import AUDIO, STRATEGY, MODEL, VALIDATION, LOG
from srt_dubbing.src.srt_parser import SRTEntry
from srt_dubbing.src.strategies.base_strategy import TimeSyncStrategy
from srt_dubbing.src.logger import get_logger, create_process_logger
from indextts.infer import IndexTTS

# 初始化项目环境
setup_project_path()

class BasicStrategy(TimeSyncStrategy):
    """基础自然合成策略实现"""
    
    def __init__(self, 
                 silence_threshold: Optional[float] = None,
                 max_speed_ratio: Optional[float] = None,
                 min_speed_ratio: Optional[float] = None):
        super().__init__()
        """
        初始化基础策略
        
        Args:
            silence_threshold: 静音阈值（秒），超过此时间才添加静音
            max_speed_ratio: 最大语速比例
            min_speed_ratio: 最小语速比例
        """
        self.silence_threshold = silence_threshold or STRATEGY.SILENCE_THRESHOLD
        self.max_speed_ratio = max_speed_ratio or STRATEGY.BASIC_MAX_SPEED_RATIO
        self.min_speed_ratio = min_speed_ratio or STRATEGY.BASIC_MIN_SPEED_RATIO
    
    def name(self) -> str:
        """策略名称"""
        return "basic"
    
    def description(self) -> str:
        """策略描述"""
        return "自然合成策略：使用自然语音合成，通过静音填充匹配时间"
    
    def process_entries(self, entries: List[SRTEntry], **kwargs) -> List[Dict[str, Any]]:
        """
        处理SRT条目，生成音频片段
        
        Args:
            entries: SRT条目列表
            **kwargs: 可选参数
                - voice_reference: 参考语音文件路径
                - model_dir: 模型目录路径
                - cfg_path: 配置文件路径
                - sample_rate: 采样率
                - verbose: 详细输出
        
        Returns:
            音频片段信息列表
        """
        # 确保TTS模型已初始化
        model_dir = kwargs.get('model_dir', MODEL.DEFAULT_MODEL_DIR)
        cfg_path = kwargs.get('cfg_path', MODEL.get_default_config_path(model_dir))
        self.ensure_model_initialized(model_dir, cfg_path)
        
        voice_reference = kwargs.get('voice_reference')
        if not voice_reference:
            raise ValueError("必须提供参考语音文件路径 (voice_reference)")
        
        validate_file_exists(voice_reference, "参考语音文件")
        
        logger = get_logger()
        verbose = kwargs.get('verbose', False)
        audio_segments = []
        
        # 创建处理进度日志器
        process_logger = create_process_logger("基础策略音频生成")
        process_logger.start(f"处理 {len(entries)} 个字幕条目")
        
        for i, entry in enumerate(entries):
            try:
                # 始终显示进度，不仅仅在verbose模式下
                text_preview = entry.text[:LOG.PROGRESS_TEXT_PREVIEW_LENGTH] + "..." if len(entry.text) > LOG.PROGRESS_TEXT_PREVIEW_LENGTH else entry.text
                process_logger.progress(i + 1, len(entries), f"条目 {entry.index}: {text_preview}")
                
                # 合成语音
                audio_data = self.synthesize_text(
                    entry.text, 
                    voice_reference=voice_reference,
                    verbose=verbose
                )
                
                # 创建音频片段信息
                segment = {
                    'audio_data': audio_data,
                    'start_time': entry.start_time,
                    'end_time': entry.end_time,
                    'text': entry.text,
                    'index': entry.index,
                    'duration': entry.duration
                }
                
                audio_segments.append(segment)
                
            except Exception as e:
                logger.error(f"条目 {entry.index} 处理失败: {e}")
                # 创建静音片段作为后备
                silence_duration = entry.duration
                # 使用标准采样率创建静音片段
                silence_data = self.add_silence(silence_duration, AUDIO.DEFAULT_SAMPLE_RATE)
                segment = {
                    'audio_data': silence_data,
                    'start_time': entry.start_time,
                    'end_time': entry.end_time,
                    'text': entry.text,
                    'index': entry.index,
                    'duration': entry.duration
                }
                audio_segments.append(segment)
        
        process_logger.complete(f"生成 {len(audio_segments)} 个音频片段")
        return audio_segments

    def synthesize_text(self, text: str, **kwargs) -> np.ndarray:
        """
        使用TTS模型合成单个文本片段，并处理返回的内存数据。
        
        Args:
            text: 要合成的文本
            **kwargs: 
                - voice_reference: 参考语音路径
                
        Returns:
            np.ndarray: float32格式的一维音频数据
        """
        voice_reference = kwargs.get('voice_reference')
        if not voice_reference:
            raise ValueError("synthesize_text需要voice_reference")
        
        # 确保模型已初始化
        assert self.tts_model is not None, "TTS模型未初始化，请先调用ensure_model_initialized"
        
        # 调用infer，触发内存返回模式
        sampling_rate, audio_data_int16 = self.tts_model.infer(
            text=text,
            audio_prompt=voice_reference,
            output_path=None
        )

        # 确保音频数据是1D float32 数组并规范化到 [-1, 1] 范围
        audio_data_float32 = normalize_audio_data(audio_data_int16)
        
        return audio_data_float32

    def add_silence(self, duration: float, sample_rate: Optional[int] = None) -> np.ndarray:
        """
        生成指定时长的静音
        
        Args:
            duration: 静音时长（秒）
            sample_rate: 采样率
        
        Returns:
            静音音频数据
        """
        sample_rate = sample_rate or AUDIO.DEFAULT_SAMPLE_RATE
        num_samples = int(duration * sample_rate)
        return np.zeros(num_samples, dtype=np.float32)
    
    def calculate_timing(self, entries: List[SRTEntry]) -> List[Dict[str, float]]:
        """
        计算每个条目的时间安排
        
        Args:
            entries: SRT条目列表
        
        Returns:
            时间安排信息列表，包含：
            - speech_duration: 语音时长  
            - silence_duration: 静音时长
            - total_duration: 总时长
        """
        timing_info = []
        
        for entry in entries:
            # 基础策略：使用自然语音时长，不强制匹配SRT时间
            speech_duration = entry.duration  # 暂时使用SRT时长作为估计
            silence_duration = 0.0  # 基础策略不添加额外静音
            total_duration = speech_duration + silence_duration
            
            timing_info.append({
                'speech_duration': speech_duration,
                'silence_duration': silence_duration, 
                'total_duration': total_duration
            })
        
        return timing_info
    
    def validate_timing(self, entries: List[SRTEntry], 
                       audio_segments: List[Dict[str, Any]]) -> bool:
        """
        验证音频段时间是否符合SRT要求
        
        Args:
            entries: 原始SRT条目
            audio_segments: 生成的音频段
        
        Returns:
            验证是否通过
        """
        logger = get_logger()
        
        if len(entries) != len(audio_segments):
            logger.warning(f"条目数量不匹配 ({len(entries)} vs {len(audio_segments)})")
            return False
        
        for entry, segment in zip(entries, audio_segments):
            # 检查音频数据是否存在
            if 'audio_data' not in segment or segment['audio_data'] is None:
                logger.error(f"条目 {entry.index} 没有音频数据")
                return False
            
            # 检查时间信息一致性
            if abs(segment['start_time'] - entry.start_time) > VALIDATION.TIME_MATCH_TOLERANCE:
                logger.warning(f"条目 {entry.index} 开始时间不匹配")
            
            if abs(segment['end_time'] - entry.end_time) > VALIDATION.TIME_MATCH_TOLERANCE:
                logger.warning(f"条目 {entry.index} 结束时间不匹配")
        
        return True 

# 注册策略（避免循环导入）
def _register_basic_strategy():
    """注册基础策略"""
    from srt_dubbing.src.strategies import _strategy_registry
    _strategy_registry['basic'] = BasicStrategy

# 在模块导入时自动注册
_register_basic_strategy() 