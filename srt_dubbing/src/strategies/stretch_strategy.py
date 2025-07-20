"""
时间拉伸策略

通过时间拉伸技术，将合成的语音精确匹配到SRT字幕的规定时长，
在保证语音完整性的同时，实现与视频的精确同步。
"""
import numpy as np
import librosa
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from srt_dubbing.src.tts_engines.base_engine import BaseTTSEngine

# 使用绝对导入，更清晰明确
from srt_dubbing.src.utils import ProgressLogger
from srt_dubbing.src.config import AUDIO, STRATEGY, LOG
from srt_dubbing.src.srt_parser import SRTEntry
from srt_dubbing.src.strategies.base_strategy import TimeSyncStrategy
from srt_dubbing.src.logger import get_logger, create_process_logger

class StretchStrategy(TimeSyncStrategy):
    """时间拉伸同步策略实现"""

    def __init__(self, 
                 tts_engine: 'BaseTTSEngine',
                 max_speed_ratio: Optional[float] = None,
                 min_speed_ratio: Optional[float] = None):
        """
        初始化时间拉伸策略
        
        Args:
            tts_engine: TTS引擎实例
            max_speed_ratio: 最大语速比例 (例如2.0表示最快加速一倍)
            min_speed_ratio: 最小语速比例 (例如0.5表示最慢减速一半)
        """
        super().__init__(tts_engine)
        self.max_speed_ratio = max_speed_ratio or STRATEGY.MAX_SPEED_RATIO
        self.min_speed_ratio = min_speed_ratio or STRATEGY.MIN_SPEED_RATIO
    
    def name(self) -> str:
        """策略名称"""
        return "stretch"

    @staticmethod
    def description() -> str:
        """策略描述"""
        return "时间拉伸策略：通过改变语速来精确匹配字幕时长"

    def process_entries(self, entries: List[SRTEntry], **kwargs) -> List[Dict[str, Any]]:
        """
        处理SRT条目，生成与字幕时长精确匹配的音频片段
        
        Args:
            entries: SRT条目列表
            **kwargs: 可选参数
                - voice_reference: 参考语音文件路径
                - verbose: 详细输出
        
        Returns:
            音频片段信息列表
        """
        logger = get_logger()
        voice_reference = kwargs.get('voice_reference')
        if not voice_reference:
            raise ValueError("必须提供参考语音文件路径 (voice_reference)")
        
        verbose = kwargs.get('verbose', False)
        audio_segments = []
        
        # 创建处理进度日志器
        process_logger = create_process_logger("时间拉伸策略音频生成")
        process_logger.start(f"处理 {len(entries)} 个字幕条目")
        
        for i, entry in enumerate(entries):
            try:
                # 始终显示进度，不仅仅在verbose模式下
                text_preview = entry.text[:LOG.PROGRESS_TEXT_PREVIEW_LENGTH] + "..." if len(entry.text) > LOG.PROGRESS_TEXT_PREVIEW_LENGTH else entry.text
                process_logger.progress(i + 1, len(entries), f"条目 {entry.index}: {text_preview}")
                
                # 1. 合成原始语音 - 使用注入的TTS引擎
                assert self.tts_engine is not None, "TTS引擎未被注入"
                audio_data, sampling_rate = self.tts_engine.synthesize(
                    text=entry.text,
                    voice_wav=voice_reference
                )
                
                # 2. 计算时长和变速比例
                source_duration = len(audio_data) / sampling_rate
                target_duration = entry.duration
                
                if target_duration == 0:
                    rate = 1.0
                else:
                    rate = source_duration / target_duration
                
                # 3. 时间拉伸/压缩
                if abs(rate - 1.0) > STRATEGY.TIME_STRETCH_THRESHOLD:  # 变化超过阈值才处理
                    clamped_rate = np.clip(rate, self.min_speed_ratio, self.max_speed_ratio)
                    if abs(clamped_rate - rate) > 0.01:
                        # 优化警告信息，使其更清晰
                        speed_type = "加速" if rate > 1.0 else "减速"
                        original_percent = int((rate - 1.0) * 100)
                        adjusted_percent = int((clamped_rate - 1.0) * 100)
                        
                        logger.warning(
                            f"条目 {entry.index} 需要{speed_type} {abs(original_percent)}% 才能匹配字幕时长，"
                            f"但超出安全范围，已限制为{speed_type} {abs(adjusted_percent)}%"
                            f"（原始变速比: {rate:.2f} → 调整后: {clamped_rate:.2f}）"
                        )
                    
                    stretched_audio = librosa.effects.time_stretch(audio_data, rate=clamped_rate)
                    
                    # 验证拉伸后的时长
                    actual_duration = len(stretched_audio) / sampling_rate
                    duration_diff = abs(actual_duration - target_duration)
                    
                    if duration_diff > STRATEGY.TIME_DURATION_TOLERANCE:
                        if verbose:
                            logger.debug(f"条目 {entry.index} 拉伸后时长 {actual_duration:.2f}s 与目标 {target_duration:.2f}s 有偏差")
                        
                        # 完全不截断策略：只处理音频偏短的情况
                        target_samples = int(target_duration * sampling_rate)
                        current_samples = len(stretched_audio)
                        
                        if current_samples > target_samples:
                            # 音频偏长时：保持完整，不截断
                            overshoot_ratio = (current_samples - target_samples) / target_samples if target_samples > 0 else 0
                            if verbose:
                                logger.debug(f"  保持完整语音: 超出目标时长 {overshoot_ratio*100:.1f}% (允许重叠)")
                        elif current_samples < target_samples:
                            # 音频偏短时：填充静音到目标时长
                            padding_samples = target_samples - current_samples
                            padding = np.zeros(padding_samples, dtype=np.float32)
                            stretched_audio = np.concatenate([stretched_audio, padding])
                            if verbose:
                                logger.debug(f"  已填充静音: {padding_samples} 样本，达到目标时长")
                else:
                    stretched_audio = audio_data

                # 4. 创建音频片段
                segment = {
                    'audio_data': stretched_audio,
                    'start_time': entry.start_time,
                    'end_time': entry.end_time,
                    'text': entry.text,
                    'index': entry.index,
                    'duration': entry.duration
                }
                audio_segments.append(segment)

            except Exception as e:
                logger.error(f"条目 {entry.index} 处理失败: {e}")
                # 后备方案：创建静音片段
                # 使用配置中的默认采样率来创建静音片段，以避免在引擎加载失败时出错
                default_sr = AUDIO.DEFAULT_SAMPLE_RATE
                silence_data = np.zeros(int(entry.duration * default_sr), dtype=np.float32)
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

# 注册逻辑将移至 __init__.py 中，以更好地管理
# def _register_stretch_strategy():
#     """注册时间拉伸策略"""
#     from srt_dubbing.src.strategies import _strategy_registry
#     _strategy_registry['stretch'] = StretchStrategy

# # 在模块导入时自动注册
# _register_stretch_strategy() 