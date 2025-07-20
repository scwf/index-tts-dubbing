"""
智能拉伸策略

该策略旨在通过优先调整生成参数来匹配字幕时长，以获得最佳音质。
仅在必要时，才使用小幅度的音频拉伸作为最终对齐的“精调”手段。
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING
import numpy as np
import librosa

if TYPE_CHECKING:
    from srt_dubbing.src.tts_engines.base_engine import BaseTTSEngine

from srt_dubbing.src.strategies.base_strategy import TimeSyncStrategy
from srt_dubbing.src.logger import get_logger
from srt_dubbing.src.srt_parser import SRTEntry
from srt_dubbing.src.config import STRATEGY, AUDIO

logger = get_logger()

class IntelligentStretchStrategy(TimeSyncStrategy):
    """
    智能拉伸策略：优先调整语速，最后微调音频长度，平衡音质和同步。
    """
    def __init__(self, tts_engine: 'BaseTTSEngine', **kwargs):
        super().__init__(tts_engine)
        self.max_speed_ratio = STRATEGY.MAX_SPEED_RATIO
        self.min_speed_ratio = STRATEGY.MIN_SPEED_RATIO

    @staticmethod
    def name() -> str:
        return "intelligent"

    @staticmethod
    def description() -> str:
        return "智能策略：优先调整生成参数匹配时长，音质更优，但速度较慢。"

    def _get_audio_duration(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """计算音频时长"""
        return len(audio_data) / sample_rate if audio_data is not None and len(audio_data) > 0 else 0.0

    def _generate_audio(self, text: str, voice_reference: str, **generation_kwargs) -> Dict[str, Any]:
        """封装TTS生成逻辑"""
        audio_data, sample_rate = self.tts_engine.synthesize(
            text=text,
            voice_wav=voice_reference,
            **generation_kwargs
        )
        duration = self._get_audio_duration(audio_data, sample_rate)
        
        return {
            "audio_data": audio_data,
            "duration": duration,
            "sample_rate": sample_rate
        }

    def process_entries(self, entries: List[SRTEntry], **kwargs) -> List[Dict[str, Any]]:
        voice_reference = kwargs.get('voice_reference')
        if not voice_reference:
            raise ValueError("必须提供参考语音文件路径 (voice_reference)")

        audio_segments = []
        total_entries = len(entries)
        
        for i, entry in enumerate(entries):
            target_duration = entry.duration
            progress_prefix = f"条目 {i+1}/{total_entries}"
            logger.info(f"{progress_prefix}: \"{entry.text}\" (目标时长: {target_duration:.2f}s)")

            try:
                if target_duration < 0.1:
                    logger.warning(f"{progress_prefix}: 目标时长无效 ({target_duration:.2f}s)，回退到自然生成模式。")
                    generated_audio = self._generate_audio(entry.text, voice_reference)
                    final_audio_data = generated_audio["audio_data"]
                else:
                    generated_audio_1 = self._generate_audio(entry.text, voice_reference, length_penalty=0.0)
                    ratio_1 = generated_audio_1["duration"] / target_duration if target_duration > 0 else float('inf')
                    logger.info(f"{progress_prefix}: 初始生成完成，时长: {generated_audio_1['duration']:.2f}s (比例: {ratio_1:.2f})")
                    
                    if 0.85 <= ratio_1 <= 1.15:
                        clamped_rate_1 = np.clip(ratio_1, self.min_speed_ratio, self.max_speed_ratio)
                        final_audio_data = librosa.effects.time_stretch(generated_audio_1["audio_data"], rate=clamped_rate_1)
                        final_duration = self._get_audio_duration(final_audio_data, generated_audio_1["sample_rate"])
                        logger.success(f"{progress_prefix}: 结果在容忍范围内，微调后时长: {final_duration:.2f}s")
                    else:
                        penalty_factor = 1.5
                        length_penalty = max(-2.0, min(2.0, -(ratio_1 - 1) * penalty_factor))
                        logger.info(f"{progress_prefix}: 调整生成参数, 新 length_penalty: {length_penalty:.2f}")

                        generated_audio_2 = self._generate_audio(entry.text, voice_reference, length_penalty=length_penalty)
                        ratio_2 = generated_audio_2["duration"] / target_duration if target_duration > 0 else float('inf')
                        logger.info(f"{progress_prefix}: 二次生成完成，时长: {generated_audio_2['duration']:.2f}s (比例: {ratio_2:.2f})")

                        clamped_rate_2 = np.clip(ratio_2, self.min_speed_ratio, self.max_speed_ratio)
                        final_audio_data = librosa.effects.time_stretch(generated_audio_2["audio_data"], rate=clamped_rate_2)
                        final_duration = self._get_audio_duration(final_audio_data, generated_audio_2["sample_rate"])
                        logger.success(f"{progress_prefix}: 精调后最终时长: {final_duration:.2f}s")

                audio_segments.append({
                    "audio_data": final_audio_data,
                    "start_time": entry.start_time,
                    "end_time": entry.end_time,
                    "index": entry.index,
                    "text": entry.text,
                })

            except Exception as e:
                logger.error(f"处理条目 {i+1} 失败: {e}")
                silent_data = np.zeros(int(target_duration * AUDIO.DEFAULT_SAMPLE_RATE), dtype=np.float32)
                audio_segments.append({
                    "audio_data": silent_data,
                    "start_time": entry.start_time,
                    "end_time": entry.end_time,
                    "index": entry.index,
                    "text": f"[静音] {entry.text}",
                })
        return audio_segments 