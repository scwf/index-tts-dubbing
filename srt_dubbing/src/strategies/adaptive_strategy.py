"""
自适应时长策略 (Adaptive Strategy)

该策略直接调用TTS引擎的自适应合成功能，以生成一个
与目标时长尽可能匹配的音频，而无需关心引擎内部的具体实现。
"""
import numpy as np
from typing import List, Dict, Any

from srt_dubbing.src.tts_engines.base_engine import BaseTTSEngine
from .base_strategy import TimeSyncStrategy
from srt_dubbing.src.logger import get_logger, create_process_logger
from srt_dubbing.src.srt_parser import SRTEntry
from srt_dubbing.src.config import AUDIO, LOG

logger = get_logger()

class AdaptiveStrategy(TimeSyncStrategy):
    """
    通过调用引擎内置的自适应功能来匹配时长的策略。
    """
    def __init__(self, tts_engine: 'BaseTTSEngine', **kwargs):
        super().__init__(tts_engine)

    @staticmethod
    def name() -> str:
        return "adaptive"

    @staticmethod
    def description() -> str:
        return "自适应策略：调用引擎的自适应功能以匹配时长，效果取决于引擎自身实现。"

    def process_entries(self, entries: List[SRTEntry], **kwargs) -> List[Dict[str, Any]]:
        
        process_logger = create_process_logger("自适应策略音频生成")
        audio_segments = []
        
        process_logger.start(f"处理 {len(entries)} 个字幕条目")
        
        for i, entry in enumerate(entries):
            try:
                text_preview = entry.text[:LOG.PROGRESS_TEXT_PREVIEW_LENGTH] + "..."
                process_logger.progress(i + 1, len(entries), f"条目 {entry.index}: {text_preview}")

                # 直接调用引擎的自适应方法
                audio_data, _ = self.tts_engine.synthesize_to_duration(
                    text=entry.text,
                    target_duration=entry.duration,
                    **kwargs
                )

                segment = {
                    'audio_data': audio_data,
                    'start_time': entry.start_time,
                    'end_time': entry.end_time,
                    'text': entry.text,
                    'index': entry.index,
                    'duration': entry.duration
                }
                audio_segments.append(segment)

            except NotImplementedError as e:
                # 捕获引擎不支持此功能的错误
                logger.error(f"处理失败: {e}")
                logger.error(f"无法使用 '{self.name()}' 策略。请为 '{type(self.tts_engine).__name__}' 引擎选择其他策略。")
                # 遇到不支持的引擎，直接终止处理
                raise e
            except Exception as e:
                logger.error(f"条目 {entry.index} 处理失败: {e}")
                # 为失败的条目创建静音片段
                silence_data = np.zeros(int(entry.duration * AUDIO.DEFAULT_SAMPLE_RATE), dtype=np.float32)
                segment = {
                    'audio_data': silence_data,
                    'start_time': entry.start_time,
                    'end_time': entry.end_time,
                    'text': f"[静音] {entry.text}",
                    'index': entry.index,
                    'duration': entry.duration
                }
                audio_segments.append(segment)

        process_logger.complete(f"生成 {len(audio_segments)} 个音频片段")
        return audio_segments 