"""
智能拉伸策略

该策略旨在通过优先调整生成参数来匹配字幕时长，以获得最佳音质。
仅在必要时，才使用小幅度的音频拉伸作为最终对齐的“精调”手段。
"""

import os
import tempfile
import time
from typing import List, Dict, Any

import torchaudio
import numpy as np
import librosa

from srt_dubbing.src.strategies.base_strategy import TimeSyncStrategy
from srt_dubbing.src.audio_processor import AudioProcessor
from srt_dubbing.src.logger import get_logger
from srt_dubbing.src.srt_parser import SRTEntry
from srt_dubbing.src.utils import safe_import_indextts, normalize_audio_data
from srt_dubbing.src.config import MODEL, STRATEGY


logger = get_logger()
IndexTTS, _indextts_available = safe_import_indextts()


class IntelligentStretchStrategy(TimeSyncStrategy):
    """
    智能拉伸策略：优先调整语速，最后微调音频长度，平衡音质和同步。
    """
    def __init__(self, model_path: str = None, hq_voice: bool = False, voice_rate: str = None):
        super().__init__("intelligent", "智能拉伸策略：优先调整语速，最后微调音频长度。")
        self.model_path = model_path
        self.hq_voice = hq_voice
        if not _indextts_available:
            raise RuntimeError("IndexTTS未安装，无法使用此策略")
        self.tts_model = None
        self.max_speed_ratio = STRATEGY.MAX_SPEED_RATIO
        self.min_speed_ratio = STRATEGY.MIN_SPEED_RATIO

    def name(self) -> str:
        return "intelligent"

    def description(self) -> str:
        return "智能策略：优先调整生成参数匹配时长，音质更优，但速度较慢。"

    def _get_audio_duration(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """计算音频时长"""
        if audio_data is None or len(audio_data) == 0:
            return 0.0
        return len(audio_data) / sample_rate
    
    def _create_silent_segment(self, duration_s: float, sample_rate: int) -> np.ndarray:
        """创建静音片段"""
        return np.zeros(int(duration_s * sample_rate), dtype=np.float32)

    def _generate_audio(self, text: str, voice_reference: str, **generation_kwargs) -> Dict[str, Any]:
        """封装TTS生成逻辑"""
        sampling_rate, audio_data_int16 = self.tts_model.infer(
            text=text,
            audio_prompt=voice_reference,
            output_path=None,  # 内存返回模式
            **generation_kwargs
        )
        audio_data = normalize_audio_data(audio_data_int16)
        duration = self._get_audio_duration(audio_data, sampling_rate)
        
        return {
            "audio_data": audio_data,
            "duration": duration,
            "sample_rate": sampling_rate
        }

    def process_entries(
        self,
        entries: List[SRTEntry],
        voice_reference: str,
        model_dir: str,
        cfg_path: str,
        verbose: bool,
    ) -> List[Dict[str, Any]]:
        
        # 1. 初始化模型（如果需要）
        if self.tts_model is None:
            logger.step("加载IndexTTS模型 (intelligent策略)")
            try:
                self.tts_model = IndexTTS(
                    cfg_path=cfg_path or MODEL.get_default_config_path(model_dir),
                    model_dir=model_dir or MODEL.DEFAULT_MODEL_DIR,
                    is_fp16=MODEL.DEFAULT_FP16
                )
                logger.success("IndexTTS模型加载成功")
            except Exception as e:
                logger.error(f"IndexTTS模型加载失败: {e}")
                raise RuntimeError(f"加载IndexTTS模型失败: {e}")
        
        processor = AudioProcessor()
        audio_segments = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"使用临时目录: {temp_dir}")

            total_entries = len(entries)
            
            for i, entry in enumerate(entries):
                # --- BUG修复：entry.duration 已经是秒，无需再除以1000 ---
                target_duration = entry.duration
                progress_prefix = f"条目 {i+1}/{total_entries}"
                logger.info(f"{progress_prefix}: \"{entry.text}\" (目标时长: {target_duration:.2f}s)")

                try:
                    # 当目标时长无效时，回退到自然生成模式
                    if target_duration < 0.1:
                        logger.warning(f"{progress_prefix}: 目标时长为零或无效 ({target_duration:.2f}s)，回退到自然生成模式。")
                        generated_audio = self._generate_audio(entry.text, voice_reference)
                        final_audio_data = generated_audio["audio_data"]
                        final_duration = generated_audio["duration"]
                        
                        audio_segments.append({
                            "audio_data": final_audio_data,
                            "start_time": entry.start_time,
                            "end_time": entry.end_time,
                            "index": entry.index,
                            "text": entry.text,
                        })
                        continue

                    # 初始生成
                    generated_audio_1 = self._generate_audio(entry.text, voice_reference, **{"length_penalty": 0.0})
                    generated_duration_1 = generated_audio_1["duration"]
                    ratio_1 = generated_duration_1 / target_duration if target_duration > 0 else float('inf')
                    
                    logger.info(f"{progress_prefix}: 初始生成完成，时长: {generated_duration_1:.2f}s (比例: {ratio_1:.2f})")
                    
                    # 检查是否在容忍区间内
                    if 0.85 <= ratio_1 <= 1.15:
                        clamped_rate_1 = np.clip(ratio_1, self.min_speed_ratio, self.max_speed_ratio)
                        final_audio_data = librosa.effects.time_stretch(generated_audio_1["audio_data"], rate=clamped_rate_1)
                        final_duration = self._get_audio_duration(final_audio_data, generated_audio_1["sample_rate"])
                        logger.success(f"{progress_prefix}: 结果在容忍范围内，微调后时长: {final_duration:.2f}s")
                        
                        audio_segments.append({
                            "audio_data": final_audio_data,
                            "start_time": entry.start_time,
                            "end_time": entry.end_time,
                            "index": entry.index,
                            "text": entry.text,
                        })
                        continue

                    # 参数调整 & 再生成
                    penalty_factor = 1.5
                    length_penalty = - (ratio_1 - 1) * penalty_factor
                    length_penalty = max(-2.0, min(2.0, length_penalty))
                    logger.info(f"{progress_prefix}: 调整生成参数, 新 length_penalty: {length_penalty:.2f}")

                    generated_audio_2 = self._generate_audio(entry.text, voice_reference, **{"length_penalty": length_penalty})
                    generated_duration_2 = generated_audio_2["duration"]
                    ratio_2 = generated_duration_2 / target_duration if target_duration > 0 else float('inf')
                    logger.info(f"{progress_prefix}: 二次生成完成，时长: {generated_duration_2:.2f}s (比例: {ratio_2:.2f})")

                    # 最终精调
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
                    sample_rate = self.tts_model.bigvgan.h.sampling_rate if self.tts_model else MODEL.DEFAULT_SAMPLE_RATE
                    silent_data = self._create_silent_segment(target_duration if target_duration > 0.1 else 0.1, sample_rate)
                    audio_segments.append({
                        "audio_data": silent_data,
                        "start_time": entry.start_time,
                        "end_time": entry.end_time,
                        "index": entry.index,
                        "text": f"[静音] {entry.text}",
                    })
        return audio_segments 