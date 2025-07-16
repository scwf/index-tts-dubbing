"""
迭代生成策略 (Iterative Generation Strategy)

本策略旨在通过多次迭代生成，逐步调整`length_penalty`参数，
来使生成的音频时长无限逼近目标时长，从而完全避免使用任何有损的时间拉伸。
这是以牺牲大量计算时间为代价，追求最高音频自然度和质量的方案。
"""

import numpy as np
import librosa
import time
from typing import List, Dict, Any

from srt_dubbing.src.strategies.base_strategy import TimeSyncStrategy
from srt_dubbing.src.logger import get_logger
from srt_dubbing.src.srt_parser import SRTEntry
from srt_dubbing.src.utils import safe_import_indextts, normalize_audio_data
from srt_dubbing.src.config import MODEL, STRATEGY

logger = get_logger()
IndexTTS, _indextts_available = safe_import_indextts()

class IterativeStrategy(TimeSyncStrategy):
    """
    通过迭代调整生成参数，逼近目标时长的策略实现。
    """
    def __init__(self):
        if not _indextts_available:
            raise RuntimeError("IndexTTS未安装，无法使用此策略")
        self.tts_model = None
        # 迭代策略的超参数
        self.max_attempts = 4  # 最大尝试次数
        self.tolerance = 0.05  # 5%的成功容忍误差
        self.adjustment_factor = 1.5 # 增大了调整因子以加快收敛

    def name(self) -> str:
        return "iterative"

    def description(self) -> str:
        return "迭代策略：通过多次生成逼近目标时长，音质最佳，速度最慢。"

    def _get_audio_duration(self, audio_data: np.ndarray, sample_rate: int) -> float:
        if audio_data is None or len(audio_data) == 0:
            return 0.0
        return len(audio_data) / sample_rate

    def _generate_audio(self, text: str, voice_reference: str, **generation_kwargs) -> Dict[str, Any]:
        sampling_rate, audio_data_int16 = self.tts_model.infer(
            text=text, audio_prompt=voice_reference, output_path=None, **generation_kwargs
        )
        audio_data = normalize_audio_data(audio_data_int16)
        duration = self._get_audio_duration(audio_data, sampling_rate)
        return {"audio_data": audio_data, "duration": duration, "sample_rate": sampling_rate}

    def process_entries(
        self, entries: List[SRTEntry], voice_reference: str, model_dir: str, cfg_path: str, verbose: bool,
    ) -> List[Dict[str, Any]]:
        if self.tts_model is None:
            logger.step("加载IndexTTS模型 (iterative策略)")
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

        audio_segments = []
        for i, entry in enumerate(entries):
            target_duration = entry.duration
            progress_prefix = f"条目 {i+1}/{len(entries)}"
            logger.info(f"{progress_prefix}: \"{entry.text}\" (目标时长: {target_duration:.2f}s)")

            try:
                if target_duration < 0.1:
                    logger.warning(f"{progress_prefix}: 目标时长无效，回退到单次自然生成。")
                    best_result = self._generate_audio(entry.text, voice_reference)
                else:
                    # --- 迭代逻辑 ---
                    length_penalty = 0.0
                    best_result = None
                    min_diff = float('inf')

                    for attempt in range(self.max_attempts):
                        logger.info(f"{progress_prefix}: 第 {attempt + 1}/{self.max_attempts} 次尝试 (penalty: {length_penalty:.2f})")
                        
                        current_result = self._generate_audio(entry.text, voice_reference, length_penalty=float(length_penalty))
                        current_duration = current_result["duration"]
                        
                        if current_duration == 0:
                            logger.warning(f"{progress_prefix}: 生成了静音音频，跳过此次尝试。")
                            continue

                        diff = abs(current_duration - target_duration)

                        if diff < min_diff:
                            min_diff = diff
                            best_result = current_result
                            logger.debug(f"{progress_prefix}: 找到更优结果，时长偏差: {diff:.2f}s")

                        if diff / target_duration <= self.tolerance:
                            logger.success(f"{progress_prefix}: 在第 {attempt + 1} 次尝试中成功匹配时长!")
                            break

                        ratio = current_duration / target_duration
                        adjustment = -(ratio - 1) * self.adjustment_factor
                        length_penalty += adjustment
                        length_penalty = np.clip(length_penalty, -2.0, 2.0)
                    else: # for-else循环，当循环正常结束（未被break）时执行
                        logger.warning(f"{progress_prefix}: 达到最大尝试次数，使用最接近的结果 (时长偏差: {min_diff:.2f}s)")
                
                # --- 使用最佳结果 ---
                final_audio_data = best_result["audio_data"]
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
                silent_data = np.zeros(int(target_duration * sample_rate), dtype=np.float32)
                audio_segments.append({
                    "audio_data": silent_data,
                    "start_time": entry.start_time,
                    "end_time": entry.end_time,
                    "index": entry.index,
                    "text": f"[静音] {entry.text}",
                })

        return audio_segments 