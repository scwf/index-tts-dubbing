"""
迭代生成策略 (Iterative Generation Strategy)

本策略旨在通过多次迭代生成，逐步调整`length_penalty`参数，
来使生成的音频时长无限逼近目标时长，从而完全避免使用任何有损的时间拉伸。
这是以牺牲大量计算时间为代价，追求最高音频自然度和质量的方案。
"""

import numpy as np
from typing import List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from srt_dubbing.src.tts_engines.base_engine import BaseTTSEngine

from srt_dubbing.src.strategies.base_strategy import TimeSyncStrategy
from srt_dubbing.src.logger import get_logger
from srt_dubbing.src.srt_parser import SRTEntry
from srt_dubbing.src.config import AUDIO

logger = get_logger()

class IterativeStrategy(TimeSyncStrategy):
    """
    通过迭代调整生成参数，逼近目标时长的策略实现。
    """
    def __init__(self, tts_engine: 'BaseTTSEngine', **kwargs):
        super().__init__(tts_engine)
        self.max_attempts = kwargs.get('max_attempts', 4)
        self.tolerance = kwargs.get('tolerance', 0.05)
        self.adjustment_factor = kwargs.get('adjustment_factor', 1.5)

    @staticmethod
    def name() -> str:
        return "iterative"

    @staticmethod
    def description() -> str:
        return "迭代策略：通过多次生成逼近目标时长，音质最佳，速度最慢。"

    def _get_audio_duration(self, audio_data: np.ndarray, sample_rate: int) -> float:
        return len(audio_data) / sample_rate if audio_data is not None and len(audio_data) > 0 else 0.0

    def _generate_audio(self, text: str, voice_reference: str, **generation_kwargs) -> Dict[str, Any]:
        audio_data, sample_rate = self.tts_engine.synthesize(
            text=text, voice_wav=voice_reference, **generation_kwargs
        )
        duration = self._get_audio_duration(audio_data, sample_rate)
        return {"audio_data": audio_data, "duration": duration, "sample_rate": sample_rate}

    def process_entries(self, entries: List[SRTEntry], **kwargs) -> List[Dict[str, Any]]:
        voice_reference = kwargs.get('voice_reference')
        if not voice_reference:
            raise ValueError("必须提供参考语音文件路径 (voice_reference)")
            
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
                    all_results = []
                    
                    logger.info(f"{progress_prefix}: 基准生成 (penalty: 0.00)")
                    base_result = self._generate_audio(entry.text, voice_reference, length_penalty=0.0)
                    base_duration = base_result["duration"]
                    base_diff = abs(base_duration - target_duration)
                    
                    all_results.append({"result": base_result, "penalty": 0.0, "duration": base_duration, "diff": base_diff, "attempt": 0})
                    logger.info(f"{progress_prefix}: 基准生成完成，时长: {base_duration:.2f}s, 偏差: {base_diff:.2f}s")
                    
                    if base_diff / target_duration <= self.tolerance:
                        logger.success(f"{progress_prefix}: 基准生成已满足精度要求!")
                        best_result = base_result
                    else:
                        length_penalty = 0.0
                        for attempt in range(1, self.max_attempts):
                            current_best = min(all_results, key=lambda x: x["diff"])
                            ratio = current_best["duration"] / target_duration
                            length_penalty += -(ratio - 1) * self.adjustment_factor
                            length_penalty = np.clip(length_penalty, -2.0, 2.0)
                            
                            logger.info(f"{progress_prefix}: 第 {attempt}/{self.max_attempts-1} 次尝试 (penalty: {length_penalty:.2f})")
                            current_result = self._generate_audio(entry.text, voice_reference, length_penalty=float(length_penalty))
                            
                            if current_result["duration"] == 0:
                                logger.warning(f"{progress_prefix}: 生成了静音音频，跳过此次尝试。")
                                continue

                            current_diff = abs(current_result["duration"] - target_duration)
                            all_results.append({"result": current_result, "penalty": length_penalty, "duration": current_result["duration"], "diff": current_diff, "attempt": attempt})
                            logger.info(f"{progress_prefix}: 尝试 {attempt} 完成，时长: {current_result['duration']:.2f}s, 偏差: {current_diff:.2f}s")

                            if current_diff / target_duration <= self.tolerance:
                                logger.success(f"{progress_prefix}: 在第 {attempt} 次尝试中成功匹配时长!")
                                break
                        
                        best_attempt = min(all_results, key=lambda x: x["diff"])
                        best_result = best_attempt["result"]
                        
                        logger.info(f"{progress_prefix}: === 所有尝试对比 ===")
                        best_attempt_num = best_attempt["attempt"]
                        for r in all_results:
                            status = "✅ 最佳" if r["attempt"] == best_attempt_num else "  "
                            logger.info(f"{progress_prefix}: {status} 尝试{r['attempt']}: penalty={r['penalty']:.2f}, 时长={r['duration']:.2f}s, 偏差={r['diff']:.2f}s")
                        
                        if len(all_results) >= self.max_attempts:
                            logger.warning(f"{progress_prefix}: 达到最大尝试次数，选择最佳结果 (时长偏差: {best_attempt['diff']:.2f}s)")
                        else:
                            logger.success(f"{progress_prefix}: 找到满足精度的结果!")
                
                audio_segments.append({
                    "audio_data": best_result["audio_data"],
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