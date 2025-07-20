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
from srt_dubbing.src.utils import normalize_audio_data
from srt_dubbing.src.config import STRATEGY, AUDIO

logger = get_logger()

class IterativeStrategy(TimeSyncStrategy):
    """
    通过迭代调整生成参数，逼近目标时长的策略实现。
    """
    def __init__(self):
        super().__init__()
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
        assert self.tts_model is not None, "TTS模型未初始化，请先调用ensure_model_initialized"
        sampling_rate, audio_data_int16 = self.tts_model.infer(
            text=text, audio_prompt=voice_reference, output_path=None, **generation_kwargs
        )
        audio_data = normalize_audio_data(audio_data_int16)
        duration = self._get_audio_duration(audio_data, sampling_rate)
        return {"audio_data": audio_data, "duration": duration, "sample_rate": sampling_rate}

    def process_entries(
        self, entries: List[SRTEntry], voice_reference: str, model_dir: str, cfg_path: str, verbose: bool,
    ) -> List[Dict[str, Any]]:
        self.ensure_model_initialized(model_dir, cfg_path)

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
                    # --- 迭代逻辑优化版 ---
                    all_results = []  # 存储所有尝试的结果
                    length_penalty_values = [0.0]  # 从0开始的penalty序列
                    
                    # 第一次生成（基准）
                    logger.info(f"{progress_prefix}: 基准生成 (penalty: 0.00)")
                    base_result = self._generate_audio(entry.text, voice_reference, length_penalty=0.0)
                    base_duration = base_result["duration"]
                    base_diff = abs(base_duration - target_duration)
                    
                    all_results.append({
                        "result": base_result,
                        "penalty": 0.0,
                        "duration": base_duration,
                        "diff": base_diff,
                        "attempt": 0
                    })
                    
                    logger.info(f"{progress_prefix}: 基准生成完成，时长: {base_duration:.2f}s, 偏差: {base_diff:.2f}s")
                    
                    # 检查基准是否已经满足要求
                    if base_diff / target_duration <= self.tolerance:
                        logger.success(f"{progress_prefix}: 基准生成已满足精度要求!")
                        best_result = base_result
                    else:
                        # 继续迭代优化
                        length_penalty = 0.0
                        
                        for attempt in range(1, self.max_attempts):
                            # 根据当前最佳结果调整penalty
                            current_best = min(all_results, key=lambda x: x["diff"])
                            ratio = current_best["duration"] / target_duration
                            adjustment = -(ratio - 1) * self.adjustment_factor
                            length_penalty += adjustment
                            length_penalty = np.clip(length_penalty, -2.0, 2.0)
                            
                            logger.info(f"{progress_prefix}: 第 {attempt}/{self.max_attempts-1} 次尝试 (penalty: {length_penalty:.2f})")
                            
                            current_result = self._generate_audio(entry.text, voice_reference, length_penalty=float(length_penalty))
                            current_duration = current_result["duration"]
                            
                            if current_duration == 0:
                                logger.warning(f"{progress_prefix}: 生成了静音音频，跳过此次尝试。")
                                continue

                            current_diff = abs(current_duration - target_duration)
                            
                            all_results.append({
                                "result": current_result,
                                "penalty": length_penalty,
                                "duration": current_duration,
                                "diff": current_diff,
                                "attempt": attempt
                            })
                            
                            logger.info(f"{progress_prefix}: 尝试 {attempt} 完成，时长: {current_duration:.2f}s, 偏差: {current_diff:.2f}s")

                            # 检查是否达到精度要求
                            if current_diff / target_duration <= self.tolerance:
                                logger.success(f"{progress_prefix}: 在第 {attempt} 次尝试中成功匹配时长!")
                                break
                        
                        # 选择所有尝试中的最佳结果
                        best_attempt = min(all_results, key=lambda x: x["diff"])
                        best_result = best_attempt["result"]
                        
                        # 输出详细的对比信息
                        logger.info(f"{progress_prefix}: === 所有尝试对比 ===")
                        best_attempt_num = best_attempt["attempt"]  # 获取最佳尝试的编号
                        for r in all_results:
                            status = "✅ 最佳" if r["attempt"] == best_attempt_num else "  "
                            logger.info(f"{progress_prefix}: {status} 尝试{r['attempt']}: penalty={r['penalty']:.2f}, 时长={r['duration']:.2f}s, 偏差={r['diff']:.2f}s")
                        
                        if len(all_results) >= self.max_attempts:
                            logger.warning(f"{progress_prefix}: 达到最大尝试次数，选择最佳结果 (时长偏差: {best_attempt['diff']:.2f}s)")
                        else:
                            logger.success(f"{progress_prefix}: 找到满足精度的结果!")
                
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
                sample_rate = self.tts_model.bigvgan.h.sampling_rate if self.tts_model else AUDIO.DEFAULT_SAMPLE_RATE
                silent_data = np.zeros(int(target_duration * sample_rate), dtype=np.float32)
                audio_segments.append({
                    "audio_data": silent_data,
                    "start_time": entry.start_time,
                    "end_time": entry.end_time,
                    "index": entry.index,
                    "text": f"[静音] {entry.text}",
                })

        return audio_segments

# 注册策略（避免循环导入）
def _register_iterative_strategy():
    """注册迭代生成策略"""
    from srt_dubbing.src.strategies import _strategy_registry
    _strategy_registry['iterative'] = IterativeStrategy

# 在模块导入时自动注册
_register_iterative_strategy() 