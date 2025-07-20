from typing import Tuple, Dict, Any, Optional
import numpy as np
from .base_engine import BaseTTSEngine
from srt_dubbing.src.config import MODEL
from srt_dubbing.src.logger import get_logger
from srt_dubbing.src.utils import normalize_audio_data

# 动态导入IndexTTS，如果不存在则给出友好提示
try:
    from indextts.infer import IndexTTS
except ImportError:
    IndexTTS = None

logger = get_logger()

class IndexTTSEngine(BaseTTSEngine):
    """IndexTTS引擎的实现"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化IndexTTS引擎。
        
        :param config: 引擎配置，需要包含 'model_dir' 和可选的 'cfg_path'。
        """
        if IndexTTS is None:
            raise ImportError("IndexTTS未安装，请执行 `pip install indextts-fork` 进行安装。")
            
        self.model_dir = config.get("model_dir", MODEL.DEFAULT_MODEL_DIR)
        self.cfg_path = config.get("cfg_path") or MODEL.get_default_config_path(self.model_dir)
        self.is_fp16 = config.get("is_fp16", MODEL.DEFAULT_FP16)
        
        logger.step("加载IndexTTS模型...")
        try:
            self.tts_model = IndexTTS(
                cfg_path=self.cfg_path,
                model_dir=self.model_dir,
                is_fp16=self.is_fp16
            )
            logger.success(f"IndexTTS模型加载成功: {self.model_dir}")
        except Exception as e:
            logger.error(f"IndexTTS模型加载失败: {e}")
            raise RuntimeError(f"加载IndexTTS模型失败: {e}")

    def synthesize(self, text: str, **kwargs) -> Tuple[np.ndarray, int]:
        voice_wav = kwargs.pop("voice_wav", None)
        if not voice_wav:
            raise ValueError("IndexTTS引擎的 `synthesize` 方法需要 'voice_wav' 参数。")

        sampling_rate, audio_data_int16 = self.tts_model.infer(
            text=text, audio_prompt=voice_wav, output_path=None, **kwargs
        )
        
        # 将int16格式的音频数据规范化到 [-1, 1] 的float32格式
        audio_data_float32 = normalize_audio_data(audio_data_int16)
        
        return audio_data_float32, sampling_rate

    def synthesize_to_duration(self, text: str, target_duration: float, **kwargs) -> Tuple[np.ndarray, int]:
        voice_wav = kwargs.get("voice_wav")
        if not voice_wav:
            raise ValueError("synthesize_to_duration 需要 'voice_wav' 参数。")

        # --- 二分查找实现 ---
        max_attempts = kwargs.get('max_attempts', 5)
        tolerance = kwargs.get('tolerance', 0.1) 
        
        low_penalty, high_penalty = -2.0, 2.0
        best_result = None
        min_diff = float('inf')

        for attempt in range(max_attempts):
            penalty = (low_penalty + high_penalty) / 2
            logger.debug(f"自适应合成尝试 {attempt+1}/{max_attempts}: penalty={penalty:.3f}")

            audio_data, sr = self.synthesize(text, voice_wav=voice_wav, length_penalty=penalty)
            current_duration = len(audio_data) / sr if sr > 0 else 0
            diff = current_duration - target_duration

            if abs(diff) < min_diff:
                min_diff = abs(diff)
                best_result = (audio_data, sr)

            if abs(diff) < tolerance:
                logger.debug("目标时长匹配成功，退出迭代。")
                break
            
            if diff > 0: # 音频太长，需要减小penalty
                high_penalty = penalty
            else: # 音频太短，需要增加penalty
                low_penalty = penalty
        
        if best_result is None:
             raise RuntimeError("自适应合成失败，无法生成任何有效音频。")

        final_duration = len(best_result[0]) / best_result[1]
        logger.info(f"自适应合成完成: 目标={target_duration:.2f}s, 最终={final_duration:.2f}s, 偏差={min_diff:.2f}s")
        return best_result

    @staticmethod
    def get_config_model() -> Dict[str, Any]:
        """返回IndexTTS引擎的默认配置模型"""
        return {
            "model_dir": MODEL.DEFAULT_MODEL_DIR,
            "cfg_path": None,
            "is_fp16": MODEL.DEFAULT_FP16,
        } 