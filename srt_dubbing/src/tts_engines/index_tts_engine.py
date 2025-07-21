import inspect
from typing import Tuple, Dict, Any, Optional
import numpy as np
from .base_engine import BaseTTSEngine
from srt_dubbing.src.config import IndexTTSConfig
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

    def __init__(self):
        """
        初始化IndexTTS引擎。
        
        :param config: 引擎配置，需要包含 'model_dir' 和可选的 'cfg_path'。
        """
        if IndexTTS is None:
            raise ImportError("IndexTTS未安装。")
            
        # 直接从配置模块获取初始化参数
        init_kwargs = IndexTTSConfig.get_init_kwargs()
        # 过滤掉值为None的参数
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
        logger.step("加载IndexTTS模型...")
        try:
            self.tts_model = IndexTTS(**init_kwargs)
            # 使用内省机制，获取底层模型真正支持的参数列表
            infer_signature = inspect.signature(self.tts_model.infer)
            self.valid_infer_params = set(infer_signature.parameters.keys())
            
            logger.success(f"IndexTTS模型加载成功: {init_kwargs}")
        except Exception as e:
            logger.error(f"IndexTTS模型加载失败: {e}")
            raise RuntimeError(f"加载IndexTTS模型失败: {e}")

    def synthesize(self, text: str, **kwargs) -> Tuple[np.ndarray, int]:
        voice_reference = kwargs.get('voice_reference')
        if not voice_reference:
            raise ValueError("必须提供参考语音文件路径 (voice_reference)")

        # 优雅地过滤出底层模型支持的参数，而不是手动pop
        filtered_kwargs = {
            key: value for key, value in kwargs.items() 
            if key in self.valid_infer_params
        }

        sampling_rate, audio_data_int16 = self.tts_model.infer(
            text=text, audio_prompt=voice_reference, output_path=None, **filtered_kwargs
        )
        
        # 将int16格式的音频数据规范化到 [-1, 1] 的float32格式
        audio_data_float32 = normalize_audio_data(audio_data_int16)
        
        return audio_data_float32, sampling_rate

    def synthesize_to_duration(self, text: str, target_duration: float, **kwargs) -> Tuple[np.ndarray, int]:
        voice_reference = kwargs.get('voice_reference')
        if not voice_reference:
            raise ValueError("必须提供参考语音文件路径 (voice_reference)")

        # --- 二分查找实现 ---
        max_attempts = kwargs.get('max_attempts', 5)
        tolerance = kwargs.get('tolerance', 0.1) 
        
        low_penalty, high_penalty = -2.0, 2.0
        best_result = None
        min_diff = float('inf')

        for attempt in range(max_attempts):
            penalty = (low_penalty + high_penalty) / 2
            logger.debug(f"自适应合成尝试 {attempt+1}/{max_attempts}: penalty={penalty:.3f}")

            # 注意：这里我们明确知道要控制 length_penalty，所以直接传递
            synthesis_kwargs = kwargs.copy()
            synthesis_kwargs['length_penalty'] = penalty
            audio_data, sr = self.synthesize(text, **synthesis_kwargs)
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
