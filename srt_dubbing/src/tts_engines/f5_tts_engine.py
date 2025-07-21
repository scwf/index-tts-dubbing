from typing import Tuple, Dict, Any
import numpy as np
from .base_engine import BaseTTSEngine
from srt_dubbing.src.logger import get_logger
from srt_dubbing.src.config import F5TTS as F5TTSConfig
import torch
import librosa
from f5_tts.api import F5TTS

logger = get_logger()

# 从F5TTS文档中提取的有效推理参数
F5TTS_INFER_PARAMS = [
    "target_rms", "cross_fade_duration", "sway_sampling_coef", "cfg_strength",
    "nfe_step", "speed", "fix_duration", "remove_silence"
]


class F5TTSEngine(BaseTTSEngine):
    """F5TTS引擎的实现 (遵循F5TTS_infer.md)"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化F5TTS引擎。
        注意：'config' 参数在此实现中被忽略，配置直接从config模块获取。
        """
        logger.step("加载F5TTS模型...")
        try:
            # 直接从配置模块获取初始化参数
            init_kwargs = F5TTSConfig.get_init_kwargs()
            # 过滤掉值为None的参数
            init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

            logger.debug(f"F5TTS初始化参数: {init_kwargs}")
            self.tts_model = F5TTS(**init_kwargs)
            logger.success("F5TTS模型加载成功")
        except Exception as e:
            logger.error(f"F5TTS模型加载失败: {e}", exc_info=True)
            raise RuntimeError(f"加载F5TTS模型失败: {e}")

    def synthesize(self, text: str, **kwargs) -> Tuple[np.ndarray, int]:
        voice_wav = kwargs.pop("voice_wav", None)
        if not voice_wav:
            raise ValueError("F5TTS引擎的 `synthesize` 方法需要 'voice_wav' 参数。")

        # F5TTS需要参考文本，我们使用它的转录功能自动获取
        try:
            ref_text = self.tts_model.transcribe(voice_wav)
            logger.debug(f"自动转录参考音频文本: {ref_text}")
        except Exception as e:
            logger.warning(f"无法自动转录参考音频: {e}。将使用默认的占位符文本。", exc_info=True)
            ref_text = "Reference audio."

        # 筛选出F5TTS infer方法所需的参数
        infer_kwargs = {
            key: value for key, value in kwargs.items() if key in F5TTS_INFER_PARAMS
        }

        wav, sr, _ = self.tts_model.infer(
            ref_file=voice_wav,
            ref_text=ref_text,
            gen_text=text,
            **infer_kwargs
        )

        # F5TTS返回的是torch.Tensor，需要转换为numpy array
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()

        if wav is None:
            raise RuntimeError("TTS引擎返回了空的音频数据。")

        return wav.astype(np.float32), sr

    def synthesize_to_duration(self, text: str, target_duration: float, **kwargs) -> Tuple[np.ndarray, int]:
        """
        （可选）合成一个精确匹配目标时长的音频。
        此实现利用F5TTS的 'fix_duration' 参数。
        """
        voice_wav = kwargs.get("voice_wav")
        if not voice_wav:
            raise ValueError("synthesize_to_duration 需要 'voice_wav' 参数。")

        try:
            # 获取参考音频的时长
            ref_duration = librosa.get_duration(path=voice_wav)
        except Exception as e:
            logger.error(f"无法读取参考音频 '{voice_wav}' 的时长: {e}", exc_info=True)
            # 如果无法获取时长，则无法使用 fix_duration，抛出错误
            raise RuntimeError("无法获取参考音频时长，无法使用 'fix_duration'。") from e

        # F5TTS文档建议参考音频在12秒内，如果超过则会截断，可能导致时长不准
        if ref_duration > 12.0:
            logger.warning(
                f"参考音频时长 ({ref_duration:.2f}s) 超过F5TTS建议的12s。"
                f"模型可能会自动截断音频，导致最终时长不精确。"
            )

        # 'fix_duration' 是参考音频和生成音频的总时长
        total_duration = ref_duration + target_duration
        logger.info(
            f"[F5TTS-Adaptive] 参考音: {ref_duration:.2f}s, "
            f"字幕目标: {target_duration:.2f}s, "
            f"引擎目标(fix_duration): {total_duration:.2f}s"
        )

        synthesis_kwargs = kwargs.copy()
        synthesis_kwargs['fix_duration'] = total_duration

        # 调用自身的synthesize方法，它会处理参数过滤和实际的TTS调用
        return self.synthesize(text, **synthesis_kwargs)


    @staticmethod
    def get_config_model() -> Dict[str, Any]:
        """返回F5TTS引擎的默认配置模型"""
        # 直接返回配置模块中定义的默认值
        return F5TTSConfig.get_init_kwargs() 