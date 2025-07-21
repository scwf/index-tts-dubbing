import inspect
from typing import Tuple, Dict, Any
import numpy as np
from .base_engine import BaseTTSEngine
from srt_dubbing.src.logger import get_logger
from srt_dubbing.src.config import F5TTSConfig
import torch
import librosa

# 动态导入F5TTS，如果不存在则给出友好提示
try:
    from f5_tts.api import F5TTS
except ImportError:
    F5TTS = None

logger = get_logger()

class F5TTSEngine(BaseTTSEngine):
    """F5TTS引擎的实现 (遵循F5TTS_infer.md)"""

    def __init__(self):
        """
        初始化F5TTS引擎。
        注意：'config' 参数在此实现中被忽略，配置直接从config模块获取。
        """
        if F5TTS is None:
            raise ImportError("F5TTS未安装，请执行 `pip install f5-tts` 进行安装。")

        logger.step("加载F5TTS模型...")
        try:
            # 直接从配置模块获取初始化参数
            init_kwargs = F5TTSConfig.get_init_kwargs()
            # 过滤掉值为None的参数
            init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

            logger.debug(f"F5TTS初始化参数: {init_kwargs}")
            self.tts_model = F5TTS(**init_kwargs)

            # 使用内省机制，获取底层模型真正支持的参数列表
            infer_signature = inspect.signature(self.tts_model.infer)
            self.valid_infer_params = set(infer_signature.parameters.keys())
            
            logger.success("F5TTS模型加载成功")
        except Exception as e:
            logger.error(f"F5TTS模型加载失败: {e}")
            raise RuntimeError(f"加载F5TTS模型失败: {e}")

    def synthesize(self, text: str, **kwargs) -> Tuple[np.ndarray, int]:
        voice_reference = kwargs.pop('voice_reference')
        if not voice_reference:
            raise ValueError("必须提供参考语音文件路径 (voice_reference)")

        # 优先从kwargs中获取参考文本(prompt_text)，如果未提供，再尝试自动转录
        ref_text = kwargs.pop("ref_text")
        if not ref_text:
            raise ValueError("F5TTS引擎的 `synthesize` 方法需要 'ref_text' 参数。") 

        # 优雅地过滤出底层模型支持的参数
        infer_kwargs = {
            key: value for key, value in kwargs.items() 
            if key in self.valid_infer_params
        }

        wav, sr, _ = self.tts_model.infer(
            ref_file=voice_reference,
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
        voice_reference = kwargs.get('voice_reference')
        if not voice_reference:
            raise ValueError("必须提供参考语音文件路径 (voice_reference)")

        try:
            # 获取参考音频的时长
            ref_duration = librosa.get_duration(path=voice_reference)
        except Exception as e:
            logger.error(f"无法读取参考音频 '{voice_reference}' 的时长: {e}", exc_info=True)
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
