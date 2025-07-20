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

class IndexTTSEngine(BaseTTSEngine):
    """IndexTTS引擎的实现"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化IndexTTS引擎。
        
        :param config: 引擎配置，需要包含 'model_dir' 和可选的 'cfg_path'。
        """
        if IndexTTS is None:
            raise ImportError("IndexTTS未安装，请执行 `pip install indextts-fork` 进行安装。")
            
        logger = get_logger()
        self.model_dir = config.get("model_dir", MODEL.DEFAULT_MODEL_DIR)
        self.cfg_path = config.get("cfg_path") or MODEL.get_default_config_path(self.model_dir)
        self.is_fp16 = config.get("is_fp16", MODEL.DEFAULT_FP16)
        
        logger.step(f"加载IndexTTS模型...")
        try:
            self.tts_model = IndexTTS(
                cfg_path=self.cfg_path,
                model_dir=self.model_dir,
                is_fp16=self.is_fp16
            )
            logger.success(f"IndexTTS模型加载成功: {self.model_dir}")
            logger.debug(f"使用配置文件: {self.cfg_path}")
        except Exception as e:
            logger.error(f"IndexTTS模型加载失败: {e}")
            raise RuntimeError(f"加载IndexTTS模型失败: {e}")

    def synthesize(self, text: str, **kwargs) -> Tuple[np.ndarray, int]:
        """
        使用IndexTTS合成音频。

        :param text: 要合成的文本。
        :param kwargs: 必须包含 'voice_wav' (参考音频路径)。
        :return: 一个元组，包含音频数据 (NumPy array, 范围 -1 到 1) 和采样率 (int)。
        """
        voice_wav = kwargs.get("voice_wav")
        if not voice_wav:
            raise ValueError("IndexTTS引擎的 `synthesize` 方法需要 'voice_wav' 参数。")

        sampling_rate, audio_data_int16 = self.tts_model.infer(
            text=text,
            audio_prompt=voice_wav,
            output_path=None  # 内存返回模式
        )
        
        # 将int16格式的音频数据规范化到 [-1, 1] 的float32格式
        audio_data_float32 = normalize_audio_data(audio_data_int16)
        
        return audio_data_float32, sampling_rate

    @staticmethod
    def get_config_model() -> Dict[str, Any]:
        """返回IndexTTS引擎的默认配置模型"""
        return {
            "model_dir": MODEL.DEFAULT_MODEL_DIR,
            "cfg_path": None,
            "is_fp16": MODEL.DEFAULT_FP16
        } 