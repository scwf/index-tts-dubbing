from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any

class BaseTTSEngine(ABC):
    """TTS引擎的抽象基类"""

    @abstractmethod
    def __init__(self):
        """
        初始化引擎。
        :param config: 引擎所需的特定配置字典。
        """
        pass

    @abstractmethod
    def synthesize(self, text: str, **kwargs) -> Tuple[np.ndarray, int]:
        """
        将文本合成为音频。

        :param text: 需要合成的文本。
        :param kwargs: 引擎特定的其他参数 (例如参考音频, 语速等)。
        :return: 一个元组，包含音频数据 (NumPy array) 和采样率 (int)。
        """
        pass

    def synthesize_to_duration(self, text: str, target_duration: float, **kwargs) -> Tuple[np.ndarray, int]:
        """
        （可选）合成一个精确匹配目标时长的音频。
        引擎内部应实现自己的迭代或优化逻辑。

        :param text: 需要合成的文本。
        :param target_duration: 目标时长（秒）。
        :param kwargs: 引擎特定的其他参数。
        :return: 一个元组，包含音频数据 (NumPy array) 和采样率 (int)。
        :raises: NotImplementedError 如果引擎不支持此功能。
        """
        raise NotImplementedError(f"引擎 '{type(self).__name__}' 不支持自适应时长合成。")