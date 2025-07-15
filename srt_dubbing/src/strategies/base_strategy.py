"""
策略抽象基类
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from srt_dubbing.src.srt_parser import SRTEntry
except ImportError:
    # 兼容直接从src目录运行的情况
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from srt_parser import SRTEntry


class TimeSyncStrategy(ABC):
    """时间同步策略抽象基类"""
    
    @abstractmethod
    def name(self) -> str:
        """策略名称"""
        pass
    
    @abstractmethod
    def description(self) -> str:
        """策略描述"""
        pass
    
    @abstractmethod
    def process_entries(self, entries: List[SRTEntry], **kwargs) -> List[Dict[str, Any]]:
        """处理SRT条目，返回音频片段信息"""
        pass 