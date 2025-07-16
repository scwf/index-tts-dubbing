"""
策略抽象基类
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any

# 使用绝对导入和新的工具模块
from srt_dubbing.src.utils import setup_project_path
from srt_dubbing.src.srt_parser import SRTEntry

# 初始化项目环境
setup_project_path()


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