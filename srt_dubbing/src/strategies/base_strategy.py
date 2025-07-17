"""
策略抽象基类
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from indextts.infer import IndexTTS

# 使用绝对导入和新的工具模块
from srt_dubbing.src.utils import setup_project_path
from srt_dubbing.src.srt_parser import SRTEntry

# 初始化项目环境
setup_project_path()


class TimeSyncStrategy(ABC):
    """时间同步策略抽象基类"""
    
    def __init__(self):
        # 类型注解会在运行时确定实际类型
        self.tts_model: Optional['IndexTTS'] = None
    
    def ensure_model_initialized(self, model_dir: str, cfg_path: Optional[str] = None) -> None:
        """
        确保TTS模型已初始化（延迟初始化）
        
        Args:
            model_dir: 模型目录路径
            cfg_path: 配置文件路径
        """
        if self.tts_model is not None:
            return
            
        from indextts.infer import IndexTTS
        from srt_dubbing.src.config import MODEL
        from srt_dubbing.src.logger import get_logger
        
        logger = get_logger()
        cfg_path = cfg_path or MODEL.get_default_config_path(model_dir)
        
        try:
            logger.step(f"加载IndexTTS模型 ({self.name()}策略)")
            self.tts_model = IndexTTS(
                cfg_path=cfg_path,
                model_dir=model_dir,
                is_fp16=MODEL.DEFAULT_FP16
            )
            logger.success(f"IndexTTS模型加载成功: {model_dir}")
            logger.debug(f"使用配置文件: {cfg_path}")
        except Exception as e:
            logger.error(f"IndexTTS模型加载失败: {e}")
            raise RuntimeError(f"加载IndexTTS模型失败: {e}")
    
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