"""
策略注册与发现模块

简化的策略注册系统，每个策略模块自行注册。
"""
from typing import Dict, Type, List
from srt_dubbing.src.strategies.base_strategy import TimeSyncStrategy

# 全局注册表，用于存储所有策略
_strategy_registry: Dict[str, Type[TimeSyncStrategy]] = {}

def init_strategies():
    """
    初始化所有策略（导入模块以触发自动注册）
    """
    if _strategy_registry:
        return
    
    # 导入所有策略模块，触发它们的自动注册
    try:
        from . import basic_strategy
        from . import stretch_strategy  
        from . import hq_stretch_strategy
        from . import iterative_strategy
        from . import intelligent_stretch_strategy
    except ImportError as e:
        print(f"导入策略模块失败: {e}")

def get_strategy(name: str) -> TimeSyncStrategy:
    """
    根据名称获取策略实例
    
    Args:
        name: 策略名称
        
    Returns:
        策略实例
        
    Raises:
        ValueError: 找不到指定策略时抛出
    """
    if not _strategy_registry:
        init_strategies()
        
    strategy_class = _strategy_registry.get(name)
    if not strategy_class:
        raise ValueError(f"未找到名为 '{name}' 的策略。可用策略: {list_strategies()}")
    return strategy_class()

def list_strategies() -> List[str]:
    """
    获取所有可用策略名称列表
    
    Returns:
        策略名称列表
    """
    if not _strategy_registry:
        init_strategies()
    return sorted(list(_strategy_registry.keys()))

def get_strategy_info() -> Dict[str, str]:
    """
    获取所有策略的信息
    
    Returns:
        策略名称到描述的映射
    """
    if not _strategy_registry:
        init_strategies()
    
    info = {}
    for name, strategy_class in _strategy_registry.items():
        try:
            # 临时实例化获取描述
            instance = strategy_class()
            info[name] = instance.description()
        except Exception:
            info[name] = "策略描述获取失败"
    return info 