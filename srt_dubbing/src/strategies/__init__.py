"""
时间同步策略模块

提供多种时间同步策略，用于处理SRT字幕与音频的时间匹配：
- BasicStrategy: 自然合成策略（基础版本）
- SpeedAdjustStrategy: 语速调节策略（未来扩展）
- HybridStrategy: 混合智能策略（未来扩展）
"""

import pkgutil
import importlib
from typing import Dict, Type, List
from .base_strategy import TimeSyncStrategy

# 全局注册表，用于存储所有发现的策略
_strategy_registry: Dict[str, Type[TimeSyncStrategy]] = {}

def init_strategies():
    """
    自动发现并导入所有策略模块
    """
    if _strategy_registry:
        return

    # 动态发现和加载策略
    package_path = __path__
    package_name = __name__

    for _, module_name, _ in pkgutil.walk_packages(package_path, prefix=f"{package_name}."):
        if module_name.endswith(('_strategy')):
            try:
                module = importlib.import_module(module_name)
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, TimeSyncStrategy) and attr is not TimeSyncStrategy:
                        # 实例化策略以获取名称
                        try:
                            strategy_instance = attr()
                            strategy_name = strategy_instance.name()
                            if strategy_name not in _strategy_registry:
                                _strategy_registry[strategy_name] = attr
                        except Exception:
                            # 忽略无法无参数实例化的类
                            pass
            except ImportError:
                print(f"警告: 导入策略模块 {module_name} 失败")

def get_strategy(name: str) -> TimeSyncStrategy:
    """
    根据名称获取策略实例
    """
    strategy_class = _strategy_registry.get(name)
    if not strategy_class:
        raise ValueError(f"未找到名为 '{name}' 的策略。可用策略: {list_strategies()}")
    return strategy_class()

def list_strategies() -> List[str]:
    """
    列出所有可用的策略名称
    """
    return sorted(list(_strategy_registry.keys())) 