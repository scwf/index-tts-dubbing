"""
策略注册与发现模块

该模块负责自动发现、导入和注册所有可用的时间同步策略。
"""
import os
import importlib
from typing import Dict, Type, List
from srt_dubbing.src.strategies.base_strategy import TimeSyncStrategy

# 全局注册表，用于存储所有发现的策略
_strategy_registry: Dict[str, Type[TimeSyncStrategy]] = {}

def init_strategies():
    """
    自动发现并导入所有策略模块
    """
    if _strategy_registry:
        return

    strategy_dir = os.path.dirname(__file__)
    for filename in os.listdir(strategy_dir):
        if filename.endswith("_strategy.py") and filename != "base_strategy.py":
            module_name = filename[:-3]
            try:
                module = importlib.import_module(f".{module_name}", package=__package__)
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, TimeSyncStrategy) and attr is not TimeSyncStrategy:
                        try:
                            # 尝试无参数实例化以获取名称
                            strategy_instance = attr()
                            strategy_name = strategy_instance.name()
                            if strategy_name not in _strategy_registry:
                                _strategy_registry[strategy_name] = attr
                        except Exception:
                            # 忽略无法无参数实例化的类
                            pass
            except ImportError as e:
                print(f"动态导入策略模块 {module_name} 失败: {e}")


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