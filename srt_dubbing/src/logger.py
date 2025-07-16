"""
日志系统模块

提供统一的日志配置和接口，支持控制台和文件输出，
包含不同级别的日志记录和美化的进度显示。
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# 尝试导入colorama，如果没有则使用无颜色版本
try:
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
    HAS_COLORAMA = True
except ImportError:
    # 如果没有colorama，定义空的颜色代码
    class Fore:
        CYAN = ""
        GREEN = ""
        YELLOW = ""
        RED = ""
        MAGENTA = ""
    
    class Style:
        RESET_ALL = ""
    
    HAS_COLORAMA = False

from srt_dubbing.src.config import LOG


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    # 定义不同日志级别的颜色
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.MAGENTA
    }
    
    def format(self, record):
        # 获取基础格式化结果
        log_message = super().format(record)
        
        # 添加颜色
        color = self.COLORS.get(record.levelname, '')
        if color and HAS_COLORAMA:
            # 只对控制台输出添加颜色
            if hasattr(record, 'colored') and record.colored:
                log_message = f"{color}{log_message}{Style.RESET_ALL}"
        
        return log_message


class SRTDubbingLogger:
    """SRT配音工具专用日志器"""
    
    def __init__(self, name: str = "srt_dubbing", log_level: str = "INFO"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.log_level = getattr(logging, log_level.upper())
        self.logger.setLevel(self.log_level)
        
        # 避免重复添加处理器
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """设置日志处理器"""
        # 控制台处理器 - 彩色输出
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        
        # 控制台格式 - 简洁版
        console_format = "%(levelname)s: %(message)s"
        console_formatter = ColoredFormatter(console_format)
        console_handler.setFormatter(console_formatter)
        
        # 为控制台记录添加标记，用于彩色显示
        console_handler.addFilter(lambda record: setattr(record, 'colored', True) or True)
        
        self.logger.addHandler(console_handler)
        
        # 文件处理器 - 详细日志（可选）
        # self._setup_file_handler()
    
    def _setup_file_handler(self, log_file: Optional[str] = None):
        """设置文件处理器"""
        if log_file is None:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"srt_dubbing_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别
        
        # 文件格式 - 详细版
        file_format = "%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s"
        file_formatter = logging.Formatter(file_format)
        file_handler.setFormatter(file_formatter)
        
        self.logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs):
        """信息级别日志"""
        self.logger.info(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """调试级别日志"""
        self.logger.debug(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """警告级别日志"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """错误级别日志"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """严重错误级别日志"""
        self.logger.critical(message, **kwargs)
    
    def success(self, message: str, **kwargs):
        """成功消息（使用INFO级别，但带特殊前缀）"""
        self.info(f"✓ {message}", **kwargs)
    
    def step(self, step_name: str, description: str = ""):
        """处理步骤日志"""
        if description:
            self.info(f"🔄 {step_name}: {description}")
        else:
            self.info(f"🔄 {step_name}")
    
    def progress(self, current: int, total: int, description: str = ""):
        """进度日志"""
        percentage = (current / total) * 100 if total > 0 else 0
        progress_bar = self._create_progress_bar(current, total)
        
        if description:
            self.info(f"{progress_bar} {current}/{total} ({percentage:.1f}%) - {description}")
        else:
            self.info(f"{progress_bar} {current}/{total} ({percentage:.1f}%)")
    
    def _create_progress_bar(self, current: int, total: int, width: int = 20) -> str:
        """创建进度条"""
        if total == 0:
            return "[" + "=" * width + "]"
        
        filled = int(width * current / total)
        bar = "=" * filled + "-" * (width - filled)
        return f"[{bar}]"


class ProcessLogger:
    """特定处理过程的日志记录器"""
    
    def __init__(self, logger: SRTDubbingLogger, process_name: str):
        self.logger = logger
        self.process_name = process_name
        self.start_time = datetime.now()
        self.step_count = 0
    
    def start(self, description: str = ""):
        """开始处理"""
        if description:
            self.logger.step(f"开始{self.process_name}", description)
        else:
            self.logger.step(f"开始{self.process_name}")
    
    def step(self, description: str, verbose: bool = False):
        """处理步骤"""
        self.step_count += 1
        if verbose:
            self.logger.debug(f"  步骤 {self.step_count}: {description}")
    
    def progress(self, current: int, total: int, item_description: str = ""):
        """进度更新"""
        self.logger.progress(current, total, item_description)
    
    def warning(self, message: str):
        """处理警告"""
        self.logger.warning(f"{self.process_name} - {message}")
    
    def error(self, message: str):
        """处理错误"""
        self.logger.error(f"{self.process_name} - {message}")
    
    def complete(self, result_description: str = ""):
        """完成处理"""
        duration = datetime.now() - self.start_time
        if result_description:
            self.logger.success(f"{self.process_name}完成: {result_description} (耗时: {duration.total_seconds():.2f}s)")
        else:
            self.logger.success(f"{self.process_name}完成 (耗时: {duration.total_seconds():.2f}s)")


# 全局日志器实例
_global_logger: Optional[SRTDubbingLogger] = None


def get_logger(name: str = "srt_dubbing", log_level: str = "INFO") -> SRTDubbingLogger:
    """获取日志器实例"""
    global _global_logger
    if _global_logger is None:
        _global_logger = SRTDubbingLogger(name, log_level)
    return _global_logger


def setup_logging(log_level: str = "INFO", enable_file_logging: bool = False, log_file: Optional[str] = None):
    """
    设置全局日志配置
    
    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_file_logging: 是否启用文件日志
        log_file: 自定义日志文件路径
    """
    global _global_logger
    _global_logger = SRTDubbingLogger("srt_dubbing", log_level)
    
    if enable_file_logging:
        _global_logger._setup_file_handler(log_file)
    
    return _global_logger


# 便捷函数
def info(message: str, **kwargs):
    """信息日志"""
    get_logger().info(message, **kwargs)


def debug(message: str, **kwargs):
    """调试日志"""
    get_logger().debug(message, **kwargs)


def warning(message: str, **kwargs):
    """警告日志"""
    get_logger().warning(message, **kwargs)


def error(message: str, **kwargs):
    """错误日志"""
    get_logger().error(message, **kwargs)


def success(message: str, **kwargs):
    """成功日志"""
    get_logger().success(message, **kwargs)


def step(step_name: str, description: str = ""):
    """步骤日志"""
    get_logger().step(step_name, description)


def progress(current: int, total: int, description: str = ""):
    """进度日志"""
    get_logger().progress(current, total, description)


def create_process_logger(process_name: str) -> ProcessLogger:
    """创建处理过程日志器"""
    return ProcessLogger(get_logger(), process_name) 