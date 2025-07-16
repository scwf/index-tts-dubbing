"""
命令行接口

提供SRT配音工具的命令行访问接口，支持各种参数配置和批量处理。
"""

import argparse
import os
from typing import List

# 使用绝对导入
from srt_dubbing.src.utils import setup_project_path
from srt_dubbing.src.config import MODEL, PATH, LOG
from srt_dubbing.src.srt_parser import SRTParser
from srt_dubbing.src.strategies import get_strategy, list_strategies, init_strategies
from srt_dubbing.src.audio_processor import AudioProcessor
from srt_dubbing.src.logger import setup_logging, create_process_logger

# 初始化项目环境
setup_project_path()


def main():
    """主函数：解析命令行参数并启动处理流程"""
    
    # 初始化所有可用策略
    init_strategies()
    
    available_strategies = list_strategies()
    
    parser = argparse.ArgumentParser(
        description="SRT字幕配音工具",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--srt",
        required=True,
        help="待处理的SRT字幕文件路径"
    )
    parser.add_argument(
        "--voice",
        required=True,
        help="参考语音WAV文件路径"
    )
    parser.add_argument(
        "--output",
        default=PATH.get_default_output_path(),
        help=f"输出的音频文件路径 (默认: {PATH.get_default_output_path()})"
    )
    parser.add_argument(
        "--strategy",
        default="stretch",
        choices=available_strategies,
        help=(
            "选择使用的时间同步策略。\n"
            f"可选策略: {', '.join(available_strategies)}\n"
            "  - basic: 自然合成，时长变化自然，音质最佳 (⭐⭐⭐⭐⭐)\n"
            "  - stretch: 通过时间拉伸精确匹配字幕时长，保证同步 (⭐⭐⭐)\n"
            "  - hq_stretch: 高质量拉伸，平衡音质与同步性 (⭐⭐⭐⭐)\n"
            "(默认: stretch)"
        )
    )
    parser.add_argument(
        "--model-dir",
        default=MODEL.DEFAULT_MODEL_DIR,
        help="IndexTTS模型目录路径"
    )
    parser.add_argument(
        "--cfg-path",
        help=f"IndexTTS模型配置文件路径 (默认: {MODEL.get_default_config_path()})"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="启用详细输出模式"
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level)
    
    # 创建主处理流程日志器
    process_logger = create_process_logger("SRT配音")
    process_logger.start(f"输入文件: {args.srt}, 策略: {args.strategy}")

    # 处理配置文件路径
    cfg_path = args.cfg_path or MODEL.get_default_config_path(args.model_dir)
    logger.debug(f"使用配置文件: {cfg_path}")

    # 1. 解析SRT文件
    try:
        process_logger.step("解析SRT文件", args.verbose)
        parser_instance = SRTParser()
        entries = parser_instance.parse_file(args.srt)
        logger.success(f"成功解析 {len(entries)} 个字幕条目")
    except Exception as e:
        logger.error(f"解析SRT文件失败: {e}")
        return 1

    # 2. 选择并初始化策略
    try:
        process_logger.step("初始化处理策略", args.verbose)
        strategy = get_strategy(args.strategy)
        logger.info(f"使用策略: {strategy.name()} - {strategy.description()}")
    except ValueError as e:
        logger.error(f"策略初始化失败: {e}")
        return 1

    # 3. 处理字幕条目生成音频片段
    try:
        process_logger.step("生成音频片段", args.verbose)
        audio_segments = strategy.process_entries(
            entries,
            voice_reference=args.voice,
            model_dir=args.model_dir,
            cfg_path=cfg_path,
            verbose=args.verbose
        )
        logger.success(f"成功生成 {len(audio_segments)} 个音频片段")
    except Exception as e:
        logger.error(f"音频生成失败: {e}")
        return 1
        
    # 4. 合并音频片段
    try:
        process_logger.step("合并音频片段", args.verbose)
        processor = AudioProcessor()
        merged_audio = processor.merge_audio_segments(
            audio_segments,
            strategy_name=args.strategy,  # 传递策略名称
            allow_overlap=True,  # 允许重叠以保证完整性（仅对stretch策略有效）
            verbose=args.verbose
        )
        
        # 5. 导出最终音频
        process_logger.step("导出音频文件", args.verbose)
        if not processor.export_audio(merged_audio, args.output):
            logger.error("音频导出失败")
            return 1
            
    except Exception as e:
        logger.error(f"音频处理失败: {e}")
        if args.verbose:
            import traceback
            logger.debug("详细错误信息:")
            logger.debug(traceback.format_exc())
        return 1
    
    process_logger.complete(f"配音文件已保存至: {args.output}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main()) 