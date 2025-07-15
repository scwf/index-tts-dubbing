"""
命令行接口

提供SRT配音工具的命令行访问接口，支持各种参数配置和批量处理。
"""

import argparse
import sys
import os
from typing import List

# 确保项目根目录在sys.path中，以便于模块解析
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .srt_parser import SRTParser
from .strategies import get_strategy, list_strategies, init_strategies
from .audio_processor import AudioProcessor


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
        default="output/output.wav",
        help="输出的音频文件路径 (默认: output/output.wav)"
    )
    parser.add_argument(
        "--strategy",
        default="stretch",
        choices=available_strategies,
        help=(
            "选择使用的时间同步策略。\n"
            f"可选策略: {', '.join(available_strategies)}\n"
            "  - basic: 自然合成，可能与字幕时长不完全匹配。\n"
            "  - stretch: 通过时间拉伸精确匹配字幕时长，保证同步。\n"
            "(默认: basic)"
        )
    )
    parser.add_argument(
        "--model-dir",
        default="model-dir/index_tts",
        help="IndexTTS模型目录路径"
    )
    parser.add_argument(
        "--cfg-path",
        help="IndexTTS模型配置文件路径 (默认: model-dir/index_tts/config.yaml)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="启用详细输出模式"
    )
    
    args = parser.parse_args()

    # 处理配置文件路径
    cfg_path = args.cfg_path or os.path.join(args.model_dir, "config.yaml")

    # 1. 解析SRT文件
    try:
        parser_instance = SRTParser()
        entries = parser_instance.parse_file(args.srt)
        print(f"成功解析 {len(entries)} 个字幕条目")
    except Exception as e:
        print(f"错误: 解析SRT文件失败 - {e}")
        return 1

    # 2. 选择并初始化策略
    try:
        strategy = get_strategy(args.strategy)
        print(f"使用策略: {strategy.name()} - {strategy.description()}")
    except ValueError as e:
        print(f"错误: {e}")
        return 1
        
    print("开始处理...")

    # 3. 处理字幕条目生成音频片段
    try:
        audio_segments = strategy.process_entries(
            entries,
            voice_reference=args.voice,
            model_dir=args.model_dir,
            cfg_path=cfg_path,
            verbose=args.verbose
        )
        print(f"成功生成 {len(audio_segments)} 个音频片段")
    except Exception as e:
        print(f"错误: 处理音频时发生错误 - {e}")
        return 1
        
    # 4. 合并音频片段
    processor = AudioProcessor()
    try:
        merged_audio = processor.merge_audio_segments(
            audio_segments,
            allow_overlap=True,  # 允许重叠以保证完整性
            verbose=args.verbose
        )
        
        # 5. 导出最终音频
        if not processor.export_audio(merged_audio, args.output):
            print("错误: 导出音频失败")
            return 1
            
    except Exception as e:
        print(f"错误: 合并或导出音频时出错 - {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    print(f"\n处理完成！配音文件已保存至: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 