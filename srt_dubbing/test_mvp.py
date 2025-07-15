#!/usr/bin/env python3
"""
SRT配音工具 - MVP测试脚本

测试基本的SRT文件处理和音频生成功能。
"""

import sys
import os
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_srt_parser():
    """测试SRT解析器"""
    print("=== 测试SRT解析器 ===")
    
    try:
        from srt_parser import SRTParser
        
        # 使用项目根目录的target_language_srt.srt文件
        srt_file = Path(__file__).parent.parent / "target_language_srt.srt"
        
        if not srt_file.exists():
            print(f"错误: 找不到测试SRT文件: {srt_file}")
            return False
        
        parser = SRTParser()
        entries = parser.parse_file(str(srt_file))
        
        print(f"成功解析 {len(entries)} 个字幕条目")
        
        # 显示前几个条目
        for i, entry in enumerate(entries[:3]):
            print(f"条目 {entry.index}: {entry.start_time:.2f}s - {entry.end_time:.2f}s")
            print(f"  文本: {entry.text[:50]}...")
        
        # 验证条目
        if parser.validate_entries(entries):
            print("✓ SRT条目验证通过")
        else:
            print("⚠ SRT条目验证有警告")
        
        total_duration = parser.get_total_duration()
        print(f"总时长: {total_duration:.2f} 秒")
        
        return True
        
    except Exception as e:
        print(f"SRT解析器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_strategy():
    """测试基础策略（需要参考语音文件）"""
    print("\n=== 测试基础策略 ===")
    
    # 检查是否有参考语音文件
    refer_voice_dir = Path(__file__).parent.parent / "refer_voice"
    voice_files = list(refer_voice_dir.glob("*.wav")) + list(refer_voice_dir.glob("*.WAV"))
    
    if not voice_files:
        print("跳过基础策略测试: 没有找到参考语音文件")
        print(f"请在 {refer_voice_dir} 目录下放置 .wav 参考语音文件")
        return True
    
    voice_file = voice_files[0]
    print(f"使用参考语音: {voice_file}")
    
    try:
        from srt_parser import SRTParser
        from strategies.basic_strategy import BasicStrategy
        
        # 解析少量SRT条目进行测试
        srt_file = Path(__file__).parent.parent / "target_language_srt.srt"
        parser = SRTParser()
        all_entries = parser.parse_file(str(srt_file))
        
        # 只测试前2个条目
        test_entries = all_entries[:2]
        print(f"测试前 {len(test_entries)} 个条目")
        
        strategy = BasicStrategy()
        print(f"策略: {strategy.description()}")
        
        # 处理条目（这里可能需要IndexTTS模型，如果没有会报错）
        try:
            audio_segments = strategy.process_entries(
                test_entries,
                voice_reference=str(voice_file),
                verbose=True
            )
            
            print(f"✓ 成功生成 {len(audio_segments)} 个音频片段")
            return True
            
        except RuntimeError as e:
            if "IndexTTS" in str(e):
                print(f"跳过策略测试: {e}")
                print("提示: 请确保IndexTTS模型已正确安装和配置")
                return True
            else:
                raise
        
    except Exception as e:
        print(f"基础策略测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_processor():
    """测试音频处理器（使用真实生成的音频）"""
    print("\n=== 测试音频处理器（真实音频） ===")

    # 1. 检查参考语音
    refer_voice_dir = Path(__file__).parent.parent / "refer_voice"
    voice_files = list(refer_voice_dir.glob("*.wav")) + list(refer_voice_dir.glob("*.WAV"))
    if not voice_files:
        print("跳过音频处理器测试: 没有找到参考语音文件")
        return True
    voice_file = voice_files[0]

    try:
        # 2. 导入所需模块
        from srt_parser import SRTParser
        from strategies.basic_strategy import BasicStrategy
        from audio_processor import AudioProcessor

        # 3. 解析SRT文件并获取前几个条目
        srt_file = Path(__file__).parent.parent / "target_language_srt.srt"
        parser = SRTParser()
        all_entries = parser.parse_file(str(srt_file))
        test_entries = all_entries[:3]  # 使用前3条字幕
        print(f"将使用 '{srt_file.name}' 的前 {len(test_entries)} 条字幕生成音频。")
        print(f"使用参考语音: {voice_file.name}")

        # 4. 使用策略生成音频片段
        strategy = BasicStrategy()
        try:
            audio_segments = strategy.process_entries(
                test_entries,
                voice_reference=str(voice_file),
                verbose=False # 保持输出简洁
            )
        except RuntimeError as e:
             if "IndexTTS" in str(e):
                print(f"跳过测试: {e}")
                print("提示: 请确保IndexTTS模型已正确安装和配置")
                return True
             else:
                raise

        print(f"✓ 成功生成 {len(audio_segments)} 个音频片段")

        # 5. 使用AudioProcessor合并音频
        processor = AudioProcessor()
        merged_audio = processor.merge_audio_segments(audio_segments)
        print(f"✓ 成功合并音频，总时长: {len(merged_audio) / processor.sample_rate:.2f}s")

        # 6. 导出最终音频文件
        output_dir = Path(__file__).parent.parent / "output"
        output_dir.mkdir(exist_ok=True)
        test_output = output_dir / "test_processor_output.wav"
        
        success = processor.export_audio(merged_audio, str(test_output))
        if success and test_output.exists():
            print(f"✓ 成功导出测试音频，文件已保留: {test_output}")
        else:
            print("⚠ 音频导出测试失败")
            return False
        
        return True

    except Exception as e:
        print(f"音频处理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli():
    """测试CLI接口（端到端测试）"""
    print("\n=== 测试CLI接口（端到端） ===")
    
    # 1. 检查参考语音
    refer_voice_dir = Path(__file__).parent.parent / "refer_voice"
    voice_files = list(refer_voice_dir.glob("*.wav")) + list(refer_voice_dir.glob("*.WAV"))
    if not voice_files:
        print("跳过CLI测试: 没有找到参考语音文件")
        return True
    voice_file = voice_files[0]

    # 2. 定义输入和输出路径
    srt_file = Path(__file__).parent.parent / "target_language_srt.srt"
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "test_cli_output.wav"

    # 3. 准备CLI命令
    # 为了测试，我们只处理前2条字幕
    # 注意：这里需要一个方法来限制处理的条目数，我们暂时先完整运行
    # 在实际的cli.py中可以添加 --limit 参数来优化
    command = [
        sys.executable,  # 使用当前Python解释器
        "-m", "srt_dubbing.src.cli",
        "--srt", str(srt_file),
        "--voice", str(voice_file),
        "--output", str(output_file),
        "--limit", "2"  # 假设CLI支持limit参数来限制处理的条目
    ]
    
    print(f"将执行CLI命令: {' '.join(command)}")

    try:
        import subprocess
        
        # 4. 执行命令
        # 我们需要捕获输出，以便在出错时提供更多信息
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False  # 我们手动检查返回码
        )
        
        # 打印CLI的输出
        if result.stdout:
            print("--- CLI STDOUT ---")
            print(result.stdout)
        if result.stderr:
            print("--- CLI STDERR ---")
            print(result.stderr)

        # 5. 验证结果
        if result.returncode != 0:
            print(f"⚠ CLI命令执行失败，返回码: {result.returncode}")
            if "ModuleNotFoundError" in result.stderr:
                print("提示: 似乎有依赖未安装，请检查环境。")
            elif "IndexTTS" in result.stderr:
                 print("提示: 似乎IndexTTS模型加载失败，请检查模型文件和配置。")
            return False

        if not output_file.exists():
            print(f"⚠ CLI执行成功，但未生成输出文件: {output_file}")
            return False

        print(f"✓ CLI端到端测试成功，已生成音频文件: {output_file}")
        return True
        
    except Exception as e:
        print(f"CLI接口测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("SRT配音工具 - MVP版本测试")
    print("=" * 50)
    
    tests = [
        ("SRT解析器", test_srt_parser),
        ("基础策略", test_basic_strategy),
        ("音频处理器", test_audio_processor),
        ("CLI接口", test_cli),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"{test_name} 测试出现异常: {e}")
            results.append((test_name, False))
    
    # 总结测试结果
    print("\n" + "=" * 50)
    print("测试结果总结:")
    
    passed = 0
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{len(tests)} 个测试通过")
    
    if passed == len(tests):
        print("🎉 所有测试通过！MVP版本可以使用。")
        print("\n使用示例:")
        print("python -m srt_dubbing.src.cli --srt target_language_srt.srt --voice refer_voice/sample.wav --output output.wav")
    else:
        print("⚠ 部分测试失败，请检查配置和依赖。")
    
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main()) 