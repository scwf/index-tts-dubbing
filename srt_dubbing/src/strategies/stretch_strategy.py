"""
时间拉伸策略

通过时间拉伸技术，将合成的语音精确匹配到SRT字幕的规定时长，
在保证语音完整性的同时，实现与视频的精确同步。
"""
import numpy as np
import librosa
from typing import List, Dict, Any, Optional
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

try:
    from indextts.infer import IndexTTS
except ImportError:
    print("警告: 无法导入IndexTTS，请确保已正确安装IndexTTS")
    IndexTTS = None

from ..srt_parser import SRTEntry
from .base_strategy import TimeSyncStrategy

class StretchStrategy(TimeSyncStrategy):
    """时间拉伸同步策略实现"""

    def __init__(self, 
                 max_speed_ratio: float = 2.0,
                 min_speed_ratio: float = 0.5):
        """
        初始化时间拉伸策略
        
        Args:
            max_speed_ratio: 最大语速比例 (例如2.0表示最快加速一倍)
            min_speed_ratio: 最小语速比例 (例如0.5表示最慢减速一半)
        """
        self.max_speed_ratio = max_speed_ratio
        self.min_speed_ratio = min_speed_ratio
        self.tts_model = None
        self._indextts_available = IndexTTS is not None
    
    def name(self) -> str:
        """策略名称"""
        return "stretch"

    def description(self) -> str:
        """策略描述"""
        return "时间拉伸策略：通过改变语速来精确匹配字幕时长"

    def process_entries(self, entries: List[SRTEntry], **kwargs) -> List[Dict[str, Any]]:
        """
        处理SRT条目，生成与字幕时长精确匹配的音频片段
        
        Args:
            entries: SRT条目列表
            **kwargs: 可选参数
                - voice_reference: 参考语音文件路径
                - model_dir: 模型目录路径
                - cfg_path: 配置文件路径
                - verbose: 详细输出
        
        Returns:
            音频片段信息列表
        """
        if not self._indextts_available:
            raise RuntimeError("IndexTTS未安装，无法进行语音合成")
            
        if self.tts_model is None:
            model_dir = kwargs.get('model_dir', 'model-dir/index_tts')
            cfg_path = kwargs.get('cfg_path', 'model-dir/index_tts/config.yaml')
            try:
                self.tts_model = IndexTTS(
                    cfg_path=cfg_path,
                    model_dir=model_dir,
                    is_fp16=True
                )
                print(f"成功加载IndexTTS模型: {model_dir}")
            except Exception as e:
                raise RuntimeError(f"加载IndexTTS模型失败: {e}")
        
        voice_reference = kwargs.get('voice_reference')
        if not voice_reference:
            raise ValueError("必须提供参考语音文件路径 (voice_reference)")
        
        verbose = kwargs.get('verbose', False)
        audio_segments = []
        
        print(f"开始处理 {len(entries)} 个字幕条目 (使用stretch策略)...")
        
        for i, entry in enumerate(entries):
            if verbose:
                print(f"处理条目 {i+1}/{len(entries)}: {entry.text[:30]}...")

            try:
                # 1. 合成原始语音
                sampling_rate, audio_data_int16 = self.tts_model.infer(
                    text=entry.text,
                    audio_prompt=voice_reference,
                    output_path=None  # 内存返回模式
                )
                audio_data = audio_data_int16.flatten().astype(np.float32) / 32768.0  # 规范化到 [-1, 1] 范围
                
                # 2. 计算时长和变速比例
                source_duration = len(audio_data) / sampling_rate
                target_duration = entry.duration
                
                if target_duration == 0:
                    rate = 1.0
                else:
                    rate = source_duration / target_duration
                
                # 3. 时间拉伸/压缩
                if abs(rate - 1.0) > 0.05: # 变化超过5%才处理
                    clamped_rate = np.clip(rate, self.min_speed_ratio, self.max_speed_ratio)
                    if abs(clamped_rate - rate) > 0.01 and verbose:
                        print(f"警告: 条目 {entry.index} 变速比 {rate:.2f} 超出范围 [{self.min_speed_ratio}, {self.max_speed_ratio}]，已调整为 {clamped_rate:.2f}")
                    
                    stretched_audio = librosa.effects.time_stretch(audio_data, rate=clamped_rate)
                    
                    # 验证拉伸后的时长
                    actual_duration = len(stretched_audio) / sampling_rate
                    duration_diff = abs(actual_duration - target_duration)
                    
                    if duration_diff > 0.1:
                        if verbose:
                            print(f"信息: 条目 {entry.index} 拉伸后时长 {actual_duration:.2f}s 与目标 {target_duration:.2f}s 有偏差")
                        
                        # 完全不截断策略：只处理音频偏短的情况
                        target_samples = int(target_duration * sampling_rate)
                        current_samples = len(stretched_audio)
                        
                        if current_samples > target_samples:
                            # 音频偏长时：保持完整，不截断
                            overshoot_ratio = (current_samples - target_samples) / target_samples if target_samples > 0 else 0
                            if verbose:
                                print(f"  保持完整语音: 超出目标时长 {overshoot_ratio*100:.1f}% (允许重叠)")
                        elif current_samples < target_samples:
                            # 音频偏短时：填充静音到目标时长
                            padding_samples = target_samples - current_samples
                            padding = np.zeros(padding_samples, dtype=np.float32)
                            stretched_audio = np.concatenate([stretched_audio, padding])
                            if verbose:
                                print(f"  已填充静音: {padding_samples} 样本，达到目标时长")
                else:
                    stretched_audio = audio_data

                # 4. 创建音频片段
                segment = {
                    'audio_data': stretched_audio,
                    'start_time': entry.start_time,
                    'end_time': entry.end_time,
                    'text': entry.text,
                    'index': entry.index,
                    'duration': entry.duration
                }
                audio_segments.append(segment)

            except Exception as e:
                print(f"处理条目 {entry.index} 失败: {e}")
                # 后备方案：创建静音片段
                silence_data = np.zeros(int(entry.duration * sampling_rate), dtype=np.float32)  # 使用动态采样率
                segment = {
                    'audio_data': silence_data,
                    'start_time': entry.start_time,
                    'end_time': entry.end_time,
                    'text': entry.text,
                    'index': entry.index,
                    'duration': entry.duration
                }
                audio_segments.append(segment)
        
        print(f"完成处理，生成 {len(audio_segments)} 个音频片段")
        return audio_segments 