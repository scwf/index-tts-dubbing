"""
音频处理器

提供音频合成、合并、格式转换等功能，支持多种音频格式的输入输出。
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import soundfile as sf

# 使用绝对导入
from srt_dubbing.src.config import AUDIO
from srt_dubbing.src.utils import create_directory_if_needed
from srt_dubbing.src.logger import get_logger


class AudioProcessor:
    """音频处理器类"""
    
    def __init__(self, sample_rate: Optional[int] = None, channels: Optional[int] = None):
        """
        初始化音频处理器
        
        Args:
            sample_rate: 采样率
            channels: 声道数
        """
        self.sample_rate = sample_rate or AUDIO.DEFAULT_SAMPLE_RATE
        self.channels = channels or AUDIO.DEFAULT_CHANNELS
        self.audio_segments: List[Dict[str, Any]] = []
    
    def merge_audio_segments(self, segments: List[Dict[str, Any]], 
                           strategy_name: str = "stretch",
                           allow_overlap: bool = True,
                           verbose: bool = False) -> np.ndarray:
        """
        根据策略类型合并音频片段
        
        Args:
            segments: 音频片段列表，每个包含：
                - audio_data: 音频数据
                - start_time: 开始时间
                - end_time: 结束时间
            strategy_name: 策略名称 ("basic", "hq_stretch", "stretch")
            allow_overlap: 是否允许音频重叠（仅对stretch策略有效）
            verbose: 是否输出详细信息
        
        Returns:
            合并后的音频数据
        """
        if not segments:
            return np.array([])
        
        logger = get_logger()
        
        # 根据策略选择合并方式
        if strategy_name in ["basic", "hq_stretch", "iterative", "adaptive"]:
            if verbose:
                logger.debug(f"使用自然拼接模式进行音频合并 (策略: {strategy_name})")
            return self._natural_concatenation(segments, verbose)
        else:
            if verbose:
                logger.debug(f"使用时间同步模式进行音频合并 (策略: {strategy_name})")
            return self._time_synchronized_merge(segments, allow_overlap, verbose)

    
    def _natural_concatenation(self, segments: List[Dict[str, Any]], 
                              verbose: bool = False) -> np.ndarray:
        """
        自然拼接模式：按字幕顺序连续拼接音频，忽略时间约束
        
        适用于basic和hq_stretch策略，优先保证语音的自然流畅性
        
        Args:
            segments: 音频片段列表
            verbose: 是否输出详细信息
            
        Returns:
            拼接后的音频数据
        """
        logger = get_logger()
        
        # 按字幕索引排序（而不是时间）
        sorted_segments = sorted(segments, key=lambda x: x.get('index', 0))
        
        if verbose:
            logger.debug("自然拼接模式详情:")
            total_expected_duration = sum(seg.get('duration', 0) for seg in sorted_segments)
            logger.debug(f"  字幕总时长: {total_expected_duration:.2f}s")
        
        # 收集所有有效的音频数据
        audio_parts = []
        total_duration = 0.0
        
        for i, segment in enumerate(sorted_segments):
            audio_data = segment['audio_data']
            
            # 确保音频数据是numpy数组
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data, dtype=np.float32)
            
            # 检查音频数据是否有效
            if len(audio_data) == 0:
                if verbose:
                    logger.warning(f"片段 {i+1} (条目 {segment.get('index', '?')}) 音频数据为空，跳过")
                continue
            
            audio_parts.append(audio_data)
            actual_duration = len(audio_data) / self.sample_rate
            total_duration += actual_duration
            
            if verbose:
                expected_duration = segment.get('duration', 0)
                text_preview = segment.get('text', '')[:30] + "..." if len(segment.get('text', '')) > 30 else segment.get('text', '')
                logger.debug(f"  片段 {i+1}: {actual_duration:.2f}s (预期{expected_duration:.2f}s) - {text_preview}")
        
        if not audio_parts:
            logger.warning("没有有效的音频数据可供拼接")
            return np.array([])
        
        # 简单直接拼接所有音频片段
        merged_audio = np.concatenate(audio_parts)
        
        if verbose:
            logger.success(f"自然拼接完成: {len(audio_parts)} 个片段，总时长 {total_duration:.2f}s")
            logger.debug(f"  最终音频: {len(merged_audio)} 样本 ({len(merged_audio)/self.sample_rate:.2f}s)")
        
        return merged_audio
    
    def _time_synchronized_merge(self, segments: List[Dict[str, Any]], 
                                allow_overlap: bool = True,
                                verbose: bool = False) -> np.ndarray:
        """
        时间同步合并模式：严格按照字幕时间定位音频片段
        
        适用于stretch策略，优先保证与字幕的时间同步
        
        Args:
            segments: 音频片段列表
            allow_overlap: 是否允许音频重叠
            verbose: 是否输出详细信息
            
        Returns:
            合并后的音频数据
        """
        logger = get_logger()
        
        # 按开始时间排序
        sorted_segments = sorted(segments, key=lambda x: x['start_time'])
        
        if verbose:
            logger.debug("时间同步合并详情:")
            for i, seg in enumerate(sorted_segments):
                audio_len = len(seg['audio_data']) if hasattr(seg['audio_data'], '__len__') else 0
                actual_duration = audio_len / self.sample_rate if audio_len > 0 else 0
                expected_duration = seg.get('end_time', seg['start_time']) - seg['start_time']
                logger.debug(f"  片段 {i+1}: 开始={seg['start_time']:.2f}s, 预期时长={expected_duration:.2f}s, 实际时长={actual_duration:.2f}s")
        
        # 计算总时长 - 考虑音频实际长度
        max_end_time = 0
        for segment in sorted_segments:
            audio_data = segment['audio_data']
            if hasattr(audio_data, '__len__') and len(audio_data) > 0:
                actual_end_time = segment['start_time'] + len(audio_data) / self.sample_rate
                max_end_time = max(max_end_time, actual_end_time)
            else:
                # 使用原始end_time作为后备
                max_end_time = max(max_end_time, segment.get('end_time', segment['start_time']))
        
        total_samples = int(max_end_time * self.sample_rate) + AUDIO.DYNAMIC_BUFFER_SIZE  # 增加一点缓冲
        merged_audio = np.zeros(total_samples, dtype=np.float32)
        
        # 将每个音频片段放置到正确位置
        for i, segment in enumerate(sorted_segments):
            audio_data = segment['audio_data']
            start_sample = int(segment['start_time'] * self.sample_rate)
            
            # 确保音频数据是numpy数组
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data, dtype=np.float32)
            
            # 检查音频数据是否有效
            if len(audio_data) == 0:
                if verbose:
                    logger.warning(f"片段 {i+1} 音频数据为空")
                continue
            
            # 计算可以放置的范围（不截断，但防止越界）
            end_sample = start_sample + len(audio_data)
            
            # 检查是否会重叠
            if not allow_overlap and i > 0:
                prev_segment = sorted_segments[i-1] 
                prev_end_sample = int(prev_segment['start_time'] * self.sample_rate) + len(prev_segment['audio_data'])
                if start_sample < prev_end_sample:
                    overlap_duration = (prev_end_sample - start_sample) / self.sample_rate
                    if verbose:
                        logger.warning(f"片段 {i+1} 与前一片段重叠 {overlap_duration:.2f}s")
                    # 调整开始位置避免重叠
                    start_sample = prev_end_sample
                    end_sample = start_sample + len(audio_data)
            
            # 确保不超出数组边界
            if end_sample > total_samples:
                if verbose:
                    logger.debug(f"片段 {i+1} 超出总时长，延长输出数组")
                # 动态扩展数组
                new_total_samples = end_sample + AUDIO.DYNAMIC_BUFFER_SIZE
                new_merged_audio = np.zeros(new_total_samples, dtype=np.float32)
                new_merged_audio[:len(merged_audio)] = merged_audio
                merged_audio = new_merged_audio
                total_samples = new_total_samples
            
            # 放置音频数据（不截断）
            if allow_overlap:
                # 混音模式：将新音频加到现有音频上
                merged_audio[start_sample:end_sample] += audio_data
            else:
                # 覆盖模式：直接替换
                merged_audio[start_sample:end_sample] = audio_data
            
            if verbose:
                actual_duration = len(audio_data) / self.sample_rate
                logger.debug(f"  ✓ 片段 {i+1} 已放置: {start_sample}-{end_sample} 样本 ({actual_duration:.2f}s)")
        
        # 防止音频过载（混音时可能超过[-1,1]范围）
        if allow_overlap:
            max_val = np.max(np.abs(merged_audio))
            if max_val > AUDIO.MAX_AMPLITUDE:
                merged_audio = merged_audio / max_val
                if verbose:
                    logger.debug(f"音频归一化: 最大值 {max_val:.2f} -> {AUDIO.MAX_AMPLITUDE}")
        
        return merged_audio

    def add_silence_between_segments(self, segments: List[Dict[str, Any]], 
                                   gap_duration: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        在音频片段之间添加静音间隔
        
        Args:
            segments: 音频片段列表
            gap_duration: 间隔时长（秒）
        
        Returns:
            添加间隔后的音频片段列表
        """
        # TODO: 实现静音间隔添加
        # 目前返回原始片段作为占位符
        return segments
    
    def normalize_audio(self, audio_data: np.ndarray, 
                       target_level: Optional[float] = -20.0) -> np.ndarray:
        """
        音频归一化处理
        
        Args:
            audio_data: 音频数据
            target_level: 目标电平（dB）
        
        Returns:
            归一化后的音频数据
        """
        # TODO: 实现音频归一化
        # 目前返回原始数据作为占位符
        return audio_data
    
    def apply_fade(self, audio_data: np.ndarray, 
                   fade_in: Optional[float] = None, fade_out: Optional[float] = None) -> np.ndarray:
        """
        应用淡入淡出效果
        
        Args:
            audio_data: 音频数据
            fade_in: 淡入时长（秒）
            fade_out: 淡出时长（秒）
        
        Returns:
            应用效果后的音频数据
        """
        # TODO: 实现淡入淡出效果
        # 目前返回原始数据作为占位符
        return audio_data
    
    def resample_audio(self, audio_data: np.ndarray, 
                      source_rate: int, target_rate: int) -> np.ndarray:
        """
        音频重采样
        
        Args:
            audio_data: 音频数据
            source_rate: 源采样率
            target_rate: 目标采样率
        
        Returns:
            重采样后的音频数据
        """
        if source_rate == target_rate:
            return audio_data
        
        # 简单的线性插值重采样
        ratio = target_rate / source_rate
        new_length = int(len(audio_data) * ratio)
        
        # 创建新的索引
        old_indices = np.linspace(0, len(audio_data) - 1, new_length)
        new_audio = np.interp(old_indices, np.arange(len(audio_data)), audio_data)
        
        return new_audio.astype(np.float32)
    
    def export_audio(self, audio_data: np.ndarray, 
                    output_path: str, format: str = "wav") -> bool:
        """
        导出音频文件
        
        Args:
            audio_data: 音频数据
            output_path: 输出路径
            format: 音频格式 (wav, mp3, flac等)
        
        Returns:
            导出是否成功
        """
        try:
            # 确保输出目录存在
            create_directory_if_needed(output_path)
            
            # 归一化音频数据到合适范围
            if len(audio_data) > 0:
                # 防止过载，限制在[-1, 1]范围内
                max_val = np.max(np.abs(audio_data))
                if max_val > AUDIO.MAX_AMPLITUDE:
                    audio_data = audio_data / max_val
            
            # 使用soundfile导出音频
            sf.write(output_path, audio_data, self.sample_rate, format=format.upper())
            
            logger = get_logger()
            logger.success(f"音频已导出到: {output_path}")
            return True
            
        except Exception as e:
            logger = get_logger()
            logger.error(f"导出音频失败: {e}")
            return False
    
    def load_audio(self, file_path: str) -> np.ndarray:
        """
        加载音频文件
        
        Args:
            file_path: 音频文件路径
        
        Returns:
            音频数据
        """
        try:
            audio_data, sample_rate = sf.read(file_path)
            
            # 如果采样率不匹配，重采样
            if sample_rate != self.sample_rate:
                audio_data = self.resample_audio(audio_data, sample_rate, self.sample_rate)
            
            # 如果是立体声，转换为单声道
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            return audio_data.astype(np.float32)
            
        except Exception as e:
            logger = get_logger()
            logger.error(f"加载音频文件失败: {e}")
            return np.array([])
    
    def get_audio_info(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        获取音频信息
        
        Args:
            audio_data: 音频数据
        
        Returns:
            音频信息字典
        """
        if len(audio_data) == 0:
            return {'duration': 0, 'peak_level': 0, 'rms_level': 0}
        
        duration = len(audio_data) / self.sample_rate
        peak_level = np.max(np.abs(audio_data))
        rms_level = np.sqrt(np.mean(audio_data ** 2))
        
        return {
            'duration': duration,
            'peak_level': peak_level,
            'rms_level': rms_level,
            'sample_count': len(audio_data)
        } 

    def merge_audio_segments_with_gaps(self, 
                                      segments: List[Dict[str, Any]], 
                                      gap_duration: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        在音频片段间添加间隔
        
        Args:
            segments: 音频片段列表
            gap_duration: 间隔时长（秒）
            
        Returns:
            处理后的音频片段列表
        """
        gap_duration = gap_duration or AUDIO.DEFAULT_GAP_DURATION
        
        if not segments:
            return []
        
        # 添加间隔的逻辑实现
        processed_segments = []
        for i, segment in enumerate(segments):
            processed_segments.append(segment)
            
            # 最后一个片段后不添加间隔
            if i < len(segments) - 1:
                gap_samples = int(gap_duration * self.sample_rate)
                gap_audio = np.zeros(gap_samples, dtype=np.float32)
                
                gap_segment = {
                    'audio_data': gap_audio,
                    'start_time': segment['end_time'],
                    'end_time': segment['end_time'] + gap_duration,
                    'text': '[间隔]',
                    'index': f"{segment['index']}_gap",
                    'duration': gap_duration
                }
                processed_segments.append(gap_segment)
        
        return processed_segments

    def apply_fade_effects(self, audio_data: np.ndarray, 
                          fade_in: Optional[float] = None, fade_out: Optional[float] = None) -> np.ndarray:
        """
        应用淡入淡出效果（使用apply_fade的包装函数）
        
        Args:
            audio_data: 音频数据
            fade_in: 淡入时长（秒）
            fade_out: 淡出时长（秒）
        
        Returns:
            应用效果后的音频数据
        """
        # 调用已有的apply_fade方法
        return self.apply_fade(audio_data, fade_in, fade_out) 