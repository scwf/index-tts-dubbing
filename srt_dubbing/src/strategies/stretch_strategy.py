"""
时间拉伸策略

通过时间拉伸技术，将合成的语音精确匹配到SRT字幕的规定时长，
在保证语音完整性的同时，实现与视频的精确同步。
"""
import numpy as np
import librosa
from typing import List, Dict, Any, Optional

# 使用绝对导入，更清晰明确
from srt_dubbing.src.utils import setup_project_path, safe_import_indextts, normalize_audio_data, ProgressLogger
from srt_dubbing.src.config import AUDIO, STRATEGY, MODEL, LOG
from srt_dubbing.src.srt_parser import SRTEntry
from srt_dubbing.src.strategies.base_strategy import TimeSyncStrategy
from srt_dubbing.src.logger import get_logger, create_process_logger

# 初始化项目环境
setup_project_path()
IndexTTS, _indextts_available = safe_import_indextts()

class StretchStrategy(TimeSyncStrategy):
    """时间拉伸同步策略实现"""

    def __init__(self, 
                 max_speed_ratio: float = None,
                 min_speed_ratio: float = None):
        """
        初始化时间拉伸策略
        
        Args:
            max_speed_ratio: 最大语速比例 (例如2.0表示最快加速一倍)
            min_speed_ratio: 最小语速比例 (例如0.5表示最慢减速一半)
        """
        self.max_speed_ratio = max_speed_ratio or STRATEGY.MAX_SPEED_RATIO
        self.min_speed_ratio = min_speed_ratio or STRATEGY.MIN_SPEED_RATIO
        self.tts_model = None
        self._indextts_available = _indextts_available
    
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
            
        logger = get_logger()
        if self.tts_model is None:
            model_dir = kwargs.get('model_dir', MODEL.DEFAULT_MODEL_DIR)
            cfg_path = kwargs.get('cfg_path', MODEL.get_default_config_path(model_dir))
            try:
                logger.step("加载IndexTTS模型")
                self.tts_model = IndexTTS(
                    cfg_path=cfg_path,
                    model_dir=model_dir,
                    is_fp16=MODEL.DEFAULT_FP16
                )
                logger.success(f"IndexTTS模型加载成功: {model_dir}")
            except Exception as e:
                logger.error(f"IndexTTS模型加载失败: {e}")
                raise RuntimeError(f"加载IndexTTS模型失败: {e}")
        
        voice_reference = kwargs.get('voice_reference')
        if not voice_reference:
            raise ValueError("必须提供参考语音文件路径 (voice_reference)")
        
        verbose = kwargs.get('verbose', False)
        audio_segments = []
        
        # 创建处理进度日志器
        process_logger = create_process_logger("时间拉伸策略音频生成")
        process_logger.start(f"处理 {len(entries)} 个字幕条目")
        
        for i, entry in enumerate(entries):
            try:
                # 始终显示进度，不仅仅在verbose模式下
                text_preview = entry.text[:LOG.PROGRESS_TEXT_PREVIEW_LENGTH] + "..." if len(entry.text) > LOG.PROGRESS_TEXT_PREVIEW_LENGTH else entry.text
                process_logger.progress(i + 1, len(entries), f"条目 {entry.index}: {text_preview}")
                
                # 1. 合成原始语音
                sampling_rate, audio_data_int16 = self.tts_model.infer(
                    text=entry.text,
                    audio_prompt=voice_reference,
                    output_path=None  # 内存返回模式
                )
                audio_data = normalize_audio_data(audio_data_int16)  # 规范化到 [-1, 1] 范围
                
                # 2. 计算时长和变速比例
                source_duration = len(audio_data) / sampling_rate
                target_duration = entry.duration
                
                if target_duration == 0:
                    rate = 1.0
                else:
                    rate = source_duration / target_duration
                
                # 3. 时间拉伸/压缩
                if abs(rate - 1.0) > STRATEGY.TIME_STRETCH_THRESHOLD:  # 变化超过阈值才处理
                    clamped_rate = np.clip(rate, self.min_speed_ratio, self.max_speed_ratio)
                    if abs(clamped_rate - rate) > 0.01:
                        # 优化警告信息，使其更清晰
                        speed_type = "加速" if rate > 1.0 else "减速"
                        original_percent = int((rate - 1.0) * 100)
                        adjusted_percent = int((clamped_rate - 1.0) * 100)
                        
                        logger.warning(
                            f"条目 {entry.index} 需要{speed_type} {abs(original_percent)}% 才能匹配字幕时长，"
                            f"但超出安全范围，已限制为{speed_type} {abs(adjusted_percent)}%"
                            f"（原始变速比: {rate:.2f} → 调整后: {clamped_rate:.2f}）"
                        )
                    
                    stretched_audio = librosa.effects.time_stretch(audio_data, rate=clamped_rate)
                    
                    # 验证拉伸后的时长
                    actual_duration = len(stretched_audio) / sampling_rate
                    duration_diff = abs(actual_duration - target_duration)
                    
                    if duration_diff > STRATEGY.TIME_DURATION_TOLERANCE:
                        if verbose:
                            logger.debug(f"条目 {entry.index} 拉伸后时长 {actual_duration:.2f}s 与目标 {target_duration:.2f}s 有偏差")
                        
                        # 完全不截断策略：只处理音频偏短的情况
                        target_samples = int(target_duration * sampling_rate)
                        current_samples = len(stretched_audio)
                        
                        if current_samples > target_samples:
                            # 音频偏长时：保持完整，不截断
                            overshoot_ratio = (current_samples - target_samples) / target_samples if target_samples > 0 else 0
                            if verbose:
                                logger.debug(f"  保持完整语音: 超出目标时长 {overshoot_ratio*100:.1f}% (允许重叠)")
                        elif current_samples < target_samples:
                            # 音频偏短时：填充静音到目标时长
                            padding_samples = target_samples - current_samples
                            padding = np.zeros(padding_samples, dtype=np.float32)
                            stretched_audio = np.concatenate([stretched_audio, padding])
                            if verbose:
                                logger.debug(f"  已填充静音: {padding_samples} 样本，达到目标时长")
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
                logger.error(f"条目 {entry.index} 处理失败: {e}")
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
        
        process_logger.complete(f"生成 {len(audio_segments)} 个音频片段")
        return audio_segments 