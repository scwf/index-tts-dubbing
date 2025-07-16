"""
高质量时间拉伸策略

在保证音质的前提下进行时间调整，使用更保守的变速范围和优化的音频处理。
当无法在音质保护范围内完成精确匹配时，会优先选择保持音质。
"""
import numpy as np
import librosa
from typing import List, Dict, Any, Optional

# 使用绝对导入
from srt_dubbing.src.utils import setup_project_path, safe_import_indextts, normalize_audio_data
from srt_dubbing.src.config import AUDIO, STRATEGY, MODEL, LOG
from srt_dubbing.src.srt_parser import SRTEntry
from srt_dubbing.src.strategies.base_strategy import TimeSyncStrategy
from srt_dubbing.src.logger import get_logger, create_process_logger

# 初始化项目环境
setup_project_path()
IndexTTS, _indextts_available = safe_import_indextts()


class HighQualityStretchStrategy(TimeSyncStrategy):
    """高质量时间拉伸策略实现"""

    def __init__(self, 
                 max_speed_ratio: float = None,
                 min_speed_ratio: float = None):
        """
        初始化高质量拉伸策略
        
        Args:
            max_speed_ratio: 最大语速比例 (默认: 1.3, 保证音质)
            min_speed_ratio: 最小语速比例 (默认: 0.8, 保证音质)
        """
        # 使用更保守的默认值
        self.max_speed_ratio = max_speed_ratio or STRATEGY.HIGH_QUALITY_MAX_SPEED
        self.min_speed_ratio = min_speed_ratio or STRATEGY.HIGH_QUALITY_MIN_SPEED
        self.tts_model = None
        self._indextts_available = _indextts_available
    
    def name(self) -> str:
        """策略名称"""
        return "hq_stretch"

    def description(self) -> str:
        """策略描述"""
        return "高质量拉伸策略：在保证音质的前提下进行时间调整"

    def process_entries(self, entries: List[SRTEntry], **kwargs) -> List[Dict[str, Any]]:
        """
        处理SRT条目，生成高质量的音频片段
        
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
        process_logger = create_process_logger("高质量拉伸策略音频生成")
        process_logger.start(f"处理 {len(entries)} 个字幕条目")
        
        for i, entry in enumerate(entries):
            try:
                # 显示进度
                text_preview = entry.text[:LOG.PROGRESS_TEXT_PREVIEW_LENGTH] + "..." if len(entry.text) > LOG.PROGRESS_TEXT_PREVIEW_LENGTH else entry.text
                process_logger.progress(i + 1, len(entries), f"条目 {entry.index}: {text_preview}")
                
                # 1. 合成原始语音
                sampling_rate, audio_data_int16 = self.tts_model.infer(
                    text=entry.text,
                    audio_prompt=voice_reference,
                    output_path=None
                )
                audio_data = normalize_audio_data(audio_data_int16)
                
                # 2. 计算时长和变速比例
                source_duration = len(audio_data) / sampling_rate
                target_duration = entry.duration
                
                if target_duration == 0:
                    rate = 1.0
                else:
                    rate = source_duration / target_duration
                
                # 3. 高质量时间调整
                processed_audio = self._high_quality_time_adjustment(
                    audio_data, rate, sampling_rate, entry, verbose, logger
                )
                
                # 4. 创建音频片段
                segment = {
                    'audio_data': processed_audio,
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
                silence_data = np.zeros(int(entry.duration * sampling_rate), dtype=np.float32)
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
    
    def _high_quality_time_adjustment(self, audio_data: np.ndarray, rate: float, 
                                     sampling_rate: int, entry: SRTEntry, 
                                     verbose: bool, logger) -> np.ndarray:
        """
        高质量时间调整处理
        
        Args:
            audio_data: 原始音频数据
            rate: 计算得出的变速比
            sampling_rate: 采样率
            entry: SRT条目
            verbose: 详细模式
            logger: 日志器
            
        Returns:
            处理后的音频数据
        """
        # 检查是否需要时间调整
        if abs(rate - 1.0) <= STRATEGY.TIME_STRETCH_THRESHOLD:
            if verbose:
                logger.debug(f"条目 {entry.index} 时长匹配良好，无需调整")
            return audio_data
        
        # 应用保守的变速限制
        clamped_rate = np.clip(rate, self.min_speed_ratio, self.max_speed_ratio)
        
        # 计算音质损失风险
        quality_risk = self._assess_quality_risk(rate, clamped_rate)
        
        if abs(clamped_rate - rate) > 0.01:
            # 变速比被限制的情况
            speed_type = "加速" if rate > 1.0 else "减速"
            original_percent = abs(int((rate - 1.0) * 100))
            adjusted_percent = abs(int((clamped_rate - 1.0) * 100))
            
            if quality_risk == "高":
                logger.warning(
                    f"条目 {entry.index} 需要{speed_type} {original_percent}% 来匹配字幕时长，"
                    f"但为保证音质已限制为{speed_type} {adjusted_percent}%。"
                    f"建议检查字幕时长是否合理。"
                )
            else:
                logger.info(
                    f"条目 {entry.index} {speed_type} {adjusted_percent}% 以匹配字幕时长 "
                    f"(音质影响: {quality_risk})"
                )
        
        # 执行高质量时间拉伸
        if abs(clamped_rate - 1.0) > STRATEGY.TIME_STRETCH_THRESHOLD:
            # 使用优化的librosa参数
            stretched_audio = librosa.effects.time_stretch(
                audio_data, 
                rate=clamped_rate,
                # 优化参数以提升音质
                hop_length=512,    # 较小的hop_length以提升质量
                n_fft=2048         # 较大的FFT窗口以减少伪影
            )
            
            if verbose:
                original_duration = len(audio_data) / sampling_rate
                new_duration = len(stretched_audio) / sampling_rate
                logger.debug(f"条目 {entry.index} 时长调整: {original_duration:.2f}s → {new_duration:.2f}s")
            
            return stretched_audio
        else:
            return audio_data
    
    def _assess_quality_risk(self, original_rate: float, clamped_rate: float) -> str:
        """
        评估音质损失风险
        
        Args:
            original_rate: 原始变速比
            clamped_rate: 限制后的变速比
            
        Returns:
            风险级别: "低", "中", "高"
        """
        rate_diff = abs(clamped_rate - 1.0)
        
        if rate_diff <= 0.15:  # 15%以内
            return "低"
        elif rate_diff <= 0.25:  # 25%以内
            return "中"
        else:
            return "高" 