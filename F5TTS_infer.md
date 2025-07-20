# F5TTS Python API 使用说明

## 概述

F5TTS提供了简洁的Python API接口，允许开发者通过代码直接调用文本转语音功能。本文档详细说明了如何使用F5TTS的Python API进行推理。

## 安装和导入

首先确保已经安装了F5TTS，然后导入相关模块：

```python
from f5_tts.api import F5TTS
```

## 基本使用示例

### 简单示例 [1](#0-0) 

```python
from f5_tts.api import F5TTS

# 初始化F5TTS实例
f5tts = F5TTS()

# 进行推理
wav, sr, spec = f5tts.infer(
    ref_file="path/to/reference_audio.wav",
    ref_text="参考音频的文本内容",
    gen_text="要生成的目标文本",
    file_wave="output_audio.wav",
    seed=42
)

print("生成完成！采样率:", sr)
```

### 完整示例 [2](#0-1) 

```python
from importlib.resources import files
from f5_tts.api import F5TTS

f5tts = F5TTS()
wav, sr, spec = f5tts.infer(
    ref_file=str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav")),
    ref_text="some call me nature, others call me mother nature.",
    gen_text="""I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring. Respect me and I'll nurture you; ignore me and you shall face the consequences.""",
    file_wave=str(files("f5_tts").joinpath("../../tests/api_out.wav")),
    file_spec=str(files("f5_tts").joinpath("../../tests/api_out.png")),
    seed=None,
)
```

## API接口详细说明

### F5TTS类初始化参数 [3](#0-2) 

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `model` | str | "F5TTS_v1_Base" | 模型名称，可选：F5TTS_v1_Base, F5TTS_Base, E2TTS_Base等 |
| `ckpt_file` | str | "" | 自定义模型检查点文件路径，留空使用默认 |
| `vocab_file` | str | "" | 词汇表文件路径，留空使用默认 |
| `ode_method` | str | "euler" | ODE求解方法 |
| `use_ema` | bool | True | 是否使用EMA模型 |
| `vocoder_local_path` | str | None | 本地声码器路径 |
| `device` | str | None | 计算设备，留空自动选择 |
| `hf_cache_dir` | str | None | Hugging Face缓存目录 |

### infer方法参数说明 [4](#0-3) 

#### 必选参数

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `ref_file` | str | 参考音频文件路径，用于克隆音色 |
| `ref_text` | str | 参考音频对应的文本内容 |
| `gen_text` | str | 要生成语音的目标文本 |

#### 可选参数 [5](#0-4) 

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `target_rms` | float | 0.1 | 目标输出音量标准化值 |
| `cross_fade_duration` | float | 0.15 | 音频片段间交叉淡化时长（秒） |
| `sway_sampling_coef` | float | -1.0 | Sway采样系数，控制生成多样性 |
| `cfg_strength` | float | 2.0 | 分类器引导强度 |
| `nfe_step` | int | 32 | 去噪步数，影响生成质量 |
| `speed` | float | 1.0 | 生成语音的播放速度 |
| `fix_duration` | float | None | 固定总时长（参考+生成音频）秒数 |
| `remove_silence` | bool | False | 是否移除生成音频中的长静音 |
| `file_wave` | str | None | 输出音频文件路径 |
| `file_spec` | str | None | 输出频谱图文件路径 |
| `seed` | int | None | 随机种子，确保结果可复现 |

#### 其他参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `show_info` | callable | print | 信息显示函数 |
| `progress` | callable | tqdm | 进度条显示函数 |

## 控制生成语音长度的方法

### 1. 使用fix_duration参数 [6](#0-5) 

`fix_duration`参数可以固定总的音频时长（包括参考音频和生成音频）：

```python
wav, sr, spec = f5tts.infer(
    ref_file="ref_audio.wav",
    ref_text="参考文本",
    gen_text="生成文本",
    fix_duration=10.0  # 固定总时长为10秒(注意，包含了参考音频的长度)
)
```

### 2. 使用speed参数 [7](#0-6) 

`speed`参数控制语音播放速度，从而间接控制时长：

```python
wav, sr, spec = f5tts.infer(
    ref_file="ref_audio.wav", 
    ref_text="参考文本",
    gen_text="生成文本",
    speed=1.5  # 1.5倍速，时长缩短
)
```

### 3. 文本长度控制 [8](#0-7) 

系统会自动将长文本分块处理，单次生成限制在30秒内。可以通过控制输入文本长度来控制生成音频长度。

### 4. 参考音频长度限制 [9](#0-8) 

参考音频建议控制在12秒以内，系统会自动截取过长的参考音频。

## 实用技巧

### 1. 音频质量优化 [10](#0-9) 

- 使用时长小于12秒的参考音频
- 在参考音频末尾留出适当的静音空间（如1秒）
- 大写字母会被逐字母朗读
- 添加空格或标点符号来引入停顿

### 2. 文本处理建议

- 如果英文标点符号结束句子，确保后面有空格
- 将数字转换为中文字符以中文方式朗读
- 适当添加标点符号来控制语音节奏

### 3. 故障排除 [11](#0-10) 

- 如果生成空白音频，检查FFmpeg安装
- 使用早期微调检查点时，尝试关闭`use_ema`

## 返回值说明

`infer`方法返回三个值：

- `wav`: 生成的音频数据（numpy数组）
- `sr`: 采样率（通常为24000Hz）
- `spec`: 频谱图数据

## 其他有用方法

### 转录功能 [12](#0-11) 

```python
# 自动转录参考音频
transcribed_text = f5tts.transcribe("ref_audio.wav", language="en")
```

### 导出音频 [13](#0-12) 

```python
# 导出音频文件
f5tts.export_wav(wav, "output.wav", remove_silence=True)
```

## Notes

F5TTS的Python API提供了灵活且强大的文本转语音功能。通过合理设置参数，可以生成高质量的语音输出。建议在使用前先用较短的文本进行测试，熟悉各参数的效果后再处理较长的文本内容。
