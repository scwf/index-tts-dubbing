# SRT配音工具

## 📖 项目介绍

SRT配音工具是一个专业的字幕配音解决方案，通过IndexTTS模型将SRT字幕文件转换为高质量的配音音频。工具支持多种时间同步策略，能够精确匹配字幕时长，生成与视频完美同步的配音。

### 主要特性

- **🎯 精确同步**: 支持时间拉伸策略，确保配音与字幕时长完全匹配
- **🎨 高质量音频**: 基于IndexTTS模型，生成自然流畅的语音
- **⚙️ 灵活策略**: 提供基础策略和拉伸策略，适应不同需求
- **📊 实时监控**: 专业日志系统，实时显示处理进度和状态
- **🔧 易于使用**: 简洁的命令行接口，支持批量处理

## 🏗️ 项目架构

```
srt_dubbing/
├── src/
│   ├── __init__.py            # 模块初始化
│   ├── config.py              # 配置管理
│   ├── utils.py               # 工具函数
│   ├── logger.py              # 日志系统
│   ├── srt_parser.py          # SRT解析器
│   ├── audio_processor.py     # 音频处理器
│   ├── cli.py                 # 命令行接口
│   └── strategies/            # 同步策略
│       ├── __init__.py        # 策略注册
│       ├── base_strategy.py   # 抽象基类
│       ├── basic_strategy.py  # 基础策略
│       ├── stretch_strategy.py # 拉伸策略
│       ├── hq_stretch_strategy.py # 高质量拉伸策略
│       ├── intelligent_stretch_strategy.py # 智能拉伸策略
│       └── iterative_strategy.py    # 迭代策略
└── README.md                  # 说明文档
```

## 🛠️ 环境配置

### 系统要求

- Python 3.10+
- Linux/Windows/macOS
- 至少8GB内存（推荐16GB）
- 支持CUDA的GPU（可选，用于加速）

### 依赖安装
0. **srt-dubbing**
   todo: clone repo
   # 创建python env环境
   conda create -n srt-dubbing python=3.10
   conda activate srt-dubbing

   # 安装ffmpeg（可选，建议用conda安装）
   conda install -c conda-forge ffmpeg

   # 安装PyTorch（请根据你的CUDA版本选择合适的指令）
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

   # 安装音频处理依赖
   pip install librosa numpy soundfile

   # 安装日志依赖
   pip install colorama tqdm

   # 安装分句依赖包
   pip install pysbd


1. **配置IndexTTS引擎 (如需使用)**

   # 克隆IndexTTS主仓库到srt-dubbing并列的目录

   ```bash
   git clone https://github.com/index-tts/index-tts.git
   cd index-tts
   pip install -r requirements.txt
   ```

   # 下载模型文件（以1.5版本为例）到指定目录（model-dir）：
   ```bash
   huggingface-cli download IndexTeam/IndexTTS-1.5 \
     config.yaml bigvgan_discriminator.pth bigvgan_generator.pth bpe.model dvae.pth gpt.pth unigram_12000.vocab \
     --local-dir model-dir
   ```

   如下载速度慢，可使用镜像：

   ```bash
   export HF_ENDPOINT="https://hf-mirror.com"
   ```

   或用wget单独下载：

   ```bash
   wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bigvgan_discriminator.pth -P model-dir
   wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bigvgan_generator.pth -P model-dir
   wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bpe.model -P model-dir
   wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/dvae.pth -P model-dir
   wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/gpt.pth -P model-dir
   wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/unigram_12000.vocab -P model-dir
   wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/config.yaml -P model-dir
   ```

   > 注意：如需使用IndexTTS-1.0模型，请将上述命令中的`IndexTeam/IndexTTS-1.5`替换为`IndexTeam/IndexTTS`。


2. **配置 CosyVoice引擎 (如需使用)**
   
   # 克隆CosyVoice主仓库到srt-dubbing并列的目录
   ```bash
   git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
   cd CosyVoice
   pip install -r requirements.txt
   # 注意：需要将CosyVoice项目路径添加到PYTHONPATH
   ```
3. **配置 F5TTS (如需使用)**

   ```bash
   pip install f5-tts
   ```

## 📝 使用说明

### 基础使用

默认使用 `index_tts` 引擎和 `stretch` 策略。

```bash
python -m srt_dubbing.src.cli \
  --srt input.srt \
  --voice reference.wav \
  --output result.wav
```

### 完整示例

```bash
# 使用时间拉伸策略，精确匹配字幕时长 (使用默认的index_tts引擎)
python -m srt_dubbing.src.cli \
  --srt subtitles/movie.srt \
  --voice voices/narrator.wav \
  --output output/movie_dubbed.wav \
  --strategy stretch \
  --model-dir model-dir/index_tts \
  --verbose

# 使用F5TTS引擎和自适应策略
python -m srt_dubbing.src.cli \
  --srt subtitles/movie.srt \
  --voice voices/narrator.wav \
  --output output/movie_f5.wav \
  --tts-engine f5_tts \
  --strategy adaptive \
  --verbose

# 使用CosyVoice引擎 (需要提供参考文本)
python -m srt_dubbing.src.cli \
  --srt subtitles/movie.srt \
  --voice voices/speaker.wav \
  --output output/movie_cosy.wav \
  --tts-engine cosy_voice \
  --prompt-text "这是参考音频说的话。" \
  --fp16 \
  --verbose

# 假设未来有一个edge_tts引擎，可以这样切换
# python -m srt_dubbing.src.cli \
#   --srt subtitles/movie.srt \
#   --voice voices/narrator.wav \
#   --output output/movie_edge.wav \
#   --tts-engine edge_tts \
#   --strategy basic

# 使用高质量拉伸策略，平衡音质和同步性
python -m srt_dubbing.src.cli \
  --srt subtitles/movie.srt \
  --voice voices/narrator.wav \
  --output output/movie_hq.wav \
  --strategy hq_stretch \
  --verbose

# 使用基础策略，自然语音合成
python -m srt_dubbing.src.cli \
  --srt subtitles/movie.srt \
  --voice voices/narrator.wav \
  --output output/movie_natural.wav \
  --strategy basic


```

## 🔧 命令行参数

### 核心参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--srt` | SRT字幕文件路径 | `--srt input.srt` |
| `--voice` | 参考语音文件路径（WAV格式） | `--voice reference.wav` |
| `--output`| 输出音频文件路径 | `--output result.wav` |

### 策略与引擎

| 参数 | 默认值 | 说明 | 示例 |
|------|--------|------|------|
| `--strategy` | `stretch` | 时间同步策略 | `--strategy basic` |
| `--tts-engine` | `index_tts` | 选择TTS引擎。可用: `index_tts`, `f5_tts`, `cosy_voice` | `--tts-engine cosy_voice` |

### TTS引擎特定参数

| 参数 | 默认值 | 说明 | 示例 |
|------|--------|------|------|
| `--model-dir` | `model-dir/index_tts` | TTS模型目录 | `--model-dir /path/to/model` |
| `--cfg-path` | 自动检测 | 模型配置文件路径 | `--cfg-path config.yaml` |
| `--prompt-text`| 无 | [CosyVoice] 参考音频对应的文本，使用 `cosy_voice` 引擎时必需 | `--prompt-text "你好世界"` |
| `--fp16` | 关闭 | [CosyVoice/IndexTTS] 启用FP16半精度推理以加速 | `--fp16` |

### 其他

| 参数 | 默认值 | 说明 | 示例 |
|------|--------|------|------|
| `--verbose` | 关闭 | 显示详细调试信息 | `--verbose` |

### 策略说明

每种策略都在**同步性**、**音质**和**处理时间**之间做出了不同的权衡。

#### 1. 基础策略 (`basic`) - ⭐⭐⭐⭐⭐ (音质)
- **目标**: 追求最高质量、最自然的语音。
- **实现方法**:
    1. 对每一条字幕文本，调用一次TTS引擎生成最自然的语音。
    2. 完全**不进行任何时间调整**或拉伸操作。
    3. 最后，将所有生成的音频片段按照字幕顺序依次拼接起来。
- **适用场景**: 对音质有极致要求，且可以容忍时长偏差的场景，例如制作有声书、播客等。

#### 2. 拉伸策略 (`stretch`) - ⭐⭐⭐ (同步性)
- **目标**: 保证音频时长与字幕时长**绝对精确**匹配，实现严格同步。
- **实现方法**:
    1. 先生成自然的语音，然后通过加速或减速音频（时间拉伸），强制将音频时长与字幕完全对齐。
    2. 为了防止音质过度劣化，该策略会有一个“安全范围”，如果计算出的变速比例超出此范围，会以最大安全值进行拉伸。
- **适用场景**: 对口型、动作等时间点要求极为严格的视频配音，例如电影、电视剧的精配。

#### 3. 高质量拉伸策略 (`hq_stretch`) - ⭐⭐⭐⭐ (平衡)
- **目标**: 在`stretch`策略的基础上，优先保护音质，做一个折中的选择。
- **实现方法**:
    1. 与`stretch`策略类似，但使用了更保守、更严格的变速“安全范围”，避免进行大幅度的拉伸。
    2. 如果计算出的变速比例超出了这个保守范围，它会选择牺牲一部分同步精度来保证音质。
- **适用场景**: 绝大多数视频配音场景，如教学视频、纪录片、Vlog等。这是在音质和同步之间取得平衡的**推荐首选**之一。

#### 4. 自适应策略 (`adaptive`) - ⭐⭐⭐⭐ (智能)
- **目标**: 在完全不使用时间拉伸的前提下，通过智能调整TTS引擎的生成参数来逼近目标时长，以获得最佳音质。
- **实现方法**:
    1. 直接调用TTS引擎自身提供的`synthesize_to_duration`（自适应时长合成）功能。
    2. 引擎内部会通过自己的优化算法（如二分查找`length_penalty`参数）来多次尝试，以生成最接近目标时长的音频。
    3. **注意**: 此策略的成功与否和效果，完全取决于所选TTS引擎自身的能力。
- **适用场景**: 追求高音质和高同步性，且使用的TTS引擎支持自适应时长功能的场景。是目前**音质优先场景下的最佳选择**。


## 📋 输出示例

### 正常处理流程

```bash
INFO: 🔄 开始SRT配音: 输入文件: movie.srt, 策略: stretch
INFO: 🔄 解析SRT文件
INFO: ✓ 成功解析 25 个字幕条目
INFO: 🔄 初始化处理策略
INFO: 🔄 生成音频片段
INFO: [================----] 20/25 (80.0%) - 条目 20: 这是第二十条字幕...
INFO: ✓ 成功生成 25 个音频片段
INFO: 🔄 合并音频片段
INFO: 🔄 导出音频文件
INFO: ✓ SRT配音完成: 配音文件已保存至: result.wav (耗时: 45.23s)
```

### 常见警告

```bash
WARNING: 条目 5 需要加速 124% 才能匹配字幕时长，但超出安全范围，已限制为加速 100%
WARNING: 条目 12 与前一条目时间重叠
```

## ❓ 常见问题

### Q: 如何选择合适的参考语音？
A: 建议使用3-10秒的清晰语音文件，包含完整的语调变化，音质越好效果越佳。

### Q: stretch策略音质不如basic策略怎么办？
A: 这是因为时间拉伸会影响音质。建议：
- 使用 `hq_stretch` 策略，它在保证音质的前提下进行时间调整
- 检查SRT字幕时长是否合理，避免过度的时间拉伸
- 如果对音质要求很高，使用 `basic` 策略并接受时长偏差

### Q: 各策略如何选择？
A: 
- **basic**: 最佳音质，适合音频材料（如播客、有声书）
- **hq_stretch**: 平衡音质和同步，适合大多数视频配音场景
- **stretch**: 严格同步，适合对时间精度要求极高的场景
- **adaptive**: 智能平衡音质与同步，是大多数视频配音的推荐首选


### Q: 处理大文件时内存不足怎么办？
A: 可以将长SRT文件分割成多个小文件分别处理，然后再合并音频。

### Q: 如何提升处理速度？
A: 使用GPU加速（确保CUDA环境配置正确），或降低音频采样率。

### Q: 支持哪些音频格式？
A: 输入支持WAV格式的参考音频，输出默认为WAV格式，可通过修改输出文件扩展名支持其他格式。

## 📄 许可证

本项目遵循MIT许可证，详见LICENSE文件。

## 🤝 贡献

欢迎提交Issue和Pull Request来帮助改进项目！ 