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

1. **安装IndexTTS**
   ```bash
   # 克隆IndexTTS项目
   git clone https://github.com/IndexTeam/Index-1.9B.git
   cd Index-1.9B
   
   # 安装依赖
   pip install -e .
   ```

2. **安装音频处理依赖**
   ```bash
   pip install librosa numpy torchaudio soundfile
   ```

3. **安装日志依赖**
   ```bash
   pip install colorama tqdm
   ```

4. **下载模型文件**
   ```bash
   # 使用HuggingFace CLI下载
   huggingface-cli download IndexTeam/Index-1.9B-Chat --local-dir model-dir/index_tts
   
   # 或使用git clone
   git clone https://huggingface.co/IndexTeam/Index-1.9B-Chat model-dir/index_tts
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
| `--tts-engine` | `index_tts` | 选择TTS引擎 | `--tts-engine edge_tts` |

### TTS引擎特定参数

| 参数 | 默认值 | 说明 | 示例 |
|------|--------|------|------|
| `--model-dir` | `model-dir/index_tts` | TTS模型目录 | `--model-dir /path/to/model` |
| `--cfg-path` | 自动检测 | 模型配置文件路径 | `--cfg-path config.yaml` |

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

#### 4. 智能策略 (`intelligent`) - ⭐⭐⭐⭐ (智能)
- **目标**: 尽可能通过调整TTS**生成参数**来匹配时长，将有损的时间拉伸作为最后的微调手段。
- **实现方法**:
    1. **参数调整**: 首先尝试调整TTS模型的`length_penalty`（时长惩罚）参数来重新生成更接近目标时长的音频。
    2. **精调**: 对第二次生成的音频，再进行一次非常小幅度的、对音质影响极小的时间拉伸进行精确对齐。
- **适用场景**: 追求高音质和高同步性，且不介意花费更长处理时间的场景（因可能需要两次TTS推理）。

#### 5. 迭代策略 (`iterative`) - ⭐⭐⭐⭐⭐ (音质) / ⭐ (速度)
- **目标**: **完全杜绝**任何有损的时间拉伸，通过多次迭代生成来无限逼近目标时长。
- **实现方法**:
    1. 对一个字幕条目进行多次循环，每次都调整`length_penalty`参数并重新生成音频。
    2. 从所有生成的结果中，选出那个时长最接近目标时长的作为最终结果。
- **适用场景**: 不计时间成本，对最终成品音质有“洁癖”式追求的最终选择。处理速度极慢。


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
- **intelligent**: 智能平衡音质与同步，是大多数视频配音的推荐首选
- **iterative**: 追求完美音质且不在乎时间成本的最终选择


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