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
│       └── stretch_strategy.py # 拉伸策略
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

```bash
python -m srt_dubbing.src.cli \
  --srt input.srt \
  --voice reference.wav \
  --output result.wav
```

### 完整示例

```bash
# 使用时间拉伸策略，精确匹配字幕时长
python -m srt_dubbing.src.cli \
  --srt subtitles/movie.srt \
  --voice voices/narrator.wav \
  --output output/movie_dubbed.wav \
  --strategy stretch \
  --model-dir model-dir/index_tts \
  --verbose

# 使用基础策略，自然语音合成
python -m srt_dubbing.src.cli \
  --srt subtitles/movie.srt \
  --voice voices/narrator.wav \
  --output output/movie_natural.wav \
  --strategy basic
```

## 🔧 命令行参数

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--srt` | SRT字幕文件路径 | `--srt input.srt` |
| `--voice` | 参考语音文件路径（WAV格式） | `--voice reference.wav` |

### 可选参数

| 参数 | 默认值 | 说明 | 示例 |
|------|--------|------|------|
| `--output` | `output.wav` | 输出音频文件路径 | `--output result.wav` |
| `--strategy` | `stretch` | 时间同步策略 | `--strategy basic` |
| `--model-dir` | `model-dir/index_tts` | IndexTTS模型目录 | `--model-dir /path/to/model` |
| `--cfg-path` | 自动检测 | 模型配置文件路径 | `--cfg-path config.yaml` |
| `--verbose` | 关闭 | 显示详细调试信息 | `--verbose` |

### 策略说明

| 策略 | 特点 | 适用场景 |
|------|------|----------|
| `basic` | 自然语音合成，可能与字幕时长不完全匹配 | 追求语音自然度，允许时长偏差 |
| `stretch` | 通过时间拉伸精确匹配字幕时长 | 需要严格同步，如视频配音 |

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