# SRT配音工具 - 日志系统优化

## 📝 日志系统概览

我们已经将所有的 `print()` 语句替换为专业的日志系统，提供了更好的日志管理、彩色输出和调试支持。

## 🎨 日志级别和颜色

| 级别 | 颜色 | 用途 | 示例 |
|------|------|------|------|
| **DEBUG** | 🔵 青色 | 详细调试信息 | 文件大小、处理步骤详情 |
| **INFO** | 🟢 绿色 | 一般信息 | 处理状态、配置信息 |
| **WARNING** | 🟡 黄色 | 警告信息 | 时间重叠、参数调整 |
| **ERROR** | 🔴 红色 | 错误信息 | 文件不存在、处理失败 |
| **SUCCESS** | ✅ 绿色 | 成功信息 | 任务完成、文件导出 |

## 🔧 使用方式

### 基础使用
```python
from srt_dubbing.src.logger import get_logger

logger = get_logger()
logger.info("这是一条信息")
logger.warning("这是一条警告")
logger.error("这是一个错误")
logger.success("操作成功完成")
```

### 进度日志
```python
from srt_dubbing.src.logger import create_process_logger

process_logger = create_process_logger("音频处理")
process_logger.start("开始处理100个文件")

for i in range(100):
    process_logger.progress(i + 1, 100, f"处理文件 {i+1}")
    # 处理逻辑...

process_logger.complete("所有文件处理完成")
```

### 设置日志级别
```python
from srt_dubbing.src.logger import setup_logging

# 普通模式
logger = setup_logging("INFO")

# 调试模式（显示更多详细信息）
logger = setup_logging("DEBUG")
```

## 📋 主要改进内容

### ❌ **改进前（使用 print）**
```python
print("开始处理 50 个字幕条目...")
print("处理条目 1/50: 这是第一条字幕...")
print("警告: 条目 5 变速比超出范围")
print("错误: 处理条目 10 失败")
print("完成处理，生成 50 个音频片段")
```

### ✅ **改进后（专业日志）**
```python
process_logger = create_process_logger("字幕处理")
process_logger.start("处理 50 个字幕条目")

for i, entry in enumerate(entries):
    process_logger.progress(i + 1, len(entries), entry.text[:30])
    # 处理逻辑...
    if warning_condition:
        logger.warning(f"条目 {entry.index} 变速比超出范围")
    if error_condition:
        logger.error(f"条目 {entry.index} 处理失败")

process_logger.complete("生成 50 个音频片段")
```

## 🔍 关键步骤日志

### 1. **CLI 主流程**
```bash
INFO: 🔄 开始SRT配音: 输入文件: test.srt, 策略: stretch
INFO: 🔄 解析SRT文件
INFO: ✓ 成功解析 25 个字幕条目
INFO: 🔄 初始化处理策略
INFO: 使用策略: stretch - 时间拉伸策略：通过改变语速来精确匹配字幕时长
INFO: 🔄 生成音频片段
INFO: ✓ 成功生成 25 个音频片段
INFO: 🔄 合并音频片段
INFO: 🔄 导出音频文件
INFO: ✓ 音频已导出到: output.wav
INFO: ✓ SRT配音完成: 配音文件已保存至: output.wav (耗时: 45.23s)
```

### 2. **SRT解析过程**
```bash
INFO: 🔄 读取SRT文件: 文件: test.srt
DEBUG: 文件读取成功，大小: 2048 字符
INFO: 🔄 解析SRT内容结构
DEBUG: 发现 25 个字幕块
INFO: ✓ SRT解析完成，共 25 个有效条目
```

### 3. **音频生成过程**
```bash
INFO: 🔄 加载IndexTTS模型
INFO: ✓ IndexTTS模型加载成功: model-dir/index_tts
INFO: 🔄 开始基础策略音频生成: 处理 25 个字幕条目
INFO: [================----] 20/25 (80.0%) - 这是第二十条字幕的内容...
INFO: ✓ 基础策略音频生成完成: 生成 25 个音频片段 (耗时: 30.15s)
```

### 4. **详细调试信息（--verbose 模式）**
```bash
DEBUG: 使用配置文件: model-dir/index_tts/config.yaml
DEBUG: 音频合并详情:
DEBUG:   片段 1: 开始=0.00s, 预期时长=2.50s, 实际时长=2.48s
DEBUG:   片段 2: 开始=2.50s, 预期时长=1.80s, 实际时长=1.85s
DEBUG:   ✓ 片段 1 已放置: 0-54560 样本 (2.48s)
DEBUG:   ✓ 片段 2 已放置: 55125-95875 样本 (1.85s)
DEBUG: 音频归一化: 最大值 1.23 -> 1.0
```

## ⚠️ 警告和错误处理

### 常见警告
```bash
WARNING: 条目 5 与前一条目时间重叠
WARNING: 条目 12 变速比 3.2 超出范围，已调整为 2.0
WARNING: 片段 3 与前一片段重叠 0.15s
```

### 常见错误
```bash
ERROR: 解析SRT文件失败: 时间戳格式错误: 00:01:30.500 --> 00:01:32.800
ERROR: IndexTTS模型加载失败: 找不到模型文件
ERROR: 条目 8 处理失败: 文本为空
ERROR: 导出音频失败: 权限被拒绝
```

## 🎯 配置选项

### 环境变量配置
```bash
# 设置日志级别
export SRT_DUBBING_LOG_LEVEL=DEBUG

# 启用文件日志
export SRT_DUBBING_FILE_LOG=true
```

### 程序内配置
```python
from srt_dubbing.src.logger import setup_logging

# 启用文件日志
logger = setup_logging(
    log_level="INFO",
    enable_file_logging=True,
    log_file="logs/srt_dubbing.log"
)
```

## 📊 性能监控

新的日志系统还包含性能监控功能：

```bash
INFO: ✓ SRT配音完成: 配音文件已保存至: output.wav (耗时: 45.23s)
INFO: ✓ 基础策略音频生成完成: 生成 25 个音频片段 (耗时: 30.15s)
INFO: ✓ IndexTTS模型加载成功: model-dir/index_tts (耗时: 5.8s)
```

## 🔇 无颜色输出

如果系统不支持 `colorama`，日志系统会自动降级为无颜色输出，保证兼容性。

## 💡 最佳实践

1. **开发调试**: 使用 `--verbose` 或 `DEBUG` 级别
2. **生产环境**: 使用 `INFO` 级别
3. **静默模式**: 使用 `WARNING` 或 `ERROR` 级别
4. **性能分析**: 关注带有耗时信息的成功日志
5. **问题排查**: 查看 `WARNING` 和 `ERROR` 级别的日志

## 📂 日志文件结构

当启用文件日志时，会生成以下格式的日志：

```
logs/srt_dubbing_20231215_143052.log
```

文件内容格式：
```
2023-12-15 14:30:52,123 | srt_dubbing | INFO | main:85 | 成功解析 25 个字幕条目
2023-12-15 14:30:53,456 | srt_dubbing | WARNING | process_entries:112 | 条目 5 变速比超出范围
```

这个新的日志系统让整个SRT配音工具更加专业、易于调试和维护！ 