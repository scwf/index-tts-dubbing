# SRT配音工具 - 导入方式说明

## 📚 Python导入方式对比

### ❌ 相对导入（旧方式）

```python
# 一个点 (.) - 当前包
from .config import AUDIO
from .utils import setup_project_path

# 两个点 (..) - 上级包  
from ..srt_parser import SRTEntry
from ...some_module import something

# 复杂的路径操作
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
```

### ✅ 绝对导入（新方式）

```python
# 清晰明确的完整路径
from srt_dubbing.src.config import AUDIO, STRATEGY, MODEL
from srt_dubbing.src.utils import setup_project_path, safe_import_indextts
from srt_dubbing.src.srt_parser import SRTEntry
from srt_dubbing.src.strategies.base_strategy import TimeSyncStrategy
```

## 🎯 绝对导入的优势

### 1. **🔍 可读性更强**
```python
# ❌ 不清楚从哪里导入
from ..config import AUDIO

# ✅ 一目了然的模块路径
from srt_dubbing.src.config import AUDIO
```

### 2. **🛠️ 更易维护**
- 文件移动时不需要修改导入路径
- IDE能提供更好的自动补全和重构支持
- 静态分析工具能更好地检测依赖关系

### 3. **🚀 更稳定**
```python
# ❌ 相对导入在某些情况下会失败
# 例如：直接运行脚本时、某些测试框架下

# ✅ 绝对导入在任何情况下都能正常工作
from srt_dubbing.src.config import AUDIO
```

### 4. **🔧 调试友好**
- 错误信息更清晰
- 导入问题更容易定位
- 支持更好的代码导航

## 📁 项目结构与导入对应关系

```
srt_dubbing/
├── src/
│   ├── config.py              → srt_dubbing.src.config
│   ├── utils.py               → srt_dubbing.src.utils  
│   ├── srt_parser.py          → srt_dubbing.src.srt_parser
│   ├── audio_processor.py     → srt_dubbing.src.audio_processor
│   ├── cli.py                 → srt_dubbing.src.cli
│   └── strategies/
│       ├── __init__.py        → srt_dubbing.src.strategies
│       ├── base_strategy.py   → srt_dubbing.src.strategies.base_strategy
│       ├── basic_strategy.py  → srt_dubbing.src.strategies.basic_strategy
│       └── stretch_strategy.py → srt_dubbing.src.strategies.stretch_strategy
└── test_mvp.py
```

## 🛡️ 设置项目路径

为了确保绝对导入正常工作，我们使用统一的路径设置：

```python
from srt_dubbing.src.utils import setup_project_path

# 在每个模块开头调用一次
setup_project_path()
```

这个函数会：
1. 自动检测项目根目录
2. 将项目路径添加到 `sys.path`
3. 确保绝对导入能正常工作

## 📝 实际使用示例

### 配置使用
```python
from srt_dubbing.src.config import AUDIO, STRATEGY, MODEL

# 音频配置
sample_rate = AUDIO.DEFAULT_SAMPLE_RATE  # 22050
normalization = AUDIO.AUDIO_NORMALIZATION_FACTOR  # 32768.0

# 策略配置  
threshold = STRATEGY.TIME_STRETCH_THRESHOLD  # 0.05
max_speed = STRATEGY.MAX_SPEED_RATIO  # 2.0

# 模型配置
model_dir = MODEL.DEFAULT_MODEL_DIR  # "model-dir/index_tts"
```

### 工具函数使用
```python
from srt_dubbing.src.utils import (
    setup_project_path,
    safe_import_indextts, 
    normalize_audio_data,
    validate_file_exists
)

# 项目初始化
setup_project_path()

# 安全导入TTS
IndexTTS, available = safe_import_indextts()

# 音频处理
audio_data = normalize_audio_data(raw_audio)

# 文件验证
validate_file_exists("voice.wav", "参考语音文件")
```

## 🔄 迁移指南

如果你在其他地方使用了相对导入，可以按以下步骤迁移：

1. **替换相对导入**
   ```python
   # 旧方式
   from .config import AUDIO
   from ..srt_parser import SRTEntry
   
   # 新方式
   from srt_dubbing.src.config import AUDIO
   from srt_dubbing.src.srt_parser import SRTEntry
   ```

2. **添加路径设置**
   ```python
   from srt_dubbing.src.utils import setup_project_path
   setup_project_path()
   ```

3. **移除手动路径操作**
   ```python
   # 删除这些代码
   sys.path.append(os.path.dirname(...))
   ```

## ✨ 总结

绝对导入让代码更：
- 📖 **可读**: 路径清晰明确
- 🔧 **可维护**: 文件移动不影响导入
- 🛡️ **稳定**: 在任何环境下都能工作
- 🚀 **专业**: 符合Python最佳实践

这种方式让项目更加健壮和专业！ 