# IndexTTS 推理接口说明

IndexTTS 是一个基于 GPT 的文本转语音（TTS）系统，提供了高质量的语音合成能力。本文档详细介绍了如何使用 IndexTTS 的推理接口。

## 目录

- [快速开始](#快速开始)
- [IndexTTS 类初始化](#indextts-类初始化)
- [推理方法](#推理方法)
  - [infer - 标准推理](#infer---标准推理)
  - [infer_fast - 快速推理](#infer_fast---快速推理)
- [生成参数详解](#生成参数详解)
- [使用示例](#使用示例)
- [注意事项](#注意事项)

## 快速开始

```python
from indextts.infer import IndexTTS

# 初始化模型
tts = IndexTTS(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    is_fp16=True
)

# 执行推理
tts.infer(
    audio_prompt="reference_audio.wav",
    text="您好，这是一个语音合成示例。",
    output_path="output.wav"
)
```

## IndexTTS 类初始化

### 构造函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cfg_path` | str | `"checkpoints/config.yaml"` | 配置文件路径 |
| `model_dir` | str | `"checkpoints"` | 模型文件目录 |
| `is_fp16` | bool | `True` | 是否启用 FP16 精度（仅 GPU 模式） |
| `device` | str/None | `None` | 指定设备（如 `"cuda:0"`, `"cpu"`），None 则自动选择 |
| `use_cuda_kernel` | bool/None | `None` | 是否使用 BigVGAN 自定义 CUDA 内核（仅 CUDA 设备） |

### 设备选择策略

- **自动选择**（`device=None`）：
  - 优先选择 CUDA 设备（如果可用）
  - 其次选择 MPS 设备（如果可用，macOS）
  - 最后回退到 CPU
- **手动指定**：可以指定具体设备如 `"cuda:0"`, `"cuda:1"`, `"cpu"` 等

### 性能优化选项

- **FP16 精度**：在 GPU 上启用可显著提升推理速度，但 CPU 模式会自动禁用
- **CUDA 内核**：启用 BigVGAN 自定义 CUDA 内核可进一步优化性能

## 推理方法

IndexTTS 提供两种推理模式：标准推理（`infer`）和快速推理（`infer_fast`）。

### infer - 标准推理

适用于对音质要求较高的场景，推理速度相对较慢但音质更好。

#### 方法签名

```python
def infer(
    self, 
    audio_prompt: str, 
    text: str, 
    output_path: str, 
    verbose: bool = False, 
    max_text_tokens_per_sentence: int = 120, 
    **generation_kwargs
) -> str or tuple
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `audio_prompt` | str | - | 参考音频文件路径（作为语音风格参考） |
| `text` | str | - | 需要合成的文本内容 |
| `output_path` | str | - | 输出音频文件路径（如为 None 则返回音频数据） |
| `verbose` | bool | `False` | 是否输出详细的调试信息 |
| `max_text_tokens_per_sentence` | int | `120` | 每句话的最大 token 数（用于长文本分句） |
| `**generation_kwargs` | - | - | 额外的生成参数（见下文详解） |

### infer_fast - 快速推理

针对长文本优化的快速推理模式，可实现 2-10 倍的速度提升。

#### 方法签名

```python
def infer_fast(
    self, 
    audio_prompt: str, 
    text: str, 
    output_path: str, 
    verbose: bool = False, 
    max_text_tokens_per_sentence: int = 100, 
    sentences_bucket_max_size: int = 4, 
    **generation_kwargs
) -> str or tuple
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `audio_prompt` | str | - | 参考音频文件路径 |
| `text` | str | - | 需要合成的文本内容 |
| `output_path` | str | - | 输出音频文件路径 |
| `verbose` | bool | `False` | 是否输出详细信息 |
| `max_text_tokens_per_sentence` | int | `100` | 分句的最大 token 数 |
| `sentences_bucket_max_size` | int | `4` | 分句分桶的最大容量 |
| `**generation_kwargs` | - | - | 额外的生成参数 |

#### 快速推理特有参数

- **`max_text_tokens_per_sentence`**：
  - 越小：batch 越多，推理速度越快，内存占用更多，可能影响质量
  - 越大：batch 越少，推理速度越慢，内存和质量更接近标准推理

- **`sentences_bucket_max_size`**：
  - 越大：bucket 数量越少，batch 越多，推理速度越快，内存占用更多
  - 越小：bucket 数量越多，batch 越少，推理速度越慢，质量更接近标准推理

## 生成参数详解

以下参数可通过 `**generation_kwargs` 传递：

### 采样相关参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `do_sample` | bool | `True` | 是否使用随机采样（False 则使用贪心搜索） |
| `temperature` | float | `1.0` | 采样温度，控制生成的随机性（0.1-2.0） |
| `top_p` | float | `0.8` | Top-p 核采样参数（0.1-1.0） |
| `top_k` | int | `30` | Top-k 采样参数 |

### Beam Search 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_beams` | int | `3` | Beam search 的 beam 数量 |
| `length_penalty` | float | `0.0` | 长度惩罚系数 |

### 质量控制参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `repetition_penalty` | float | `10.0` | 重复惩罚系数，防止生成重复内容 |
| `max_mel_tokens` | int | `600` | 最大生成的 mel token 数量 |

## 语音长度控制

IndexTTS 提供了多个参数来精确控制生成语音的长度，适应不同的应用场景需求。

### 核心长度控制参数

| 参数 | 类型 | 默认值 | 说明 | 影响范围 |
|------|------|--------|------|----------|
| `max_mel_tokens` | int | `600` | 最大生成的 mel token 数量 | 直接控制单句语音最大长度 |
| `length_penalty` | float | `0.0` | 长度惩罚系数 | 影响 beam search 的长度偏好 |
| `max_text_tokens_per_sentence` | int | `120`/`100` | 每句话的最大文本 token 数 | 控制分句粒度，间接影响语音长度 |

### 长度控制策略详解

#### 1. mel token 与语音时长的关系

```
语音时长（秒） ≈ mel_tokens × 0.0106
```

常用的 `max_mel_tokens` 配置：

| 目标语音长度 | 推荐 max_mel_tokens | 说明 |
|-------------|-------------------|------|
| 短句（1-3秒） | 200-400 | 适合快速响应场景 |
| 中等（3-6秒） | 400-600 | 默认配置，平衡质量和效率 |
| 长句（6-12秒） | 600-1200 | 适合朗读、讲解等场景 |
| 超长（12秒以上） | 1200+ | 需要足够的 GPU 内存 |

> ⚠️ **重要提醒：截断风险**
> 
> `max_mel_tokens` 设置过小是导致**语音截断**的主要原因！当生成的 mel token 数量达到限制时，模型会强制停止生成，导致：
> - ❌ **语音不完整**：句子可能说到一半就突然停止
> - ❌ **音质下降**：突然中断会让语音听起来不自然  
> - ❌ **语义缺失**：重要信息可能被截断丢失
>
> **如何识别截断：**
> ```
> WARN: generation stopped due to exceeding max_mel_tokens (600)
> ```
> 看到此警告说明发生了截断，需要调整参数。

#### max_mel_tokens 安全配置指南

**📋 配置检查清单：**

✅ **文本长度评估**
```python
# 快速评估：预估所需的 mel tokens
text_chars = len(your_text)
estimated_tokens = text_chars * 0.5  # 保守估算
recommended_max = int(estimated_tokens * 1.3)  # 加30%安全余量
print(f"推荐 max_mel_tokens: {recommended_max}")
```

**💡 最佳实践总结：**

1. **预估优于猜测**：使用计算公式预估所需 tokens，避免盲目设置
2. **宁大勿小**：设置安全余量，避免截断比略微浪费内存更重要
3. **分段处理**：对于超长文本，考虑使用 `infer_fast` 或手动分段
4. **监控警告**：始终关注截断警告，及时调整参数
5. **动态调整**：根据不同文本类型和长度动态调整参数

#### 2. 长度惩罚机制

`length_penalty` 参数控制模型对生成长度的偏好：

```python
# 鼓励生成较长的语音
length_penalty = 1.0  # 正值鼓励长序列

# 鼓励生成较短的语音  
length_penalty = -0.5  # 负值鼓励短序列

# 中性设置（默认）
length_penalty = 0.0  # 不施加长度偏好
```

> ⚠️ **重要说明**：
> 根据 HuggingFace Transformers 的标准实现：
> - **正值** (`length_penalty > 0`)：**鼓励较长的语音生成**
> - **负值** (`length_penalty < 0`)：**鼓励较短的语音生成**  
> - **零值** (`length_penalty = 0`)：不施加长度偏好

#### 3. 文本分句对长度的影响

`max_text_tokens_per_sentence` 通过控制分句粒度来间接影响语音长度：

```python
# 细粒度分句 - 生成多个短音频片段
max_text_tokens_per_sentence = 50

# 粗粒度分句 - 生成较长的连续音频
max_text_tokens_per_sentence = 150
```

### 长度控制配置示例

#### 短音频配置（适合对话、提示音）
```python
short_audio_config = {
    "max_mel_tokens": 300,
    "length_penalty": -0.5,  # 鼓励较短生成
    "max_text_tokens_per_sentence": 60,
    "temperature": 0.8,
    "num_beams": 3
}

tts.infer(
    audio_prompt="reference.wav",
    text="简短的提示信息。",
    output_path="short_audio.wav",
    **short_audio_config
)
```

#### 中等长度配置（适合新闻、故事）
```python
medium_audio_config = {
    "max_mel_tokens": 800,
    "length_penalty": 0.0,  # 中性长度偏好
    "max_text_tokens_per_sentence": 100,
    "temperature": 0.7,
    "num_beams": 4
}
```

#### 长音频配置（适合朗读、演讲）
```python
long_audio_config = {
    "max_mel_tokens": 1500,
    "length_penalty": 0.3,  # 鼓励较长生成
    "max_text_tokens_per_sentence": 150,
    "temperature": 0.6,
    "num_beams": 5,
    "repetition_penalty": 12.0  # 避免长音频中的重复
}
```

#### 快速推理中的长度控制
```python
# 长文本快速推理，精确控制每个片段长度
tts.infer_fast(
    audio_prompt="reference.wav",
    text=long_text,
    output_path="output.wav",
    max_text_tokens_per_sentence=80,  # 控制单句长度
    sentences_bucket_max_size=4,      # 控制批处理大小
    max_mel_tokens=600,               # 控制最大生成长度
    length_penalty=0.1                # 轻微鼓励较长生成
)
```

### 长度控制最佳实践

#### 1. 根据内容类型调整参数

```python
# 新闻播报 - 稳定的中等长度
news_config = {
    "max_mel_tokens": 700,
    "length_penalty": 0.2,  # 轻微鼓励较长语音
    "max_text_tokens_per_sentence": 90
}

# 有声书朗读 - 较长连贯片段
audiobook_config = {
    "max_mel_tokens": 1200,
    "length_penalty": 0.4,  # 鼓励较长语音
    "max_text_tokens_per_sentence": 140
}

# 智能助手回复 - 简洁明了
assistant_config = {
    "max_mel_tokens": 400,
    "length_penalty": -0.3,  # 鼓励较短语音
    "max_text_tokens_per_sentence": 70
}
```

#### 2. 动态长度调整

```python
def get_length_config(text_length):
    """根据文本长度动态调整参数"""
    if text_length < 50:
        return {
            "max_mel_tokens": 300,
            "length_penalty": -0.2,  # 短文本鼓励简洁
            "max_text_tokens_per_sentence": 50
        }
    elif text_length < 200:
        return {
            "max_mel_tokens": 600,
            "length_penalty": 0.1,   # 中等文本轻微鼓励较长语音
            "max_text_tokens_per_sentence": 80
        }
    else:
        return {
            "max_mel_tokens": 1000,
            "length_penalty": 0.3,   # 长文本鼓励较长语音
            "max_text_tokens_per_sentence": 120
        }

# 使用示例
text = "您的文本内容..."
config = get_length_config(len(text))
tts.infer(audio_prompt="ref.wav", text=text, output_path="out.wav", **config)
```

### 长度控制常见问题

#### 1. 生成停止警告
```
WARN: generation stopped due to exceeding max_mel_tokens (600)
```

**解决方案：**
- 增大 `max_mel_tokens` 参数
- 减小 `max_text_tokens_per_sentence` 进行更细分句
- 检查文本是否包含过长句子

#### 2. 音频过短或截断

**可能原因和解决方案：**
```python
# 问题：音频意外截断
# 解决：增加最大长度限制
config = {
    "max_mel_tokens": 1000,  # 增大限制
    "length_penalty": 0.2    # 鼓励完整生成
}

# 问题：音频过短
# 解决：鼓励更长的生成
config = {
    "length_penalty": 0.3,   # 正值鼓励较长生成
    "temperature": 0.6       # 降低随机性
}
```

#### 3. 内存不足问题

当 `max_mel_tokens` 设置过大时可能导致内存不足：

```python
# GPU 内存优化配置
memory_efficient_config = {
    "max_mel_tokens": 800,                    # 适中的长度限制
    "sentences_bucket_max_size": 2,           # 减小批处理大小
    "max_text_tokens_per_sentence": 80        # 更细的分句
}
```

### 长度预估工具

```python
def estimate_audio_duration(text, avg_tokens_per_second=94):
    """
    估算音频时长
    
    Args:
        text: 输入文本
        avg_tokens_per_second: 平均每秒 mel tokens（约94）
    
    Returns:
        estimated_duration: 预估时长（秒）
    """
    # 粗略估算：中文约 2 字符/token，英文约 4 字符/token
    char_count = len(text)
    estimated_tokens = char_count * 0.5  # 保守估计
    estimated_duration = estimated_tokens / avg_tokens_per_second
    
    return estimated_duration

# 使用示例
text = "这是一段测试文本，用于估算生成的音频长度。"
duration = estimate_audio_duration(text)
print(f"预估音频时长: {duration:.2f} 秒")

# 根据预估时长设置 max_mel_tokens
recommended_max_tokens = int(duration * 94 * 1.2)  # 增加 20% 余量
```

### 参数配置建议

#### 高质量配置
```python
generation_kwargs = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "num_beams": 5,
    "repetition_penalty": 15.0,
    "max_mel_tokens": 800
}
```

#### 快速配置
```python
generation_kwargs = {
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 0.8,
    "top_k": 30,
    "num_beams": 1,
    "repetition_penalty": 5.0,
    "max_mel_tokens": 400
}
```

#### 稳定配置
```python
generation_kwargs = {
    "do_sample": False,  # 使用贪心搜索
    "num_beams": 3,
    "repetition_penalty": 10.0,
    "max_mel_tokens": 600
}
```

## 使用示例

### 基础使用

```python
from indextts.infer import IndexTTS

# 初始化
tts = IndexTTS()

# 标准推理
output_file = tts.infer(
    audio_prompt="reference.wav",
    text="这是一个测试文本。",
    output_path="output.wav",
    verbose=True
)
print(f"音频已保存到: {output_file}")
```

### 快速推理（适合长文本）

```python
long_text = """
人工智能技术的发展日新月异，语音合成作为其中的重要分支，
已经在诸多领域得到了广泛应用。IndexTTS 作为新一代的
文本转语音系统，具有音质自然、响应迅速的特点。
"""

tts.infer_fast(
    audio_prompt="reference.wav",
    text=long_text,
    output_path="long_output.wav",
    max_text_tokens_per_sentence=80,
    sentences_bucket_max_size=6
)
```

### 自定义生成参数

```python
# 高质量生成
tts.infer(
    audio_prompt="reference.wav",
    text="需要高质量合成的文本。",
    output_path="high_quality.wav",
    do_sample=True,
    temperature=0.6,
    top_p=0.95,
    num_beams=5,
    repetition_penalty=15.0,
    max_mel_tokens=1000
)
```

### 返回音频数据而非保存文件

```python
# 不指定 output_path，直接返回音频数据
sample_rate, audio_data = tts.infer(
    audio_prompt="reference.wav",
    text="测试文本",
    output_path=None  # 返回音频数据
)

# 可以进一步处理音频数据
print(f"采样率: {sample_rate}, 音频形状: {audio_data.shape}")
```

### 批量处理

```python
texts = [
    "第一段文本。",
    "第二段文本。", 
    "第三段文本。"
]

for i, text in enumerate(texts):
    tts.infer_fast(
        audio_prompt="reference.wav",
        text=text,
        output_path=f"output_{i}.wav"
    )
```

## 注意事项

### 性能优化

1. **参考音频缓存**：IndexTTS 会自动缓存参考音频的特征，连续使用相同参考音频时会跳过重复计算
2. **GPU 内存管理**：推理完成后会自动清理 GPU 缓存
3. **长文本处理**：建议使用 `infer_fast` 方法处理超过 100 字的文本

### 参数调优

1. **音质 vs 速度**：
   - 追求音质：使用 `infer` 方法，增大 `num_beams`，降低 `temperature`
   - 追求速度：使用 `infer_fast` 方法，减小 `max_text_tokens_per_sentence`

2. **内存优化**：
   - GPU 内存不足时，减小 `sentences_bucket_max_size`
   - 减小 `max_mel_tokens` 可降低内存占用

3. **质量控制**：
   - 增大 `repetition_penalty` 可减少重复
   - 调整 `temperature` 控制生成的多样性

### 常见问题

1. **生成停止警告**：如果看到超出 `max_mel_tokens` 的警告，可以：
   - 增大 `max_mel_tokens` 参数
   - 减小 `max_text_tokens_per_sentence` 进行更细粒度的分句

2. **CUDA 内核加载失败**：这不会影响功能，只是性能略有下降，可通过重新安装解决

3. **DeepSpeed 加载失败**：系统会自动回退到标准推理，不影响正常使用

### 环境要求

- Python 3.7+
- PyTorch 1.8+
- CUDA 11.0+（如使用 GPU）
- 足够的内存（CPU 模式需要更多内存）

---

更多详细信息请参考项目的官方文档和示例代码。 