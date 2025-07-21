
# 使用 Python API 调用 CosyVoice 进行声音克隆与语音合成

本文档旨在说明如何通过 `CosyVoice` 的 Python API 进行声音克隆 (Zero-Shot TTS) 和预训练音色语音合成 (SFT TTS)。

## 本地环境设置 (Local Setup)

在直接使用 Python API 之前，您需要先搭建好本地开发环境。相比于主要用于服务化场景的 Docker 部署，本地环境对于开发和测试更为直接方便。

### 1. 克隆代码库
```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
```

### 2. 创建 Conda 环境
官方推荐使用 Conda 管理 Python 环境，以避免依赖冲突。
```bash
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```
> **重要提示：无需源码安装**
>
> `CosyVoice` 被设计为直接从克隆下来的代码库中运行，它**不需要**执行 `pip install .` 或 `python setup.py install` 这样的源码安装命令。
>
> 您只需确保您的 Python 脚本（或交互式会话）是在 `CosyVoice` 项目的**根目录下**启动的。这样，Python 就可以自动找到 `cosyvoice` 模块并成功导入。

### 5. 在项目外调用 (Advanced)

如果您希望将自己的代码和克隆下来的 `CosyVoice` 目录分开存放，您需要手动将 `CosyVoice` 的路径告知 Python。以下是两种推荐的方法：

#### 方法一：在代码中修改 `sys.path` (推荐)

在您的 Python 脚本的开头，导入 `sys` 模块并把 `CosyVoice` 源码的绝对路径添加到 `sys.path` 列表中。

```python
import sys
import os

# 将 '/path/to/your/CosyVoice' 替换为 CosyVoice 的实际路径
cosyvoice_path = '/path/to/your/CosyVoice'
sys.path.append(cosyvoice_path)

# 现在可以正常导入了
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

# ... 后续代码不变 ...
model_id = 'iic/CosyVoice2-0.5B' 
cosyvoice = CosyVoice2(model_id)

```

#### 方法二：设置 PYTHONPATH 环境变量

在启动 Python 脚本的终端中，通过 `export` 命令设置 `PYTHONPATH` 环境变量。

```bash
# 将 /path/to/your/CosyVoice 替换为 CosyVoice 的实际路径
export PYTHONPATH="/path/to/your/CosyVoice:$PYTHONPATH"

# 然后在同一个终端中运行您的 Python 脚本
python your_script.py
```
> 要使此环境变量永久生效，您可以将 `export` 命令添加到您的 shell 配置文件中（如 `~/.bashrc` 或 `~/.zshrc`）。

> **注意**: 如果在安装过程中遇到 sox 相关的兼容性问题，您可能需要安装 sox 开发库。
> - Ubuntu/Debian: `sudo apt-get install sox libsox-dev`
> - CentOS/RHEL: `sudo yum install sox sox-devel`

### 4. 下载预训练模型
您需要将预训练模型下载到本地。可以使用 `modelscope` 提供的SDK自动下载。
```python
# 您可以新建一个 Python 文件来运行以下代码
from modelscope import snapshot_download

# 我们以推荐的 CosyVoice2 为例
model_dir = snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
print(f"模型已下载到: {model_dir}")

# 您也可以按需下载其他模型
# snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')
```
完成以上步骤后，您的本地环境便已就绪，可以开始编写代码调用 API 了。

## 1. 初始化

### 模型推荐

**官方强烈推荐使用 `CosyVoice2` 模型** (`iic/CosyVoice2-0.5B`)，因为它在多语言支持、低延迟、发音准确性和音色稳定性方面都优于第一代 `CosyVoice` 模型。

### 基础初始化

首先，需要实例化 `CosyVoice2` 类，并指定模型的 `model_dir`。这个参数非常灵活，**既可以是 ModelScope 的模型ID，也可以是本地的模型路径**。

> **它是如何区分模型ID和本地路径的？**
>
> 答案是通过 `os.path.exists()` 函数。代码会检查您传入的字符串在当前工作目录下是否为一个已存在的路径。
> - 如果**不存在** (`os.path.exists()` 返回 `False`)，它就会被当作**模型ID**处理，并触发自动下载。
> - 如果**存在** (`os.path.exists()` 返回 `True`)，它就会被当作**本地路径**处理，直接从该路径加载模型。
>
> **最佳实践**：为了完全避免歧义，推荐对本地路径使用明确的相对 (`./`) 或绝对 (`/`) 格式，例如 `'./pretrained_models/my_model'`。

> **提示：** `CosyVoice` 的初始化非常智能。如果您传入的是一个 ModelScope 模型ID（如 `'iic/CosyVoice2-0.5B'`），且该模型在本地不存在，程序会自动从网上下载模型。如果传入的是一个本地已经存在的路径，则会直接加载。
>
> **模型缓存位置**：当使用模型ID自动下载时，模型文件默认会缓存到用户主目录下的 `~/.cache/modelscope/hub` (Linux/macOS) 或 `C:\\Users\\<Your-Username>\\.cache\\modelscope\\hub` (Windows) 中。

- **CosyVoice2-0.5B**: [https://www.modelscope.cn/models/iic/CosyVoice2-0.5B](https://www.modelscope.cn/models/iic/CosyVoice2-0.5B) (官方强烈推荐)
- **CosyVoice-300M**: [https://www.modelscope.cn/models/iic/CosyVoice-300M](https://www.modelscope.cn/models/iic/CosyVoice-300M)
- **CosyVoice-300M-Instruct**: [https://www.modelscope.cn/models/iic/CosyVoice-300M-Instruct](https://www.modelscope.cn/models/iic/CosyVoice-300M-Instruct)
- **CosyVoice-300M-SFT**: [https://www.modelscope.cn/models/iic/CosyVoice-300M-SFT](https://www.modelscope.cn/models/iic/CosyVoice-300M-SFT)

```python
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

# --- 用法1: 传入模型ID (推荐，会自动下载和缓存) ---
model_id = 'iic/CosyVoice2-0.5B' 
cosyvoice = CosyVoice2(model_id)

# --- 用法2: 传入本地路径 (需要提前下载好模型) ---
# 假设模型已下载到 'pretrained_models/CosyVoice2-0.5B'
local_model_path = 'pretrained_models/CosyVoice2-0.5B'
cosyvoice_local = CosyVoice2(local_model_path) 
```

### 初始化参数说明

在初始化 `CosyVoice` 或 `CosyVoice2` 时，可以传入以下参数来优化模型的加载和推理性能。这些参数在NVIDIA GPU环境下尤其有效。

`CosyVoice2(model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False, trt_concurrent=1)`

- `load_jit` (bool): 是否加载通过 JIT (Just-In-Time) 编译优化的模型。可以提升推理速度，但会增加初次加载时间。默认为 `False`。
- `load_trt` (bool): 是否加载通过 TensorRT 优化的模型。这是针对 NVIDIA GPU 的最高性能优化，可以显著提升推理速度。默认为 `False`。
- `load_vllm` (bool): **(仅适用于 CosyVoice2)** 是否使用 vLLM 进行推理加速。需要预先安装 vLLM。默认为 `False`。
- `fp16` (bool): 是否使用半精度浮点数 (FP16) 进行推理。可以减少约一半的显存占用并提升速度，但可能会有轻微的精度损失。默认为 `False`。
- `trt_concurrent` (int): TensorRT 的并发数。默认为 `1`。

**注意**:
- `load_trt`, `load_vllm` 和 `fp16` 仅在 CUDA 环境下可用。
- 首次使用 `load_trt` 时，模型会进行编译和优化，这可能需要几分钟时间。编译好的模型会被缓存，后续加载会很快。

### 最佳实践配置

对于在 NVIDIA GPU 上的生产环境部署，推荐开启所有优化选项以获得最佳性能：

```python
# CosyVoice2 的最佳性能配置
cosyvoice_optimized = CosyVoice2('iic/CosyVoice2-0.5B', load_jit=True, load_trt=True, load_vllm=True, fp16=True)

# 如果不使用 vLLM
cosyvoice_optimized_no_vllm = CosyVoice2('iic/CosyVoice2-0.5B', load_jit=True, load_trt=True, fp16=True)
```

## 2. 推理接口说明

`CosyVoice` 提供了多种推理模式，其中声音克隆和（SFT）语音合成是最常用的两种。

### 2.1 声音克隆 (Zero-Shot TTS)

声音克隆允许您使用一个简短的音频片段（prompt），来合成具有相同音色的新语音。这在 `inference_zero_shot` 方法中实现。

#### 示例代码

```python
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio

# 初始化 (此处示例使用本地路径，您也可以用模型ID)
model_dir = 'pretrained_models/CosyVoice-300M'
cosyvoice = CosyVoice(model_dir)

# 需要合成的文本
tts_text = "我是通义实验室语音团队全新推出的生成式语音大模型，提供舒适自然的语音合成能力。"

# Prompt 音频和文本
# Prompt 音频需要为 16k 采样率
prompt_wav = "asset/zero_shot_prompt.wav"
prompt_text = "希望你以后能够做的比我还好呦"
prompt_speech_16k = load_wav(prompt_wav, 16000)

# 执行推理
output_wav_path = "zero_shot_out.wav"
output_speech = []
for speech in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k):
    output_speech.append(speech['tts_speech'])

# 保存音频
torchaudio.save(output_wav_path, torch.cat(output_speech, dim=1), sample_rate=cosyvoice.sample_rate)
```

#### 参数说明

`inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True)`

- **必选参数**:
    - `tts_text` (str): 需要合成的目标文本。
    - `prompt_text` (str): Prompt 音频对应的文本。
    - `prompt_speech_16k` (torch.Tensor): 16kHz 采样率的 Prompt 音频数据。

- **可选参数**:
    - `zero_shot_spk_id` (str): 为该 prompt 音色指定一个ID，用于后续的快速调用。默认为空。
    - `stream` (bool): 是否以流式模式返回音频。默认为 `False`。流式模式下，音频会以小块的形式逐步返回。
    - `speed` (float): 语速调节，仅在非流式模式 (`stream=False`) 下生效。默认为 `1.0`。
    - `text_frontend` (bool): 是否对输入文本进行前端处理（如文本正则化）。默认为 `True`。

### 2.2 语音合成 (SFT TTS)

如果您的模型经过了特定说话人的微调（SFT），您可以使用 `inference_sft` 方法直接通过说话人ID来合成语音。

#### 示例代码
```python
from cosyvoice.cli.cosyvoice import CosyVoice
import torchaudio

# 初始化 SFT 模型 (此处示例使用模型ID，会自动下载)
model_dir = 'iic/CosyVoice-300M-SFT'
cosyvoice = CosyVoice(model_dir)

# 查看可用的说话人 ID
available_spks = cosyvoice.list_available_spks()
print("Available speakers:", available_spks)

# 选择一个 spk_id
spk_id = available_spks[0]
tts_text = "这是一个使用预训练音色合成的例子。"

# 执行推理
output_wav_path = "sft_out.wav"
output_speech = []
for speech in cosyvoice.inference_sft(tts_text, spk_id):
    output_speech.append(speech['tts_speech'])

# 保存音频
torchaudio.save(output_wav_path, torch.cat(output_speech, dim=1), sample_rate=cosyvoice.sample_rate)
```

#### 参数说明

`inference_sft(self, tts_text, spk_id, stream=False, speed=1.0, text_frontend=True)`

- **必选参数**:
    - `tts_text` (str): 需要合成的目标文本。
    - `spk_id` (str): 预设的说话人ID。

- **可选参数**:
    - `stream` (bool): 是否以流式模式返回音频。默认为 `False`。
    - `speed` (float): 语速调节，仅在非流式模式 (`stream=False`) 下生效。默认为 `1.0`。
    - `text_frontend` (bool): 是否对输入文本进行前端处理。默认为 `True`。


## 3. 如何控制生成语音的长度

### 通过调整语速控制 (间接控制时长)

您还可以通过调整 `speed` 参数来间接控制音频的时长。这个参数可以改变合成语音的语速。

- `speed` > 1.0：语速变快，音频时长变短。
- `speed` < 1.0：语速变慢，音频时长变长。

**注意**：`speed` 参数只在非流式模式（`stream=False`）下生效。

例如，生成一段语速为 1.2 倍的音频：

```python
# ... (初始化代码)
tts_text = "通过调整语速，可以改变音频的播放时长。"
# 此处以 SFT 模式为例，spk_id 需要根据您使用的SFT模型来确定
available_spks = cosyvoice.list_available_spks()
if available_spks:
    spk_id = available_spks[0]
    output_speech = []
    for speech in cosyvoice.inference_sft(tts_text, spk_id, stream=False, speed=1.2):
        output_speech.append(speech['tts_speech'])

    torchaudio.save("faster_speech.wav", torch.cat(output_speech, dim=1), sample_rate=cosyvoice.sample_rate)
``` 

## 4. 服务部署 (Deployment)

CosyVoice 提供了基于 gRPC 和 FastAPI 的服务化部署方案，并推荐使用 Docker 进行快速环境隔离和部署。

### 4.1 构建 Docker 镜像

首先，需要进入 `runtime/python` 目录并构建 Docker 镜像。

```bash
cd runtime/python
docker build -t cosyvoice:v1.0 .
```

### 4.2 使用 gRPC 服务部署

使用以下命令启动 gRPC 服务。我们推荐使用 `CosyVoice2` 模型以获得更佳效果。此处的 `--model_dir` 同样既可以是模型ID也可以是本地路径。

```bash
# --runtime=nvidia 参数确保容器能使用 NVIDIA GPU
# -p 50000:50000 将容器的50000端口映射到宿主机
# 使用模型ID (会自动在容器内下载)
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v1.0 /bin/bash -c \
  "cd /opt/CosyVoice/CosyVoice/runtime/python/grpc && \
   python3 server.py --port 50000 --max_conc 4 --model_dir 'iic/CosyVoice2-0.5B' && \
   sleep infinity"
```

服务启动后，可以使用 `grpc/client.py` 进行测试：
```bash
python3 runtime/python/grpc/client.py --port 50000 --mode <sft|zero_shot|cross_lingual|instruct> ... (其他参数)
```

### 4.3 使用 FastAPI 服务部署

使用以下命令启动 FastAPI 服务。

```bash
# 使用模型ID (会自动在容器内下载)
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v1.0 /bin/bash -c \
  "cd /opt/CosyVoice/CosyVoice/runtime/python/fastapi && \
   python3 server.py --port 50000 --model_dir 'iic/CosyVoice2-0.5B' && \
   sleep infinity"
```

服务启动后，可以使用 `fastapi/client.py` 进行测试：
```bash
python3 runtime/python/fastapi/client.py --port 50000 --mode <sft|zero_shot|cross_lingual|instruct> ... (其他参数)
``` 