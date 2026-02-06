# 本地 LLM 部署指南 - Qwen2-1.5B-Instruct

## 概述

MediaAITools 默认使用 **Qwen2-1.5B-Instruct** 本地模型，无需 API 密钥，完全离线运行。

## 系统要求

### 最低配置
- **内存**: 8GB RAM
- **存储**: 5GB 可用空间（模型约 4GB）
- **Python**: 3.8+

### 推荐配置
- **内存**: 16GB+ RAM
- **GPU**: NVIDIA GPU with 6GB+ VRAM (可选，提升速度 10-50x)
- **存储**: 10GB+ 可用空间
- **CUDA**: 11.8+ (如使用 GPU)

## 安装步骤

### 1. 安装依赖

```powershell
# 安装 PyTorch (CPU 版本)
pip install torch torchvision torchaudio

# 或者安装 GPU 版本 (如果有 NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 Transformers
pip install transformers accelerate
```

### 2. 验证安装

```powershell
cd D:\MediaAITools
python test_local_llm.py
```

首次运行会自动下载模型（约 4GB），需要几分钟到半小时不等，取决于网络速度。

### 3. 配置文件说明

配置文件: [src/config/config.yaml](code/config/config.yaml)

```yaml
llm:
  # 本地 Qwen2-1.5B 配置（默认启用）
  local_qwen:
    enabled: true
    model_path: "Qwen/Qwen2-1.5B-Instruct"  # HuggingFace 模型名
    device: "auto"  # auto/cuda/cpu
```

#### 设备选项

- `auto`: 自动选择（有 GPU 用 GPU，否则用 CPU）
- `cuda`: 强制使用 GPU（需要 NVIDIA GPU + CUDA）
- `cpu`: 强制使用 CPU（较慢但兼容性好）

#### 模型路径选项

- **HuggingFace 名称**: `"Qwen/Qwen2-1.5B-Instruct"` (自动下载)
- **本地路径**: `"D:/models/Qwen2-1.5B-Instruct"` (手动下载)

## 自然语言配置

### 通过自然语言切换 LLM

MediaAITools 支持通过自然语言切换 LLM 提供商：

```python
from code.skills.config_skill import ConfigSkill

config = ConfigSkill()

# 使用本地模型
config.execute("使用本地模型")

# 切换到 OpenAI
config.execute("切换到OpenAI")

# 使用 DeepSeek
config.execute("用DeepSeek")

# 使用远程千问 API
config.execute("用远程千问API")
```

### 支持的配置命令

| 命令示例 | 效果 |
|---------|------|
| "使用本地模型" | 启用本地 Qwen2-1.5B |
| "local model" | 启用本地 Qwen2-1.5B |
| "切换到OpenAI" | 启用 OpenAI GPT-4 |
| "use gpt" | 启用 OpenAI GPT-4 |
| "用DeepSeek" | 启用 DeepSeek Chat |
| "switch to deepseek" | 启用 DeepSeek Chat |
| "用远程千问API" | 启用千问云 API |
| "remote qwen api" | 启用千问云 API |

## 性能对比

### 本地模型 vs 云 API

| 指标 | 本地 Qwen2-VL (CPU) | 本地 Qwen2-VL (GPU) | OpenAI GPT-4 | DeepSeek |
|------|---------------------|---------------------|--------------|----------|
| **成本** | 免费 ✓ | 免费 ✓ | $0.03/1K tokens | $0.0001/1K tokens |
| **隐私** | 完全离线 ✓ | 完全离线 ✓ | 上传到云端 | 上传到云端 |
| **速度** | ~5-10 tokens/s | ~50-200 tokens/s | ~30-50 tokens/s | ~30-50 tokens/s |
| **质量** | 良好 | 良好 | 优秀 | 优秀 |
| **网络** | 不需要 ✓ | 不需要 ✓ | 需要 | 需要 |

### 推荐使用场景

- **本地模型**: 日常字幕翻译、离线使用、注重隐私
- **OpenAI**: 需要最高质量翻译
- **DeepSeek**: 性价比优先（价格是 OpenAI 的 1/300）
- **千问**: 国内用户，阿里云生态

## 手动下载模型（可选）

如果网络不稳定，可以手动下载模型：

### 方法 1: 使用 HuggingFace CLI

```powershell
# 安装 CLI
pip install huggingface-hub

# 下载模型
huggingface-cli download Qwen/Qwen2-1.5B-Instruct --local-dir D:/models/Qwen2-1.5B-Instruct
```

然后修改配置文件：

```yaml
local_qwen:
  model_path: "D:/models/Qwen2-1.5B-Instruct"
```

### 方法 2: 使用镜像站（国内用户）

```powershell
# 设置镜像环境变量
$env:HF_ENDPOINT = "https://hf-mirror.com"

# 运行测试脚本会自动从镜像下载
python test_local_llm.py
```

## 故障排除

### 问题 1: 内存不足

**症状**: `RuntimeError: out of memory`

**解决**:
1. 关闭其他应用释放内存
2. 使用 CPU 模式: `device: "cpu"`
3. 或切换到云 API

### 问题 2: 模型下载失败

**症状**: `Connection error` 或超时

**解决**:
1. 使用国内镜像: `$env:HF_ENDPOINT = "https://hf-mirror.com"`
2. 或手动下载模型（见上文）
3. 或临时切换到云 API: `config.execute("用DeepSeek")`

### 问题 3: GPU 不可用

**症状**: `CUDA not available`

**解决**:
1. 确认安装了 CUDA 版本的 PyTorch
2. 或使用 CPU 模式（较慢但可用）

### 问题 4: 推理速度慢

**优化建议**:
1. 使用 GPU: 配置 `device: "cuda"`
2. 安装 Flash Attention: `pip install flash-attn`
3. 或切换到云 API（速度更稳定）

## 配置远程 API（备选方案）

如果本地资源不足，可以配置远程 API：

### DeepSeek（推荐性价比）

```yaml
deepseek:
  enabled: true
  api_key: "sk-xxxxxxxxxxxxxx"  # 从 https://platform.deepseek.com 获取
  
default_models:
  subtitle_translation: "deepseek"
```

### OpenAI（推荐质量）

```yaml
openai:
  enabled: true
  api_key: "sk-xxxxxxxxxxxxxx"  # 从 https://platform.openai.com 获取
  
default_models:
  subtitle_translation: "openai"
```

### 千问（推荐国内）

```yaml
qwen:
  enabled: true
  api_key: "sk-xxxxxxxxxxxxxx"  # 从 https://dashscope.console.aliyun.com 获取
  
default_models:
  subtitle_translation: "qwen"
```

## 测试命令

```powershell
# 测试配置切换
python test_local_llm.py

# 测试字幕翻译（本地模型）
cd code
python test_bilingual.py

# 查看当前配置
python check_llm_config.py
```

## 总结

✓ **默认配置**: 本地 Qwen2-1.5B-Instruct（免费、离线、隐私）
✓ **自动下载**: 首次运行自动下载模型
✓ **灵活切换**: 通过自然语言切换任何 LLM
✓ **多种选择**: 本地/OpenAI/DeepSeek/千问 任选

根据您的需求选择：
- **没有GPU，离线使用** → 本地 CPU 模式（慢但可用）
- **有 GPU** → 本地 GPU 模式（快速且免费）
- **追求速度和质量** → DeepSeek 或 OpenAI
- **网络受限** → 本地模型
