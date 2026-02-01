# 快速开始 - 本地 LLM 使用指南

## 1. 立即开始使用（默认配置）

MediaAITools 已经默认配置为使用**本地 Qwen2-1.5B-Instruct** 模型。无需任何 API 密钥即可开始使用！

### 安装依赖

```powershell
# 安装 PyTorch 和 Transformers
pip install torch transformers accelerate

# 对于 GPU 加速（可选，需要 NVIDIA GPU）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 测试配置

```powershell
cd D:\MediaAITools
python test_local_llm.py
```

首次运行会自动下载模型（约 4GB），需要等待几分钟。

## 2. 生成双语字幕（使用本地模型）

```powershell
cd D:\MediaAITools\code
$env:PATH = "D:\temp\bin;" + $env:PATH

# 测试双语字幕生成
python test_bilingual.py
```

这将使用本地 Qwen2-1.5B 模型进行字幕智能纠错，完全离线运行！

## 3. 通过自然语言切换 LLM

### Python 代码方式

```python
from code.skills.config_skill import ConfigSkill
import asyncio

async def switch_llm():
    config = ConfigSkill()
    
    # 使用本地模型（默认）
    result = await config.execute("使用本地模型")
    print(result)
    # Output: {'success': True, 'message': '✓ 已切换到 local_qwen (本地)', ...}
    
    # 切换到 DeepSeek（需要 API 密钥）
    result = await config.execute("切换到DeepSeek")
    
    # 切换回本地
    result = await config.execute("用本地模型")

asyncio.run(switch_llm())
```

### 支持的命令

| 命令 | 效果 |
|------|------|
| `"使用本地模型"` | 使用本地 Qwen2-1.5B（免费、离线） |
| `"切换到OpenAI"` | 使用 OpenAI GPT-4（需要配置 API 密钥） |
| `"用DeepSeek"` | 使用 DeepSeek（需要配置 API 密钥） |
| `"用远程千问API"` | 使用千问云 API（需要配置 API 密钥） |

## 4. 配置远程 API（可选）

如果您想使用云端 LLM，编辑 [code/config/config.yaml](code/config/config.yaml)：

### 使用 DeepSeek（性价比最高）

```yaml
deepseek:
  enabled: true  # 改为 true
  api_key: "sk-xxxxxx"  # 填入您的 API 密钥
```

然后通过自然语言切换：

```python
result = await config.execute("切换到DeepSeek")
```

### 使用 OpenAI

```yaml
openai:
  enabled: true
  api_key: "sk-xxxxxx"
```

```python
result = await config.execute("切换到OpenAI")
```

## 5. 当前配置状态

### 查看当前配置

```python
from code.skills.config_skill import ConfigSkill

config = ConfigSkill()
current = config.get_current_config()

print(f"启用的提供商: {current['enabled_providers']}")
print(f"默认模型: {current['default_models']}")
```

### 使用命令行工具

```powershell
python check_llm_config.py
```

输出示例：
```
1. 配置文件状态:
   ✓ local_qwen: 已配置 (本地模型)
   ✗ openai: 未配置
   ✗ deepseek: 未配置
   ✗ qwen: 未配置

2. 默认任务模型配置:
   视频理解: local_qwen
   字幕翻译: local_qwen
   中文处理: local_qwen

3. LLM Manager初始化:
   ✓ 1 个可用提供商: local_qwen
```

## 6. 性能说明

### 本地模型性能

- **CPU 模式**: ~5-10 tokens/秒（可用但较慢）
- **GPU 模式**: ~50-200 tokens/秒（推荐）

### 翻译速度参考

以 1 分钟视频为例（约 100 个字幕片段）：

- **本地模型 (CPU)**: ~2-5 分钟
- **本地模型 (GPU)**: ~30-60 秒
- **云端 API**: ~30-60 秒

### 成本对比

- **本地模型**: 完全免费 ✓
- **DeepSeek**: ~$0.01-0.02/视频
- **OpenAI**: ~$3-5/视频

## 7. 常见问题

### Q: 本地模型会很慢吗？

A: 取决于硬件：
- 有 GPU：速度接近云端 API
- 仅 CPU：较慢但完全可用，适合离线场景

### Q: 模型占用多少空间？

A: 
- 模型文件：~4GB
- 运行时内存：4-6GB (CPU) 或 2-4GB VRAM (GPU)

### Q: 可以混合使用吗？

A: 可以！随时通过自然语言命令切换：
```python
# 日常使用本地模型（免费）
await config.execute("使用本地模型")

# 重要任务切换到 GPT-4（质量优先）
await config.execute("切换到OpenAI")

# 大批量任务用 DeepSeek（成本优先）
await config.execute("用DeepSeek")
```

### Q: 本地模型支持哪些功能？

A: 
- ✓ 字幕翻译（中英互译）
- ✓ 视频理解
- ✓ 中文文本处理
- ✓ 完全离线运行
- ✓ 数据隐私保护

## 8. 故障排除

### 问题：模型加载失败

```
RuntimeError: Failed to load local model
```

**解决**：
```powershell
# 确保安装了依赖
pip install transformers torch accelerate

# 检查是否为网络问题（使用镜像）
$env:HF_ENDPOINT = "https://hf-mirror.com"
python test_local_llm.py
```

### 问题：内存不足

```
RuntimeError: out of memory
```

**解决**：
1. 关闭其他应用
2. 或临时切换到云 API：
```python
await config.execute("用DeepSeek")  # 不占用本地内存
```

### 问题：GPU 未被使用

**检查**：
```python
import torch
print(torch.cuda.is_available())  # 应该输出 True
```

**解决**：
如果输出 False，重新安装 CUDA 版本的 PyTorch：
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 9. 完整示例

```python
"""完整的字幕生成示例 - 使用本地 LLM"""
import asyncio
from code.skills.config_skill import ConfigSkill
from code.mcp_server.tools import SubtitleTool

async def generate_bilingual_subtitles():
    # 1. 确保使用本地模型
    config = ConfigSkill()
    result = await config.execute("使用本地模型")
    print(result['message'])
    
    # 2. 生成双语字幕
    tool = SubtitleTool()
    result = await tool.execute(
        video_path="test/trade.avi",
        output_path="test/trade-subtitle-local.mp4",
        bilingual=True
    )
    
    print(f"✓ 字幕生成完成: {result['output_path']}")

# 运行
asyncio.run(generate_bilingual_subtitles())
```

## 10. 下一步

- 阅读完整文档: [LOCAL_LLM_GUIDE.md](LOCAL_LLM_GUIDE.md)
- 查看配置详情: [BILINGUAL_CONFIG_GUIDE.md](BILINGUAL_CONFIG_GUIDE.md)
- 测试所有功能: `python test_local_llm.py`

---

**推荐配置**: 本地模型（免费、隐私、离线） + DeepSeek 备用（成本极低、质量好）
