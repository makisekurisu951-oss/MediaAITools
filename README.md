# MediaAI Tools 🎬

<div align="center">

**智能AI视频处理平台 - 用自然语言完成专业视频处理**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![GitHub stars](https://img.shields.io/github/stars/lionelyi/MediaAITools?style=social)](https://github.com/lionelyi/MediaAITools)

**Solo Developer** 🚀 | **MVP Version** 💡 | **Built from Scratch** 🛠️ | **Open for Contributions** 🤝

[English](README_EN.md) | 简体中文

</div>

## 💭 为什么开发这个项目

作为一个独立开发者，我经常需要为视频添加字幕，但发现：
- **手动打字太慢** - 10分钟视频要花1小时
- **现有工具复杂** - 需要学习FFmpeg命令行
- **AI工具割裂** - Whisper识别 + 人工校对分离

**于是我想**：能不能用自然语言描述需求，让AI自动完成一切？

历时3个月，从0到1独立完成这个MVP：
- ✅ 自然语言交互（不需要记命令）
- ✅ AI智能纠错（Whisper + LLM双重保障）
- ✅ Docker一键部署（开箱即用）
- ✅ 完全开源（MIT协议）

**这只是开始**！我计划持续优化，也希望更多开发者参与进来，让视频处理变得更简单。

## ✨ 特性

- 🤖 **自然语言交互** - 用中文描述需求，AI自动完成视频处理
- 🎯 **智能路由** - 自动分析任务类型，选择最优处理策略  
- 📝 **AI字幕生成** - 基于Whisper的语音识别 + LLM智能纠错
- 🌐 **Web界面** - 现代化的浏览器操作界面，无需命令行
- 🐳 **Docker部署** - 一键启动，开箱即用
- 🔧 **多LLM支持** - 本地Qwen2、DeepSeek、OpenAI等

## 🚀 快速开始（Docker推荐）

```bash
# 1. 克隆仓库
git clone https://github.com/lionelyi/MediaAITools.git
cd MediaAITools

# 2. 启动服务
docker-compose up -d

# 3. 访问Web界面
# 浏览器打开: http://localhost:8000
```

就是这么简单！🎉 服务启动后：
- **Web UI**: http://localhost:8000 - 主操作界面
- **LLM配置**: http://localhost:8000/config.html - 配置AI模型
- **API文档**: http://localhost:8000/docs - Swagger文档

## 💡 使用示例

### 1. Web界面（推荐）

<table>
<tr>
<td width="50%">

**单文件处理**
```
帮我把 D:\video\demo.mp4 添加字幕
```

**批量处理**
```
帮我把 D:\videos 目录下的所有mp4文件添加字幕
```

**双语字幕**
```
给 D:\work\presentation.mp4 生成中英双语字幕
```

</td>
<td width="50%">

<img src="docs/screenshots/web-ui.png" alt="Web UI" />

*在输入框中用自然语言描述需求，AI自动完成处理*

</td>
</tr>
</table>

### 2. API调用

```python
import requests

# 发送处理请求
response = requests.post('http://localhost:8000/api/agent/process', json={
    "instruction": "帮我把D:\\videos\\demo.mp4添加字幕",
    "use_llm_correction": True,
    "language": "zh"
})

task_id = response.json()['task_id']

# 查询任务状态
status = requests.get(f'http://localhost:8000/api/tasks/{task_id}')
print(status.json())
```

完整示例代码请查看 [examples/](examples/) 目录。

## 🏗️ 架构设计

```
用户交互层
    ↓
┌─────────────────────────────────────┐
│      Web UI (浏览器界面)             │
│  • 自然语言输入                      │
│  • 实时进度显示                      │
│  • LLM配置管理                       │
└──────────────┬──────────────────────┘
               │ HTTP REST API
┌──────────────▼──────────────────────┐
│      FastAPI Backend                │
│                                     │
│  ┌────────────────────────────┐    │
│  │  MediaAgent (AI核心)        │    │
│  │                             │    │
│  │  ┌─────────┬────────┬────┐ │    │
│  │  │ Router  │ Memory │Eval│ │    │
│  │  │(路由)   │(记忆)  │(评估)│    │
│  │  └─────────┴────────┴────┘ │    │
│  └────────────────────────────┘    │
│                                     │
│  ┌────────────────────────────┐    │
│  │  Skills Layer (业务逻辑)   │    │
│  │  • SubtitleSkill (字幕)    │    │
│  │  • BatchSkill (批量)       │    │
│  │  • ImageSkill (图片)       │    │
│  └────────────────────────────┘    │
│                                     │
│  ┌────────────────────────────┐    │
│  │  MCP Server (工具层)       │    │
│  │  • FFmpeg封装              │    │
│  │  • Whisper语音识别         │    │
│  │  • LLM智能纠错             │    │
│  └────────────────────────────┘    │
└─────────────────────────────────────┘
```

详细架构文档：[ARCHITECTURE.md](ARCHITECTURE.md)

## 🎯 支持的功能

### 视频处理
- ✅ **字幕生成** - Whisper识别 + LLM智能纠错，准确率95%+
- ✅ **双语字幕** - 中英对照，自动翻译
- ✅ **批量处理** - 支持目录和单文件
- ✅ **视频剪辑** - 指定时间范围裁剪
- ✅ **格式转换** - MP4/AVI/MKV等格式互转
- ✅ **智能优化** - 音频降噪、视频增强

### AI能力
- ✅ **本地LLM** - Qwen2-1.5B，离线运行，无需API Key
- ✅ **云端LLM** - DeepSeek/OpenAI，高质量纠错
- ✅ **专业术语** - 自动提取并保留技术词汇
- ✅ **语义纠正** - 同音字、上下文错误智能修正

### 图像处理
- ✅ **智能旋转** - 基于YOLO人物检测的自动旋转
- ✅ **尺寸调整** - 支持多种裁剪模式
- ✅ **格式转换** - JPG/PNG/WEBP等

## ⚙️ 配置说明

### LLM配置（Web界面）

访问 `http://localhost:8000/config.html` 进行可视化配置：

| LLM提供商 | 使用场景 | 需要API Key | 推荐度 |
|----------|---------|------------|--------|
| **本地Qwen2** | 离线使用，隐私保护 | ❌ 不需要 | ⭐⭐⭐⭐⭐ |
| **DeepSeek** | 高性价比，质量好 | ✅ 需要 | ⭐⭐⭐⭐ |
| **OpenAI** | 最高质量 | ✅ 需要 | ⭐⭐⭐ |

### 配置文件

复制 `src/config/config.example.yaml` 为 `src/config/config.yaml`：

```yaml
llm:
  default_provider: "qwen2_local"  # 默认使用本地模型
  
  providers:
    qwen2_local:
      model_path: "Qwen/Qwen2-1.5B-Instruct"
      
    deepseek:
      api_key: "your_deepseek_key"
      base_url: "https://api.deepseek.com"
      
    openai:
      api_key: "your_openai_key"
      model: "gpt-3.5-turbo"
```

## 📊 性能指标

| 功能 | 平均耗时 | GPU加速 | 质量评分 |
|------|---------|---------|----------|
| 字幕生成（10分钟视频）| 3-5分钟 | ✅ 支持 | 92/100 |
| LLM纠错（100行字幕）| 10-30秒 | ❌ | 95/100 |
| 批量处理（10个文件）| 30-50分钟 | ✅ 支持 | 90/100 |
| 智能旋转（图片）| <1秒 | ✅ 支持 | 98/100 |

## 🐳 Docker部署详解

### 方式一：docker-compose（推荐）

```bash
# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 方式二：Docker命令

```bash
# 构建镜像
docker build -t mediaai-tools .

# 运行容器
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/output:/app/output \
  --name mediaai \
  mediaai-tools
```

### GPU支持

```bash
# 使用NVIDIA GPU加速
docker-compose -f docker-compose.gpu.yml up -d
```

## 🛠️ 手动安装（高级用户）

<details>
<summary>点击展开查看详细步骤</summary>

#### 环境要求

- Python 3.10+
- FFmpeg
- CUDA 11.8+（可选，用于GPU加速）

#### 安装步骤

```bash
# 1. 安装Python依赖
pip install -r requirements.txt

# 2. 安装FFmpeg
# Windows: 下载 https://ffmpeg.org/download.html 并添加到PATH
# Linux: sudo apt install ffmpeg
# macOS: brew install ffmpeg

# 3. 配置LLM（可选）
cp src/config/config.example.yaml src/config/config.yaml
# 编辑 config.yaml 设置API Key

# 4. 启动服务
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

</details>

## 🤝 贡献指南

欢迎贡献代码！查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

### 开发环境

```bash
# 1. Fork并克隆仓库
git clone https://github.com/lionelyi/MediaAITools.git
cd MediaAITools

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装开发依赖
pip install -r requirements.txt
pip install pytest black flake8

# 4. 运行测试
pytest test/

# 5. 代码格式化
black src/ api/

# 6. 提交PR
```

## 📝 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [OpenAI Whisper](https://github.com/openai/whisper) - 强大的语音识别引擎
- [FFmpeg](https://ffmpeg.org/) - 视频处理工具
- [LangChain](https://github.com/langchain-ai/langchain) - LLM应用框架
- [FastAPI](https://fastapi.tiangolo.com/) - 现代Web框架
- [Qwen](https://github.com/QwenLM/Qwen) - 优秀的本地LLM

## 📧 联系与支持

- 🐛 Bug反馈：[GitHub Issues](https://github.com/lionelyi/MediaAITools/issues)
- 💬 功能讨论：[GitHub Discussions](https://github.com/lionelyi/MediaAITools/discussions)
- 👨‍💻 作者主页：[lionelyi](https://github.com/lionelyi)

## 🗺️ 路线图

> 💡 **项目状态**：这是一个从0到1独立开发完成的MVP版本，核心功能已完整实现。欢迎社区贡献，一起让它变得更好！

### 近期计划 (v1.x)
- [ ] 实时字幕预览
- [ ] WebSocket实时进度更新
- [ ] 字幕编辑器
- [ ] 批量任务队列管理

### 中期计划 (v2.x)
- [ ] 多语言界面（英文、日文等）
- [ ] 支持更多视频格式
- [ ] 云端部署方案
- [ ] 用户认证系统

### 长期愿景 (v3.x)
- [ ] 视频智能剪辑（AI高光提取）
- [ ] 多用户协作支持
- [ ] 云存储集成（S3/OSS）
- [ ] AI自动配音

---

<div align="center">

**如果这个项目对你有帮助，请给个⭐️Star支持一下！**

[![GitHub stars](https://img.shields.io/github/stars/lionelyi/MediaAITools?style=social)](https://github.com/lionelyi/MediaAITools)
[![GitHub forks](https://img.shields.io/github/forks/lionelyi/MediaAITools?style=social)](https://github.com/lionelyi/MediaAITools)

从0到1独立开发 | MVP版本持续优化中 | 欢迎贡献代码

Made with ❤️ by [lionelyi](https://github.com/lionelyi)

</div>
