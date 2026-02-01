# MediaAI Tools 🎬

<div align="center">

**Intelligent AI-Powered Video Processing Platform - Professional Video Editing with Natural Language**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![GitHub stars](https://img.shields.io/github/stars/lionelyi/MediaAITools?style=social)](https://github.com/lionelyi/MediaAITools)

**Solo Developer** 🚀 | **MVP Version** 💡 | **Built from Scratch** 🛠️ | **Open for Contributions** 🤝

English | [简体中文](README.md)

</div>

## 💭 Why I Built This

As a solo developer, I frequently needed to add subtitles to videos, but found:
- **Manual typing is too slow** - 10-minute video takes 1 hour to subtitle
- **Existing tools are complex** - Need to learn FFmpeg command-line
- **AI tools are fragmented** - Whisper recognition + manual correction are separate

**So I thought**: What if you could just describe what you want in plain language, and AI does everything?

After 3 months of solo development from scratch, here's the MVP:
- ✅ Natural language interface (no commands to memorize)
- ✅ AI intelligent correction (Whisper + LLM double guarantee)
- ✅ Docker one-click deployment (ready to use)
- ✅ Fully open source (MIT license)

**This is just the beginning**! I plan to continuously improve it, and I hope more developers will join to make video processing simpler.

## ✨ Features

- 🤖 **Natural Language Interface** - Describe tasks in plain language, AI handles the rest
- 🎯 **Intelligent Router** - Automatically selects optimal processing strategy  
- 📝 **AI Subtitle Generation** - Whisper speech recognition + LLM intelligent correction
- 🌐 **Modern Web UI** - Browser-based interface, no command line needed
- 🐳 **Docker Ready** - One-command deployment, production-ready
- 🔧 **Multi-LLM Support** - Local Qwen2, DeepSeek, OpenAI, and more

## 🚀 Quick Start (Docker Recommended)

```bash
# 1. Clone repository
git clone https://github.com/lionelyi/MediaAITools.git
cd MediaAITools

# 2. Start service
docker-compose up -d

# 3. Access Web UI
# Open browser: http://localhost:8000
```

That's it! 🎉 After service starts:
- **Web UI**: http://localhost:8000 - Main interface
- **LLM Config**: http://localhost:8000/config.html - Configure AI models
- **API Docs**: http://localhost:8000/docs - Swagger documentation

## 💡 Usage Examples

### Web Interface (Recommended)

```
Add subtitles to D:\video\demo.mp4
```

```
Generate bilingual subtitles for all videos in D:\work\videos
```

### API Call

```python
import requests

response = requests.post('http://localhost:8000/api/agent/process', json={
    "instruction": "Add subtitles to D:\\videos\\demo.mp4",
    "use_llm_correction": True,
    "language": "en"
})

task_id = response.json()['task_id']
status = requests.get(f'http://localhost:8000/api/tasks/{task_id}')
```

See [examples/](examples/) for complete code samples.

## 🏗️ Architecture

```
User Interface
    ↓
┌─────────────────────────────────────┐
│      Web UI (Browser)               │
│  • Natural language input           │
│  • Real-time progress               │
│  • LLM configuration                │
└──────────────┬──────────────────────┘
               │ HTTP REST API
┌──────────────▼──────────────────────┐
│      FastAPI Backend                │
│                                     │
│  ┌────────────────────────────┐    │
│  │  MediaAgent (AI Core)       │    │
│  │                             │    │
│  │  ┌─────────┬────────┬────┐ │    │
│  │  │ Router  │ Memory │Eval│ │    │
│  │  └─────────┴────────┴────┘ │    │
│  └────────────────────────────┘    │
│                                     │
│  ┌────────────────────────────┐    │
│  │  Skills Layer              │    │
│  │  • SubtitleSkill           │    │
│  │  • BatchSkill              │    │
│  │  • ImageSkill              │    │
│  └────────────────────────────┘    │
│                                     │
│  ┌────────────────────────────┐    │
│  │  MCP Server (Tools)        │    │
│  │  • FFmpeg wrapper          │    │
│  │  • Whisper recognition     │    │
│  │  • LLM correction          │    │
│  └────────────────────────────┘    │
└─────────────────────────────────────┘
```

## 🎯 Supported Features

### Video Processing
- ✅ **Subtitle Generation** - Whisper + LLM correction, 95%+ accuracy
- ✅ **Bilingual Subtitles** - Chinese-English, auto-translation
- ✅ **Batch Processing** - Directories and single files
- ✅ **Video Clipping** - Trim by time range
- ✅ **Format Conversion** - MP4/AVI/MKV conversion
- ✅ **Intelligent Optimization** - Audio denoising, video enhancement

### AI Capabilities
- ✅ **Local LLM** - Qwen2-1.5B, offline operation, no API key needed
- ✅ **Cloud LLM** - DeepSeek/OpenAI for high-quality correction
- ✅ **Professional Terms** - Auto-extract and preserve technical vocabulary
- ✅ **Semantic Correction** - Homophones, contextual error correction

## ⚙️ Configuration

### LLM Setup (Web Interface)

Visit `http://localhost:8000/config.html` for visual configuration:

| LLM Provider | Use Case | API Key Required | Recommendation |
|-------------|----------|------------------|----------------|
| **Local Qwen2** | Offline, privacy-focused | ❌ No | ⭐⭐⭐⭐⭐ |
| **DeepSeek** | Cost-effective, high quality | ✅ Yes | ⭐⭐⭐⭐ |
| **OpenAI** | Highest quality | ✅ Yes | ⭐⭐⭐ |

## 📊 Performance Metrics

| Feature | Avg Time | GPU Acceleration | Quality Score |
|---------|----------|------------------|---------------|
| Subtitle Generation (10min video) | 3-5 min | ✅ Supported | 92/100 |
| LLM Correction (100 lines) | 10-30 sec | ❌ | 95/100 |
| Batch Processing (10 files) | 30-50 min | ✅ Supported | 90/100 |
| Smart Rotation (image) | <1 sec | ✅ Supported | 98/100 |

## 🐳 Docker Deployment

### Method 1: docker-compose (Recommended)

```bash
docker-compose up -d
docker-compose logs -f
docker-compose down
```

### Method 2: Docker Commands

```bash
docker build -t mediaai-tools .
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/output:/app/output \
  --name mediaai \
  mediaai-tools
```

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [FFmpeg](https://ffmpeg.org/) - Video processing
- [LangChain](https://github.com/langchain-ai/langchain) - LLM framework
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [Qwen](https://github.com/QwenLM/Qwen) - Local LLM

## 📧 Contact

- 🐛 Bug Reports: [GitHub Issues](https://github.com/lionelyi/MediaAITools/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/lionelyi/MediaAITools/discussions)
- 👨‍💻 Author: [lionelyi](https://github.com/lionelyi)

---

<div align="center">

**If this project helps you, please give it a ⭐️ Star!**

[![GitHub stars](https://img.shields.io/github/stars/lionelyi/MediaAITools?style=social)](https://github.com/lionelyi/MediaAITools)
[![GitHub forks](https://img.shields.io/github/forks/lionelyi/MediaAITools?style=social)](https://github.com/lionelyi/MediaAITools)

**Solo Developer** | Built from scratch | **MVP** in active development | **Contributions welcome**

Made with ❤️ by [lionelyi](https://github.com/lionelyi)

</div>
