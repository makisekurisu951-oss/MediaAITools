# MediaAI Tools - 技术架构文档

## 🏗️ 核心技术栈

### 1. **LangChain** 
用于 LLM 集成和消息处理

**使用位置：**
- `src/llm/providers.py` - LLM 提供商封装
- `src/agent/media_agent.py` - Agent 消息处理
- `src/agent/base_agent.py` - 基础 Agent 类

**功能：**
- ✅ LLM 消息格式化（HumanMessage, AIMessage, SystemMessage）
- ✅ 多个 LLM 提供商统一接口（OpenAI, DeepSeek, 通义千问）
- ✅ 本地模型集成（Qwen2-VL）
- ✅ 对话历史管理

### 2. **LangGraph** 
用于复杂任务的工作流编排

**使用位置：**
- `src/agent/workflow.py` - 状态图工作流

**功能：**
- ✅ 多步骤任务编排
- ✅ 条件分支
- ✅ 状态管理
- ✅ 错误处理

**工作流示例：**
```
用户指令 → 分析意图 → 规划步骤 → 执行工具 → 返回结果
            ↓          ↓           ↓
         (LLM)    (规划器)     (MCP Tools)
```

### 3. **MCP (Model Context Protocol)**
工具和技能的标准化接口

**使用位置：**
- `src/mcp_server/media_mcp_server.py` - MCP 服务器
- `src/mcp_server/tools.py` - 工具集合

**工具列表：**
- ✅ SubtitleTool - 字幕生成（Whisper + LLM纠错）
- ✅ ClipTool - 视频剪辑
- ✅ ConcatTool - 视频拼接
- ✅ FormatTool - 格式转换
- ✅ OptimizeTool - 视频优化
- ✅ ImageTool - 图像处理

**MCP 优势：**
- 统一的工具调用接口
- 参数验证
- 错误处理
- 日志记录

### 4. **Agent 架构**
智能任务调度和执行

**组件：**
- `src/agent/base_agent.py` - 基础 Agent
- `src/agent/media_agent.py` - 媒体处理 Agent

**功能：**
- ✅ 自然语言理解
- ✅ 意图分析（使用 LLM）
- ✅ 任务路由
- ✅ 工具选择
- ✅ 上下文管理

**处理流程：**
```python
# 用户输入
"为这个视频添加中文字幕并智能纠错"

# Agent 分析
→ 识别任务类型：subtitle
→ 提取参数：语言=中文, 纠错=启用
→ 选择工具：SubtitleTool
→ 调用 MCP Server
→ 返回结果
```

### 5. **Skills 系统**
可复用的技能模块

**使用位置：**
- `src/skills/` - 技能模块目录
- `src/skills/skill_registry.py` - 技能注册表

**技能列表：**
- ✅ ClipSkill - 剪辑技能
- ✅ SubtitleSkill - 字幕技能
- ✅ FormatSkill - 格式转换技能
- ✅ OptimizeSkill - 优化技能
- ✅ ImageSkill - 图像处理技能

**优势：**
- 模块化设计
- 易于扩展
- 独立测试
- 动态加载

## 📊 完整架构图

```
┌─────────────────────────────────────────────────────┐
│                   Web 前端                           │
│  (自然语言输入 + 文件上传 + 配置界面)                  │
└────────────────────┬────────────────────────────────┘
                     │ HTTP/WebSocket
                     ↓
┌─────────────────────────────────────────────────────┐
│                 FastAPI 后端                         │
│  - 文件上传管理                                      │
│  - 任务队列                                          │
│  - 配置管理                                          │
│  - 实时进度更新                                      │
└────────────────────┬────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        ↓            ↓            ↓
   ┌─────────┐  ┌─────────┐  ┌─────────┐
   │ MediaAgent│ │   MCP   │  │ Skills  │
   │(LangChain)│ │ Server  │  │Registry │
   └─────┬───┘  └────┬────┘  └────┬────┘
         │           │            │
         └───────────┼────────────┘
                     │
         ┌───────────┴───────────┐
         ↓                       ↓
   ┌──────────┐           ┌──────────┐
   │LangGraph │           │   Tools  │
   │ Workflow │           │  (MCP)   │
   └─────┬────┘           └────┬─────┘
         │                     │
         └──────────┬──────────┘
                    ↓
         ┌──────────────────────┐
         │   底层工具/模型       │
         ├──────────────────────┤
         │ Whisper (语音识别)   │
         │ Qwen2-1.5B (智能纠错)  │
         │ FFmpeg (视频处理)    │
         │ Pillow (图像处理)    │
         └──────────────────────┘
```

## 🔄 数据流示例

### 场景：添加智能纠错字幕

```
1. 用户输入
   "为 video.mp4 添加中文字幕，发音不标准需要智能纠错"
   
2. FastAPI 接收请求
   POST /api/process
   
3. 调用 MediaAgent
   agent.process(instruction, context)
   
4. LangChain 意图分析
   → 任务类型：subtitle
   → 参数：language=zh, use_llm_correction=true
   
5. LangGraph 工作流编排
   analyze → plan → execute
   
6. MCP Server 调用工具
   call_tool("generate_subtitle", params)
   
7. SubtitleTool 执行
   ├─ FFmpeg 提取音频
   ├─ Whisper 转录
   ├─ Qwen2-VL 智能纠错（上下文分析）
   ├─ 生成 SRT
   └─ FFmpeg 嵌入字幕
   
8. 返回结果
   → 带字幕视频
   → SRT 文件
```

## 🚀 快速集成示例

### 使用 Agent + LangChain
```python
from agent.media_agent import MediaAgent

agent = MediaAgent()
result = await agent.process(
    "为这个视频添加字幕",
    context={"video_paths": ["video.mp4"]}
)
```

### 使用 MCP Server
```python
from mcp_server.media_mcp_server import MediaMCPServer

mcp = MediaMCPServer()
result = await mcp.call_tool("generate_subtitle", {
    "video_path": "video.mp4",
    "language": "zh",
    "use_llm_correction": True
})
```

### 使用 LangGraph 工作流
```python
from agent.workflow import run_workflow

result = await run_workflow(
    "为视频添加字幕",
    video_paths=["video.mp4"]
)
```

### 使用 Skills
```python
from skills import SkillRegistry

registry = SkillRegistry()
subtitle_skill = registry.get_skill("subtitle")
result = await subtitle_skill.execute({
    "video_path": "video.mp4",
    "language": "zh"
})
```

## 📦 依赖库

```
# LangChain 相关
langchain-core>=0.1.0
langchain-openai>=0.0.5
langgraph>=0.0.20

# AI 模型
transformers>=4.36.0
torch>=2.1.0
openai-whisper>=20231117

# 媒体处理
ffmpeg-python>=0.2.0
Pillow>=10.0.0

# Web 服务
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
```

## 🎯 技术亮点

1. **模块化设计**
   - Agent、MCP、Skills 各司其职
   - 松耦合，易扩展

2. **智能纠错**
   - 结合 Whisper + LLM
   - 上下文理解
   - 处理发音不标准

3. **工作流编排**
   - LangGraph 状态图
   - 复杂任务自动化
   - 条件分支

4. **统一接口**
   - MCP 协议
   - 所有工具标准化
   - 易于集成

5. **灵活配置**
   - 支持多种 LLM
   - 本地/云端切换
   - Web 界面配置

---

这个架构充分利用了现代 AI 框架的优势，提供了一个强大、灵活、易用的视频处理平台！
