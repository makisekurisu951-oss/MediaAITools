# 贡献指南 Contributing Guide

感谢你对 MediaAI Tools 项目的关注！我们欢迎任何形式的贡献。

## 🚀 快速开始

### 1. Fork 项目

访问 [https://github.com/lionelyi/MediaAITools](https://github.com/lionelyi/MediaAITools)，点击右上角的 "Fork" 按钮。

### 2. 克隆代码

```bash
git clone https://github.com/你的用户名/MediaAITools.git
cd MediaAITools
```

### 3. 设置开发环境

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r src/requirements.txt
pip install -r api/requirements.txt

# 安装开发工具
pip install pytest black flake8 mypy
```

### 4. 配置 LLM

```bash
# 复制配置模板
cp src/config/config.example.yaml src/config/config.yaml

# 编辑配置文件（可选择本地模型或API）
# 默认使用本地 Qwen2 模型，无需 API Key
```

### 5. 运行测试

```bash
# 运行所有测试
pytest test/

# 运行特定测试
pytest test/test_subtitle_tool.py

# 查看覆盖率
pytest --cov=src --cov-report=html
```

## 📝 代码规范

### Python 代码风格

我们使用 **Black** 和 **flake8** 来保持代码风格一致：

```bash
# 格式化代码
black src/ api/ test/

# 检查代码风格
flake8 src/ api/ test/ --max-line-length=100

# 类型检查
mypy src/
```

### 代码规范要点

1. **Imports** - 使用绝对导入，按照标准库、第三方库、本地模块排序
   ```python
   import os
   import sys
   from pathlib import Path
   
   from fastapi import FastAPI
   from langchain_core import BaseMessage
   
   from utils.logger import get_logger
   from config.config_manager import ConfigManager
   ```

2. **日志** - 使用统一的日志系统
   ```python
   from utils.logger import get_logger
   logger = get_logger(__name__)
   
   logger.info("操作开始")
   logger.error("错误信息", exc_info=True)
   ```

3. **异步函数** - Skills 和 Agent 必须使用 async/await
   ```python
   async def execute(self, instruction: str, **kwargs):
       result = await self.some_async_operation()
       return result
   ```

4. **类型注解** - 为函数参数和返回值添加类型注解
   ```python
   def process_video(file_path: str, duration: int) -> Dict[str, Any]:
       pass
   ```

5. **文档字符串** - 为所有公共函数和类添加文档
   ```python
   def clip_video(input_file: str, start_time: float, end_time: float) -> str:
       """裁剪视频片段
       
       Args:
           input_file: 输入视频路径
           start_time: 开始时间（秒）
           end_time: 结束时间（秒）
           
       Returns:
           输出视频路径
           
       Raises:
           FileNotFoundError: 输入文件不存在
           FFmpegError: FFmpeg 执行失败
       """
       pass
   ```

## 🏗️ 项目架构

```
src/
├── agent/              # Agent 层（路由、记忆、评估）
│   ├── router.py       # 智能路由器
│   ├── memory.py       # 会话记忆管理
│   └── evaluator.py    # 任务质量评估
├── skills/             # Skills 层（业务逻辑）
│   ├── subtitle_skill.py
│   ├── batch_skill.py
│   └── image_skill.py
├── mcp_server/         # MCP 层（工具封装）
│   ├── tools.py        # 所有工具实现
│   └── server.py       # MCP 服务器
├── llm/                # LLM 层（模型管理）
│   ├── llm_manager.py  # LLM 管理器
│   └── providers.py    # 各种 LLM 提供商
├── config/             # 配置管理
├── utils/              # 工具函数
└── main.py             # 主入口

api/
├── main.py             # FastAPI 主应用
└── config_routes.py    # LLM 配置路由

web/
├── index.html          # Web UI 主页面
└── config.html         # LLM 配置页面
```

## 🔧 添加新功能

### 添加新工具（Tool）

1. 在 `src/mcp_server/tools.py` 中创建新类，继承 `MediaTool`：

```python
class NewTool(MediaTool):
    """新工具描述"""
    
    def __init__(self):
        super().__init__(
            name="new_tool",
            description="工具功能描述"
        )
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """执行工具逻辑"""
        # 实现具体逻辑
        return {
            "success": True,
            "message": "操作成功",
            "data": result
        }
```

2. 在 `MediaMCPServer` 的 `available_tools` 中注册：

```python
self.available_tools = {
    # ... 现有工具 ...
    "new_tool": NewTool(),
}
```

### 添加新技能（Skill）

1. 在 `src/skills/` 创建新文件 `new_skill.py`：

```python
from pathlib import Path
import sys
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from skills.base_skill import BaseSkill
from utils.logger import get_logger

logger = get_logger(__name__)

class NewSkill(BaseSkill):
    """新技能描述"""
    
    def __init__(self, mcp_server):
        super().__init__(
            name="new_skill",
            description="技能功能描述",
            mcp_server=mcp_server
        )
    
    async def execute(self, instruction: str, **kwargs):
        """执行技能逻辑"""
        logger.info(f"执行新技能: {instruction}")
        
        # 调用工具
        tool_result = self.mcp_server.execute_tool(
            "some_tool",
            param1="value1"
        )
        
        return {
            "success": True,
            "message": "技能执行成功",
            "result": tool_result
        }
```

2. 在 `SkillRegistry._register_default_skills()` 中注册：

```python
self.register(
    NewSkill(mcp_server),
    aliases=["新技能", "new_skill", "别名1", "别名2"]
)
```

3. 更新 `IntelligentRouter` 的 `keyword_patterns`：

```python
'NEW_SKILL': [r'新技能', r'别名1', r'别名2']
```

### 添加新 LLM 提供商

1. 在 `src/llm/providers.py` 创建新类：

```python
class NewLLMProvider(BaseLLMProvider):
    """新 LLM 提供商"""
    
    def __init__(self, config: Dict):
        super().__init__("new_provider", config)
        self.client = None  # 初始化客户端
    
    def get_llm(self):
        """返回 LangChain LLM 实例"""
        # 实现逻辑
        pass
    
    def is_available(self) -> bool:
        """检查是否可用"""
        return self.config.get("api_key") is not None
```

2. 在 `LLMManager._initialize_providers()` 添加初始化：

```python
if "new_provider" in providers:
    self.providers["new_provider"] = NewLLMProvider(
        providers["new_provider"]
    )
```

3. 在 `config.yaml` 添加配置节：

```yaml
llm:
  providers:
    new_provider:
      api_key: "your_key"
      base_url: "https://api.example.com"
```

## 🧪 测试指南

### 编写测试

在 `test/` 目录创建测试文件：

```python
import pytest
import asyncio
from pathlib import Path

from agent.media_agent import MediaAgent

class TestNewFeature:
    """测试新功能"""
    
    @pytest.fixture
    async def agent(self):
        """创建 Agent 实例"""
        return MediaAgent()
    
    @pytest.mark.asyncio
    async def test_feature(self, agent):
        """测试具体功能"""
        result = await agent.process("测试指令")
        
        assert result["success"] is True
        assert "data" in result
```

### 运行测试

```bash
# 所有测试
pytest

# 特定文件
pytest test/test_new_feature.py

# 特定测试
pytest test/test_new_feature.py::TestNewFeature::test_feature

# 详细输出
pytest -v -s

# 覆盖率报告
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

## 📤 提交 PR

### Commit 规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

**类型（type）：**
- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 重构（不是新功能也不是修复）
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

**示例：**
```
feat(subtitle): 添加双语字幕生成功能

- 支持中英双语
- 自动翻译中文字幕
- 调整时间轴对齐

Closes #123
```

### PR 流程

1. **创建分支**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **提交代码**
   ```bash
   git add .
   git commit -m "feat: 添加新功能"
   git push origin feature/your-feature-name
   ```

3. **创建 Pull Request**
   - 访问 GitHub 仓库页面
   - 点击 "New Pull Request"
   - 填写 PR 描述，说明改动内容
   - 关联相关 Issue

4. **代码审查**
   - 等待维护者审查
   - 根据反馈修改代码
   - 所有检查通过后合并

### PR 模板

```markdown
## 描述
简要描述这个 PR 的目的和改动内容。

## 改动类型
- [ ] Bug 修复
- [ ] 新功能
- [ ] 重构
- [ ] 文档更新
- [ ] 测试
- [ ] 其他

## 测试
描述如何测试这些改动：
- [ ] 添加了单元测试
- [ ] 手动测试通过
- [ ] 已有测试通过

## 相关 Issue
Closes #issue_number

## 截图（如果适用）
添加截图帮助说明改动。

## Checklist
- [ ] 代码符合项目规范
- [ ] 添加了必要的文档
- [ ] 所有测试通过
- [ ] 更新了 CHANGELOG.md
```

## 🐛 报告 Bug

使用 [GitHub Issues](https://github.com/lionelyi/MediaAITools/issues) 报告 Bug：

**Bug 报告模板：**
```markdown
### 问题描述
清晰简洁地描述问题。

### 复现步骤
1. 执行 '...'
2. 点击 '...'
3. 滚动到 '...'
4. 看到错误

### 预期行为
描述你期望发生的行为。

### 实际行为
描述实际发生的行为。

### 环境信息
- OS: [Windows 10 / Ubuntu 22.04 / macOS 13]
- Python 版本: [3.10.0]
- 项目版本: [v1.0.0]
- Docker: [是/否]
- GPU: [NVIDIA RTX 3090 / CPU only]

### 日志
粘贴相关日志：
```
[日志内容]
```

### 截图
如果适用，添加截图帮助说明问题。
```

## 💡 功能请求

使用 GitHub Issues 提交功能请求：

```markdown
### 功能描述
清晰简洁地描述你想要的功能。

### 使用场景
描述这个功能的使用场景和价值。

### 可能的实现方式
如果有想法，描述可能的实现方式。

### 替代方案
描述你考虑过的替代方案。
```

## 📚 文档贡献

文档同样重要！你可以：

- 修复文档中的错误
- 改进现有文档的清晰度
- 添加新的示例和教程
- 翻译文档到其他语言

## 🎯 优先级标签

我们使用以下标签来标识 Issue 优先级：

- `priority: critical` - 严重 Bug，需要立即修复
- `priority: high` - 重要功能或 Bug
- `priority: medium` - 中等优先级
- `priority: low` - 可以稍后处理

## 🙏 行为准则

- 尊重所有贡献者
- 欢迎新手和提问
- 建设性地提供反馈
- 专注于代码质量而非个人

## 📧 联系方式

- 问题讨论：[GitHub Discussions](https://github.com/lionelyi/MediaAITools/discussions)
- Bug 报告：[GitHub Issues](https://github.com/lionelyi/MediaAITools/issues)
- 作者主页：[lionelyi](https://github.com/lionelyi)

> 💡 **关于作者**：这个项目由 lionelyi 独立开发完成，从0到1构建。目前是MVP版本，欢迎社区贡献让它变得更好！

---

再次感谢你的贡献！🎉
