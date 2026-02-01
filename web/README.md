# MediaAI Tools Web 界面说明

## 文件结构

- **index.html** - 主用户界面（自然语言交互）
- **config.html** - LLM 配置管理页面

## 主界面功能

### 1. 自然语言输入
用户可以直接用中文描述需求，例如：
- "帮我把 D:\videos 目录下的所有 MP4 文件添加字幕"
- "给 D:\test\video.mp4 生成中英双语字幕"
- "把 D:\images 目录下的图片调整为 9:16 竖屏"
- "切换到 DeepSeek 模型"

### 2. 实时进度显示
- 进度条显示处理百分比
- 状态文本实时更新
- 处理结果展示（成功/失败）
- 输出文件路径展示

### 3. 快速示例
点击示例标签可以快速填充常用指令

## 使用方法

1. 启动 API 服务：
```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

2. 访问网页界面：
```
http://localhost:8000/
```

3. 访问 LLM 配置页面：
```
http://localhost:8000/config.html
```

4. 访问 API 文档：
```
http://localhost:8000/docs
```

## 快捷键

- **Ctrl + Enter** - 发送指令

## 技术栈

- 纯 HTML + CSS + JavaScript（无框架依赖）
- 异步轮询机制（任务状态监控）
- RESTful API 调用
