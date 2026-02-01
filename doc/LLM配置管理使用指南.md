# LLM 配置管理使用指南

## 🎯 功能概述

现在您可以通过网页前端直接配置和切换 LLM 提供商，无需手动编辑配置文件。支持：

- ✅ **DeepSeek API** - 在线推理，性价比最高
- ✅ **Qwen2 本地** - 本地纯文本模型，适合字幕处理
- ✅ **OpenAI API** - GPT-4o，效果最佳
- ✅ **通义千问 API** - 阿里云在线服务

## 📋 使用步骤

### 1. 启动后端服务

```powershell
cd D:\MediaAITools
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. 访问配置页面

浏览器打开：`http://localhost:8000/config.html`

或从主页点击 "⚙️ LLM 配置管理" 按钮

### 3. 选择 LLM 提供商

点击任一提供商卡片：

#### 选项 A：DeepSeek API（推荐）

**优点：**
- 🎯 价格最低：1 元 = 100万 token
- ⚡ 速度快：在线推理，无需本地显卡
- 🎨 效果好：专业术语提取 95%+ 准确率
- 💾 无内存占用：不需要加载模型到本地

**配置步骤：**
1. 访问 https://platform.deepseek.com 注册账号
2. 充值 5-10 元（可用很久）
3. 获取 API Key（sk-开头）
4. 在配置页面填入：
   - API Key: `sk-your-key-here`
   - Base URL: `https://api.deepseek.com`（默认）
   - 模型: `deepseek-chat`（推荐）

#### 选项 B：Qwen2.5 本地（离线使用）

**优点：**
- 🔒 隐私保护：本地运行，数据不外传
- 💰 完全免费：无 API 调用费用
- 📝 文本优化：纯文本模型，比 Qwen2-VL 更适合字幕

**配置步骤：**
1. 下载模型（首次使用）：
   ```powershell
   cd D:\MediaAITools
   python download_qwen25.py
   ```
   
2. 等待下载完成（约 6GB，需要 10-30 分钟）

3. 在配置页面选择：
   - 模型路径: `Qwen/Qwen2.5-3B-Instruct`（推荐）
   - 设备: `auto`（自动检测 GPU/CPU）

4. **内存要求：**
   - CPU 模式：8-12GB 系统内存
   - GPU 模式：6-8GB 显存

#### 选项 C：OpenAI API（效果最佳但贵）

**优点：**
- 🏆 效果最好：GPT-4o 智能理解能力强
- 🌍 国际标准：成熟稳定

**缺点：**
- 💸 价格较高：约 $0.005/1K tokens
- 🚫 需要科学上网（或使用镜像）

**配置步骤：**
1. 访问 https://platform.openai.com 注册
2. 充值（最低 $5）
3. 获取 API Key
4. 配置：
   - API Key: `sk-...`
   - Base URL: 留空（或填镜像地址）
   - 模型: `gpt-4o-mini`（便宜）或 `gpt-4o`（最强）

#### 选项 D：通义千问 API

**适用场景：**
- 已有阿里云账号
- 需要国内稳定服务

**配置步骤：**
1. 阿里云控制台开通通义千问
2. 获取 API Key
3. 配置即可

### 4. 测试连接

配置完成后，点击 **"🔗 测试连接"** 按钮验证配置是否正确。

成功示例：
```
✓ 连接测试成功！响应时间: 1234ms
```

失败示例：
```
✗ 连接测试失败: Invalid API key
```

### 5. 保存配置

点击 **"💾 保存配置"** 按钮，配置会自动写入 `src/config/config.yaml`

### 6. 开始使用

返回主页，现在所有字幕生成任务都会使用您选择的 LLM 提供商！

## 🔧 高级配置

### 动态切换提供商

您可以随时切换提供商，无需重启服务：

1. 打开配置页面
2. 选择新的提供商
3. 填写配置
4. 点击保存

系统会自动重新加载 LLM 管理器。

### 配置文件位置

手动编辑配置文件（可选）：
```
D:\MediaAITools\src\config\config.yaml
```

### 环境变量覆盖

优先级更高的配置方式（适合服务器部署）：

```powershell
# Windows
$env:DEEPSEEK_API_KEY="sk-..."
$env:OPENAI_API_KEY="sk-..."

# Linux/Mac
export DEEPSEEK_API_KEY="sk-..."
export OPENAI_API_KEY="sk-..."
```

## 📊 性能对比

| 提供商 | 速度 | 准确率 | 成本 | 内存占用 |
|--------|------|--------|------|----------|
| DeepSeek API | ⭐⭐⭐⭐⭐ | 95% | ¥0.001/次 | 0 MB |
| Qwen2.5 本地 | ⭐⭐⭐⭐ | 85% | 免费 | 6-8 GB |
| OpenAI API | ⭐⭐⭐⭐⭐ | 98% | $0.005/次 | 0 MB |
| Qwen2-VL 本地 | ⭐⭐⭐ | 25% | 免费 | 8-16 GB |

**推荐配置：**
- 💰 **预算优先**：DeepSeek API（性价比最高）
- 🔒 **隐私优先**：Qwen2.5 本地（完全离线）
- 🏆 **效果优先**：OpenAI GPT-4o（准确率最高）

## 🐛 常见问题

### Q1: 保存后提示 "配置已更新" 但实际未生效？

**解决方案：** 点击 "🔄 刷新" 按钮重新加载配置

### Q2: Qwen2.5 本地加载失败？

**可能原因：**
1. 模型未下载：运行 `python download_qwen25.py`
2. 内存不足：增加虚拟内存（参考 `doc/Windows内存优化指南.md`）
3. 模型路径错误：检查 HuggingFace 缓存目录

### Q3: DeepSeek API 返回 401 错误？

**解决方案：**
1. 检查 API Key 是否正确
2. 确认账户有余额
3. 检查 Base URL 是否为 `https://api.deepseek.com`

### Q4: 切换提供商后字幕质量反而变差？

**分析：**
- Qwen2-VL → DeepSeek/Qwen2.5：应该变好 ✅
- DeepSeek → Qwen2-VL：肯定变差（Qwen2-VL 不适合纯文本）❌

**建议：** 使用 DeepSeek API 或 Qwen2.5 本地模型

### Q5: 如何查看当前使用的提供商？

1. 方法一：配置页面底部 "当前提供商" 显示
2. 方法二：查看日志 `src/logs/mediaai.log`
3. 方法三：API 请求 `GET http://localhost:8000/api/config/llm`

## 📝 API 文档

### 获取配置

```bash
curl http://localhost:8000/api/config/llm
```

响应：
```json
{
  "success": true,
  "providers": {
    "deepseek": {
      "enabled": true,
      "api_key": "sk-...",
      "base_url": "https://api.deepseek.com",
      "model": "deepseek-chat"
    },
    "local_qwen": {
      "enabled": false,
      "model_path": "Qwen/Qwen2.5-3B-Instruct",
      "device": "auto"
    }
  }
}
```

### 更新配置

```bash
curl -X POST http://localhost:8000/api/config/llm \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "deepseek",
    "api_key": "sk-...",
    "model": "deepseek-chat"
  }'
```

### 测试连接

```bash
curl -X POST http://localhost:8000/api/config/llm/test \
  -H "Content-Type: application/json" \
  -d '{"provider": "deepseek"}'
```

## 🎓 下一步

配置完成后，您可以：

1. **测试字幕生成：**
   ```powershell
   python test_agent_subtitle.py
   ```

2. **批量处理视频：**
   使用 Web 界面上传视频并输入指令

3. **查看处理日志：**
   ```powershell
   Get-Content src\logs\mediaai.log -Tail 50
   ```

4. **优化提示词：**
   编辑 `src/mcp_server/tools.py` 中的 LLM prompts

---

**提示：** 如遇到问题，请查看：
- 日志文件：`src/logs/mediaai.log`
- 错误诊断：`doc/Qwen2-VL问题分析.md`
- 内存优化：`doc/Windows内存优化指南.md`

