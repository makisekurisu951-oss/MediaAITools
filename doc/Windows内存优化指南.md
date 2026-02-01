# Windows 虚拟内存优化指南

## 错误说明

```
Failed to load local model: 页面文件太小，无法完成操作。 (os error 1455)
```

**原因：** Qwen2-VL-2B 模型需要约 8-16GB 内存，但 Windows 虚拟内存（页面文件）设置太小。

## 快速修复步骤

### 方法1：增加虚拟内存（推荐）

1. **打开系统属性**
   - 按 `Win + Pause/Break` 或右键"此电脑" → 属性
   - 点击"高级系统设置"

2. **进入性能设置**
   - 在"高级"选项卡中，点击"性能"下的"设置"按钮
   - 切换到"高级"选项卡

3. **修改虚拟内存**
   - 点击"虚拟内存"下的"更改"按钮
   - 取消勾选"自动管理所有驱动器的分页文件大小"
   
4. **设置页面文件大小**
   ```
   选择系统盘（通常是 C:）
   
   推荐设置：
   - 初始大小：16384 MB (16 GB)
   - 最大大小：32768 MB (32 GB)
   
   如果内存充足：
   - 初始大小：24576 MB (24 GB)
   - 最大大小：49152 MB (48 GB)
   ```

5. **应用设置**
   - 点击"设置" → "确定"
   - 重启电脑（重要！）

### 方法2：使用 PowerShell 快速设置

以管理员身份运行 PowerShell：

```powershell
# 设置虚拟内存为 16GB-32GB
$computersys = Get-WmiObject Win32_ComputerSystem -EnableAllPrivileges
$computersys.AutomaticManagedPagefile = $False
$computersys.Put()

$pagefile = Get-WmiObject -Query "SELECT * FROM Win32_PageFileSetting WHERE Name='C:\\pagefile.sys'"
if ($pagefile) {
    $pagefile.Delete()
}

Set-WmiInstance -Class Win32_PageFileSetting -Arguments @{
    name="C:\pagefile.sys"; 
    InitialSize=16384;    # 16 GB
    MaximumSize=32768     # 32 GB
}

Write-Host "虚拟内存已设置为 16-32GB，请重启电脑使其生效" -ForegroundColor Green
```

### 方法3：临时释放内存

在运行前执行：

```powershell
# 清理内存缓存
Clear-RecycleBin -Force -ErrorAction SilentlyContinue
[System.GC]::Collect()
[System.GC]::WaitForPendingFinalizers()

# 停止不必要的应用
# 关闭浏览器、IDE 等占用内存的程序
```

## 验证设置

### 检查当前虚拟内存

```powershell
# 查看页面文件设置
Get-WmiObject Win32_PageFileUsage | Select-Object Name, AllocatedBaseSize, CurrentUsage

# 查看物理内存
Get-WmiObject Win32_ComputerSystem | Select-Object TotalPhysicalMemory
```

### 检查可用内存

```powershell
# 查看内存使用情况
Get-Counter '\Memory\Available MBytes'
Get-Counter '\Memory\% Committed Bytes In Use'
```

## 模型内存需求

| 模型 | 最小内存 | 推荐内存 | 虚拟内存推荐 |
|------|---------|---------|-------------|
| Qwen2-VL-2B | 8 GB | 16 GB | 16-32 GB |
| Qwen2-VL-7B | 16 GB | 32 GB | 32-64 GB |

## 优化后测试

修改虚拟内存并重启后，运行测试：

```powershell
cd D:\MediaAITools
$env:PATH = "D:\temp\bin;" + $env:PATH

# 1. 测试 LLM 加载
python test_llm_status.py

# 2. 测试字幕生成
python test_agent_subtitle.py
```

## 常见问题

### Q1: 虚拟内存设置多大合适？

**A:** 
- 物理内存 8GB：虚拟内存 16-24GB
- 物理内存 16GB：虚拟内存 16-32GB
- 物理内存 32GB+：虚拟内存 16-48GB

### Q2: 设置后仍然失败？

**A:** 可能原因：
1. 未重启电脑
2. 磁盘空间不足（需要预留足够空间）
3. 同时运行了其他占内存的程序

**解决方案：**
```powershell
# 1. 检查磁盘空间
Get-PSDrive C | Select-Object Used, Free

# 2. 关闭其他程序
Stop-Process -Name chrome, firefox, code -Force -ErrorAction SilentlyContinue

# 3. 确认重启
Restart-Computer -Confirm
```

### Q3: 还是内存不足怎么办？

**A:** 临时方案：
1. 关闭所有浏览器和 IDE
2. 只保留必要进程
3. 分批处理视频（一次只处理一个）

或使用在线 API：
- 配置 DeepSeek API（更便宜）
- 配置 OpenAI API

## 重要提示

⚠️ **修改虚拟内存后必须重启电脑才能生效！**

⚠️ **确保系统盘有足够空间**（至少预留 50GB）

✓ **推荐设置**：初始 16GB，最大 32GB

✓ **最佳实践**：关闭不必要的应用再运行模型

## 下一步

优化完成后，继续测试：

```powershell
# 清理旧字幕
Remove-Item "D:\MediaAITools\test\subtitle-test\*.srt" -ErrorAction SilentlyContinue

# 运行 Agent v2.0（带 LLM 纠错）
$env:PATH = "D:\temp\bin;" + $env:PATH
cd D:\MediaAITools
python test_agent_subtitle.py
```

预期效果：
- ✓ LLM 自动提取专业术语
- ✓ 智能还原英文和数字
- ✓ 高质量字幕输出
