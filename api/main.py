"""FastAPI 后端服务 - MediaAI Tools
集成 LangChain, MCP, Agent, Skills 架构
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import sys
import asyncio
import uuid
from pathlib import Path
import shutil

# 添加 src 到路径
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

# 导入核心组件
from agent.media_agent import MediaAgent
from mcp_server.media_mcp_server import MediaMCPServer
from skills import SkillRegistry
from llm.llm_manager import get_llm_manager
from config import load_config

# 配置logging（必须在其他模块导入前）
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# 设置所有相关logger为DEBUG级别
for logger_name in ['MediaAITools', 'src', 'mcp_server', 'tools']:
    logging.getLogger(logger_name).setLevel(logging.DEBUG)

# 导入配置路由
from .config_routes import router as config_router

app = FastAPI(title="MediaAI Tools API", version="1.0.0")

# 注册配置管理路由
app.include_router(config_router)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化核心组件
media_agent: Optional[MediaAgent] = None
mcp_server: Optional[MediaMCPServer] = None
skill_registry: Optional[SkillRegistry] = None

# 任务状态存储
tasks_status: Dict[str, Dict[str, Any]] = {}

# 配置目录
UPLOAD_DIR = project_root / "uploads"
OUTPUT_DIR = project_root / "output"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def make_json_serializable(obj: Any) -> Any:
    """将对象转换为 JSON 可序列化的格式"""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif hasattr(obj, '__dict__'):
        # 对于有 __dict__ 的对象，只保留简单的字符串表示
        return str(obj)
    else:
        return str(obj)


@app.on_event("startup")
async def startup_event():
    """启动时初始化组件"""
    global media_agent, mcp_server, skill_registry
    
    print("🚀 初始化 MediaAI Tools...")
    
    # 初始化 MCP Server
    mcp_server = MediaMCPServer()
    print("✅ MCP Server 初始化完成")
    
    # 初始化 Skills Registry
    skill_registry = SkillRegistry()
    print("✅ Skills Registry 初始化完成")
    
    # 初始化 MediaAgent (LangChain)
    try:
        media_agent = MediaAgent()
        print("✅ MediaAgent 初始化完成")
    except Exception as e:
        print(f"⚠️  MediaAgent 初始化失败: {e}")
        media_agent = None
    
    print("✨ 系统就绪！")


class TaskRequest(BaseModel):
    """任务请求模型"""
    instruction: str
    video_paths: Optional[List[str]] = None
    language: Optional[str] = "zh"
    bilingual: Optional[bool] = False
    use_llm_correction: Optional[bool] = True


class ConfigUpdate(BaseModel):
    """配置更新模型"""
    llm_provider: Optional[str] = None
    model_path: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@app.get("/api/info")
async def api_info():
    """API 信息（原 / 路径）"""
    return {
        "message": "MediaAI Tools API",
        "version": "2.0.0",
        "status": "running",
        "description": "智能媒体处理服务 - Router/Memory/Evaluator/LangGraph/LangChain/MCP",
        "components": {
            "agent": media_agent is not None,
            "mcp_server": mcp_server is not None,
            "skills": skill_registry is not None,
            "router": media_agent.router is not None if media_agent else False,
            "memory": media_agent.memory is not None if media_agent else False,
            "evaluator": media_agent.evaluator is not None if media_agent else False
        }
    }


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "agent_ready": media_agent is not None,
        "mcp_ready": mcp_server is not None,
        "skills_count": len(skill_registry.list_skills()) if skill_registry else 0,
        "router_enabled": media_agent.router is not None if media_agent else False,
        "memory_enabled": media_agent.memory is not None if media_agent else False,
        "evaluator_enabled": media_agent.evaluator is not None if media_agent else False
    }


@app.get("/api/config")
async def get_config():
    """获取当前配置"""
    config = load_config()
    llm_manager = get_llm_manager()
    
    return {
        "llm": config.get("llm", {}),
        "media": config.get("media", {}),
        "current_provider": llm_manager.get_provider().__class__.__name__ if llm_manager.get_provider() else None
    }


@app.post("/api/config")
async def update_config(config_update: ConfigUpdate):
    """更新配置"""
    try:
        # 注意：当前 load_config 不支持动态保存，这里只返回成功
        # 实际配置修改需要手动编辑 config.yaml
        return {"status": "success", "message": "配置更新功能暂不可用，请手动编辑 src/config/config.yaml"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传视频文件"""
    try:
        file_id = str(uuid.uuid4())
        file_ext = Path(file.filename).suffix
        save_path = UPLOAD_DIR / f"{file_id}{file_ext}"
        
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "path": str(save_path),
            "size": save_path.stat().st_size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tasks/subtitle")
async def create_subtitle_task(
    background_tasks: BackgroundTasks,
    video_path: str = Form(...),
    language: str = Form("zh"),
    bilingual: bool = Form(False),
    use_llm_correction: bool = Form(True),
    instruction: Optional[str] = Form(None)
):
    """创建字幕生成任务"""
    task_id = str(uuid.uuid4())
    
    # 初始化任务状态
    tasks_status[task_id] = {
        "status": "pending",
        "progress": 0,
        "message": "任务已创建",
        "result": None
    }
    
    # 后台执行任务
    background_tasks.add_task(
        process_subtitle_task,
        task_id,
        video_path,
        language,
        bilingual,
        use_llm_correction
    )
    
    return {"task_id": task_id}


async def process_subtitle_task(
    task_id: str,
    video_path: str,
    language: str,
    bilingual: bool,
    use_llm_correction: bool
):
    """处理字幕生成任务 - 使用 MCP SubtitleTool"""
    try:
        tasks_status[task_id]["status"] = "processing"
        tasks_status[task_id]["message"] = "正在提取音频..."
        tasks_status[task_id]["progress"] = 10
        
        # 确定输出路径
        video_file = Path(video_path)
        output_path = OUTPUT_DIR / f"{video_file.stem}-字幕{video_file.suffix}"
        
        tasks_status[task_id]["message"] = "正在转录音频（Whisper）..."
        tasks_status[task_id]["progress"] = 30
        
        # 使用 MCP Server 的 SubtitleTool
        if mcp_server:
            result = await mcp_server.call_tool("generate_subtitle", {
                "video_path": video_path,
                "output_path": str(output_path),
                "language": language,
                "embed_subtitle": True,
                "bilingual": bilingual,
                "use_llm_correction": use_llm_correction
            })
        else:
            # Fallback: 直接使用工具
            from mcp_server.tools import SubtitleTool
            tool = SubtitleTool()
            result = tool.execute(
                video_path=video_path,
                output_path=str(output_path),
                language=language,
                embed_subtitle=True,
                bilingual=bilingual,
                use_llm_correction=use_llm_correction
            )
        
        if result.get("success"):
            tasks_status[task_id]["status"] = "completed"
            tasks_status[task_id]["progress"] = 100
            tasks_status[task_id]["message"] = "字幕生成成功"
            tasks_status[task_id]["result"] = {
                "output_path": result.get("output_path"),
                "srt_path": result.get("srt_path")
            }
        else:
            tasks_status[task_id]["status"] = "failed"
            tasks_status[task_id]["message"] = result.get("error", "未知错误")
    
    except Exception as e:
        tasks_status[task_id]["status"] = "failed"
        tasks_status[task_id]["message"] = str(e)


@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    """获取任务状态"""
    if task_id not in tasks_status:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    # 清理结果确保可以 JSON 序列化
    return make_json_serializable(tasks_status[task_id])


@app.get("/api/download/{file_id}")
async def download_file(file_id: str):
    """下载处理后的文件"""
    # 在输出目录中查找文件
    for file_path in OUTPUT_DIR.glob("*"):
        if file_id in str(file_path):
            return FileResponse(
                path=file_path,
                filename=file_path.name,
                media_type="application/octet-stream"
            ) 
    #使用 MediaAgent (LangChain + Skills)
    if not media_agent:
        raise HTTPException(status_code=503, detail="MediaAgent 未初始化")
    
    task_id = str(uuid.uuid4())
    
    tasks_status[task_id] = {
        "status": "pending",
        "progress": 0,
        "message": "正在通过 AI Agent 解析指令...",
        "result": None
    }
    
    # 后台执行 Agent 任务
    background_tasks.add_task(
        process_agent_task,
        task_id,
        request.instruction,
        request.video_paths,
        {
            "language": request.language,
            "bilingual": request.bilingual,
            "use_llm_correction": request.use_llm_correction
        }
    )
    
    return {"task_id": task_id}


async def process_agent_task(
    task_id: str,
    instruction: str,
    video_paths: Optional[List[str]],
    options: Dict[str, Any]
):
    """使用 MediaAgent 处理任务"""
    try:
        tasks_status[task_id]["status"] = "processing"
        tasks_status[task_id]["message"] = "AI Agent 正在分析任务..."
        tasks_status[task_id]["progress"] = 10
        
        # 构建上下文
        context = {
            "video_paths": video_paths or [],
            "options": options,
            "output_dir": str(OUTPUT_DIR)
        }
        
        # 使用 MediaAgent 处理（LangChain + Skills）
        result = await media_agent.process(instruction, context)
        
        tasks_status[task_id]["progress"] = 90
        
        # 清理结果，确保可以 JSON 序列化
        clean_result = make_json_serializable(result)
        
        # 判断任务是否成功：有 success 字段且为 True，或者没有 error/error为空
        is_success = result.get("success", True) if "success" in result else not result.get("error")
        
        if is_success:
            tasks_status[task_id]["status"] = "completed"
            tasks_status[task_id]["progress"] = 100
            tasks_status[task_id]["message"] = "任务完成"
            tasks_status[task_id]["result"] = clean_result
        else:
            tasks_status[task_id]["status"] = "failed"
            tasks_status[task_id]["message"] = result.get("error", "任务失败")
            tasks_status[task_id]["result"] = clean_result
    
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        tasks_status[task_id]["status"] = "failed"
        tasks_status[task_id]["message"] = str(e)
        tasks_status[task_id]["error_details"] = error_msg


@app.post("/api/mcp/tools")
async def list_mcp_tools():
    """列出所有 MCP 工具"""
    if not mcp_server:
        raise HTTPException(status_code=503, detail="MCP Server 未初始化")
    
    tools = mcp_server.list_tools()
    return {"tools": tools}


@app.post("/api/mcp/execute")
async def execute_mcp_tool(
    tool_name: str = Form(...),
    parameters: str = Form(...)  # JSON string
):
    """执行 MCP 工具"""
    if not mcp_server:
        raise HTTPException(status_code=503, detail="MCP Server 未初始化")
    
    import json
    params = json.loads(parameters)
    
    result = await mcp_server.call_tool(tool_name, params)
    return result


@app.post("/api/agent/process")
async def process_with_agent(
    request: TaskRequest,
    background_tasks: BackgroundTasks
):
    """使用 MediaAgent 处理自然语言任务请求
    
    示例请求:
    {
        "instruction": "帮我把D:\\MediaAITools\\test\\subtitle-test目录下面的mp4文件添加字幕,源文件不变"
    }
    """
    if not media_agent:
        raise HTTPException(status_code=503, detail="MediaAgent 未初始化")
    
    task_id = str(uuid.uuid4())
    
    tasks_status[task_id] = {
        "status": "pending",
        "progress": 0,
        "message": "正在通过 AI Agent 解析指令...",
        "result": None
    }
    
    # 后台执行 Agent 任务
    background_tasks.add_task(
        process_agent_task,
        task_id,
        request.instruction,
        request.video_paths,
        {
            "language": request.language,
            "bilingual": request.bilingual,
            "use_llm_correction": request.use_llm_correction
        }
    )
    
    return {"task_id": task_id, "message": "任务已创建，Agent 正在分析指令"}


@app.get("/api/skills")
async def list_skills():
    """列出所有技能"""
    if not skill_registry:
        raise HTTPException(status_code=503, detail="Skills Registry 未初始化")
    
    skill_names = skill_registry.list_skills()
    return {
        "skills": [
            {
                "name": name,
                "class": skill_registry.get_skill(name).__class__.__name__
            }
            for name in skill_names
        ]
    }


@app.get("/api/memory/summary")
async def get_memory_summary():
    """获取记忆摘要"""
    if not media_agent or not media_agent.memory:
        raise HTTPException(status_code=503, detail="记忆系统未初始化")
    
    summary = media_agent.get_memory_summary()
    return summary


@app.get("/api/memory/history")
async def get_conversation_history(n: int = 10):
    """获取对话历史"""
    if not media_agent or not media_agent.memory:
        raise HTTPException(status_code=503, detail="记忆系统未初始化")
    
    history = media_agent.get_conversation_history(n)
    return {"history": history}


@app.get("/api/evaluator/stats")
async def get_evaluator_stats():
    """获取评估统计"""
    if not media_agent or not media_agent.evaluator:
        raise HTTPException(status_code=503, detail="评估器未初始化")
    
    stats = media_agent.evaluator.get_statistics()
    return stats


# 挂载静态文件（Web界面）
web_dir = project_root / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
else:
    print(f"⚠️ Web 目录不存在: {web_dir}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
