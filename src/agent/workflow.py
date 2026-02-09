"""LangGraph 工作流 - 复杂任务编排
用于处理多步骤的视频处理任务
"""

from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator

# 定义状态
class AgentState(TypedDict):
    """Agent 状态"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    instruction: str
    video_paths: list
    task_type: str
    parameters: dict
    current_step: str
    results: list
    error: str


def analyze_intent(state: AgentState) -> AgentState:
    """分析用户意图"""
    instruction = state["instruction"]
    
    # 简单的意图分析（可以用 LLM 增强）
    if "字幕" in instruction or "subtitle" in instruction.lower():
        state["task_type"] = "subtitle"
    elif "剪辑" in instruction or "clip" in instruction.lower():
        state["task_type"] = "clip"
    elif "转换" in instruction or "convert" in instruction.lower():
        state["task_type"] = "format"
    elif "优化" in instruction or "optimize" in instruction.lower():
        state["task_type"] = "optimize"
    else:
        state["task_type"] = "unknown"
    
    state["current_step"] = "analyzed"
    return state


def plan_execution(state: AgentState) -> AgentState:
    """规划执行步骤"""
    task_type = state["task_type"]
    
    if task_type == "subtitle":
        state["parameters"] = {
            "steps": [
                {"action": "extract_audio", "tool": "ffmpeg"},
                {"action": "transcribe", "tool": "whisper"},
                {"action": "correct", "tool": "llm"},
                {"action": "embed", "tool": "ffmpeg"}
            ]
        }
    elif task_type == "clip":
        state["parameters"] = {
            "steps": [
                {"action": "clip_video", "tool": "ffmpeg"}
            ]
        }
    
    state["current_step"] = "planned"
    return state


def execute_tools(state: AgentState) -> AgentState:
    """执行工具调用"""
    from mcp_server import MediaMCPServer
    from utils.logger import get_logger
    
    logger = get_logger(__name__)
    state["current_step"] = "executing"
    
    # 实际调用 MCP Server 工具
    task_type = state.get("task_type")
    video_paths = state.get("video_paths", [])
    instruction = state.get("instruction", "")
    
    if not video_paths:
        # 尝试从指令中提取视频路径
        import re
        path_pattern = r'([A-Za-z]:[\\//][^"<>|*?\n]+\.(mp4|avi|mov|mkv|wmv|flv|webm))'
        match = re.search(path_pattern, instruction, re.IGNORECASE)
        if match:
            video_paths = [match.group(1)]
    
    if not video_paths:
        state["error"] = "No video file found"
        return state
    
    # 根据任务类型调用相应工具
    mcp_server = MediaMCPServer()
    results = []
    
    try:
        for video_path in video_paths:
            if task_type == "subtitle":
                # 调用字幕生成工具
                # 检测语言
                language = "en"  # 默认英文
                if "中文" in instruction or "中文" in instruction or "汉语" in instruction:
                    language = "zh"
                elif "英文" in instruction or "英语" in instruction or "english" in instruction.lower():
                    language = "en"
                
                result = mcp_server.call_tool(
                    "generate_subtitle",
                    video_path=video_path,
                    language=language
                )
                results.append(result)
            elif task_type == "clip":
                # 调用视频剪辑工具（需要提取时间参数）
                result = mcp_server.call_tool(
                    "clip_video",
                    input_path=video_path,
                    start_time="00:00:00",
                    end_time="00:00:10",
                    output_path=video_path.replace(".mp4", "-clip.mp4")
                )
                results.append(result)
            # 可以添加更多任务类型
        
        state["results"] = results
        logger.info(f"Workflow executed {len(results)} tasks")
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        state["error"] = str(e)
    
    return state


def should_continue(state: AgentState) -> str:
    """决定是否继续"""
    if state.get("error"):
        return "error"
    if state["current_step"] == "analyzed":
        return "plan"
    if state["current_step"] == "planned":
        return "execute"
    return "end"


# 创建 LangGraph 工作流
def create_media_workflow() -> StateGraph:
    """创建媒体处理工作流"""
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("analyze", analyze_intent)
    workflow.add_node("plan", plan_execution)
    workflow.add_node("execute", execute_tools)
    
    # 添加边
    workflow.set_entry_point("analyze")
    
    workflow.add_conditional_edges(
        "analyze",
        should_continue,
        {
            "plan": "plan",
            "error": END,
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "plan",
        should_continue,
        {
            "execute": "execute",
            "error": END,
            "end": END
        }
    )
    
    workflow.add_edge("execute", END)
    
    return workflow.compile()


# 使用示例
async def run_workflow(instruction: str, video_paths: list):
    """运行工作流"""
    workflow = create_media_workflow()
    
    initial_state = {
        "messages": [HumanMessage(content=instruction)],
        "instruction": instruction,
        "video_paths": video_paths,
        "task_type": "",
        "parameters": {},
        "current_step": "start",
        "results": [],
        "error": ""
    }
    
    result = await workflow.ainvoke(initial_state)
    return result
