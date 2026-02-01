"""Main Media Agent with Router, Memory, Evaluator, and Benchmark support"""

from typing import Dict, Any, Optional, List
import time

# Lazy import for langchain
try:
    from langchain_core.messages import HumanMessage
except ImportError:
    # Create dummy class for testing
    class HumanMessage:
        def __init__(self, content=""):
            self.content = content
from .base_agent import BaseAgent
from .router import IntelligentRouter, RouteStrategy
from .memory import MemoryManager
from .evaluator import Evaluator
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from mcp_server import MediaMCPServer
from skills import SkillRegistry
from utils.logger import get_logger
import json
import re

logger = get_logger(__name__)


class MediaAgent(BaseAgent):
    """Main agent for media processing tasks with intelligent routing, memory, and evaluation"""
    
    def __init__(self):
        system_prompt = """You are a professional media processing AI assistant with advanced capabilities.
Your role is to understand user requests and coordinate with specialized agents and tools to complete media processing tasks.

You can:
1. Understand natural language instructions for video editing, subtitle generation, format conversion, etc.
2. Break down complex tasks into steps using intelligent routing
3. Remember conversation history and task context
4. Call appropriate tools and skills to complete tasks
5. Evaluate task execution quality and provide feedback
6. Provide clear feedback on task progress and results

Always be helpful, accurate, and provide detailed explanations of what you're doing."""
        
        super().__init__("MediaAgent", system_prompt)
        self.mcp_server = MediaMCPServer()
        self.skill_registry = SkillRegistry()
        
        # 增强组件
        self.router = IntelligentRouter()
        self.memory = MemoryManager()
        self.evaluator = Evaluator()
        
        logger.info("MediaAgent 已初始化，包含 Router、Memory、Evaluator 组件")
    
    async def process(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process user input and execute media processing tasks with routing, memory, and evaluation"""
        logger.info(f"MediaAgent processing: {user_input}")
        
        # 1. 添加对话到记忆
        self.memory.add_conversation("user", user_input)
        
        # 2. 智能路由决策
        route_result = self.router.route(user_input, context or {})
        logger.info(f"路由决策: {route_result.reasoning}")
        
        # 3. 添加任务到记忆
        task_id = self.memory.add_task(
            instruction=user_input,
            task_type=route_result.task_type.value,
            parameters=route_result.parameters,
            metadata={
                "strategy": route_result.strategy.value,
                "target": route_result.target,
                "confidence": route_result.confidence
            }
        )
        
        # 4. 执行任务
        start_time = time.time()
        try:
            if route_result.strategy == RouteStrategy.DIRECT_TOOL:
                # 直接调用工具
                result = await self._execute_tool(route_result, context)
            elif route_result.strategy == RouteStrategy.WORKFLOW:
                # 使用工作流 - 传递user_input
                result = await self._execute_workflow(route_result, context, user_input)
            elif route_result.strategy == RouteStrategy.SKILL:
                # 使用技能
                result = await self._execute_skill(route_result, context)
            else:
                # 使用智能体分析
                result = await self._execute_with_analysis(user_input, context)
            
            execution_time = time.time() - start_time
            success = True
            
        except Exception as e:
            logger.error(f"任务执行失败: {str(e)}")
            execution_time = time.time() - start_time
            success = False
            result = {"error": str(e)}
        
        # 5. 添加执行结果到记忆
        self.memory.add_result(
            task_id=task_id,
            success=success,
            result=result,
            execution_time=execution_time
        )
        
        # 6. 评估执行结果
        if success:
            evaluation = self.evaluator.evaluate(
                task_id=task_id,
                task_type=route_result.task_type.value,
                result=result,
                metadata={"execution_time": execution_time}
            )
            
            logger.info(f"任务评估: {evaluation.quality_level.value} (分数: {evaluation.overall_score:.2f})")
            
            result["evaluation"] = {
                "score": evaluation.overall_score,
                "quality": evaluation.quality_level.value,
                "recommendations": evaluation.recommendations
            }
        
        # 7. 添加助手回复到记忆
        response_text = f"任务完成，耗时 {execution_time:.2f}s"
        if success and "evaluation" in result:
            response_text += f"，质量评分: {result['evaluation']['score']:.1f}"
        self.memory.add_conversation("assistant", response_text)
        
        # 8. 返回结果（包含路由和评估信息）
        result["routing"] = {
            "task_type": route_result.task_type.value,
            "strategy": route_result.strategy.value,
            "target": route_result.target,
            "confidence": route_result.confidence
        }
        result["execution_time"] = execution_time
        result["task_id"] = task_id
        
        return result
    
    async def _execute_tool(self, route_result, context: Optional[Dict]) -> Dict:
        """直接执行 MCP 工具"""
        tool_name = route_result.target
        parameters = {**route_result.parameters}
        
        if context:
            parameters.update(context)
        
        logger.info(f"调用工具: {tool_name}")
        result = self.mcp_server.call_tool(tool_name, **parameters)  # 解包参数
        return result
    
    async def _execute_workflow(self, route_result, context: Optional[Dict], user_input: str) -> Dict:
        """使用 LangGraph 工作流执行或智能路由到合适的Skill"""
        import re
        from pathlib import Path
        
        # 使用传入的user_input作为指令
        instruction = user_input
        
        logger.info(f"Workflow处理指令: {instruction}")
        
        # 检测是否包含目录路径 - 如果是批量处理任务，路由到BatchSkill
        path_pattern = r'([A-Za-z]:\\(?:[^\s]+)?目录|[A-Za-z]:\\[^\s]+)'
        if re.search(path_pattern, instruction) or "目录" in instruction or "批量" in instruction:
            logger.info("检测到批量处理任务，路由到BatchSkill")
            
            # 使用BatchSkill处理
            from skills.batch_skill import BatchSkill
            batch_skill = BatchSkill()
            result = await batch_skill.execute(instruction, context)
            return result
        
        # 否则使用LangGraph工作流
        from .workflow import run_workflow
        
        logger.info("使用工作流编排执行")
        
        video_paths = []
        if context and "video_paths" in context:
            video_paths = context["video_paths"]
        elif context and "video_path" in context:
            video_paths = [context["video_path"]]
        
        result = await run_workflow(instruction, video_paths)
        return result
    
    async def _execute_skill(self, route_result, context: Optional[Dict]) -> Dict:
        """使用技能策略 - 通过 MCP Server 调用对应工具"""
        # 任务类型到工具名称的映射
        tool_map = {
            "subtitle": "generate_subtitle",
            "clip": "clip_video",
            "format": "convert_format",
            "optimize": "optimize_media",
            "image": "process_image"
        }
        
        tool_name = tool_map.get(route_result.task_type.value)
        if not tool_name:
            raise ValueError(f"未找到对应工具: {route_result.task_type.value}")
        
        logger.info(f"使用技能策略，调用工具: {tool_name}")
        
        # 合并参数
        parameters = {**route_result.parameters}
        if context:
            parameters.update(context)
        
        # 通过 MCP Server 调用工具（解包参数）
        result = self.mcp_server.call_tool(tool_name, **parameters)
        return result
    
    async def _execute_with_analysis(self, user_input: str, context: Optional[Dict]) -> Dict:
        """使用 LLM 分析后执行"""
        # 原有的分析逻辑
        intent = await self._analyze_intent(user_input, context)
        logger.info(f"检测到意图: {intent}")
        
        # 路由到适当的技能或工具
        result = await self._route_and_execute(intent, user_input, context)
        return result
    
    def get_memory_summary(self) -> Dict:
        """获取记忆摘要"""
        return self.memory.summarize_session()
    
    def get_conversation_history(self, n: int = 10) -> List[Dict]:
        """获取对话历史"""
        return self.memory.get_conversation_history(n)
    
    async def _analyze_intent(self, user_input: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user intent using LLM"""
        analysis_prompt = f"""Analyze the following user request and extract:
1. Task type (clip, subtitle, convert, optimize, batch, etc.)
2. Key parameters (file paths, time ranges, formats, etc.)
3. Priority/urgency

User request: {user_input}

Return a JSON object with:
{{
    "task_type": "clip|subtitle|convert|optimize|batch|other",
    "parameters": {{}},
    "confidence": 0.0-1.0
}}"""
        
        messages = [
            {"role": "system", "content": "You are a task analysis assistant. Return only valid JSON."},
            {"role": "user", "content": analysis_prompt}
        ]
        
        try:
            response = await self.llm_manager.generate(messages, task_type="chinese_processing")
            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                intent = json.loads(json_match.group())
                return intent
        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
        
        # Fallback: simple keyword-based intent detection
        intent = self._fallback_intent_detection(user_input)
        return intent
    
    def _fallback_intent_detection(self, user_input: str) -> Dict[str, Any]:
        """Fallback intent detection using keywords"""
        user_lower = user_input.lower()
        
        # Check for batch processing first (highest priority)
        if any(word in user_lower for word in ["批量", "batch", "所有", "多个", "目录", "子目录"]) or \
           ("所有" in user_input and any(word in user_input for word in ["视频", "文件"])):
            return {"task_type": "batch", "parameters": {}, "confidence": 0.9}
        # Check for image-related keywords
        elif any(word in user_lower for word in ["图片", "image", "图像", "照片", "photo", "png", "jpg", "jpeg", "处理图片"]):
            return {"task_type": "image", "parameters": {}, "confidence": 0.8}
        elif any(word in user_lower for word in ["剪辑", "裁剪", "剪", "clip", "cut"]):
            return {"task_type": "clip", "parameters": {}, "confidence": 0.7}
        elif any(word in user_lower for word in ["字幕", "subtitle", "转写"]):
            return {"task_type": "subtitle", "parameters": {}, "confidence": 0.7}
        elif any(word in user_lower for word in ["转换", "convert", "格式"]) and not any(word in user_lower for word in ["图片", "image", "图像"]):
            return {"task_type": "convert", "parameters": {}, "confidence": 0.7}
        elif any(word in user_lower for word in ["优化", "优化", "enhance", "降噪"]):
            return {"task_type": "optimize", "parameters": {}, "confidence": 0.7}
        else:
            return {"task_type": "other", "parameters": {}, "confidence": 0.5}
    
    async def _route_and_execute(
        self,
        intent: Dict[str, Any],
        user_input: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Route task to appropriate skill or tool and execute"""
        task_type = intent.get("task_type")
        
        # Try to use skill first
        skill = self.skill_registry.get_skill(task_type)
        if skill:
            logger.info(f"Using skill: {skill.__class__.__name__}")
            try:
                result = await skill.execute(user_input, context)
                # Return skill result directly, but add metadata
                if isinstance(result, dict):
                    result["_meta"] = {
                        "method": "skill",
                        "skill": skill.__class__.__name__
                    }
                    return result
                else:
                    return {
                        "success": True,
                        "result": result,
                        "method": "skill",
                        "skill": skill.__class__.__name__
                    }
            except Exception as e:
                logger.error(f"Skill execution failed: {e}")
                # Fallback to direct tool call
        
        # Fallback to direct tool call via MCP
        return await self._execute_via_mcp(intent, user_input, context)
    
    async def _execute_via_mcp(
        self,
        intent: Dict[str, Any],
        user_input: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute task via MCP tools"""
        task_type = intent.get("task_type")
        
        # Map task types to MCP tools
        tool_mapping = {
            "clip": "clip_video",
            "subtitle": "generate_subtitle",
            "convert": "convert_format",
            "optimize": "optimize_media",
            "image": "process_image",
            "resize": "process_image",
            "图片": "process_image",
        }
        
        tool_name = tool_mapping.get(task_type)
        if not tool_name:
            return {
                "success": False,
                "error": f"Unknown task type: {task_type}"
            }
        
        # Extract parameters from user input using LLM
        params = await self._extract_parameters(user_input, tool_name)
        
        # Call MCP tool
        result = self.mcp_server.call_tool(tool_name, **params)
        
        return {
            "success": result.get("success", False),
            "result": result,
            "method": "mcp_tool",
            "tool": tool_name
        }
    
    async def _extract_parameters(self, user_input: str, tool_name: str) -> Dict[str, Any]:
        """Extract parameters from user input using LLM"""
        tool_schema = self.mcp_server.get_tool_schema(tool_name)
        if not tool_schema:
            return {}
        
        extraction_prompt = f"""Extract parameters from the user request for tool: {tool_name}

Tool schema: {json.dumps(tool_schema, indent=2, ensure_ascii=False)}
User request: {user_input}

Return a JSON object with the extracted parameters. Only include parameters that are clearly mentioned in the user request."""
        
        messages = [
            {"role": "system", "content": "You are a parameter extraction assistant. Return only valid JSON."},
            {"role": "user", "content": extraction_prompt}
        ]
        
        try:
            response = await self.llm_manager.generate(messages, task_type="chinese_processing")
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                params = json.loads(json_match.group())
                return params
        except Exception as e:
            logger.error(f"Parameter extraction failed: {e}")
        
        return {}
