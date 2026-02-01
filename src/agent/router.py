"""智能路由器 - 根据任务类型和上下文选择最佳处理路径"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    """任务类型枚举"""
    SUBTITLE = "subtitle"
    CLIP = "clip"
    CONCAT = "concat"
    FORMAT = "format"
    OPTIMIZE = "optimize"
    IMAGE = "image"
    COMPLEX = "complex"  # 需要多步骤处理
    UNKNOWN = "unknown"


class RouteStrategy(Enum):
    """路由策略"""
    DIRECT_TOOL = "direct_tool"  # 直接调用工具
    WORKFLOW = "workflow"  # 使用工作流
    SKILL = "skill"  # 使用技能
    AGENT = "agent"  # 使用智能体


@dataclass
class RouteResult:
    """路由结果"""
    task_type: TaskType
    strategy: RouteStrategy
    target: str  # 目标工具/技能/工作流名称
    confidence: float  # 置信度 0-1
    parameters: Dict
    reasoning: str  # 路由原因


class IntelligentRouter:
    """智能路由器"""
    
    def __init__(self):
        # 关键词映射到任务类型
        self.keyword_patterns = {
            TaskType.SUBTITLE: [
                r'字幕', r'subtitle', r'转录', r'transcribe',
                r'语音识别', r'srt', r'vtt', r'纠错'
            ],
            TaskType.CLIP: [
                r'剪辑', r'clip', r'裁剪', r'cut', r'trim',
                r'截取', r'片段', r'segment'
            ],
            TaskType.CONCAT: [
                r'拼接', r'concat', r'合并', r'merge',
                r'连接', r'join', r'组合'
            ],
            TaskType.FORMAT: [
                r'转换', r'convert', r'格式', r'format',
                r'编码', r'encode', r'mp4', r'avi', r'mkv'
            ],
            TaskType.OPTIMIZE: [
                r'优化', r'optimize', r'压缩', r'compress',
                r'质量', r'quality', r'大小', r'size'
            ],
            TaskType.IMAGE: [
                r'图片', r'image', r'照片', r'photo',
                r'旋转', r'rotate', r'缩放', r'resize'
            ]
        }
        
        # 复杂任务模式（包含多个动作）
        self.complex_patterns = [
            r'先.*然后', r'首先.*接着', r'并且', r'同时',
            r'and.*and', r'then', r'after'
        ]
        
        # 工具能力映射
        self.tool_capabilities = {
            "generate_subtitle": {
                "tasks": [TaskType.SUBTITLE],
                "complexity": "medium",
                "avg_time": 30.0
            },
            "clip_video": {
                "tasks": [TaskType.CLIP],
                "complexity": "low",
                "avg_time": 10.0
            },
            "concat_videos": {
                "tasks": [TaskType.CONCAT],
                "complexity": "low",
                "avg_time": 15.0
            },
            "convert_format": {
                "tasks": [TaskType.FORMAT],
                "complexity": "low",
                "avg_time": 20.0
            },
            "optimize_media": {
                "tasks": [TaskType.OPTIMIZE],
                "complexity": "medium",
                "avg_time": 25.0
            },
            "process_image": {
                "tasks": [TaskType.IMAGE],
                "complexity": "low",
                "avg_time": 5.0
            }
        }
    
    def route(self, instruction: str, context: Dict = None) -> RouteResult:
        """
        智能路由决策
        
        Args:
            instruction: 用户指令
            context: 上下文信息（文件路径、历史等）
            
        Returns:
            RouteResult: 路由结果
        """
        # 1. 识别任务类型
        task_type, confidence = self._identify_task_type(instruction)
        
        # 2. 检测是否为复杂任务
        is_complex = self._is_complex_task(instruction)
        
        # 3. 选择路由策略
        strategy = self._select_strategy(task_type, is_complex, confidence)
        
        # 4. 确定目标
        target = self._determine_target(task_type, strategy)
        
        # 5. 提取参数
        parameters = self._extract_parameters(instruction, task_type, context)
        
        # 6. 生成推理说明
        reasoning = self._generate_reasoning(
            task_type, strategy, confidence, is_complex
        )
        
        return RouteResult(
            task_type=task_type,
            strategy=strategy,
            target=target,
            confidence=confidence,
            parameters=parameters,
            reasoning=reasoning
        )
    
    def _identify_task_type(self, instruction: str) -> Tuple[TaskType, float]:
        """识别任务类型"""
        instruction_lower = instruction.lower()
        scores = {}
        
        for task_type, patterns in self.keyword_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, instruction_lower):
                    score += 1
            if score > 0:
                scores[task_type] = score
        
        if not scores:
            return TaskType.UNKNOWN, 0.0
        
        # 选择得分最高的任务类型
        max_task = max(scores.items(), key=lambda x: x[1])
        confidence = min(max_task[1] / 3.0, 1.0)  # 归一化到 0-1
        
        return max_task[0], confidence
    
    def _is_complex_task(self, instruction: str) -> bool:
        """检测是否为复杂任务"""
        for pattern in self.complex_patterns:
            if re.search(pattern, instruction):
                return True
        
        # 检查是否包含多个任务类型
        task_count = 0
        for patterns in self.keyword_patterns.values():
            if any(re.search(p, instruction.lower()) for p in patterns):
                task_count += 1
        
        return task_count > 1
    
    def _select_strategy(
        self, 
        task_type: TaskType, 
        is_complex: bool,
        confidence: float
    ) -> RouteStrategy:
        """选择路由策略"""
        # 复杂任务使用工作流
        if is_complex:
            return RouteStrategy.WORKFLOW
        
        # 低置信度任务使用智能体
        if confidence < 0.5:
            return RouteStrategy.AGENT
        
        # 未知任务使用智能体
        if task_type == TaskType.UNKNOWN:
            return RouteStrategy.AGENT
        
        # 简单明确的任务直接调用工具
        if confidence >= 0.7:
            return RouteStrategy.DIRECT_TOOL
        
        # 中等置信度使用技能
        return RouteStrategy.SKILL
    
    def _determine_target(
        self, 
        task_type: TaskType,
        strategy: RouteStrategy
    ) -> str:
        """确定目标"""
        if strategy == RouteStrategy.AGENT:
            return "media_agent"
        
        if strategy == RouteStrategy.WORKFLOW:
            return "media_workflow"
        
        # 任务类型到工具/技能的映射
        type_to_target = {
            TaskType.SUBTITLE: "generate_subtitle",
            TaskType.CLIP: "clip_video",
            TaskType.CONCAT: "concat_videos",
            TaskType.FORMAT: "convert_format",
            TaskType.OPTIMIZE: "optimize_media",
            TaskType.IMAGE: "process_image"
        }
        
        return type_to_target.get(task_type, "media_agent")
    
    def _extract_parameters(
        self,
        instruction: str,
        task_type: TaskType,
        context: Dict = None
    ) -> Dict:
        """提取参数"""
        params = {}
        
        if context:
            params.update(context)
        
        # 提取时间范围（用于剪辑）
        time_pattern = r'(\d+):(\d+)(?::(\d+))?'
        times = re.findall(time_pattern, instruction)
        if times and task_type == TaskType.CLIP:
            if len(times) >= 2:
                params["start_time"] = self._format_time(times[0])
                params["end_time"] = self._format_time(times[1])
        
        # 提取语言
        if '中文' in instruction or '中' in instruction:
            params["language"] = "zh"
        elif 'english' in instruction.lower() or '英文' in instruction:
            params["language"] = "en"
        
        # 提取智能纠错需求
        if '智能' in instruction or '纠错' in instruction or 'llm' in instruction.lower():
            params["use_llm_correction"] = True
        
        # 提取格式
        formats = ['mp4', 'avi', 'mkv', 'mov', 'flv', 'webm']
        for fmt in formats:
            if fmt in instruction.lower():
                params["output_format"] = fmt
                break
        
        return params
    
    def _format_time(self, time_tuple: Tuple) -> str:
        """格式化时间"""
        h, m, s = time_tuple[0], time_tuple[1], time_tuple[2] or '0'
        return f"{h.zfill(2)}:{m.zfill(2)}:{s.zfill(2)}"
    
    def _generate_reasoning(
        self,
        task_type: TaskType,
        strategy: RouteStrategy,
        confidence: float,
        is_complex: bool
    ) -> str:
        """生成推理说明"""
        reasons = []
        
        reasons.append(f"识别到任务类型: {task_type.value}")
        reasons.append(f"置信度: {confidence:.2f}")
        
        if is_complex:
            reasons.append("检测到复杂任务（多步骤），使用工作流编排")
        
        strategy_desc = {
            RouteStrategy.DIRECT_TOOL: "任务明确，直接调用工具执行",
            RouteStrategy.WORKFLOW: "复杂任务，使用LangGraph工作流",
            RouteStrategy.SKILL: "使用技能模块处理",
            RouteStrategy.AGENT: "使用智能体进行分析和执行"
        }
        
        reasons.append(f"策略: {strategy_desc[strategy]}")
        
        return " | ".join(reasons)
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict]:
        """获取工具信息"""
        return self.tool_capabilities.get(tool_name)
    
    def recommend_tools(self, task_type: TaskType) -> List[str]:
        """推荐适合的工具"""
        recommended = []
        for tool_name, info in self.tool_capabilities.items():
            if task_type in info["tasks"]:
                recommended.append(tool_name)
        return recommended


# 使用示例
if __name__ == "__main__":
    router = IntelligentRouter()
    
    # 测试用例
    test_cases = [
        "为这个视频添加中文字幕，需要智能纠错",
        "剪辑视频从 00:10 到 00:30",
        "先添加字幕然后转换成mp4格式",
        "优化视频质量并压缩大小",
        "帮我处理一下这个文件"
    ]
    
    for instruction in test_cases:
        print(f"\n指令: {instruction}")
        result = router.route(instruction)
        print(f"  任务类型: {result.task_type.value}")
        print(f"  策略: {result.strategy.value}")
        print(f"  目标: {result.target}")
        print(f"  置信度: {result.confidence:.2f}")
        print(f"  参数: {result.parameters}")
        print(f"  推理: {result.reasoning}")
