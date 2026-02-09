"""Batch Processing Skill"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import re
import sys
import os
from pathlib import Path as PathLib

# Add parent directory to path for imports
parent_dir = PathLib(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from .base_skill import BaseSkill
from mcp_server import MediaMCPServer
from utils.logger import get_logger

logger = get_logger(__name__)


class BatchSkill(BaseSkill):
    """Skill for batch processing media files"""
    
    def __init__(self):
        super().__init__(
            name="batch",
            description="Batch process multiple media files"
        )
        self.mcp_server = MediaMCPServer()
    
    async def execute(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute batch processing task"""
        logger.info(f"BatchSkill processing: {user_input}")
        
        # Extract parameters from user input
        params = await self._extract_batch_parameters(user_input)
        
        if not params.get("directory"):
            return {
                "success": False,
                "error": "No directory specified for batch processing"
            }
        
        target_path = params["directory"]
        
        # 智能判断：是文件还是目录
        if os.path.isfile(target_path):
            # 单个文件直接处理
            logger.info(f"Detected single file: {target_path}")
            video_files = [Path(target_path)]
        elif os.path.isdir(target_path):
            # 目录则批量处理
            logger.info(f"Detected directory: {target_path}")
            video_files = self._get_video_files(
                target_path,
                recursive=params.get("recursive", True)
            )
        else:
            return {
                "success": False,
                "error": f"Path not found or invalid: {target_path}"
            }
        
        if not video_files:
            return {
                "success": False,
                "error": f"No video files found in {target_path}"
            }
        
        logger.info(f"Found {len(video_files)} video files to process")
        
        # Determine operation type
        operation = params.get("operation", "subtitle")  # Default to subtitle
        
        # Process each file
        results = []
        success_count = 0
        failed_count = 0
        
        for video_file in video_files:
            logger.info(f"Processing: {video_file}")
            try:
                result = self._process_single_file(
                    video_file,
                    operation,
                    params
                )
                results.append(result)
                if result.get("success"):
                    success_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                logger.error(f"Failed to process {video_file}: {e}")
                results.append({
                    "success": False,
                    "file": str(video_file),
                    "error": str(e)
                })
                failed_count += 1
        
        # 聚合评估所需的信息
        aggregated_result = {
            "success": True,
            "total": len(video_files),
            "success_count": success_count,
            "failed_count": failed_count,
            "results": results
        }
        
        # 从results中提取评估所需的字段
        if success_count > 0 and results:
            # 找到第一个成功的结果作为代表
            first_success = next((r for r in results if r.get("success")), None)
            if first_success:
                # 传递评估器需要的字段
                if "srt_path" in first_success:
                    aggregated_result["srt_path"] = first_success["srt_path"]
                if "output_path" in first_success:
                    aggregated_result["output_path"] = first_success["output_path"]
                if "used_llm_correction" in first_success:
                    aggregated_result["used_llm_correction"] = first_success["used_llm_correction"]
                if "correction_count" in first_success:
                    # 聚合所有成功任务的纠正数量
                    total_corrections = sum(
                        r.get("correction_count", 0) 
                        for r in results 
                        if r.get("success")
                    )
                    aggregated_result["correction_count"] = total_corrections
        
        return aggregated_result
    
    async def _extract_batch_parameters(self, user_input: str) -> Dict[str, Any]:
        """Use LLM to extract parameters from user input"""
        params = {}
        
        # 尝试使用 LLM 智能解析
        try:
            from llm import get_llm_manager
            llm_manager = get_llm_manager()
            
            # ⚠️ 不要调用 reload()！Provider 会自动清理旧的客户端
            # llm_manager.reload() 会导致事件循环错误
            
            provider = llm_manager.get_provider(task_type="chinese_processing")
            if not provider:
                provider = llm_manager.get_provider()
            
            if provider:
                prompt = f"""你是一个参数提取助手。请仔细分析用户指令，提取其中的视频处理参数。

【用户指令】
{user_input}

【任务】
从上述指令中提取以下参数，以JSON格式返回：
- directory: 完整的目录或文件路径（必须从指令中提取，保持原始格式）
- recursive: 是否包含子目录（只有明确提到"子目录"或"recursive"才为true，否则一律false）
- bilingual: 是否生成双语字幕（只有明确提到"双语"、"bilingual"、"中英"才为true，否则一律false）
- language: 字幕语言（"中文/中文/汉语" → "zh"，"英文/英语/english" → "en"，未提及时不返回此字段）
- operation: 操作类型，严格按以下规则判断：
  * 如果提到"字幕"、"subtitle"、"转录"、"添加字幕" → subtitle
  * 如果提到"剪辑"、"clip"、"裁剪"、"截取" → clip
  * 如果提到"转换格式"、"convert"、"格式转换" → convert
  * 如果提到"优化"、"optimize"、"压缩" → optimize
  * 默认：subtitle

【重要提示】
1. directory必须从用户指令中原样提取（可以是文件或目录），不要使用任何示例路径
2. 路径分隔符保持用户输入的格式（\\ 或 /）
3. operation只能是subtitle、clip、convert、optimize之一，不要包含中文
4. 只返回JSON对象，不要添加任何解释

【输出格式】
{{
  "directory": "提取的实际路径",
  "recursive": true或false,
  "bilingual": true或false,
  "language": "zh或en（仅当明确提到时）",
  "operation": "操作类型"
}}"""

                # 使用异步调用 LLM
                messages = [{"role": "user", "content": prompt}]
                response = await provider.generate(messages)
                
                # 解析JSON
                import json
                
                # 清理响应（去除代码块标记）
                cleaned_response = response.strip()
                if cleaned_response.startswith('```'):
                    lines = cleaned_response.split('\n')
                    if lines[0].startswith('```'):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == '```':
                        lines = lines[:-1]
                    cleaned_response = '\n'.join(lines).strip()
                
                # 提取JSON对象
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned_response, re.DOTALL)
                if json_match:
                    params = json.loads(json_match.group())
                    logger.info(f"✅ LLM 解析参数成功: {params}")
                    
                    # 验证目录路径
                    if "directory" in params:
                        directory = params["directory"]
                        if not os.path.exists(directory):
                            logger.warning(f"⚠️ LLM提取的路径不存在: {directory}")
                            # 尝试修正路径（去除多余的转义）
                            directory = directory.replace('\\\\', '\\')
                            if os.path.exists(directory):
                                params["directory"] = directory
                                logger.info(f"✅ 路径修正成功: {directory}")
                    
                    return params
                else:
                    logger.warning(f"⚠️ LLM响应中未找到有效JSON: {response[:200]}")
        
        except Exception as e:
            logger.warning(f"⚠️ LLM解析失败，降级到正则提取: {str(e)}")
        
        # 降级方案：使用正则表达式
        logger.info("使用正则表达式提取参数（LLM降级）")
        
        # Extract directory path - 在常见关键词处停止
        stop_words = r'(?:目录|文件|下面|下的|里面|里的|中的|添加|生成|转换|处理)'
        path_match = re.search(rf'([A-Za-z]:[\\\/][^"<>|*?\n]+?)(?:{stop_words}|$)', user_input)
        
        if path_match:
            directory = path_match.group(1).strip().rstrip('\\/').strip()
            if os.path.exists(directory):
                params["directory"] = directory
                logger.info(f"Extracted directory: {params['directory']}")
        
        # Check if recursive (子目录)
        params["recursive"] = "子目录" in user_input or "recursive" in user_input.lower()
        
        # Check if bilingual (双语) - 只在明确提到"双语"或"中英"时为 true
        params["bilingual"] = any(word in user_input for word in ["双语", "bilingual", "中英"])
        
        # Detect language (语言) - 只在明确提到时设置
        user_input_lower = user_input.lower()
        if any(word in user_input_lower for word in ["中文", "中文", "汉语"]):
            params["language"] = "zh"
        elif any(word in user_input_lower for word in ["英文", "英语", "english"]):
            params["language"] = "en"
        # 否则不设置 language，使用 SubtitleTool 的默认值（en）
        
        # Detect operation type
        if "字幕" in user_input or "subtitle" in user_input.lower():
            params["operation"] = "subtitle"
        elif "剪辑" in user_input or "clip" in user_input.lower():
            params["operation"] = "clip"
        elif "转换" in user_input or "convert" in user_input.lower():
            params["operation"] = "convert"
        elif "优化" in user_input or "optimize" in user_input.lower():
            params["operation"] = "optimize"
        else:
            params["operation"] = "subtitle"  # Default
        
        # Extract output naming pattern
        # Look for patterns like "-subtitle.*" or "-processed.*"
        suffix_pattern = r'[-_](\w+)\.\*'
        suffix_match = re.search(suffix_pattern, user_input)
        if suffix_match:
            params["output_suffix"] = suffix_match.group(1)
        else:
            # Default based on operation
            params["output_suffix"] = params["operation"]
        
        logger.info(f"Extracted batch parameters: {params}")
        return params
    
    def _get_video_files(self, directory: str, recursive: bool = True) -> List[Path]:
        """Get all video files from directory"""
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.error(f"Directory not found: {directory}")
            return []
        
        video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']
        video_files = []
        
        if recursive:
            for ext in video_extensions:
                video_files.extend(dir_path.rglob(f'*{ext}'))
        else:
            for ext in video_extensions:
                video_files.extend(dir_path.glob(f'*{ext}'))
        
        return sorted(video_files)
    
    def _process_single_file(
        self,
        video_file: Path,
        operation: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a single video file"""
        # Generate output filename
        output_suffix = params.get("output_suffix", operation)
        output_file = video_file.parent / f"{video_file.stem}-{output_suffix}{video_file.suffix}"
        
        # Map operation to MCP tool
        tool_mapping = {
            "subtitle": "generate_subtitle",
            "clip": "clip_video",
            "convert": "convert_format",
            "optimize": "optimize_media"
        }
        
        tool_name = tool_mapping.get(operation)
        if not tool_name:
            return {
                "success": False,
                "file": str(video_file),
                "error": f"Unknown operation: {operation}"
            }
        
        # Prepare tool parameters
        if operation == "subtitle":
            # SubtitleTool uses video_path and output_path
            tool_params = {
                "video_path": str(video_file),
                "output_path": str(output_file),
                "embed_subtitle": True,  # 将字幕流嵌入mp4，原地替换源文件
                "use_llm_correction": True,  # 启用 LLM 纠错（通用性更好）
                "bilingual": params.get("bilingual", False)
            }
            # 只有明确指定语言时才传递，否则使用 SubtitleTool 的默认值（en）
            if "language" in params:
                tool_params["language"] = params["language"]
        elif operation == "convert":
            # FormatTool uses input_path, output_format, output_path
            output_format = params.get("output_format", "mp4")  # 默认转mp4
            tool_params = {
                "input_path": str(video_file),
                "output_format": output_format,
                "output_path": str(output_file),
                "resolution": params.get("resolution")
            }
        elif operation == "clip":
            # ClipTool uses input_path, start_time, end_time, output_path
            tool_params = {
                "input_path": str(video_file),
                "start_time": params.get("start_time", "00:00:00"),
                "end_time": params.get("end_time", "00:00:10"),
                "output_path": str(output_file)
            }
        else:
            # OptimizeTool and others use input_path and output_path
            tool_params = {
                "input_path": str(video_file),
                "output_path": str(output_file)
            }
        
        # Call MCP tool
        logger.info(f"Calling {tool_name} for {video_file.name}")
        result = self.mcp_server.call_tool(tool_name, **tool_params)
        
        # 构建返回结果，包含评估所需字段
        processed_result = {
            "success": result.get("success", False),
            "file": str(video_file),
            "output": str(output_file),
            "message": result.get("message", ""),
            "error": result.get("error")
        }
        
        # 透传SubtitleTool的关键字段供Evaluator使用
        if operation == "subtitle" and result.get("success"):
            if "srt_path" in result:
                processed_result["srt_path"] = result["srt_path"]
            if "output_path" in result:
                processed_result["output_path"] = result["output_path"]
            # LLM纠错相关
            processed_result["used_llm_correction"] = tool_params.get("use_llm_correction", False)
            if "correction_count" in result:
                processed_result["correction_count"] = result["correction_count"]
        
        return processed_result
    
    def _validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate extracted parameters"""
        if not params.get("directory"):
            return False
        
        directory = Path(params["directory"])
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory}")
            return False
        
        return True
