"""Subtitle Skill for subtitle generation"""

from typing import Dict, Any, Optional
import re
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from .base_skill import BaseSkill
from mcp_server import MediaMCPServer
from utils.logger import get_logger

logger = get_logger(__name__)


class SubtitleSkill(BaseSkill):
    """Skill for subtitle generation and processing"""
    
    def __init__(self):
        super().__init__(
            "SubtitleSkill",
            "Handles subtitle generation, translation, and editing operations"
        )
        self.mcp_server = MediaMCPServer()
    
    async def execute(self, instruction: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute subtitle operation"""
        logger.info(f"SubtitleSkill executing: {instruction}")
        
        # Extract parameters
        params = self._extract_subtitle_parameters(instruction, context)
        
        if not params.get("video_path"):
            return {
                "success": False,
                "error": "Video path not found in instruction or context"
            }
        
        # Call MCP tool
        result = self.mcp_server.call_tool("generate_subtitle", **params)
        
        return {
            "success": result.get("success", False),
            "result": result,
            "skill": self.name
        }
    
    def _extract_subtitle_parameters(self, instruction: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract subtitle parameters from instruction"""
        params = {}
        
        # Get video path
        if context and "video_path" in context:
            params["video_path"] = context["video_path"]
        else:
            # 支持带引号和不带引号的路径
            # 方式1: 带引号的路径
            path_match = re.search(r'["\']([^"\']+\.(mp4|avi|mov|mkv|wmv|flv|webm))["\']', instruction, re.IGNORECASE)
            if path_match:
                params["video_path"] = path_match.group(1)
            else:
                # 方式2: 不带引号的路径（盘符开头）
                path_match = re.search(r'([A-Za-z]:[\\//][^<>|*?\n]+\.(mp4|avi|mov|mkv|wmv|flv|webm))', instruction, re.IGNORECASE)
                if path_match:
                    # 提取路径，并去除末尾的标点符号和中文字符
                    raw_path = path_match.group(1)
                    # 去除路径末尾的"文件"、"."等词和标点
                    params["video_path"] = re.sub(r'[。，、\s文件]+$', '', raw_path).strip()
        
        # Extract language
        language_map = {
            "中文": "zh", "汉语": "zh",
            "英文": "en", "英语": "en", "english": "en",
            "日文": "ja", "日语": "ja", "japanese": "ja",
        }
        
        instruction_lower = instruction.lower()
        for key, lang_code in language_map.items():
            if key in instruction_lower:
                params["language"] = lang_code
                break
        
        # Default to English if not specified
        if "language" not in params:
            params["language"] = "en"  # Default to English
        
        # 启用 LLM 智能纠错（针对语法、术语、上下文）
        params["use_llm_correction"] = True
        
        # Extract output path
        if context and "output_path" in context:
            params["output_path"] = context["output_path"]
        elif params.get("video_path"):
            video_path = params["video_path"]
            base = video_path.rsplit('.', 1)[0]
            params["output_path"] = f"{base}.srt"
        
        return params
