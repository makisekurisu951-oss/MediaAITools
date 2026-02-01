"""Clip Skill for video clipping operations"""

from typing import Dict, Any, Optional
import re
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from .base_skill import BaseSkill
from mcp_server import MediaMCPServer
from utils.media_utils import parse_time
from utils.logger import get_logger

logger = get_logger(__name__)


class ClipSkill(BaseSkill):
    """Skill for video clipping"""
    
    def __init__(self):
        super().__init__(
            "ClipSkill",
            "Handles video clipping operations based on natural language instructions"
        )
        self.mcp_server = MediaMCPServer()
    
    async def execute(self, instruction: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute clipping operation"""
        logger.info(f"ClipSkill executing: {instruction}")
        
        # Extract parameters from instruction
        params = self._extract_clip_parameters(instruction, context)
        
        if not params.get("input_path"):
            return {
                "success": False,
                "error": "Input video path not found in instruction or context"
            }
        
        # Call MCP tool
        result = self.mcp_server.call_tool("clip_video", **params)
        
        return {
            "success": result.get("success", False),
            "result": result,
            "skill": self.name
        }
    
    def _extract_clip_parameters(self, instruction: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract clipping parameters from instruction"""
        params = {}
        
        # Get input path from context or instruction
        if context and "input_path" in context:
            params["input_path"] = context["input_path"]
        else:
            # Try to extract from instruction (simple pattern matching)
            path_match = re.search(r'["\']([^"\']+\.(mp4|avi|mov|mkv))["\']', instruction)
            if path_match:
                params["input_path"] = path_match.group(1)
        
        # Extract time ranges
        # Pattern: HH:MM:SS-HH:MM:SS or 00:01:20-00:02:30
        time_pattern = r'(\d{1,2}:\d{2}:\d{2})\s*[-~到至]\s*(\d{1,2}:\d{2}:\d{2})'
        time_match = re.search(time_pattern, instruction)
        if time_match:
            params["start_time"] = time_match.group(1)
            params["end_time"] = time_match.group(2)
        else:
            # Try other patterns
            start_match = re.search(r'从\s*(\d{1,2}:\d{2}:\d{2})', instruction)
            end_match = re.search(r'到\s*(\d{1,2}:\d{2}:\d{2})', instruction)
            if start_match:
                params["start_time"] = start_match.group(1)
            if end_match:
                params["end_time"] = end_match.group(1)
        
        # Extract output path
        if context and "output_path" in context:
            params["output_path"] = context["output_path"]
        else:
            # Generate default output path
            if params.get("input_path"):
                input_path = params["input_path"]
                base = input_path.rsplit('.', 1)[0]
                params["output_path"] = f"{base}_clipped.mp4"
        
        return params
