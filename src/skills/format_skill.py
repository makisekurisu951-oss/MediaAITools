"""Format Skill for video format conversion"""

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


class FormatSkill(BaseSkill):
    """Skill for video format conversion"""
    
    def __init__(self):
        super().__init__(
            "FormatSkill",
            "Handles video format conversion operations"
        )
        self.mcp_server = MediaMCPServer()
    
    async def execute(self, instruction: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute format conversion"""
        logger.info(f"FormatSkill executing: {instruction}")
        
        params = self._extract_format_parameters(instruction, context)
        
        if not params.get("input_path"):
            return {
                "success": False,
                "error": "Input path not found"
            }
        
        result = self.mcp_server.call_tool("convert_format", **params)
        
        return {
            "success": result.get("success", False),
            "result": result,
            "skill": self.name
        }
    
    def _extract_format_parameters(self, instruction: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract format conversion parameters"""
        params = {}
        
        # Get input path
        if context and "input_path" in context:
            params["input_path"] = context["input_path"]
        else:
            path_match = re.search(r'["\']([^"\']+\.(mp4|avi|mov|mkv|flv))["\']', instruction)
            if path_match:
                params["input_path"] = path_match.group(1)
        
        # Extract target format
        format_match = re.search(r'(mp4|avi|mov|mkv|flv|webm)', instruction, re.IGNORECASE)
        if format_match:
            params["output_format"] = format_match.group(1).lower()
        
        # Extract resolution
        resolution_match = re.search(r'(\d+p|1080p|720p|480p)', instruction, re.IGNORECASE)
        if resolution_match:
            params["resolution"] = resolution_match.group(1).lower()
        
        # Generate output path
        if context and "output_path" in context:
            params["output_path"] = context["output_path"]
        elif params.get("input_path") and params.get("output_format"):
            input_path = params["input_path"]
            base = input_path.rsplit('.', 1)[0]
            params["output_path"] = f"{base}.{params['output_format']}"
        
        return params
