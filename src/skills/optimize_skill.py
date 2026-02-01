"""Optimize Skill for audio/video optimization"""

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


class OptimizeSkill(BaseSkill):
    """Skill for audio/video optimization"""
    
    def __init__(self):
        super().__init__(
            "OptimizeSkill",
            "Handles audio and video quality optimization operations"
        )
        self.mcp_server = MediaMCPServer()
    
    async def execute(self, instruction: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute optimization"""
        logger.info(f"OptimizeSkill executing: {instruction}")
        
        params = self._extract_optimize_parameters(instruction, context)
        
        if not params.get("input_path"):
            return {
                "success": False,
                "error": "Input path not found"
            }
        
        result = self.mcp_server.call_tool("optimize_media", **params)
        
        return {
            "success": result.get("success", False),
            "result": result,
            "skill": self.name
        }
    
    def _extract_optimize_parameters(self, instruction: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract optimization parameters"""
        params = {}
        
        # Get input path
        if context and "input_path" in context:
            params["input_path"] = context["input_path"]
        else:
            path_match = re.search(r'["\']([^"\']+\.(mp4|avi|mov|mkv))["\']', instruction)
            if path_match:
                params["input_path"] = path_match.group(1)
        
        # Determine optimization type
        instruction_lower = instruction.lower()
        if any(word in instruction_lower for word in ["音频", "声音", "降噪", "audio", "sound", "noise"]):
            params["optimize_type"] = "audio"
        elif any(word in instruction_lower for word in ["视频", "画质", "video", "quality", "enhance"]):
            params["optimize_type"] = "video"
        else:
            params["optimize_type"] = "audio"  # Default
        
        # Generate output path
        if context and "output_path" in context:
            params["output_path"] = context["output_path"]
        elif params.get("input_path"):
            input_path = params["input_path"]
            base = input_path.rsplit('.', 1)[0]
            opt_type = params.get("optimize_type", "optimized")
            params["output_path"] = f"{base}_{opt_type}.mp4"
        
        return params
