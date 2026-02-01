"""Image Skill for image processing operations"""

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


class ImageSkill(BaseSkill):
    """Skill for image processing"""
    
    def __init__(self):
        super().__init__(
            "ImageSkill",
            "Handles image processing operations like resizing, format conversion, aspect ratio adjustment"
        )
        self.mcp_server = MediaMCPServer()
    
    async def execute(self, instruction: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute image processing operation"""
        logger.info(f"ImageSkill executing: {instruction}")
        
        params = self._extract_image_parameters(instruction, context)
        
        if not params.get("input_path"):
            return {
                "success": False,
                "error": "Input image path not found in instruction or context"
            }
        
        # Call MCP tool
        result = self.mcp_server.call_tool("process_image", **params)
        
        return {
            "success": result.get("success", False),
            "result": result,
            "skill": self.name
        }
    
    def _extract_image_parameters(self, instruction: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract image processing parameters from instruction"""
        params = {}
        
        # Get input path from context or instruction
        if context and "input_path" in context:
            params["input_path"] = context["input_path"]
        else:
            # Try to extract from instruction - look for file paths
            # Pattern 1: quoted paths
            path_match = re.search(r'["\']([^"\']+\.(png|jpg|jpeg|gif|bmp|webp))["\']', instruction, re.IGNORECASE)
            if path_match:
                params["input_path"] = path_match.group(1)
            else:
                # Pattern 2: Windows absolute paths (D:\path\file.png)
                # Match drive letter, colon, backslash, then path components
                path_match = re.search(r'\b([A-Z]:[\\/][^\s<>"\'|]+\.(png|jpg|jpeg|gif|bmp|webp))\b', instruction, re.IGNORECASE)
                if path_match:
                    params["input_path"] = path_match.group(1)
                else:
                    # Pattern 3: Relative paths or paths without drive letter
                    path_match = re.search(r'\b([^\s<>"\'|]+\.(png|jpg|jpeg|gif|bmp|webp))\b', instruction, re.IGNORECASE)
                    if path_match:
                        params["input_path"] = path_match.group(1)
        
        # Extract output path
        if context and "output_path" in context:
            params["output_path"] = context["output_path"]
        else:
            # Try to extract from instruction
            # Pattern 1: Chinese keywords with quotes
            output_match = re.search(r'保存为["\']([^"\']+)["\']|输出为["\']([^"\']+)["\']|存为["\']([^"\']+)["\']', instruction)
            if output_match:
                output_path = output_match.group(1) or output_match.group(2) or output_match.group(3)
                params["output_path"] = output_path.replace('/', '\\')
            else:
                # Pattern 2: Windows paths after keywords - look for path after keyword
                output_match = re.search(r'(保存为|输出为|存为)\s+([A-Z]:[\\/][^\s<>"\'|]+\.(png|jpg|jpeg|gif|bmp|webp))', instruction, re.IGNORECASE)
                if output_match:
                    params["output_path"] = output_match.group(2)
                else:
                    # Pattern 3: Any path-like string after keywords (more flexible)
                    output_match = re.search(r'(保存为|输出为|存为)\s+([^\s<>"\'|]+\.(png|jpg|jpeg|gif|bmp|webp))', instruction, re.IGNORECASE)
                    if output_match:
                        params["output_path"] = output_match.group(2)
                    elif params.get("input_path"):
                        # Generate default output path
                        input_path = params["input_path"]
                        base = Path(input_path).stem
                        ext = Path(input_path).suffix
                        params["output_path"] = str(Path(input_path).parent / f"{base}_processed{ext}")
        
        # Detect vertical/portrait mode
        instruction_lower = instruction.lower()
        if any(word in instruction_lower for word in ["竖屏", "竖", "portrait", "vertical", "手机竖屏"]):
            params["aspect_ratio"] = "9:16"
            # Prefer cover to avoid top/bottom white bars
            params["resize_mode"] = "cover"
        
        # Extract dimensions if specified
        dim_match = re.search(r'(\d+)\s*[xX×]\s*(\d+)', instruction)
        if dim_match:
            params["width"] = int(dim_match.group(1))
            params["height"] = int(dim_match.group(2))
        
        # Extract resize mode
        if "裁剪" in instruction or "crop" in instruction_lower:
            # Map to cover for our implementation
            params["resize_mode"] = "cover"
        elif "拉伸" in instruction or "stretch" in instruction_lower:
            params["resize_mode"] = "stretch"
        else:
            # Default to cover for nicer portrait results; callers can explicitly request "fit"
            params["resize_mode"] = params.get("resize_mode", "cover")
        
        return params
