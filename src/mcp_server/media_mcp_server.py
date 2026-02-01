"""MCP Server implementation for media processing"""

from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from .tools import ClipTool, ConcatTool, SubtitleTool, FormatTool, OptimizeTool, ImageTool
from config import get_config
from utils.logger import get_logger

logger = get_logger(__name__)


class MediaMCPServer:
    """MCP Server for media processing tools"""
    
    def __init__(self):
        config = get_config()
        media_config = config.get("media", {})
        
        ffmpeg_path = media_config.get("ffmpeg_path", "ffmpeg")
        ffprobe_path = media_config.get("ffprobe_path", "ffprobe")
        
        # Initialize tools
        self.clip_tool = ClipTool(ffmpeg_path, ffprobe_path)
        self.concat_tool = ConcatTool(ffmpeg_path, ffprobe_path)
        self.subtitle_tool = SubtitleTool(ffmpeg_path, ffprobe_path)
        self.format_tool = FormatTool(ffmpeg_path, ffprobe_path)
        self.optimize_tool = OptimizeTool(ffmpeg_path, ffprobe_path)
        
        # Initialize image tool (if available)
        try:
            self.image_tool = ImageTool()
        except ImportError:
            self.image_tool = None
            logger.warning("ImageTool not available (Pillow not installed)")
        
        # Tool registry
        self.tools: Dict[str, Any] = {
            "clip_video": self.clip_tool,
            "concat_videos": self.concat_tool,
            "generate_subtitle": self.subtitle_tool,
            "convert_format": self.format_tool,
            "optimize_media": self.optimize_tool,
        }
        
        # Add image tool if available
        if self.image_tool:
            self.tools["process_image"] = self.image_tool
    
    def list_tools(self) -> List[str]:
        """List available tools"""
        return list(self.tools.keys())
    
    def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Call a tool by name"""
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Tool not found: {tool_name}"
            }
        
        tool = self.tools[tool_name]
        logger.info(f"Calling tool: {tool_name} with args: {kwargs}")
        
        try:
            result = tool.execute(**kwargs)
            logger.info(f"Tool {tool_name} completed: {result.get('success', False)}")
            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get tool schema for MCP protocol"""
        schemas = {
            "clip_video": {
                "name": "clip_video",
                "description": "Clip video segment from start_time to end_time",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input_path": {"type": "string", "description": "Input video path"},
                        "start_time": {"type": "string", "description": "Start time (HH:MM:SS)"},
                        "end_time": {"type": "string", "description": "End time (HH:MM:SS)"},
                        "output_path": {"type": "string", "description": "Output video path"},
                    },
                    "required": ["input_path", "start_time", "end_time", "output_path"]
                }
            },
            "concat_videos": {
                "name": "concat_videos",
                "description": "Concatenate multiple videos",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of video paths to concatenate"
                        },
                        "output_path": {"type": "string", "description": "Output video path"},
                    },
                    "required": ["video_paths", "output_path"]
                }
            },
            "generate_subtitle": {
                "name": "generate_subtitle",
                "description": "Generate subtitle file from video",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_path": {"type": "string", "description": "Input video path"},
                        "language": {"type": "string", "description": "Language code (zh, en, etc.)"},
                        "output_path": {"type": "string", "description": "Output subtitle path"},
                    },
                    "required": ["video_path"]
                }
            },
            "convert_format": {
                "name": "convert_format",
                "description": "Convert video format",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input_path": {"type": "string", "description": "Input video path"},
                        "output_format": {"type": "string", "description": "Output format (mp4, avi, etc.)"},
                        "output_path": {"type": "string", "description": "Output video path"},
                        "resolution": {"type": "string", "description": "Target resolution (1080p, 720p, etc.)"},
                    },
                    "required": ["input_path", "output_path"]
                }
            },
            "optimize_media": {
                "name": "optimize_media",
                "description": "Optimize audio or video quality",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input_path": {"type": "string", "description": "Input media path"},
                        "output_path": {"type": "string", "description": "Output media path"},
                        "optimize_type": {"type": "string", "description": "Type: 'audio' or 'video'"},
                    },
                    "required": ["input_path", "output_path", "optimize_type"]
                }
            },
            "process_image": {
                "name": "process_image",
                "description": "Process image: resize, convert format, adjust aspect ratio",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input_path": {"type": "string", "description": "Input image path"},
                        "output_path": {"type": "string", "description": "Output image path"},
                        "width": {"type": "integer", "description": "Target width in pixels"},
                        "height": {"type": "integer", "description": "Target height in pixels"},
                        "aspect_ratio": {"type": "string", "description": "Aspect ratio (e.g., '9:16' for vertical)"},
                        "resize_mode": {"type": "string", "description": "Resize mode: 'fit', 'crop', or 'stretch'"},
                    },
                    "required": ["input_path", "output_path"]
                }
            }
        }
        
        return schemas.get(tool_name)
