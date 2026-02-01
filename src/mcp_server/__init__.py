"""MCP Server for Media Processing Tools"""

from .media_mcp_server import MediaMCPServer
from .tools import MediaTool, ClipTool, SubtitleTool, FormatTool, OptimizeTool

# Lazy import for ImageTool
try:
    from .tools import ImageTool
    __all__ = [
        "MediaMCPServer",
        "MediaTool",
        "ClipTool",
        "SubtitleTool",
        "FormatTool",
        "OptimizeTool",
        "ImageTool",
    ]
except ImportError:
    __all__ = [
        "MediaMCPServer",
        "MediaTool",
        "ClipTool",
        "SubtitleTool",
        "FormatTool",
        "OptimizeTool",
    ]
