"""Media file utilities"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import subprocess
import json


def validate_video_file(file_path: str) -> bool:
    """Validate if file is a valid video file"""
    if not os.path.exists(file_path):
        return False
    
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}
    ext = Path(file_path).suffix.lower()
    return ext in valid_extensions


def get_video_info(video_path: str, ffprobe_path: str = "ffprobe") -> Dict:
    """Get video information using ffprobe"""
    try:
        cmd = [
            ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        
        # Extract key information
        video_stream = next(
            (s for s in info.get("streams", []) if s.get("codec_type") == "video"),
            None
        )
        audio_stream = next(
            (s for s in info.get("streams", []) if s.get("codec_type") == "audio"),
            None
        )
        
        format_info = info.get("format", {})
        
        return {
            "duration": float(format_info.get("duration", 0)),
            "size": int(format_info.get("size", 0)),
            "bitrate": int(format_info.get("bit_rate", 0)),
            "format": format_info.get("format_name", ""),
            "video": {
                "codec": video_stream.get("codec_name", "") if video_stream else None,
                "width": video_stream.get("width", 0) if video_stream else 0,
                "height": video_stream.get("height", 0) if video_stream else 0,
                "fps": eval(video_stream.get("r_frame_rate", "0/1")) if video_stream else 0,
            } if video_stream else None,
            "audio": {
                "codec": audio_stream.get("codec_name", "") if audio_stream else None,
                "sample_rate": int(audio_stream.get("sample_rate", 0)) if audio_stream else 0,
                "channels": audio_stream.get("channels", 0) if audio_stream else 0,
            } if audio_stream else None,
        }
    except Exception as e:
        return {"error": str(e)}


def format_time(seconds: float) -> str:
    """Format seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_time(time_str: str) -> float:
    """Parse time string (HH:MM:SS or MM:SS) to seconds"""
    parts = time_str.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = map(float, parts)
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        minutes, seconds = map(float, parts)
        return minutes * 60 + seconds
    else:
        return float(time_str)


def ensure_output_dir(output_path: str) -> Path:
    """Ensure output directory exists"""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return Path(output_path)
