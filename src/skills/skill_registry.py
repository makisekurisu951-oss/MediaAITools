"""Skill Registry for managing agent skills"""

from typing import Dict, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from .base_skill import BaseSkill
from .clip_skill import ClipSkill
from .subtitle_skill import SubtitleSkill
from .format_skill import FormatSkill
from .optimize_skill import OptimizeSkill
from .config_skill import ConfigSkill
from utils.logger import get_logger

# Lazy import for ImageSkill
try:
    from .image_skill import ImageSkill
    IMAGE_SKILL_AVAILABLE = True
except ImportError:
    IMAGE_SKILL_AVAILABLE = False
    ImageSkill = None

# Import BatchSkill
from .batch_skill import BatchSkill

logger = get_logger(__name__)


class SkillRegistry:
    """Registry for managing and retrieving agent skills"""
    
    def __init__(self):
        self.skills: Dict[str, BaseSkill] = {}
        self._register_default_skills()
    
    def _register_default_skills(self):
        """Register default skills"""
        self.register_skill("clip", ClipSkill())
        self.register_skill("subtitle", SubtitleSkill())
        self.register_skill("convert", FormatSkill())
        self.register_skill("optimize", OptimizeSkill())
        self.register_skill("batch", BatchSkill())
        self.register_skill("config", ConfigSkill())
        
        # Also register by alternative names
        self.register_skill("clipping", ClipSkill())
        self.register_skill("format", FormatSkill())
        self.register_skill("format_conversion", FormatSkill())
        self.register_skill("configuration", ConfigSkill())
        self.register_skill("settings", ConfigSkill())
        self.register_skill("批量", BatchSkill())
        
        # Register image skill if available
        if IMAGE_SKILL_AVAILABLE and ImageSkill:
            self.register_skill("image", ImageSkill())
            self.register_skill("resize", ImageSkill())
            self.register_skill("图片", ImageSkill())
    
    def register_skill(self, name: str, skill: BaseSkill):
        """Register a skill"""
        self.skills[name.lower()] = skill
        logger.info(f"Registered skill: {name}")
    
    def get_skill(self, name: str) -> Optional[BaseSkill]:
        """Get skill by name"""
        return self.skills.get(name.lower())
    
    def list_skills(self) -> list:
        """List all registered skill names"""
        return list(self.skills.keys())
    
    def unregister_skill(self, name: str):
        """Unregister a skill"""
        if name.lower() in self.skills:
            del self.skills[name.lower()]
            logger.info(f"Unregistered skill: {name}")
