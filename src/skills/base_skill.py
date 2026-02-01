"""Base Skill class"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from llm import get_llm_manager
from utils.logger import get_logger

logger = get_logger(__name__)


class BaseSkill(ABC):
    """Base class for all agent skills"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.llm_manager = get_llm_manager()
    
    @abstractmethod
    async def execute(self, instruction: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the skill based on instruction"""
        pass
    
    def validate(self, context: Optional[Dict[str, Any]] = None) -> bool:
        """Validate if skill can be executed with given context"""
        return True
