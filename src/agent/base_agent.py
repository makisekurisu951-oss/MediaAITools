"""Base Agent class"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Lazy import for langchain
try:
    from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Create dummy classes for testing
    class BaseMessage:
        def __init__(self, content=""):
            self.content = content
    class HumanMessage(BaseMessage):
        pass
    class SystemMessage(BaseMessage):
        pass

from llm import get_llm_manager
from utils.logger import get_logger

logger = get_logger(__name__)


class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, name: str, system_prompt: str = ""):
        self.name = name
        self.system_prompt = system_prompt
        self.llm_manager = get_llm_manager()
        self.conversation_history: List[BaseMessage] = []
        
        if system_prompt:
            self.conversation_history.append(SystemMessage(content=system_prompt))
    
    @abstractmethod
    async def process(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process user input and return result"""
        pass
    
    async def _generate_response(
        self,
        messages: List[BaseMessage],
        task_type: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate LLM response"""
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                formatted_messages.append({"role": "system", "content": msg.content})
            else:
                formatted_messages.append({"role": "assistant", "content": msg.content})
        
        response = await self.llm_manager.generate(
            formatted_messages,
            task_type=task_type,
            **kwargs
        )
        
        return response
    
    def add_to_history(self, message: BaseMessage):
        """Add message to conversation history"""
        self.conversation_history.append(message)
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        if self.system_prompt:
            self.conversation_history.append(SystemMessage(content=self.system_prompt))
