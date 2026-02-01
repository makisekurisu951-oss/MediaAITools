"""LLM Manager for dynamic model scheduling"""

from typing import Dict, Optional, List, Any

# Lazy import to avoid errors when dependencies are not installed
try:
    from config import get_config
    CONFIG_AVAILABLE = True
except Exception:
    CONFIG_AVAILABLE = False
    def get_config():
        return {}

try:
    from .providers import BaseLLMProvider, OpenAIProvider, DeepSeekProvider, QwenProvider, LocalQwenProvider
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    BaseLLMProvider = None
    OpenAIProvider = None
    DeepSeekProvider = None
    QwenProvider = None
    LocalQwenProvider = None


class LLMManager:
    """Manages multiple LLM providers and schedules them based on task type"""
    
    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.default_models: Dict[str, str] = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize LLM providers from config"""
        if not LLM_AVAILABLE:
            # LLM dependencies not installed, skip initialization
            return
        
        try:
            config = get_config()
        except Exception:
            # Config not available, use defaults
            config = {}
        
        llm_config = config.get("llm", {})
        
        # Initialize Local Qwen (Priority - no API key needed)
        if LocalQwenProvider and (local_qwen_config := llm_config.get("local_qwen")):
            if local_qwen_config.get("enabled", True):
                try:
                    model_path = local_qwen_config.get("model_path", "Qwen/Qwen2-1.5B-Instruct")
                    self.providers["local_qwen"] = LocalQwenProvider(
                        model_path=model_path,
                        device=local_qwen_config.get("device", "auto")
                    )
                    # Provider will print its own initialization message
                except Exception as e:
                    print(f"[WARN] Failed to initialize local Qwen: {e}")
        
        # Initialize OpenAI
        if OpenAIProvider and (openai_config := llm_config.get("openai")):
            if openai_config.get("enabled", False):
                api_key = openai_config.get("api_key")
                if api_key and api_key != "your-openai-api-key":
                    try:
                        self.providers["openai"] = OpenAIProvider(
                            api_key=api_key,
                            model=openai_config.get("model", "gpt-4o"),
                            base_url=openai_config.get("base_url"),
                            temperature=0.7
                        )
                    except Exception:
                        pass  # Skip if initialization fails
        
        # Initialize DeepSeek
        if DeepSeekProvider and (deepseek_config := llm_config.get("deepseek")):
            if deepseek_config.get("enabled", False):
                api_key = deepseek_config.get("api_key")
                if api_key and api_key != "your-deepseek-api-key":
                    try:
                        self.providers["deepseek"] = DeepSeekProvider(
                            api_key=api_key,
                            model=deepseek_config.get("model", "deepseek-chat"),
                            base_url=deepseek_config.get("base_url"),
                            temperature=0.7
                        )
                    except Exception:
                        pass
        
        # Initialize Qwen (Remote)
        if QwenProvider and (qwen_config := llm_config.get("qwen")):
            if qwen_config.get("enabled", False):
                api_key = qwen_config.get("api_key")
                if api_key and api_key != "your-qwen-api-key":
                    try:
                        self.providers["qwen"] = QwenProvider(
                            api_key=api_key,
                            model=qwen_config.get("model", "qwen-turbo"),
                            base_url=qwen_config.get("base_url"),
                            temperature=0.7
                        )
                    except Exception:
                        pass
        
        # Set default models for different tasks
        self.default_models = llm_config.get("default_models", {
            "video_understanding": "local_qwen",
            "subtitle_translation": "local_qwen",
            "chinese_processing": "local_qwen"
        })
    
    def get_provider(self, provider_name: Optional[str] = None, task_type: Optional[str] = None) -> Optional[BaseLLMProvider]:
        """Get LLM provider by name or task type"""
        if provider_name:
            return self.providers.get(provider_name)
        
        if task_type:
            provider_name = self.default_models.get(task_type)
            if provider_name:
                return self.providers.get(provider_name)
        
        # Return first available provider as fallback
        if self.providers:
            return next(iter(self.providers.values()))
        
        return None
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        provider_name: Optional[str] = None,
        task_type: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate response using appropriate provider"""
        provider = self.get_provider(provider_name, task_type)
        if not provider:
            raise ValueError("No LLM provider available")
        
        return await provider.generate(messages, **kwargs)
    
    def list_providers(self) -> List[str]:
        """List available provider names"""
        return list(self.providers.keys())
    
    def reload(self):
        """Reload providers from config"""
        self.providers.clear()
        self.default_models.clear()
        self._initialize_providers()


# Global instance
_llm_manager: Optional[LLMManager] = None


def get_llm_manager() -> LLMManager:
    """Get global LLM manager instance"""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager
