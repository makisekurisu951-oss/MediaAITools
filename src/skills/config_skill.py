"""Configuration management skill - allows natural language LLM configuration"""

import re
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from .base_skill import BaseSkill
from utils.logger import get_logger


class ConfigSkill(BaseSkill):
    """Skill for managing LLM configuration through natural language"""
    
    def __init__(self):
        super().__init__(
            name="config",
            description="Manage LLM configuration through natural language"
        )
        self.config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        self.logger = get_logger(__name__)
    
    async def execute(self, intent: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Execute configuration management based on natural language intent
        
        Args:
            intent: Natural language description of configuration change
                Examples:
                - "使用本地模型"
                - "切换到OpenAI"
                - "用远程千问API"
                - "使用DeepSeek"
            context: Optional context dictionary
                
        Returns:
            dict with status and message
        """
        self.logger.info(f"ConfigSkill executing with intent: {intent}")
        
        try:
            # Parse intent to determine desired configuration
            config_change = self._parse_intent(intent)
            
            if not config_change:
                return {
                    "success": False,
                    "message": "无法理解配置意图，请说明要使用哪个LLM（本地模型/OpenAI/DeepSeek/千问）"
                }
            
            # Apply configuration change
            self._update_config(config_change)
            
            # Reload LLM manager to apply changes
            self._reload_llm_manager()
            
            return {
                "success": True,
                "message": f"✓ 已切换到 {config_change['provider']} ({'本地' if config_change['is_local'] else '远程'})",
                "provider": config_change["provider"],
                "is_local": config_change["is_local"]
            }
            
        except Exception as e:
            self.logger.error(f"Configuration update failed: {e}")
            return {
                "success": False,
                "message": f"配置更新失败: {str(e)}"
            }
    
    def _parse_intent(self, intent: str) -> Optional[Dict[str, Any]]:
        """Parse natural language intent to configuration change
        
        Returns:
            dict with provider name and whether it's local, or None if can't parse
        """
        intent_lower = intent.lower()
        
        # Check for local model keywords
        if any(keyword in intent_lower for keyword in ["本地", "local", "离线", "本机"]):
            return {
                "provider": "local_qwen",
                "is_local": True,
                "enabled": True
            }
        
        # Check for specific remote providers
        if any(keyword in intent_lower for keyword in ["openai", "gpt"]):
            return {
                "provider": "openai",
                "is_local": False,
                "enabled": True
            }
        
        if any(keyword in intent_lower for keyword in ["deepseek", "深度求索"]):
            return {
                "provider": "deepseek",
                "is_local": False,
                "enabled": True
            }
        
        if any(keyword in intent_lower for keyword in ["qwen", "千问", "通义"]):
            # Check if it's explicitly remote
            if any(keyword in intent_lower for keyword in ["远程", "remote", "api", "在线"]):
                return {
                    "provider": "qwen",
                    "is_local": False,
                    "enabled": True
                }
            # Default to local qwen
            return {
                "provider": "local_qwen",
                "is_local": True,
                "enabled": True
            }
        
        # Check for remote/API keywords (default to best available)
        if any(keyword in intent_lower for keyword in ["远程", "remote", "api", "在线"]):
            # Default to DeepSeek for remote (best value)
            return {
                "provider": "deepseek",
                "is_local": False,
                "enabled": True
            }
        
        return None
    
    def _update_config(self, config_change: Dict[str, Any]):
        """Update configuration file"""
        # Read current config
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        provider = config_change["provider"]
        
        # Disable all providers first
        for p in ["local_qwen", "openai", "deepseek", "qwen"]:
            if p in config["llm"]:
                config["llm"][p]["enabled"] = False
        
        # Enable selected provider
        if provider in config["llm"]:
            config["llm"][provider]["enabled"] = True
        
        # Update default models to use selected provider
        for task in config["llm"]["default_models"]:
            config["llm"]["default_models"][task] = provider
        
        # Write updated config
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        
        self.logger.info(f"Updated config: {provider} enabled")
    
    def _reload_llm_manager(self):
        """Reload LLM manager to apply configuration changes"""
        try:
            from llm.llm_manager import LLMManager
            import llm.llm_manager as llm_module
            
            # Reset global instance
            llm_module._llm_manager = None
            
            # Create new instance with updated config
            new_manager = LLMManager()
            llm_module._llm_manager = new_manager
            
            self.logger.info("LLM Manager reloaded successfully")
        except Exception as e:
            self.logger.warning(f"Could not reload LLM manager: {e}")
    
    def validate(self) -> bool:
        """Validate that configuration file exists and is readable"""
        return self.config_path.exists() and self.config_path.is_file()
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current LLM configuration
        
        Returns:
            dict with current provider settings
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            llm_config = config.get("llm", {})
            enabled_providers = []
            
            for provider, settings in llm_config.items():
                if isinstance(settings, dict) and settings.get("enabled", False):
                    enabled_providers.append(provider)
            
            return {
                "enabled_providers": enabled_providers,
                "default_models": llm_config.get("default_models", {}),
                "available_providers": list(llm_config.keys())
            }
        except Exception as e:
            self.logger.error(f"Failed to get current config: {e}")
            return {}
