"""Configuration module for MediaAITools"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any

_config: Dict[str, Any] = None


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    global _config
    
    if _config is not None:
        return _config
    
    if config_path is None:
        # Default to config.yaml in config directory
        config_dir = Path(__file__).parent
        config_path = config_dir / "config.yaml"
        
        # If config.yaml doesn't exist, try config.example.yaml
        if not config_path.exists():
            example_path = config_dir / "config.example.yaml"
            if example_path.exists():
                print(f"Warning: config.yaml not found, using {example_path}")
                config_path = example_path
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        _config = yaml.safe_load(f)
    
    # Override with environment variables if present
    _override_with_env(_config)
    
    return _config


def _override_with_env(config: Dict[str, Any]):
    """Override config values with environment variables"""
    # LLM API keys
    if os.getenv("OPENAI_API_KEY"):
        config.setdefault("llm", {}).setdefault("openai", {})["api_key"] = os.getenv("OPENAI_API_KEY")
    if os.getenv("DEEPSEEK_API_KEY"):
        config.setdefault("llm", {}).setdefault("deepseek", {})["api_key"] = os.getenv("DEEPSEEK_API_KEY")
    if os.getenv("QWEN_API_KEY"):
        config.setdefault("llm", {}).setdefault("qwen", {})["api_key"] = os.getenv("QWEN_API_KEY")


def get_config() -> Dict[str, Any]:
    """Get current configuration"""
    if _config is None:
        return load_config()
    return _config


def save_config(config: Dict[str, Any], config_path: str = None) -> None:
    """Save configuration to YAML file"""
    global _config
    
    if config_path is None:
        config_dir = Path(__file__).parent
        config_path = config_dir / "config.yaml"
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    # Update cached config
    _config = config


def reload_config(config_path: str = None) -> Dict[str, Any]:
    """Reload configuration from file"""
    global _config
    _config = None
    return load_config(config_path)


def update_llm_provider(provider: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Update LLM provider settings"""
    if provider not in ['deepseek', 'local_qwen', 'openai', 'qwen']:
        raise ValueError(f"Unsupported provider: {provider}")
    
    # Disable all providers first
    for p in ['deepseek', 'local_qwen', 'openai', 'qwen']:
        if p in config.get('llm', {}):
            config['llm'][p]['enabled'] = False
    
    # Enable selected provider
    if provider in config.get('llm', {}):
        config['llm'][provider]['enabled'] = True
        
        # Update default models to use this provider
        if provider == 'local_qwen':
            model_name = 'local_qwen'
        else:
            model_name = provider
            
        config['llm']['default_models'] = {
            'chinese_processing': model_name,
            'subtitle_translation': model_name,
            'video_understanding': model_name
        }
    
    return config
