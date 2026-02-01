"""配置管理 API 路由"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import time
from pathlib import Path
import sys

# 添加 src 到路径
project_root = Path(__file__).parent.parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from config import get_config, save_config, reload_config, update_llm_provider
from llm.llm_manager import get_llm_manager

router = APIRouter()


class LLMConfigUpdate(BaseModel):
    """LLM 配置更新模型"""
    provider: str  # deepseek, local_qwen, openai, qwen
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    model_path: Optional[str] = None
    device: Optional[str] = None


class TestConnectionRequest(BaseModel):
    """测试连接请求"""
    provider: str


@router.get("/api/config/llm")
async def get_llm_config():
    """获取 LLM 配置"""
    try:
        config = get_config()
        llm_config = config.get('llm', {})
        
        return {
            "success": True,
            "providers": {
                "deepseek": llm_config.get("deepseek", {}),
                "local_qwen": llm_config.get("local_qwen", {}),
                "openai": llm_config.get("openai", {}),
                "qwen": llm_config.get("qwen", {})
            },
            "default_models": llm_config.get("default_models", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/config/llm")
async def update_llm_config(config_update: LLMConfigUpdate):
    """更新 LLM 配置"""
    try:
        config = get_config()
        provider = config_update.provider
        
        # 验证提供商
        if provider not in ['deepseek', 'local_qwen', 'openai', 'qwen']:
            raise HTTPException(status_code=400, detail=f"不支持的提供商: {provider}")
        
        # 更新配置
        config = update_llm_provider(provider, config)
        
        # 更新提供商特定配置
        if provider == 'deepseek':
            if config_update.api_key:
                config['llm']['deepseek']['api_key'] = config_update.api_key
            if config_update.base_url:
                config['llm']['deepseek']['base_url'] = config_update.base_url
            if config_update.model:
                config['llm']['deepseek']['model'] = config_update.model
                
        elif provider == 'local_qwen':
            if config_update.model_path:
                config['llm']['local_qwen']['model_path'] = config_update.model_path
            if config_update.device:
                config['llm']['local_qwen']['device'] = config_update.device
                
        elif provider == 'openai':
            if config_update.api_key:
                config['llm']['openai']['api_key'] = config_update.api_key
            if config_update.base_url:
                config['llm']['openai']['base_url'] = config_update.base_url
            if config_update.model:
                config['llm']['openai']['model'] = config_update.model
                
        elif provider == 'qwen':
            if config_update.api_key:
                config['llm']['qwen']['api_key'] = config_update.api_key
            if config_update.base_url:
                config['llm']['qwen']['base_url'] = config_update.base_url
            if config_update.model:
                config['llm']['qwen']['model'] = config_update.model
        
        # 保存配置
        save_config(config)
        
        # 重新加载 LLM 管理器
        try:
            llm_manager = get_llm_manager()
            llm_manager.reload()
        except Exception as e:
            print(f"警告: 重新加载 LLM 管理器失败: {e}")
        
        return {
            "success": True,
            "message": f"已切换到 {provider}",
            "provider": provider
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/config/llm/test")
async def test_llm_connection(request: TestConnectionRequest):
    """测试 LLM 连接"""
    try:
        provider = request.provider
        config = get_config()
        
        # 检查提供商是否启用
        if not config['llm'].get(provider, {}).get('enabled'):
            raise HTTPException(status_code=400, detail=f"{provider} 未启用")
        
        # 获取 LLM 管理器
        llm_manager = get_llm_manager()
        llm_provider = llm_manager.get_provider()
        
        if not llm_provider:
            raise HTTPException(status_code=500, detail="LLM 提供商未初始化")
        
        # 测试简单调用
        start_time = time.time()
        
        try:
            # 尝试调用 chat 方法
            if hasattr(llm_provider, 'chat'):
                response = llm_provider.chat([{
                    "role": "user",
                    "content": "你好，请回复'测试成功'"
                }])
            else:
                # 尝试异步方法
                import asyncio
                response = await llm_provider.generate([{
                    "role": "user",
                    "content": "你好，请回复'测试成功'"
                }])
            
            response_time = int((time.time() - start_time) * 1000)
            
            return {
                "success": True,
                "provider": provider,
                "response_time": response_time,
                "message": "连接测试成功"
            }
            
        except Exception as call_error:
            raise HTTPException(
                status_code=500, 
                detail=f"调用失败: {str(call_error)}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/config/reload")
async def reload_configuration():
    """重新加载配置"""
    try:
        reload_config()
        
        # 重新加载 LLM 管理器
        try:
            llm_manager = get_llm_manager()
            llm_manager.reload()
        except Exception as e:
            print(f"警告: 重新加载 LLM 管理器失败: {e}")
        
        return {
            "success": True,
            "message": "配置已重新加载"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
