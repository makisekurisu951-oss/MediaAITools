"""Test local Qwen2-VL model configuration and natural language config changes"""

import sys
import asyncio
from pathlib import Path

# Add code directory to path
code_dir = Path(__file__).parent / "code"
sys.path.insert(0, str(code_dir))

from skills.config_skill import ConfigSkill
from llm.llm_manager import get_llm_manager


async def test_config_skill():
    """Test natural language configuration changes"""
    print("=" * 60)
    print("Testing ConfigSkill - Natural Language Configuration")
    print("=" * 60)
    
    config_skill = ConfigSkill()
    
    # Test 1: Get current configuration
    print("\n1. Current Configuration:")
    current = config_skill.get_current_config()
    print(f"   Enabled providers: {current.get('enabled_providers', [])}")
    print(f"   Default models: {current.get('default_models', {})}")
    
    # Test 2: Switch to local model
    print("\n2. Testing: '使用本地模型'")
    result = await config_skill.execute("使用本地模型")
    print(f"   Result: {result}")
    
    # Verify change
    current = config_skill.get_current_config()
    print(f"   Enabled providers: {current.get('enabled_providers', [])}")
    
    # Test 3: Switch to OpenAI
    print("\n3. Testing: '切换到OpenAI'")
    result = await config_skill.execute("切换到OpenAI")
    print(f"   Result: {result}")
    
    current = config_skill.get_current_config()
    print(f"   Enabled providers: {current.get('enabled_providers', [])}")
    
    # Test 4: Switch to DeepSeek
    print("\n4. Testing: '用DeepSeek'")
    result = await config_skill.execute("用DeepSeek")
    print(f"   Result: {result}")
    
    current = config_skill.get_current_config()
    print(f"   Enabled providers: {current.get('enabled_providers', [])}")
    
    # Test 5: Switch to remote Qwen
    print("\n5. Testing: '用远程千问API'")
    result = await config_skill.execute("用远程千问API")
    print(f"   Result: {result}")
    
    current = config_skill.get_current_config()
    print(f"   Enabled providers: {current.get('enabled_providers', [])}")
    
    # Test 6: Switch back to local
    print("\n6. Testing: '切换回本地模型'")
    result = await config_skill.execute("切换回本地模型")
    print(f"   Result: {result}")
    
    current = config_skill.get_current_config()
    print(f"   Enabled providers: {current.get('enabled_providers', [])}")


def test_llm_manager():
    """Test LLM Manager with local model"""
    print("\n" + "=" * 60)
    print("Testing LLM Manager - Local Model Initialization")
    print("=" * 60)
    
    try:
        manager = get_llm_manager()
        
        print("\n1. Available Providers:")
        providers = manager.list_providers()
        for provider in providers:
            print(f"   ✓ {provider}")
        
        print("\n2. Default Models:")
        for task, provider in manager.default_models.items():
            print(f"   {task}: {provider}")
        
        print("\n3. Testing Provider Access:")
        provider = manager.get_provider(task_type="subtitle_translation")
        if provider:
            print(f"   ✓ Got provider for subtitle_translation: {provider.__class__.__name__}")
        else:
            print(f"   ✗ No provider available for subtitle_translation")
        
    except Exception as e:
        print(f"\n✗ LLM Manager test failed: {e}")
        import traceback
        traceback.print_exc()


def test_local_model_inference():
    """Test local model inference (if model is available)"""
    print("\n" + "=" * 60)
    print("Testing Local Model Inference")
    print("=" * 60)
    
    try:
        manager = get_llm_manager()
        provider = manager.get_provider(provider_name="local_qwen")
        
        if not provider:
            print("\n⚠ Local Qwen provider not initialized")
            print("  This is expected if:")
            print("  1. transformers/torch not installed")
            print("  2. Model not downloaded yet")
            print("  3. Insufficient memory/GPU")
            return
        
        print("\n✓ Local Qwen provider found")
        print(f"  Model path: {provider.model_path}")
        print(f"  Device: {provider.device}")
        
        # Try a simple inference
        print("\n  Testing inference with: '你好，请介绍一下你自己'")
        response = provider.chat("你好，请介绍一下你自己")
        print(f"  Response: {response[:100]}...")
        
    except Exception as e:
        print(f"\n⚠ Local model inference test skipped: {e}")
        print("  To enable local model:")
        print("  1. Install dependencies: pip install transformers torch")
        print("  2. Model will auto-download on first use")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("MediaAITools - Local LLM Configuration Test")
    print("=" * 60)
    
    # Test 1: Config Skill (async)
    asyncio.run(test_config_skill())
    
    # Test 2: LLM Manager
    test_llm_manager()
    
    # Test 3: Local Model Inference (if available)
    test_local_model_inference()
    
    print("\n" + "=" * 60)
    print("Tests Complete")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. If local model not loaded, install: pip install transformers torch")
    print("2. First run will download ~4GB Qwen2-VL-2B-Instruct model")
    print("3. Use natural language to switch LLMs:")
    print("   - '使用本地模型' - Local Qwen2-VL")
    print("   - '切换到OpenAI' - OpenAI GPT-4")
    print("   - '用DeepSeek' - DeepSeek Chat")
    print("   - '用远程千问API' - Qwen Cloud API")


if __name__ == "__main__":
    main()
