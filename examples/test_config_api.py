"""
测试 LLM 配置管理功能

验证：
1. 配置读取/保存
2. 提供商切换
3. API 端点
"""

import requests
import json
import time

API_BASE = "http://localhost:8000"

def test_get_config():
    """测试获取配置"""
    print("=" * 60)
    print("测试 1: 获取 LLM 配置")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_BASE}/api/config/llm")
        response.raise_for_status()
        
        data = response.json()
        print("✓ 请求成功")
        print()
        print("当前配置：")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False


def test_update_config_deepseek():
    """测试切换到 DeepSeek"""
    print()
    print("=" * 60)
    print("测试 2: 切换到 DeepSeek API")
    print("=" * 60)
    
    try:
        config = {
            "provider": "deepseek",
            "api_key": "sk-test-key-for-demo",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat"
        }
        
        response = requests.post(
            f"{API_BASE}/api/config/llm",
            json=config,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        data = response.json()
        print("✓ 配置更新成功")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False


def test_update_config_local_qwen():
    """测试切换到 Qwen2.5 本地"""
    print()
    print("=" * 60)
    print("测试 3: 切换到 Qwen2.5 本地")
    print("=" * 60)
    
    try:
        config = {
            "provider": "local_qwen",
            "model_path": "Qwen/Qwen2.5-3B-Instruct",
            "device": "auto"
        }
        
        response = requests.post(
            f"{API_BASE}/api/config/llm",
            json=config,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        data = response.json()
        print("✓ 配置更新成功")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False


def test_connection():
    """测试连接（如果可用）"""
    print()
    print("=" * 60)
    print("测试 4: 测试 LLM 连接")
    print("=" * 60)
    print("(如果 API Key 无效或模型未加载，此测试会失败)")
    
    try:
        response = requests.post(
            f"{API_BASE}/api/config/llm/test",
            json={"provider": "local_qwen"},
            headers={"Content-Type": "application/json"}
        )
        
        data = response.json()
        
        if response.status_code == 200 and data.get("success"):
            print("✓ 连接测试成功")
            print(f"  响应时间: {data.get('response_time')}ms")
        else:
            print("✗ 连接测试失败")
            print(f"  原因: {data.get('detail', '未知错误')}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False


def test_reload_config():
    """测试重载配置"""
    print()
    print("=" * 60)
    print("测试 5: 重载配置")
    print("=" * 60)
    
    try:
        response = requests.post(f"{API_BASE}/api/config/reload")
        response.raise_for_status()
        
        data = response.json()
        print("✓ 配置重载成功")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False


def main():
    """运行所有测试"""
    print()
    print("LLM 配置管理功能测试")
    print()
    print(f"API 地址: {API_BASE}")
    print()
    
    # 检查服务是否运行
    try:
        response = requests.get(f"{API_BASE}/")
        print("✓ API 服务运行中")
    except Exception as e:
        print(f"✗ 无法连接到 API 服务: {e}")
        print()
        print("请先启动 API 服务：")
        print("  .\\start_api.ps1")
        print()
        return
    
    print()
    time.sleep(1)
    
    # 运行测试
    results = []
    
    results.append(("获取配置", test_get_config()))
    time.sleep(0.5)
    
    results.append(("切换到 DeepSeek", test_update_config_deepseek()))
    time.sleep(0.5)
    
    results.append(("切换到 Qwen2.5", test_update_config_local_qwen()))
    time.sleep(0.5)
    
    results.append(("测试连接", test_connection()))
    time.sleep(0.5)
    
    results.append(("重载配置", test_reload_config()))
    
    # 汇总结果
    print()
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print()
    print(f"通过率: {passed}/{total} ({passed/total*100:.0f}%)")
    
    if passed == total:
        print()
        print("🎉 所有测试通过！")
        print()
        print("下一步：")
        print("1. 访问配置页面：http://localhost:8000/config.html")
        print("2. 选择您想要的 LLM 提供商")
        print("3. 填写配置并保存")
        print("4. 测试连接确保配置正确")
    else:
        print()
        print("⚠ 部分测试失败")
        print()
        print("提示：")
        print("- '测试连接' 失败通常是因为 API Key 无效或模型未加载")
        print("- 其他测试失败请检查 API 服务日志")


if __name__ == "__main__":
    main()
