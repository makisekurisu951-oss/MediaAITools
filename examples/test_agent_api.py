"""测试 Agent REST API - 自然语言处理视频"""

import requests
import time
import json

# API 地址
API_BASE = "http://localhost:8000"

def test_agent_subtitle():
    """测试 Agent 自然语言字幕生成"""
    
    print("\n" + "=" * 70)
    print("测试 Agent REST API - 自然语言处理")
    print("=" * 70)
    
    # 1. 检查健康状态
    print("\n[1] 检查 API 状态...")
    try:
        response = requests.get(f"{API_BASE}/api/health")
        health = response.json()
        print(f"✓ API 状态: {health}")
        
        if not health.get("agent_ready"):
            print("✗ MediaAgent 未就绪，请先启动 API 服务器")
            return
    except requests.exceptions.ConnectionError:
        print("✗ 无法连接到 API 服务器")
        print("请先运行: python -m uvicorn api.main:app --reload")
        return
    
    # 2. 发送自然语言指令
    print("\n[2] 发送自然语言指令...")
    instruction = "帮我把D:\\MediaAITools\\test\\subtitle-test目录下面的mp4文件添加字幕"
    print(f"指令: {instruction}")
    
    payload = {
        "instruction": instruction,
        "use_llm_correction": True,
        "language": "zh"
    }
    
    response = requests.post(
        f"{API_BASE}/api/agent/process",
        json=payload
    )
    
    if response.status_code != 200:
        print(f"✗ 请求失败: {response.status_code}")
        print(response.text)
        return
    
    result = response.json()
    task_id = result["task_id"]
    print(f"✓ 任务已创建: {task_id}")
    print(f"  消息: {result['message']}")
    
    # 3. 轮询任务状态
    print("\n[3] 等待任务完成...")
    max_wait = 1800  # 最多等待30分钟
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        response = requests.get(f"{API_BASE}/api/tasks/{task_id}")
        status = response.json()
        
        current_status = status.get("status")
        progress = status.get("progress", 0)
        message = status.get("message", "")
        
        print(f"\r[{current_status}] {progress}% - {message}", end="", flush=True)
        
        if current_status == "completed":
            print("\n✓ 任务完成！")
            print("\n结果：")
            print(json.dumps(status.get("result"), indent=2, ensure_ascii=False))
            break
        elif current_status == "failed":
            print(f"\n✗ 任务失败: {message}")
            if "error_details" in status:
                print("\n详细错误信息：")
                print(status["error_details"])
            print("\n完整状态：")
            print(json.dumps(status, indent=2, ensure_ascii=False))
            break
        
        time.sleep(5)  # 每5秒检查一次
    else:
        print(f"\n⚠ 任务超时（超过{max_wait}秒）")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_agent_subtitle()
