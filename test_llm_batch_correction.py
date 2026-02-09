"""测试 LLM 分批次纠错是否还有事件循环问题"""

import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcp_server.tools import SubtitleTool
from utils.logger import get_logger

logger = get_logger(__name__)

def create_test_segments(count=150):
    """创建测试字幕数据（模拟 Whisper 输出）"""
    test_texts = [
        "大家好,欢迎观看亚伯智能最新善价的数码派官方缅称色潜头夜市版的夜市视频",
        "今天我们来介绍一下这个数码总统版的摄像头怎么使用",
        "首先我们需要把色潜头擦入到数码派的接口上",
        "然后打开我们的流气气访问树莓派的IP地址",
        "在记忆上我们就可以看到实时的视频流",
        "这个夜市功能在晚上也能清晰地看到画面",
        "接下来我们看一下如何配置摄像头的参数",
        "在配置文件里面我们可以修改分辨率和帧率",
        "安装这一个工具我们先看取加C",
        "然后到了这边之后直接点击属该路件就可以把过去那边",
    ]
    
    segments = []
    for i in range(count):
        text = test_texts[i % len(test_texts)]
        segments.append({
            'text': text,
            'start': i * 3.0,
            'end': (i + 1) * 3.0
        })
    
    return segments

def main():
    print("=" * 80)
    print("开始测试 LLM 分批次纠错")
    print("=" * 80)
    
    logger.info("=" * 80)
    logger.info("开始测试 LLM 分批次纠错")
    logger.info("=" * 80)
    
    # 创建工具实例
    print("创建 SubtitleTool 实例...")
    tool = SubtitleTool()
    print("✅ SubtitleTool 创建成功")
    
    # 测试大批次（300条，分6批）
    segment_count = 300
    print(f"\n创建 {segment_count} 条测试字幕（大批次测试，每批50条，共6批）...")
    segments = create_test_segments(segment_count)
    print(f"✅ 创建了 {len(segments)} 条测试字幕")
    print(f"前3条示例:")
    for i, s in enumerate(segments[:3], 1):
        print(f"  {i}. {s['text'][:50]}...")
    
    # 调用 LLM 纠错
    try:
        print("\n" + "=" * 80)
        print("🔄 开始调用 _correct_subtitle_with_llm()...")
        print("=" * 80)
        
        corrected = tool._correct_subtitle_with_llm(
            segments, 
            use_llm_correction=True,
            tech_terms={'树莓派': '树莓派', '摄像头': '摄像头', '夜视': '夜视'}
        )
        
        print("\n" + "=" * 80)
        print(f"✅✅✅ 测试成功！共纠正 {len(corrected)} 条字幕")
        print("=" * 80)
        
        # 显示前5条纠正结果
        print("\n前5条纠正结果：")
        for i, seg in enumerate(corrected[:5], 1):
            print(f"{i}. {seg['text']}")
        
        # 检查是否有明显纠错
        has_correction = any('树莓派' in seg['text'] for seg in corrected[:10])
        if has_correction:
            print("\n✅ 检测到纠错生效（找到'树莓派'）")
        else:
            print("\n⚠️ 未检测到明显纠错，可能 LLM 未生效")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        logger.error(f"❌ 测试失败: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
