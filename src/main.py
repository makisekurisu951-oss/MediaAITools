"""Main entry point for MediaAITools"""

import asyncio
import sys
from pathlib import Path

# Add code directory to path
code_dir = Path(__file__).parent
sys.path.insert(0, str(code_dir))

from config import load_config
from utils.logger import setup_logger
from agent import MediaAgent


async def main():
    """Main function"""
    # Load configuration
    try:
        config = load_config()
        print("Configuration loaded successfully")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Please create config/config.yaml from config.example.yaml")
        return
    
    # Setup logger
    log_config = config.get("logging", {})
    logger = setup_logger(
        level=log_config.get("level", "INFO"),
        log_file=log_config.get("file"),
        max_bytes=log_config.get("max_bytes", 10485760),
        backup_count=log_config.get("backup_count", 5)
    )
    
    logger.info("MediaAITools starting...")
    
    # Initialize agent
    agent = MediaAgent()
    
    # Interactive mode
    print("\n" + "="*60)
    print("MediaAITools - AI-Powered Media Processing Agent")
    print("="*60)
    print("Type your media processing request in natural language.")
    print("Examples:")
    print("  - 剪辑视频 video.mp4 从 00:01:20 到 00:02:30")
    print("  - 为 video.mp4 生成中文字幕")
    print("  - 将 video.avi 转换为 mp4 格式，分辨率 1080p")
    print("  - 优化 video.mp4 的音频质量")
    print("\nType 'exit' or 'quit' to exit.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', '退出']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Process request
            print("Processing...")
            result = await agent.process(user_input)
            
            # Display result
            if result.get("success"):
                print(f"✓ Success! Method: {result.get('method', 'unknown')}")
                if "result" in result:
                    print(f"Result: {result['result']}")
            else:
                print(f"✗ Error: {result.get('error', 'Unknown error')}")
            
            print()  # Empty line for readability
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            print(f"Error: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
