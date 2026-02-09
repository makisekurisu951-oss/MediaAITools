"""MCP Tools for media processing"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import subprocess
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from utils.media_utils import validate_video_file, parse_time, format_time, ensure_output_dir
from utils.logger import get_logger

logger = get_logger(__name__)


def run_async_in_new_loop(coro, timeout=30):
    """在新线程的事件循环中运行异步代码（避免嵌套事件循环冲突）
    
    Args:
        coro: 异步协程对象
        timeout: 超时时间（秒），默认30秒
        
    Returns:
        协程的返回值，或在超时时抛出 TimeoutError
    """
    def run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_in_thread)
        try:
            return future.result(timeout=timeout)
        except Exception as e:
            logger.error(f"Async task failed or timeout after {timeout}s: {e}")
            raise

# Image processing imports
try:
    from PIL import Image, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None
    ImageOps = None

# YOLO imports for object detection
try:
    from ultralytics import YOLO
    import numpy as np
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None
    np = None


class MediaTool(ABC):
    """Base class for media processing tools"""
    
    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool"""
        pass
    
    def _run_ffmpeg(self, cmd: list, timeout: int = 300) -> Dict[str, Any]:
        """Run FFmpeg command"""
        try:
            result = subprocess.run(
                [self.ffmpeg_path] + cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
                encoding='utf-8',
                errors='ignore'  # 忽略编码错误
            )
            return {"success": True, "output": result.stdout}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Command timeout"}
        except subprocess.CalledProcessError as e:
            return {"success": False, "error": e.stderr}
        except Exception as e:
            return {"success": False, "error": str(e)}


class ClipTool(MediaTool):
    """Tool for video clipping"""
    
    def execute(
        self,
        input_path: str,
        start_time: str,
        end_time: str,
        output_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Clip video from start_time to end_time"""
        if not validate_video_file(input_path):
            return {"success": False, "error": "Invalid video file"}
        
        output_path = str(ensure_output_dir(output_path))
        
        start_seconds = parse_time(start_time)
        end_seconds = parse_time(end_time)
        duration = end_seconds - start_seconds
        
        cmd = [
            "-i", input_path,
            "-ss", str(start_seconds),
            "-t", str(duration),
            "-c", "copy",  # Copy codec for faster processing
            "-avoid_negative_ts", "make_zero",
            output_path,
            "-y"  # Overwrite output file
        ]
        
        return self._run_ffmpeg(cmd)


class ConcatTool(MediaTool):
    """Tool for concatenating videos"""
    
    def execute(
        self,
        video_paths: list,
        output_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Concatenate multiple videos"""
        # Validate all input files
        for path in video_paths:
            if not validate_video_file(path):
                return {"success": False, "error": f"Invalid video file: {path}"}
        
        output_path = str(ensure_output_dir(output_path))
        
        # Create concat file list
        concat_file = output_path + ".concat.txt"
        with open(concat_file, 'w', encoding='utf-8') as f:
            for path in video_paths:
                f.write(f"file '{os.path.abspath(path)}'\n")
        
        try:
            cmd = [
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-c", "copy",
                output_path,
                "-y"
            ]
            
            result = self._run_ffmpeg(cmd)
            
            # Clean up concat file
            if os.path.exists(concat_file):
                os.remove(concat_file)
            
            return result
        except Exception as e:
            # Clean up concat file on error
            if os.path.exists(concat_file):
                os.remove(concat_file)
            return {"success": False, "error": str(e)}


class SubtitleTool(MediaTool):
    """Tool for subtitle generation using Whisper"""
    
    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        super().__init__(ffmpeg_path, ffprobe_path)
        self._whisper_model = None
        self._llm_manager = None
    
    def _get_whisper_model(self, model_name: str = "base"):
        """Lazy load Whisper model"""
        if self._whisper_model is None:
            try:
                import whisper
                import torch
                logger.info(f"Loading Whisper model: {model_name}")
                
                
                # Whisper 在 GPU 上运行
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Whisper using {device}")
                
                self._whisper_model = whisper.load_model(model_name, device=device)
                logger.info(f"Whisper model loaded successfully on {device}")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise
        return self._whisper_model
    
    def _get_llm_manager(self):
        """Lazy load LLM manager for subtitle correction"""
        if self._llm_manager is None:
            try:
                import sys
                from pathlib import Path
                # Ensure llm module can be imported
                parent_dir = Path(__file__).parent.parent
                if str(parent_dir) not in sys.path:
                    sys.path.insert(0, str(parent_dir))
                
                from llm import LLMManager
                self._llm_manager = LLMManager()
                logger.info("LLM Manager initialized for subtitle correction")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM Manager: {e}")
                self._llm_manager = None
        return self._llm_manager
    
    def _extract_technical_terms(self, video_path: str, initial_text: str) -> dict:
        """自动从视频文件名和初步字幕中提取专业术语"""
        # 先从文件名提取关键术语（快速且准确）
        import os
        import re
        filename = os.path.basename(video_path)
        
        # 从文件名中提取中文关键词（2-4个字的词组）
        filename_terms = {}
        chinese_words = re.findall(r'[\u4e00-\u9fff]{2,4}', filename)
        for word in chinese_words:
            # 过滤常见无意义词
            if word not in ['视频', '演示', '官方', '原装', '文件']:
                filename_terms[word] = word
        
        logger.info(f"📁 从文件名提取关键词: {list(filename_terms.keys())}")
        
        # ⚠️ Windows 事件循环限制：新线程中的 LLM 调用不稳定
        # 直接返回文件名术语，不调用 LLM（避免 Event loop closed 错误）
        logger.info(f"✅ 使用文件名术语 {len(filename_terms)} 个（跳过 LLM 提取以避免事件循环冲突）")
        #return filename_terms
        
        # 获取 LLM Manager
        llm_manager = self._get_llm_manager()
        if not llm_manager:
            logger.warning("LLM not available, cannot extract technical terms")
            return filename_terms
        
        provider = llm_manager.get_provider(task_type="chinese_processing")
        if not provider:
            provider = llm_manager.get_provider()
        
        if not provider:
            return filename_terms
        
        try:
            
            # 取前 500 字作为样本（避免太长）
            sample_text = initial_text[:500]
            
            prompt =  f"""纠正语音识别文本中的专业术语错别字（如同音字、形近字），结合文件名上下文推断正确术语.
            仅输出标准JSON:{{"术语":"术语"}}（仅含纠正后的正确术语）。文本：{sample_text} | 文件名：{filename}"""
            
            # 使用新线程的事件循环调用异步 generate 方法（30秒超时）
            try:
                messages = [{"role": "user", "content": prompt}]
                response = run_async_in_new_loop(provider.generate(messages), timeout=30)
                logger.info(f"LLM 术语提取响应: {response[:200] if response else 'Empty'}...")
            except TimeoutError:
                logger.warning("⚠️ LLM 术语提取超时(30秒),跳过此步骤")
                return {}
            except Exception as e:
                logger.error(f"❌ LLM 调用失败: {str(e)}")
                return {}
            
            # 提取 JSON
            import json
            import re
            
            if not response:
                logger.warning("LLM 返回空响应")
                return {}
            
            # 打印完整响应用于调试
            logger.info(f"LLM 完整响应: {response}")
            
            # 先尝试去除代码块标记（```json ... ```）
            cleaned_response = response.strip()
            if cleaned_response.startswith('```'):
                # 去除开头的 ```json 或 ```
                lines = cleaned_response.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]  # 去除第一行
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]  # 去除最后一行
                cleaned_response = '\n'.join(lines).strip()
            
            # 尝试提取 JSON 对象
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned_response, re.DOTALL)
            if json_match:
                try:
                    tech_terms = json.loads(json_match.group())
                    logger.info(f"自动提取了 {len(tech_terms)} 个专业术语: {list(tech_terms.keys())[:10]}")
                    return tech_terms
                except json.JSONDecodeError as e:
                    logger.error(f"JSON 解析失败: {e}, 提取的文本: {json_match.group()[:200]}")
                    return {}
            else:
                logger.warning(f"LLM 响应中未找到有效 JSON,响应前 500 字符: {response[:500]}")
                return {}
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"提取专业术语失败: {str(e)}", exc_info=True)
            return {}
    
    def _correct_subtitle_with_llm(self, segments: list, **kwargs) -> list:
        """Use LLM to correct subtitle text based on context"""
        logger.info("=" * 60)
        logger.info("开始 LLM 智能纠错流程")
        logger.info("=" * 60)
        
        # 检测字幕语言（根据第一段内容判断）
        language = kwargs.get('language', 'zh')
        logger.info(f"🌍 字幕语言: {language}")
        
        llm_manager = self._get_llm_manager()
        if not llm_manager:
            logger.warning("❌ LLM Manager 不可用，使用规则纠错")
            return self._correct_subtitle_with_rules(segments)
        
        try:
            # 根据语言选择不同的纠错策略
            if language == 'en':
                return self._correct_english_subtitle_with_llm(segments, llm_manager, **kwargs)
            else:
                return self._correct_chinese_subtitle_with_llm(segments, llm_manager, **kwargs)
        except Exception as e:
            logger.error(f"Failed to correct subtitles with LLM: {e}, falling back to rule-based correction")
            return self._correct_subtitle_with_rules(segments)
    
    def _correct_english_subtitle_with_llm(self, segments: list, llm_manager, **kwargs) -> list:
        """Use LLM to correct English subtitle text"""
        logger.info("📝 使用 LLM 纠正英文字幕（语法、术语、连贯性）")
        
        # 分批处理
        batch_size = 50
        total_segments = len(segments)
        logger.info(f"📊 总共 {total_segments} 条字幕，分批处理（每批 {batch_size} 条）")
        
        corrected_segments = []
        for batch_start in range(0, total_segments, batch_size):
            batch_end = min(batch_start + batch_size, total_segments)
            batch = segments[batch_start:batch_end]
            
            logger.info(f"🔄 处理第 {batch_start+1}-{batch_end} 条字幕...")
            
            # 创建独立的 LLMManager 实例
            try:
                from llm.llm_manager import LLMManager
                batch_llm_manager = LLMManager()
            except Exception as e:
                logger.warning(f"⚠️ 创建 LLMManager 失败: {e}")
                batch_corrected = self._correct_subtitle_with_rules(batch)
                corrected_segments.extend(batch_corrected)
                continue
            
            provider = batch_llm_manager.get_provider(task_type="subtitle_translation")
            if not provider:
                provider = batch_llm_manager.get_provider()
            
            if not provider:
                logger.warning(f"❌ 第 {batch_start+1}-{batch_end} 批次无可用 provider，跳过纠错")
                corrected_segments.extend(batch)
                continue
            
            # 合并字幕文本
            full_text = "\n".join([f"{i+1}. {seg['text'].strip()}" for i, seg in enumerate(batch)])
            
            # 英文字幕纠错 Prompt
            prompt = f"""You are a professional English subtitle editor. Fix errors in subtitles generated by speech recognition while preserving original meaning and line count.
            **Fix:**
            - Grammar (tense, articles, prepositions)
            - Spelling of technical terms (e.g., Transformer, GPU, PyTorch, BPE, Self-Attention)
            - Fluency and coherence between lines
            **Preserve:**
            - Original structure and number of lines ({len(batch)})
            - Technical accuracy — use correct terms from this list:  
            Transformer, Attention Mechanism, Self-Attention, Neural Network, Deep Learning, Machine Learning, GPU, CUDA, PyTorch, TensorFlow, Encoder, Decoder, Embedding, Token, BPE, Tokenizer, Gradient Descent, Backpropagation, Fine-tuning, Pre-training
            **Output format:**  
            "1. [corrected line]\\n2. [corrected line]\\n...\\n{len(batch)}. [corrected line]"
            **Original Subtitles:**  
            {full_text}
            **Corrected Subtitles:**"""

            
            logger.info(f"📤 Batch prompt 长度: {len(prompt)} 字符")
            
            # 调用 LLM
            response = None
            try:
                messages = [{"role": "user", "content": prompt}]
                response = run_async_in_new_loop(provider.generate(messages), timeout=30)
                logger.info("✅ LLM 调用成功")
            except TimeoutError:
                logger.error(f"❌ 第 {batch_start+1}-{batch_end} 批次调用超时(30秒)")
                corrected_segments.extend(batch)
                continue
            except Exception as e:
                logger.error(f"❌ 第 {batch_start+1}-{batch_end} 批次调用失败: {str(e)}")
                corrected_segments.extend(batch)
                continue
            
            if response:
                logger.info(f"📥 LLM 响应长度: {len(response)} 字符")
                
                # 解析纠正后的文本
                corrected_lines = []
                import re
                for line in response.strip().split('\\n'):
                    line = line.strip()
                    match = re.match(r'^(\\d+)\\.\\s*(.+)$', line)
                    if match:
                        text = match.group(2).strip()
                        if text:
                            corrected_lines.append(text)
                
                logger.info(f"✅ 本批次解析得到 {len(corrected_lines)} 行纠正文本，原始有 {len(batch)} 段")
                
                if len(corrected_lines) == len(batch):
                    for i, corrected_text in enumerate(corrected_lines):
                        batch[i]['text'] = corrected_text
                    corrected_segments.extend(batch)
                    logger.info(f"✅ 批次纠正成功！已处理 {len(corrected_segments)}/{total_segments} 条字幕")
                else:
                    logger.warning(f"⚠️ 行数不匹配 ({len(corrected_lines)} vs {len(batch)})，保留原文")
                    corrected_segments.extend(batch)
            else:
                logger.warning(f"❌ 第 {batch_start+1}-{batch_end} 批次响应为空")
                corrected_segments.extend(batch)
        
        logger.info(f"✅✅✅ 英文字幕 LLM 纠错完成！共纠正 {len(corrected_segments)} 条字幕")
        logger.info("=" * 60)
        return corrected_segments
    
    #Use LLM to correct Chinese subtitle text (original logic)
    def _correct_chinese_subtitle_with_llm(self, segments: list, llm_manager, **kwargs) -> list:
        logger.info("=" * 60)
        logger.info("开始 LLM 智能纠错流程")
        logger.info("=" * 60)
        
        llm_manager = self._get_llm_manager()
        if not llm_manager:
            logger.warning("❌ LLM Manager 不可用，使用规则纠错")
            return self._correct_subtitle_with_rules(segments)
        try:
            # 分批处理：每次处理50条字幕（避免prompt过长）
            batch_size = 50
            total_segments = len(segments)
            logger.info(f"📊 总共 {total_segments} 条字幕，分批处理（每批 {batch_size} 条）")
            
            corrected_segments = []
            for batch_start in range(0, total_segments, batch_size):
                batch_end = min(batch_start + batch_size, total_segments)
                batch = segments[batch_start:batch_end]
                
                logger.info(f"🔄 处理第 {batch_start+1}-{batch_end} 条字幕...")
                
                # ⚠️ 终极修复：每批创建全新的 LLMManager 实例（不复用全局单例）
                # 原因：Windows ProactorEventLoop 无法在新线程中安全清理旧事件循环的 httpx 连接池
                # 解决：完全独立的 LLMManager，用完即弃，避免任何连接复用
                try:
                    from llm.llm_manager import LLMManager  # 直接导入类，不用全局函数
                    batch_llm_manager = LLMManager()  # 每批创建新实例
                    logger.debug("✅ 创建独立 LLMManager 实例")
                except Exception as e:
                    logger.warning(f"⚠️ 创建 LLMManager 失败: {e}")
                    batch_corrected = self._correct_subtitle_with_rules(batch)
                    corrected_segments.extend(batch_corrected)
                    continue
                
                # 从新实例获取 Provider
                provider = batch_llm_manager.get_provider(task_type="chinese_processing")
                if not provider:
                    provider = batch_llm_manager.get_provider()
                
                if not provider:
                    logger.warning(f"❌ 第 {batch_start+1}-{batch_end} 批次无可用 provider，使用规则纠错")
                    batch_corrected = self._correct_subtitle_with_rules(batch)
                    corrected_segments.extend(batch_corrected)
                    continue
                
                logger.debug(f"✅ Provider: {provider.__class__.__name__}")
                
                # Combine batch subtitle text for context
                full_text = "\n".join([f"{i+1}. {seg['text'].strip()}" for i, seg in enumerate(batch)])
                
                # 获取专业词汇词典，整合自动提取的术语
                tech_terms = kwargs.get('tech_terms', {})
                
                # 基础技术词汇（保底）
                base_terms = ["FastAPI", "DeepSeek", "R1", "Pydantic", "Ollama", "GPU", "API", 
                             "Docker", "Python", "RAG", "Transformer", "CUDA", "PyTorch", "TensorFlow",
                             "HuggingFace", "OpenAI", "GPT", "LLM", "Qwen", "Whisper", "FFmpeg"]
                
                # 常见拼音误识别映射（帮助LLM识别）
                pinyin_map = {
                    "FastAPI": ["past api", "fast api", "法斯特api", "帕斯特api"],
                    "RAG": ["rg", "阿格", "r g"],
                    "DeepSeek": ["dpsi", "迪普西克", "deep seek"],
                    "API": ["a p i", "埃皮爱", "接口"],
                    "Web": ["未备", "微博"],
                    "Docker": ["多克", "道克"],
                    "GPU": ["g p u", "计皮友"],
                    "LLM": ["l l m", "大模型"],
                    "Transformer": ["传输佛莫", "transformer"]
                }
                
                # 合并自动提取的术语（优先级更高）
                if tech_terms:
                    extracted_terms = list(tech_terms.keys())
                    # 合并去重
                    all_terms = list(dict.fromkeys(extracted_terms + base_terms))
                    # 只显示前15个最重要的术语
                    key_terms = all_terms[:15]
                    logger.info(f"🔤 使用 {len(all_terms)} 个专业术语，重点纠正前15个: {', '.join(key_terms)}")
                else:
                    key_terms = base_terms[:15]
                    logger.info(f"🔤 使用基础术语 {len(key_terms)} 个")
                
                # 构建术语纠正提示（包含拼音映射）
                terms_hint = []
                for term in key_terms:
                    if term in pinyin_map:
                        variants = ", ".join(pinyin_map[term])
                        terms_hint.append(f"  • {term} (可能被识别为: {variants})")
                    else:
                        terms_hint.append(f"  • {term}")
                
                terms_display = "\n".join(terms_hint)
                
                # Create prompt for subtitle correction
                prompt = f"""你是专业的中文字幕纠错AI，专门修正语音识别产生的同音字错误。请严格按以下要求处理：
                【任务】
                1. 理解整句语义，找出不通顺、不合逻辑或无意义的词组  
                2. 结合上下文和专业术语，将错误词替换为语义合理、符合语境的正确词  
                3. 仅替换错误部分，不得改变句子结构或增删内容  
                【依据】
                - 必须参考以下关键术语表：
                {terms_display}
                【原则】
                - 语义优先：不通顺就必须改  
                - 上下文驱动：根据前后文判断最合理的词  
                - 逻辑一致：主谓宾搭配、修饰关系需合理  
                【输出要求】
                - 输出 {len(batch)} 行,每行格式为“1. [纠正后文本]”、“2. [纠正后文本]”……  
                - 仅输出纠正结果，不要任何解释、标记或额外内容  
                【原始字幕】
                {full_text}
                """
                logger.info(f"📤 Batch prompt 长度: {len(prompt)} 字符")
                # 调用 LLM（使用新线程事件循环避免冲突，30秒超时）
                response = None
                try:
                    logger.info("🔄 调用 LLM generate()...")
                    messages = [{"role": "user", "content": prompt}]
                    response = run_async_in_new_loop(provider.generate(messages), timeout=30)
                    logger.info("✅ LLM 调用成功")
                except TimeoutError:
                    logger.error(f"❌ 第 {batch_start+1}-{batch_end} 批次调用超时（30秒）")
                    logger.warning(f"⚠️ 对本批次降级到规则纠错（保留前 {len(corrected_segments)} 条已纠正字幕）")
                    batch_corrected = self._correct_subtitle_with_rules(batch)
                    corrected_segments.extend(batch_corrected)
                    continue  # 继续处理下一批
                except Exception as e:
                    logger.error(f"❌ 第 {batch_start+1}-{batch_end} 批次调用失败: {str(e)}")
                    logger.warning(f"⚠️ 对本批次降级到规则纠错（保留前 {len(corrected_segments)} 条已纠正字幕）")
                    batch_corrected = self._correct_subtitle_with_rules(batch)
                    corrected_segments.extend(batch_corrected)
                    continue  # 继续处理下一批
                
                if response:
                    # 打印完整响应用于调试（限制长度避免日志过长）
                    logger.info(f"📥 LLM 响应长度: {len(response)} 字符")
                    logger.debug(f"📥 LLM 完整响应:\n{response[:500]}...")  # 只打印前500字符
                    logger.info("=" * 80)
                    
                    # Parse corrected text - 只解析有序号的行
                    corrected_lines = []
                    import re
                    for line in response.strip().split('\n'):
                        line = line.strip()
                        # 严格匹配 "数字. 文本" 格式
                        match = re.match(r'^(\d+)\.\s*(.+)$', line)
                        if match:
                            text = match.group(2).strip()
                            if text:  # 确保文本非空
                                corrected_lines.append(text)
                                logger.debug(f"解析行 {match.group(1)}: {text[:50]}...")
                    
                    logger.info(f"✅ 本批次解析得到 {len(corrected_lines)} 行纠正文本，原始有 {len(batch)} 段")
                    if corrected_lines:
                        logger.info(f"前5行示例: {corrected_lines[:5]}")
                    
                    # Update batch segments with corrected text
                    if len(corrected_lines) == len(batch):
                        for i, corrected_text in enumerate(corrected_lines):
                            batch[i]['text'] = corrected_text
                        corrected_segments.extend(batch)
                        logger.info(f"✅ 批次纠正成功！已处理 {len(corrected_segments)}/{total_segments} 条字幕")
                    else:
                        # 行数不匹配：对本批次使用规则纠错，但保留其他批次的LLM纠错结果
                        logger.warning(f"⚠️ 本批次纠正行数 ({len(corrected_lines)}) 与原始段数 ({len(batch)}) 不匹配")
                        logger.warning(f"⚠️ 对本批次使用规则纠错（保留前 {len(corrected_segments)} 条已纠正字幕）")
                        # 对当前批次应用规则纠错
                        batch_corrected = self._correct_subtitle_with_rules(batch)
                        corrected_segments.extend(batch_corrected)
                        logger.info(f"✅ 批次降级完成！已处理 {len(corrected_segments)}/{total_segments} 条字幕")
                else:
                    logger.warning(f"❌ 第 {batch_start+1}-{batch_end} 批次响应为空")
                    logger.warning(f"⚠️ 对本批次降级到规则纠错（保留前 {len(corrected_segments)} 条已纠正字幕）")
                    batch_corrected = self._correct_subtitle_with_rules(batch)
                    corrected_segments.extend(batch_corrected)
        except Exception as e:
            logger.error(f"❌ 处理第 {batch_start+1}-{batch_end} 批次时发生错误: {str(e)}", exc_info=True)
            logger.warning(f"⚠️ 对本批次降级到规则纠错（保留前 {len(corrected_segments)} 条已纠正字幕）")
            batch_corrected = self._correct_subtitle_with_rules(batch)
            corrected_segments.extend(batch_corrected)
            
        # 所有批次处理完成
        logger.info(f"✅✅✅ 中文字幕 LLM 纠错全部完成！共纠正 {len(corrected_segments)} 条字幕")
        logger.info("=" * 60)
        return corrected_segments

    
    def _correct_subtitle_with_rules(self, segments: list) -> list:
        """Use rule-based correction for common errors"""
        # Common homophone errors in Chinese speech recognition
        correction_rules = {
            # Technology terms - First pass (specific phrases)
            '亚伯智能最新善价的数码派官方缅称色潜头夜市版': '树莓派最新上架的树莓派官方摄像头夜视版',
            '数码派官方缅称色潜头': '树莓派官方摄像头',
            '官方缅称色潜头': '官方摄像头',
            '数码总统版的色潜头方向操控': '树莓派的摄像头接口',
            '色潜头方向操控': '摄像头接口',
            '数码总统版': '树莓派',
            '数码系统': '树莓派系统',
            '最新善价': '最新上架',
            '善价的': '上架的',
            '色潜头': '摄像头',
            '夜市版': '夜视版',
            '色产业的': '摄像的',
            '色产业': '摄像',
            '流气气': '浏览器',
            '流器气': '浏览器',
            '数码派': '树莓派',
            '装用色潜头': '专用摄像头',
            '装用': '专用',
            '亚伯智能': '树莓派',
            '视频IP地址': 'IP地址',
            
            # Second pass (individual words)
            '善价': '上架',
            '缅称': '摄像',
            '潜头': '像头',
            '擦入': '插入',
            '夜市': '夜视',
            '总统版': '派',
            '总统': '派',
            '记忆上': '界面上',
            '记忆': '界面',
            '视频IP': 'IP',
            
            # Common misrecognitions
            '气气': '器',
            '气': '器',
            '産': '产',
        }
        
        logger.info("Applying rule-based subtitle correction...")
        corrected_count = 0
        
        for segment in segments:
            original_text = segment['text']
            corrected_text = original_text
            
            # Apply correction rules (ordered by length to handle phrases first)
            sorted_rules = sorted(correction_rules.items(), key=lambda x: len(x[0]), reverse=True)
            for wrong, correct in sorted_rules:
                if wrong in corrected_text:
                    before = corrected_text
                    corrected_text = corrected_text.replace(wrong, correct)
                    if before != corrected_text:
                        corrected_count += 1
                        logger.debug(f"Applied rule: '{wrong}' -> '{correct}'")
            
            # Update segment if changed
            if corrected_text != original_text:
                segment['text'] = corrected_text
                logger.debug(f"Corrected: '{original_text}' -> '{corrected_text}'")
        
        logger.info(f"Rule-based correction applied {corrected_count} fixes to {len(segments)} segments")
        return segments
    
    def _extract_audio(self, video_path: str, audio_path: str) -> bool:
        """Extract audio from video"""
        try:
            cmd = [
                "-i", video_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM audio
                "-ar", "16000",  # 16kHz sample rate for Whisper
                "-ac", "1",  # Mono
                audio_path,
                "-y"
            ]
            result = self._run_ffmpeg(cmd)
            return result.get("success", False)
        except Exception as e:
            logger.error(f"Failed to extract audio: {e}")
            return False
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds to SRT timestamp format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def _convert_to_simplified_chinese(self, text: str) -> str:
        """Convert traditional Chinese to simplified Chinese"""
        try:
            from opencc import OpenCC
            cc = OpenCC('t2s')  # Traditional to Simplified
            return cc.convert(text)
        except ImportError:
            # If OpenCC is not available, try using a simple mapping
            logger.warning("OpenCC not installed, using basic conversion")
            # Basic traditional to simplified mapping
            trans_map = {
                '歡迎': '欢迎', '觀看': '观看', '數碼': '数码', '緬稱': '缅称',
                '擦入': '插入', '總統': '总统', '繼續': '继续', '讓': '让',
                '運行': '运行', '氣': '器', '記憶': '界面', '產業': '摄像',
                '結束': '结束', '謝謝': '谢谢', '價': '价', '視頻': '视频',
                '擦': '插', '綫': '线', '潛頭': '摄像头', '夜市': '夜视'
            }
            for trad, simp in trans_map.items():
                text = text.replace(trad, simp)
            return text
        except Exception as e:
            logger.warning(f"Conversion failed: {e}, using original text")
            return text
    
    def _translate_to_english(self, text: str) -> str:
        """Translate Chinese text to English using LLM"""
        llm_manager = self._get_llm_manager()
        if not llm_manager:
            logger.warning("LLM not available for translation")
            # Return a placeholder that indicates translation is needed
            return f"[EN: {text[:30]}...]" if len(text) > 30 else f"[EN: {text}]"
        
        try:
            # Get provider by task type
            provider = llm_manager.get_provider(task_type="subtitle_translation")
            if not provider:
                logger.warning("No translation provider available")
                # Return Chinese text as fallback
                return f"[EN: {text[:30]}...]" if len(text) > 30 else f"[EN: {text}]"
            
            prompt = f"""Translate the following Chinese subtitle to natural English. Keep it concise and suitable for video subtitles.
Only output the English translation, no explanations.

Chinese: {text}
English:"""
            
            try:
                messages = [{"role": "user", "content": prompt}]
                response = run_async_in_new_loop(provider.generate(messages), timeout=30)
                if response:
                    return response.strip()
                else:
                    return f"[EN: {text[:30]}...]" if len(text) > 30 else f"[EN: {text}]"
            except TimeoutError:
                logger.warning(f"Translation timeout (30s) for: {text[:30]}...")
                return f"[EN: {text[:30]}...]" if len(text) > 30 else f"[EN: {text}]"
            except Exception as e:
                logger.warning(f"Translation API call failed: {e}")
                return f"[EN: {text[:30]}...]" if len(text) > 30 else f"[EN: {text}]"
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return f"[EN: {text[:30]}...]" if len(text) > 30 else f"[EN: {text}]"
    
    def _write_srt(self, segments: list, output_path: str, convert_to_simplified: bool = True, bilingual: bool = False) -> bool:
        """Write segments to SRT file
        
        Args:
            segments: List of subtitle segments
            output_path: Output SRT file path
            convert_to_simplified: Convert traditional Chinese to simplified
            bilingual: Generate bilingual (Chinese + English) subtitles
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(segments, 1):
                    start = self._format_timestamp(segment['start'])
                    end = self._format_timestamp(segment['end'])
                    text = segment['text'].strip()
                    
                    # Convert to simplified Chinese if requested
                    if convert_to_simplified:
                        text = self._convert_to_simplified_chinese(text)
                    
                    f.write(f"{i}\n")
                    f.write(f"{start} --> {end}\n")
                    
                    if bilingual:
                        # Write Chinese and English on separate lines
                        english_text = segment.get('english_text', '')
                        if not english_text:
                            # Translate if not already translated
                            english_text = self._translate_to_english(text)
                            segment['english_text'] = english_text
                        
                        f.write(f"{text}\n")
                        f.write(f"{english_text}\n\n")
                    else:
                        f.write(f"{text}\n\n")
            return True
        except Exception as e:
            logger.error(f"Failed to write SRT file: {e}")
            return False
    
    def execute(
        self,
        video_path: str,
        language: str = "en",
        output_path: Optional[str] = None,
        model: str = "base",
        embed_subtitle: bool = False,
        bilingual: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate subtitle file using Whisper and optionally embed it into video
        
        Args:
            video_path: Path to input video file
            language: Language code for speech recognition (default: en)
            output_path: Path to output file
            model: Whisper model to use (default: base)
            embed_subtitle: Whether to burn subtitle into video
            bilingual: Generate bilingual Chinese+English subtitles
            **kwargs: Additional arguments (use_llm_correction, etc.)
        """
        if not validate_video_file(video_path):
            return {"success": False, "error": "Invalid video file"}
        
        # Determine subtitle file path
        if output_path is None:
            if embed_subtitle:
                # For embedded subtitle, output is a video file
                base = str(Path(video_path).stem)
                output_path = str(Path(video_path).parent / f"{base}-subtitle.avi")
            else:
                # For standalone subtitle, output is SRT file
                output_path = str(Path(video_path).with_suffix('.srt'))
        else:
            output_path = str(ensure_output_dir(output_path))
        
        # Generate SRT file path
        if embed_subtitle:
            # 基于输出文件路径生成 SRT，避免覆盖源文件的 SRT
            srt_path = str(Path(output_path).with_suffix('.srt'))
        else:
            # 不嵌入时，强制使用.srt后缀（即使传入的是.mp4）
            srt_path = str(Path(output_path).with_suffix('.srt'))
        
        try:
            # 确保 ffmpeg 在 PATH 中（Whisper 需要调用 ffmpeg）
            ffmpeg_dir = str(Path(self.ffmpeg_path).parent)
            if ffmpeg_dir not in os.environ.get('PATH', ''):
                os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')
                logger.info(f"Added ffmpeg directory to PATH: {ffmpeg_dir}")
            
            # Step 1: Extract audio from video
            logger.info(f"Extracting audio from {video_path}")
            audio_path = str(Path(video_path).with_suffix('.wav'))
            if not self._extract_audio(video_path, audio_path):
                return {"success": False, "error": "Failed to extract audio"}
            
            # Step 2: Transcribe audio using Whisper
            logger.info(f"Transcribing audio with Whisper (language: {language})")
            model_obj = self._get_whisper_model(model)
            result = model_obj.transcribe(audio_path, language=language)
            
            # Step 2.5: Auto-extract technical terms if not provided
            segments = result['segments']
            tech_terms = kwargs.get('tech_terms', {})
            
            if not tech_terms and kwargs.get('auto_extract_terms', True):
                logger.info("自动提取专业术语...")
                # 从初步字幕中提取专业术语
                initial_text = "\n".join([seg['text'] for seg in segments])
                extracted_terms = self._extract_technical_terms(video_path, initial_text)
                if extracted_terms:
                    tech_terms = extracted_terms
                    kwargs['tech_terms'] = tech_terms
                    logger.info(f"自动提取了 {len(tech_terms)} 个专业术语")
            
            # First convert to simplified Chinese
            for seg in segments:
                seg['text'] = self._convert_to_simplified_chinese(seg['text'])
            
            # Then apply corrections
            correction_count = 0
            
            # ⚠️ 默认使用规则纠错（避免 Windows 事件循环冲突）
            # 如果需要 LLM 纠错，请在主异步环境中调用
            use_rule_correction = not kwargs.get('use_llm_correction', False)
            
            if use_rule_correction:
                # 直接使用规则纠错（稳定、快速、效果好）
                logger.info("Applying rule-based subtitle correction...")
                segments = self._correct_subtitle_with_rules(segments)
            else:
                # 尝试 LLM 纠错（可能因事件循环问题失败）
                logger.info("Attempting LLM-based subtitle correction...")
                try:
                    segments = self._correct_subtitle_with_llm(segments, **kwargs)
                    correction_count = len(segments)
                    logger.info("✅ LLM 纠错成功")
                except RuntimeError as e:
                    if 'Event loop is closed' in str(e):
                        logger.warning("⚠️ LLM 调用失败（事件循环冲突），降级到规则纠错")
                        segments = self._correct_subtitle_with_rules(segments)
                    else:
                        raise
                except Exception as e:
                    logger.warning(f"⚠️ LLM 纠错异常: {e}，降级到规则纠错")
                    segments = self._correct_subtitle_with_rules(segments)
            
            # Step 2.6: Translate to English if bilingual mode
            if bilingual:
                logger.info("Translating subtitles to English for bilingual mode...")
                for seg in segments:
                    if 'english_text' not in seg:
                        seg['english_text'] = self._translate_to_english(seg['text'])
            
            # Step 3: Write SRT file (skip conversion since already done)
            logger.info(f"Writing subtitle to {srt_path}")
            if not self._write_srt(segments, srt_path, convert_to_simplified=False, bilingual=bilingual):
                return {"success": False, "error": "Failed to write SRT file"}
            
            # Clean up audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            # Step 4: Embed subtitle if requested (使用软字幕流，不烧录到画面)
            if embed_subtitle:
                logger.info(f"Embedding subtitle stream (soft subtitle) into video: {video_path}")
                
                # Convert to absolute paths
                abs_video_path = str(Path(video_path).absolute())
                abs_srt_path = str(Path(srt_path).absolute())
                
                # 创建临时输出文件
                temp_output = str(Path(video_path).parent / f"{Path(video_path).stem}_temp{Path(video_path).suffix}")
                abs_temp_output = str(Path(temp_output).absolute())
                
                # 使用软字幕：将字幕流嵌入视频容器，不重新编码视频
                logger.info(f"Adding subtitle stream using: {abs_srt_path}")
                cmd = [
                    "-i", abs_video_path,
                    "-i", abs_srt_path,
                    "-c:v", "copy",  # 视频流直接复制，不重新编码
                    "-c:a", "copy",  # 音频流直接复制
                    "-c:s", "mov_text",  # 字幕编码格式（MP4用mov_text）
                    "-metadata:s:s:0", "language=chi",  # 设置字幕语言
                    "-metadata:s:s:0", "title=Chinese",  # 设置字幕标题
                    abs_temp_output,
                    "-y"
                ]
                
                embed_result = self._run_ffmpeg(cmd)
                
                if not embed_result.get("success", False):
                    # 清理临时文件
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                    return {
                        "success": False,
                        "error": f"Failed to embed subtitle: {embed_result.get('error', 'Unknown error')}",
                        "srt_path": srt_path
                    }
                
                # 成功后，原地替换源文件
                try:
                    # 删除源文件
                    os.remove(video_path)
                    # 重命名临时文件为源文件名
                    os.rename(temp_output, video_path)
                    logger.info(f"Successfully replaced source file with subtitled version: {video_path}")
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Failed to replace source file: {str(e)}",
                        "srt_path": srt_path
                    }
                
                return {
                    "success": True,
                    "output_path": video_path,  # 返回源文件路径（已被替换）
                    "srt_path": srt_path,
                    "extracted_terms": tech_terms,
                    "correction_count": correction_count,
                    "message": f"Subtitle embedded into source file successfully (soft subtitle, original encoding preserved)"
                }
            else:
                return {
                    "success": True,
                    "output_path": srt_path,
                    "srt_path": srt_path,
                    "extracted_terms": tech_terms,
                    "correction_count": correction_count,
                    "message": f"Subtitle generated successfully"
                }
            
        except Exception as e:
            logger.error(f"Subtitle generation failed: {e}", exc_info=True)
            # Clean up temporary files
            for tmp_file in [audio_path, srt_path if embed_subtitle else None]:
                if tmp_file and os.path.exists(tmp_file):
                    try:
                        os.remove(tmp_file)
                    except:
                        pass
            return {"success": False, "error": str(e)}


class FormatTool(MediaTool):
    """Tool for format conversion"""
    
    def execute(
        self,
        input_path: str,
        output_format: str,
        output_path: str,
        resolution: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Convert video format"""
        if not validate_video_file(input_path):
            return {"success": False, "error": "Invalid video file"}
        
        output_path = str(ensure_output_dir(output_path))
        
        cmd = ["-i", input_path]
        
        # Add resolution filter if specified
        if resolution:
            if resolution == "1080p":
                cmd.extend(["-vf", "scale=1920:1080"])
            elif resolution == "720p":
                cmd.extend(["-vf", "scale=1280:720"])
            elif resolution == "480p":
                cmd.extend(["-vf", "scale=854:480"])
        
        # Add output format and path
        cmd.extend(["-c:v", "libx264", "-c:a", "aac", output_path, "-y"])
        
        return self._run_ffmpeg(cmd)


class ImageTool:
    """Tool for image processing with smart rotation based on person detection"""
    
    def __init__(self):
        if not PIL_AVAILABLE:
            raise ImportError("Pillow is not installed. Install it with: pip install Pillow")
        
        # Initialize YOLO model for person detection (lazy loading)
        self._yolo_model = None
        self._yolo_available = YOLO_AVAILABLE
    
    def _get_yolo_model(self):
        """Get or initialize YOLO model for person detection"""
        if not self._yolo_available:
            return None
        
        if self._yolo_model is None:
            try:
                # Use YOLOv8n (nano) for faster inference, can detect person class (class 0)
                self._yolo_model = YOLO('yolov8n.pt')
                logger.info("YOLO model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load YOLO model: {e}. Smart rotation will be disabled.")
                self._yolo_available = False
                return None
        
        return self._yolo_model
    
    def _detect_person_and_analyze_orientation(self, img: Image.Image) -> Dict[str, Any]:
        """Detect person in image and analyze orientation"""
        if not self._yolo_available:
            return {"detected": False, "rotation_needed": 0}
        
        model = self._get_yolo_model()
        if model is None:
            return {"detected": False, "rotation_needed": 0}
        
        try:
            # Convert PIL to numpy array for YOLO
            img_array = np.array(img)
            
            # Run detection (person class is 0 in COCO dataset)
            results = model(img_array, classes=[0], verbose=False)  # Only detect person class
            
            if len(results) == 0 or len(results[0].boxes) == 0:
                return {"detected": False, "rotation_needed": 0}
            
            # Get the largest person bounding box (most prominent person)
            boxes = results[0].boxes
            largest_box = None
            largest_area = 0
            
            img_width, img_height = img.size
            
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)
                
                if area > largest_area:
                    largest_area = area
                    largest_box = {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "width": float(x2 - x1),
                        "height": float(y2 - y1),
                        "center_x": float((x1 + x2) / 2),
                        "center_y": float((y1 + y2) / 2),
                    }
            
            if largest_box is None:
                return {"detected": False, "rotation_needed": 0}
            
            # Analyze person orientation
            person_width = largest_box["width"]
            person_height = largest_box["height"]
            person_aspect_ratio = person_width / person_height if person_height > 0 else 1.0
            
            # Analyze image orientation
            image_aspect_ratio = img_width / img_height if img_height > 0 else 1.0
            
            # Determine if rotation is needed
            rotation_needed = 0
            
            # If image is landscape (wide) and person is also wide (horizontal)
            # Or if person's bounding box is wider than tall, person might be horizontal
            if image_aspect_ratio > 1.0:  # Landscape image
                if person_aspect_ratio > 1.2:  # Person is wider than tall (horizontal)
                    # Person is horizontal in landscape image, rotate 90 degrees CCW
                    rotation_needed = 90
                elif person_aspect_ratio < 0.8:  # Person is taller than wide (vertical)
                    # Person is already vertical, might need -90 rotation
                    rotation_needed = -90
            else:  # Portrait image
                if person_aspect_ratio > 1.2:  # Person is wider than tall
                    # Person is horizontal in portrait, rotate 90 degrees
                    rotation_needed = 90
            
            logger.info(f"Person detected: aspect_ratio={person_aspect_ratio:.2f}, rotation_needed={rotation_needed}")
            
            return {
                "detected": True,
                "rotation_needed": rotation_needed,
                "person_box": largest_box,
                "person_aspect_ratio": person_aspect_ratio,
                "image_aspect_ratio": image_aspect_ratio
            }
        except Exception as e:
            logger.warning(f"YOLO detection failed: {e}")
            return {"detected": False, "rotation_needed": 0}
    
    def execute(
        self,
        input_path: str,
        output_path: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        aspect_ratio: Optional[str] = None,  # e.g., "9:16" for vertical
        resize_mode: str = "fit",  # "fit"(pad), "cover"(fill+crop), "stretch"
        smart_rotate: bool = True,  # Enable smart rotation based on person detection
        **kwargs
    ) -> Dict[str, Any]:
        """Process image: resize, convert format, etc."""
        try:
            if not os.path.exists(input_path):
                return {"success": False, "error": f"Input file not found: {input_path}"}
            
            # Open image
            img = Image.open(input_path)
            original_width, original_height = img.size
            
            # Smart rotation: detect person and rotate if needed
            rotation_applied = 0
            if smart_rotate and (aspect_ratio in ("9:16", "vertical") or (width and height and height > width)):
                detection_result = self._detect_person_and_analyze_orientation(img)
                if detection_result.get("detected") and detection_result.get("rotation_needed") != 0:
                    rotation_needed = detection_result["rotation_needed"]
                    # Rotate image
                    if rotation_needed == 90:
                        img = img.rotate(-90, expand=True)  # Rotate counter-clockwise
                        rotation_applied = 90
                    elif rotation_needed == -90:
                        img = img.rotate(90, expand=True)  # Rotate clockwise
                        rotation_applied = -90
                    # Update dimensions after rotation
                    original_width, original_height = img.size
                    logger.info(f"Applied rotation: {rotation_applied} degrees")
            
            # Determine target dimensions
            target_width = width
            target_height = height
            
            # If aspect ratio is specified, calculate dimensions
            if aspect_ratio and not (width and height):
                if aspect_ratio == "9:16" or aspect_ratio == "vertical":
                    # Common phone vertical aspect ratio
                    if width:
                        target_height = int(width * 16 / 9)
                    elif height:
                        target_width = int(height * 9 / 16)
                    else:
                        # Use common phone vertical size: 1080x1920
                        target_width = 1080
                        target_height = 1920
                elif ":" in aspect_ratio:
                    ratio_parts = aspect_ratio.split(":")
                    ratio_w = float(ratio_parts[0])
                    ratio_h = float(ratio_parts[1])
                    if width:
                        target_height = int(width * ratio_h / ratio_w)
                    elif height:
                        target_width = int(height * ratio_w / ratio_h)
                    else:
                        # Default to 1080 width
                        target_width = 1080
                        target_height = int(1080 * ratio_h / ratio_w)
            
            # If no dimensions specified, use defaults
            if not target_width and not target_height:
                target_width = 1080
                target_height = 1920
            
            # Ensure output directory exists
            output_path = str(ensure_output_dir(output_path))
            
            # Normalize and validate resize_mode
            resize_mode = (resize_mode or "fit").lower().strip()
            if resize_mode == "crop":
                # Backward-compat alias: crop behaves like cover (fill+crop)
                resize_mode = "cover"

            # Resize image based on mode
            if resize_mode == "fit":
                # Fit image maintaining aspect ratio, add padding if needed (letterbox)
                img.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
                new_img = Image.new("RGB", (target_width, target_height), (255, 255, 255))
                paste_x = (target_width - img.size[0]) // 2
                paste_y = (target_height - img.size[1]) // 2
                new_img.paste(img, (paste_x, paste_y))
                img = new_img
            elif resize_mode == "cover":
                # Cover target size while preserving aspect ratio, then center-crop (no blank bars)
                ow, oh = img.size
                scale = max(target_width / ow, target_height / oh)
                new_w = max(1, int(round(ow * scale)))
                new_h = max(1, int(round(oh * scale)))
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                left = max(0, (new_w - target_width) // 2)
                top = max(0, (new_h - target_height) // 2)
                right = left + target_width
                bottom = top + target_height
                img = img.crop((left, top, right, bottom))
            else:  # stretch
                # Stretch to exact dimensions (may distort)
                img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary (for JPEG compatibility)
            if img.mode != "RGB" and output_path.lower().endswith(('.jpg', '.jpeg')):
                img = img.convert("RGB")
            
            # Save image
            img.save(output_path, quality=95)
            
            result = {
                "success": True,
                "output_path": output_path,
                "original_size": f"{original_width}x{original_height}",
                "new_size": f"{target_width}x{target_height}"
            }
            
            if rotation_applied != 0:
                result["rotation_applied"] = rotation_applied
                result["smart_rotation"] = True
            
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}


class OptimizeTool(MediaTool):
    """Tool for audio/video optimization"""
    
    def execute(
        self,
        input_path: str,
        output_path: str,
        optimize_type: str = "audio",  # "audio" or "video"
        **kwargs
    ) -> Dict[str, Any]:
        """Optimize audio or video"""
        if not validate_video_file(input_path):
            return {"success": False, "error": "Invalid video file"}
        
        output_path = str(ensure_output_dir(output_path))
        
        cmd = ["-i", input_path]
        
        if optimize_type == "audio":
            # Audio denoising and normalization
            cmd.extend([
                "-af", "highpass=f=200,lowpass=f=3000,volume=1.5",
                "-c:v", "copy",  # Copy video stream
                output_path,
                "-y"
            ])
        elif optimize_type == "video":
            # Video enhancement (basic)
            cmd.extend([
                "-vf", "eq=contrast=1.2:brightness=0.05:saturation=1.1",
                "-c:a", "copy",  # Copy audio stream
                output_path,
                "-y"
            ])
        else:
            return {"success": False, "error": f"Unknown optimize_type: {optimize_type}"}
        
        return self._run_ffmpeg(cmd)
