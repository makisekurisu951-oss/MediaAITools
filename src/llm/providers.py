"""LLM provider implementations"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

# Lazy imports to avoid errors when dependencies are not installed
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ChatOpenAI = None
    BaseMessage = object
    HumanMessage = None
    AIMessage = None
    SystemMessage = None

try:
    from langchain_community.chat_models import ChatAnthropic
except ImportError:
    ChatAnthropic = None


class BaseLLMProvider(ABC):
    """Base class for LLM providers"""
    
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None, **kwargs):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.kwargs = kwargs
        self._client = None
    
    @abstractmethod
    def get_client(self):
        """Get LLM client instance"""
        pass
    
    @abstractmethod
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response from messages"""
        pass
    
    def chat(self, prompt: str, **kwargs) -> str:
        """Synchronous chat method for simple prompts"""
        import asyncio
        import concurrent.futures
        
        messages = [{"role": "user", "content": prompt}]
        
        # 检查是否在运行的事件循环中
        try:
            loop = asyncio.get_running_loop()
            # 在运行的事件循环中，使用线程池运行新的事件循环
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._run_generate_sync, messages, **kwargs)
                return future.result(timeout=120)
        except RuntimeError:
            # 没有运行的事件循环，直接使用 asyncio.run()
            return asyncio.run(self.generate(messages, **kwargs))
    
    def _run_generate_sync(self, messages, **kwargs):
        """在新事件循环中运行 generate 方法"""
        import asyncio
        return asyncio.run(self.generate(messages, **kwargs))
    
    def format_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain messages to provider format"""
        formatted = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                formatted.append({"role": "system", "content": msg.content})
        return formatted


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider"""
    
    def get_client(self):
        """Get OpenAI client"""
        if not LANGCHAIN_AVAILABLE or ChatOpenAI is None:
            raise ImportError("langchain_openai is not installed. Install it with: pip install langchain-openai")
        
        if self._client is None:
            self._client = ChatOpenAI(
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                **self.kwargs
            )
        return self._client
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using OpenAI"""
        if not LANGCHAIN_AVAILABLE or HumanMessage is None:
            raise ImportError("langchain_core is not installed. Install it with: pip install langchain-core")
        
        client = self.get_client()
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
        
        response = await client.ainvoke(langchain_messages)
        return response.content


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek LLM provider (compatible with OpenAI API)"""
    
    def get_client(self):
        """Get DeepSeek client (using OpenAI-compatible API)"""
        if not LANGCHAIN_AVAILABLE or ChatOpenAI is None:
            raise ImportError("langchain_openai is not installed. Install it with: pip install langchain-openai")
        
        if self._client is None:
            self._client = ChatOpenAI(
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url or "https://api.deepseek.com/v1",
                **self.kwargs
            )
        return self._client
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using DeepSeek"""
        if not LANGCHAIN_AVAILABLE or HumanMessage is None:
            raise ImportError("langchain_core is not installed. Install it with: pip install langchain-core")
        
        client = self.get_client()
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
        
        response = await client.ainvoke(langchain_messages)
        return response.content


class QwenProvider(BaseLLMProvider):
    """Qwen (千问) LLM provider"""
    
    def get_client(self):
        """Get Qwen client"""
        if not LANGCHAIN_AVAILABLE or ChatOpenAI is None:
            raise ImportError("langchain_openai is not installed. Install it with: pip install langchain-openai")
        
        if self._client is None:
            # Qwen uses OpenAI-compatible API
            self._client = ChatOpenAI(
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
                **self.kwargs
            )
        return self._client
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using Qwen"""
        if not LANGCHAIN_AVAILABLE or HumanMessage is None:
            raise ImportError("langchain_core is not installed. Install it with: pip install langchain-core")
        
        client = self.get_client()
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
        
        response = await client.ainvoke(langchain_messages)
        return response.content


class LocalQwenProvider(BaseLLMProvider):
    """Local Qwen model provider (supports both Qwen2 and Qwen2-VL) using transformers"""
    
    def __init__(self, model_path: str, device: str = "auto", **kwargs):
        super().__init__(api_key="", model=model_path, **kwargs)
        self.device = device
        self.model_path = model_path
        self._model = None
        self._tokenizer = None
    
    def get_client(self):
        """Load local Qwen model (支持 Qwen2 和 Qwen2-VL)"""
        if self._model is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                
                # 智能检测：VL 模型使用专用加载器，纯文本模型使用通用加载器
                if "VL" in self.model_path or "vl" in self.model_path:
                    from transformers import Qwen2VLForConditionalGeneration
                    model_class = Qwen2VLForConditionalGeneration
                    model_type = "Qwen2-VL"
                else:
                    model_class = AutoModelForCausalLM
                    model_type = "Qwen2"
                
                # Load model and tokenizer
                self._model = model_class.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    device_map=self.device,
                    trust_remote_code=True
                )
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
                
                print(f"[OK] Local {model_type} provider initialized ({self.model_path})")
            except ImportError:
                raise ImportError(
                    "transformers and torch are required for local models. "
                    "Install with: pip install transformers torch"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load local model: {e}")
        
        return self._model
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using local Qwen2-VL model (正确的调用方式)"""
        import torch
        
        model = self.get_client()
        
        # 使用 apply_chat_template 格式化消息（官方推荐方式）
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        model_inputs = self._tokenizer([text], return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=kwargs.get("max_tokens", 4096),  # 增加到4096支持长字幕
                temperature=kwargs.get("temperature", 0.3),  # 降低温度提高准确性
                do_sample=kwargs.get("do_sample", True),
                top_p=kwargs.get("top_p", 0.9)
            )
        
        # Decode only the newly generated tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self._tokenizer.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return response.strip()
