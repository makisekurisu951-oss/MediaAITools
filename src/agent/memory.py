"""记忆系统 - 保存对话历史、上下文、执行结果"""

import json
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from collections import deque


@dataclass
class MemoryItem:
    """记忆项"""
    id: str
    timestamp: float
    type: str  # conversation, task, result, context
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryItem':
        return cls(**data)


class ShortTermMemory:
    """短期记忆（会话级别）"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.items: deque = deque(maxlen=max_size)
        self.current_session_id = self._generate_session_id()
    
    def add(self, item: MemoryItem):
        """添加记忆项"""
        self.items.append(item)
    
    def get_recent(self, n: int = 10) -> List[MemoryItem]:
        """获取最近的n条记忆"""
        return list(self.items)[-n:]
    
    def get_by_type(self, item_type: str) -> List[MemoryItem]:
        """按类型获取记忆"""
        return [item for item in self.items if item.type == item_type]
    
    def clear(self):
        """清空短期记忆"""
        self.items.clear()
        self.current_session_id = self._generate_session_id()
    
    def _generate_session_id(self) -> str:
        """生成会话ID"""
        return f"session_{int(time.time())}"


class LongTermMemory:
    """长期记忆（持久化存储）"""
    
    def __init__(self, storage_path: str = "logs/memory"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_path / "index.json"
        self.index = self._load_index()
    
    def save(self, item: MemoryItem):
        """保存到长期记忆"""
        # 按日期组织文件
        date_str = datetime.fromtimestamp(item.timestamp).strftime("%Y-%m-%d")
        file_path = self.storage_path / f"memory_{date_str}.jsonl"
        
        # 转换为字典并清理不可序列化的对象
        item_dict = item.to_dict()
        item_dict = self._make_serializable(item_dict)
        
        # 追加写入
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(item_dict, ensure_ascii=False) + '\n')
        
        # 更新索引
        self._update_index(item.id, date_str, item.type)
    
    def _make_serializable(self, obj: Any) -> Any:
        """将对象转换为 JSON 可序列化的格式"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif hasattr(obj, '__dict__'):
            # 对于有 __dict__ 的对象（如 HumanMessage），转换为字符串
            return str(obj)
        else:
            return str(obj)
    
    def load(self, item_id: str) -> Optional[MemoryItem]:
        """加载指定记忆"""
        if item_id not in self.index:
            return None
        
        date_str = self.index[item_id]["date"]
        file_path = self.storage_path / f"memory_{date_str}.jsonl"
        
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if data["id"] == item_id:
                    return MemoryItem.from_dict(data)
        
        return None
    
    def search(
        self, 
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        item_type: Optional[str] = None,
        limit: int = 100
    ) -> List[MemoryItem]:
        """搜索记忆"""
        results = []
        
        # 确定要搜索的文件
        files = sorted(self.storage_path.glob("memory_*.jsonl"))
        
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(results) >= limit:
                        break
                    
                    data = json.loads(line)
                    item = MemoryItem.from_dict(data)
                    
                    # 过滤条件
                    if start_time and item.timestamp < start_time:
                        continue
                    if end_time and item.timestamp > end_time:
                        continue
                    if item_type and item.type != item_type:
                        continue
                    
                    results.append(item)
        
        return results
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {
            "total_items": len(self.index),
            "by_type": {},
            "by_date": {}
        }
        
        for item_info in self.index.values():
            # 按类型统计
            item_type = item_info["type"]
            stats["by_type"][item_type] = stats["by_type"].get(item_type, 0) + 1
            
            # 按日期统计
            date = item_info["date"]
            stats["by_date"][date] = stats["by_date"].get(date, 0) + 1
        
        return stats
    
    def _load_index(self) -> Dict:
        """加载索引"""
        if self.index_file.exists():
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _update_index(self, item_id: str, date: str, item_type: str):
        """更新索引"""
        self.index[item_id] = {
            "date": date,
            "type": item_type
        }
        
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)


class MemoryManager:
    """记忆管理器 - 统一管理短期和长期记忆"""
    
    def __init__(self, storage_path: str = "logs/memory"):
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory(storage_path)
        self.conversation_history: List[Dict] = []
    
    def add_conversation(
        self,
        role: str,
        content: str,
        metadata: Dict = None
    ) -> str:
        """添加对话记忆"""
        item_id = f"conv_{int(time.time() * 1000)}"
        
        item = MemoryItem(
            id=item_id,
            timestamp=time.time(),
            type="conversation",
            content={
                "role": role,
                "content": content
            },
            metadata=metadata or {}
        )
        
        self.short_term.add(item)
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        
        return item_id
    
    def add_task(
        self,
        instruction: str,
        task_type: str,
        parameters: Dict,
        metadata: Dict = None
    ) -> str:
        """添加任务记忆"""
        item_id = f"task_{int(time.time() * 1000)}"
        
        item = MemoryItem(
            id=item_id,
            timestamp=time.time(),
            type="task",
            content={
                "instruction": instruction,
                "task_type": task_type,
                "parameters": parameters
            },
            metadata=metadata or {}
        )
        
        self.short_term.add(item)
        self.long_term.save(item)
        
        return item_id
    
    def add_result(
        self,
        task_id: str,
        success: bool,
        result: Any,
        execution_time: float,
        metadata: Dict = None
    ) -> str:
        """添加执行结果记忆"""
        item_id = f"result_{int(time.time() * 1000)}"
        
        item = MemoryItem(
            id=item_id,
            timestamp=time.time(),
            type="result",
            content={
                "task_id": task_id,
                "success": success,
                "result": result,
                "execution_time": execution_time
            },
            metadata=metadata or {}
        )
        
        self.short_term.add(item)
        self.long_term.save(item)
        
        return item_id
    
    def add_context(
        self,
        key: str,
        value: Any,
        metadata: Dict = None
    ) -> str:
        """添加上下文信息"""
        item_id = f"ctx_{int(time.time() * 1000)}"
        
        item = MemoryItem(
            id=item_id,
            timestamp=time.time(),
            type="context",
            content={
                "key": key,
                "value": value
            },
            metadata=metadata or {}
        )
        
        self.short_term.add(item)
        
        return item_id
    
    def get_conversation_history(self, n: int = 10) -> List[Dict]:
        """获取对话历史"""
        return self.conversation_history[-n:]
    
    def get_recent_tasks(self, n: int = 5) -> List[MemoryItem]:
        """获取最近的任务"""
        tasks = self.short_term.get_by_type("task")
        return tasks[-n:]
    
    def get_task_results(self, task_id: str) -> Optional[MemoryItem]:
        """获取任务结果"""
        results = self.short_term.get_by_type("result")
        for result in results:
            if result.content.get("task_id") == task_id:
                return result
        return None
    
    def get_context(self, key: str) -> Optional[Any]:
        """获取上下文"""
        contexts = self.short_term.get_by_type("context")
        for ctx in reversed(contexts):
            if ctx.content.get("key") == key:
                return ctx.content.get("value")
        return None
    
    def summarize_session(self) -> Dict:
        """总结当前会话"""
        tasks = self.short_term.get_by_type("task")
        results = self.short_term.get_by_type("result")
        
        successful_tasks = sum(
            1 for r in results if r.content.get("success")
        )
        
        total_time = sum(
            r.content.get("execution_time", 0) for r in results
        )
        
        return {
            "session_id": self.short_term.current_session_id,
            "total_conversations": len(self.conversation_history),
            "total_tasks": len(tasks),
            "successful_tasks": successful_tasks,
            "failed_tasks": len(results) - successful_tasks,
            "total_execution_time": total_time,
            "avg_execution_time": total_time / len(results) if results else 0
        }
    
    def clear_session(self):
        """清空当前会话"""
        self.short_term.clear()
        self.conversation_history.clear()


# 使用示例
if __name__ == "__main__":
    memory = MemoryManager()
    
    # 添加对话
    memory.add_conversation("user", "为视频添加字幕")
    memory.add_conversation("assistant", "好的，我来帮您处理")
    
    # 添加任务
    task_id = memory.add_task(
        instruction="为视频添加字幕",
        task_type="subtitle",
        parameters={"video_path": "test.mp4", "language": "zh"}
    )
    
    # 添加结果
    memory.add_result(
        task_id=task_id,
        success=True,
        result={"output_path": "test-字幕.mp4"},
        execution_time=35.5
    )
    
    # 获取总结
    summary = memory.summarize_session()
    print("会话总结:", json.dumps(summary, ensure_ascii=False, indent=2))
    
    # 获取统计
    stats = memory.long_term.get_statistics()
    print("长期记忆统计:", json.dumps(stats, ensure_ascii=False, indent=2))
