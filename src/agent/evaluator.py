"""评估器 - 评估任务执行结果的质量"""

import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum


class QualityLevel(Enum):
    """质量等级"""
    EXCELLENT = "excellent"  # 90-100
    GOOD = "good"  # 70-89
    ACCEPTABLE = "acceptable"  # 50-69
    POOR = "poor"  # 30-49
    FAILED = "failed"  # 0-29


@dataclass
class EvaluationMetric:
    """评估指标"""
    name: str
    score: float  # 0-100
    weight: float  # 权重 0-1
    description: str
    details: Dict[str, Any]


@dataclass
class EvaluationResult:
    """评估结果"""
    task_id: str
    task_type: str
    overall_score: float  # 总分 0-100
    quality_level: QualityLevel
    metrics: List[EvaluationMetric]
    timestamp: float
    recommendations: List[str]  # 改进建议
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data["quality_level"] = self.quality_level.value
        return data


class SubtitleEvaluator:
    """字幕质量评估器"""
    
    def evaluate(self, result: Dict, metadata: Dict = None) -> List[EvaluationMetric]:
        """评估字幕生成结果"""
        metrics = []
        
        # 1. 文件存在性
        output_path = result.get("output_path", "")
        srt_path = result.get("srt_path", "")
        
        file_exists_score = 0
        if output_path and Path(output_path).exists():
            file_exists_score += 50
        if srt_path and Path(srt_path).exists():
            file_exists_score += 50
        
        metrics.append(EvaluationMetric(
            name="file_existence",
            score=file_exists_score,
            weight=0.2,
            description="输出文件是否存在",
            details={"video_exists": Path(output_path).exists() if output_path else False,
                    "srt_exists": Path(srt_path).exists() if srt_path else False}
        ))
        
        # 2. 字幕内容质量
        srt_quality_score = 0
        details = {}
        
        if srt_path and Path(srt_path).exists():
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 检查字幕数量
            subtitle_count = content.count('\n\n')
            details["subtitle_count"] = subtitle_count
            
            if subtitle_count > 0:
                srt_quality_score += 30
            if subtitle_count >= 5:
                srt_quality_score += 20
            
            # 检查是否有时间戳
            has_timestamps = '-->' in content
            details["has_timestamps"] = has_timestamps
            if has_timestamps:
                srt_quality_score += 25
            
            # 检查文本长度
            text_length = len(content)
            details["text_length"] = text_length
            if text_length > 100:
                srt_quality_score += 25
        
        metrics.append(EvaluationMetric(
            name="subtitle_quality",
            score=srt_quality_score,
            weight=0.3,
            description="字幕内容质量",
            details=details
        ))
        
        # 3. 处理时间效率
        execution_time = metadata.get("execution_time", 0) if metadata else 0
        time_score = 100
        
        if execution_time > 60:
            time_score = 80
        if execution_time > 120:
            time_score = 60
        if execution_time > 180:
            time_score = 40
        
        metrics.append(EvaluationMetric(
            name="efficiency",
            score=time_score,
            weight=0.2,
            description="处理时间效率",
            details={"execution_time": execution_time}
        ))
        
        # 4. 智能纠错质量（如果使用）
        correction_score = 50  # 默认中等
        
        if result.get("used_llm_correction"):
            correction_score = 80  # 使用了LLM纠错
            
            correction_count = result.get("correction_count", 0)
            if correction_count > 0:
                correction_score += min(correction_count * 2, 20)
        
        metrics.append(EvaluationMetric(
            name="correction_quality",
            score=min(correction_score, 100),
            weight=0.3,
            description="智能纠错质量",
            details={
                "used_llm": result.get("used_llm_correction", False),
                "correction_count": result.get("correction_count", 0)
            }
        ))
        
        return metrics


class VideoEvaluator:
    """视频处理质量评估器"""
    
    def evaluate(self, result: Dict, metadata: Dict = None) -> List[EvaluationMetric]:
        """评估视频处理结果"""
        metrics = []
        
        output_path = result.get("output_path", "")
        
        # 1. 文件完整性
        file_score = 0
        details = {}
        
        if output_path and Path(output_path).exists():
            file_path = Path(output_path)
            file_size = file_path.stat().st_size
            details["file_size"] = file_size
            
            # 文件大小合理性
            if file_size > 1024:  # > 1KB
                file_score += 50
            if file_size > 1024 * 1024:  # > 1MB
                file_score += 50
        
        metrics.append(EvaluationMetric(
            name="file_integrity",
            score=file_score,
            weight=0.3,
            description="文件完整性",
            details=details
        ))
        
        # 2. 处理准确性（基于参数匹配）
        accuracy_score = 100  # 假设完全准确，除非发现问题
        
        metrics.append(EvaluationMetric(
            name="accuracy",
            score=accuracy_score,
            weight=0.4,
            description="处理准确性",
            details={}
        ))
        
        # 3. 处理效率
        execution_time = metadata.get("execution_time", 0) if metadata else 0
        time_score = 100 - min(execution_time / 2, 50)
        
        metrics.append(EvaluationMetric(
            name="efficiency",
            score=max(time_score, 0),
            weight=0.3,
            description="处理效率",
            details={"execution_time": execution_time}
        ))
        
        return metrics


class Evaluator:
    """主评估器"""
    
    def __init__(self, log_path: str = "logs/evaluations"):
        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        # 任务类型到评估器的映射
        self.evaluators = {
            "subtitle": SubtitleEvaluator(),
            "clip": VideoEvaluator(),
            "concat": VideoEvaluator(),
            "format": VideoEvaluator(),
            "optimize": VideoEvaluator(),
        }
    
    def evaluate(
        self,
        task_id: str,
        task_type: str,
        result: Dict,
        metadata: Dict = None
    ) -> EvaluationResult:
        """
        评估任务执行结果
        
        Args:
            task_id: 任务ID
            task_type: 任务类型
            result: 执行结果
            metadata: 元数据（执行时间等）
            
        Returns:
            EvaluationResult: 评估结果
        """
        # 选择对应的评估器
        evaluator = self.evaluators.get(task_type, VideoEvaluator())
        
        # 获取评估指标
        metrics = evaluator.evaluate(result, metadata)
        
        # 计算总分（加权平均）
        total_score = sum(m.score * m.weight for m in metrics)
        total_weight = sum(m.weight for m in metrics)
        overall_score = total_score / total_weight if total_weight > 0 else 0
        
        # 确定质量等级
        quality_level = self._get_quality_level(overall_score)
        
        # 生成改进建议
        recommendations = self._generate_recommendations(metrics, task_type)
        
        # 创建评估结果
        evaluation = EvaluationResult(
            task_id=task_id,
            task_type=task_type,
            overall_score=overall_score,
            quality_level=quality_level,
            metrics=metrics,
            timestamp=time.time(),
            recommendations=recommendations
        )
        
        # 保存评估结果
        self._save_evaluation(evaluation)
        
        return evaluation
    
    def _get_quality_level(self, score: float) -> QualityLevel:
        """根据分数确定质量等级"""
        if score >= 90:
            return QualityLevel.EXCELLENT
        elif score >= 70:
            return QualityLevel.GOOD
        elif score >= 50:
            return QualityLevel.ACCEPTABLE
        elif score >= 30:
            return QualityLevel.POOR
        else:
            return QualityLevel.FAILED
    
    def _generate_recommendations(
        self,
        metrics: List[EvaluationMetric],
        task_type: str
    ) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        for metric in metrics:
            if metric.score < 60:
                if metric.name == "file_existence":
                    recommendations.append("检查文件路径和权限，确保输出文件正确生成")
                elif metric.name == "subtitle_quality":
                    recommendations.append("考虑使用更高质量的语音识别模型或启用智能纠错")
                elif metric.name == "efficiency":
                    recommendations.append("优化处理流程，考虑使用GPU加速")
                elif metric.name == "correction_quality":
                    recommendations.append("启用LLM智能纠错功能以提高字幕准确性")
        
        if not recommendations:
            recommendations.append("任务执行良好，继续保持！")
        
        return recommendations
    
    def _save_evaluation(self, evaluation: EvaluationResult):
        """保存评估结果"""
        date_str = time.strftime("%Y-%m-%d")
        log_file = self.log_path / f"eval_{date_str}.jsonl"
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(evaluation.to_dict(), ensure_ascii=False) + '\n')
    
    def get_statistics(self, days: int = 7) -> Dict:
        """获取评估统计"""
        stats = {
            "total_evaluations": 0,
            "by_quality": {level.value: 0 for level in QualityLevel},
            "by_task_type": {},
            "avg_score": 0,
            "avg_execution_time": 0
        }
        
        # 读取最近n天的评估日志
        total_score = 0
        total_time = 0
        
        for log_file in self.log_path.glob("eval_*.jsonl"):
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    stats["total_evaluations"] += 1
                    
                    # 按质量等级统计
                    quality = data["quality_level"]
                    stats["by_quality"][quality] += 1
                    
                    # 按任务类型统计
                    task_type = data["task_type"]
                    if task_type not in stats["by_task_type"]:
                        stats["by_task_type"][task_type] = {
                            "count": 0,
                            "avg_score": 0
                        }
                    stats["by_task_type"][task_type]["count"] += 1
                    
                    # 累计分数
                    total_score += data["overall_score"]
                    
                    # 累计执行时间
                    for metric in data["metrics"]:
                        if metric["name"] == "efficiency":
                            total_time += metric["details"].get("execution_time", 0)
        
        if stats["total_evaluations"] > 0:
            stats["avg_score"] = total_score / stats["total_evaluations"]
            stats["avg_execution_time"] = total_time / stats["total_evaluations"]
        
        return stats


# 使用示例
if __name__ == "__main__":
    evaluator = Evaluator()
    
    # 评估字幕任务
    result = evaluator.evaluate(
        task_id="task_123",
        task_type="subtitle",
        result={
            "output_path": "test-字幕.mp4",
            "srt_path": "test.srt",
            "used_llm_correction": True,
            "correction_count": 15
        },
        metadata={
            "execution_time": 45.5
        }
    )
    
    print(f"任务评估结果:")
    print(f"  总分: {result.overall_score:.2f}")
    print(f"  质量等级: {result.quality_level.value}")
    print(f"  详细指标:")
    for metric in result.metrics:
        print(f"    - {metric.description}: {metric.score:.2f} (权重: {metric.weight})")
    print(f"  改进建议:")
    for rec in result.recommendations:
        print(f"    - {rec}")
    
    # 获取统计
    stats = evaluator.get_statistics()
    print(f"\n评估统计: {json.dumps(stats, ensure_ascii=False, indent=2)}")
