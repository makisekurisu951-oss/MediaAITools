"""基准测试 - 测试系统性能和准确性"""

import time
import json
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime


@dataclass
class BenchmarkCase:
    """基准测试用例"""
    id: str
    name: str
    task_type: str
    instruction: str
    input_files: List[str]
    expected_output: Dict[str, Any]
    parameters: Dict[str, Any]
    timeout: float = 300.0  # 5分钟超时


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    case_id: str
    success: bool
    execution_time: float
    accuracy_score: float  # 准确性分数 0-100
    performance_score: float  # 性能分数 0-100
    error_message: str = ""
    output: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class BenchmarkSuite:
    """基准测试套件"""
    
    def __init__(self):
        self.test_cases: List[BenchmarkCase] = []
        self.results: List[BenchmarkResult] = []
        
        # 加载预定义的测试用例
        self._load_test_cases()
    
    def _load_test_cases(self):
        """加载测试用例"""
        # 字幕生成测试
        self.test_cases.append(BenchmarkCase(
            id="subtitle_001",
            name="基础字幕生成",
            task_type="subtitle",
            instruction="为视频添加中文字幕",
            input_files=["test/trade.avi"],
            expected_output={
                "has_video": True,
                "has_srt": True,
                "min_subtitle_count": 5
            },
            parameters={
                "language": "zh",
                "use_llm_correction": False
            }
        ))
        
        self.test_cases.append(BenchmarkCase(
            id="subtitle_002",
            name="智能纠错字幕生成",
            task_type="subtitle",
            instruction="为视频添加中文字幕并智能纠错",
            input_files=["test/trade.avi"],
            expected_output={
                "has_video": True,
                "has_srt": True,
                "min_subtitle_count": 5,
                "used_llm": True
            },
            parameters={
                "language": "zh",
                "use_llm_correction": True
            }
        ))
        
        # 视频剪辑测试
        self.test_cases.append(BenchmarkCase(
            id="clip_001",
            name="视频剪辑",
            task_type="clip",
            instruction="剪辑视频从00:00:05到00:00:15",
            input_files=["test/trade.avi"],
            expected_output={
                "has_output": True,
                "duration_approx": 10.0
            },
            parameters={
                "start_time": "00:00:05",
                "end_time": "00:00:15"
            }
        ))
        
        # 格式转换测试
        self.test_cases.append(BenchmarkCase(
            id="format_001",
            name="格式转换",
            task_type="format",
            instruction="转换视频为mp4格式",
            input_files=["test/trade.avi"],
            expected_output={
                "has_output": True,
                "format": "mp4"
            },
            parameters={
                "output_format": "mp4"
            }
        ))
    
    def add_test_case(self, test_case: BenchmarkCase):
        """添加自定义测试用例"""
        self.test_cases.append(test_case)
    
    async def run_benchmark(
        self,
        executor_func,
        test_cases: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        运行基准测试
        
        Args:
            executor_func: 执行任务的函数
            test_cases: 要运行的测试用例ID列表，None表示运行所有
            
        Returns:
            测试报告
        """
        start_time = time.time()
        
        # 确定要运行的测试用例
        cases_to_run = self.test_cases
        if test_cases:
            cases_to_run = [c for c in self.test_cases if c.id in test_cases]
        
        print(f"开始运行 {len(cases_to_run)} 个基准测试用例...")
        
        # 运行测试
        for i, case in enumerate(cases_to_run, 1):
            print(f"\n[{i}/{len(cases_to_run)}] 运行测试: {case.name} ({case.id})")
            result = await self._run_single_test(case, executor_func)
            self.results.append(result)
            
            # 显示结果
            status = "✓ 成功" if result.success else "✗ 失败"
            print(f"  {status} - 耗时: {result.execution_time:.2f}s, "
                  f"准确性: {result.accuracy_score:.1f}, "
                  f"性能: {result.performance_score:.1f}")
            
            if not result.success:
                print(f"  错误: {result.error_message}")
        
        total_time = time.time() - start_time
        
        # 生成报告
        report = self._generate_report(total_time)
        
        # 保存报告
        self._save_report(report)
        
        return report
    
    async def _run_single_test(
        self,
        case: BenchmarkCase,
        executor_func
    ) -> BenchmarkResult:
        """运行单个测试用例"""
        start_time = time.time()
        
        try:
            # 执行任务
            result = await asyncio.wait_for(
                executor_func(case.instruction, case.parameters),
                timeout=case.timeout
            )
            
            execution_time = time.time() - start_time
            
            # 验证输出
            accuracy_score = self._calculate_accuracy(
                result, case.expected_output
            )
            
            # 计算性能分数（基于执行时间）
            performance_score = self._calculate_performance(
                execution_time, case.timeout
            )
            
            return BenchmarkResult(
                case_id=case.id,
                success=True,
                execution_time=execution_time,
                accuracy_score=accuracy_score,
                performance_score=performance_score,
                output=result
            )
        
        except asyncio.TimeoutError:
            return BenchmarkResult(
                case_id=case.id,
                success=False,
                execution_time=case.timeout,
                accuracy_score=0.0,
                performance_score=0.0,
                error_message=f"超时 (>{case.timeout}s)"
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            return BenchmarkResult(
                case_id=case.id,
                success=False,
                execution_time=execution_time,
                accuracy_score=0.0,
                performance_score=0.0,
                error_message=str(e)
            )
    
    def _calculate_accuracy(
        self,
        actual_output: Dict,
        expected_output: Dict
    ) -> float:
        """计算准确性分数"""
        if not actual_output:
            return 0.0
        
        score = 0.0
        total_checks = len(expected_output)
        
        for key, expected_value in expected_output.items():
            if key == "has_video":
                output_path = actual_output.get("output_path", "")
                if output_path and Path(output_path).exists():
                    score += 100.0 / total_checks
            
            elif key == "has_srt":
                srt_path = actual_output.get("srt_path", "")
                if srt_path and Path(srt_path).exists():
                    score += 100.0 / total_checks
            
            elif key == "min_subtitle_count":
                srt_path = actual_output.get("srt_path", "")
                if srt_path and Path(srt_path).exists():
                    with open(srt_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    count = content.count('\n\n')
                    if count >= expected_value:
                        score += 100.0 / total_checks
            
            elif key == "used_llm":
                if actual_output.get("used_llm_correction") == expected_value:
                    score += 100.0 / total_checks
            
            elif key == "has_output":
                output_path = actual_output.get("output_path", "")
                if output_path and Path(output_path).exists():
                    score += 100.0 / total_checks
            
            elif key == "format":
                output_path = actual_output.get("output_path", "")
                if output_path and output_path.endswith(f".{expected_value}"):
                    score += 100.0 / total_checks
        
        return score
    
    def _calculate_performance(
        self,
        execution_time: float,
        timeout: float
    ) -> float:
        """计算性能分数"""
        # 性能分数基于执行时间相对于超时时间的比例
        # 执行时间越短，分数越高
        ratio = execution_time / timeout
        
        if ratio <= 0.2:  # 非常快，<20%
            return 100.0
        elif ratio <= 0.4:  # 快，20-40%
            return 90.0
        elif ratio <= 0.6:  # 正常，40-60%
            return 70.0
        elif ratio <= 0.8:  # 慢，60-80%
            return 50.0
        else:  # 很慢，>80%
            return 30.0
    
    def _generate_report(self, total_time: float) -> Dict:
        """生成测试报告"""
        total_cases = len(self.results)
        successful_cases = sum(1 for r in self.results if r.success)
        failed_cases = total_cases - successful_cases
        
        avg_accuracy = sum(r.accuracy_score for r in self.results) / total_cases if total_cases > 0 else 0
        avg_performance = sum(r.performance_score for r in self.results) / total_cases if total_cases > 0 else 0
        avg_execution_time = sum(r.execution_time for r in self.results) / total_cases if total_cases > 0 else 0
        
        # 按任务类型统计
        by_task_type = {}
        for i, result in enumerate(self.results):
            case = self.test_cases[i]
            task_type = case.task_type
            
            if task_type not in by_task_type:
                by_task_type[task_type] = {
                    "total": 0,
                    "success": 0,
                    "avg_accuracy": 0,
                    "avg_performance": 0
                }
            
            by_task_type[task_type]["total"] += 1
            if result.success:
                by_task_type[task_type]["success"] += 1
            by_task_type[task_type]["avg_accuracy"] += result.accuracy_score
            by_task_type[task_type]["avg_performance"] += result.performance_score
        
        # 计算平均值
        for stats in by_task_type.values():
            total = stats["total"]
            stats["avg_accuracy"] /= total
            stats["avg_performance"] /= total
            stats["success_rate"] = (stats["success"] / total * 100) if total > 0 else 0
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_cases": total_cases,
                "successful": successful_cases,
                "failed": failed_cases,
                "success_rate": (successful_cases / total_cases * 100) if total_cases > 0 else 0,
                "total_time": total_time,
                "avg_execution_time": avg_execution_time,
                "avg_accuracy_score": avg_accuracy,
                "avg_performance_score": avg_performance
            },
            "by_task_type": by_task_type,
            "detailed_results": [
                {
                    "case_id": r.case_id,
                    "case_name": self.test_cases[i].name,
                    "task_type": self.test_cases[i].task_type,
                    "success": r.success,
                    "execution_time": r.execution_time,
                    "accuracy_score": r.accuracy_score,
                    "performance_score": r.performance_score,
                    "error": r.error_message
                }
                for i, r in enumerate(self.results)
            ]
        }
        
        return report
    
    def _save_report(self, report: Dict):
        """保存报告"""
        report_dir = Path("logs/benchmarks")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"benchmark_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n基准测试报告已保存: {report_file}")


# 使用示例
async def example_executor(instruction: str, parameters: Dict) -> Dict:
    """示例执行器（需要替换为实际的执行函数）"""
    # 模拟执行
    await asyncio.sleep(2)
    
    return {
        "output_path": "test-output.mp4",
        "srt_path": "test.srt",
        "used_llm_correction": parameters.get("use_llm_correction", False)
    }


async def main():
    """主函数"""
    benchmark = BenchmarkSuite()
    
    # 运行基准测试
    report = await benchmark.run_benchmark(example_executor)
    
    # 显示摘要
    print("\n" + "="*60)
    print("基准测试报告摘要")
    print("="*60)
    summary = report["summary"]
    print(f"总用例数: {summary['total_cases']}")
    print(f"成功: {summary['successful']} | 失败: {summary['failed']}")
    print(f"成功率: {summary['success_rate']:.1f}%")
    print(f"平均执行时间: {summary['avg_execution_time']:.2f}s")
    print(f"平均准确性分数: {summary['avg_accuracy_score']:.1f}")
    print(f"平均性能分数: {summary['avg_performance_score']:.1f}")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
