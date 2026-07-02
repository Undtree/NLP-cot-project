"""
评估与日志系统 (Logger)
------------------------
负责:
  - 将实验结果导出为 .jsonl 和 .csv 格式
  - 记录每个样本的详细信息：问题、标准答案、推理路径、最终答案、对错
  - 生成汇总统计报告
  - 支持追加模式，避免实验中断导致数据丢失
"""

import json
import csv
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

from .evaluator import EvalReport, EvalResult

logger = logging.getLogger("harness.logger")


@dataclass
class ExperimentRun:
    """
    一次完整的实验运行记录。

    包含:
    - 实验元数据（策略名称、时间戳、配置等）
    - 每个样本的详细结果
    - 汇总统计
    """
    run_id: str = ""
    strategy_name: str = ""
    timestamp: str = ""
    dataset_name: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    total_samples: int = 0
    correct_count: int = 0
    accuracy: float = 0.0
    extraction_failures: int = 0
    per_sample_results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperimentLogger:
    """
    实验日志记录器。

    功能:
    - 将每次实验运行保存为 JSONL (每行一个样本的完整信息)
    - 同时导出 CSV (方便在 Excel 中快速查看)
    - 生成 JSON 格式的汇总报告
    - 支持追加写入（避免实验中断丢失数据）

    使用方式:
        logger = ExperimentLogger(results_dir="results")
        logger.log_run(eval_report, strategy_name, config, dataset_name)
    """

    def __init__(self, results_dir: str = "results"):
        """
        Args:
            results_dir: 实验结果保存目录
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def _generate_run_id(self) -> str:
        """生成唯一的运行 ID"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def log_run(
        self,
        eval_report: EvalReport,
        strategy_name: str,
        config: Optional[Dict[str, Any]] = None,
        dataset_name: str = "aqua",
        traces: Optional[List[Any]] = None,
    ) -> str:
        """
        记录一次完整的实验运行。

        Args:
            eval_report: 评估报告
            strategy_name: 策略名称
            config: 策略配置
            dataset_name: 数据集名称
            traces: CoT 推理路径追踪列表 (可选)

        Returns:
            运行 ID 字符串
        """
        run_id = self._generate_run_id()
        timestamp = datetime.now().isoformat()

        # 构建逐样本结果
        per_sample = []
        for i, result in enumerate(eval_report.per_sample_results):
            sample_record = {
                "sample_id": result.sample_id,
                "question": result.question[:500],  # 截断过长问题
                "ground_truth": result.ground_truth,
                "predicted_answer": result.predicted_answer,
                "is_correct": result.is_correct,
                "raw_output_truncated": result.raw_output[:2000],
                "extraction_method": result.extraction_method,
                "error_info": result.error_info,
            }

            # 如果有 trace，附加上推理路径
            if traces and i < len(traces):
                trace = traces[i]
                sample_record["reasoning_steps"] = trace.reasoning_steps
                sample_record["tool_calls"] = trace.tool_calls
                sample_record["total_tokens"] = trace.total_tokens
                sample_record["total_time_seconds"] = trace.total_time_seconds

            per_sample.append(sample_record)

        # 构建运行记录
        run = ExperimentRun(
            run_id=run_id,
            strategy_name=strategy_name,
            timestamp=timestamp,
            dataset_name=dataset_name,
            config=config or {},
            total_samples=eval_report.total_samples,
            correct_count=eval_report.correct_count,
            accuracy=eval_report.accuracy,
            extraction_failures=eval_report.extraction_failures,
            per_sample_results=per_sample,
        )

        # 1. 保存为 JSONL（每行一个样本的详细信息）
        jsonl_path = os.path.join(self.results_dir, f"run_{run_id}_{strategy_name}.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for sample in per_sample:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        # 2. 保存汇总 JSON
        summary_path = os.path.join(self.results_dir, f"run_{run_id}_{strategy_name}_summary.json")
        summary_data = {
            "run_id": run_id,
            "strategy_name": strategy_name,
            "timestamp": timestamp,
            "dataset_name": dataset_name,
            "config": config or {},
            "total_samples": eval_report.total_samples,
            "correct_count": eval_report.correct_count,
            "accuracy": eval_report.accuracy,
            "extraction_failures": eval_report.extraction_failures,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)

        # 3. 保存为 CSV（方便 Excel 查看）
        csv_path = os.path.join(self.results_dir, f"run_{run_id}_{strategy_name}.csv")
        if per_sample:
            # 展平 CSV 列
            csv_columns = [
                "sample_id", "question", "ground_truth",
                "predicted_answer", "is_correct", "extraction_method",
            ]
            with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(per_sample)

        # 4. 追加到全局汇总表
        self._append_to_global_summary(summary_data)

        logger.info(
            f"实验运行记录已保存:\n"
            f"  JSONL: {jsonl_path}\n"
            f"  JSON:  {summary_path}\n"
            f"  CSV:   {csv_path}"
        )

        return run_id

    def log_run_qa(
        self,
        eval_report,  # QAEvalReport
        strategy_name: str,
        config: Optional[Dict[str, Any]] = None,
        dataset_name: str = "hotpotqa",
        traces: Optional[List[Any]] = None,
    ) -> str:
        """
        记录一次 QA 实验运行（HotpotQA 等自由文本 QA 任务）。

        Args:
            eval_report: QAEvalReport 对象
            strategy_name: 策略名称
            config: 策略配置
            dataset_name: 数据集名称
            traces: CoT 推理路径追踪列表 (可选)

        Returns:
            运行 ID 字符串
        """
        run_id = self._generate_run_id()
        timestamp = datetime.now().isoformat()

        # 构建逐样本结果
        per_sample = []
        for i, result in enumerate(eval_report.per_sample_results):
            sample_record = {
                "sample_id": result.sample_id,
                "question": result.question[:500],
                "ground_truth": result.ground_truth,
                "prediction": result.prediction,
                "em": result.em,
                "f1": result.f1,
                "title_recall": result.title_recall,
                "retrieved_count": result.retrieved_count,
                "llm_calls": result.llm_calls,
            }

            if traces and i < len(traces):
                trace = traces[i]
                sample_record["reasoning_steps"] = trace.reasoning_steps
                sample_record["total_time_seconds"] = trace.total_time_seconds

            per_sample.append(sample_record)

        # 1. JSONL
        jsonl_path = os.path.join(self.results_dir, f"run_{run_id}_{strategy_name}.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for sample in per_sample:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        # 2. 汇总 JSON
        summary_path = os.path.join(self.results_dir, f"run_{run_id}_{strategy_name}_summary.json")
        summary_data = {
            "run_id": run_id,
            "strategy_name": strategy_name,
            "timestamp": timestamp,
            "dataset_name": dataset_name,
            "config": config or {},
            **eval_report.to_dict(),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)

        # 3. CSV
        csv_path = os.path.join(self.results_dir, f"run_{run_id}_{strategy_name}.csv")
        if per_sample:
            csv_columns = [
                "sample_id", "question", "ground_truth", "prediction",
                "em", "f1", "title_recall", "retrieved_count", "llm_calls",
            ]
            with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(per_sample)

        # 4. 全局汇总
        self._append_to_global_summary(summary_data)

        logger.info(
            f"QA 实验运行记录已保存:\n"
            f"  JSONL: {jsonl_path}\n"
            f"  JSON:  {summary_path}\n"
            f"  CSV:   {csv_path}"
        )

        return run_id

    def _append_to_global_summary(self, summary: Dict[str, Any]):
        """
        追加到全局汇总 JSONL 文件。
        所有实验运行汇总在一行，方便横向对比。
        """
        global_path = os.path.join(self.results_dir, "all_runs_summary.jsonl")
        with open(global_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    def load_run(self, run_id: str, strategy_name: str) -> Optional[ExperimentRun]:
        """
        从磁盘加载指定实验运行。

        Args:
            run_id: 运行 ID
            strategy_name: 策略名称

        Returns:
            ExperimentRun 或 None
        """
        summary_path = os.path.join(
            self.results_dir, f"run_{run_id}_{strategy_name}_summary.json"
        )
        if not os.path.exists(summary_path):
            return None

        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 加载逐样本数据
        jsonl_path = os.path.join(
            self.results_dir, f"run_{run_id}_{strategy_name}.jsonl"
        )
        per_sample = []
        if os.path.exists(jsonl_path):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        per_sample.append(json.loads(line))

        return ExperimentRun(
            run_id=run_id,
            strategy_name=data.get("strategy_name", strategy_name),
            timestamp=data.get("timestamp", ""),
            dataset_name=data.get("dataset_name", ""),
            config=data.get("config", {}),
            total_samples=data.get("total_samples", 0),
            correct_count=data.get("correct_count", 0),
            accuracy=data.get("accuracy", 0.0),
            extraction_failures=data.get("extraction_failures", 0),
            per_sample_results=per_sample,
        )

    def compare_runs(self, strategy_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        读取全局汇总表，返回所有运行记录的列表。
        可用于比较不同策略的表现。

        Args:
            strategy_names: 要筛选的策略名称列表（可选，不指定则返回全部）

        Returns:
            运行记录列表
        """
        global_path = os.path.join(self.results_dir, "all_runs_summary.jsonl")
        if not os.path.exists(global_path):
            return []

        runs = []
        with open(global_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    if strategy_names is None or record.get("strategy_name") in strategy_names:
                        runs.append(record)
        return runs

    def print_comparison_table(self):
        """打印所有实验运行的对比表"""
        runs = self.compare_runs()
        if not runs:
            print("暂无实验记录。")
            return

        print(f"\n{'='*80}")
        print(f"{'策略名称':<30} {'准确率':>8} {'正确/总数':>12} {'提取失败':>8} {'时间':<20}")
        print(f"{'-'*80}")
        for r in runs:
            name = r.get("strategy_name", "unknown")[:28]
            acc = r.get("accuracy", 0.0)
            correct = r.get("correct_count", 0)
            total = r.get("total_samples", 0)
            failures = r.get("extraction_failures", 0)
            ts = r.get("timestamp", "")[:19]
            print(f"{name:<30} {acc:>7.2%} {correct:>4}/{total:<7} {failures:>8} {ts:<20}")
        print(f"{'='*80}\n")
