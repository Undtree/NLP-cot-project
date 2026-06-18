"""
Harness Engineering Framework
==============================
核心评测与管理框架，为 CoT (Chain-of-Thought) Agentic 任务提供统一的：
  - 数据集加载
  - 云端 LLM 客户端（带重试与并发控制）
  - 任务抽象基类
  - 答案提取与正确率评估
  - 实验日志与结果导出

使用方式:
    from harness import (
        load_dataset,
        LLMClient,
        BaseTask,
        Evaluator,
        ExperimentLogger,
    )
"""

from .dataset import load_dataset, AQuADataset
from .llm_client import LLMClient
from .base_task import BaseTask
from .evaluator import Evaluator, extract_final_answer, normalize_answer
from .logger import ExperimentLogger

__all__ = [
    "load_dataset",
    "AQuADataset",
    "LLMClient",
    "BaseTask",
    "Evaluator",
    "extract_final_answer",
    "normalize_answer",
    "ExperimentLogger",
]
