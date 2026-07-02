"""
Harness Engineering Framework
==============================
核心评测与管理框架，为 CoT (Chain-of-Thought) Agentic 任务提供统一的：
  - 数据集加载（AQuA / HotpotQA）
  - 云端 LLM 客户端（带重试与并发控制）
  - 任务抽象基类（支持 Prompt 模板化 + Agentic 推理 + Sample 接口）
  - 结构化 Prompt 模板引擎
  - 检索模块（BM25）
  - ReAct 状态机解析器
  - 答案提取与正确率评估（字母匹配 / QA 指标）
  - 实验日志与结果导出

使用方式:
    from harness import (
        Sample,
        load_dataset,
        SimpleBM25,
        LLMClient,
        BaseTask,
        PromptTemplate,
        ToolDefinition,
        StateMachineParser,
        Evaluator,
        QAMatchEvaluator,
        ExperimentLogger,
    )
"""

from .sample import Sample
from .dataset import load_dataset, AQuADataset, load_hotpotqa, build_corpus_from_samples
from .retrieval import SimpleBM25, merge_unique_passages, format_passages
from .llm_client import LLMClient
from .base_task import BaseTask, CoTTrace
from .prompt_template import (
    PromptTemplate,
    ToolDefinition,
    ToolParameter,
    ConversationHistory,
    MATH_COT_TEMPLATE,
    REACT_AGENT_TEMPLATE,
)
from .state_machine import (
    StateMachineParser,
    ReActParser,
    XMLTagParser,
    ParsedTrace,
    ParsedBlock,
    BlockType,
)
from .evaluator import (
    Evaluator,
    extract_final_answer,
    normalize_answer,
    QAMatchEvaluator,
    QAEvalReport,
    qa_exact_match,
    qa_token_f1,
    qa_title_recall,
)
from .logger import ExperimentLogger

__all__ = [
    # 样本
    "Sample",
    # 数据集
    "load_dataset",
    "AQuADataset",
    "load_hotpotqa",
    "build_corpus_from_samples",
    # 检索
    "SimpleBM25",
    "merge_unique_passages",
    "format_passages",
    # LLM 客户端
    "LLMClient",
    # 任务基类
    "BaseTask",
    "CoTTrace",
    # Prompt 模板引擎
    "PromptTemplate",
    "ToolDefinition",
    "ToolParameter",
    "ConversationHistory",
    "MATH_COT_TEMPLATE",
    "REACT_AGENT_TEMPLATE",
    # 状态机解析器
    "StateMachineParser",
    "ReActParser",
    "XMLTagParser",
    "ParsedTrace",
    "ParsedBlock",
    "BlockType",
    # 评估器
    "Evaluator",
    "extract_final_answer",
    "normalize_answer",
    "QAMatchEvaluator",
    "QAEvalReport",
    "qa_exact_match",
    "qa_token_f1",
    "qa_title_recall",
    # 日志
    "ExperimentLogger",
]
