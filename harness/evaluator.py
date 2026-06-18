"""
评估器 (Evaluator)
-------------------
负责:
  - 从 CoT 推理输出中提取最终答案
  - 将提取答案与标准答案对比，计算正确率
  - 支持多种答案提取模式
  - 生成评估统计报告

CoT 推理的输出通常很长，包含大量中间推理步骤。
评估器需要从中精确提取形如 "The answer is: X" 的最终答案。
"""

import re
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("harness.evaluator")


@dataclass
class EvalResult:
    """单个样本的评估结果"""
    sample_id: str = ""
    question: str = ""
    ground_truth: str = ""
    raw_output: str = ""            # 模型原始输出
    predicted_answer: str = ""      # 提取后的预测答案
    is_correct: bool = False
    extraction_method: str = ""     # 使用的答案提取方法
    error_info: str = ""            # 错误信息 (如有)


@dataclass
class EvalReport:
    """整体评估报告"""
    total_samples: int = 0
    correct_count: int = 0
    accuracy: float = 0.0
    extraction_failures: int = 0    # 答案提取失败的样本数
    per_sample_results: List[EvalResult] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"EvalReport(total={self.total_samples}, "
            f"correct={self.correct_count}, "
            f"accuracy={self.accuracy:.2%}, "
            f"extraction_failures={self.extraction_failures})"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "correct_count": self.correct_count,
            "accuracy": self.accuracy,
            "extraction_failures": self.extraction_failures,
        }


# ---------- 答案规范化 ----------

def normalize_answer(answer: str) -> str:
    """
    规范化答案字符串。

    处理流程:
    1. 去除首尾空白
    2. 转大写
    3. 去除末尾的句号
    4. 提取括号内的字母 (如 "(A)" -> "A")
    5. 去除多余引号

    Args:
        answer: 原始答案字符串

    Returns:
        规范化后的答案
    """
    if not answer:
        return ""

    answer = answer.strip().upper()
    # 去掉首尾的句号
    answer = answer.strip(".")

    # 如果答案被括号包裹，提取括号内的内容: "(A)" -> "A"
    bracket_match = re.match(r"\(([A-E])\)", answer)
    if bracket_match:
        return bracket_match.group(1)

    # 如果答案以 "OPTION " 开头，去掉前缀: "OPTION A" -> "A"
    if answer.startswith("OPTION "):
        answer = answer.replace("OPTION ", "").strip()

    # 去掉引号
    answer = answer.strip("\"'")

    # 如果答案太长（超过20个字符），可能不是直接的选项
    # 尝试从中提取选项字母
    if len(answer) > 20:
        simple_match = re.search(r"\b([A-E])\b", answer)
        if simple_match:
            return simple_match.group(1)

    return answer


# ---------- 答案提取模式 ----------

# 多种答案提取正则模式（按优先级排列）
ANSWER_PATTERNS = [
    # 模式 1: "The answer is X" 或 "answer is X"
    (r"(?:the\s+)?answer\s+(?:is|:)\s*\(?([A-E])\)?", "answer_is"),

    # 模式 2: "Therefore, the answer is X"
    (r"(?:therefore|thus|so|hence)[,\s]+(?:the\s+)?answer\s+(?:is|:)\s*\(?([A-E])\)?", "therefore_answer_is"),

    # 模式 3: "I choose X" 或 "I select X"
    (r"(?:i\s+)?(?:choose|select|pick)\s+\(?([A-E])\)?", "choose_select"),

    # 模式 4: "The correct option is X"
    (r"(?:the\s+)?correct\s+(?:option|choice|answer)\s+(?:is|:)\s*\(?([A-E])\)?", "correct_option"),

    # 模式 5: 行首孤立的大写字母 (最后手段)
    (r"^\(?([A-E])\)?[\.\s]*$", "isolated_letter"),

    # 模式 6: 中文答案格式 "答案是：X" 或 "答案为 X"
    (r"(?:答案|正确选项)(?:是|为|：|:)\s*\(?([A-E])\)?", "chinese_answer"),

    # 模式 7: "#### X" (Markdown 风格的最终答案)
    (r"#{1,4}\s*(?:answer|答案)?[:：]?\s*\(?([A-E])\)?", "markdown_answer"),

    # 模式 8: "\boxed{X}" (LaTeX 风格)
    (r"\\boxed\{([A-E])\}", "latex_boxed"),
]


def extract_final_answer(raw_output: str, verbose: bool = False) -> Tuple[str, str]:
    """
    从 CoT 推理输出中提取最终答案。

    采用多模式级联匹配策略:
    1. 按优先级依次尝试多种正则模式
    2. 首个匹配成功的模式返回结果
    3. 如果所有模式都失败，返回空字符串

    Args:
        raw_output: 模型原始输出文本
        verbose: 是否打印详细提取过程

    Returns:
        (提取的答案, 匹配模式名称) 或 ("", "none")
    """
    if not raw_output:
        return "", "none"

    # 将文本规范化为单行以便正则匹配（保留原有换行用于某些模式）
    text_single_line = " ".join(raw_output.split())

    for pattern, method_name in ANSWER_PATTERNS:
        # 在原始文本（多行）和单行文本中都尝试匹配
        for text in [raw_output, text_single_line]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted = normalize_answer(match.group(1))
                if extracted and extracted in "ABCDE":
                    if verbose:
                        logger.info(
                            f"答案提取成功: '{extracted}' "
                            f"（方法: {method_name}, "
                            f"匹配文本: ...{match.group(0)}...）"
                        )
                    return extracted, method_name

    # 所有模式都失败
    if verbose:
        logger.warning(f"答案提取失败，原始输出前200字符: {raw_output[:200]}...")
    return "", "none"


def quick_extract(raw_output: str) -> str:
    """快速提取答案（只返回答案字符串）"""
    answer, _ = extract_final_answer(raw_output)
    return answer


# ---------- 评估器类 ----------

class Evaluator:
    """
    评估器 —— 答案提取 + 正确率计算。

    使用方式:
        evaluator = Evaluator()
        report = evaluator.evaluate(predictions, ground_truths)
        print(f"Accuracy: {report.accuracy:.2%}")
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def evaluate_single(
        self,
        raw_output: str,
        ground_truth: str,
        sample_id: str = "",
        question: str = "",
    ) -> EvalResult:
        """
        评估单个样本。

        Args:
            raw_output: 模型原始输出
            ground_truth: 标准答案
            sample_id: 样本 ID
            question: 问题文本

        Returns:
            EvalResult 对象
        """
        predicted, method = extract_final_answer(raw_output, verbose=self.verbose)

        gt_normalized = normalize_answer(ground_truth)
        is_correct = (predicted == gt_normalized) if predicted else False

        result = EvalResult(
            sample_id=sample_id,
            question=question,
            ground_truth=gt_normalized,
            raw_output=raw_output,
            predicted_answer=predicted,
            is_correct=is_correct,
            extraction_method=method,
            error_info="" if predicted else "答案提取失败",
        )

        if self.verbose and not result.is_correct:
            logger.info(
                f"样本 {sample_id}: 预测={predicted or '(空)'}, "
                f"标准答案={gt_normalized}, "
                f"正确={is_correct}"
            )

        return result

    def evaluate(
        self,
        raw_outputs: List[str],
        ground_truths: List[str],
        sample_ids: Optional[List[str]] = None,
        questions: Optional[List[str]] = None,
    ) -> EvalReport:
        """
        批量评估。

        Args:
            raw_outputs: 模型原始输出列表
            ground_truths: 标准答案列表
            sample_ids: 样本 ID 列表 (可选)
            questions: 问题文本列表 (可选)

        Returns:
            EvalReport 评估报告
        """
        if len(raw_outputs) != len(ground_truths):
            raise ValueError(
                f"raw_outputs 和 ground_truths 长度不匹配: "
                f"{len(raw_outputs)} vs {len(ground_truths)}"
            )

        n = len(raw_outputs)
        if sample_ids is None:
            sample_ids = [f"sample_{i:04d}" for i in range(n)]
        if questions is None:
            questions = [""] * n

        results = []
        correct_count = 0
        extraction_failures = 0

        for i in range(n):
            result = self.evaluate_single(
                raw_output=raw_outputs[i],
                ground_truth=ground_truths[i],
                sample_id=sample_ids[i],
                question=questions[i],
            )
            results.append(result)

            if result.is_correct:
                correct_count += 1
            if not result.predicted_answer:
                extraction_failures += 1

        accuracy = correct_count / n if n > 0 else 0.0

        report = EvalReport(
            total_samples=n,
            correct_count=correct_count,
            accuracy=accuracy,
            extraction_failures=extraction_failures,
            per_sample_results=results,
        )

        logger.info(f"评估完成: {report}")
        return report


# ---------- 辅助：直接从 CoT Trace 评估 ----------

def evaluate_from_traces(
    traces: List[Any],  # CoTTrace 列表
    ground_truths: List[str],
) -> EvalReport:
    """
    从 CoTTrace 对象列表直接评估。
    方便组员使用了 solve_with_trace 后直接计算准确率。

    Args:
        traces: CoTTrace 对象列表
        ground_truths: 标准答案列表

    Returns:
        EvalReport
    """
    evaluator = Evaluator()
    raw_outputs = [t.final_answer for t in traces]
    sample_ids = [f"trace_{i:04d}" for i in range(len(traces))]
    questions = [t.question for t in traces]

    return evaluator.evaluate(raw_outputs, ground_truths, sample_ids, questions)
