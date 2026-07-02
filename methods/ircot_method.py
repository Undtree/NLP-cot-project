"""
IRCoT: Interleaving Retrieval with Chain-of-Thought
====================================================
基于检索增强的交错式思维链推理策略。

参考论文: "Interleaving Retrieval with Chain-of-Thought Reasoning
for Knowledge-Intensive Multi-Step Questions" (Trivedi et al., 2023)

本模块实现三个可注册到 harness 的策略:
1. NoRetrievalCoT       : 纯 CoT，不使用检索（HotpotQA baseline）
2. OneStepRAGCoT        : 一次性检索 + CoT
3. IRCoTTask            : 完整 IRCoT（推理与检索交替进行）

所有策略继承 BaseTask，通过 solve_sample_with_trace 访问 HotpotQA 的
context 和 supporting_titles 元数据。

使用方式:
    from methods.ircot_method import IRCoTTask
    from harness.retrieval import SimpleBM25

    task = IRCoTTask(client, retriever=bm25_retriever, max_steps=3)
    trace = task.solve_sample_with_trace(sample)
"""

from __future__ import annotations

import re
import time
from typing import Dict, List, Optional, Tuple

from harness.base_task import BaseTask, CoTTrace
from harness.llm_client import LLMClient
from harness.retrieval import SimpleBM25, format_passages, merge_unique_passages
from harness.evaluator import qa_exact_match, qa_token_f1, qa_title_recall


# ============================================================
# Prompt 模板（移植自 IRCoT src/prompts.py）
# ============================================================

SYSTEM_PROMPT = (
    "You are a careful question answering assistant. "
    "Use retrieved evidence when it is provided. "
    "End the final response with exactly: So the answer is: <answer>"
)

STEP_SYSTEM_PROMPT = (
    "You are helping an iterative retrieval system. "
    "For intermediate steps, do not give the final answer. "
    "Write only one concise intermediate clue, bridge fact, or search query "
    "that can help retrieve more evidence."
)


def _cot_prompt(question: str) -> str:
    return (
        f"Question: {question}\n\n"
        "Think step by step. End with exactly one final line:\n"
        "So the answer is: <answer>"
    )


def _one_step_rag_prompt(question: str, passages: List[Dict[str, str]]) -> str:
    return (
        "Retrieved evidence:\n"
        f"{format_passages(passages)}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using the evidence. Think step by step, "
        "then end with exactly one final line:\n"
        "So the answer is: <answer>"
    )


def _reason_next_prompt(
    question: str,
    passages: List[Dict[str, str]],
    chain: List[str],
) -> str:
    chain_text = "\n".join(
        f"{i + 1}. {step}" for i, step in enumerate(chain)
    ) or "(none yet)"
    return (
        "Retrieved evidence:\n"
        f"{format_passages(passages)}\n\n"
        f"Question: {question}\n\n"
        f"Reasoning so far:\n{chain_text}\n\n"
        "Generate only the next intermediate retrieval clue. "
        "Do not answer the original question yet. "
        "Do not use the phrase 'So the answer is'. "
        "Return exactly one short sentence or query that names the next entity, "
        "relationship, or fact that should be retrieved."
    )


def _final_reader_prompt(
    question: str,
    passages: List[Dict[str, str]],
    chain: List[str],
) -> str:
    chain_text = "\n".join(
        f"{i + 1}. {step}" for i, step in enumerate(chain)
    ) or "(none)"
    return (
        "Retrieved evidence:\n"
        f"{format_passages(passages)}\n\n"
        f"Question: {question}\n\n"
        f"Intermediate reasoning:\n{chain_text}\n\n"
        "Now give the final answer. Keep it concise and end with exactly:\n"
        "So the answer is: <answer>"
    )


# ============================================================
# 答案解析
# ============================================================

_ANSWER_RE = re.compile(
    r"so\s+the\s+answer\s+is\s*[:：]\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)


def _parse_final_answer(text: str) -> str:
    """从模型输出中提取 'So the answer is: ...' 之后的文本"""
    matches = _ANSWER_RE.findall(text)
    if matches:
        answer = matches[-1].strip()
    else:
        answer = text.strip()
    answer = answer.split("\n")[0].strip()
    return answer.rstrip(" .")


def _first_reasoning_sentence(text: str) -> str:
    """提取文本的第一句推理"""
    text = text.strip()
    answer_match = _ANSWER_RE.search(text)
    if answer_match:
        return text[:answer_match.end()].strip()
    parts = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    return parts[0].strip() if parts and parts[0].strip() else text


def _clean_retrieval_step(text: str) -> str:
    """将 LLM 中间输出转化为检索查询"""
    text = text.strip()
    text = _ANSWER_RE.sub(r"\1", text).strip()
    text = re.sub(
        r"^(next\s+)?(search\s+)?(query|clue|step)\s*[:：-]\s*",
        "", text, flags=re.IGNORECASE,
    )
    text = _first_reasoning_sentence(text)
    return text.strip().strip('"').rstrip(" .")


# ============================================================
# IRCoT 策略实现
# ============================================================

class NoRetrievalCoT(BaseTask):
    """HotpotQA 纯 CoT baseline（不使用检索）。

    使用与 IRCoT 相同的 prompt 格式以便公平对比。
    """

    def __init__(
        self,
        client: LLMClient,
        temperature: float = 0.0,
        max_tokens: int = 512,
        name: Optional[str] = None,
    ):
        super().__init__(
            client=client,
            name=name or "NoRetrievalCoT",
            system_prompt=SYSTEM_PROMPT,
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    def solve(self, question: str) -> str:
        response = self.client.chat(
            user_message=_cot_prompt(question),
            system_message=self.system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return _parse_final_answer(response.content)

    def solve_sample_with_trace(self, sample) -> CoTTrace:
        trace = CoTTrace(question=sample.question)
        start_time = time.time()

        response = self.client.chat(
            user_message=_cot_prompt(sample.question),
            system_message=self.system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        pred = _parse_final_answer(response.content)
        gold = str(sample.ground_truth)

        trace.final_answer = pred
        trace.total_tokens = response.usage.get("total_tokens", 0)
        trace.total_time_seconds = time.time() - start_time
        trace.intermediate_responses = [response.content]
        trace.metadata.update({
            "method": "no_retrieval",
            "prediction": pred,
            "gold": gold,
            "em": qa_exact_match(pred, gold),
            "f1": qa_token_f1(pred, gold),
            "title_recall": 0.0,
            "retrieved_count": 0,
            "llm_calls": 1,
            "retrieved": [],
            "reasoning_steps": [],
        })
        return trace

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "method": "no_retrieval",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        })
        return config


class OneStepRAGCoT(BaseTask):
    """一次性检索 + CoT（RAG baseline）。

    先用问题检索 top-k 段落，再让模型基于段落推理。
    """

    def __init__(
        self,
        client: LLMClient,
        retriever: SimpleBM25,
        top_k: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 512,
        name: Optional[str] = None,
    ):
        super().__init__(
            client=client,
            name=name or "OneStepRAGCoT",
            system_prompt=SYSTEM_PROMPT,
        )
        self.retriever = retriever
        self.top_k = top_k
        self.temperature = temperature
        self.max_tokens = max_tokens

    def solve(self, question: str) -> str:
        passages = self.retriever.search(question, top_k=self.top_k)
        response = self.client.chat(
            user_message=_one_step_rag_prompt(question, passages),
            system_message=self.system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return _parse_final_answer(response.content)

    def solve_sample_with_trace(self, sample) -> CoTTrace:
        trace = CoTTrace(question=sample.question)
        start_time = time.time()

        passages = self.retriever.search(sample.question, top_k=self.top_k)
        response = self.client.chat(
            user_message=_one_step_rag_prompt(sample.question, passages),
            system_message=self.system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        pred = _parse_final_answer(response.content)
        gold = str(sample.ground_truth)

        trace.final_answer = pred
        trace.total_tokens = response.usage.get("total_tokens", 0)
        trace.total_time_seconds = time.time() - start_time
        trace.intermediate_responses = [response.content]
        trace.metadata.update({
            "method": "one_step",
            "prediction": pred,
            "gold": gold,
            "em": qa_exact_match(pred, gold),
            "f1": qa_token_f1(pred, gold),
            "title_recall": qa_title_recall(passages, sample.supporting_titles),
            "retrieved_count": len(passages),
            "llm_calls": 1,
            "retrieved": passages,
            "reasoning_steps": [],
        })
        return trace

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "method": "one_step_rag",
            "top_k": self.top_k,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        })
        return config


class IRCoTTask(BaseTask):
    """完整 IRCoT 策略：推理与检索交替进行。

    算法流程:
    1. 用问题做初始 BM25 检索
    2. 进入 max_steps 轮交错循环:
       a. 用当前证据和推理链生成下一步检索线索
       b. 用线索再次检索
       c. 去重合并段落
    3. 最终 reader 综合全部证据和推理链输出答案

    Args:
        client: LLMClient 实例
        retriever: SimpleBM25 检索器（需预先对语料建索引）
        max_steps: 最大推理-检索轮数（默认 3）
        top_k: 每次检索返回的段落数（默认 5）
        max_context_paragraphs: 累积上下文段落上限（默认 15）
        temperature: 生成温度
        max_tokens: 最终答案生成的最大 token 数
    """

    def __init__(
        self,
        client: LLMClient,
        retriever: SimpleBM25,
        max_steps: int = 3,
        top_k: int = 5,
        max_context_paragraphs: int = 15,
        temperature: float = 0.0,
        max_tokens: int = 512,
        name: Optional[str] = None,
    ):
        super().__init__(
            client=client,
            name=name or "IRCoT",
            system_prompt=SYSTEM_PROMPT,
        )
        self.retriever = retriever
        self.max_steps = max_steps
        self.top_k = top_k
        self.max_context_paragraphs = max_context_paragraphs
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _run_ircot(
        self, question: str
    ) -> Tuple[str, str, List[Dict[str, str]], List[str], int]:
        """执行完整的 IRCoT 推理-检索循环。

        Returns:
            (prediction, raw_output, retrieved_passages, reasoning_chain, llm_calls)
        """
        # Step 1: 初始检索
        passages = self.retriever.search(question, top_k=self.top_k)
        chain: List[str] = []
        llm_calls = 0

        # Step 2: 交错循环
        for _ in range(self.max_steps):
            step_response = self.client.chat(
                user_message=_reason_next_prompt(question, passages, chain),
                system_message=STEP_SYSTEM_PROMPT,
                temperature=self.temperature,
                max_tokens=96,
            )
            llm_calls += 1
            step = _clean_retrieval_step(step_response.content)
            if not step:
                break
            chain.append(step)

            # 用线索再次检索
            new_passages = self.retriever.search(step, top_k=self.top_k)
            passages = merge_unique_passages(
                passages, new_passages, limit=self.max_context_paragraphs
            )

        # Step 3: 最终答案生成
        final_response = self.client.chat(
            user_message=_final_reader_prompt(question, passages, chain),
            system_message=self.system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        llm_calls += 1

        pred = _parse_final_answer(final_response.content)
        return pred, final_response.content, passages, chain, llm_calls

    def solve(self, question: str) -> str:
        pred, _, _, _, _ = self._run_ircot(question)
        return pred

    def solve_sample_with_trace(self, sample) -> CoTTrace:
        trace = CoTTrace(question=sample.question)
        start_time = time.time()

        pred, raw_output, passages, chain, llm_calls = self._run_ircot(
            sample.question
        )
        gold = str(sample.ground_truth)

        trace.final_answer = pred
        trace.total_tokens = 0  # LLMClient 不跨请求累计，由 metadata 记录
        trace.total_time_seconds = time.time() - start_time
        trace.intermediate_responses = [raw_output]
        trace.reasoning_steps = chain
        trace.metadata.update({
            "method": "ircot",
            "prediction": pred,
            "gold": gold,
            "em": qa_exact_match(pred, gold),
            "f1": qa_token_f1(pred, gold),
            "title_recall": qa_title_recall(passages, sample.supporting_titles),
            "retrieved_count": len(passages),
            "llm_calls": llm_calls,
            "retrieved": passages,
            "reasoning_steps": chain,
            "max_steps": self.max_steps,
            "top_k": self.top_k,
        })
        return trace

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "method": "ircot",
            "max_steps": self.max_steps,
            "top_k": self.top_k,
            "max_context_paragraphs": self.max_context_paragraphs,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        })
        return config
