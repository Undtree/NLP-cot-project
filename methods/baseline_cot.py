"""
基线思维链策略 (Baseline CoT)
------------------------------
最基础的 Zero-shot Chain-of-Thought 实现。

策略流程:
1. 构造 Prompt: "Let's think step by step. ... The answer is:"
2. 调用云端 LLM 获取推理 + 答案
3. 返回完整输出，由 Evaluator 自动提取答案

这是最简实现，作为组员实现更复杂策略的参考模板。
"""

import time
from typing import Optional
from harness.base_task import BaseTask, CoTTrace
from harness.llm_client import LLMClient


# 零样本 CoT 的标准 Prompt 模板
ZERO_SHOT_COT_SYSTEM_PROMPT = """You are a helpful AI assistant that solves multiple-choice questions using step-by-step reasoning.
Follow these rules strictly:
1. Read the question and all options carefully.
2. Think through the problem step by step.
3. At the end of your reasoning, clearly state your final answer in the format:
   The answer is: X
   where X is one of A, B, C, D, or E.
4. Only output the final answer once at the very end."""

ZERO_SHOT_COT_USER_TEMPLATE = """Solve the following multiple-choice question by thinking step by step.

{question}

Let's think step by step."""


class BaselineCoT(BaseTask):
    """
    零样本思维链基线策略。

    使用方式:
        from methods.baseline_cot import BaselineCoT

        task = BaselineCoT(client, temperature=0.0)
        answer = task.solve("What is 2+2?\nA. 3\nB. 4\nC. 5")
    """

    def __init__(
        self,
        client: LLMClient,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        name: Optional[str] = None,
    ):
        super().__init__(
            client=client,
            name=name or "BaselineCoT",
            system_prompt=ZERO_SHOT_COT_SYSTEM_PROMPT,
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _build_user_prompt(self, question: str) -> str:
        """构建用户 Prompt"""
        return ZERO_SHOT_COT_USER_TEMPLATE.format(question=question)

    def solve(self, question: str) -> str:
        """
        使用零样本 CoT 解决问题。

        Args:
            question: 问题文本

        Returns:
            模型原始输出 (包含推理过程和最终答案)
        """
        user_prompt = self._build_user_prompt(question)
        response = self.client.chat(
            user_message=user_prompt,
            system_message=self.system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.content

    def solve_with_trace(self, question: str) -> CoTTrace:
        """
        带推理追踪的求解。
        记录完整推理过程和 token 消耗。
        """
        trace = CoTTrace(question=question)
        start_time = time.time()

        response = self.client.chat(
            user_message=self._build_user_prompt(question),
            system_message=self.system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        trace.final_answer = response.content
        trace.total_tokens = response.usage.get("total_tokens", 0)
        trace.total_time_seconds = time.time() - start_time
        trace.metadata["finish_reason"] = response.finish_reason
        trace.metadata["model"] = response.model

        # 尝试将输出按段落拆分作为推理步骤
        steps = [s.strip() for s in response.content.split("\n\n") if s.strip()]
        trace.reasoning_steps = steps

        return trace

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        })
        return config
