"""
self consistency
思路：同一道题让模型回答好几次，然后投票选出最终答案
参考论文：Wang et al. 2022, Self-Consistency Improves Chain of Thought Reasoning in Language Models
"""

import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from harness.base_task import BaseTask, CoTTrace
from harness.evaluator import quick_extract
from harness.llm_client import LLMClient


# 系统提示词，要求模型逐步推理并在最后输出固定格式的答案
SYSTEM_PROMPT = """You are a helpful AI assistant that solves multiple-choice questions using step-by-step reasoning.
Follow these rules strictly:
1. Read the question and all options carefully.
2. Develop one clear reasoning path for the problem.
3. At the end, state the final answer exactly in this format:
   The answer is: X
   where X is one of A, B, C, D, or E.
4. Do not output more than one final answer line."""


class SelfConsistencyTask(BaseTask):

    def __init__(
        self,
        client: LLMClient,
        paths: int = 5,            # 采样路径数
        temperature: float = 0.7,  # 温度不能太低，否则每条路径都一样，投票没意义
        max_tokens: int = 2048,
        name: Optional[str] = None,
    ):
        super().__init__(
            client=client,
            name=name or "SelfConsistency",
            system_prompt=SYSTEM_PROMPT,
            paths=paths,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.paths = paths
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _call_once(self, args):
        """
        单次调用一条推理路径，返回 (response, 提取到的答案)
        """
        question, path_idx = args
        user_msg = (
            question
            + f"\n\nThis is reasoning path #{path_idx}. Use your own independent reasoning."
            + "\n\nThink step by step, then finish with:\nThe answer is: X"
        )
        resp = self.client.chat(
            user_message=user_msg,
            system_message=self.system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        ans = quick_extract(resp.content)
        return resp, ans

    def _run(self, question: str):
        """
        并发采样所有路径，并发数受 LLM_MAX_CONCURRENT 控制
        """
        rate_limiter = getattr(self.client, "_rate_limiter", None)
        max_concurrent = getattr(rate_limiter, "max_concurrent", self.paths)
        n_workers = max(1, min(self.paths, max_concurrent))

        args_list = [(question, i + 1) for i in range(self.paths)]
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(self._call_once, args_list))

        responses = [r for r, _ in results]
        answers = [a for _, a in results]
        total_tokens = sum(r.usage.get("total_tokens", 0) for r in responses)

        # 投票：过滤掉提取失败的结果，只对合法选项计数
        valid = [a for a in answers if a in "ABCDE"]
        vote_counts = {}
        final_ans = "?"
        if valid:
            counts = Counter(valid)
            final_ans = counts.most_common(1)[0][0]  # 票数最多
            vote_counts = dict(counts)

        return responses, answers, vote_counts, final_ans, total_tokens

    def solve(self, question: str) -> str:
        _, _, vote_counts, final_ans, _ = self._run(question)

        if final_ans == "?":
            # 都没有提取出合法答案
            return "Self-consistency could not extract any valid answer from sampled paths.\nThe answer is: ?"

        # 输出投票详情，方便后续分析
        result = "Self-consistency vote summary: "
        for opt in "ABCDE":
            result += f"{opt}={vote_counts.get(opt, 0)}, "
        result = result.rstrip(", ")
        result += f"\nThe answer is: {final_ans}"
        return result

    def solve_with_trace(self, question: str) -> CoTTrace:
        """
        带推理追踪的版本，输出到 .jsonl 结果文件
        """
        trace = CoTTrace(question=question)
        start_time = time.time()

        responses, answers, vote_counts, final_ans, total_tokens = self._run(question)

        if final_ans == "?":
            final_output = "Self-consistency could not extract any valid answer from sampled paths.\nThe answer is: ?"
        else:
            final_output = "Self-consistency vote summary: "
            for opt in "ABCDE":
                final_output += f"{opt}={vote_counts.get(opt, 0)}, "
            final_output = final_output.rstrip(", ")
            final_output += f"\nThe answer is: {final_ans}"

        trace.final_answer = final_output
        trace.total_tokens = total_tokens
        trace.total_time_seconds = time.time() - start_time
        trace.intermediate_responses = [r.content for r in responses]
        # 保存每条路径提取到的答案
        trace.reasoning_steps = [
            f"Path {i+1}: extracted_answer={ans or '?'}"
            for i, ans in enumerate(answers)
        ]
        trace.metadata.update({
            "paths": self.paths,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "path_answers": answers,
            "vote_counts": vote_counts,
            "models": [r.model for r in responses],
            "finish_reasons": [r.finish_reason for r in responses],
        })
        return trace

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "paths": self.paths,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        })
        return config
