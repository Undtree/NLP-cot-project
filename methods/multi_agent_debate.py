"""
多智能体辩论式 CoT 策略 (Multi-Agent Debate)
---------------------------------------------

本文件实现两个可直接注册到 harness 的策略:
1. DebateTask: 两个求解 Agent 独立推理 + Judge 裁决。
2. ReflectiveDebateTask: 在 DebateTask 基础上加入一轮互看观点后的反思修正。

设计目标:
- 不引入 AutoGen/LangGraph 等额外框架，保持在课程 harness 内可复现。
- 用多个角色 prompt 模拟 agent 分工、讨论、反思与裁决。
- 输出始终包含 "The answer is: X"，方便 evaluator 自动提取答案。
"""

import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

from harness.base_task import BaseTask, CoTTrace
from harness.evaluator import quick_extract
from harness.llm_client import LLMClient, LLMResponse


ANSWER_OPTIONS = "ABCDE"


JUDGE_SYSTEM_PROMPT = """You are a strict debate judge for multiple-choice reasoning problems.
Your job is to compare different agents' reasoning, identify mistakes, and choose the best final option.
Rules:
1. Prefer mathematically valid reasoning over majority vote.
2. If agents disagree, explain which argument is more reliable.
3. End with exactly one final line in this format:
   The answer is: X
   where X is one of A, B, C, D, or E."""


FINALIZER_SYSTEM_PROMPT = """You convert a reasoning result into a clean multiple-choice final answer.
Read the provided debate and output only a short final answer.
End with exactly:
The answer is: X
where X is one of A, B, C, D, or E."""


SOLVER_PROFILES = {
    "Analyst": {
        "system": """You are Agent Analyst, a careful mathematical reasoner.
Focus on solving the problem from first principles. Show concise step-by-step reasoning.
End with one line: The answer is: X""",
        "style": "Use direct calculation and algebraic reasoning.",
    },
    "Verifier": {
        "system": """You are Agent Verifier, an independent checker.
Solve the problem independently, then verify the result against the answer choices.
End with one line: The answer is: X""",
        "style": "Use sanity checks, units, and option verification.",
    },
    "Skeptic": {
        "system": """You are Agent Skeptic, a critical reasoner.
Look for traps, hidden assumptions, and tempting wrong options while solving.
End with one line: The answer is: X""",
        "style": "Emphasize edge cases and common mistakes.",
    },
}


class DebateTask(BaseTask):
    """
    基础多智能体辩论策略。

    流程:
    1. 多个 solver agent 独立作答。
    2. 可选若干轮 reflection，让 agent 读取其他人的答案后修正。
    3. Judge agent 汇总所有观点并裁决。

    Args:
        client: LLMClient 实例。
        num_agents: 使用几个 solver agent，默认 2。
        num_rounds: 反思轮数，默认 0，即基础 debate。
        temperature: 生成温度。multi-agent 建议略高于 baseline 以增加路径多样性。
        max_tokens: 每次 LLM 调用的 token 上限。
    """

    def __init__(
        self,
        client: LLMClient,
        num_agents: int = 2,
        num_rounds: int = 0,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        name: Optional[str] = None,
    ):
        super().__init__(
            client=client,
            name=name or "MultiAgentDebate",
            system_prompt=JUDGE_SYSTEM_PROMPT,
            num_agents=num_agents,
            num_rounds=num_rounds,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if num_agents < 2:
            raise ValueError("DebateTask 至少需要 2 个 solver agent")
        if num_agents > len(SOLVER_PROFILES):
            raise ValueError(f"当前最多支持 {len(SOLVER_PROFILES)} 个 solver agent")

        self.num_agents = num_agents
        self.num_rounds = num_rounds
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.agent_names = list(SOLVER_PROFILES.keys())[:num_agents]

    def _chat(
        self,
        user_message: str,
        system_message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        return self.client.chat(
            user_message=user_message,
            system_message=system_message,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )

    def _build_solver_prompt(self, question: str, agent_name: str) -> str:
        profile = SOLVER_PROFILES[agent_name]
        return f"""Solve the following multiple-choice question independently.

Role guidance: {profile["style"]}

Question:
{question}

Think step by step, keep the reasoning concise, and finish with:
The answer is: X"""

    def _run_initial_solvers(self, question: str) -> List[Dict[str, str]]:
        responses = []
        for agent_name in self.agent_names:
            profile = SOLVER_PROFILES[agent_name]
            response = self._chat(
                user_message=self._build_solver_prompt(question, agent_name),
                system_message=profile["system"],
            )
            responses.append({
                "agent": agent_name,
                "round": "initial",
                "content": response.content,
                "answer": quick_extract(response.content),
                "tokens": response.usage.get("total_tokens", 0),
            })
        return responses

    def _format_peer_views(
        self,
        current_agent: str,
        latest_by_agent: Dict[str, Dict[str, str]],
    ) -> str:
        chunks = []
        for agent_name, item in latest_by_agent.items():
            if agent_name == current_agent:
                continue
            chunks.append(
                f"[{agent_name}] proposed answer: {item.get('answer') or '?'}\n"
                f"{item['content']}"
            )
        return "\n\n".join(chunks)

    def _run_reflection_round(
        self,
        question: str,
        latest_by_agent: Dict[str, Dict[str, str]],
        round_index: int,
    ) -> List[Dict[str, str]]:
        reflected = []
        for agent_name in self.agent_names:
            profile = SOLVER_PROFILES[agent_name]
            own_previous = latest_by_agent[agent_name]
            peer_views = self._format_peer_views(agent_name, latest_by_agent)
            prompt = f"""You are revising your answer after reading other agents' reasoning.

Question:
{question}

Your previous reasoning:
{own_previous['content']}

Other agents' reasoning:
{peer_views}

Task:
1. Identify whether your previous reasoning or any peer reasoning contains an error.
2. Decide whether to keep or revise your answer.
3. Finish with exactly one line: The answer is: X"""

            response = self._chat(
                user_message=prompt,
                system_message=profile["system"],
            )
            reflected.append({
                "agent": agent_name,
                "round": f"reflection_{round_index}",
                "content": response.content,
                "answer": quick_extract(response.content),
                "tokens": response.usage.get("total_tokens", 0),
            })
        return reflected

    def _format_debate_transcript(self, agent_outputs: List[Dict[str, str]]) -> str:
        chunks = []
        for item in agent_outputs:
            chunks.append(
                f"## {item['agent']} ({item['round']})\n"
                f"Extracted answer: {item.get('answer') or '?'}\n"
                f"{item['content']}"
            )
        return "\n\n".join(chunks)

    def _majority_answer(self, agent_outputs: List[Dict[str, str]]) -> str:
        latest_answers = {}
        for item in agent_outputs:
            answer = item.get("answer") or ""
            if answer in ANSWER_OPTIONS:
                latest_answers[item["agent"]] = answer
        if not latest_answers:
            return ""
        return Counter(latest_answers.values()).most_common(1)[0][0]

    def _run_judge(
        self,
        question: str,
        agent_outputs: List[Dict[str, str]],
    ) -> Tuple[str, int]:
        transcript = self._format_debate_transcript(agent_outputs)
        majority = self._majority_answer(agent_outputs)
        prompt = f"""Question:
{question}

Debate transcript:
{transcript}

The latest solver majority answer is: {majority or "unknown"}.

As the judge, compare the reasoning quality and choose the final answer.
End with exactly one line:
The answer is: X"""

        response = self._chat(
            user_message=prompt,
            system_message=JUDGE_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=self.max_tokens,
        )

        final_answer = response.content
        if not quick_extract(final_answer):
            final_answer = self._finalize_answer(question, transcript, majority)

        return final_answer, response.usage.get("total_tokens", 0)

    def _finalize_answer(self, question: str, transcript: str, majority: str) -> str:
        prompt = f"""Question:
{question}

Debate transcript:
{transcript}

Majority answer if available: {majority or "unknown"}

Return a final answer in the required format."""
        response = self._chat(
            user_message=prompt,
            system_message=FINALIZER_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=128,
        )
        if quick_extract(response.content):
            return response.content
        if majority:
            return f"The answer is: {majority}"
        return response.content

    def _run_debate(self, question: str) -> Tuple[str, List[Dict[str, str]], int]:
        total_tokens = 0
        agent_outputs = self._run_initial_solvers(question)
        total_tokens += sum(item.get("tokens", 0) for item in agent_outputs)

        latest_by_agent = {item["agent"]: item for item in agent_outputs}
        for round_index in range(1, self.num_rounds + 1):
            reflected = self._run_reflection_round(
                question=question,
                latest_by_agent=latest_by_agent,
                round_index=round_index,
            )
            agent_outputs.extend(reflected)
            total_tokens += sum(item.get("tokens", 0) for item in reflected)
            latest_by_agent = {item["agent"]: item for item in reflected}

        final_answer, judge_tokens = self._run_judge(question, agent_outputs)
        total_tokens += judge_tokens
        return final_answer, agent_outputs, total_tokens

    def solve(self, question: str) -> str:
        final_answer, _, _ = self._run_debate(question)
        return final_answer

    def solve_with_trace(self, question: str) -> CoTTrace:
        trace = CoTTrace(question=question)
        start_time = time.time()

        final_answer, agent_outputs, total_tokens = self._run_debate(question)

        trace.final_answer = final_answer
        trace.total_tokens = total_tokens
        trace.total_time_seconds = time.time() - start_time
        trace.intermediate_responses = [item["content"] for item in agent_outputs]
        trace.reasoning_steps = [
            f"{item['agent']} ({item['round']}): answer={item.get('answer') or '?'}"
            for item in agent_outputs
        ]
        trace.metadata.update({
            "num_agents": self.num_agents,
            "num_rounds": self.num_rounds,
            "agents": self.agent_names,
            "solver_answers": [
                {
                    "agent": item["agent"],
                    "round": item["round"],
                    "answer": item.get("answer") or "",
                }
                for item in agent_outputs
            ],
            "final_extracted_answer": quick_extract(final_answer),
        })
        return trace

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "num_agents": self.num_agents,
            "num_rounds": self.num_rounds,
            "agent_names": self.agent_names,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        })
        return config


class ReflectiveDebateTask(DebateTask):
    """
    带一轮反思的多智能体辩论策略。

    相比 DebateTask，solver agent 会看到其他 agent 的推理后再修正一次，
    更贴合“讨论 + 反思”的 multi-agent 贡献点。
    """

    def __init__(
        self,
        client: LLMClient,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        num_agents: int = 2,
        num_rounds: int = 1,
        name: Optional[str] = None,
    ):
        super().__init__(
            client=client,
            num_agents=num_agents,
            num_rounds=num_rounds,
            temperature=temperature,
            max_tokens=max_tokens,
            name=name or "ReflectiveDebate",
        )


class ThreeAgentReflectiveDebateTask(ReflectiveDebateTask):
    """
    三 solver agent + 一轮反思 + Judge。

    这个版本调用次数更多，但报告展示效果更强，适合小样本消融实验。
    """

    def __init__(
        self,
        client: LLMClient,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        name: Optional[str] = None,
    ):
        super().__init__(
            client=client,
            temperature=temperature,
            max_tokens=max_tokens,
            num_agents=3,
            num_rounds=1,
            name=name or "ThreeAgentReflectiveDebate",
        )
