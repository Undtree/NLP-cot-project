"""
带 Verifier 的多样化投票策略 (Verifier CoT)
-------------------------------------------
使用多条推理路径 + Verifier 评分 + 加权投票提升推理可靠性。

策略流程:
1. 3 个 Solver（Baseline / Skeptic / DoubleChecker）独立推理
2. LLM Verifier 对每条路径评分（0-1 分），输出 JSON
3. 加权投票聚合答案，全票一致时跳过验证
"""

import json
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from harness.base_task import BaseTask, CoTTrace
from harness.evaluator import quick_extract
from harness.llm_client import LLMClient, LLMResponse


# Solver 用户 Prompt 模板
VERIFIER_COT_USER_TEMPLATE = """Solve the following multiple-choice question by thinking step by step.

{question}

Let's think step by step."""

# Solver 系统 Prompt
SOLVER_PROFILES = {
    "Baseline": {
        "system": """You are a helpful AI assistant that solves multiple-choice questions using step-by-step reasoning.
Follow these rules strictly:
1. Read the question and all options carefully.
2. Think through the problem step by step.
3. At the end of your reasoning, clearly state your final answer in the format:
   The answer is: X
   where X is one of A, B, C, D, or E.
4. Only output the final answer once at the very end.""",
        "style": "Standard zero-shot chain-of-thought reasoning.",
    },
    "Skeptic": {
        "system": """You are a skeptical problem solver. For each option, actively try to find why it could be WRONG before accepting it as correct.
Follow these rules strictly:
1. Read the question and all options carefully.
2. For each option A through E, explain one reason it might be incorrect (a calculation trap, a wrong assumption, an edge case). Then identify which option survives your scrutiny.
3. At the end of your reasoning, clearly state your final answer in the format:
   The answer is: X
   where X is one of A, B, C, D, or E.
4. Only output the final answer once at the very end.""",
        "style": "For each option, find a reason it could be wrong. The correct answer is the one that survives all attacks.",
    },
    "DoubleChecker": {
        "system": """You are a careful problem solver who solves first, then verifies.
Follow these rules strictly:
1. Read the question and all options carefully.
2. First, solve the problem independently without looking at the options — derive your own answer from scratch. Then compare your answer to A, B, C, D, E and select the matching one. If none match, re-check your work.
3. At the end of your reasoning, clearly state your final answer in the format:
   The answer is: X
   where X is one of A, B, C, D, or E.
4. Only output the final answer once at the very end.""",
        "style": "Solve from scratch without looking at options, then find which option matches your result.",
    },
}

SOLVER_PROFILE_NAMES = list(SOLVER_PROFILES.keys())

# Verifier 评分 Prompt
VERIFIER_PATH_SYSTEM_PROMPT = """You are a strict evaluator who scores the quality of math reasoning.
Follow these rules strictly:
1. Read the question and the agent's reasoning carefully.
2. Score the reasoning quality on a scale of 0.0 to 1.0:
   - 0.9-1.0: Flawless logic, all steps correct.
   - 0.7-0.8: Mostly correct, minor issues only.
   - 0.4-0.6: Partially correct but with notable gaps or weak justification.
   - 0.1-0.3: Significant errors or logical fallacies.
   - 0.0: Completely incoherent or empty.
3. Output ONLY a JSON object on a single line:
   {{"score": 0.85, "extracted_answer": "B", "critique": "brief one-sentence assessment"}}
   Do NOT include markdown code fences or any other text."""

VERIFIER_PATH_USER_TEMPLATE = """Evaluate the reasoning quality of the following solution.

Question:
{question}

Agent's reasoning:
{reasoning}

Output your evaluation as JSON only."""

# 回退 Prompt
FALLBACK_SYSTEM_PROMPT = """You are a helpful AI assistant that solves multiple-choice questions using step-by-step reasoning.
Follow these rules strictly:
1. Read the question and all options carefully.
2. Think through the problem step by step.
3. At the end of your reasoning, clearly state your final answer in the format:
   The answer is: X
   where X is one of A, B, C, D, or E.
4. Only output the final answer once at the very end."""


def parse_verifier_json(raw: str, default_score: float = 0.5) -> dict:
    """
    鲁棒地从 LLM 输出中提取 verifier JSON。

    三层回退策略:
    1. 正则匹配 JSON 对象 { ... }
    2. 尝试直接解析整个输出
    3. 正则提取 score 数值
    4. 最终回退默认值
    """
    # 清理可能的 markdown 代码块
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    def _clip_result(result: dict) -> dict:
        """确保 score 在 [0, 1] 范围内，extracted_answer 有效"""
        result["score"] = min(max(float(result.get("score", default_score)), 0.0), 1.0)
        ans = result.get("extracted_answer", "")
        if ans and isinstance(ans, str) and len(ans) == 1 and ans in "ABCDE":
            result["extracted_answer"] = ans
        else:
            result["extracted_answer"] = ""
        return result

    # 策略 1: 正则匹配 JSON 对象
    json_match = re.search(r'\{[^{}]*"score"[^{}]*\}', cleaned, re.DOTALL)
    if json_match:
        try:
            return _clip_result(json.loads(json_match.group(0)))
        except json.JSONDecodeError:
            pass

    # 策略 2: 直接解析整段输出
    try:
        return _clip_result(json.loads(cleaned))
    except json.JSONDecodeError:
        pass

    # 策略 3: 正则提取 score 数值
    score_match = re.search(r'"score"\s*:\s*([\d.]+)', cleaned)
    if score_match:
        score = float(score_match.group(1))
        return {
            "score": min(max(score, 0.0), 1.0),
            "extracted_answer": "",
            "critique": "score_regex_extracted",
        }

    # 策略 4: 最终回退
    return {
        "score": default_score,
        "extracted_answer": "",
        "critique": "parse_failed",
    }


class VerifierCoT(BaseTask):
    """多条推理路径 + Verifier 评分 + 加权投票的 CoT 策略。"""

    def __init__(
        self,
        client: LLMClient,
        num_paths: int = 3,
        temperature: float = 0.0,
        verifier_temperature: float = 0.0,
        path_max_tokens: int = 2048,
        verifier_max_tokens: int = 512,
        skip_when_unanimous: bool = True,
        default_score: float = 0.5,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            client=client,
            name=name or "VerifierCoT",
            system_prompt=VERIFIER_PATH_SYSTEM_PROMPT,
            num_paths=num_paths,
            temperature=temperature,
            verifier_temperature=verifier_temperature,
            path_max_tokens=path_max_tokens,
            verifier_max_tokens=verifier_max_tokens,
            skip_when_unanimous=skip_when_unanimous,
            default_score=default_score,
            **kwargs,
        )
        self.num_paths = num_paths
        self.temperature = temperature
        self.verifier_temperature = verifier_temperature
        self.path_max_tokens = path_max_tokens
        self.verifier_max_tokens = verifier_max_tokens
        self.skip_when_unanimous = skip_when_unanimous
        self.default_score = default_score

    def _build_solver_prompt(self, question: str, profile_name: str) -> str:
        """为指定 solver profile 构建 user prompt（使用统一模板）"""
        return VERIFIER_COT_USER_TEMPLATE.format(question=question)

    def _generate_diverse_paths(
        self, question: str
    ) -> List[Dict[str, object]]:
        """
        Phase 1: 生成 M 条多样化推理路径。

        Returns:
            List[dict]: 每条路径包含
                - profile: 使用的 profile 名称
                - content: 模型原始输出
                - answer: quick_extract 提取的答案
                - tokens: token 消耗
        """
        paths = []
        for i in range(self.num_paths):
            profile_name = SOLVER_PROFILE_NAMES[i % len(SOLVER_PROFILE_NAMES)]
            profile = SOLVER_PROFILES[profile_name]

            try:
                response = self.client.chat(
                    user_message=self._build_solver_prompt(question, profile_name),
                    system_message=profile["system"],
                    temperature=self.temperature,
                    max_tokens=self.path_max_tokens,
                )
                content = response.content
                answer = quick_extract(content) or ""
                tokens = response.usage.get("total_tokens", 0)
            except Exception as e:
                content = f"[ERROR] {e}"
                answer = ""
                tokens = 0

            paths.append({
                "profile": profile_name,
                "content": content,
                "answer": answer,
                "tokens": tokens,
            })

        return paths

    def _verify_path_level(
        self, question: str, path_content: str
    ) -> dict:
        """
        路径级验证：对整个推理路径打分。

        Returns:
            dict: {"score": float, "extracted_answer": str, "critique": str}
        """
        try:
            response = self.client.chat(
                user_message=VERIFIER_PATH_USER_TEMPLATE.format(
                    question=question,
                    reasoning=path_content,
                ),
                system_message=VERIFIER_PATH_SYSTEM_PROMPT,
                temperature=self.verifier_temperature,
                max_tokens=self.verifier_max_tokens,
            )
            result = parse_verifier_json(response.content, self.default_score)
            result["_verifier_tokens"] = response.usage.get("total_tokens", 0)
            return result
        except Exception:
            return {
                "score": self.default_score,
                "extracted_answer": "",
                "critique": "verifier_call_failed",
                "_verifier_tokens": 0,
            }

    def _verify_path(
        self, question: str, path_content: str
    ) -> dict:
        """对一条推理路径评分"""
        return self._verify_path_level(question, path_content)

    def _weighted_vote(
        self, scored_paths: List[dict]
    ) -> Tuple[str, float, str]:
        """
        加权投票：按 verifier score 加权聚合答案。

        Args:
            scored_paths: list of {
                "answer": str (A-E or ""),
                "score": float (0-1),
                "profile": str,
            }

        Returns:
            (winning_answer, total_weight, method_description)
        """
        # 过滤有效答案
        valid = [
            p for p in scored_paths
            if p["answer"] and p["answer"] in "ABCDE"
        ]

        if not valid:
            return ("", 0.0, "no_valid_answers")

        # 按答案累加分数
        weights: Dict[str, float] = defaultdict(float)
        best_per_answer: Dict[str, float] = {}  # 每个答案的最高单路径分

        for p in valid:
            ans = p["answer"]
            weights[ans] += p["score"]
            if ans not in best_per_answer or p["score"] > best_per_answer[ans]:
                best_per_answer[ans] = p["score"]

        # 找最高权重的答案
        max_weight = max(weights.values())
        candidates = [ans for ans, w in weights.items() if w == max_weight]

        if len(candidates) == 1:
            winner = candidates[0]
        else:
            # 平局：取单路径最高分的答案
            winner = max(candidates, key=lambda a: best_per_answer.get(a, 0.0))

        return (winner, weights[winner], f"weighted_vote_{len(valid)}_paths")

    # 回退策略

    def _fallback_solve(self, question: str) -> str:
        """回退：单次 baseline CoT 调用"""
        try:
            response = self.client.chat(
                user_message=f"Solve step by step.\n\n{question}\n\nThe answer is:",
                system_message=FALLBACK_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=self.path_max_tokens,
            )
            return response.content
        except Exception as e:
            return f"[FALLBACK_ERROR] {e}\nThe answer is: A"

    # 核心 solve 方法

    def solve(self, question: str) -> str:
        """
        三阶段求解: 多样化路径 → 验证评分 → 加权投票。

        Args:
            question: 问题文本（含选项）

        Returns:
            包含 "The answer is: X" 的完整回答
        """
        # Phase 1: 生成多样化推理路径
        paths = self._generate_diverse_paths(question)
        answers_from_paths = [p["answer"] for p in paths]

        # 短路优化：全票一致时跳过后续阶段
        if self.skip_when_unanimous:
            valid_answers = [a for a in answers_from_paths if a and a in "ABCDE"]
            if valid_answers and len(set(valid_answers)) == 1:
                unanimous = valid_answers[0]
                return (
                    f"All {len(valid_answers)} diverse solvers agree on the same answer. "
                    f"The answer is: {unanimous}"
                )

        # Phase 2: Verifier 评分
        scored_paths = []
        for p in paths:
            if p["content"].startswith("[ERROR]"):
                scored_paths.append({
                    "answer": "",
                    "score": 0.0,
                    "profile": p["profile"],
                    "verifier_critique": "solver_error",
                })
                continue

            verifier_result = self._verify_path(question, p["content"])

            # 优先用 verifier 提取的答案，回退到 quick_extract
            answer = verifier_result.get("extracted_answer", "")
            if not answer or answer not in "ABCDE":
                answer = p["answer"]

            scored_paths.append({
                "answer": answer,
                "score": verifier_result["score"],
                "profile": p["profile"],
                "verifier_critique": verifier_result.get("critique", ""),
                "verifier_tokens": verifier_result.get("_verifier_tokens", 0),
            })

        # Phase 3: 加权投票
        winner, weight, method = self._weighted_vote(scored_paths)

        if not winner:
            return self._fallback_solve(question)

        return (
            f"After generating {self.num_paths} diverse reasoning paths "
            f"and scoring them with a verifier ({method}), "
            f"the weighted vote selects: "
            f"The answer is: {winner}"
        )

    # 带 Trace 的求解

    def solve_with_trace(self, question: str) -> CoTTrace:
        """
        解决问题并记录完整推理路径。

        在 CoTTrace 中详细记录:
        - 每条路径的内容和答案
        - verifier 对每条路径的评分
        - 最终加权投票结果
        """
        trace = CoTTrace(question=question)
        start_time = time.time()
        total_tokens = 0

        # Phase 1: 生成路径
        paths = self._generate_diverse_paths(question)
        total_tokens += sum(p["tokens"] for p in paths)

        answers_from_paths = [p["answer"] for p in paths]

        # 短路检查
        if self.skip_when_unanimous:
            valid_answers = [a for a in answers_from_paths if a and a in "ABCDE"]
            if valid_answers and len(set(valid_answers)) == 1:
                unanimous = valid_answers[0]
                trace.final_answer = (
                    f"All {len(valid_answers)} diverse solvers agree. "
                    f"The answer is: {unanimous}"
                )
                trace.total_tokens = total_tokens
                trace.total_time_seconds = time.time() - start_time
                trace.intermediate_responses = [p["content"] for p in paths]
                trace.reasoning_steps = [
                    f"[{p['profile']}] → answer={p['answer']}" for p in paths
                ]
                trace.metadata.update({
                    "method": "unanimous_shortcut",
                    "num_paths": self.num_paths,
                    "paths": [{
                        "profile": p["profile"],
                        "answer": p["answer"],
                        "tokens": p["tokens"],
                    } for p in paths],
                })
                return trace

        # Phase 2: Verifier 评分
        scored_paths = []
        verifier_total_tokens = 0
        for p in paths:
            if p["content"].startswith("[ERROR]"):
                scored_paths.append({
                    "answer": "",
                    "score": 0.0,
                    "profile": p["profile"],
                })
                continue

            verifier_result = self._verify_path(question, p["content"])
            verifier_total_tokens += verifier_result.get("_verifier_tokens", 0)

            answer = verifier_result.get("extracted_answer", "")
            if not answer or answer not in "ABCDE":
                answer = p["answer"]

            scored_paths.append({
                "answer": answer,
                "score": verifier_result["score"],
                "profile": p["profile"],
                "verifier_critique": verifier_result.get("critique", ""),
            })

        total_tokens += verifier_total_tokens

        # Phase 3: 加权投票
        winner, weight, method = self._weighted_vote(scored_paths)

        if not winner:
            fallback = self._fallback_solve(question)
            trace.final_answer = fallback
            trace.metadata["method"] = "fallback"
        else:
            trace.final_answer = (
                f"After {self.num_paths} diverse paths + verifier scoring "
                f"({method}, winner_weight={weight:.3f}), "
                f"The answer is: {winner}"
            )
            trace.metadata["method"] = method
            trace.metadata["winner_weight"] = weight

        trace.total_tokens = total_tokens
        trace.total_time_seconds = time.time() - start_time

        # 记录中间输出
        trace.intermediate_responses = [
            f"[{p['profile']}] {p['content']}" for p in paths
        ]
        trace.reasoning_steps = [
            f"[{p['profile']}] answer={p['answer']}" for p in paths
        ]

        # 详细元数据
        trace.metadata.update({
            "num_paths": self.num_paths,
            "temperature": self.temperature,
            "verifier_temperature": self.verifier_temperature,
            "unanimous_skip": False,
            "total_solver_tokens": sum(p["tokens"] for p in paths),
            "total_verifier_tokens": verifier_total_tokens,
            "paths": [{
                "profile": p["profile"],
                "answer": p["answer"],
                "tokens": p["tokens"],
            } for p in paths],
            "verifier_scores": [{
                "profile": sp["profile"],
                "score": sp["score"],
                "answer": sp["answer"],
                "critique": sp.get("verifier_critique", ""),
            } for sp in scored_paths],
            "final_extracted_answer": quick_extract(trace.final_answer),
        })

        return trace

    # 配置

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "num_paths": self.num_paths,
            "temperature": self.temperature,
            "verifier_temperature": self.verifier_temperature,
            "path_max_tokens": self.path_max_tokens,
            "verifier_max_tokens": self.verifier_max_tokens,
            "skip_when_unanimous": self.skip_when_unanimous,
            "default_score": self.default_score,
        })
        return config
