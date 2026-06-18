"""
任务标准化接口 (Task / Agent Interface)
----------------------------------------
Harness 框架的精髓 —— 定义抽象基类，强制组员实现的策略类遵循统一规范。

所有组员的 CoT 策略实现必须:
1. 继承 BaseTask
2. 实现 solve(question: str) -> str 方法
3. (可选) 实现 solve_with_trace(question: str) -> CoTTrace 以记录完整推理路径

这样可以保证:
- 不同策略可以在统一的 main.py 流水线上互换
- 评估器能无差别地对所有策略进行打分
- 日志系统能统一记录推理路径
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class CoTTrace:
    """
    思维链推理路径追踪。

    记录 Agent 在解决问题时的完整推理过程，
    便于后续使用 Verifier 进行验证和分析。
    """
    question: str = ""                      # 原始问题
    final_answer: str = ""                  # 最终答案
    reasoning_steps: List[str] = field(default_factory=list)  # 中间推理步骤
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)  # 工具调用记录
    intermediate_responses: List[str] = field(default_factory=list)  # 中间模型响应
    total_tokens: int = 0                   # 总 token 消耗
    total_time_seconds: float = 0.0         # 总耗时
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典，方便日志记录"""
        return {
            "question": self.question,
            "final_answer": self.final_answer,
            "reasoning_steps": self.reasoning_steps,
            "tool_calls": self.tool_calls,
            "intermediate_responses": self.intermediate_responses,
            "total_tokens": self.total_tokens,
            "total_time_seconds": self.total_time_seconds,
            "metadata": self.metadata,
        }

    def add_step(self, step: str):
        """添加一个推理步骤"""
        self.reasoning_steps.append(step)

    def add_tool_call(self, tool_name: str, tool_input: Any, tool_output: Any):
        """记录一次工具调用"""
        self.tool_calls.append({
            "tool": tool_name,
            "input": tool_input,
            "output": tool_output,
        })


class BaseTask(ABC):
    """
    CoT 任务的抽象基类。

    所有组员的策略类必须继承此类并实现 solve 方法。

    使用示例 (组员视角):
        from harness.base_task import BaseTask, CoTTrace

        class MyCoTStrategy(BaseTask):
            def __init__(self, client, **kwargs):
                super().__init__(client, **kwargs)
                # 组员自定义初始化

            def solve(self, question: str) -> str:
                # 组员实现自己的 CoT 策略
                prompt = self._build_prompt(question)
                response = self.client.chat(prompt)
                return response.content

            def solve_with_trace(self, question: str) -> CoTTrace:
                # (推荐) 实现带推理路径追踪的版本
                trace = CoTTrace(question=question)
                # ... 组员的推理流程 ...
                return trace

    Args:
        client: LLMClient 实例，用于调用云端模型
        name: 策略名称 (用于日志和报告)
        system_prompt: 系统提示词 (可选)
    """

    def __init__(
        self,
        client,
        name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        self.client = client
        self.name = name or self.__class__.__name__
        self.system_prompt = system_prompt
        self.extra_config = kwargs  # 组员的自定义配置

    @abstractmethod
    def solve(self, question: str) -> str:
        """
        解决问题的核心方法。

        【组员必须实现此方法】

        Args:
            question: 问题文本 (含选项)

        Returns:
            模型的最终答案字符串 (如 "A", "B", "C", "D", "E")
        """
        pass

    def solve_with_trace(self, question: str) -> CoTTrace:
        """
        解决问题并记录完整推理路径。

        【组员可选择重写此方法以提供更详细的推理追踪】
        默认实现只记录最终答案，不记录中间步骤。

        Args:
            question: 问题文本

        Returns:
            CoTTrace 对象，包含完整推理路径
        """
        trace = CoTTrace(question=question)
        final_answer = self.solve(question)
        trace.final_answer = final_answer
        return trace

    def solve_batch(self, questions: List[str]) -> List[str]:
        """
        批量求解（串行版本）。
        组员可重写为并行版本以加速。

        Args:
            questions: 问题列表

        Returns:
            答案列表
        """
        answers = []
        for q in questions:
            ans = self.solve(q)
            answers.append(ans)
        return answers

    def get_config(self) -> Dict[str, Any]:
        """返回当前策略的配置信息（用于日志记录）"""
        return {
            "strategy_name": self.name,
            "system_prompt": self.system_prompt,
            **self.extra_config,
        }

    def __repr__(self) -> str:
        return f"{self.name}(config={self.extra_config})"
