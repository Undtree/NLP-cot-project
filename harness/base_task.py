"""
任务标准化接口 (Task / Agent Interface)
----------------------------------------
Harness 框架的精髓 —— 定义抽象基类，强制组员实现的策略类遵循统一规范。

所有组员的 CoT 策略实现必须:
1. 继承 BaseTask
2. 实现 solve(question: str) -> str 方法
3. (可选) 实现 solve_with_trace(question: str) -> CoTTrace 以记录完整推理路径
4. (可选) 实现 solve_agentic(question: str) -> ParsedTrace 以支持多步 Agent 推理

这样可以保证:
- 不同策略可以在统一的 main.py 流水线上互换
- 评估器能无差别地对所有策略进行打分
- 日志系统能统一记录推理路径
- 状态机解析器能精准提取思考与动作块
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field

from .prompt_template import (
    PromptTemplate, ToolDefinition, ConversationHistory, MATH_COT_TEMPLATE,
)
from .state_machine import (
    StateMachineParser, ParsedTrace, ParsedBlock, BlockType, ReActParser,
)

logger = logging.getLogger("harness.base_task")


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

    新增能力:
    - 结构化 Prompt 模板：分离系统指令、任务描述、输出格式
    - 工具注册与管理：注册可调用工具，自动注入 Prompt
    - Agentic 推理循环：多步 Thought→Action→Observation 循环
    - 状态机解析：自动解析 LLM 输出中的思考与动作块

    所有组员的策略类必须继承此类并实现 solve 方法。

    使用示例 (组员视角):
        from harness.base_task import BaseTask, CoTTrace
        from harness.prompt_template import PromptTemplate, ToolDefinition

        class MyAgentStrategy(BaseTask):
            def __init__(self, client, **kwargs):
                super().__init__(
                    client,
                    name="MyAgent",
                    prompt_template=PromptTemplate(
                        system_instruction="你是一个数学助手。",
                        output_format="react",
                    ),
                )
                # 注册工具
                self.register_tool(ToolDefinition(
                    name="calculator",
                    description="执行数学计算",
                    parameters=[...],
                ))

            def solve(self, question: str) -> str:
                # 单次调用
                messages = self.build_messages(question)
                response = self.client.chat_multi_turn(messages)
                return response.content

            def solve_agentic(self, question: str) -> ParsedTrace:
                # 多步 Agent 推理
                return self.run_agent_loop(question, max_steps=5)

    Args:
        client: LLMClient 实例，用于调用云端模型
        name: 策略名称 (用于日志和报告)
        system_prompt: 系统提示词 (简单模式，与 prompt_template 二选一)
        prompt_template: 结构化 Prompt 模板 (推荐使用)
        **kwargs: 组员的自定义配置
    """

    def __init__(
        self,
        client,
        name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        prompt_template: Optional[PromptTemplate] = None,
        **kwargs,
    ):
        self.client = client
        self.name = name or self.__class__.__name__

        # ---- Prompt 模板系统 ----
        if prompt_template is not None:
            self.prompt_template = prompt_template
        elif system_prompt is not None:
            # 向后兼容：从旧 system_prompt 自动构建模板
            self.prompt_template = MATH_COT_TEMPLATE.__class__(
                system_instruction=system_prompt,
                output_format="standard",
            )
        else:
            self.prompt_template = MATH_COT_TEMPLATE

        # 保留旧接口兼容
        self.system_prompt = system_prompt or self.prompt_template.system_instruction

        # ---- 工具系统 ----
        self._tools: Dict[str, ToolDefinition] = {}
        self._tool_handlers: Dict[str, Callable] = {}

        # ---- 对话历史 ----
        self._conversation_history: Optional[ConversationHistory] = None

        # ---- 状态机解析器 ----
        self._state_parser = StateMachineParser()

        self.extra_config = kwargs  # 组员的自定义配置

    # ============================================================
    # 工具管理
    # ============================================================

    def register_tool(self, tool: ToolDefinition, handler: Optional[Callable] = None):
        """
        注册一个工具。

        Args:
            tool: 工具定义
            handler: 工具的执行函数。签名为 handler(input: str) -> str
                     如果不提供，Agentic 循环会在调用时跳过实际执行。
        """
        self._tools[tool.name] = tool
        if handler:
            self._tool_handlers[tool.name] = handler
        # 同步到 Prompt 模板
        if tool not in self.prompt_template.tools:
            self.prompt_template.tools.append(tool)
        logger.info(f"[{self.name}] 已注册工具: {tool.name}")

    def unregister_tool(self, tool_name: str):
        """移除一个工具"""
        self._tools.pop(tool_name, None)
        self._tool_handlers.pop(tool_name, None)
        self.prompt_template.tools = [
            t for t in self.prompt_template.tools if t.name != tool_name
        ]

    def get_tools(self) -> List[ToolDefinition]:
        """获取所有已注册的工具"""
        return list(self._tools.values())

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """获取 OpenAI Function Calling 格式的工具列表"""
        return self.prompt_template.get_tool_schemas()

    # ============================================================
    # Prompt 构建
    # ============================================================

    def build_messages(
        self,
        question: str,
        include_history: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        构建完整的消息列表。

        Args:
            question: 问题文本
            include_history: 是否包含对话历史（Agentic 模式）

        Returns:
            OpenAI 兼容的消息列表
        """
        history = self._conversation_history if include_history else None
        return self.prompt_template.render_messages(question, history)

    def build_system_prompt(self) -> str:
        """渲染当前模板的系统 Prompt"""
        return self.prompt_template.render_system_prompt()

    # ============================================================
    # 对话历史管理
    # ============================================================

    def start_conversation(self):
        """开始一轮新的对话（清空历史）"""
        self._conversation_history = ConversationHistory()

    def add_to_history(self, role: str, content: str, **kwargs):
        """向对话历史添加一条消息"""
        if self._conversation_history is None:
            self.start_conversation()
        self._conversation_history.add(role, content, **kwargs)

    def get_history_messages(self) -> List[Dict[str, Any]]:
        """获取当前对话历史的消息列表"""
        if self._conversation_history is None:
            return []
        return self._conversation_history.to_messages()

    # ============================================================
    # Agentic 推理循环（核心新增）
    # ============================================================

    def run_agent_loop(
        self,
        question: str,
        max_steps: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        verbose: bool = False,
    ) -> ParsedTrace:
        """
        执行 ReAct Agent 推理循环。

        循环流程:
        1. 发送 Prompt + 历史 → LLM
        2. StateMachineParser 解析输出 → 提取 Thought / Action
        3. 如果 Action 是 FinalAnswer → 退出循环，返回 ParsedTrace
        4. 如果 Action 是工具调用 → 执行工具 → Observation → 回到步骤 1
        5. 达到 max_steps 仍无答案 → 返回当前 ParsedTrace

        Args:
            question: 问题文本
            max_steps: 最大推理步数
            temperature: 采样温度
            max_tokens: 单次最大生成 token 数
            verbose: 是否打印详细过程

        Returns:
            ParsedTrace 包含完整的状态块解析结果
        """
        self.start_conversation()
        parsed_trace = ParsedTrace()

        for step in range(max_steps):
            if verbose:
                logger.info(f"[{self.name}] Agent 步骤 {step + 1}/{max_steps}")

            # 构建消息（含历史）
            messages = self.build_messages(question, include_history=True)

            # 调用 LLM
            response = self.client.chat_multi_turn(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # 记录助手回复到历史
            self.add_to_history("assistant", response.content)

            # 状态机解析
            step_trace = self._state_parser.parse(response.content)
            parsed_trace.blocks.extend(step_trace.blocks)
            parsed_trace.raw_output = response.content

            if verbose:
                for block in step_trace.blocks:
                    if block.block_type == BlockType.THOUGHT:
                        logger.info(f"  💭 Thought: {block.content[:80]}...")
                    elif block.block_type == BlockType.ACTION:
                        logger.info(f"  🔧 Action: {block.action_name}")
                    elif block.block_type == BlockType.FINAL_ANSWER:
                        logger.info(f"  ✅ FinalAnswer: {block.content}")

            # 检查是否有 FinalAnswer
            if step_trace.final_answer is not None:
                parsed_trace.metadata["total_steps"] = step + 1
                parsed_trace.metadata["termination"] = "final_answer"
                return parsed_trace

            # 处理工具调用
            tool_called = False
            for block in step_trace.blocks:
                if block.block_type == BlockType.ACTION:
                    tool_name = block.action_name
                    tool_input = block.action_input

                    if tool_name in self._tool_handlers:
                        try:
                            observation = self._tool_handlers[tool_name](tool_input)
                            observation_str = str(observation)
                        except Exception as e:
                            observation_str = f"工具执行错误: {e}"
                            logger.error(f"[{self.name}] 工具 {tool_name} 执行失败: {e}")
                    else:
                        observation_str = (
                            f"未知工具 '{tool_name}'。"
                            f"可用工具: {list(self._tools.keys())}"
                        )

                    # 注入 Observation
                    self.add_to_history(
                        "user",
                        f"Observation: {observation_str}",
                    )
                    tool_called = True

                    if verbose:
                        logger.info(f"  📊 Observation: {observation_str[:80]}...")

            if not tool_called:
                # 没有工具调用也没有 FinalAnswer → 可能格式不对
                logger.warning(
                    f"[{self.name}] 步骤 {step + 1} 未检测到 Action 或 FinalAnswer"
                )

        # 达到最大步数
        parsed_trace.metadata["total_steps"] = max_steps
        parsed_trace.metadata["termination"] = "max_steps_reached"
        logger.warning(
            f"[{self.name}] 达到最大步数 {max_steps}，强制终止 Agent 循环"
        )
        return parsed_trace

    # ============================================================
    # 核心抽象方法（组员必须实现）
    # ============================================================

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

    # ============================================================
    # 可选重写方法
    # ============================================================

    def solve_with_trace(self, question: str) -> CoTTrace:
        """
        解决问题并记录完整推理路径。

        使用状态机解析提取推理步骤。

        Args:
            question: 问题文本

        Returns:
            CoTTrace 对象，包含完整推理路径
        """
        trace = CoTTrace(question=question)
        start_time = time.time()

        final_answer = self.solve(question)
        trace.final_answer = final_answer
        trace.total_time_seconds = time.time() - start_time

        return trace

    # ============================================================
    # Sample-based 接口（泛用化扩展）
    # ============================================================

    def solve_sample(self, sample) -> str:
        """基于 Sample 对象的求解接口。

        默认从 Sample 中提取 question 字段调用 solve(question)。
        需要访问数据集特定元数据（如 HotpotQA 的 context）的策略可重写此方法。

        Args:
            sample: Sample 对象，含 question, ground_truth, metadata

        Returns:
            模型输出（最终答案文本）
        """
        return self.solve(sample.question)

    def solve_sample_with_trace(self, sample) -> CoTTrace:
        """基于 Sample 对象的带 trace 求解接口。

        默认从 Sample 中提取 question 字段调用 solve_with_trace(question)。

        Args:
            sample: Sample 对象

        Returns:
            CoTTrace 对象
        """
        return self.solve_with_trace(sample.question)

    def solve_agentic(self, question: str, **kwargs) -> ParsedTrace:
        """
        Agentic 推理求解（组员可重写）。

        默认使用 run_agent_loop。组员可重写以实现自定义 Agent 逻辑。

        Args:
            question: 问题文本
            **kwargs: 传递给 run_agent_loop 的参数

        Returns:
            ParsedTrace
        """
        return self.run_agent_loop(question, **kwargs)

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

    # ============================================================
    # 配置与序列化
    # ============================================================

    def get_config(self) -> Dict[str, Any]:
        """返回当前策略的配置信息（用于日志记录）"""
        return {
            "strategy_name": self.name,
            "system_prompt": self.system_prompt,
            "output_format": self.prompt_template.output_format,
            "tools": list(self._tools.keys()),
            **self.extra_config,
        }

    def __repr__(self) -> str:
        tools_str = f", tools={list(self._tools.keys())}" if self._tools else ""
        return (f"{self.name}("
                f"format={self.prompt_template.output_format}"
                f"{tools_str}"
                f")")
