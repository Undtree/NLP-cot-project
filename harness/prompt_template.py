"""
Prompt 模板引擎 (Prompt Template Engine)
------------------------------------------
将系统指令、任务描述、可用工具列表和历史对话分离管理，
支持结构化组装和多种 Agent 范式（Standard CoT / ReAct / Tool-Use）。

设计目标:
  - 组员只需填写「插槽」，模板引擎负责拼接
  - 支持工具定义注入（OpenAI Function Calling 格式）
  - 支持历史对话自动截断与轮换
  - 所有模板可序列化，方便日志记录与复现

使用方式:
    from harness.prompt_template import PromptTemplate, ToolDefinition

    template = PromptTemplate(
        system_instruction="你是一个数学助手。",
        task_description="逐步推理并给出最终答案。",
        tools=[ToolDefinition(name="calculator", ...)],
        output_format="react",  # 或 "standard"
    )
    messages = template.render(question="2+2=?", history=[])
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Literal
from enum import Enum


# ============================================================
# 工具定义
# ============================================================

class ToolType(Enum):
    """工具类型枚举"""
    FUNCTION = "function"       # 普通函数调用
    CODE_INTERPRETER = "code"   # 代码解释器
    RETRIEVAL = "retrieval"     # 知识检索


@dataclass
class ToolParameter:
    """工具参数定义"""
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    enum: Optional[List[str]] = None  # 枚举值约束


@dataclass
class ToolDefinition:
    """
    工具定义 —— 描述一个 Agent 可调用的工具。

    输出为 OpenAI Function Calling 兼容的 JSON Schema。
    """
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    tool_type: ToolType = ToolType.FUNCTION

    def to_openai_schema(self) -> Dict[str, Any]:
        """转换为 OpenAI Function Calling 格式"""
        properties = {}
        required = []
        for param in self.parameters:
            prop = {"type": param.type, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_text_description(self) -> str:
        """生成人类可读的工具描述（用于纯文本 Prompt）"""
        param_str = ", ".join(
            f"{p.name}: {p.type}" + (f" (可选值: {p.enum})" if p.enum else "")
            for p in self.parameters
        )
        return f"- **{self.name}**: {self.description}\n  参数: {param_str}"


# ============================================================
# 对话消息与历史管理
# ============================================================

@dataclass
class ConversationTurn:
    """一轮对话"""
    role: str           # "user" | "assistant" | "system" | "tool"
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    def to_openai_message(self) -> Dict[str, Any]:
        msg: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg


class ConversationHistory:
    """
    对话历史管理器。

    功能:
    - 追加消息并自动维护角色交替
    - 基于 token 估算的自动截断（保留最近 N 轮）
    - 导出为 OpenAI 兼容的消息列表
    """

    def __init__(self, max_turns: int = 20):
        self.turns: List[ConversationTurn] = []
        self.max_turns = max_turns

    def add(self, role: str, content: str,
            tool_calls: Optional[List[Dict[str, Any]]] = None,
            tool_call_id: Optional[str] = None):
        self.turns.append(ConversationTurn(
            role=role, content=content,
            tool_calls=tool_calls, tool_call_id=tool_call_id,
        ))

    def add_user(self, content: str):
        self.add("user", content)

    def add_assistant(self, content: str,
                      tool_calls: Optional[List[Dict[str, Any]]] = None):
        self.add("assistant", content, tool_calls=tool_calls)

    def add_tool_result(self, content: str, tool_call_id: str):
        self.add("tool", content, tool_call_id=tool_call_id)

    def to_messages(self) -> List[Dict[str, Any]]:
        """导出为 OpenAI 兼容的消息列表，自动截断超长历史"""
        turns = self.turns
        if len(turns) > self.max_turns * 2:
            turns = turns[-(self.max_turns * 2):]
        return [t.to_openai_message() for t in turns]

    def clear(self):
        self.turns.clear()

    def __len__(self) -> int:
        return len(self.turns)

    def __bool__(self) -> bool:
        return len(self.turns) > 0


# ============================================================
# Prompt 模板
# ============================================================

OutputFormat = Literal["standard", "react", "tool_use"]


@dataclass
class PromptTemplate:
    """
    结构化 Prompt 模板。

    将 Prompt 拆分为独立插槽，支持:
    - 系统指令 (system_instruction): 定义 Agent 角色与行为约束
    - 任务描述 (task_description): 描述本次任务目标
    - 工具列表 (tools): 可用工具的定义
    - 输出格式 (output_format): 控制 Agent 的输出范式
    - 示例 (examples): Few-shot 示例
    - 额外规则 (extra_rules): 补充约束

    使用方式:
        template = PromptTemplate(
            system_instruction="You are a math assistant.",
            task_description="Solve the following problem step by step.",
            output_format="react",
        )
        system_prompt = template.render_system_prompt()
    """

    # ---- 核心插槽 ----
    system_instruction: str = (
        "You are a helpful AI assistant that solves problems "
        "using careful step-by-step reasoning."
    )
    task_description: str = ""
    tools: List[ToolDefinition] = field(default_factory=list)
    output_format: OutputFormat = "standard"

    # ---- 可选插槽 ----
    examples: List[Dict[str, str]] = field(default_factory=list)
    extra_rules: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ============================================================
    # 输出格式定义（可扩展）
    # ============================================================

    # 标准 CoT 格式指引
    STANDARD_COT_FORMAT = (
        "Follow these rules strictly:\n"
        "1. Read the question and all options carefully.\n"
        "2. Think through the problem step by step.\n"
        "3. At the end of your reasoning, clearly state your final answer in the format:\n"
        "   The answer is: X\n"
        "   where X is one of A, B, C, D, or E.\n"
        "4. Only output the final answer once at the very end."
    )

    # ReAct 格式指引（Thought / Action / Observation 循环）
    REACT_FORMAT = (
        "You MUST follow this exact format for every step:\n\n"
        "Thought: <your reasoning about what to do next>\n"
        "Action: <the tool name to call, or 'FinalAnswer'>\n"
        "Action Input: <the input to the tool, or your final answer>\n\n"
        "After receiving an Observation, continue with another Thought.\n"
        "When you have the final answer, use:\n"
        "Thought: I now know the final answer.\n"
        "Action: FinalAnswer\n"
        "Action Input: <your answer as a single letter A/B/C/D/E>"
    )

    # Tool-Use 格式指引（OpenAI Function Calling 兼容）
    TOOL_USE_FORMAT = (
        "You have access to the following tools. Use them when needed.\n"
        "Think step by step, and call tools using the function calling format.\n"
        "When ready, state your final answer as: The answer is: X"
    )

    _FORMAT_GUIDES = {
        "standard": STANDARD_COT_FORMAT,
        "react": REACT_FORMAT,
        "tool_use": TOOL_USE_FORMAT,
    }

    def _render_tools_section(self) -> str:
        """渲染工具列表为 Prompt 文本"""
        if not self.tools:
            return ""

        lines = ["## Available Tools", ""]
        for tool in self.tools:
            lines.append(tool.to_text_description())
            lines.append("")
        return "\n".join(lines)

    def _render_examples_section(self) -> str:
        """渲染 Few-shot 示例"""
        if not self.examples:
            return ""

        lines = ["## Examples", ""]
        for i, ex in enumerate(self.examples, 1):
            lines.append(f"### Example {i}")
            if "question" in ex:
                lines.append(f"**Question:** {ex['question']}")
                lines.append("")
            if "reasoning" in ex:
                lines.append(f"**Reasoning:** {ex['reasoning']}")
                lines.append("")
            if "answer" in ex:
                lines.append(f"**Answer:** {ex['answer']}")
                lines.append("")
        return "\n".join(lines)

    def _render_extra_rules_section(self) -> str:
        """渲染额外规则"""
        if not self.extra_rules:
            return ""

        lines = ["## Additional Rules", ""]
        for i, rule in enumerate(self.extra_rules, 1):
            lines.append(f"{i}. {rule}")
        return "\n".join(lines)

    def render_system_prompt(self) -> str:
        """
        渲染完整的系统 Prompt。

        组装顺序:
        1. 系统指令
        2. 任务描述
        3. 工具列表（如有）
        4. 输出格式指引
        5. Few-shot 示例（如有）
        6. 额外规则（如有）

        Returns:
            完整的系统 Prompt 字符串
        """
        sections = [self.system_instruction]

        if self.task_description:
            sections.append(f"\n## Task\n{self.task_description}")

        tools_section = self._render_tools_section()
        if tools_section:
            sections.append(f"\n{tools_section}")

        format_guide = self._FORMAT_GUIDES.get(
            self.output_format, self.STANDARD_COT_FORMAT
        )
        sections.append(f"\n## Output Format\n{format_guide}")

        examples_section = self._render_examples_section()
        if examples_section:
            sections.append(f"\n{examples_section}")

        rules_section = self._render_extra_rules_section()
        if rules_section:
            sections.append(f"\n{rules_section}")

        return "\n\n".join(sections)

    def render_user_prompt(self, question: str) -> str:
        """
        渲染用户 Prompt。

        Args:
            question: 问题文本

        Returns:
            用户 Prompt 字符串
        """
        return f"{question}\n\nLet's think step by step."

    def render_messages(
        self,
        question: str,
        history: Optional[ConversationHistory] = None,
    ) -> List[Dict[str, Any]]:
        """
        渲染完整的消息列表（用于 OpenAI API）。

        Args:
            question: 当前问题
            history: 对话历史（可选）

        Returns:
            可直接传入 API 的消息列表
        """
        messages: List[Dict[str, Any]] = []

        # 系统消息
        system_prompt = self.render_system_prompt()
        messages.append({"role": "system", "content": system_prompt})

        # 历史对话
        if history and history.turns:
            messages.extend(history.to_messages())

        # 当前用户消息
        messages.append({"role": "user", "content": self.render_user_prompt(question)})

        return messages

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """获取 OpenAI Function Calling 格式的工具列表"""
        return [t.to_openai_schema() for t in self.tools]

    def to_dict(self) -> Dict[str, Any]:
        """序列化模板配置"""
        return {
            "system_instruction": self.system_instruction,
            "task_description": self.task_description,
            "tools": [t.to_openai_schema() for t in self.tools],
            "output_format": self.output_format,
            "examples": self.examples,
            "extra_rules": self.extra_rules,
            "metadata": self.metadata,
        }


# ============================================================
# 预设模板（开箱即用）
# ============================================================

# 标准 CoT 数学推理模板
MATH_COT_TEMPLATE = PromptTemplate(
    system_instruction=(
        "You are a highly skilled mathematical reasoning assistant. "
        "Your goal is to solve multiple-choice math problems with "
        "precise, step-by-step logical deduction."
    ),
    task_description=(
        "Solve the given multiple-choice math problem. "
        "Think through every step carefully and verify your reasoning "
        "before stating the final answer."
    ),
    output_format="standard",
    extra_rules=[
        "Always show your complete reasoning process.",
        "Double-check arithmetic calculations.",
        "The final answer MUST be a single letter: A, B, C, D, or E.",
        "Use the exact format 'The answer is: X' at the very end.",
    ],
)

# ReAct Agent 模板（用于多步推理 + 工具调用）
REACT_AGENT_TEMPLATE = PromptTemplate(
    system_instruction=(
        "You are an agent that solves complex problems through "
        "interleaved reasoning and tool use. You think before acting, "
        "and use tools to gather information or perform calculations."
    ),
    task_description=(
        "Solve the problem using the ReAct format. "
        "You may need multiple cycles of thought and action."
    ),
    output_format="react",
)
