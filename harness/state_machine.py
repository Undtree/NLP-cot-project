"""
状态机解析器 (State Machine Parser)
------------------------------------
将 LLM 的结构化输出（ReAct / Tool-Use 等范式）解析为离散的状态块，
使代码能精准提取「思考」和「动作」。

支持的输出范式:
  - ReAct: Thought → Action → Action Input → Observation 循环
  - XML-Tag: <thinking>...</thinking> <action>...</action>
  - Markdown-Fenced: ```thought ... ``` ```action ... ```

设计目标:
  - 解析后的状态块具有明确类型，方便下游 Agent 循环控制
  - 支持流式解析（增量式提取已完成的状态块）
  - 与 PromptTemplate 的输出格式对齐

使用方式:
    from harness.state_machine import ReActParser, ParsedBlock

    parser = ReActParser()
    blocks = parser.parse(llm_output)
    for block in blocks:
        if block.block_type == BlockType.THOUGHT:
            print(f"思考: {block.content}")
        elif block.block_type == BlockType.ACTION:
            print(f"动作: {block.action_name} -> {block.action_input}")
"""

import re
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple

logger = logging.getLogger("harness.state_machine")


# ============================================================
# 状态块定义
# ============================================================

class BlockType(Enum):
    """解析后的状态块类型"""
    THOUGHT = "thought"               # 思考/推理
    ACTION = "action"                 # 工具调用
    OBSERVATION = "observation"       # 工具返回结果
    FINAL_ANSWER = "final_answer"     # 最终答案
    RAW_TEXT = "raw_text"             # 未识别的原始文本


@dataclass
class ParsedBlock:
    """
    解析后的单个状态块。

    属性:
        block_type: 块类型
        content: 块内容（原始文本）
        action_name: 工具名称（仅 ACTION 类型）
        action_input: 工具输入（仅 ACTION 类型）
        confidence: 解析置信度 (0.0 ~ 1.0)
        raw_span: 在原文本中的起止位置
    """
    block_type: BlockType
    content: str = ""
    action_name: str = ""
    action_input: str = ""
    confidence: float = 1.0
    raw_span: Tuple[int, int] = (0, 0)

    def __repr__(self) -> str:
        if self.block_type == BlockType.ACTION:
            return (f"ParsedBlock(ACTION, name={self.action_name}, "
                    f"input={self.action_input[:50]}...)")
        return (f"ParsedBlock({self.block_type.value}, "
                f"content={self.content[:50]}...)")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_type": self.block_type.value,
            "content": self.content,
            "action_name": self.action_name,
            "action_input": self.action_input,
            "confidence": self.confidence,
        }


@dataclass
class ParsedTrace:
    """
    完整解析后的推理路径。

    包含所有解析出的状态块，并提供便捷的访问方法。
    """
    raw_output: str = ""
    blocks: List[ParsedBlock] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def thoughts(self) -> List[ParsedBlock]:
        """所有思考块"""
        return [b for b in self.blocks if b.block_type == BlockType.THOUGHT]

    @property
    def actions(self) -> List[ParsedBlock]:
        """所有动作块"""
        return [b for b in self.blocks if b.block_type == BlockType.ACTION]

    @property
    def final_answer(self) -> Optional[str]:
        """最终答案（如有）"""
        for b in reversed(self.blocks):
            if b.block_type == BlockType.FINAL_ANSWER:
                return b.content
        return None

    @property
    def reasoning_chain(self) -> str:
        """拼接所有思考文本"""
        return "\n".join(b.content for b in self.thoughts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blocks": [b.to_dict() for b in self.blocks],
            "final_answer": self.final_answer,
            "num_thoughts": len(self.thoughts),
            "num_actions": len(self.actions),
            "metadata": self.metadata,
        }


# ============================================================
# ReAct 解析器
# ============================================================

class ReActParser:
    """
    ReAct 范式解析器。

    解析格式:
        Thought: <推理内容>
        Action: <工具名称>
        Action Input: <工具输入>
        ... (Observation 由外部注入)
        Thought: <推理内容>
        Action: FinalAnswer
        Action Input: <最终答案>

    特性:
    - 支持大小写不敏感匹配
    - 支持中英文混排
    - 支持 "FinalAnswer" / "Final Answer" / "final_answer" 变体
    - 返回带类型的 ParsedBlock 列表
    """

    # ReAct 步骤的正则模式
    _THOUGHT_PATTERN = re.compile(
        r"(?:Thought|思考|思路)[：:]\s*(.+?)(?=\n(?:Action|动作|Observation|观察)|$)",
        re.IGNORECASE | re.DOTALL,
    )

    _ACTION_PATTERN = re.compile(
        r"(?:Action|动作)[：:]\s*(.+?)(?=\n(?:Action\s*Input|动作输入|Thought|思考)|$)",
        re.IGNORECASE,
    )

    _ACTION_INPUT_PATTERN = re.compile(
        r"(?:Action\s*Input|动作输入)[：:]\s*(.+?)(?=\n(?:Thought|思考|Action|动作|Observation|观察)|$)",
        re.IGNORECASE | re.DOTALL,
    )

    _OBSERVATION_PATTERN = re.compile(
        r"(?:Observation|观察|结果)[：:]\s*(.+?)(?=\n(?:Thought|思考|Action|动作)|$)",
        re.IGNORECASE | re.DOTALL,
    )

    # 最终答案的变体识别
    _FINAL_ANSWER_ACTIONS = {
        "finalanswer", "final answer", "final_answer",
        "finish", "终结", "最终答案",
    }

    def parse(self, raw_output: str) -> ParsedTrace:
        """
        解析 LLM 的 ReAct 输出。

        Args:
            raw_output: LLM 原始输出文本

        Returns:
            ParsedTrace 包含所有解析出的状态块
        """
        trace = ParsedTrace(raw_output=raw_output)
        remaining = raw_output
        offset = 0

        while remaining.strip():
            # 尝试匹配 Thought
            thought_match = self._THOUGHT_PATTERN.search(remaining)
            # 尝试匹配 Action
            action_match = self._ACTION_PATTERN.search(remaining)
            # 尝试匹配 Observation
            obs_match = self._OBSERVATION_PATTERN.search(remaining)

            # 找到最早出现的关键字
            candidates = []
            if thought_match:
                candidates.append((thought_match.start(), "thought", thought_match))
            if action_match:
                candidates.append((action_match.start(), "action", action_match))
            if obs_match:
                candidates.append((obs_match.start(), "observation", obs_match))

            if not candidates:
                # 没有匹配到任何关键字，剩余内容作为 RAW_TEXT
                remaining_stripped = remaining.strip()
                if remaining_stripped:
                    trace.blocks.append(ParsedBlock(
                        block_type=BlockType.RAW_TEXT,
                        content=remaining_stripped,
                        raw_span=(offset, offset + len(remaining_stripped)),
                    ))
                break

            # 按出现位置排序
            candidates.sort(key=lambda x: x[0])
            pos, kind, match = candidates[0]

            # 如果关键字前有文本，作为 RAW_TEXT
            prefix = remaining[:match.start()].strip()
            if prefix:
                trace.blocks.append(ParsedBlock(
                    block_type=BlockType.RAW_TEXT,
                    content=prefix,
                    raw_span=(offset, offset + match.start()),
                ))

            if kind == "thought":
                content = match.group(1).strip()
                trace.blocks.append(ParsedBlock(
                    block_type=BlockType.THOUGHT,
                    content=content,
                    raw_span=(offset + match.start(), offset + match.end()),
                ))

            elif kind == "action":
                action_name = match.group(1).strip()
                # 查找对应的 Action Input
                remaining_after_action = remaining[match.end():]
                input_match = self._ACTION_INPUT_PATTERN.search(remaining_after_action)
                action_input = ""
                input_end = match.end()

                if input_match:
                    action_input = input_match.group(1).strip()
                    input_end = match.end() + input_match.end()

                # 判断是否为 FinalAnswer
                action_key = action_name.lower().replace(" ", "").replace("-", "")
                if action_key in self._FINAL_ANSWER_ACTIONS:
                    trace.blocks.append(ParsedBlock(
                        block_type=BlockType.FINAL_ANSWER,
                        content=action_input,
                        action_name="FinalAnswer",
                        action_input=action_input,
                        raw_span=(offset + match.start(), offset + input_end),
                    ))
                else:
                    trace.blocks.append(ParsedBlock(
                        block_type=BlockType.ACTION,
                        content=action_input,
                        action_name=action_name,
                        action_input=action_input,
                        raw_span=(offset + match.start(), offset + input_end),
                    ))

                remaining = remaining[input_end:]
                offset += input_end
                continue

            elif kind == "observation":
                content = match.group(1).strip()
                trace.blocks.append(ParsedBlock(
                    block_type=BlockType.OBSERVATION,
                    content=content,
                    raw_span=(offset + match.start(), offset + match.end()),
                ))

            remaining = remaining[match.end():]
            offset += match.end()

        return trace

    def extract_final_answer(self, raw_output: str) -> Optional[str]:
        """
        便捷方法：直接从 ReAct 输出中提取最终答案。

        Args:
            raw_output: LLM 原始输出

        Returns:
            最终答案字符串，或 None
        """
        trace = self.parse(raw_output)
        return trace.final_answer


# ============================================================
# XML-Tag 解析器
# ============================================================

class XMLTagParser:
    """
    XML 标签式解析器。

    解析格式:
        <thinking>推理内容</thinking>
        <action>工具名称</action>
        <action_input>输入参数</action_input>
        <answer>最终答案</answer>

    适用于明确要求 LLM 使用 XML 标签输出的场景。
    """

    _TAG_PATTERNS = {
        BlockType.THOUGHT: re.compile(
            r"<thinking>(.*?)</thinking>", re.IGNORECASE | re.DOTALL
        ),
        BlockType.ACTION: re.compile(
            r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL
        ),
        BlockType.FINAL_ANSWER: re.compile(
            r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL
        ),
        BlockType.OBSERVATION: re.compile(
            r"<observation>(.*?)</observation>", re.IGNORECASE | re.DOTALL
        ),
    }

    def parse(self, raw_output: str) -> ParsedTrace:
        """
        解析 XML 标签式输出。

        Args:
            raw_output: LLM 原始输出

        Returns:
            ParsedTrace
        """
        trace = ParsedTrace(raw_output=raw_output)

        for block_type, pattern in self._TAG_PATTERNS.items():
            for match in pattern.finditer(raw_output):
                content = match.group(1).strip()
                if block_type == BlockType.ACTION:
                    trace.blocks.append(ParsedBlock(
                        block_type=block_type,
                        action_name=content,
                        action_input=content,
                        content=content,
                        raw_span=match.span(),
                    ))
                else:
                    trace.blocks.append(ParsedBlock(
                        block_type=block_type,
                        content=content,
                        raw_span=match.span(),
                    ))

        # 按出现位置排序
        trace.blocks.sort(key=lambda b: b.raw_span[0])
        return trace

    def extract_final_answer(self, raw_output: str) -> Optional[str]:
        trace = self.parse(raw_output)
        return trace.final_answer


# ============================================================
# 统一解析入口
# ============================================================

class StateMachineParser:
    """
    统一的状态机解析入口。

    自动检测输出范式并选择合适的解析器。

    使用方式:
        parser = StateMachineParser()
        trace = parser.parse(llm_output)
        print(f"推理步骤: {len(trace.thoughts)}")
        print(f"工具调用: {len(trace.actions)}")
        print(f"最终答案: {trace.final_answer}")
    """

    def __init__(self, default_mode: str = "auto"):
        """
        Args:
            default_mode: 默认解析模式
                - "auto": 自动检测
                - "react": 强制使用 ReAct 解析器
                - "xml": 强制使用 XML 解析器
        """
        self.default_mode = default_mode
        self._react_parser = ReActParser()
        self._xml_parser = XMLTagParser()

    def detect_format(self, raw_output: str) -> str:
        """
        自动检测输出范式。

        检测逻辑:
        - 如果包含 <thinking> 或 <answer> 标签 → "xml"
        - 如果包含 "Thought:" 或 "Action:" 关键字 → "react"
        - 否则 → "react"（默认回退）
        """
        # 检测 XML 标签
        if re.search(r"<(thinking|answer|action)>", raw_output, re.IGNORECASE):
            return "xml"

        # 检测 ReAct 关键字
        if re.search(r"(Thought|Action)[：:]", raw_output, re.IGNORECASE):
            return "react"

        # 默认回退到 ReAct（兼容性最好）
        return "react"

    def parse(self, raw_output: str, mode: Optional[str] = None) -> ParsedTrace:
        """
        统一解析入口。

        Args:
            raw_output: LLM 原始输出
            mode: 解析模式（None 则自动检测）

        Returns:
            ParsedTrace
        """
        mode = mode or self.default_mode
        if mode == "auto":
            mode = self.detect_format(raw_output)

        if mode == "xml":
            return self._xml_parser.parse(raw_output)
        else:
            return self._react_parser.parse(raw_output)

    def extract_final_answer(
        self, raw_output: str, mode: Optional[str] = None
    ) -> Optional[str]:
        """便捷方法：提取最终答案"""
        trace = self.parse(raw_output, mode)
        return trace.final_answer


# ============================================================
# 辅助：从 CoTTrace 升级为 ParsedTrace
# ============================================================

def upgrade_cot_trace(raw_output: str) -> ParsedTrace:
    """
    将原始 LLM 输出升级为 ParsedTrace。

    这是一个便捷函数，供已有代码快速接入状态机解析。
    """
    parser = StateMachineParser()
    return parser.parse(raw_output)
