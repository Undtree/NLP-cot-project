"""
统一样本表示 (Sample)
=====================
跨数据集的通用样本数据结构，替代原始 dict 传递，
使得 BaseTask 可以访问数据集特定的元数据（如 context、supporting_titles 等）。

使用方式:
    from harness.sample import Sample

    sample = Sample(
        id="hotpotqa_001",
        question="Who won the Nobel Prize in 2023?",
        ground_truth="John Doe",
        metadata={"context": [...], "supporting_titles": [...]},
    )
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Sample:
    """跨数据集的统一样本表示。

    Attributes:
        id: 样本唯一标识
        question: 问题文本（纯问题，不包含选项等附加内容）
        ground_truth: 标准答案，类型因数据集而异
            - AQuA: "A"/"B"/"C"/"D"/"E"
            - HotpotQA: 自由文本答案
        metadata: 数据集特定的附加字段
            - AQuA: {"options": [...], "rationale": "..."}
            - HotpotQA: {"context": [...], "supporting_titles": [...]}
    """
    id: str = ""
    question: str = ""
    ground_truth: Any = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def options(self) -> List[str]:
        """获取选项列表（仅 AQuA 等选择题数据集）"""
        return self.metadata.get("options", [])

    @property
    def rationale(self) -> str:
        """获取参考推理过程"""
        return self.metadata.get("rationale", "")

    @property
    def context(self) -> List[Dict[str, str]]:
        """获取上下文段落列表（仅 HotpotQA 等 QA 数据集）"""
        return self.metadata.get("context", [])

    @property
    def supporting_titles(self) -> List[str]:
        """获取支持证据的标题列表（仅 HotpotQA）"""
        return self.metadata.get("supporting_titles", [])

    @classmethod
    def from_aqua_dict(cls, d: Dict[str, Any]) -> "Sample":
        """从 AQuA 格式的 dict 构建 Sample"""
        return cls(
            id=d.get("id", ""),
            question=d.get("question", ""),
            ground_truth=d.get("ground_truth", ""),
            metadata={
                "options": d.get("options", []),
                "rationale": d.get("rationale", ""),
            },
        )

    @classmethod
    def from_hotpotqa_dict(cls, d: Dict[str, Any]) -> "Sample":
        """从 HotpotQA 格式的 dict 构建 Sample"""
        return cls(
            id=str(d.get("id", "")),
            question=d.get("question", ""),
            ground_truth=d.get("answer", ""),
            metadata={
                "context": d.get("context", []),
                "supporting_titles": d.get("supporting_titles", []),
                "raw": d.get("raw", {}),
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "id": self.id,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "metadata": self.metadata,
        }
