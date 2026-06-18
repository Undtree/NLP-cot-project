"""
数据集加载器 (Dataset Loader)
-----------------------------
负责统一下载、清洗和格式化数据集。
目前主要支持 AQuA 数据集格式，可扩展支持其他数据集。

AQuA 数据集格式要求:
    每个样本为一个 JSON 对象，包含:
    - question: str          # 问题文本
    - options: list[str]     # 选项列表 (可选)
    - correct: str           # 正确答案
    - rationale: str         # 推理过程 (可选，用于训练/参考)

统一输出格式:
    {
        "id": str,              # 样本唯一标识
        "question": str,        # 完整问题文本 (含选项)
        "ground_truth": str,    # 标准答案
        "options": list[str],   # 选项列表
        "rationale": str,       # 参考推理过程
    }
"""

import json
import os
import re
import hashlib
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class AQuADataset:
    """AQuA 数据集的标准化数据结构"""
    samples: List[Dict[str, Any]] = field(default_factory=list)
    name: str = "aqua"
    split: str = "test"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)

    def get_questions_only(self) -> List[str]:
        """只返回问题文本列表，方便组员直接调用"""
        return [s["question"] for s in self.samples]

    def get_ground_truths(self) -> List[str]:
        """只返回标准答案列表"""
        return [s["ground_truth"] for s in self.samples]


def _generate_sample_id(question: str, index: int) -> str:
    """为样本生成唯一 ID"""
    hash_str = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
    return f"aqua_{index:04d}_{hash_str}"


def _build_full_question(question: str, options: Optional[List[str]] = None) -> str:
    """
    构建包含选项的完整问题文本。
    如果提供了选项，以 A. B. C. D. E. 的格式拼接。
    """
    if not options:
        return question.strip()

    labels = [chr(ord("A") + i) for i in range(len(options))]
    options_text = "\n".join(
        f"{label}. {opt}" for label, opt in zip(labels, options)
    )
    return f"{question.strip()}\n{options_text}"


def _validate_sample(sample: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    """
    验证并清洗单个样本。
    返回清洗后的样本字典；如果样本无效则返回 None。
    """
    # 必需字段检查
    question = sample.get("question", "").strip()
    if not question:
        print(f"[警告] 样本 #{index} 缺少 question 字段，已跳过")
        return None

    correct = sample.get("correct", "").strip()
    if not correct:
        print(f"[警告] 样本 #{index} 缺少 correct 字段，已跳过")
        return None

    # 可选字段
    options = sample.get("options", [])
    if isinstance(options, str):
        # 有些数据集的 options 是字符串，尝试解析
        options = [o.strip() for o in options.split(",") if o.strip()]

    rationale = sample.get("rationale", "")

    return {
        "id": _generate_sample_id(question, index),
        "question": _build_full_question(question, options),
        "ground_truth": correct,
        "options": options,
        "rationale": rationale.strip() if rationale else "",
    }


def load_aqua_json(filepath: str) -> AQuADataset:
    """
    从 JSON 文件加载 AQuA 格式的数据集。

    支持的 JSON 格式:
    1. 顶层为列表: [{"question": ..., "correct": ...}, ...]
    2. 顶层为字典且含 "samples" 或 "data" 键: {"samples": [...]}
    3. 每行一个 JSON 的 .jsonl 格式会自动检测

    Args:
        filepath: JSON/JSONL 文件路径

    Returns:
        AQuADataset 对象

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: JSON 格式错误或无法解析
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据集文件不存在: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    # 尝试解析 JSON
    samples_raw = []
    try:
        data = json.loads(raw_text)
        if isinstance(data, list):
            samples_raw = data
        elif isinstance(data, dict):
            # 尝试常见键名
            for key in ["samples", "data", "items", "examples"]:
                if key in data:
                    samples_raw = data[key]
                    break
            else:
                # 整个字典本身可能就是一个样本
                samples_raw = [data]
    except json.JSONDecodeError:
        # 尝试按 .jsonl 格式逐行解析
        samples_raw = []
        for line in raw_text.split("\n"):
            line = line.strip()
            if line:
                try:
                    samples_raw.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"[警告] 无法解析行: {line[:80]}...")

    if not samples_raw:
        raise ValueError(f"未能从 {filepath} 中解析出任何样本数据")

    # 验证和清洗每个样本
    cleaned_samples = []
    skipped = 0
    for i, sample in enumerate(samples_raw):
        cleaned = _validate_sample(sample, i)
        if cleaned:
            cleaned_samples.append(cleaned)
        else:
            skipped += 1

    if skipped > 0:
        print(f"[信息] 共跳过 {skipped} 个无效样本")

    if not cleaned_samples:
        raise ValueError("清洗后没有有效样本，请检查数据集格式")

    # 推断 split 名称
    split = "test"
    basename = os.path.basename(filepath).lower()
    if "train" in basename:
        split = "train"
    elif "dev" in basename or "val" in basename:
        split = "dev"

    print(f"[信息] 成功加载 {len(cleaned_samples)} 个样本 (split={split}) 从 {filepath}")

    return AQuADataset(samples=cleaned_samples, split=split)


def load_dataset(filepath: str, dataset_type: str = "aqua") -> AQuADataset:
    """
    统一的数据集加载入口。
    可根据 dataset_type 扩展支持不同数据集格式。

    Args:
        filepath: 数据集文件路径
        dataset_type: 数据集类型，目前支持 "aqua"

    Returns:
        AQuADataset 对象
    """
    if dataset_type == "aqua":
        return load_aqua_json(filepath)
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}。目前仅支持 'aqua'。")


# ---------- 便捷函数：下载 AQuA 数据集 ----------

AQUA_DOWNLOAD_URL = "https://raw.githubusercontent.com/deepmind/AQuA/master/test.json"


def download_aqua_dataset(save_dir: str = "data") -> str:
    """
    从 GitHub 下载 AQuA 数据集到本地。
    如果本地已存在则跳过下载。

    Args:
        save_dir: 保存目录

    Returns:
        本地文件路径
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "aqua_test.json")

    if os.path.exists(save_path):
        print(f"[信息] AQuA 数据集已存在于 {save_path}，跳过下载")
        return save_path

    print(f"正在从 {AQUA_DOWNLOAD_URL} 下载 AQuA 数据集...")
    try:
        import urllib.request
        urllib.request.urlretrieve(AQUA_DOWNLOAD_URL, save_path)
        print(f"[信息] 下载完成，保存至 {save_path}")
    except Exception as e:
        print(f"[错误] 下载失败: {e}")
        print("请手动下载 AQuA 数据集并放置到 data/ 目录下")
        raise

    return save_path
