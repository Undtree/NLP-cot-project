"""
数据集加载器 (Dataset Loader)
-----------------------------
负责统一下载、清洗和格式化数据集。
支持 AQuA（数学选择题）和 HotpotQA（多跳问答）两种格式。

AQuA 数据集格式要求:
    每个样本为一个 JSON 对象，包含:
    - question: str          # 问题文本
    - options: list[str]     # 选项列表 (可选)
    - correct: str           # 正确答案
    - rationale: str         # 推理过程 (可选，用于训练/参考)

HotpotQA 数据集格式要求:
    每个样本包含:
    - question: str          # 问题文本
    - answer: str            # 标准答案
    - context: list          # 上下文段落 (title + sentences)
    - supporting_facts: list # 支持证据

统一输出格式:
    Sample 对象或 AQuADataset 对象
"""

import json
import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

from .sample import Sample


# ============================================================
# AQuA 数据集
# ============================================================

class AQuADataset:
    """AQuA 数据集的标准化数据结构"""
    def __init__(self, samples=None, name="aqua", split="test"):
        self.samples = samples or []
        self.name = name
        self.split = split

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)

    def get_questions_only(self) -> List[str]:
        """只返回问题文本列表"""
        return [s["question"] if isinstance(s, dict) else s.question for s in self.samples]

    def get_ground_truths(self) -> List[str]:
        """只返回标准答案列表"""
        return [s["ground_truth"] if isinstance(s, dict) else s.ground_truth for s in self.samples]

    def to_samples(self) -> List[Sample]:
        """将内部 dict 转换为 Sample 对象列表"""
        return [
            s if isinstance(s, Sample) else Sample.from_aqua_dict(s)
            for s in self.samples
        ]


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


def load_dataset(filepath: str, dataset_type: str = "aqua", **kwargs) -> Any:
    """
    统一的数据集加载入口。

    Args:
        filepath: 数据集文件路径
        dataset_type: 数据集类型
            - "aqua": AQuA 数学选择题 (默认)
            - "hotpotqa": HotpotQA 多跳问答
        **kwargs: 传递给具体加载器的参数
            - max_samples: 最大样本数
            - split: HotpotQA split (默认 "validation")

    Returns:
        - dataset_type="aqua": AQuADataset 对象
        - dataset_type="hotpotqa": (List[Sample], List[Dict]) 即 (samples, corpus)
    """
    if dataset_type == "aqua":
        return load_aqua_json(filepath)
    elif dataset_type == "hotpotqa":
        return load_hotpotqa(filepath, **kwargs)
    else:
        raise ValueError(
            f"不支持的数据集类型: {dataset_type}。"
            f"目前支持: 'aqua', 'hotpotqa'。"
        )


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


# ============================================================
# HotpotQA 数据集加载
# ============================================================

def _clean_text(text: Any) -> str:
    """清洗文本：合并多余空白"""
    return re.sub(r"\s+", " ", str(text)).strip()


def _extract_hotpotqa_context(record: Dict[str, Any]) -> List[Dict[str, str]]:
    """从 HotpotQA 记录中提取 context 段落"""
    context = record.get("context") or []
    paragraphs: List[Dict[str, str]] = []

    if isinstance(context, dict):
        titles = context.get("title") or context.get("titles") or []
        sentence_groups = context.get("sentences") or context.get("sentence") or []
        for title, sentences in zip(titles, sentence_groups):
            text = " ".join(map(str, sentences)) if isinstance(sentences, list) else str(sentences)
            text = _clean_text(text)
            if text:
                paragraphs.append({"title": str(title), "text": text})
        return paragraphs

    if isinstance(context, list):
        for item in context:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title = str(item[0])
                sentences = item[1]
                text = " ".join(map(str, sentences)) if isinstance(sentences, list) else str(sentences)
            elif isinstance(item, dict):
                title = str(item.get("title", ""))
                sentences = item.get("sentences", item.get("text", ""))
                text = " ".join(map(str, sentences)) if isinstance(sentences, list) else str(sentences)
            else:
                continue
            text = _clean_text(text)
            if text:
                paragraphs.append({"title": title, "text": text})

    return paragraphs


def _extract_hotpotqa_supporting_titles(record: Dict[str, Any]) -> List[str]:
    """从 HotpotQA 记录中提取 supporting_facts 的标题"""
    sf = record.get("supporting_facts") or record.get("supportingFacts") or []
    titles: List[str] = []

    if isinstance(sf, dict):
        raw_titles = sf.get("title") or sf.get("titles") or []
        titles = [str(t) for t in raw_titles]
    elif isinstance(sf, list):
        for item in sf:
            if isinstance(item, (list, tuple)) and item:
                titles.append(str(item[0]))
            elif isinstance(item, dict) and "title" in item:
                titles.append(str(item["title"]))

    seen = set()
    unique = []
    for t in titles:
        if t not in seen:
            unique.append(t)
            seen.add(t)
    return unique


def _normalize_hotpotqa_record(record: Dict[str, Any], index: int) -> Dict[str, Any]:
    """规范化 HotpotQA 记录为统一格式"""
    question = record.get("question")
    answer = record.get("answer")
    if question is None or answer is None:
        raise ValueError(f"Record {index} has no question or answer field")

    return {
        "id": str(record.get("_id", record.get("id", index))),
        "question": _clean_text(question),
        "answer": _clean_text(answer),
        "context": _extract_hotpotqa_context(record),
        "supporting_titles": _extract_hotpotqa_supporting_titles(record),
        "raw": record,
    }


def _read_hotpotqa_records(path: Path, max_records: Optional[int] = None) -> List[Dict[str, Any]]:
    """从本地文件读取 HotpotQA 记录（支持 .json / .jsonl / .parquet）"""
    if not path.exists():
        raise FileNotFoundError(f"HotpotQA 文件不存在: {path}")

    if path.suffix.lower() == ".parquet":
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("读取 .parquet 文件需要安装 pandas 和 pyarrow")
        return pd.read_parquet(path).to_dict(orient="records")

    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    # .json
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "examples", "records", "samples"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"不支持的 HotpotQA JSON 结构: {path}")


def build_corpus_from_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """从 HotpotQA 样本的 context 段落构建 BM25 检索语料。

    这是一个轻量本地语料方案，适用于课程项目。
    相比索引完整 Wikipedia，它只用样本自带的 context 段落。

    Args:
        samples: 已规范化的 HotpotQA 样本列表

    Returns:
        语料文档列表，每项为 {"id": str, "title": str, "text": str}
    """
    corpus: List[Dict[str, str]] = []
    seen = set()

    for sample in samples:
        for para in sample.get("context", []):
            title = para["title"]
            text = para["text"]
            key = (title, text)
            if key in seen:
                continue
            seen.add(key)
            corpus.append({
                "id": f"p{len(corpus)}",
                "title": title,
                "text": text,
            })

    if not corpus:
        raise ValueError(
            "未找到任何 context 段落。请使用 HotpotQA distractor 格式数据，"
            "或提供自定义语料构建器。"
        )
    return corpus


def load_hotpotqa(
    data_path: str,
    split: str = "validation",
    max_samples: Optional[int] = None,
) -> Tuple[List[Sample], List[Dict[str, str]]]:
    """加载 HotpotQA 数据集并返回 (samples, corpus)。

    支持本地 .json / .jsonl / .parquet 文件。

    Args:
        data_path: 数据文件路径
        split: 数据集 split（仅用于标识）
        max_samples: 最大样本数

    Returns:
        (Sample 列表, 语料文档列表)
    """
    records = _read_hotpotqa_records(Path(data_path), max_records=max_samples)

    if max_samples is not None:
        records = records[:max_samples]

    # 规范化
    normalized = [_normalize_hotpotqa_record(r, i) for i, r in enumerate(records)]

    # 构建语料
    corpus = build_corpus_from_samples(normalized)

    # 转换为 Sample 对象
    samples = [Sample.from_hotpotqa_dict(n) for n in normalized]

    print(f"[信息] 成功加载 {len(samples)} 个 HotpotQA 样本 (split={split})")
    print(f"[信息] 构建语料: {len(corpus)} 个段落")

    return samples, corpus
