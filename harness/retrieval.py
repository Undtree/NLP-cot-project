"""
检索模块 (Retrieval)
====================
轻量级 BM25 检索器，为 IRCoT 等多跳 QA 策略提供段落检索能力。

移植自 IRCoT 原版 src/retrieval.py，适配 Harness 框架。
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    """简单分词：提取字母数字 token 并转小写"""
    return TOKEN_RE.findall(text.lower())


class SimpleBM25:
    """轻量 BM25 实现，无外部搜索依赖。

    用于对 HotpotQA 样本中的 context 段落建索引并检索。

    Args:
        documents: 文档列表，每个文档为 {"id": str, "title": str, "text": str}
        k1: BM25 k1 参数 (默认 1.5)
        b: BM25 b 参数 (默认 0.75)
    """

    def __init__(self, documents: Iterable[Dict[str, str]], k1: float = 1.5, b: float = 0.75):
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self.doc_freq = defaultdict(int)
        self.term_freqs: List[Counter[str]] = []
        self.doc_lens: List[int] = []

        for doc in self.documents:
            tokens = tokenize(f"{doc.get('title', '')} {doc.get('text', '')}")
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            self.doc_lens.append(len(tokens))
            for term in tf:
                self.doc_freq[term] += 1

        self.num_docs = len(self.documents)
        self.avgdl = sum(self.doc_lens) / max(1, self.num_docs)
        self.idf = {
            term: math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))
            for term, df in self.doc_freq.items()
        }

    def score(self, query: str, idx: int) -> float:
        """计算 query 对第 idx 篇文档的 BM25 得分"""
        query_terms = tokenize(query)
        if not query_terms:
            return 0.0

        tf = self.term_freqs[idx]
        dl = self.doc_lens[idx]
        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            freq = tf[term]
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
            score += self.idf.get(term, 0.0) * numerator / denominator
        return score

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        """检索 top_k 篇最相关的文档。

        Returns:
            文档列表，每项含 id, title, text, score
        """
        scored = []
        for idx, doc in enumerate(self.documents):
            s = self.score(query, idx)
            if s > 0:
                item = dict(doc)
                item["score"] = s
                scored.append(item)
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return self.num_docs


def merge_unique_passages(
    existing: List[Dict[str, str]],
    new_items: List[Dict[str, str]],
    limit: int,
) -> List[Dict[str, str]]:
    """合并两批检索结果，按 id 去重，不超过 limit。

    Args:
        existing: 已有的段落列表
        new_items: 新检索到的段落列表
        limit: 合并后最大段落数

    Returns:
        去重合并后的段落列表
    """
    seen = {item["id"] for item in existing}
    merged = list(existing)
    for item in new_items:
        if item["id"] in seen:
            continue
        merged.append(item)
        seen.add(item["id"])
        if len(merged) >= limit:
            break
    return merged


def format_passages(passages: List[Dict[str, str]], max_chars_per_passage: int = 900) -> str:
    """将段落列表格式化为模型可读的文本。

    格式:
        [1] Wikipedia Title: <title>
        <text>

    Args:
        passages: 段落列表
        max_chars_per_passage: 每段最大字符数（截断）

    Returns:
        格式化的段落文本
    """
    chunks = []
    for i, passage in enumerate(passages, 1):
        text = passage["text"][:max_chars_per_passage]
        chunks.append(f"[{i}] Wikipedia Title: {passage['title']}\n{text}")
    return "\n\n".join(chunks)
