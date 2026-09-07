"""
检索评测指标 —— Hit Rate@k 与 MRR@k

学习要点：
- Hit Rate@k：top-k 检索结果中命中标准答案来源片段的问题比例（衡量召回能力）
- MRR@k：平均倒数排名，命中位置越靠前分数越高（衡量排序能力）
- 匹配判定：标准片段与检索片段互为包含即命中；否则用字符相似度兜底

这两个指标只依赖"问题 → 标准来源片段"标注数据，无需 LLM 判分，
结果确定、可复现，适合作为每次优化的回归基线。
"""

import json
import logging
import re
from difflib import SequenceMatcher


def normalize_text(text):
    """归一化文本：去除所有空白字符并转小写"""
    return re.sub(r"\s+", "", str(text)).lower()


def is_match(retrieved_text, gold_text, similarity_threshold=0.8):
    """
    判断检索片段是否命中标准片段

    匹配规则（宽松度递进）：
    1. 归一化后互为包含（LLM 生成的 gold_chunk 可能截断或拼接，双向包含都要查）
    2. 字符序列相似度 >= similarity_threshold（防止两端边界截断导致的假阴性）
    """
    if not gold_text:
        return False
    r, g = normalize_text(retrieved_text), normalize_text(gold_text)
    if not r:
        return False
    if g in r or r in g:
        return True
    return SequenceMatcher(None, g, r).ratio() >= similarity_threshold


def first_match_rank(retrieved_texts, gold_text, k):
    """
    返回 gold_text 在前 k 个检索结果中首次命中的排名（1-based），未命中返回 None
    """
    for rank, text in enumerate(retrieved_texts[:k], start=1):
        if is_match(text, gold_text):
            return rank
    return None


def evaluate_retrieval(qa_items, retrieve_fn, ks=(1, 2, 3, 5)):
    """
    批量评测检索质量

    Args:
        qa_items: 评测数据列表，每项需含 question 与 gold_chunk 字段
        retrieve_fn: 检索函数 (question) -> 按相关性降序的文本片段列表
        ks: 计算 Hit Rate 的 k 值列表

    Returns:
        {'hit_rate': {k: float}, 'mrr': float, 'details': [...]}
    """
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    reciprocal_ranks = []
    details = []

    for item in qa_items:
        question = item["question"]
        gold_chunk = item["gold_chunk"]
        retrieved_texts = retrieve_fn(question)

        ranks = {k: first_match_rank(retrieved_texts, gold_chunk, k) for k in ks}
        for k in ks:
            if ranks[k] is not None:
                hits[k] += 1

        first_rank = ranks[max_k]
        rr = 1.0 / first_rank if first_rank is not None else 0.0
        reciprocal_ranks.append(rr)

        details.append({
            "question": question,
            "hit_ranks": ranks,
            "mrr": rr,
            "hit": first_rank is not None,
        })

    total = len(qa_items)
    if total == 0:
        logging.warning("评测数据为空")
        return {"hit_rate": {k: 0.0 for k in ks}, "mrr": 0.0, "details": []}

    return {
        "hit_rate": {k: hits[k] / total for k in ks},
        "mrr": sum(reciprocal_ranks) / total,
        "details": details,
    }


def format_report(results, ks=(1, 2, 3, 5), title="检索评测结果"):
    """将评测结果格式化为可读报告"""
    lines = [f"===== {title} ====="]
    lines.append(f"评测问题数: {len(results['details'])}")
    lines.append("Hit Rate@k:")
    for k in ks:
        lines.append(f"  @{k}: {results['hit_rate'][k]:.2%}")
    lines.append(f"MRR@{max(ks)}: {results['mrr']:.4f}")

    missed = [d for d in results["details"] if not d["hit"]]
    if missed:
        lines.append(f"未命中问题 ({len(missed)} 条):")
        for d in missed:
            lines.append(f"  - {d['question']}")
    return "\n".join(lines)


def load_dataset(path, include_draft=False):
    """
    加载评测数据集

    默认只加载 status='reviewed' 的人工校对条目，保证基线可信。
    """
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    if include_draft:
        return items
    reviewed = [it for it in items if it.get("status") == "reviewed"]
    if not reviewed:
        logging.warning("数据集中没有 status='reviewed' 的条目，请先人工校对 build_dataset 的产出")
    return reviewed
