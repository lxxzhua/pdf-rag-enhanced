"""
评测运行入口 —— 构建索引 → 逐题检索 → 计算指标

用法（在项目根目录）：
    python -m evaluation.run_eval
    python -m evaluation.run_eval --docs "文档A.pdf" "文档B.docx" --ks 1 2 3 5

流程说明：
1. 将评测文档走与生产完全一致的管线（提取 → 分块 → 嵌入 → FAISS/BM25 索引）
2. 每个问题执行"单轮混合检索 + 重排序"（不含递归改写，纯测检索质量）
3. 与 dataset.json 中的 gold_chunk 比对，输出 Hit Rate@k / MRR@k

注意：基线数字请使用人工校对后的数据集（status='reviewed'），
草稿条目只用于管道冒烟测试。
"""

import argparse
import logging
import os
import time

from config import RERANK_TOP_K, RETRIEVAL_TOP_K
from core.bm25_index import bm25_manager
from core.document_loader import extract_text
from core.embeddings import encode_query, encode_texts
from core.generator import query_answer  # noqa: F401  导入以保持与生产一致的初始化路径
from core.retriever import hybrid_merge, rerank_results
from core.text_splitter import split_text
from core.vector_store import vector_store
from evaluation.retrieval_metrics import evaluate_retrieval, format_report, load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_eval")

DEFAULT_DOCS = ["挖掘机维修案例(样例）.pdf"]
DEFAULT_DATASET = os.path.join(os.path.dirname(__file__), "dataset.json")


def build_index_from_docs(doc_paths):
    """用与 rag_demo.process_multiple_files 完全一致的管线构建索引（不依赖 Gradio）"""
    vector_store.clear()
    bm25_manager.clear()

    all_chunks, all_metadatas, all_ids = [], [], []
    for idx, path in enumerate(doc_paths, 1):
        text = extract_text(path)
        if not text:
            raise ValueError(f"文档内容为空或无法提取: {path}")
        chunks = split_text(text)
        doc_id = f"doc_{int(time.time())}_{idx}"
        all_chunks.extend(chunks)
        all_metadatas.extend({"source": os.path.basename(path), "doc_id": doc_id} for _ in chunks)
        all_ids.extend(f"{doc_id}_chunk_{i}" for i in range(len(chunks)))
        logger.info("已处理 %s: %d 个文本块", path, len(chunks))

    if not all_chunks:
        raise ValueError("没有生成任何文本块")

    embeddings = encode_texts(all_chunks, show_progress=True)
    vector_store.build_index(all_chunks, all_ids, all_metadatas, embeddings)
    bm25_manager.build_index(all_chunks, all_ids)
    logger.info("索引构建完成: %d 个文本块", len(all_chunks))
    return all_chunks


def retrieve_once(query, top_k=RETRIEVAL_TOP_K):
    """
    单轮检索：语义 + BM25 混合 → 重排序

    与 recursive_retrieval 第一轮逻辑一致，但不含 LLM 查询改写，
    用于纯粹度量检索层的质量。
    """
    query_embedding = encode_query(query)
    sem_docs, sem_ids, sem_metas = vector_store.search(query_embedding, k=top_k)
    prepared = {"ids": [sem_ids], "documents": [sem_docs], "metadatas": [sem_metas]}
    bm25_res = bm25_manager.search(query, top_k=top_k) if bm25_manager.bm25_index else []

    hybrid = hybrid_merge(prepared, bm25_res)
    ids_iter = [doc_id for doc_id, _ in hybrid[:top_k]]
    docs_iter = [data["content"] for _, data in hybrid[:top_k]]
    meta_iter = [data["metadata"] for _, data in hybrid[:top_k]]

    if not docs_iter:
        return []

    try:
        reranked = rerank_results(query, docs_iter, ids_iter, meta_iter, top_k=RERANK_TOP_K)
    except Exception as e:
        logger.error("重排序失败，使用混合排序结果: %s", e)
        reranked = [(did, {"content": d, "metadata": m, "score": 1.0})
                    for did, d, m in zip(ids_iter, docs_iter, meta_iter)]

    return [data["content"] for _, data in reranked]


def main():
    parser = argparse.ArgumentParser(description="RAG 检索层评测")
    parser.add_argument("--docs", nargs="+", default=DEFAULT_DOCS, help="评测文档路径")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="评测数据集 JSON 路径")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3, 5], help="Hit Rate 的 k 值")
    parser.add_argument("--include-draft", action="store_true", help="包含未校对的草稿条目（仅冒烟测试用）")
    args = parser.parse_args()

    qa_items = load_dataset(args.dataset, include_draft=args.include_draft)
    if not qa_items:
        logger.error("没有可用的评测条目，退出")
        return

    logger.info("加载 %d 条评测数据 (k=%s)", len(qa_items), args.ks)
    build_index_from_docs(args.docs)

    results = evaluate_retrieval(qa_items, retrieve_once, ks=tuple(args.ks))
    print(format_report(results, ks=tuple(args.ks)))

    report_path = os.path.join(os.path.dirname(__file__), "baseline_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        import json
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("详细结果已写入 %s", report_path)


if __name__ == "__main__":
    main()
