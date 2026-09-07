"""
文档处理管线 —— 提取 → 分块 → 向量化 → 构建索引

将 rag_demo.process_multiple_files 的核心逻辑抽取到此模块，
供 Gradio UI 和 FastAPI 服务共用，避免 API 服务被迫加载整个 Gradio。
"""

import logging
import os
import time

from core.document_loader import extract_text
from core.text_splitter import split_parent_child
from core.embeddings import encode_texts
from core.vector_store import vector_store
from core.bm25_index import bm25_manager

logger = logging.getLogger(__name__)


def _default_progress(ratio, desc=""):
    """默认进度回调：仅打印日志"""
    logger.info("进度 %.0f%%: %s", ratio * 100, desc)


def process_files(file_paths, progress_callback=None):
    """
    处理多个文件：提取文本 → 父子分块 → 向量化 → 构建 FAISS/BM25 索引

    Args:
        file_paths: 文件路径列表
        progress_callback: 进度回调函数 progress_callback(ratio: float, desc: str)
                           不传则用默认日志回调

    Returns:
        (result_text, file_display_names)
        - result_text: 处理结果汇总文本
        - file_display_names: 文件名显示列表
    """
    if progress_callback is None:
        progress_callback = _default_progress

    if not file_paths:
        return "请选择要上传的文件(支持PDF, Word, Excel, PPT, TXT, Markdown等)", []

    try:
        progress_callback(0.1, desc="清理历史数据...")
        vector_store.clear()
        bm25_manager.clear()

        total_files = len(file_paths)
        processed_results = []
        all_chunks, all_metadatas, all_ids = [], [], []
        all_parents = {}

        for idx, file_path in enumerate(file_paths, 1):
            try:
                file_name = os.path.basename(file_path)
                progress_callback((idx - 1) / total_files, desc=f"处理文件 {idx}/{total_files}: {file_name}")

                text = extract_text(file_path)
                if not text:
                    raise ValueError("文档内容为空或无法提取文本")

                doc_id = f"doc_{int(time.time())}_{idx}"
                # 父子分块：子块用于检索，父块喂给 LLM
                chunks, child_metas, parents_map = split_parent_child(text, doc_id=doc_id)
                for meta in child_metas:
                    meta["source"] = file_name
                    meta["doc_id"] = doc_id
                chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]

                all_chunks.extend(chunks)
                all_metadatas.extend(child_metas)
                all_ids.extend(chunk_ids)
                all_parents.update(parents_map)
                processed_results.append(
                    f"✅ {file_name}: 成功处理 {len(chunks)} 个子块（{len(parents_map)} 个父块）"
                )

            except Exception as e:
                logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
                processed_results.append(f"❌ {file_name}: 处理失败 - {str(e)}")

        if all_chunks:
            progress_callback(0.8, desc="生成文本嵌入...")
            embeddings = encode_texts(all_chunks, show_progress=True)

            progress_callback(0.9, desc="构建FAISS索引...")
            vector_store.build_index(all_chunks, all_ids, all_metadatas, embeddings, parents_map=all_parents)

        progress_callback(0.95, desc="构建BM25检索索引...")
        bm25_manager.build_index(all_chunks, all_ids)

        summary = f"\n总计处理 {total_files} 个文件，{len(all_chunks)} 个子块"
        processed_results.append(summary)
        display_names = [f"📄 {os.path.basename(p)}" for p in file_paths]
        return "\n".join(processed_results), display_names

    except Exception as e:
        logger.error(f"处理过程出错: {str(e)}")
        return f"处理过程出错: {str(e)}", []
