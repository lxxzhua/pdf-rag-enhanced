"""
文本分块器 —— 将长文本切分为检索友好的片段

学习要点：
- chunk_size：每个片段的最大字符数。过大则检索粒度粗，过小则上下文缺失
- chunk_overlap：相邻片段的重叠字符数。避免关键信息被切断
- separators：按优先级尝试的分割符。中文文档应包含中文标点
- 父子分块（Parent-Child Chunking）：小块（子块）用于检索，大块（父块）喂给 LLM
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP, PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP


def split_text(text, chunk_size=None, chunk_overlap=None):
    """
    将长文本切分为多个片段

    使用 RecursiveCharacterTextSplitter 递归切分：
    先尝试按段落分割，若片段仍过大则按句子分割，以此类推。

    Args:
        text: 待切分的长文本
        chunk_size: 每个片段的最大字符数（默认使用配置值 400）
        chunk_overlap: 相邻片段的重叠字符数（默认使用配置值 40）

    Returns:
        切分后的文本片段列表
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or CHUNK_SIZE,
        chunk_overlap=chunk_overlap or CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", "；", "：", " ", ""]
    )
    return text_splitter.split_text(text)


def split_parent_child(text, doc_id="doc"):
    """
    父子分块：先生成父块（大块，保留上下文），再将每个父块切成子块（小块，用于检索）

    策略：
    - 父块：按章节/段落边界切（PARENT_CHUNK_SIZE，默认 1500 字符），保证语义完整
    - 子块：对每个父块用现有参数（CHUNK_SIZE=400）切，用于精准检索
    - 子块 metadata 中记录 parent_id，检索命中后可回溯父块

    Args:
        text: 待切分的长文本
        doc_id: 文档 ID，用于生成父子块的唯一标识

    Returns:
        (child_chunks, child_metadatas, parents_map)
        - child_chunks: 子块文本列表（用于建索引和检索）
        - child_metadatas: 子块元数据列表，每项含 parent_id
        - parents_map: {parent_id: parent_text} 父块文本映射
    """
    # 第一步：切父块（按段落/章节优先，保留完整语义单元）
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", ""]
    )
    parent_chunks = parent_splitter.split_text(text)

    child_chunks = []
    child_metadatas = []
    parents_map = {}

    for p_idx, parent_text in enumerate(parent_chunks):
        parent_id = f"{doc_id}_parent_{p_idx}"
        parents_map[parent_id] = parent_text

        # 第二步：将每个父块切成子块（用于检索）
        children = split_text(parent_text)
        for c_idx, child_text in enumerate(children):
            child_chunks.append(child_text)
            child_metadatas.append({
                "parent_id": parent_id,
                "parent_index": p_idx,
                "child_index": c_idx,
            })

    return child_chunks, child_metadatas, parents_map
