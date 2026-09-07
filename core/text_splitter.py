"""
文本分块器 —— 将长文本切分为检索友好的片段

学习要点：
- chunk_size：每个片段的最大字符数。过大则检索粒度粗，过小则上下文缺失
- chunk_overlap：相邻片段的重叠字符数。避免关键信息被切断
- separators：按优先级尝试的分割符。中文文档应包含中文标点
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


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
