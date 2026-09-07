"""
向量化模型 —— 将文本映射到高维向量空间

学习要点：
- Embedding 将文本转换为固定维度的向量，使语义相似的文本在向量空间中距离更近
- BAAI/bge-m3 是中文 RAG 主流选择（1024 维，支持 8192 上下文）
- 模型名可通过 .env 的 EMBED_MODEL_NAME 配置，默认 BAAI/bge-m3
"""

import logging
import numpy as np
from functools import lru_cache

from config import EMBED_MODEL_NAME


@lru_cache(maxsize=1)
def get_embed_model():
    """
    获取向量化模型（单例 + 缓存）

    首次调用时加载模型，后续调用直接返回缓存的实例。
    """
    import os
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    # 强制刷新 huggingface_hub 常量，防止其他模块提前导入导致镜像不生效
    try:
        import huggingface_hub.constants
        huggingface_hub.constants.ENDPOINT = 'https://hf-mirror.com'
        huggingface_hub.constants.HUGGINGFACE_CO_URL_TEMPLATE = (
            'https://hf-mirror.com/{repo_id}/resolve/{revision}/{filename}'
        )
    except Exception:
        pass

    from sentence_transformers import SentenceTransformer
    logging.info(f"加载向量化模型: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    logging.info(f"向量化模型加载完成，输出维度: {dim}")
    return model


def encode_texts(texts, show_progress=False):
    """
    将文本列表编码为向量

    Args:
        texts: 文本列表
        show_progress: 是否显示进度条

    Returns:
        numpy 数组，形状为 (n_texts, embedding_dim)
    """
    model = get_embed_model()
    embeddings = model.encode(texts, show_progress_bar=show_progress)
    return np.array(embeddings).astype('float32')


def encode_query(query):
    """
    将单个查询文本编码为向量

    Returns:
        numpy 数组，形状为 (1, embedding_dim)
    """
    model = get_embed_model()
    embedding = model.encode([query])
    return np.array(embedding).astype('float32')
