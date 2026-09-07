"""
评测模块 —— 量化 RAG 系统的检索与生成质量

分层设计：
- 第一层（本模块）：纯检索指标 Hit Rate@k / MRR@k，确定性计算，不依赖 LLM
- 第二层（后续接入）：RAGAS 评测 Faithfulness / Context Precision，LLM 判分
"""
