"""
QA 评测集生成 —— LLM 基于文档分块生成问答对草稿

用法（在项目根目录）：
    python -m evaluation.build_dataset
    python -m evaluation.build_dataset --docs "文档.pdf" --num-per-chunk 3

流程：
1. 用生产管线提取并分块文档
2. 对每个分块调用 LLM 生成问题与标准答案（仅基于该分块内容，保证可溯源）
3. 产出 dataset.json 草稿（status='draft'），等待人工校对后改为 'reviewed'

人工校对要点：
- 问题应自然、有检索价值（避免"这段话说了什么"式提问）
- 答案必须能仅凭对应分块得出
- gold_chunk 原则上不改（它决定检索命中判定），确需修改时保持原文子串
"""

import argparse
import json
import logging
import os
import re

from config import DEFAULT_MODEL_CHOICE
from core.document_loader import extract_text
from core.generator import call_cloud_api
from core.text_splitter import split_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_dataset")

DEFAULT_DOCS = ["挖掘机维修案例(样例）.pdf"]
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "dataset.json")

QA_PROMPT = """你是一个评测集构建助手。请仅基于下面的文档片段，生成 {num} 条问答对。

要求：
1. 问题必须是用户真实会问的自然问题，有检索价值，不要用"文中提到"之类的表述
2. 答案必须仅凭该片段内容即可得出，简洁准确
3. 严格输出 JSON 数组，不要输出任何其他文字或代码块标记
4. 格式：[{{"question": "...", "answer": "..."}}]

文档片段：
{chunk}
"""


def parse_llm_json(raw):
    """从 LLM 回复中解析 JSON 数组（容忍代码块标记与前后杂文本）"""
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1)
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"回复中未找到 JSON 数组: {raw[:100]}")
    return json.loads(raw[start:end + 1])


def generate_qa_for_chunk(chunk, num, model_choice):
    """对单个分块调用 LLM 生成问答对"""
    prompt = QA_PROMPT.format(num=num, chunk=chunk)
    raw = call_cloud_api(prompt, model_choice, temperature=0.3, max_tokens=1024)
    if "<think>" in raw:
        raw = raw.split("<think>")[0]
    qa_list = parse_llm_json(raw)
    return [qa for qa in qa_list if qa.get("question") and qa.get("answer")]


def build_dataset(doc_paths, num_per_chunk, out_path, model_choice):
    qa_items = []
    for path in doc_paths:
        text = extract_text(path)
        if not text:
            logger.warning("跳过（无法提取文本）: %s", path)
            continue
        chunks = split_text(text)
        logger.info("%s: %d 个分块，开始生成 QA（每块 %d 条）", path, len(chunks), num_per_chunk)

        for i, chunk in enumerate(chunks, 1):
            try:
                qa_list = generate_qa_for_chunk(chunk, num_per_chunk, model_choice)
            except Exception as e:
                logger.error("分块 %d QA 生成失败: %s", i, e)
                continue
            for qa in qa_list:
                qa_items.append({
                    "id": f"qa_{len(qa_items) + 1:03d}",
                    "question": qa["question"].strip(),
                    "answer": qa["answer"].strip(),
                    "gold_chunk": chunk,
                    "source": os.path.basename(path),
                    "status": "draft",
                })
            logger.info("分块 %d/%d 完成，累计 %d 条", i, len(chunks), len(qa_items))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(qa_items, f, ensure_ascii=False, indent=2)
    logger.info("草稿已写入 %s（共 %d 条，status='draft'）", out_path, len(qa_items))
    return qa_items


def main():
    parser = argparse.ArgumentParser(description="生成 QA 评测集草稿")
    parser.add_argument("--docs", nargs="+", default=DEFAULT_DOCS, help="用于生成 QA 的文档")
    parser.add_argument("--num-per-chunk", type=int, default=2, help="每个分块生成的问题数")
    parser.add_argument("--out", default=DEFAULT_OUT, help="输出 JSON 路径")
    parser.add_argument("--model", default=DEFAULT_MODEL_CHOICE, help="生成所用模型服务")
    args = parser.parse_args()

    build_dataset(args.docs, args.num_per_chunk, args.out, args.model)


if __name__ == "__main__":
    main()
