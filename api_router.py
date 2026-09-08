"""
REST API 模块（FastAPI 实现）

提供文档上传、问答（阻塞/流式）、会话管理、状态查询等接口。
核心管线已抽取到 core/pipeline.py，API 服务不再依赖 Gradio。
"""
import json
import logging
import os
import re
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import asyncio
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import MAGICK_API_KEY, SILICONFLOW_API_KEY, is_configured_api_key
from core.generator import query_answer, stream_answer
from core.pipeline import process_files
from core.vector_store import vector_store
from features.web_search import check_serpapi_key
from utils.network import is_port_available
from version import __version__

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rag-api")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 会话管理（内存存储，多轮对话历史）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sessions: Dict[str, List[Dict[str, str]]] = {}
SESSION_TTL = 3600  # 会话过期时间（秒）
session_timestamps: Dict[str, float] = {}


def get_or_create_session(session_id: Optional[str]) -> str:
    """获取已有会话或创建新会话，返回 session_id"""
    if session_id and session_id in sessions:
        session_timestamps[session_id] = time.time()
        return session_id
    new_id = session_id or str(uuid.uuid4())
    sessions[new_id] = []
    session_timestamps[new_id] = time.time()
    return new_id


def cleanup_expired_sessions():
    """清理过期会话"""
    now = time.time()
    expired = [sid for sid, ts in session_timestamps.items() if now - ts > SESSION_TTL]
    for sid in expired:
        sessions.pop(sid, None)
        session_timestamps.pop(sid, None)
    if expired:
        logger.info(f"清理过期会话 {len(expired)} 个")


def build_history_context(session_id: str, max_turns: int = 5) -> str:
    """从会话历史构建多轮对话上下文文本"""
    history = sessions.get(session_id, [])
    if not history:
        return ""
    recent = history[-(max_turns * 2):]
    lines = []
    for msg in recent:
        role = "用户" if msg["role"] == "user" else "助手"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FastAPI 应用
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API 服务启动")
    yield
    logger.info("API 服务已关闭")


app = FastAPI(
    title="本地RAG API服务",
    description="提供基于本地大模型、云端模型服务和SERPAPI的文档问答API接口",
    version=__version__,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 请求/响应模型
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class QuestionRequest(BaseModel):
    question: str
    enable_web_search: bool = False
    model_choice: str = "siliconflow"
    session_id: Optional[str] = None


class AnswerResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    session_id: str
    metadata: Dict[str, Any]


class FileProcessResult(BaseModel):
    status: str
    message: str
    file_info: Optional[Dict[str, Any]] = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 接口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """处理文档并存入向量数据库（SSE 流式上传进度）"""
    from fastapi.responses import StreamingResponse

    # 先把文件读到原始文件名的临时路径（在线程外，避免阻塞）
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    original_dir = os.path.dirname(tmp_path)
    target_path = os.path.join(original_dir, file.filename)
    os.rename(tmp_path, target_path)
    tmp_path = target_path

    filename = file.filename

    async def upload_stream():
        import threading
        try:
            loop = asyncio.get_running_loop()

            def progress_cb(ratio, desc=""):
                try:
                    asyncio.run_coroutine_threadsafe(
                        queue.put({"ratio": ratio, "desc": desc}), loop
                    )
                except Exception:
                    pass

            queue: asyncio.Queue = asyncio.Queue()

            def run_processing():
                try:
                    result = process_files([tmp_path], progress_callback=progress_cb)
                    asyncio.run_coroutine_threadsafe(
                        queue.put({"result": result}), loop
                    )
                except Exception as exc:
                    asyncio.run_coroutine_threadsafe(
                        queue.put({"error": str(exc)}), loop
                    )

            thread = threading.Thread(target=run_processing, daemon=True)
            thread.start()

            while True:
                item = await queue.get()
                if "error" in item:
                    raise Exception(item["error"])
                if "result" in item:
                    os.unlink(tmp_path)
                    result_text = item["result"]
                    result = result_text[0] if isinstance(result_text, tuple) else result_text
                    chunk_match = re.search(r'(\d+) 个子块', result)
                    chunks = int(chunk_match.group(1)) if chunk_match else 0
                    payload = json.dumps({
                        "done": True,
                        "status": "success" if "成功" in result else "error",
                        "message": result,
                        "filename": filename,
                        "chunks": chunks
                    }, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
                    break
                payload = json.dumps({
                    "progress": item["ratio"],
                    "desc": item["desc"]
                }, ensure_ascii=False)
                yield f"data: {payload}\n\n"
        except Exception as e:
            logger.error(f"文件处理失败: {str(e)}")
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            payload = json.dumps({"done": True, "error": str(e)}, ensure_ascii=False)
            yield f"data: {payload}\n\n"

    return StreamingResponse(upload_stream(), media_type="text/event-stream")


@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(req: QuestionRequest):
    """问答接口（阻塞式，等待完整回答后返回）"""
    if not req.question:
        raise HTTPException(400, "问题不能为空")
    try:
        cleanup_expired_sessions()
        session_id = get_or_create_session(req.session_id)
        history = build_history_context(session_id)

        answer, sources = await asyncio.to_thread(
            query_answer, req.question, req.enable_web_search, req.model_choice, None, history
        )

        # 记录会话历史
        sessions[session_id].append({"role": "user", "content": req.question})
        sessions[session_id].append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "sources": sources,
            "session_id": session_id,
            "metadata": {"enable_web_search": req.enable_web_search, "model": req.model_choice}
        }
    except Exception as e:
        logger.error(f"问答失败: {str(e)}")
        raise HTTPException(500, f"问答处理失败: {str(e)}") from e


@app.post("/api/ask/stream")
async def ask_question_stream(req: QuestionRequest):
    """问答接口（SSE 流式输出，逐 token 返回）"""
    if not req.question:
        raise HTTPException(400, "问题不能为空")

    cleanup_expired_sessions()
    session_id = get_or_create_session(req.session_id)
    history = build_history_context(session_id)

    async def event_generator():
        full_answer = ""
        final_sources = []
        try:
            # stream_answer 现在 yield 三元组: (answer_chunk, status, sources_or_None)
            for answer_chunk, status, src in stream_answer(
                req.question, req.enable_web_search, req.model_choice, None, history
            ):
                full_answer = answer_chunk
                if src is not None:
                    final_sources = src
                payload = json.dumps({"answer": answer_chunk, "status": status}, ensure_ascii=False)
                yield f"data: {payload}\n\n"

            sessions[session_id].append({"role": "user", "content": req.question})
            sessions[session_id].append({"role": "assistant", "content": full_answer})

            # 结束事件带 sources，前端据此渲染引用详情
            end_payload = json.dumps(
                {"done": True, "session_id": session_id, "sources": final_sources},
                ensure_ascii=False
            )
            yield f"data: {end_payload}\n\n"
        except Exception as e:
            logger.error(f"流式问答失败: {str(e)}")
            err_payload = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {err_payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """获取会话历史"""
    if session_id not in sessions:
        raise HTTPException(404, "会话不存在")
    return {"session_id": session_id, "messages": sessions[session_id]}


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    sessions.pop(session_id, None)
    session_timestamps.pop(session_id, None)
    return {"status": "success", "message": "会话已删除"}


@app.get("/api/status")
async def check_status():
    return {
        "status": "healthy",
        "siliconflow_configured": is_configured_api_key(SILICONFLOW_API_KEY),
        "magick_configured": is_configured_api_key(MAGICK_API_KEY),
        "serpapi_configured": check_serpapi_key(),
        "vector_store_ready": vector_store.is_ready,
        "total_chunks": vector_store.total_chunks,
        "version": __version__
    }


# 静态前端挂载：必须放在所有 API 路由之后，/api/* 优先匹配
_frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
if os.path.isdir(_frontend_dir):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
    logger.info(f"前端已挂载: {_frontend_dir}")


if __name__ == "__main__":
    import uvicorn
    port = next((p for p in [17995, 17996, 17997, 17998, 17999] if is_port_available(p)), 17995)
    logger.info(f"启动API服务，端口: {port}，前端访问: http://localhost:{port}/")
    uvicorn.run(app, host="0.0.0.0", port=port)
