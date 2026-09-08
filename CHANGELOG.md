# Changelog

All notable changes to this project are documented here.

## [3.0.0] - 2026-09-08

二次开发大版本：以「评测先行、逐阶段迭代、同尺对比」方法论完成检索质量、分块策略、溯源能力与工程化改造。

### Added

- **评测体系（`evaluation/`）**：LLM 生成 QA 草稿 → 人工校对入库（13 条）→ 脚本计算 Hit Rate@k / MRR，各阶段报告（`baseline_report.json` / `stage1_report.json` / `stage2_report.json`）入库可追溯。
- **中文语义检索全家桶**：嵌入切换为 BGE 中文系列，重排序器修复为 `BAAI/bge-reranker-v2-m3` 官方交叉编码器；模型名全部支持 `.env` 配置。
- **父子分块（Parent-Child Chunking）**：子块 400 字符用于向量检索，命中后回溯父块（1500 字符、按段落边界切分）喂给 LLM，按父块去重并控制上下文上限（5000 字符）。
- **引用溯源体系**：检索上下文注入 `[1]..[N]` 编号，回答完成后做引用后校验删除假引用；来源以结构化 `sources`（ref_id / 文档名 / 父块全文 / doc_id）随 API 返回；前端点击 `[n]` 弹出引用详情。
- **FastAPI 生产化服务（`api_router.py`）**：问答（阻塞 / SSE 流式）、上传（SSE 实时进度回调：清理 → 嵌入 → 索引 → BM25 → 完成）、会话管理（`session_id` 多轮记忆 + TTL 自动清理）、`/api/chunks` 分块可视化接口；直接挂载前端静态文件。
- **Vue 3 前端（`frontend/`）**：CDN 免构建单文件，支持流式渲染、引用弹窗、上传进度条、父子分块可视化、已处理文件分区展示。
- **CPU 性能优化**：FP16 模型加载（常驻内存 3.6 GB → 472 MB，−87%）、服务启动时预加载嵌入模型（首次上传 30s+ → ~2s）、默认轻量模型 `bge-base-zh-v1.5`（大 PDF 编码 ~34 min → ~7.5 min，提速 4.5 倍）。
- **检索质量跃升**（13 条人工校对 QA，同尺对比）：Hit Rate@1 7.69% → 100%，MRR@5 0.3167 → 1.0000。

### Fixed

- 上游真 bug：`distiluse-v2` 双塔模型被当作交叉编码器使用（分类头随机初始化，重排序结果近乎随机）。
- 并发上传导致向量索引互相污染：新增上传互斥锁强制串行化。
- Windows 下文件覆盖 `FileExistsError`：改用 `os.replace`。
- 引用 `[n]` 点击无响应：改为事件委托 + 结构化 sources 驱动。
- 上传文件名与来源显示不一致：上传时保留原始文件名。
- `bge-*-zh-v1.5` 查询侧官方指令前缀缺失。

### Changed

- 文档处理管线抽取到 `core/pipeline.py`，API 服务与 Gradio 彻底解耦。
- 默认嵌入模型：`BAAI/bge-base-zh-v1.5`（CPU 友好），追求极致精度可在 `.env` 配置 `BAAI/bge-m3`。
- 重写 README（项目背景 / 架构图 / 创新点 / 指标对比 / 截图 / 部署步骤）。

## [2.1.0] - 2026-08-12

### Added

- MIT license recognized by GitHub.
- English README and a concise Chinese project guide.
- Automated tests for configuration, document loading, hybrid retrieval, and missing-key behavior.
- GitHub Actions CI for source compilation and tests.
- Contribution, security, conduct, issue, and pull request guidance.
- A current application screenshot and centralized version metadata.

### Changed

- Repositioned the repository as a transparent educational and reference RAG implementation.
- Clarified supported document types, setup steps, provider choices, and known limitations.
- Added the runtime dependencies required for Excel parsing.
- Updated Gradio support to the 6.x line used by the current interface.

### Removed

- Commercial book, course, community, and store promotion from the repository.

## [2.0.0] - 2026-03-18

### Added

- Modular `core/` and `features/` structure.
- Gradio 6.x compatibility updates.
- Configurable model names and provider selection.
- FAISS and BM25 hybrid retrieval pipeline.
