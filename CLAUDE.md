# 项目开发规范与文档索引（CLAUDE.md）

> 本文档是小组与所有 AI Coding Agent 的**统一入口**。开工前请先通过下方索引按需读取相关文档，严格遵守协作与工程红线。

---

## 1. 文档导航索引

| 文档路径 | 核心内容 | 何时阅读 |
| :--- | :--- | :--- |
| [docs/proposal.md](docs/proposal.md) | 项目背景、目标、需求范围与非目标、成功标准、课程硬约束 | 首次上手或确认业务边界、交付标准时 |
| [docs/design.md](docs/design.md) | 系统架构图、模块职责、核心流程、概念数据模型、演进设计 | 设计新模块或修改跨模块交互前 |
| [docs/tasks.md](docs/tasks.md) | 带编号的实施任务清单（`T01`–`T18`）、依赖关系、验收标准与进度 | 领取任务、明确工作范围或更新进度时 |
| [docs/contracts/README.md](docs/contracts/README.md) | 模块间接口契约索引（开发过程中动态维护） | 对接或变更模块输入输出、错误语义时 |
| [docs/decisions/README.md](docs/decisions/README.md) | 关键架构取舍记录（模型、Embedding、Web 框架、Git 协作） | 了解技术选型背景或提出方案变更时 |

---

## 2. Git 协作与双仓库规范

本项目为**小组协作开发**，并需配合课程的「持续更新计中间能力提升分」机制：

- **小组 Fork 仓库（`origin`）**：`https://github.com/zhang040705/LLM-Course.git`
- **老师官方仓库（`upstream`）**：`https://github.com/CAI-testbooks/LLM-Course.git`

### 必守协作规则
1. **分支开发**：基于最新 `origin/main` 创建分支，禁止直接在 `main` 上提交未验证代码。
   - 命名：`feat/<成员>-<模块>`、`fix/<成员>-<问题>`、`exp/<版本>-<实验>`、`docs/<主题>`
2. **及时推送到 Fork 仓库**：每完成一个编号任务或可验证子功能，检查状态与自测后，使用 Conventional Commits（`feat/fix/exp/docs/refactor/chore`）提交并**及时 `push` 到 `origin`**，随后在小组仓库内通过 PR 合并至 `main`。
3. **定期同步至老师仓库**：按阶段里程碑（`V0`→`V1`→`V2`→`V3`→`V4`）或每周固定节奏，同步 `upstream/main` 后将小组稳定 `main` 更新至面向老师仓库的课程 PR。
   - PR 标题格式：`[作业01-NLP] 学号 姓名 - 领域特定RAG问答系统`
   - PR 描述须完整填写 `.github/pull_request_template.md` 全部字段。
4. **安全操作边界**：
   - 严禁对共享分支执行 `git push --force` 或覆盖远端历史。
   - **合并小组 `main` 及向老师 `upstream` 仓库推送/更新 PR 前，必须经人类组员确认，Agent 不得擅自推送 `upstream`。**

---

## 3. 关键开发红线（摘要）

- **按编号聚焦范围**：开工前核对 `docs/tasks.md` 指定任务编号（如 `T02`），只改相关范围，不顺手大改他人模块；完成后更新任务状态。
- **契约与文档同步**：跨模块接口有变更时，动态更新 `docs/contracts/` 对应文件，并在 Commit/PR 中显式告知组员；新增依赖或启动方式同步更新 `requirements.txt` 与 `README.md`。
- **架构解耦底线**：
  - LLM 走 OpenAI 兼容抽象（支持 DeepSeek 无缝切本地微调模型）。
  - Embedding 走统一接口（默认本地 `BAAI/bge-small-zh-v1.5`，兼容云端 API）。
  - RAG 核心逻辑全部下沉在 `src/rag/`，Gradio 与后期 FastAPI 仅作薄封装。
- **密钥与体积红线**：严禁提交 `.env` / 真实 API Key（如 `DEEPSEEK_API_KEY`）；严禁将原始数据集、向量库索引、模型权重直接提交 Git。
- **医疗业务红线**：严禁评测集数据泄漏进知识库；不得移除医疗免责声明；不得放松拒答机制或伪造引用编号。
