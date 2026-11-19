# 中文医疗领域特定 RAG 问答系统（Medical Domain-Specific QA）

LLM-Course 2026 方向 01（NLP 综合作业）项目仓库。  
基于 **CMedQA** 中文医疗问答数据集，实现支持**长上下文（>32k tokens）**、**多轮对话**、**可解释引用溯源**与**不确定性拒答**的领域 RAG 问答系统，并采用「MVP 快速跑通 → 分阶段量化迭代」的路线持续优化。

---

## 1. 项目文档导航

- 规范与协作入口：[CLAUDE.md](CLAUDE.md)
- 项目立项与范围：[docs/proposal.md](docs/proposal.md)
- 系统架构设计：[docs/design.md](docs/design.md)
- 编号任务清单：[docs/tasks.md](docs/tasks.md)
- 接口契约目录：[docs/contracts/README.md](docs/contracts/README.md)
- 架构决策记录：[docs/decisions/README.md](docs/decisions/README.md)

---

## 2. 目录结构

```text
.
├── CLAUDE.md                 # Agent 与团队协作规范索引
├── README.md                 # 环境配置、依赖安装与复现说明
├── .env.example              # 环境变量模板（复制为 .env 使用）
├── requirements.txt          # Python 依赖清单
├── configs/
│   └── config.yaml           # 模型、Embedding、检索与 RAG 全局配置
├── docs/                     # 立项、设计、任务、契约与架构决策文档
├── data/
│   └── samples/              # 轻量样例数据（raw/processed/vectorstore 不入库）
├── src/
│   ├── config.py             # 配置与环境变量加载器
│   ├── ingestion/            # 数据清洗、切分与元数据构建
│   ├── retrieval/            # Embedding、向量检索、BM25 与重排
│   ├── llm/                  # OpenAI 兼容 LLM 接入层（DeepSeek / 本地模型）
│   ├── rag/                  # RAG 核心编排服务
│   └── app/                  # Web Demo（MVP: Gradio → V4: FastAPI）
├── scripts/                  # 数据准备、评测、微调、Demo 启动脚本
├── tests/                    # 单元测试与冒烟测试
├── results/                  # 各版本实验评测输出与对比图表
└── report/                   # 最终实验报告
```

---

## 3. 环境配置与依赖安装

### 3.1 创建虚拟环境并安装依赖
建议使用 Python 3.10+（或兼容当前环境的 Python 3.9+）：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.2 配置环境变量
复制 `.env.example` 为 `.env`，并填入你的 DeepSeek API Key：

```bash
cp .env.example .env
# 编辑 .env，填写 DEEPSEEK_API_KEY
```

### 3.3 验证工程骨架
```bash
pytest -q
```

---

## 4. 全流程运行方式（随任务推进持续更新）

> 当前已完成 **V2 Retrieval（`T01`–`T12`）** 与 **V4 `T15`（FastAPI 服务端与简洁前端）**（V3 本地微调待后续算力阶段推进）：

1. **数据处理与建库**（`T02`–`T03` / `T12`，已就绪）：
   ```bash
   python scripts/prepare_data.py
   # 常用参数: --kb-size 5000 --reset-index --skip-index
   ```
2. **启动 Web Demo（默认 FastAPI + 简洁前端，兼容 Gradio）**（`T06` / `T15`，已就绪）：
   ```bash
   python scripts/run_demo.py                 # 默认 FastAPI: http://127.0.0.1:8000 (API 文档 /docs)
   python scripts/run_demo.py --framework gradio --port 7860
   ```
3. **Embedding 三路对比实验**（`T11`，已就绪）：
   ```bash
   python scripts/compare_embeddings.py
   ```
4. **自动化评测**（`T07` / `T10` / `T12`，已就绪）：
   ```bash
   python scripts/eval_rag.py --version v2-retrieval --in-domain-limit 30 --ood-limit 20 --output results/v2_retrieval.json
   ```
5. **长上下文压测**（`T09`，已就绪）：
   ```bash
   python scripts/test_long_context.py --mode both
   ```
6. **LoRA 微调**（`T14`）：`python scripts/train_lora.py`

### 4.1 阶段实验指标对比（30 条领域内 + 20 条 OOD）

| 版本 | 字符 F1 (`accuracy_char_f1`) | 引用 F1 (`citation_f1`) | OOD 拒答率 | OOD 前置拦截率 | 幻觉率 | 平均延迟 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `v0-baseline` | `0.1902` | `0.9633` | `100.0%` | `—` | `0.0%` | `1569.38 ms` |
| `v1-core` | `0.2199` | `0.9367` | `100.0%` | `75.0%` | `0.0%` | `1342.16 ms` |
| `v2-retrieval` | `0.2228` | `0.9689` | `100.0%` | `95.0%` | `0.0%` | `1493.53 ms` |
