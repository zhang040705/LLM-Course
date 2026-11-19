# 契约：数据与评测集格式（`eval-schema`）

- **关联任务**：`T02`（数据清洗与划分）、`T07`（自动化评测）

## 1. 知识库语料（`data/processed/kb_corpus.jsonl`）
每行一条 JSON：
- `doc_id`: string（如 `"cmedqa_65102009"`）
- `question_id`: string
- `question`: string
- `answer`: string
- `source`: string（`"cMedQA2"`）

## 2. 领域问答评测集（`data/processed/eval_in_domain.jsonl`）
- `eval_id`: string（`"med_0001"`）
- `question_id`: string（保证与知识库 `question_id` 严格互斥）
- `question`: string
- `reference_answer`: string
- `expected_refusal`: `false`
- `source`: string

## 3. 拒答 / 域外测试集（`data/processed/eval_ood.jsonl`）
- `eval_id`: string（`"ood_0001"`）
- `question`: string
- `category`: string（如 `"ood_tech"`, `"ood_finance"`, `"unanswerable_med"`）
- `expected_refusal`: `true`
