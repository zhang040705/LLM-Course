# 契约：数据与评测集格式（`eval-schema`）

- **关联任务**：`T02`（数据清洗与划分）、`T07`（V0 自动化评测）、`T10`（V1 对比评测）

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

## 4. 评测结果汇总格式（`results/v*_*.json`）
- `version`: string（如 `"v0-baseline"`, `"v1-core"`）
- `timestamp_utc`: ISO-8601 时间戳
- `metrics`: object
  - `in_domain_count`, `ood_count`, `in_domain_answer_rate`, `accuracy_char_f1`
  - `citation_precision`, `citation_recall`, `citation_f1`
  - `ood_refusal_accuracy`, `ood_pre_refusal_rate`（前置阈值零成本拦截率）, `hallucination_rate`
  - `invalid_citations_removed_total`, `avg_latency_ms`, `avg_total_tokens`
- `samples`: list（抽样预测与拒答明细）
- `multiturn_rewrite`（可选，开启 `--with-multiturn` 时生成）：
  - `followup_cases`, `avg_raw_top1_score`, `avg_rewritten_top1_score`, `avg_score_gain`, `improved_count`, `topic_switch_ood_refused`, `cases`
