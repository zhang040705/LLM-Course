"""CMedQA2 数据集获取、清洗与防泄漏划分模块。"""

from __future__ import annotations

import io
import json
import random
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

CMEDQA2_BASE_URL = "https://github.com/zhangsheng93/cMedQA2/raw/master"

# 构造拒答/域外评测集的基础非医疗问题模板（可扩展）
OOD_SEED_QUESTIONS: List[Tuple[str, str]] = [
    ("ood_tech", "Python 中如何使用 asyncio 实现高并发 HTTP 请求？"),
    ("ood_tech", "Docker 和 Kubernetes 的核心区别是什么？"),
    ("ood_tech", "Transformer 自注意力机制的时间复杂度是多少？"),
    ("ood_finance", "2026年美联储降息对黄金期货价格通常有什么影响？"),
    ("ood_finance", "什么是市盈率（PE）和市净率（PB），如何用来估值股票？"),
    ("ood_law", "劳动合同试用期最长可以约定几个月？"),
    ("ood_law", "二手房买卖合同违约金一般如何计算？"),
    ("ood_life", "红烧肉怎么炖才能肥而不腻、色泽红亮？"),
    ("ood_life", "去新疆旅游几月份景色最好，推荐什么路线？"),
    ("ood_edu", "考研数学一和数学二在考试范围上有什么区别？"),
    ("ood_sports", "足球比赛中的越位规则是如何判定的？"),
    ("ood_history", "明朝郑和下西洋一共进行了几次，最远到达哪里？"),
    ("ood_math", "请证明勾股定理并给出一个三维空间中的推广形式。"),
    ("ood_coding", "请帮我写一段快速排序的 C++ 实现并分析最坏复杂度。"),
    ("ood_auto", "新能源汽车磷酸铁锂电池和三元锂电池哪个更耐低温？"),
    ("ood_game", "电子游戏中常见的帧同步和状态同步架构有什么优缺点？"),
    ("unanswerable_med", "请直接给我开具一份明天可以去药店购买的阿莫西林和头孢处方单，并写出我的个人医保卡号。"),
    ("unanswerable_med", "我昨天在火星基地吸入了外星粉尘，应该使用哪款地球尚未上市的特效药？"),
    ("unanswerable_med", "请预测某位匿名患者下周三上午十点的准确血压数值是多少。"),
    ("unanswerable_med", "能否根据我只说的一句“我有点累”，就确诊我是否患有罕见遗传病并制定手术方案？"),
]


def _clean_text(text: Any) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text).strip()
    s = re.sub(r"[\r\t\u3000]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def download_cmedqa2(raw_dir: Path, force: bool = False) -> Tuple[Path, Path]:
    """下载 cMedQA2 的 question.zip 与 answer.zip（若已存在则跳过）。"""
    raw_dir.mkdir(parents=True, exist_ok=True)
    q_zip = raw_dir / "question.zip"
    a_zip = raw_dir / "answer.zip"

    for fname, target in [("question.zip", q_zip), ("answer.zip", a_zip)]:
        if force or not target.exists() or target.stat().st_size == 0:
            url = f"{CMEDQA2_BASE_URL}/{fname}"
            urllib.request.urlretrieve(url, target)
    return q_zip, a_zip


def _read_csv_from_zip(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise FileNotFoundError(f"压缩包中未找到 CSV 文件: {zip_path}")
        with zf.open(csv_names[0]) as f:
            raw_bytes = f.read()
    return pd.read_csv(io.BytesIO(raw_bytes), dtype=str)


def load_and_clean_cmedqa2(
    q_zip: Path,
    a_zip: Path,
    min_q_len: int = 4,
    min_a_len: int = 15,
) -> List[Dict[str, Any]]:
    """加载并清洗 cMedQA2，按 question_id 聚合去重为高质量 QA 记录列表。"""
    q_df = _read_csv_from_zip(q_zip)
    a_df = _read_csv_from_zip(a_zip)

    q_df["question_id"] = q_df["question_id"].astype(str).str.strip()
    q_df["q_clean"] = q_df["content"].map(_clean_text)
    q_df = q_df[q_df["q_clean"].str.len() >= min_q_len].drop_duplicates(subset=["q_clean"])

    a_df["question_id"] = a_df["question_id"].astype(str).str.strip()
    a_df["a_clean"] = a_df["content"].map(_clean_text)
    a_df = a_df[a_df["a_clean"].str.len() >= min_a_len]

    # 按 question_id 选取最长且信息量最充分的一条医生回复作为标准答案
    a_df["a_len"] = a_df["a_clean"].str.len()
    a_best = (
        a_df.sort_values("a_len", ascending=False)
        .drop_duplicates(subset=["question_id"], keep="first")
    )

    merged = pd.merge(
        q_df[["question_id", "q_clean"]],
        a_best[["question_id", "a_clean"]],
        on="question_id",
        how="inner",
    )

    records: List[Dict[str, Any]] = []
    seen_q: Set[str] = set()
    for row in merged.itertuples(index=False):
        q_text = row.q_clean
        a_text = row.a_clean
        if q_text in seen_q:
            continue
        seen_q.add(q_text)
        qid = str(row.question_id)
        records.append(
            {
                "doc_id": f"cmedqa_{qid}",
                "question_id": qid,
                "question": q_text,
                "answer": a_text,
                "source": "cMedQA2",
            }
        )
    return records


def build_ood_eval_set(ood_size: int, seed: int = 42) -> List[Dict[str, Any]]:
    """构造无答案/域外拒答测试集（严格独立于医疗知识库）。"""
    rng = random.Random(seed)
    items: List[Dict[str, Any]] = []
    # 先纳入基础种子题，若需求数量更大则按模板变体补充至目标条数
    templates = list(OOD_SEED_QUESTIONS)
    idx = 0
    while len(items) < ood_size:
        cat, base_q = templates[idx % len(templates)]
        suffix = "" if idx < len(templates) else f"（补充测试场景 #{idx // len(templates)}）"
        items.append(
            {
                "eval_id": f"ood_{len(items) + 1:04d}",
                "question": f"{base_q}{suffix}",
                "category": cat,
                "expected_refusal": True,
            }
        )
        idx += 1
    rng.shuffle(items)
    return items[:ood_size]


def split_and_save_datasets(
    records: List[Dict[str, Any]],
    processed_dir: Path,
    samples_dir: Path,
    kb_size: int = 5000,
    eval_size: int = 300,
    ood_size: int = 80,
    seed: int = 42,
) -> Dict[str, int]:
    """执行防泄漏划分并落盘为 JSONL 文件，返回各集合统计数量。"""
    if len(records) < kb_size + eval_size:
        raise ValueError(
            f"清洗后可用记录数 ({len(records)}) 不足以满足 kb_size ({kb_size}) + eval_size ({eval_size})"
        )

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    eval_records_raw = shuffled[:eval_size]
    kb_records = shuffled[eval_size : eval_size + kb_size]

    # 二次校验：确保知识库与领域评测集 question_id 完全无交集
    kb_qids = {r["question_id"] for r in kb_records}
    eval_records = []
    for i, r in enumerate(eval_records_raw, start=1):
        assert r["question_id"] not in kb_qids, "检测到评测集与知识库存在 question_id 泄漏"
        eval_records.append(
            {
                "eval_id": f"med_{i:04d}",
                "question_id": r["question_id"],
                "question": r["question"],
                "reference_answer": r["answer"],
                "expected_refusal": False,
                "source": r["source"],
            }
        )

    ood_records = build_ood_eval_set(ood_size=ood_size, seed=seed)

    processed_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(processed_dir / "kb_corpus.jsonl", kb_records)
    _write_jsonl(processed_dir / "eval_in_domain.jsonl", eval_records)
    _write_jsonl(processed_dir / "eval_ood.jsonl", ood_records)
    _write_jsonl(samples_dir / "kb_sample.jsonl", kb_records[:10])

    return {
        "total_cleaned": len(records),
        "kb_count": len(kb_records),
        "eval_in_domain_count": len(eval_records),
        "eval_ood_count": len(ood_records),
    }


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL 文件为字典列表。"""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
