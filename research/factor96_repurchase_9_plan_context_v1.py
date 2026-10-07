"""复核21个用途未知批次的9个原方案，区分同文档未知与当时可见的方案证据。"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

from research.factor96_repurchase_purpose_v1 import ROOT, now, digest, read, save, normalize

OUT = ROOT / "reports/research/510300_factor96_repurchase_9_plan_context_v1"
BASE = ROOT / "reports/research/510300_factor96_repurchase_purpose_v1"
PRIOR = ROOT / "reports/research/510300_factor96_repurchase_body_purpose_v1"
CHANGES = ROOT / "reports/research/510300_factor96_repurchase_32_originals_v1/combined_change_ledger.json"
STUDY = "510300_FACTOR96_REPURCHASE_9_PLAN_CONTEXT_V1"


def prepare():
    assert not (OUT / "protocol.json").exists()
    batches = read(PRIOR / "remaining_unknown_positive_batches.json")
    roots = {b["root_id"] for b in batches}
    nodes = [n for n in read(BASE / "inputs/plan_nodes.json") if n["root_id"] in roots]
    changes = [c for c in read(CHANGES) if any(r in roots for r in c["confirmed_target_roots"])]
    ids = {n["document_id"] for n in nodes} | {c["document_id"] for c in changes}
    docs = [d for d in read(BASE / "inputs/documents.json") if d["document_id"] in ids]
    assert len(batches) == 21 and len(roots) == 9 and len(nodes) == 15 and len(docs) == 16
    inputs = [PRIOR / "remaining_unknown_positive_batches.json", PRIOR / "publication_batches.json",
              BASE / "inputs/plan_nodes.json", BASE / "inputs/documents.json", CHANGES]
    save(OUT / "target_batches.json", batches, True)
    save(OUT / "plan_nodes.json", nodes, True)
    save(OUT / "existing_linked_changes.json", changes, True)
    save(OUT / "local_documents.json", docs, True)
    save(OUT / "protocol.json", {
        "at": now(), "study_id": STUDY, "previous_goal_turn": "PROGRESS_194_POSITIVE_BATCH_PURPOSES_REVIEWED",
        "scope": "上一轮剩余21个正向批次、9个方案，全部纳入；本地15个方案节点及1个已关联变更，先核对原用途与审批条款。",
        "field_separation": "保留执行公告同文档用途未知；新增prior_plan_context字段，不改写旧purpose_consensus，不称完整生效链已经建成。",
        "time": "每份原方案和变更使用各自known_at，按原目录日期末代理；后出的变更、批准、报告书不能被早期查询看见。",
        "approval": "原方案的董事会已通过、股东大会待批、股东大会已批分别记录；不把初始意向当已生效。",
        "identity": "先按既有直接方案身份关联，再核对审批日、方案编号、公告编号和正文用途；同公司不同方案不串联。",
        "same_document_absence": "执行公告未重述用途属于文档字段缺失；历史方案证据可以单列，但不能据此声称期间不存在用途变更。",
        "bounded_missing_source_search": [
            {"symbol": "600010.SH", "start": "2024-08-26", "end": "2024-08-29", "target": "2024年8月26日董事会通过的回购用途变更公告"},
            {"symbol": "600010.SH", "start": "2024-12-09", "end": "2024-12-11", "target": "2024年第四次临时股东大会决议及回购用途变更批准"},
        ],
        "missing_source_policy": "仅补这两个已有明确日期的包钢原文缺件；优先复用本地目录，若无则官方巨潮或交易所来源定向查询，传输失败最多重试一次，403/429停止。",
        "execution": "本轮不改现金差分，不运行账户，不读取新策略收益。自由流通分母缺失，T12保持未运行。",
        "new_accounts": 0, "new_returns": 0, "goal_achieved": False, "delivery_package_required": False,
        "inputs": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in inputs],
    }, True)
    print("已固定21批次/9方案及16份本地原文，另有2个明确日期的包钢变更与批准缺件窗口。")


def excerpts():
    assert not (OUT / "source_clauses.json").exists()
    records = []
    for doc in read(OUT / "local_documents.json"):
        assert digest(BASE / doc["text_snapshot"]) == doc["text_sha256"]
        clauses = []
        pages = read(BASE / doc["text_snapshot"])
        for number, raw in enumerate(pages, 1):
            text = normalize(raw)
            spans = []
            for m in re.finditer("用于|用作|注销|股东.{0,12}(?:通过|审议)|尚需|无需|变更|其他内容不变", text):
                a, b = max(0, m.start()-55), min(len(text), m.end()+110)
                if spans and a <= spans[-1][1]:
                    spans[-1][1] = b
                else:
                    spans.append([a, b])
            clauses.extend({"page": number, "start": a, "end": b, "text": text[a:b]} for a, b in spans)
        records.append({"document_id": doc["document_id"], "symbol": doc["symbol"], "title": doc["title"], "clauses": clauses})
    save(OUT / "source_clauses.json", records, True)
    print(f"已提取{len(records)}份本地原文的用途与批准条款，尚未赋予生效状态。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="九方案原文和审批时钟复核")
    parser.add_argument("action", choices=["prepare", "excerpts"])
    {"prepare": prepare, "excerpts": excerpts}[parser.parse_args().action]()
