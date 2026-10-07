"""复核剩余23份配股完整预案的当次发行条款，不读取账户收益。"""
from collections import Counter
import argparse
import re

from research.factor96_rights_event_chains_v1 import ROOT, read, save, digest, normalized, now

OUT = ROOT / "reports/research/510300_factor96_rights_preplans_v1"
PREVIOUS = ROOT / "reports/research/510300_factor96_rights_plan_admin_v1"
SOURCE = ROOT / "reports/research/510300_factor96_rights_issue_documents_v1"


def freeze():
    rows = read(PREVIOUS / "remaining_63_long_documents.json")["documents"]
    targets = [r for r in rows if r["body_role_candidate"] == "PROPOSED_PLAN_VERSION"]
    remaining = [r for r in rows if r["body_role_candidate"] != "PROPOSED_PLAN_VERSION"]
    assert len(targets) == 23 and len(remaining) == 40
    save(OUT / "protocol.json", {
        "at": now(), "study_id": "510300_FACTOR96_RIGHTS_PREPLANS_V1",
        "scope": "既有63份待审长文件中全部23份完整预案，复核当次配股比例、数量及股本基准、A/H范围、募资上限、批准条件、期限和持有约束。",
        "selection_rule": "按既有正文分类全取23份，不按最终发行成败、价格或收益筛选；其余40份保留待审。",
        "source_rule": "只读已有PDF及文本；源文件哈希保留。相关条款阅读与全文阅读分开，不验证经营预测及投资可行性。",
        "clock_rule": "目录日末作可得日期代理；正文内更早会议、报告期和签署日期不提前可得时点；未证明历史最早公开。",
        "quantity_rule": "有条件股本测算、发行上限、实际认购、A股、H股分别保存；不设统一取整规则；未知数不从后来的实施回填。",
        "holding_rule": "认购承诺不是不减持承诺；一般无持有期与法定例外分开。未出现条款不推断无约束，未来限售期限未定不推算解锁日。",
        "version_rule": "同日修订说明和预案并列，不能以公告编号排序覆盖；已终止方案保持终止，原来源字段不覆盖，不新建实际发行日历。",
        "new_network_requests": 0, "new_accounts": 0, "new_returns": 0, "new_models": 0,
        "orders_authorized": False, "delivery_package_required": False})
    save(OUT / "targets.json", targets)
    save(OUT / "remaining_40_long_documents.json", {"at": now(), "documents": remaining,
        "count": len(remaining), "role_counts": dict(Counter(r["body_role_candidate"] for r in remaining)),
        "status": "BODY_REVIEW_PENDING"})
    paths = [PREVIOUS / "remaining_63_long_documents.json", PREVIOUS / "combined_source_facts.json",
             OUT / "protocol.json", OUT / "targets.json", OUT / "remaining_40_long_documents.json"]
    save(OUT / "freeze.json", {"at": now(), "files": [{"path": str(p), "sha256": digest(p)} for p in paths]})
    print("已固定23份完整预案，剩余40份长文件。")


def extract():
    for f in read(OUT / "freeze.json")["files"]:
        assert digest(f["path"]) == f["sha256"]
    pattern = r"每10股|每十股|募集资金总额|募集资金规模|持有期|不减持|限售期|锁定期|有效期|尚需|尚须|尚待|仍需|仍须"
    outputs = []
    for row in read(OUT / "targets.json"):
        source = row["source"]
        assert digest(SOURCE / source["raw_snapshot"]) == source["raw_sha256"]
        assert digest(row["text_path"]) == source["text_sha256"]
        pages = read(row["text_path"])["pages"]
        candidates = []
        for p in pages:
            t = normalized(p["text"])
            spans = []
            for m in re.finditer(pattern, t):
                a, b = max(0, m.start()-90), min(len(t), m.end()+300)
                if spans and a <= spans[-1][1]:
                    spans[-1][1] = max(b, spans[-1][1])
                else:
                    spans.append([a,b])
            for a,b in spans:
                candidates.append({"page": p["page"], "start": a, "end": b,
                                   "literal": t[a:b], "reviewed": False})
        outputs.append({"document_id": row["document_id"], "symbol": row["symbol"],
            "title": row["title"], "known_at": row["known_at"], "pages": len(pages),
            "candidates": candidates, "candidate_chars": sum(len(c["literal"]) for c in candidates)})
    save(OUT / "source_candidates.json", outputs)
    print("固定预案", len(outputs), "总页数", sum(r["pages"] for r in outputs),
          "候选片段", sum(len(r["candidates"]) for r in outputs),
          "候选字符", sum(r["candidate_chars"] for r in outputs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze", "extract"])
    args = parser.parse_args()
    freeze() if args.stage == "freeze" else extract()
