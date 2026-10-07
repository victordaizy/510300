"""核对固定说明书剩余关键词候选，保存实际阅读窗口和未解决的事项归属。"""
import argparse
import json
import re

from research.factor96_rights_event_chains_v1 import ROOT, read, save, digest, normalized, now
from research.factor96_rights_prospectus_constraints_v1 import PATTERN

PREVIOUS = ROOT / "reports/research/510300_factor96_rights_prospectus_constraints_v1"
OUT = ROOT / "reports/research/510300_factor96_rights_holding_remainder_v1"


def freeze():
    source = read(PREVIOUS / "remaining_holding_candidates.json")
    targets = read(PREVIOUS / "targets.json")
    lookup = {r["document_id"]: (i, r) for i, r in enumerate(targets)}
    assert source["count"] == len(source["candidates"]) == 248
    save(OUT / "protocol.json", {
        "at": now(), "study_id": "510300_FACTOR96_RIGHTS_HOLDING_REMAINDER_V1",
        "previous_goal_turn_classification": "PROGRESS_6_CONTEXT_SOURCES_AND_248_PENDING_CANDIDATES",
        "scope": "固定上一轮未读248段，逐一核对全部关键词的60字前文、100字后文；不足判断时补读原候选或跨页正文。",
        "pattern": PATTERN, "before_characters": 60, "after_characters": 100,
        "reading_rule": "保存真实输出阅读窗口；窗口阅读不声称整段或全文已读。仅当事项归属明确才记录归类，不明确则保留后续证据缺口。",
        "semantic_rule": "发行前股份分类、历史融资、第三方股票和经营金融业务的锁定不自动计为本次配股限制；现有限售仍可能影响获配股份，必须单独连接证据。",
        "time_rule": "仅在固定来源目录日末代理时点读取该来源；旧事实不倒填到早年公开时点。",
        "quantity_rule": "不推定未知数量为零，不将旧持股量机械乘当前配股比例，不从无限售登记数量自动减去承诺持股。",
        "new_accounts": 0, "new_models": 0, "new_network_requests": 0,
        "orders_authorized": False, "delivery_package_required": False,
    })
    rows = []
    for i, r in enumerate(source["candidates"]):
        spans = []
        for m in re.finditer(PATTERN, r["literal"]):
            a, b = max(0, m.start() - 60), min(len(r["literal"]), m.end() + 100)
            if spans and a <= spans[-1][1]:
                spans[-1][1] = max(b, spans[-1][1])
            else:
                spans.append([a, b])
        assert spans
        index, target = lookup[r["document_id"]]
        rows.append({**r, "pending_index": i, "document_index": index,
                     "known_at": target["known_at"], "windows": [
                         {"page": r["page"], "start": r["start"] + a,
                          "end": r["start"] + b, "literal": r["literal"][a:b]}
                         for a, b in spans], "full_candidate_read": False})
    save(OUT / "targets.json", rows)
    paths = [OUT / "protocol.json", OUT / "targets.json",
             PREVIOUS / "remaining_holding_candidates.json", PREVIOUS / "combined_source_facts.json",
             PREVIOUS / "targets.json", PREVIOUS / "reviewed_candidates.json"]
    save(OUT / "freeze.json", {"at": now(), "files": [
        {"path": str(p), "sha256": digest(p)} for p in paths]})
    print(json.dumps({"固定候选": len(rows), "文件": len({r['document_id'] for r in rows}),
                      "阅读窗口": sum(len(r['windows']) for r in rows),
                      "窗口字符": sum(len(w['literal']) for r in rows for w in r['windows'])}, ensure_ascii=False))


def show(start, end, full=False):
    rows = read(OUT / "targets.json")
    for r in rows[start:end]:
        print(f"[{r['pending_index']}] 文档{r['document_index']} {r['document_id']} 第{r['page']}页 候选{r['candidate_index']}")
        if full:
            print(r["literal"])
        else:
            for w in r["windows"]:
                print(w["literal"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze", "show"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=248)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
    else:
        show(args.start, args.end, args.full)
