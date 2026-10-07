"""基于固定本地配股原文建立发行身份及日历版本，保留未知与更正时钟。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/research/510300_factor96_rights_issue_documents_v1"
OUT = ROOT / "reports/research/510300_factor96_rights_event_chains_v1"
MANIFEST = SOURCE / "transport_completion_v1/effective_documents.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def now():
    return datetime.now().astimezone().isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def title_role(title):
    if "超额配股" in title:
        return "OVERSUBSCRIPTION_OUTSIDE_A_RIGHTS"
    if "终止" in title or "批复到期失效" in title:
        return "TERMINATION_OR_EXPIRY"
    if "更正" in title:
        return "CORRECTION_NOTICE"
    if "发行结果" in title:
        return "SUBSCRIPTION_RESULT"
    if "上市公告" in title:
        return "LISTING_NOTICE"
    if "发行公告" in title:
        return "REVISED_ISSUANCE_NOTICE" if "修订" in title else "ISSUANCE_NOTICE"
    return None


def freeze():
    documents = read(MANIFEST)
    targets = [{**row, "review_role": title_role(row["title"])} for row in documents if title_role(row["title"])]
    assert len(documents) == 491
    assert all(row["status"] == "PDF_TEXT_SAVED" for row in targets)
    protocol = {
        "at": now(), "study_id": "510300_FACTOR96_RIGHTS_EVENT_CHAINS_V1",
        "scope": "固定491份本地配股原文中，标题含发行公告、发行结果、上市公告、更正、终止、批复到期失效或超额配股的全部文件。",
        "targets": len(targets), "title_roles": dict(Counter(row["review_role"] for row in targets)),
        "selection_uses_returns": False,
        "source_schema_seen_before_freeze": "491项目录字段及标题；旧兴业证券2015发行四文档示例已在先前阶段阅读。其余本轮目标正文尚未新读。",
        "event_identity": "公司证券代码加原文核准/注册文号及A股发行对象为优先锚点；登记日、价格和配股代码用于交叉检查；不得仅凭相邻日期合并两次发行。文号未识别时保留未连结状态。",
        "time_policy": "原文目录日期末23:59:59+08:00作为保守日期代理；保留原目录时钟和本次取得时间，不宣称历史首版已验证。后续结果与更正只能从各自目录代理时钟开始可见。",
        "field_policy": "计划数量、预算上限、发行价格、认购付款窗口、停牌清算期、实际认购、募集总额、募集净额和拟上市日分别记录。金额按分、股数按股；A/H分别，不能混合双市场总量。",
        "correction_policy": "更正引用的旧值保留为旧版本；同日重发及重复结果不计新增发行；未能确证覆盖范围的字段保持未知。",
        "termination_policy": "未实施计划的终止不生成缴款、上市或新供给事件；独董意见不作为新一次终止决策。",
        "remaining_source_documents": "其余说明书、提示公告、预案及反馈材料仍在原库，本轮不宣称其版本关系已全部核实。",
        "source_only": True, "trading_feature_admitted": False, "full_M06_calendar_established": False,
        "new_accounts": 0, "new_returns": 0, "new_models": 0, "new_network_requests": 0,
        "orders_authorized": False, "delivery_package_required": False,
    }
    save(OUT / "protocol.json", protocol)
    save(OUT / "targets.json", targets)
    save(OUT / "freeze.json", {"at": now(), "manifest_sha256": digest(MANIFEST),
         "protocol_sha256": digest(OUT / "protocol.json"), "targets_sha256": digest(OUT / "targets.json"),
         "source_files": [{"document_id": row["document_id"], "text_path": row["text_snapshot"],
                           "text_sha256": row["text_sha256"], "raw_path": row["raw_snapshot"],
                           "raw_sha256": row["raw_sha256"]} for row in targets]})
    print(json.dumps(protocol, ensure_ascii=False))


def extract():
    frozen = read(OUT / "freeze.json")
    assert digest(OUT / "targets.json") == frozen["targets_sha256"]
    patterns = {
        "IDENTITY": r"证监许可.{0,45}?号|证监会.{0,45}?号|配股代码.{0,35}|配股简称.{0,35}",
        "RECORD_DATE": r"股权登记日",
        "PAYMENT": r"缴款起止|缴款期|缴款时间|缴款日期|缴款期限|认购时间|认购期",
        "PRICE": r"配股价格|发行价格",
        "ACTUAL": r"有效认购|实际配售|实际认购|实收|募集资金总额|募集资金净额|认购股份|认购资金",
        "LISTING": r"上市时间|上市日期|上市流通日|新增股份上市|新增股票上市|上市日",
        "REVISION": r"现更正为|更正后|更正前|终止|失效",
    }
    rows = []
    for row in read(OUT / "targets.json"):
        assert digest(SOURCE / row["text_snapshot"]) == row["text_sha256"]
        pages = read(SOURCE / row["text_snapshot"])["pages"]
        hits = []
        for page in pages:
            text = normalized(page["text"])
            for kind, pattern in patterns.items():
                for match in re.finditer(pattern, text):
                    start, end = max(0, match.start() - 55), min(len(text), match.end() + 210)
                    hits.append({"kind": kind, "page": page["page"], "start": start, "end": end,
                                 "evidence": text[start:end]})
        rows.append({"document_id": row["document_id"], "symbol": row["symbol"], "title": row["title"],
                     "known_at": row["catalogue_date"] + "T23:59:59+08:00", "role": row["review_role"],
                     "page_count": len(pages), "candidates": hits, "event_field_admitted": False})
    save(OUT / "source_candidates.json", rows)
    print("固定范围原文定位完成：", len(rows), "份文档；", sum(len(row["candidates"]) for row in rows), "段候选。")


def show(start, stop, kind):
    rows = read(OUT / "source_candidates.json")
    if kind:
        rows = [row for row in rows if row["role"] == kind]
    for row in rows[start:stop]:
        print("\n", row["symbol"], row["document_id"], row["known_at"], row["role"], row["title"])
        seen = set()
        limits = Counter()
        for hit in row["candidates"]:
            if hit["evidence"] in seen or limits[hit["kind"]] >= 3:
                continue
            seen.add(hit["evidence"])
            limits[hit["kind"]] += 1
            print(hit["kind"], "页", hit["page"], hit["evidence"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze", "extract", "show"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=5)
    parser.add_argument("--kind")
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
    elif args.stage == "extract":
        extract()
    else:
        show(args.start, args.stop, args.kind)
