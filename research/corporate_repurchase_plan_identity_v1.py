"""用原始方案引用及披露时点解析执行记录身份，避免把预算修订当新方案。"""
from __future__ import annotations

from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.selected_mix_reappraisal_v1 import read, save, now, digest
import research.corporate_repurchase_execution_extractor_v1 as prior
import research.corporate_repurchase_index_documents_v1 as source

OUT = ROOT / "reports/research/510300_corporate_repurchase_plan_identity_v1"
STUDY = "510300_CORPORATE_REPURCHASE_PLAN_IDENTITY_V1"
ANN = re.compile(r"(?:公告)?编号[:：]?(?:临)?[〔\[（(]?(20\d{2})[〕\]）)]?[-－—]?([0-9]{1,4})(?![0-9])")


def annos(text):
    return sorted({f"{m.group(1)}-{int(m.group(2)):03d}" for m in ANN.finditer(text)})


def publication_date(text):
    match = re.search(r"(?:A股)?回购方案首次披露日(20\d{2})[/年.-](\d{1,2})[/月.-](\d{1,2})日?", text[:1600])
    return prior.calendar(match.groups()) if match else None


def approvals(text):
    """容许日期与会议之间有公司名称，原方案与金额/期限修订分开。"""
    output = []
    for match in re.finditer(prior.DATE, text[:4500]):
        before = text[max(0, match.start() - 180):match.start()]
        after = text[match.end():match.end() + 450]
        if re.search(prior.DATE + r"[,、及和]$", before):
            continue
        head = re.split(r"。|；", after)[0]
        meeting = re.search(r"董事会|股东大会|股东会", head[:160])
        if not meeting:
            continue
        lead = head[:meeting.start()]
        additional_dates = list(re.finditer(prior.DATE, lead))
        if additional_dates and not re.match(r"[,、及和](?:20\d{2}年)?\d{1,2}月\d{1,2}日", lead):
            continue
        proposal = re.search(r"审议(?:并)?通过(?:了)?《([^》]+)》", head)
        if proposal:
            name = proposal.group(1)
        else:
            raw = re.search(r"审议(?:并)?通过(?:了)?([^。；]{1,60})", head)
            previous = re.findall(r"《([^》]+)》", before)
            name = raw.group(1) if raw and "回购" in raw.group(1) else (previous[-1] if "审议通过" in head and previous else "")
        if "回购" not in name or any(word in name for word in ["限制性", "补偿", "贷款", "授信", "债券回购"]):
            continue
        action = "AMENDMENT" if any(word in name for word in ["增加", "调整", "变更", "延长", "延期", "终止"]) else "ORIGINAL"
        output.append({"date": prior.calendar(match.groups()), "body": "BOARD" if meeting.group(0) == "董事会" else "SHAREHOLDER",
            "action": action, "proposal": name, "evidence": text[match.start():match.end() + min(310, len(head))]})
    unique = {(r["date"], r["body"], r["action"]): r for r in output}
    return sorted(unique.values(), key=lambda r: (r["date"], r["body"], r["action"]))


def metadata(row):
    text = prior.page_text(read(ROOT / row["text_path"]))
    title = prior.normalized(row["title"])
    signed = list(re.finditer(prior.DATE, text[-260:]))
    day = pd.Timestamp(row["catalogue_date"])
    signed_day = prior.calendar(signed[-1].groups()) if signed else None
    known_day = max(day, signed_day) if signed_day is not None else day
    header = annos(text[:260])
    numbers = annos(text)
    meetings = approvals(text)
    first = publication_date(text)
    non_core = any(word in title for word in ["提议", "提案", "董事长", "子公司", "联营", "合营", "限制性", "补偿", "法律意见", "独立", "注销", "借款", "贷款", "转让"])
    if "H股" in title and "A股" not in title:
        non_core = True
    original_plan = (row["title_category"] == "PLAN_OR_OTHER_REPURCHASE_DOCUMENT" and not non_core
        and any(word in title for word in ["方案", "报告书"]) and ("集中竞价" in text or "二级市场" in text))
    amendment = original_plan and any(word in title for word in ["增加", "调整", "变更", "延长", "延期", "终止", "更正"])
    return {"symbol": row["symbol"], "document_id": row["document_id"], "title": row["title"], "source_url": row["source_url"],
        "catalogue_date": day, "signed_date": signed_day,
        "known_at": known_day.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=23, minutes=59, seconds=59),
        "document_announcements": header, "referenced_announcements": [value for value in numbers if value not in header],
        "explicit_first_publication": first, "approvals": meetings,
        "original_board_dates": sorted({m["date"] for m in meetings if m["action"] == "ORIGINAL" and m["body"] == "BOARD"}),
        "budget": prior.budget_terms(text),
        "phase": next(iter(re.findall(r"第([一二三四五六七八九十\d]+)期", title)), None),
        "plan_kind": "AMENDMENT" if amendment else "ORIGINAL_PLAN_DOCUMENT" if original_plan else "NOT_A_PLAN_NODE",
        "raw_path": row["raw_path"], "raw_sha256": row["raw_sha256"]}


def roots_by_reference(meta, nodes):
    referenced = set(meta["referenced_announcements"])
    return {node["root_id"] for node in nodes if node["symbol"] == meta["symbol"] and node["known_at"] <= meta["known_at"]
        and referenced.intersection(node["document_announcements"])}


def compatible_budget(left, right):
    return (left["unambiguous_pair"] and right["unambiguous_pair"]
        and left["floor_cents"] == right["floor_cents"] and left["cap_cents"] == right["cap_cents"])


def build_nodes(metas):
    nodes, excluded = [], []
    selected = [m for m in metas if m["plan_kind"] != "NOT_A_PLAN_NODE"]
    selected.sort(key=lambda m: (m["known_at"], "报告书" in m["title"] and "方案" not in m["title"], m["document_id"]))
    for item in selected:
        meta = dict(item)
        candidates = roots_by_reference(meta, nodes)
        method = "EXPLICIT_PRIOR_PLAN_ANNOUNCEMENT"
        if not candidates:
            by_board = [n for n in nodes if n["symbol"] == meta["symbol"] and n["known_at"] <= meta["known_at"]
                and set(n["original_board_dates"]).intersection(meta["original_board_dates"])
                and n["phase"] == meta["phase"] and compatible_budget(n["budget"], meta["budget"])]
            candidates = {n["root_id"] for n in by_board}
            method = "SAME_BOARD_PHASE_AND_BUDGET"
        if len(candidates) > 1 or (meta["plan_kind"] == "AMENDMENT" and not candidates):
            meta.update(node_status="UNRESOLVED_PLAN_REFERENCE", candidate_roots=sorted(candidates))
            excluded.append(meta)
            continue
        root = next(iter(candidates)) if candidates else meta["symbol"] + "_ORIGINAL_" + meta["document_id"]
        meta.update(root_id=root, node_status="LINKED_PLAN_DOCUMENT" if candidates else "ORIGINAL_PLAN_ROOT",
            root_method=method if candidates else "ISSUER_ORIGINAL_DOCUMENT_ID")
        nodes.append(meta)
    return nodes, excluded


def resolve(meta, nodes):
    available = [n for n in nodes if n["symbol"] == meta["symbol"] and n["known_at"] <= meta["known_at"]]
    first = meta["explicit_first_publication"]
    by_first = {n["root_id"] for n in available if first is not None and first in [n["catalogue_date"], n["signed_date"], n["explicit_first_publication"]]}
    refs = roots_by_reference(meta, available)
    board = {n["root_id"] for n in available if set(n["original_board_dates"]).intersection(meta["original_board_dates"])
        and n["phase"] == meta["phase"] and compatible_budget(n["budget"], meta["budget"])}
    if by_first:
        roots, method = by_first, "EXPLICIT_FIRST_PUBLICATION_DATE"
        if refs and not roots.intersection(refs):
            return {"status": "CONFLICTING_FIRST_DATE_AND_PLAN_REFERENCE", "candidate_roots": sorted(by_first | refs)}
        if refs:
            roots = roots.intersection(refs)
    elif refs:
        roots, method = refs, "EXPLICIT_ORIGINAL_ANNOUNCEMENT_REFERENCE"
    else:
        roots, method = board, "BOARD_PHASE_AND_BUDGET_ONLY"
    if len(roots) != 1:
        return {"status": "MULTIPLE_ORIGINAL_PLANS" if roots else "NO_ORIGINAL_PLAN_IN_AVAILABLE_SOURCES", "candidate_roots": sorted(roots)}
    root = next(iter(roots))
    evidence = [n for n in available if n["root_id"] == root]
    return {"status": "LINKED_ASOF_ORIGINAL_PLAN", "root_id": root, "method": method,
        "evidence_document_ids": [n["document_id"] for n in evidence],
        "latest_used_plan_known_at": max(n["known_at"] for n in evidence),
        "original_board_dates": sorted({d for n in evidence for d in n["original_board_dates"]}),
        "strong_direct_reference": method != "BOARD_PHASE_AND_BUDGET_ONLY"}


def checks():
    assert annos("公告编号：临2024－025；公告编号2024-25") == ["2024-025"]
    assert publication_date("回购方案首次披露日2024/9/21") == pd.Timestamp("2024-09-21")
    meetings = approvals("2023年10月19日,公司召开第八届董事会第十七次会议审议通过了《关于回购公司股份的议案》。2024年3月1日,公司第八届董事会审议通过了《关于增加股份回购金额的议案》。")
    assert [m["action"] for m in meetings] == ["ORIGINAL", "AMENDMENT"]
    meta = {"symbol": "A", "known_at": pd.Timestamp("2025-01-03", tz="Asia/Shanghai"), "explicit_first_publication": None,
        "referenced_announcements": ["2025-001"], "original_board_dates": [], "phase": None,
        "budget": {"unambiguous_pair": False}}
    future = {"symbol": "A", "root_id": "X", "document_id": "1", "known_at": pd.Timestamp("2025-01-04", tz="Asia/Shanghai"),
        "catalogue_date": pd.Timestamp("2025-01-04"), "signed_date": None, "explicit_first_publication": None,
        "document_announcements": ["2025-001"], "original_board_dates": [], "phase": None, "budget": {"unambiguous_pair": False}}
    assert resolve(meta, [future])["status"] == "NO_ORIGINAL_PLAN_IN_AVAILABLE_SOURCES"
    earlier = {**future, "known_at": pd.Timestamp("2025-01-02", tz="Asia/Shanghai")}
    assert resolve(meta, [earlier, future])["root_id"] == "X"
    assert resolve(meta, [earlier]) == resolve(meta, [earlier, future])
    return {"announcement_number_canonicalization": True, "original_vs_amendment": True,
        "source_reference_must_be_available": True, "future_document_does_not_relabel_prior_identity": True}


def run(dry=False):
    tests = checks()
    docs = read(source.OUT / "documents.json")
    metas = [metadata(row) for row in docs]
    indexed = {m["document_id"]: m for m in metas}
    nodes, excluded = build_nodes(metas)
    old = read(prior.OUT / "results/parsed_documents.json")
    records = [r for r in old if r["status"].startswith("EXTRACTED_")]
    links = []
    for row in records:
        meta = indexed[row["document_id"]]
        identity = resolve(meta, nodes)
        links.append({"symbol": row["symbol"], "document_id": row["document_id"], "title": row["title"],
            "known_at": meta["known_at"], "explicit_first_publication": meta["explicit_first_publication"],
            "referenced_announcements": meta["referenced_announcements"], "original_board_dates_in_execution": meta["original_board_dates"],
            "identity": identity, "cumulative_cents_unchanged": row["cumulative_cents"], "trading_feature_admitted": False})
    counts = pd.Series([r["identity"]["status"] for r in links]).value_counts().to_dict()
    methods = pd.Series([r["identity"].get("method", "UNRESOLVED") for r in links]).value_counts().to_dict()
    print({"方案文件": len(nodes), "原方案根": len({n['root_id'] for n in nodes}), "方案文件未解": len(excluded), "执行连接": counts, "连接方式": methods}, flush=True)
    if dry:
        return
    for folder in ["code", "results"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY,
        "question": "将执行披露连接到当时已经公开的原始方案，保留修订和相似方案的区别，不通过未来文档回写过去身份。",
        "source_hashes": {p.relative_to(ROOT).as_posix(): digest(p) for p in [source.OUT / "documents.json", prior.OUT / "results/parsed_documents.json"]},
        "code_sha256": digest(Path(__file__)), "implementation_checks": tests,
        "precedence": "明确首次披露日及原方案公告编号优先；仅会议日期/期次/预算一致的连接单独标记，不能视为直接引用。",
        "amounts": "复用已保存金额，不修改本轮输入的累计金额或收益资料；尚不生成交易信号。",
        "new_accounts": 0, "new_fits": 0, "independent_forward_observations": 0}, True)
    (OUT / "code" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    save(OUT / "results/document_metadata.json", metas, True)
    save(OUT / "results/plan_nodes.json", nodes, True)
    save(OUT / "results/unresolved_plan_nodes.json", excluded, True)
    save(OUT / "results/execution_identity_links.json", links, True)
    result = {"at": now(), "study_id": STUDY, "status": "ORIGINAL_PLAN_LINKAGE_COMPLETED_SOURCE_GAPS_RETAINED",
        "plan_nodes": len(nodes), "original_plan_roots": len({n['root_id'] for n in nodes}), "unresolved_plan_nodes": len(excluded),
        "execution_records": len(links), "identity_status_counts": counts, "identity_method_counts": methods,
        "new_accounts": 0, "new_fits": 0, "independent_forward_observations": 0, "goal_status": "active",
        "goal_achieved": False, "current_market_view": "NO_VIEW", "orders_authorized": False, "review_package_created": False}
    save(OUT / "result.json", result, True)


if __name__ == "__main__":
    run(dry=len(sys.argv) > 1 and sys.argv[1] == "inspect")
