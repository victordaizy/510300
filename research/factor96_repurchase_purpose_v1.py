"""提取明确勾选的回购用途，保留多用途、未知及未归属的变更公告。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import unicodedata

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_factor96_repurchase_purpose_v1"
SOURCE = ROOT / "reports/research/510300_corporate_repurchase_disclosed_demand_v1"
SUPPLEMENT = ROOT / "reports/research/510300_corporate_repurchase_original_plan_completion_v1"
ORIGINAL = ROOT / "reports/research/510300_corporate_repurchase_index_documents_v1"
STUDY = "510300_FACTOR96_REPURCHASE_PURPOSE_V1"
LABELS = {"CANCEL_CAPITAL": "减少注册资本", "EMPLOYEE_INCENTIVE": "用于员工持股计划或股权激励",
          "CONVERTIBLE_BOND": "用于转换公司可转债", "VALUE_SUPPORT": "为维护公司价值及股东权益"}
MARKS = {"√": True, "✓": True, "✔": True, "☑": True, "□": False, "☐": False}


def now():
    return datetime.now().astimezone().isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, data, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, default=str)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def normalize(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def purpose_fields(pages):
    blocks = []
    for number, page in enumerate(pages, 1):
        text = normalize(page)
        for match in re.finditer("回购用途", text):
            start = match.end()
            stop = re.search("累计已回购|累计回购股|实际回购价格|回购方案|预计回购|一、|二、", text[start:start + 450])
            end = start + stop.start() if stop else min(len(text), start + 450)
            block = text[start:end]
            marks, unknown = {}, []
            for key, label in LABELS.items():
                positions = [m.start() for m in re.finditer(re.escape(label), block)]
                if len(positions) != 1 or positions[0] == 0:
                    unknown.append(key)
                    continue
                mark = block[positions[0] - 1]
                if mark not in MARKS:
                    unknown.append(key)
                    continue
                marks[key] = MARKS[mark]
            selected = sorted(key for key, checked in marks.items() if checked)
            complete = len(marks) == len(LABELS) and not unknown and bool(selected)
            blocks.append({"page": number, "normalized_start": start, "normalized_end": end,
                "evidence": block, "marks": marks, "unresolved_labels": unknown, "selected_labels": selected,
                "complete_four_label_menu": complete,
                "multiple_choice_allocation_unknown": len(selected) > 1,
                "alternative_or_combined_wording": bool(re.search("二者之一|二者皆有|之一或|一种或多种", block))})
    admitted = [b for b in blocks if b["complete_four_label_menu"]]
    selections = {tuple(b["selected_labels"]) for b in admitted}
    if not blocks:
        status, chosen = "NO_STANDARD_PURPOSE_HEADER", None
    elif not admitted:
        status, chosen = "PARTIAL_OR_UNKNOWN_MENU_RETAINED", None
    elif len(selections) > 1:
        status, chosen = "CONFLICTING_COMPLETE_MENUS", None
    elif any(b["marks"] and not b["complete_four_label_menu"] for b in blocks):
        status, chosen = "COMPLETE_AND_PARTIAL_MENUS_REQUIRE_REVIEW", None
    else:
        chosen = list(next(iter(selections)))
        status = "EXPLICIT_MULTIPLE_PURPOSES_NO_ALLOCATION" if len(chosen) > 1 else "EXPLICIT_SINGLE_PURPOSE"
    return {"status": status, "selected_purposes": chosen, "blocks": blocks,
            "trading_feature_admitted": False, "fallback_cancellation_used_as_current_purpose": False}


def asof_purpose(rows, root_id, timestamp):
    eligible = [r for r in rows if r.get("root_id") == root_id and r.get("known_at")
                and pd.Timestamp(r["known_at"]) <= pd.Timestamp(timestamp)]
    if not eligible:
        return {"status": "NO_VIEW", "document_id": None, "selected_purposes": None}
    last_clock = max(pd.Timestamp(r["known_at"]) for r in eligible)
    latest = [r for r in eligible if pd.Timestamp(r["known_at"]) == last_clock]
    if len(latest) != 1:
        return {"status": "SAME_CLOCK_MULTIPLE_DOCUMENTS_REQUIRE_REVIEW", "document_id": None, "selected_purposes": None}
    row = latest[0]
    return {"status": row["purpose"]["status"], "document_id": row["document_id"],
            "selected_purposes": row["purpose"]["selected_purposes"], "known_at": row["known_at"]}


def freeze():
    assert not (OUT / "freeze.json").exists()
    events = read(SOURCE / "results/confirmed_disclosure_events.json")
    roots = {r["root_id"] for r in events}
    nodes = [r for r in read(SOURCE / "results/plan_nodes.json") if r["root_id"] in roots]
    original_docs = read(ORIGINAL / "documents.json")
    documents = {r["document_id"]: r for r in original_docs}
    for name in ["supplement_documents.json", "gateway_supplement_documents.json"]:
        for row in read(SUPPLEMENT / "results" / name):
            if row["status"] == "PDF_TEXT_SAVED":
                documents[row["document_id"]] = row
    changes = [r for r in original_docs if any(w in r["title"] for w in ["用途", "终止", "调整回购股份方案", "回购金额", "下调回购"])]
    selected_ids = sorted({r["document_id"] for r in [*events, *nodes, *changes]})
    targets = []
    for key in selected_ids:
        original = documents[key]
        row = dict(original)
        for kind, old_field, suffix in [("raw", "raw_path", ".pdf"), ("text", "text_path", ".json"), ("receipts", "receipt_path", ".json")]:
            old = ROOT / original[old_field]
            if kind == "raw":
                assert digest(old) == original["raw_sha256"]
            target = OUT / "inputs" / kind / (key + suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, target)
            row[kind + "_snapshot"] = target.relative_to(OUT).as_posix()
            row[kind + "_sha256"] = digest(target)
        targets.append(row)
    save(OUT / "inputs/documents.json", targets, True)
    save(OUT / "inputs/execution_events.json", events, True)
    save(OUT / "inputs/plan_nodes.json", nodes, True)
    save(OUT / "inputs/change_notice_ids.json", [r["document_id"] for r in changes], True)
    source_map = {"inputs/prior_document_metadata.json": SUPPLEMENT / "results/document_metadata.json",
                  "source_evidence/previous_demand_result.json": SOURCE / "result.json",
                  "source_evidence/previous_demand_protocol.json": SOURCE / "protocol.json",
                  "source_evidence/original_document_protocol.json": ORIGINAL / "protocol.json",
                  "source_evidence/current_mandate.json": ROOT / "config/510300_existing_data_training_mandate_v1.json"}
    for name, source in source_map.items():
        target = OUT / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
    assert read(OUT / "prefreeze_test_receipt.json")["exit_code"] == 0
    for name in ["research/factor96_repurchase_purpose_v1.py", "tests/test_factor96_repurchase_purpose_v1.py"]:
        target = OUT / "code" / Path(name).name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT / name, target)
    protocol = {"at": now(), "study_id": STUDY, "phase": "SOURCE_PURPOSE_FIELDS_ONLY",
        "selection": "全部948条已有直接方案身份支持的执行记录、它们所属176方案的全部已存节点，以及已有原文中全部54个用途/终止/金额标题候选；不按收益筛选。",
        "documents": len(targets), "execution_records": len(events), "scheme_roots": len(roots), "plan_nodes": len(nodes), "change_notices": len(changes),
        "schema_exploration_before_freeze": "只对原2407份文本进行用途标题样式盘点：1040份包含回购用途，216种片段；未读策略收益或按绩效选模板。",
        "purpose_rule": "仅认定同一页回购用途表内四个标准标签的明确勾选/未勾选；菜单不全、私用字形或多处冲突保留未知。正文条件性未来注销不当当前注销目的。",
        "multiple_purposes": "多项勾选保存全部；不强制归单类、不虚构资金比例，二者之一或二者皆有另存。",
        "budget_progress": "执行公告自身已解析且无歧义的计划下限为分母，保存累计实施金额/该下限；不能把下限或上限当已回购，分母未知不计算。",
        "clock": "继承同文档原元数据与已保存执行/节点的最晚known_at；只在该时钟后可查询。查询显示最新文档即使其字段未知，不跨未知文档沿用旧值。",
        "boundary": "这是部分M02来源字段，不是完整用途/终止/减额版本链。54个标题候选不自动归属某一原方案；尚缺自由流通市值及完整时间/用途版本，T12仍NOT_RUN。",
        "source_coverage": "原目录未下载的限制性注销、持股清单、债权人和价格调整文件仍属旧范围之外，不宣称已覆盖全部用途变更。",
        "new_network_requests": 0, "new_accounts": 0, "new_models": 0, "goal_status": "active", "goal_achieved": False,
        "orders_authorized": False, "external_review": "NOT_PERFORMED"}
    save(OUT / "protocol.json", protocol, True)
    save(OUT / "freeze.json", {"at": now(), "before_fixed_purpose_extraction_and_new_outcomes": True,
        "files": [{"path": p.relative_to(OUT).as_posix(), "sha256": digest(p)} for p in sorted(OUT.rglob("*")) if p.is_file()]}, True)
    print(f"回购用途范围已冻结：{len(targets)}份原文、{len(events)}执行记录、{len(changes)}变更标题候选。", flush=True)


def run():
    assert not (OUT / "run_started.json").exists()
    for item in read(OUT / "freeze.json")["files"]:
        assert digest(OUT / item["path"]) == item["sha256"], item["path"]
    assert digest(Path(__file__)) == digest(OUT / "code" / Path(__file__).name)
    save(OUT / "run_started.json", {"at": now(), "freeze_sha256": digest(OUT / "freeze.json")}, True)
    events = {r["document_id"]: r for r in read(OUT / "inputs/execution_events.json")}
    nodes = {r["document_id"]: r for r in read(OUT / "inputs/plan_nodes.json")}
    metas = {r["document_id"]: r for r in read(OUT / "inputs/prior_document_metadata.json")}
    changes = set(read(OUT / "inputs/change_notice_ids.json"))
    fields = []
    for source in read(OUT / "inputs/documents.json"):
        key = source["document_id"]
        origins = [r for r in [metas.get(key), nodes.get(key), events.get(key)] if r is not None]
        clocks = [pd.Timestamp(r["known_at"]) for r in origins if r.get("known_at")]
        clock = max(clocks).isoformat() if clocks else None
        links = {r["root_id"] for r in [nodes.get(key), events.get(key)] if r and r.get("root_id")}
        event = events.get(key)
        budget = event.get("budget_terms", {}) if event else {}
        floor = budget.get("floor_cents") if budget.get("unambiguous_pair") else None
        progress = event["cumulative_cents"] / floor if event and floor and floor > 0 else None
        row = {"document_id": key, "symbol": source["symbol"], "title": source["title"], "source_url": source["source_url"],
            "known_at": clock, "known_clock_candidates": sorted({c.isoformat() for c in clocks}),
            "root_id": next(iter(links)) if len(links) == 1 else None,
            "is_prior_direct_execution": event is not None, "is_selected_plan_node": key in nodes,
            "is_change_title_candidate": key in changes, "purpose": purpose_fields(read(OUT / source["text_snapshot"])),
            "reported_cumulative_cents": event["cumulative_cents"] if event else None,
            "reported_plan_floor_cents": floor, "reported_execution_to_floor_ratio": progress,
            "prior_change_status": event.get("change_status") if event else None,
            "prior_usable_positive_disclosure": event.get("usable_positive_disclosure", False) if event else False,
            "ratio_status": "SAME_DOCUMENT_REPORTED_CUMULATIVE_OVER_PLAN_FLOOR" if progress is not None else "NOT_COMPUTED_NO_SAME_DOCUMENT_UNAMBIGUOUS_FLOOR",
            "historical_first_publication_verified": False, "trading_feature_admitted": False}
        fields.append(row)
    save(OUT / "document_purpose_fields.json", fields, True)
    execution_fields = [r for r in fields if r["is_prior_direct_execution"]]
    save(OUT / "execution_purpose_fields.json", execution_fields, True)
    save(OUT / "change_notice_fields.json", [r for r in fields if r["is_change_title_candidate"]], True)
    flat = [{k: v for k, v in r.items() if k != "purpose"} | {"purpose_status": r["purpose"]["status"],
                "selected_purposes": "|".join(r["purpose"]["selected_purposes"] or [])} for r in fields]
    pd.DataFrame(flat).to_csv(OUT / "逐公告用途与实施比例.csv", index=False, encoding="utf-8-sig")
    result = {"at": now(), "study_id": STUDY, "status": "EXPLICIT_CHECKBOX_PURPOSE_FIELDS_COMPLETE_FULL_M02_CHAIN_PENDING",
        "documents": len(fields), "execution_records": len(execution_fields),
        "purpose_statuses": dict(Counter(r["purpose"]["status"] for r in fields)),
        "execution_purpose_statuses": dict(Counter(r["purpose"]["status"] for r in execution_fields)),
        "execution_ratios_available": sum(r["reported_execution_to_floor_ratio"] is not None for r in execution_fields),
        "change_title_candidates": len(changes), "change_titles_without_existing_root_link": sum(r["is_change_title_candidate"] and r["root_id"] is None for r in fields),
        "full_purpose_termination_chain_established": False, "free_float_denominator_established": False,
        "T12": "NOT_RUN", "new_accounts": 0, "new_returns": 0, "new_network_requests": 0,
        "goal_status": "active", "goal_achieved": False, "external_review": "NOT_PERFORMED", "orders_authorized": False}
    save(OUT / "result.json", result, True)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "run"])
    {"freeze": freeze, "run": run}[parser.parse_args().action]()
