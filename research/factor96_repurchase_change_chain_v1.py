"""逐公告区分回购方案引用、实际变更对象与审批阶段，只整理来源字段。"""
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
OUT = ROOT / "reports/research/510300_factor96_repurchase_change_chain_v1"
PRIOR = ROOT / "reports/research/510300_factor96_repurchase_purpose_v1"
DEMAND = ROOT / "reports/research/510300_corporate_repurchase_disclosed_demand_v1"
ORIGINAL = ROOT / "reports/research/510300_corporate_repurchase_index_documents_v1"
SUPPLEMENT = ROOT / "reports/research/510300_corporate_repurchase_original_plan_completion_v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def now():
    return datetime.now().astimezone().isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def normalize(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def anchors(pages, phrase):
    """只在单页内定位，不把下一页的页码拼到数字前面。"""
    needle = normalize(phrase)
    found = []
    for number, page in enumerate(pages, 1):
        text = normalize(page)
        for match in re.finditer(re.escape(needle), text):
            # 数字短语不得从更长的数字中截取，页码串接造成的假数字也不能命中。
            if needle[0].isdigit() and match.start() and text[match.start() - 1].isdigit():
                continue
            if needle[-1].isdigit() and match.end() < len(text) and text[match.end()].isdigit():
                continue
            found.append({"page": number, "start": match.start(), "end": match.end(),
                          "normalized_quote": needle,
                          "context": text[max(0, match.start() - 70):match.end() + 70]})
    if not found:
        raise ValueError("原文中缺少复核短语：" + phrase)
    return found


def prior_candidates(change, metadata, nodes):
    refs = set(metadata["referenced_announcements"])
    boards = set(metadata["original_board_dates"])
    candidates = []
    for node in nodes:
        if node["symbol"] != change["symbol"] or pd.Timestamp(node["known_at"]) >= pd.Timestamp(change["known_at"]):
            continue
        reference_hits = refs.intersection(node["document_announcements"])
        board_hits = boards.intersection(node["original_board_dates"])
        if reference_hits or board_hits:
            candidates.append({"root_id": node["root_id"], "document_id": node["document_id"],
                               "announcement_hits": sorted(reference_hits), "board_date_hits": sorted(board_hits),
                               "role": "MENTION_ONLY_REQUIRES_TARGET_CLAUSE"})
    return candidates


def latest_reviewed_change(rows, root_id, timestamp):
    """仅回答本轮已复核变更；不声称覆盖该方案全部公告或完整当前状态。"""
    eligible = [row for row in rows if root_id in row["confirmed_target_roots"]
                and pd.Timestamp(row["known_at"]) <= pd.Timestamp(timestamp)]
    if not eligible:
        return {"status": "NO_REVIEWED_CHANGE_BEFORE_QUERY", "document_ids": [], "effective_new_purpose": None}
    latest = max(pd.Timestamp(row["known_at"]) for row in eligible)
    group = [row for row in eligible if pd.Timestamp(row["known_at"]) == latest]
    if len(group) > 1:
        return {"status": "SAME_CLOCK_MULTIPLE_DOCUMENTS_UNRESOLVED", "document_ids": sorted(r["document_id"] for r in group),
                "effective_new_purpose": None}
    row = group[0]
    approved = row["approval_state"] == "BOARD_APPROVED_NO_SHAREHOLDER_REQUIRED"
    return {"status": row["approval_state"], "document_ids": [row["document_id"]],
            "action": row["action"], "known_at": row["known_at"],
            "proposed_purpose": row.get("proposed_purpose"),
            "effective_new_purpose": row.get("proposed_purpose") if approved else None,
            "approved_terms": row.get("terms", {}) if approved else {},
            "scope": "LATEST_REVIEWED_CHANGE_ONLY_NOT_COMPLETE_PLAN_STATE"}


def prepare():
    assert not (OUT / "protocol.json").exists()
    changes = read(PRIOR / "change_notice_fields.json")
    metadata = read(PRIOR / "inputs/prior_document_metadata.json")
    metas = {row["document_id"]: row for row in metadata}
    nodes = read(DEMAND / "results/plan_nodes.json")
    candidates = {row["document_id"]: prior_candidates(row, metas[row["document_id"]], nodes) for row in changes}
    selected = {row["document_id"] for row in changes}
    for rows in candidates.values():
        for row in rows:
            selected.add(row["document_id"])
            selected.add(row["root_id"].split("_ORIGINAL_")[1])
    documents = {r["document_id"]: r for r in read(ORIGINAL / "documents.json")}
    for name in ["supplement_documents.json", "gateway_supplement_documents.json"]:
        for row in read(SUPPLEMENT / "results" / name):
            if row["status"] == "PDF_TEXT_SAVED":
                documents[row["document_id"]] = row
    snapshots = []
    for key in sorted(selected):
        source = documents[key]
        row = {"document_id": key, "symbol": source["symbol"], "title": source["title"],
               "source_url": source["source_url"], "catalogue_date": source["catalogue_date"]}
        for kind, field, suffix in [("raw", "raw_path", ".pdf"), ("text", "text_path", ".json"), ("receipts", "receipt_path", ".json")]:
            origin = ROOT / source[field]
            target = OUT / "inputs" / kind / (key + suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            assert not target.exists()
            if kind == "raw":
                assert digest(origin) == source["raw_sha256"]
            shutil.copyfile(origin, target)
            row[kind + "_path"] = target.relative_to(OUT).as_posix()
            row[kind + "_sha256"] = digest(target)
        snapshots.append(row)
    save(OUT / "inputs/documents.json", snapshots)
    save(OUT / "inputs/changes.json", changes)
    save(OUT / "inputs/metadata.json", metadata)
    save(OUT / "inputs/plan_nodes.json", nodes)
    save(OUT / "candidate_mentions.json", candidates)
    for name, source in {"prior_purpose_result.json": PRIOR / "result.json",
                         "prior_purpose_protocol.json": PRIOR / "protocol.json",
                         "mandate_before.json": ROOT / "config/510300_existing_data_training_mandate_v1.json",
                         "program_before.json": ROOT / "reports/research/510300_factor96_program_v1/status.json"}.items():
        save(OUT / "inputs" / name, read(source))
    save(OUT / "protocol.json", {"study_id": "510300_FACTOR96_REPURCHASE_CHANGE_CHAIN_V1", "at": now(),
         "selection": "完整保留既有54个标题候选。只对用公告编号或原批准日能与既有早期原方案产生候选关系的全部文件，逐段复核变更对象；其余保留缺原方案。另检查终止一词的实际对象。",
         "schema_exploration": "准备前只读原文及已存元数据，识别多方案提及、待批准、数字跨页和同编号双文档；未读市场价格序列、收益或按绩效挑样本。",
         "change_documents": len(changes), "selected_source_documents": len(snapshots),
         "candidate_document_count": sum(bool(v) for v in candidates.values()),
         "association_rule": "同公司、严格先于变更保守已知时钟的原方案，且原文变更对象条款明确对应。提到原方案或相同批准日仅是候选，不自动认定其用途改变。",
         "approval_rule": "董事长提议、董事会通过但待股东会、无需股东会的已通过事项分开；拟注销数量不记成已完成注销。",
         "multi_target_rule": "逐目标保存数量和原方案；一个目标缺原件不妨碍记录另一明确目标，但整篇/全链不标完成。",
         "clock_rule": "沿用来源保守known_at，不将董事会日期提前当公开日；未证明首次历史发布，所有记录trading_feature_admitted=false。",
         "same_clock_rule": "同一方案同一时钟多文档保留，未明修订次序不选一个当唯一最新值；不把数量加总两次。",
         "state_query_scope": "本轮54个候选中的已复核变更，非完整方案存续状态，尚未包含后续批准、登记和全部后续未知公告。",
         "quantity_rule": "同一页内找数值证据，避免串接页码；预算增加、已实施金额、剩余股份和新成交金额分开。",
         "free_float_denominator": "MISSING_UNCHANGED", "T12": "NOT_RUN", "new_network_requests": 0,
         "new_accounts": 0, "goal_achieved": False, "orders_authorized": False})
    print(json.dumps({"阶段": "来源准备完成", "变更候选": len(changes), "有既有方案候选关系": sum(bool(v) for v in candidates.values()),
                      "复制原文": len(snapshots)}, ensure_ascii=False), flush=True)


def freeze():
    assert not (OUT / "freeze.json").exists()
    assert read(OUT / "prefreeze_test_receipt.json")["exit_code"] == 0
    for name in ["research/factor96_repurchase_change_chain_v1.py", "tests/test_factor96_repurchase_change_chain_v1.py"]:
        target = OUT / "code" / Path(name).name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    files = [{"path": path.relative_to(OUT).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)}
             for path in sorted(OUT.rglob("*")) if path.is_file() and "__pycache__" not in path.parts]
    save(OUT / "freeze.json", {"at": now(), "files": files})
    print("原文、逐条复核卡及代码已冻结。", flush=True)


def run():
    frozen = read(OUT / "freeze.json")
    assert not (OUT / "result.json").exists()
    for row in frozen["files"]:
        assert digest(OUT / row["path"]) == row["sha256"]
    save(OUT / "run_started.json", {"at": now(), "freeze_sha256": digest(OUT / "freeze.json")})
    changes = read(OUT / "inputs/changes.json")
    nodes = read(OUT / "inputs/plan_nodes.json")
    by_id = {r["document_id"]: r for r in nodes}
    sources = {r["document_id"]: r for r in read(OUT / "inputs/documents.json")}
    cards = read(OUT / "review_cards.json")
    mentions = read(OUT / "candidate_mentions.json")
    assert {r["document_id"] for r in changes if mentions[r["document_id"]]}.issubset(cards)
    output = []
    for item in changes:
        key = item["document_id"]
        card = cards.get(key)
        row = {"document_id": key, "symbol": item["symbol"], "title": item["title"], "known_at": item["known_at"],
               "prior_root_id": item["root_id"], "candidate_mentions": mentions[key],
               "confirmed_target_roots": [], "source_url": sources[key]["source_url"],
               "raw_sha256": sources[key]["raw_sha256"], "historical_first_publication_verified": False,
               "trading_feature_admitted": False, "new_executed_cashflow": "NOT_COMPUTED",
               "full_plan_lifecycle_established": False}
        if card is None:
            row.update(review_status="NO_EXISTING_PRIOR_ROOT_CANDIDATE_FULL_TARGET_REVIEW_PENDING",
                       action="UNRESOLVED", approval_state="UNRESOLVED", evidence={})
        else:
            pages = read(OUT / sources[key]["text_path"])
            row.update(card)
            row["evidence"] = {label: anchors(pages, phrase) for label, phrase in card["proofs"].items()}
            del row["proofs"]
            targets = []
            for target in card.get("targets", []):
                target = dict(target)
                original_id = target.get("original_id")
                if original_id:
                    node = by_id[original_id]
                    assert node["symbol"] == item["symbol"] and pd.Timestamp(node["known_at"]) < pd.Timestamp(item["known_at"])
                    assert node["root_id"] in {n["root_id"] for n in mentions[key]}
                    target["root_id"] = node["root_id"]
                    target["identity_status"] = "EXPLICIT_TARGET_WITH_EXISTING_PRIOR_ORIGINAL"
                    row["confirmed_target_roots"].append(node["root_id"])
                else:
                    target.update(root_id=None, identity_status="TARGET_IDENTIFIED_ORIGINAL_DOCUMENT_MISSING")
                targets.append(target)
            row["targets"] = targets
            row["confirmed_target_roots"] = sorted(set(row["confirmed_target_roots"]))
            row["context_only_roots"] = sorted({m["root_id"] for m in mentions[key]} - set(row["confirmed_target_roots"]))
        output.append(row)
    save(OUT / "change_ledger.json", output)
    flat = [{"document_id": r["document_id"], "symbol": r["symbol"], "known_at": r["known_at"], "title": r["title"],
             "action": r["action"], "approval_state": r["approval_state"], "review_status": r["review_status"],
             "confirmed_target_roots": "|".join(r["confirmed_target_roots"]),
             "context_only_roots": "|".join(r.get("context_only_roots", [])),
             "missing_target_originals": sum(t.get("original_id") is None for t in r.get("targets", [])),
             "trading_feature_admitted": False} for r in output]
    pd.DataFrame(flat).to_csv(OUT / "逐公告变更对象与审批状态.csv", index=False, encoding="utf-8-sig")
    queries = []
    for row in output:
        for root_id in row["confirmed_target_roots"]:
            for side, clock in [("BEFORE", pd.Timestamp(row["known_at"]) - pd.Timedelta(seconds=1)), ("AT", pd.Timestamp(row["known_at"]))]:
                queries.append({"root_id": root_id, "query_at": clock.isoformat(), "side": side,
                                "result": latest_reviewed_change(output, root_id, clock)})
    save(OUT / "asof_change_queries.json", queries)
    result = {"at": now(), "study_id": "510300_FACTOR96_REPURCHASE_CHANGE_CHAIN_V1",
              "status": "CHANGE_TARGETS_AND_APPROVAL_STAGES_REVIEWED_FULL_M02_CHAIN_PENDING",
              "change_documents": len(output), "reviewed_cards": len(cards),
              "documents_with_prior_candidates": sum(bool(v) for v in mentions.values()),
              "documents_with_confirmed_target_root": sum(bool(r["confirmed_target_roots"]) for r in output),
              "distinct_confirmed_target_roots": len({v for r in output for v in r["confirmed_target_roots"]}),
              "new_document_root_links_vs_prior": sum(bool(r["confirmed_target_roots"]) and r["prior_root_id"] is None for r in output),
              "context_only_root_mentions": sum(len(r.get("context_only_roots", [])) for r in output),
              "partial_target_documents": sum(any(t.get("original_id") is None for t in r.get("targets", [])) and bool(r["confirmed_target_roots"]) for r in output),
              "review_status_counts": dict(Counter(r["review_status"] for r in output)),
              "approval_state_counts": dict(Counter(r["approval_state"] for r in output)),
              "asof_queries": len(queries), "same_clock_ambiguous_queries": sum(q["result"]["status"] == "SAME_CLOCK_MULTIPLE_DOCUMENTS_UNRESOLVED" for q in queries),
              "full_purpose_termination_chain_established": False, "free_float_denominator_established": False,
              "T12": "NOT_RUN", "new_accounts": 0, "new_returns": 0, "new_network_requests": 0,
              "goal_status": "active", "goal_achieved": False, "external_review": "NOT_PERFORMED", "orders_authorized": False}
    save(OUT / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="回购变更对象与审批阶段的来源整理")
    parser.add_argument("action", choices=["prepare", "freeze", "run"])
    args = parser.parse_args()
    {"prepare": prepare, "freeze": freeze, "run": run}[args.action]()
