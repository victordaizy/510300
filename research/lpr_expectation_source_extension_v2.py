"""合并七个核对月份，保持旧事实，复算全月母集；不读取证券行情。"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from lpr_expectation_identification_v1 import compute, require, read_csv, write_json


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_lpr_expectation_source_extension_v2"
FIELDS = ["month", "uid", "n", "cut1_min", "cut1_max", "cut5_min", "cut5_max", "hold1_min", "hold5_min", "median1_min_bp", "median1_max_bp", "median5_min_bp", "median5_max_bp", "paragraphs", "explanation"]
ADDITIONS = [
    ("2021-02", "yahoo_2021_02", 35, 1, 1, 0, 1, 31, 31, 0, 0, 0, 0, "31人双不变、3人双加息5bp、1人只明确降一年期；后者五年期未交代，保留0至1降息票。"),
    ("2021-09", "yahoo_2021_09", 20, 1, 1, 0, 0, 19, 20, 0, 0, 0, 0, "19人双不变、1人一年期降5bp且五年期不变。"),
    ("2022-07", "yahoo_2022_07", 22, 0, 0, 9, 9, 22, 0, 0, 0, None, None, "22人一年期不变；9人明确五年期降息，其余13人没有逐项明示，不能强填持平或中位数。"),
    ("2023-04", "kfgo_2023_04", 30, 0, 3, 0, 3, 27, 27, 0, 0, 0, 0, "27人双不变、3人预计至少一期降5bp；逐期限票数保留0至3。旧标题筛选漏掉may hold，本轮按正文补入。"),
    ("2023-05", "dunya_2023_05", 26, 1, 3, 3, 3, 23, 23, 0, 0, 0, 0, "23人双不变；2人只明示五年期降息，1人双降5bp。一年期降息票保留1至3；采用页面更新时间作为可用时间上界。"),
    ("2024-05", "yahoo_2024_05", 33, 2, 2, 6, 6, 31, 27, 0, 0, 0, 0, "27人双不变、4人一年期不变及五年期降5至20bp、2人双降同类幅度。"),
    ("2026-06", "yahoo_2026_06", 30, 0, 0, 0, 0, 30, 30, 0, 0, 0, 0, "30人均预计两期限不变；不得误用别月20人或23人。"),
]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_write(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def article_nodes(value):
    if isinstance(value, dict):
        if value.get("datePublished") and value.get("dateModified"):
            yield value
        for nested in value.values():
            yield from article_nodes(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from article_nodes(nested)


def collect_sources() -> tuple[list[dict], list[dict], list[dict]]:
    facts, candidates, sources = [], [], []
    for values in ADDITIONS:
        month, key, *numeric, note = values
        receipt = load(OUT / "receipts" / (key + ".json"))
        require(receipt["status"] in {"SAVED_HTTP", "SAVED_LOCAL_CATALOGUE"}, "来源没有保存成功。")
        require(digest(OUT / receipt["raw_path"]) == receipt["sha256"], "来源身份改变。")
        body = (OUT / receipt["text_path"]).read_text(encoding="utf-8")
        require("Reuters" in body and ("survey" in body or "poll" in body), "没有Reuters调查正文。")
        if key.startswith("yahoo_"):
            nodes = list(article_nodes(load(OUT / "raw_local_only" / (key + "_structured.json"))))
            require(len(nodes) == 1, "需要唯一的有日期新闻对象。")
            published, modified = nodes[0]["datePublished"], nodes[0]["dateModified"]
            clock = "JSONLD_PUBLICATION_AND_MODIFICATION"
        elif key == "kfgo_2023_04":
            published, modified = receipt["published_utc"] + "+00:00", receipt["modified_utc"] + "+00:00"
            clock = "SAVED_CATALOGUE_UTC"
        else:
            require("19 May 23, 14:01:24 UTC" in body and "Updated on" in body, "页面更新时间核对失败。")
            published = modified = "2023-05-19T14:01:24+00:00"
            clock = "VISIBLE_UPDATE_UPPER_BOUND_NOT_FIRST_PUBLICATION"
        source = {
            "month": month, "uid": key, "source_url": receipt["url"], "source_sha256": receipt["sha256"],
            "published_available_upper_at": published, "modified_upper_at": modified, "clock_kind": clock,
            "history_first_version_certified": False, "note": note,
        }
        candidates.append({"uid": key, "month": month, "source_url": receipt["url"], "published_at": published, "modified_at": modified, "candidate_status": "CANDIDATE_PRE_RELEASE_POLL", "page_sha256": receipt["sha256"], "content_sha256": digest(OUT / receipt["text_path"]), "clock_kind": clock})
        facts.append(dict(zip(FIELDS, [month, key] + ["" if value is None else str(value) for value in numeric] + ["人工全文核对，见new_sources.json", note])))
        sources.append(source)
    return facts, candidates, sources


def finalize_summary(root: Path, ledger: list[dict], summary: dict) -> dict:
    old = read_csv(root / "inputs/old_survey_facts.csv")
    new = load(root / "new_facts.json")
    combined = read_csv(root / "inputs/survey_facts.csv")
    require(combined == sorted(old + new, key=lambda r: r["month"]), "旧事实改变或合并表不符。")
    logs = load(root / "evidence/本轮检索清单.json")
    queries = sum(len(row["queries"]) for row in logs)
    require(queries == 36, "本轮检索应为12次初查加24次冻结后检索。")
    for source in load(root / "new_sources.json"):
        receipt = load(root / "receipts" / (source["uid"] + ".json"))
        require(receipt["sha256"] == source["source_sha256"], "保存来源回执与准入不符。")
    receipts = [load(path) for path in (root / "receipts").glob("*.json")]
    http = [r for r in receipts if "http_status" in r]
    require(len(http) <= load(root / "source_plan.json")["new_download_limit"], "下载超过登记预算。")
    summary.update(study_id="510300_LPR_EXPECTATION_SOURCE_EXTENSION_V2", status="COMPLETED_BOUNDED_SOURCE_EXTENSION_NO_PREDICTIVE_VIEW", parent_survey_months=len(old), new_survey_months=len(new), old_fact_records_unchanged=True, preflight_queries=12, additional_queries=queries-12, new_http_attempts=len(http), new_http_saved=sum(r["status"] == "SAVED_HTTP" for r in http), new_http_failed=sum(r["status"] != "SAVED_HTTP" for r in http), recovered_local_months=sum(r["status"] == "SAVED_LOCAL_CATALOGUE" for r in receipts))
    summary["direction_scope"] = "仅基于扩样后56个月调查及其数值约束；不是完整84个月全部政策意外，不是证券方向。"
    summary["remaining_missing_months"] = [r["month"] for r in ledger if r["status"] == "NO_ADMITTED_NUMERIC_SURVEY_IN_SAVED_SOURCE_SET"]
    summary["same_search_not_to_restart_under_new_version"] = True
    return summary


def build() -> None:
    require(not (OUT / "source_freeze_receipt.json").exists(), "来源已经封存，禁止覆盖。")
    new, candidates, sources = collect_sources()
    old = read_csv(OUT / "inputs/old_survey_facts.csv")
    require(not ({r["month"] for r in old} & {r["month"] for r in new}), "新旧月份重复。")
    write_json(OUT / "new_facts.json", new)
    write_json(OUT / "new_sources.json", sources)
    csv_write(OUT / "inputs/survey_facts.csv", sorted(old + new, key=lambda r: r["month"]))
    csv_write(OUT / "inputs/candidates.csv", read_csv(OUT / "inputs/old_candidates.csv") + candidates)
    csv_write(OUT / "新增七个月调查事实.csv", new)
    protocol = {
        "study_id": "510300_LPR_EXPECTATION_SOURCE_EXTENSION_V2",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {name: {"sha256": digest(OUT / "inputs" / name)} for name in ["actual.csv", "survey_facts.csv", "candidates.csv", "old_survey_facts.csv", "old_candidates.csv"]},
        "calculation": "继承lpr_expectation_identification_v1.compute，只计算票数、调查中位数与实际政策差异；证券收益未读取。",
        "clock_note": "published_at对Dunya记录是更新时间给出的可用上界，不声称是首次发布时间；其他新记录使用JSONLD或原目录UTC。",
        "equity_outcome_reads_before_freeze": 0, "new_models": 0, "new_accounts": 0,
    }
    write_json(OUT / "protocol.json", protocol)
    ledger, summary = compute(OUT)
    summary = finalize_summary(OUT, ledger, summary)
    write_json(OUT / "ledger.json", ledger)
    write_json(OUT / "summary.json", summary)
    csv_write(OUT / "全部84月事前预期识别.csv", ledger)
    write_json(OUT / "source_freeze_receipt.json", {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_files": {n: digest(OUT / n) for n in ["source_plan.json", "protocol.json", "new_facts.json", "new_sources.json", "summary.json", "ledger.json"]},
        "source_set": [{"uid": r["uid"], "source_sha256": r["source_sha256"]} for r in sources],
        "new_equity_labels": 0, "new_models": 0, "new_accounts": 0,
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def verify(root: Path) -> None:
    for name, value in load(root / "source_freeze_receipt.json")["frozen_files"].items():
        require(digest(root / name) == value, "冻结文件身份改变。")
    ledger, summary = compute(root)
    summary = finalize_summary(root, ledger, summary)
    require(ledger == load(root / "ledger.json") and summary == load(root / "summary.json"), "保存结果与复算不同。")
    displayed = read_csv(root / "全部84月事前预期识别.csv")
    require(len(displayed) == len(ledger), "CSV月份数量不同。")
    for row, derived in zip(displayed, ledger):
        require(all(value == ("" if derived.get(k) is None else str(derived[k])) for k, value in row.items()), "CSV显示与结果不同。")
    feasibility = load(root / "prediction_feasibility.json")
    parent = load(root / "inputs/parent_lpr_prediction_protocol.json")
    require(digest(root / "inputs/parent_lpr_prediction_protocol.json") == feasibility["parent_protocol_sha256"], "父级样本要求身份不符。")
    train = parent["minimum_train_months"]
    evaluation = parent["minimum_eval_months"]
    require(feasibility["minimum_train_months"] == train == 36 and feasibility["minimum_eval_months"] == evaluation == 24, "继承样本门不符。")
    bounds = {
        "optimistic_max_evaluation_any_representation": summary["saved_survey_months"] - train,
        "exact_1y_median_optimistic_max_evaluation": summary["counts"]["t1_exact_median_months"] - train,
        "exact_both_median_optimistic_max_evaluation": summary["counts"]["both_tenors_median_change_exact"] - train,
        "exact_both_vote_optimistic_max_evaluation": summary["counts"]["both_tenors_cut_vote_share_exact"] - train,
    }
    require(all(feasibility[k] == max(0, v) for k, v in bounds.items()), "样本上界复算不符。")
    unresolved = load(root / "evidence/可见原文但版本时钟未齐的三个月.json")
    require(feasibility["three_version_unresolved_candidates"] == [r["month"] for r in unresolved] and all(r["admitted"] is False for r in unresolved), "候选未准入状态不符。")
    require(feasibility["upper_bound_if_all_three_eventually_admitted"] == summary["saved_survey_months"] + len(unresolved) - train < evaluation, "候选全部补齐的乐观上界不符。")
    require(not feasibility["source_set_could_pass_inherited_sample_requirement"] and feasibility["account_stage"] == "NOT_RUN_INHERITED_LPR_SAMPLE_GATE", "账户阶段不符。")
    print(json.dumps({"status": "PASS_SOURCE_EXTENSION_SAVED_FACT_RECOMPUTATION", "months": 84, "surveys": summary["saved_survey_months"], "new_equity_labels": 0}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="有限扩展事前调查并保持历史事实")
    parser.add_argument("action", choices=["build", "verify"])
    parser.add_argument("--root", type=Path, default=OUT)
    args = parser.parse_args()
    if args.action == "build":
        build()
    else:
        verify(args.root)
