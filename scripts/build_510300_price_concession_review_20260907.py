"""构建论证核查包并验证结构；不运行研究策略、读取新行情或连接券商。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/research/510300_price_concession_review_20260907"
ZIP_NAME = "510300_PRICE_CONCESSION_REVIEW_20260907_GPT_REVIEW.zip"
BYTE_LIMIT = 512_000_000
CN = timezone(timedelta(hours=8))
CONTEXT_FILES = [
    "config/510300_research_authority_v6.json",
    "docs/510300_SHARPE_1_2_RESEARCH_RESUMPTION_20260906.md",
    "config/510300_post_close_stale_price_capture_v2.yaml",
    "config/510300_post_close_stale_price_capture_v2_source_receipt.json",
    "config/510300_post_close_stale_price_capture_v2_manifest.json",
    "docs/510300_POST_CLOSE_STALE_PRICE_CAPTURE_V2_SPEC.md",
    "research/post_close_stale_price_capture_v2.py",
    "scripts/collect_510300_post_close_stale_price_capture_v2.py",
    "scripts/freeze_510300_post_close_stale_price_capture_v2.py",
    "scripts/mature_510300_post_close_stale_price_capture_v2.py",
    "scripts/render_510300_post_close_stale_price_capture_v2_status.py",
    "tests/test_510300_post_close_stale_price_capture_v2.py",
    "reports/forward/510300_POST_CLOSE_STALE_PRICE_CAPTURE_V2/g0_status.json",
    "reports/forward/510300_POST_CLOSE_STALE_PRICE_CAPTURE_V2/g0_evidence_20260904.json",
    "reports/forward/510300_POST_CLOSE_STALE_PRICE_CAPTURE_V2/G0_STATUS.md",
    "config/510300_anchored_sparse_mean_reversion_grid_v1.yaml",
    "config/510300_anchored_sparse_mean_reversion_grid_v1_manifest.json",
    "reports/research/510300_anchored_sparse_mean_reversion_grid_v1.json",
    "reports/research/510300_ANCHORED_SPARSE_MEAN_REVERSION_GRID_V1.md",
    "reports/research/510300_anchored_sparse_mean_reversion_grid_v1_input_audit.json",
    "reports/frozen/510300_anchored_sparse_mean_reversion_grid_v1_freeze_receipt.json",
    "config/510300_episodic_alpha_library_v1.yaml",
    "docs/510300_EPISODIC_ALPHA_LIBRARY_V1_SPEC.md",
    "reports/research/510300_episodic_alpha_library_v1_post_run_adjudication.json",
    "reports/research/510300_EPISODIC_ALPHA_LIBRARY_V1_POST_RUN_ADJUDICATION.md",
    "reports/research/510300_episodic_alpha_library_batch_2_resource_adjudication.json",
    "reports/research/510300_EPISODIC_ALPHA_LIBRARY_BATCH_2_RESOURCE_ADJUDICATION.md",
    "reports/data_quality/510300_primary_market_readiness.json",
    "data/reference/sse_trade_calendar_2026.csv",
    "data/reference/sse_trade_calendar_2026.metadata.json",
]


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def copy_evidence(source: Path, target: Path) -> dict:
    raw = source.read_bytes()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != raw:
        raise RuntimeError(f"来源与已保存副本不同，保留原副本：{source}")
    target.write_bytes(raw)
    return {"source_path": str(source), "member_path": target.relative_to(REPORT).as_posix(),
            "bytes": len(raw), "sha256": digest(raw),
            "role": "本轮输入或保存背景；原始日期不变"}


def economic_arithmetic() -> dict:
    calendar_path = REPORT / "context/data/reference/sse_trade_calendar_2026.csv"
    with calendar_path.open(encoding="utf-8-sig", newline="") as handle:
        calendar = list(csv.DictReader(handle))
    days = [r["trade_date"] for r in calendar if "2026-07-06" <= r["trade_date"] <= "2026-09-04"]
    sharpe_rows = []
    for n in (10, 25, 50):
        mu = 1.2 * 0.01 / math.sqrt(n - 1.2**2 * (1 - n / 252))
        recovered = math.sqrt(n) * mu / math.sqrt(0.01**2 + (1 - n / 252) * mu**2)
        if not math.isclose(recovered, 1.2, abs_tol=1e-12):
            raise RuntimeError("说明性夏普公式复算不一致")
        sharpe_rows.append({"annual_events": n, "required_event_net_mean_bps": mu * 10_000,
                            "recovered_annual_sharpe": recovered})
    costs = []
    for notional in (25_000, 20_000, 10_000, 5_000):
        commission = 2 * max(notional * 0.0002, 5) / notional * 10_000
        costs.append({"notional_cny": notional, "round_trip_commission_bps": commission,
                      "v2_style_base_5bp_each_leg_bps": commission + 10,
                      "v2_style_stress_10bp_each_leg_bps": commission + 20,
                      "illustrative_fixed_close_buy_zero_plus_sell_5bp_bps": commission + 5})
    unweighted = 0.5 * 0.006 + 0.5 * (-0.002)
    weighted = 0.5 * 0.1 * 0.006 + 0.5 * 0.9 * (-0.002)
    if not math.isclose(weighted * 10_000, -6.0, abs_tol=1e-12):
        raise RuntimeError("说明性逆向选择算术不一致")
    return {"classification": "ILLUSTRATIVE_NOT_EMPIRICAL", "market_price_series_read": False,
            "cost_assumptions": "等额买卖，仅示意佣金与假定滑点；未测量实际券商费用，不改V2成本", 
            "round_trip_costs": costs,
            "adverse_selection": {"signal_mean_gross_bps": unweighted * 10_000,
                                  "plan_capital_mean_gross_bps": weighted * 10_000},
            "sharpe_assumptions": "252日；独立单日事件；现金日超额0；事件条件标准差1%；等规模充分成交",
            "formula": "mu = S*sigma / sqrt(N - S^2*(1-N/252))",
            "event_mean_requirements": sharpe_rows,
            "calendar_count": {"from": "2026-07-06", "through": "2026-09-04",
                "completed_trading_days": len(days), "trade_dates": days,
                "qualification": "日历计数，不是完整可研究或实际成交样本数"}}


def verify(path: Path) -> dict:
    failures = []
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        bad = archive.testzip()
        if bad is not None:
            failures.append(f"CRC失败：{bad}")
        if len(names) != len(set(names)):
            failures.append("成员名重复")
        rows = list(csv.DictReader(io.StringIO(archive.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        if len(rows) != len({r["path"] for r in rows}):
            failures.append("索引行重复")
        if {r["path"] for r in rows} != set(names) - {"FILE_INDEX.csv"}:
            failures.append("索引覆盖不一致")
        for row in rows:
            raw = archive.read(row["path"])
            if len(raw) != int(row["bytes"]) or digest(raw) != row["sha256"]:
                failures.append(f"大小或哈希不一致：{row['path']}")
        mapping = list(csv.DictReader(io.StringIO(archive.read("07_证据映射.csv").decode("utf-8-sig"))))
        for row in mapping:
            raw = archive.read(row["member_path"])
            if len(raw) != int(row["bytes"]) or digest(raw) != row["sha256"]:
                failures.append(f"直接证据与复制记录不一致：{row['member_path']}")
        facts = json.loads(archive.read("05_说明性算术.json"))
        if facts["calendar_count"]["completed_trading_days"] != 45:
            failures.append("制度后日历计数不一致")
        for row in facts["event_mean_requirements"]:
            mu = row["required_event_net_mean_bps"] / 10_000
            n = row["annual_events"]
            reproduced = math.sqrt(n) * mu / math.sqrt(0.01**2 + (1 - n/252) * mu**2)
            if not math.isclose(reproduced, 1.2, abs_tol=1e-12):
                failures.append("保存的说明性夏普算术未通过复算")
        raw_receipt = json.loads(archive.read("sources/raw_capture_receipt.json"))
        for row in raw_receipt["sources"]:
            if row["status"] == "CAPTURED_RAW_BYTES":
                raw = archive.read("sources/" + row["filename"])
                if len(raw) != row["bytes"] or digest(raw) != row["sha256"]:
                    failures.append(f"制度原文回执不一致：{row['filename']}")
    size = path.stat().st_size
    if size >= BYTE_LIMIT:
        failures.append("压缩包达到或超过512,000,000字节上限")
    return {"status": "PASS_STRUCTURAL_AND_ILLUSTRATIVE_ARITHMETIC" if not failures else "FAILED",
            "verified_at": datetime.now(CN).isoformat(), "zip_path": str(path), "bytes": size,
            "member_count": len(names), "indexed_count": len(rows),
            "copied_evidence_count": len(mapping), "sha256": digest(path.read_bytes()),
            "checks": ["CRC", "DUPLICATE_MEMBERS", "INDEX_COVERAGE", "SIZE_AND_SHA256",
                       "COPIED_EVIDENCE_IDENTITY", "RULE_CAPTURE_RECEIPTS", "ILLUSTRATIVE_FORMULA_RECOMPUTATION"],
            "security_audit": False, "external_gpt_review": False,
            "new_accounts": 0, "model_fits": 0, "new_market_price_reads": 0,
            "failures": failures}


def build() -> dict:
    final = ROOT / "deliverables" / ZIP_NAME
    if final.exists():
        raise RuntimeError("本轮ZIP已存在；请使用--verify进行只读复核")
    required_docs = ["00_README_FIRST.md", "01_GPT_REVIEW_PROMPT.md", "02_论证核实与研究判断.md",
                     "03_下一步研究方案_待执行.md", "04_来源与证据说明.md"]
    if any(not (REPORT / name).is_file() for name in required_docs):
        raise RuntimeError("本轮论证文件尚未齐备")
    mapping = [copy_evidence(ROOT / path, REPORT / "context" / path) for path in CONTEXT_FILES]
    attachment = Path(r"E:\CodexData\.codex\attachments\5505f5b7-b772-4e69-97b4-c0f4c27fa2b4\pasted-text.txt")
    mapping.append(copy_evidence(attachment, REPORT / "user/用户提供原文.txt"))
    for name in (Path(__file__).name, "capture_510300_price_concession_rule_sources_20260907.py"):
        mapping.append(copy_evidence(ROOT / "scripts" / name, REPORT / "code" / name))
    with (REPORT / "07_证据映射.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_path", "member_path", "bytes", "sha256", "role"])
        writer.writeheader()
        writer.writerows(mapping)
    facts = economic_arithmetic()
    write_json(REPORT / "05_说明性算术.json", facts)
    g0 = read_json(REPORT / "context/reports/forward/510300_POST_CLOSE_STALE_PRICE_CAPTURE_V2/g0_status.json")
    authority = read_json(REPORT / "context/config/510300_research_authority_v6.json")
    original_rule_receipt = read_json(REPORT / "context/config/510300_post_close_stale_price_capture_v2_source_receipt.json")
    raw_receipt = read_json(REPORT / "sources/raw_capture_receipt.json")
    captured = {r["filename"]: r for r in raw_receipt["sources"]}
    rules_match = {
        "rule": captured["sse_2026_rule.docx"].get("sha256") == original_rule_receipt["sse_rule"]["rule_document_sha256"],
        "delayed_provisions": captured["sse_2026_delayed_provisions.docx"].get("sha256") == original_rule_receipt["sse_rule"]["delayed_provisions_document_sha256"],
    }
    ledger_paths = ["data/forward/510300_post_close_stale_price_capture_v2/observation_ledger.jsonl",
                    "data/forward/510300_post_close_stale_price_capture_v2/target_maturity_ledger.jsonl"]
    state = {
        "review_id": "510300_PRICE_CONCESSION_MECHANISM_REVIEW_20260907",
        "status": "REVIEW_COMPLETE_PLAN_NOT_RUN", "generated_at": datetime.now(CN).isoformat(),
        "scope_assumption": "用户提供机制文字未单列执行要求；澄清等待期间先完成论证核实及待执行建议",
        "recommendation": "FIRST_MECHANISM_SOURCE_AND_EXECUTION_FEASIBILITY_FIRST",
        "new_strategy_registered": False, "new_forward_ledger_created": False,
        "current_authority": authority["authority_id"],
        "new_method_research_authorized": authority["new_method_research_authorized"],
        "historical_walk_forward_training_authorized": authority["historical_walk_forward_training_authorized"],
        "current_model_action": authority["model_action"],
        "current_model_position_target": authority["model_position_target"],
        "position_impact": 0, "broker_connection_authorized": False, "live_trading_authorized": False,
        "current_validated_high_sharpe_strategy": authority["current_validated_high_sharpe_strategy"],
        "net_sharpe": "NOT_COMPUTED", "new_event_returns": "NOT_COMPUTED",
        "actual_fill_fraction": "UNKNOWN_NO_ORDER_EVIDENCE",
        "old_branch_status_writes": 0, "new_market_data_downloads": 0,
        "model_fits": 0, "new_accounts": 0, "new_observations": 0,
        "public_rule_raw_download_count": sum(r["status"] == "CAPTURED_RAW_BYTES" for r in raw_receipt["sources"]),
        "rule_bytes_match_original_v2_receipt": rules_match,
        "v2_last_saved_g0": {"updated_at": g0["updated_at"], "state": g0["g0_state"],
            "blockers": g0["blockers"], "authoritative_forward_start": g0["authoritative_forward_start"],
            "saved_ledger_counts": g0["ledger_counts"],
            "filesystem_observed_at": datetime.now(CN).isoformat(),
            "configured_ledger_file_existence": {p: (ROOT / p).exists() for p in ledger_paths},
            "clock_and_broker_remeasured_this_round": False},
        "sources": "来源与范围见04_来源与证据说明.md及sources原始回执",
        "security_audit": False, "external_gpt_review_received": False,
    }
    write_json(REPORT / "06_本轮状态.json", state)
    write_json(REPORT / "08_包含与排除说明.json", {
        "included": ["本轮报告、方案、说明性计算、输入原文、来源回执及制度原文", "直接引用的现存研究状态与配置", "本轮构建与来源保存脚本"],
        "excluded": ["既有研究的完整市场原始数据、全部事件路径和未引用代码依赖", "其他研究ZIP及与本次论证无关的研究树"],
        "coverage": "本轮论证核查自包含；旧策略只作背景状态核对，不提供旧策略重跑声明",
        "missing": ["本轮510300同步参考价值样例", "合格盘后行情样例及自身订单回执", "外部GPT实际审阅反馈"],
        "original_frozen_artifacts_mutated": False,
    })
    files = {p.relative_to(REPORT).as_posix(): p.read_bytes() for p in REPORT.rglob("*")
             if p.is_file() and p.name not in {"FILE_INDEX.csv", "交付结构核验.json"}}
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["path", "bytes", "sha256"])
    writer.writeheader()
    for name, raw in sorted(files.items()):
        writer.writerow({"path": name, "bytes": len(raw), "sha256": digest(raw)})
    index_raw = buffer.getvalue().encode("utf-8-sig")
    (REPORT / "FILE_INDEX.csv").write_bytes(index_raw)
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.with_suffix(".building.zip")
    with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in sorted(files.items()):
            archive.writestr(name, raw)
        archive.writestr("FILE_INDEX.csv", index_raw)
    result = verify(temporary)
    if result["status"] != "PASS_STRUCTURAL_AND_ILLUSTRATIVE_ARITHMETIC":
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    os.replace(temporary, final)
    result["zip_path"] = str(final)
    write_json(REPORT / "交付结构核验.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="构建或只读校验510300价格让步论证核查包")
    parser.add_argument("--verify", type=Path, help="对指定ZIP做只读结构与说明性算术复核")
    args = parser.parse_args()
    result = verify(args.verify.resolve()) if args.verify else build()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS_STRUCTURAL_AND_ILLUSTRATIVE_ARITHMETIC":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
