"""用双源行情、完整上交所公告和原已核实分红续算固定策略，不恢复执行。"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.new_daily_input_adapter_v1 import read, location, relative, completed_dates, parse_prices, compare_prices, append_inputs, raw
from research.intraday_overnight_increment_v1 import now, digest, require, write_json
from research.official_dividend_coverage_refresh_v1 import check_http_clock, validate_announcements
from research.fixed_date_continuation_v1 import run as run_accounts

PARENT = ROOT / "reports/research/510300_original_frozen_new_evidence_20260925"
OUT = ROOT / "reports/research/510300_original_frozen_sse_completion_20260925"


def record_timing(source_folder, account_folder, dates):
    """只采用实际输入检查点的保存时间，不沿用更早研究起点的时钟。"""
    folders = sorted(account_folder.glob("accounts/*/*"))
    require(len(folders) == 22, "固定依赖图账户数不完整")
    source_clocks, output_clocks = {}, {}
    for folder in folders:
        key = (folder.parent.name, folder.name)
        previous = source_folder / "accounts" / key[0] / key[1]
        source_clocks[key] = read(previous / "recording.json")["actual_recorded_at"]
        output_clocks[key] = read(folder / "recording.json")["actual_recorded_at"]
    by_date, rows = [], []
    for day in dates:
        opening = day.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=30)
        clocks = source_clocks if day == dates[0] else output_clocks
        timely = sum(pd.Timestamp(moment) < opening for moment in clocks.values())
        by_date.append({"date": day.date().isoformat(), "timely_source_accounts": timely,
                        "all_22_incoming_decisions_saved_before_open": timely == 22})
        for cost in ["BASE", "STRESS"]:
            ledger = pd.read_parquet(account_folder / "accounts" / cost / "SELECTED_MIX_BAND10_SIMPLE2" / "ledger.parquet")
            row = ledger[ledger.date.eq(day)].iloc[0]
            rows.append({**by_date[-1], "cost": cost, "equity": float(row.equity), "net_return": float(row.net_return),
                         "shares": int(row.shares), "filled_quantity": int(row.filled_quantity),
                         "commission": float(row.commission), "slippage_cost": float(row.slippage_cost)})
    write_json(OUT / "actual_source_recording_times.json", {
        "source": [{"cost": key[0], "node": key[1], "recorded_at": value} for key, value in source_clocks.items()],
        "rule": "第一新增日用对应原检查点的实际记录时刻；之后用本次新决定实际记录时刻。",
        "unrelated_earlier_origin_timestamp_not_used": True}, exclusive=True)
    pd.DataFrame(rows).to_parquet(OUT / "new_observed_daily_returns.parquet", index=False)
    return by_date, rows


def main():
    require(not OUT.exists(), "本次来源补充已开始或完成，先读取保存记录，不能覆盖")
    settings = read(PARENT / "settings.json")
    source = settings["initial_source"]
    official = read(location(settings["official_configuration"]))
    source_dir = PARENT / "2026-09-24"
    old_prices = pd.read_parquet(location(source["prices"]))
    old_features = pd.read_parquet(location(source["features"]))
    dates, cutoff, next_day = completed_dates(pd.to_datetime(pd.read_csv(location(settings["calendar"])).trade_date),
                                             old_prices.date.iloc[-1], now())
    require(cutoff == "2026-09-24" and len(dates) == 6, "来源补充限定原已请求的六个新增交易日")
    OUT.mkdir(parents=True)
    references = {name: read(source_dir / f"{name}_attempt1.json") for name in ["sina", "tencent"]}
    sse_receipts = [read(source_dir / "sse_page_1_attempt2.json")]
    pages = [json.loads(raw(sse_receipts[0]))]
    require(int(pages[0]["pageHelp"]["pageCount"]) == len(pages), "已保存上交所分页不完整")
    for receipt in sse_receipts:
        check_http_clock(receipt["headers"], pd.Timestamp(receipt["retrieved_at"]).to_pydatetime(),
                         pd.Timestamp(cutoff).date(), official)
    coverage = read(location(source["coverage"]))
    announcements = validate_announcements(pages, official, pd.Timestamp(cutoff).date(), coverage)
    start = old_prices.date.iloc[max(0, len(old_prices) - 10)].date().isoformat()
    sina, tencent = [parse_prices(name, raw(references[name]), start, cutoff) for name in ["sina", "tencent"]]
    comparison = compare_prices(sina, tencent, old_prices, dates)
    protocol = {
        "at": now(), "study_id": "510300_ORIGINAL_FROZEN_SSE_COMPLETION_20260925",
        "user_authority": "reports/research/510300_new_evidence_resume_20260925/authority_update.json",
        "purpose": "旧夏普1.2固定规则的新日期研究对照，不改规则、不新增训练或选择候选。",
        "source_contract": "新浪和腾讯逐字段交叉核对、旧行情前缀完全不变；此前已核实分红事件不变，上交所从2026-01-01至截止日完整公告无新分红或拆合事件。",
        "source_change": "基金官网三次请求代理失败；本次分红新增日期覆盖仅由完整上交所公告支持，不能宣称通过原双官方页面接纳流程。",
        "prior_failures_preserved": [relative(source_dir / name) for name in ["attempt1_failure.json", "attempt2_failure.json", "manager_default_client/receipt.json"]],
        "source_checks": announcements, "new_trading_days": [d.date().isoformat() for d in dates],
        "input_source": source, "capital": 200000, "fixed_strategy": "SELECTED_MIX_BAND10_SIMPLE2",
        "current_two_year_tail_contract_compliant": False, "new_parameter_search": 0, "new_model_fits": 0,
        "evidence_limit": "新取得的后续日期价格可以检查原固定规则；当时决定文件未在开盘前保存的日期，不计为及时前向验证。六日也不能证明长期稳定性。",
        "automatic_tasks_resumed": False, "orders_authorized": False, "goal_achieved": False,
    }
    write_json(OUT / "protocol.json", protocol, exclusive=True)
    write_json(OUT / "freeze.json", {"at": now(), "code_sha256": digest(Path(__file__)),
        "strategy_engine_sha256": digest(ROOT / "research/fixed_date_continuation_v1.py"),
        "protocol_sha256": digest(OUT / "protocol.json"),
        "price_receipts": [r["receipt_file"] for r in references.values()],
        "sse_receipts": [r["receipt_file"] for r in sse_receipts],
        "source_checkpoints": {relative(p): digest(p) for p in sorted(location(source["source_folder"]).glob("accounts/*/*/checkpoint.json"))}}, exclusive=True)
    prices, features = append_inputs(old_prices, old_features, sina, pd.read_csv(location(source["dividends"])), references["sina"]["retrieved_at"])
    prices.to_parquet(OUT / "candidate_prices.parquet", index=False)
    features.to_parquet(OUT / "candidate_features.parquet", index=False)
    comparison.to_parquet(OUT / "price_comparison.parquet", index=False)
    coverage.update(coverage_end=cutoff, retrieved_at=now(), event_ledger_changed=False,
                    coverage_extension_method="PRIOR_VERIFIED_REGISTER_PLUS_FRESH_COMPLETE_SSE_NOTICES_RESEARCH_ONLY",
                    coverage_refresh_evidence=announcements,
                    current_manager_page_confirmed=False)
    write_json(OUT / "candidate_dividend_coverage.json", coverage, exclusive=True)
    admission = {"accepted": True, "completed_at": now(), "cutoff": cutoff, "dividend_coverage_through": cutoff,
        "features_sha256": digest(OUT / "candidate_features.parquet"), "dividends_sha256": digest(location(source["dividends"])),
        "evidence_class": "NEW_COMPLETE_DAILY_INPUTS", "source_contract": protocol["source_contract"],
        "source_contract_changed": True, "original_dual_official_page_admission_passed": False,
        "old_price_rows_preserved": len(old_prices), "old_feature_rows_preserved": len(old_features),
        "new_trading_days": len(dates), "old_prefix_exact": True, "new_model_fits": 0,
        "raw_source_receipts": [r["receipt_file"] for r in [*references.values(), *sse_receipts]],
        "source_cost_cny": 0}
    write_json(OUT / "admission_receipt.json", admission, exclusive=True)
    account_settings = {"configuration": settings["strategy_configuration"], "source_folder": source["source_folder"],
        "output_folder": relative(OUT / "accounts_run"), "source_features": source["features"],
        "features": relative(OUT / "candidate_features.parquet"), "dividends": source["dividends"],
        "ridge_models": source["ridge_models"], "within_models": source["within_models"], "cutoff": cutoff,
        "admission_receipt": relative(OUT / "admission_receipt.json"), "mode": "FIXED_RESEARCH_CONTINUATION",
        "research_origin": settings["research_origin"]}
    write_json(OUT / "account_settings.json", account_settings, exclusive=True)
    result = run_accounts(OUT / "account_settings.json")
    days, rows = record_timing(location(source["source_folder"]), OUT / "accounts_run", dates)
    periods = []
    for cost in ["BASE", "STRESS"]:
        frame = pd.DataFrame([r for r in rows if r["cost"] == cost])
        previous = pd.read_parquet(location(source["source_folder"]) / "accounts" / cost / "SELECTED_MIX_BAND10_SIMPLE2/ledger.parquet")
        previous_equity = float(previous.equity.iloc[-1])
        periods.append({"cost": cost, "start_equity": previous_equity, "end_equity": float(frame.equity.iloc[-1]),
                        "new_period_net_return": float(np.prod(1 + frame.net_return) - 1),
                        "filled_days": int(frame.filled_quantity.ne(0).sum()), "exposed_days": int(frame.shares.gt(0).sum()),
                        "fees": float(frame.commission.sum() + frame.slippage_cost.sum()),
                        "new_period_sharpe": None, "short_period_not_annualized": True})
    final = {"at": now(), "status": "COMPLETED_FIXED_ORIGINAL_REPLAY_WITH_SSE_SOURCE_SUBSTITUTION",
        "study_id": protocol["study_id"], "cutoff": cutoff, "next_official_trading_day": next_day,
        "new_observed_trading_days": len(dates), "new_timely_incoming_decision_days": sum(d["all_22_incoming_decisions_saved_before_open"] for d in days),
        "timing": days, "new_periods": periods, "all_metrics": result["all_metrics"],
        "old_account_rows_reused": result["old_account_rows_reused"], "incremental_account_rows": result["incremental_account_rows"],
        "new_model_fits": 0, "new_candidates": 0, "goal_achieved": False, "orders_authorized": False,
        "current_two_year_tail_contract_compliant": False,
        "source_contract_changed": True, "independent_stability_proven": False,
        "actual_source_recording_clock_used": True, "current_market_view": "NO_VIEW"}
    write_json(OUT / "result.json", final, exclusive=True)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
