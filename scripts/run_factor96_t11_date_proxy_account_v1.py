"""固定T11日期代理账户实验；只写研究文件，不制作交付包。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports/research/510300_factor96_t11_date_proxy_account_v1"
PREVIOUS = ROOT / "reports/research/510300_factor96_daily_state_shrink_v1"
COHORT = ROOT / "reports/research/510300_factor96_t11_financial_cohorts_v1_0_1"
O02 = ROOT / "reports/research/510300_factor96_t11_o02_date_proxy_definition_v1"


def now():
    return pd.Timestamp.now(tz="Asia/Shanghai").isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(x).isoformat() if pd.notna(x) else None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(clean(value), stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_manifest(base, manifest):
    for row in read(manifest)["files"]:
        assert digest(base / row["path"]) == row["sha256"], f"冻结文件变更：{row['path']}"


def prepare():
    assert not (OUT / "implementation_input_freeze.json").exists(), "已冻结，不能覆盖"
    verify_manifest(OUT, OUT / "economic_freeze.json")
    prior_hashes = {r["path"]: r["sha256"] for r in read(PREVIOUS / "freeze.json")["files"]}
    cohort_hashes = {r["path"]: r["sha256"] for r in read(COHORT / "result_freeze.json")["files"]}
    sources = []

    def copy(source, destination, expected=None):
        actual = digest(source)
        assert expected is None or actual == expected, f"来源不等于冻结版本：{source}"
        target = OUT / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists(), f"不覆盖来源副本：{target}"
        shutil.copy2(source, target)
        sources.append({"original": str(source.relative_to(ROOT)), "snapshot": destination, "sha256": actual})

    for name in ["market.parquet", "dividends.csv", "price_features.parquet", "labels.parquet", "risk_training_records.json"]:
        copy(PREVIOUS / "inputs" / name, "inputs/"+name, prior_hashes["inputs/"+name])
    for name in ["account_engine.py", "factor96_margin_repair_v1.py", "factor96_library_intake_v1.py"]:
        copy(PREVIOUS / "code" / name, "code/"+name, prior_hashes["code/"+name])
    for name in ["daily_financial_cohorts.parquet", "company_financial_cohorts.parquet"]:
        copy(COHORT / name, "inputs/"+name, cohort_hashes[name])
    name = "inputs/repaired_member_report_measurements.parquet"
    copy(COHORT / name, name, cohort_hashes[name])
    copy(COHORT / "code/factor96_t11_financial_cohorts_v1.py", "code/factor96_t11_financial_cohorts_v1.py",
         cohort_hashes["code/factor96_t11_financial_cohorts_v1.py"])
    copy(O02 / "code/factor96_t11_o02_date_proxy_v1.py", "code/factor96_t11_o02_date_proxy_v1.py",
         "8e46c0dda27ce221e9589030343fdd3e8aff431a9e586527423e96aa824f3a1c")
    for name in ["source_receipt.json", "freeze.json", "saved_verification_receipt.json"]:
        copy(PREVIOUS / name, "source_evidence/price_risk_"+name)
    copy(ROOT / "reports/research/510300_factor96_mechanism_batch_v1/completed_run/inputs/dividend_coverage.json",
         "inputs/dividend_coverage.json")
    copy(ROOT / "config/510300_existing_data_training_mandate_v1.json", "source_evidence/latest_mandate.json")
    for folder, name in [("research", "factor96_t11_date_proxy_account_v1.py"),
                         ("tests", "test_factor96_t11_date_proxy_account_v1.py"),
                         ("scripts", Path(__file__).name)]:
        copy(ROOT / folder / name, "code/"+name)
    save(OUT / "prefreeze_test_receipt.json", {"at": now(), "command": ".venv/Scripts/python.exe -m pytest tests/test_factor96_t11_date_proxy_account_v1.py -q",
        "observed_result": "11 passed in 2.93s", "tests_sha256": digest(ROOT / "tests/test_factor96_t11_date_proxy_account_v1.py"),
        "account_code_sha256": digest(ROOT / "research/factor96_t11_date_proxy_account_v1.py"),
        "real_O02_already_computed": False, "actual_accounts_already_run": 0})
    save(OUT / "source_receipt.json", {"at": now(), "sources": sources, "network_requests": 0,
        "price_admission": "沿用先前固定账户研究已经保存并复核的原始价和分红账本；不把回取时间当历史首次发布时刻。",
        "risk_extension": "同一固定算法按成熟信息延伸，须核对旧缓存全部重叠ES和训练记录。",
        "financial_clock": "名义公告日期代理；完整更正链和历史first_seen未建立。",
        "delivery_package_required": False, "orders_authorized": False})
    files = [{"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)}
             for p in sorted(OUT.rglob("*")) if p.is_file()]
    save(OUT / "implementation_input_freeze.json", {"at": now(), "files": files, "new_O02_computed": False, "new_accounts": 0})
    print("实现、输入与测试已冻结；实际O02及账户尚未运行。", flush=True)


def run():
    assert not (OUT / "run_started.json").exists(), "实际研究只能首次运行，修复必须另立版本"
    verify_manifest(OUT, OUT / "implementation_input_freeze.json")
    save(OUT / "run_started.json", {"at": now(), "mode": "ONE_FIXED_DATE_PROXY_EXPERIMENT", "planned_accounts": 120})
    code = OUT / "code"
    load("research.factor96_library_intake_v1", code / "factor96_library_intake_v1.py")
    core = load("t11_frozen_risk_core", code / "factor96_margin_repair_v1.py")
    engine = load("t11_frozen_engine", code / "account_engine.py")
    model = load("t11_frozen_date_proxy_account", code / "factor96_t11_date_proxy_account_v1.py")
    measurement = load("t11_frozen_o02", code / "factor96_t11_o02_date_proxy_v1.py")
    financial = load("t11_frozen_financial_cohorts", code / "factor96_t11_financial_cohorts_v1.py")
    protocol = read(OUT / "protocol.json")
    market = pd.read_parquet(OUT / "inputs/market.parquet")
    market.date = pd.to_datetime(market.date)
    assert set(market.symbol) == {"510300.SH"}
    assert market.date.is_unique and market.date.is_monotonic_increasing
    assert np.isfinite(market[["open", "high", "low", "close", "dividend"]]).all().all()
    assert market[["open", "high", "low", "close"]].gt(0).all().all()
    assert market.high.ge(market[["open", "close", "low"]].max(axis=1)).all()
    assert market.low.le(market[["open", "close", "high"]].min(axis=1)).all()
    np.testing.assert_allclose(market.previous_close.iloc[1:], market.close.iloc[:-1], rtol=0, atol=1e-12)
    dividends = core.normalize_dividends(pd.read_csv(OUT / "inputs/dividends.csv"))
    coverage = read(OUT / "inputs/dividend_coverage.json")
    assert coverage["complete_history_confirmed"]
    assert coverage["distribution_file_sha256"] == digest(OUT / "inputs/dividends.csv")
    expected_dividends = market.date.map(dividends.groupby("ex_date").cash_dividend_per_share.sum()).fillna(0.)
    np.testing.assert_allclose(market.dividend, expected_dividends, rtol=0, atol=1e-12)
    assert market.date.between(coverage["coverage_start"], coverage["coverage_end"]).all()
    market = market.loc[market.date.le(protocol["periods"]["FULL"][1])].reset_index(drop=True)
    features = core.price_features(market)
    features["es95"], labels, records = core.mature_risk(market, features, dividends)
    old_features = pd.read_parquet(OUT / "inputs/price_features.parquet")
    n = len(old_features)
    assert market.date.iloc[:n].equals(old_features.date)
    fields = ["wealth", "high_w", "low_w", "pressure5", "trend20", "log_rv5_rv60", "es95"]
    np.testing.assert_allclose(features[fields].iloc[:n], old_features[fields], rtol=0, atol=1e-12, equal_nan=True)
    old_records = read(OUT / "inputs/risk_training_records.json")
    assert clean(records[:len(old_records)]) == old_records, "旧风险训练记录重算不一致"
    old_labels = pd.read_parquet(OUT / "inputs/labels.parquet")
    known = old_labels.gross_return5.notna()
    np.testing.assert_allclose(labels.gross_return5.iloc[:n][known], old_labels.gross_return5[known], rtol=0, atol=1e-12)
    for rec in records:
        t = rec["decision_idx"]
        chosen = np.array(rec["selected_indices"])
        assert rec["latest_training_exit_idx"] <= t and (chosen+6 <= t).all()
        assert market.date.iloc[chosen].ge(market.date.iloc[t]-pd.DateOffset(years=2)).all()
        assert rec["training_count"] >= 252 and len(chosen) == 126
    features = features[["date"]+fields]
    features.to_parquet(OUT / "risk_features_extended.parquet", index=False)
    labels.to_parquet(OUT / "risk_labels_extended.parquet", index=False)
    save(OUT / "risk_training_records_extended.json", records)
    save(OUT / "risk_extension_receipt.json", {"at": now(), "overlap_days_reproduced": n,
         "overlap_training_records_reproduced": len(old_records), "extended_market_days": len(market)-n,
         "total_training_records": len(records), "future_label_rows_used": 0, "parameter_changes": 0})
    print(f"风险缓存旧区间{n}日一致，新延伸{len(market)-n}日，未使用未成熟标签。", flush=True)
    daily = pd.read_parquet(OUT / "inputs/daily_financial_cohorts.parquet")
    companies = pd.read_parquet(OUT / "inputs/company_financial_cohorts.parquet")
    reports = pd.read_parquet(OUT / "inputs/repaired_member_report_measurements.parquet")
    measured, refs = measurement.measure_first_responses(daily, pd.DataFrame({"date": market.date,
         "raw_close": market.close, "cash_dividend": market.dividend, "price_source_admitted": True, "cash_dividend_known": True}))
    measured.to_parquet(OUT / "o02_first_response_measurement.parquet", index=False)
    refs.to_parquet(OUT / "o02_reference_dependencies.parquet", index=False)
    frontier = financial.disclosure_frontier(reports)
    signals = model.signal_table(measured, companies, market, features)
    facts, fact_audit = model.fact_events(frontier, companies, market)
    fact_audit.to_parquet(OUT / "all_new_report_events.parquet", index=False)
    serial_signals = [{**r, "anchors": json.dumps(clean(r["anchors"]), ensure_ascii=False)} for r in signals]
    pd.DataFrame(serial_signals).to_parquet(OUT / "signals.parquet", index=False)
    counts = {p: sum(s[p] for s in signals) for p in model.POLICIES}
    save(OUT / "measurement_result.json", {"at": now(), "disclosure_days": len(measured),
        "status_counts": measured.status.value_counts().to_dict(), "signals": counts,
        "reference_dependencies": len(refs), "new_report_events": len(fact_audit),
        "clock": "NOMINAL_DATE_PROXY_NOT_STRICT_T11", "no_independent_validation": True})
    print("首轮反应测量完成，固定信号数："+json.dumps(counts, ensure_ascii=False), flush=True)
    summaries, annual, all_cycles, primary_ledgers = [], [], [], {}
    accounts_dir = OUT / "accounts"
    accounts_dir.mkdir()
    for period, (start, end) in protocol["periods"].items():
        for capital in protocol["capital_cny"]:
            for cost_name in protocol["costs"]:
                for policy in model.POLICIES + model.BENCHMARKS:
                    for lag in ((1, 2) if policy in model.POLICIES else (1,)):
                        identity = f"{period}_{capital}_{cost_name}_{policy}_L{lag}"
                        ledger, decisions, orders, fact_checks = model.simulate(market, features, dividends, signals, facts,
                            policy, lag, capital, cost_name, start, end, engine, core.target_quantity)
                        folder = accounts_dir / identity
                        folder.mkdir()
                        for filename, frame in [("ledger", ledger), ("decisions", decisions), ("orders", orders), ("fact_checks", fact_checks)]:
                            frame.to_parquet(folder / (filename+".parquet"), index=False)
                        values = {"account_id": identity, "period": period, "capital": capital, "cost": cost_name,
                                  "policy": policy, "lag": lag, **core.metrics(ledger, capital)}
                        values["cash_days"] = int(ledger.shares.eq(0).sum())
                        values["new_entries"] = int((ledger.shares_before.eq(0) & ledger.filled_quantity.gt(0)).sum())
                        values["account_stopped"] = bool(ledger.risk_stopped.any())
                        summaries.append(values)
                        start_equity = float(capital)
                        for year, block in ledger.groupby(ledger.date.dt.year):
                            annual.append({"account_id": identity, "year": int(year), **core.metrics(block, start_equity)})
                            start_equity = float(block.equity.iloc[-1])
                        cycles = core.cycle_records(ledger, capital)
                        cycles.insert(0, "account_id", identity)
                        all_cycles.append(cycles)
                        if period == "FULL" and capital == 200000 and cost_name == "STRESS" and lag == 1 and policy in ("FULL", "FINANCIAL_COMMON", "ABOVE_MEDIAN"):
                            primary_ledgers[policy] = ledger
                        if len(summaries) % 10 == 0:
                            print(f"已完成固定账户 {len(summaries)}/120。", flush=True)
    assert len(summaries) == 120
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "account_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(annual).to_csv(OUT / "annual_accounts.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_cycles, ignore_index=True).to_csv(OUT / "cycles.csv", index=False, encoding="utf-8-sig")
    uncertainty = model.paired_uncertainty(primary_ledgers)
    save(OUT / "paired_block_uncertainty.json", uncertainty)
    primary = summary.loc[summary.account_id.eq("FULL_200000_STRESS_FULL_L1")].iloc[0].to_dict()
    point_pass = bool(pd.notna(primary["net_sharpe"]) and primary["net_sharpe"] >= 1.2
                      and primary["cagr"] >= .1 and primary["max_drawdown"] <= .1)
    result = {"at": now(), "study_id": protocol["study_id"], "status": "HISTORICAL_POINT_PASS_NOT_VALIDATED" if point_pass else "FIXED_DATE_PROXY_TARGET_FAILED",
        "primary_account": primary, "primary_historical_point_pass": point_pass, "accounts": len(summary),
        "signals": counts, "original_T11": "NOT_RUN_STRICT_CLOCK_AND_CORRECTION_CHAIN_NOT_ADMITTED",
        "independent_forward_observations": 0, "qualified_candidates": [], "goal_achieved": False,
        "parameter_search": False, "orders_authorized": False, "external_review": "NOT_PERFORMED",
        "delivery_package_required": False, "scope": "固定历史日期代理变体，不是严格first_seen或完整更正链验证。",
        "source_limits": ["公告时刻多数为零点日期代理，历史更正/取消链不完整。", "财务修复范围外旧字段未全部逐页验证。",
                          "23个正向财务日中15日仅一家披露公司，不等于整个沪深300盈利更新。",
                          "50%为成交前目标限制，持仓价格变动可使收盘权重超过50%。", "日线开盘成交为模拟，非真实成交证据。"]}
    save(OUT / "result.json", result)
    manifest = [{"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)}
                for p in sorted(OUT.rglob("*")) if p.is_file()]
    save(OUT / "result_freeze.json", {"at": now(), "files": manifest})
    print(json.dumps(clean({"状态": result["status"], "主账户": primary}), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T11日期代理固定研究，无交付包")
    parser.add_argument("action", choices=["prepare", "run"])
    args = parser.parse_args()
    prepare() if args.action == "prepare" else run()
