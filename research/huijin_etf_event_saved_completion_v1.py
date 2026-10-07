"""完成已保存四账户的复算；仅统一日期存储精度，不改策略或重新搜索。"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.selected_mix_reappraisal_v1 import read, save, now, digest, local_import_closure
from research.huijin_etf_event_probe_v1 import OUT as PARENT, SOURCE, PRICE, OLD_TAIL, prepare, verification
from research.huijin_etf_event_core_v1 import public_states, simulate, PRIMARY, CONTROL
from research.selected_mix_daily_two_year_v1 import paired_interval, summarize_accounts
from research.strategy_review_diagnostics_v1 import metrics

OUT = ROOT / "reports/research/510300_huijin_etf_event_saved_completion_v1"


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("保存结果完成步骤已固定")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "code").mkdir()
    (OUT / "results").mkdir()
    save(OUT / "protocol.json", {"at": now(), "study_id": "510300_HUIJIN_ETF_EVENT_SAVED_COMPLETION_V1",
        "parent": PARENT.relative_to(ROOT).as_posix(),
        "failure": read(PARENT / "RUN_FAILURE.json"),
        "correction": "比较数据帧前，把所有datetime列统一到纳秒精度；不改日期值、账户账本、决策、价格、费用、窗口或策略。",
        "evaluation": "复用四个完整账户，完成原协议中的两档配对比较、逐年及滚动两年，并重放一次截断前缀。",
        "new_full_accounts": 0, "saved_full_accounts": 4, "new_prefix_verification_replays": 1,
        "new_return_model_fits": 0, "new_parameter_searches": 0, "orders_authorized": False, "goal_achieved": False}, True)
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    sources = local_import_closure({Path(__file__)})
    sources.update(ROOT / name for name in read(PARENT / "freeze.json")["sources"])
    sources.update([PARENT / "protocol.json", PARENT / "implementation_clarifications.json", PARENT / "freeze.json",
                    PARENT / "RUN_FAILURE.json", PARENT / "daily_two_year_tail_forecasts.json", PARENT / "episodes.json"])
    for cost in ["BASE", "STRESS"]:
        for name in [CONTROL, PRIMARY]:
            sources.update((PARENT / "accounts" / cost / name).glob("*"))
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
         "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(sources) if p.is_file()}}, True)
    print("保存结果完成步骤已固定：只处理datetime存储精度，四个完整账户保持原样。", flush=True)


def dates_to_ns(frame):
    result = frame.copy()
    for name in result.columns:
        if pd.api.types.is_datetime64_any_dtype(result[name]):
            result[name] = result[name].dt.as_unit("ns")
    return result


def run():
    frozen = read(OUT / "freeze.json")
    assert digest(OUT / "protocol.json") == frozen["protocol_sha256"]
    for name, sha in frozen["sources"].items():
        assert digest(ROOT / name) == sha, "保存来源变化：" + name
    save(OUT / "RUN_STARTED.json", {"at": now()}, True)
    data, dividends, config, facts, states, episodes, receipts = prepare()
    tails = read(PARENT / "daily_two_year_tail_forecasts.json")
    accounts, checks, snapshots = {}, [], {}
    for name in [CONTROL, PRIMARY]:
        for cost in ["BASE", "STRESS"]:
            folder = PARENT / "accounts" / cost / name
            ledger, decisions = [pd.read_parquet(folder / filename) for filename in ["ledger.parquet", "decisions.parquet"]]
            accounts[name, cost] = ledger
            checks.append({"policy": name, "cost": cost, **verification(ledger, decisions)})
            snapshots[name, cost] = read(folder / "checkpoint.json")
    cutoff = int(data.date.searchsorted("2024-12-31", side="right")) - 1
    prefix_data = data.iloc[:cutoff+1].copy()
    prefix_states, prefix_episodes, _ = public_states(prefix_data, facts)
    prefix_tails = [row for row in tails if row["origin_index"] <= cutoff]
    prefix, prefix_decisions, _ = simulate(prefix_data, dividends, config, prefix_states, prefix_episodes,
                                         prefix_tails, PRIMARY, "STRESS", next_execution_date=data.date.iloc[cutoff+1])
    original = accounts[PRIMARY, "STRESS"]
    pd.testing.assert_frame_equal(dates_to_ns(prefix), dates_to_ns(original[original.date.le(data.date.iloc[cutoff])].reset_index(drop=True)), check_exact=True)
    original_decisions = pd.read_parquet(PARENT / "accounts/STRESS" / PRIMARY / "decisions.parquet")
    pd.testing.assert_frame_equal(dates_to_ns(prefix_decisions),
        dates_to_ns(original_decisions[original_decisions.origin_index.le(cutoff)].reset_index(drop=True)), check_exact=True)
    measurements, windows = summarize_accounts(OUT, data, accounts)
    for row in measurements:
        name, cost = row["policy"], row["cost"]
        closed = pd.read_parquet(PARENT / "accounts" / cost / name / "cycles.parquet")
        row.update(completed_cycles=len(closed), stopped=snapshots[name, cost]["risk_governor"]["stopped"])
    rng = np.random.default_rng(20260925)
    comparisons = [{"cost": cost, "left": PRIMARY, "right": CONTROL,
                    **paired_interval(accounts[PRIMARY, cost], accounts[CONTROL, cost], rng)} for cost in ["BASE", "STRESS"]]
    primary = next(row for row in measurements if row["policy"] == PRIMARY and row["cost"] == "STRESS")
    old_origins = {row["origin_index"] for row in read(OLD_TAIL) if row["policy"] == "BASELINE"}
    result = {"at": now(), "study_id": "510300_HUIJIN_ETF_EVENT_SAVED_COMPLETION_V1",
              "original_study": "510300_HUIJIN_ETF_EVENT_PROBE_V1",
              "status": "HISTORICAL_POINT_PASS_DEVELOPMENT_ONLY" if primary["historical_point_targets_met"] else "FROZEN_NO_QUALIFIED_HUIJIN_DISCLOSURE_STRATEGY",
              "primary": primary, "all_accounts": measurements, "comparisons": comparisons,
              "source_documents": 114, "etf_disclosures": 8, "account_period_buy_episodes": sum(row["review_date"].year >= 2015 for row in episodes),
              "source_first_vintage_verified": False, "daily_risk_forecasts": len(tails),
              "saved_risk_forecasts_reproduced": sum(row["origin_index"] in old_origins for row in tails),
              "full_accounts_computed_in_parent": 4, "new_full_accounts_in_completion": 0,
              "parent_prefix_replay_before_dtype_failure": 1, "completion_prefix_replay": 1,
              "prefix_replay_days": len(prefix), "prefix_account_and_decisions_exact_after_datetime_unit_normalization": True,
              "date_values_changed": False, "checks": checks, "new_return_model_fits": 0,
              "new_parameter_searches": 0, "new_independent_forward_observations": 0,
              "primary_rolling_two_year_joint_passes": sum(row["joint_point_pass"] for row in windows if row["policy"] == PRIMARY and row["cost"] == "STRESS"),
              "goal_achieved": False, "orders_authorized": False}
    save(OUT / "result.json", result, True)
    save(PARENT / "completion_reference.json", {"at": now(), "original_failure_preserved": True,
         "saved_result_completion": (OUT / "result.json").relative_to(ROOT).as_posix(), "status": result["status"]}, True)
    for row in measurements:
        print(f"{row['policy']} {row['cost']}：夏普{row['sharpe']:.6f}、年化{row['annual_return']:.4%}、期末{row['end_equity']:.2f}元。", flush=True)
    print("原四账户及全部决定前缀复算完成，日期值和策略没有改变。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="汇金披露四账户的保存结果完成。")
    parser.add_argument("command", choices=["freeze", "run"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.command]()
