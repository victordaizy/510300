"""总社融、政府债券融资构成与资金意外的两年日更固定检验。"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from types import FunctionType, SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.tsf_government_composition_source_v1 as source
import research.funding_repurchase_demand_daily_v1 as primitives
import research.funding_forecast_maturity_account_v1 as maturity
from research.selected_mix_reappraisal_v1 import read, save, now, digest
from research.selected_mix_daily_two_year_v1 import summarize_accounts, paired_interval
from research.strategy_review_diagnostics_v1 import metrics, cycles

OUT = ROOT / "reports/research/510300_tsf_composition_funding_daily_v1"
STUDY = "510300_TSF_COMPOSITION_FUNDING_DAILY_V1"
GOVERNMENT = "government_share_yoy_change_pp"
CREDIT = "credit_acceleration3_pp"
PRICE = primitives.parent.BASE
FUNDING = "PRICE_AND_FUNDING"
AGGREGATE = "PRICE_FUNDING_AND_TOTAL_CREDIT"
COMPOSITION = "PRICE_FUNDING_CREDIT_AND_COMPOSITION"
MODELS = {"HISTORY": [], FUNDING: [*PRICE, "funding_innovation"],
          AGGREGATE: [*PRICE, "funding_innovation", CREDIT],
          COMPOSITION: [*PRICE, "funding_innovation", CREDIT, GOVERNMENT]}
TRADED = [FUNDING, AGGREGATE, COMPOSITION]
VARIANTS = ["ROLL_NO_ADD", "FIXED5_NO_ADD"]
PRIMARY = COMPOSITION + "__FIXED5_NO_ADD"
START, END = primitives.START, primitives.END


def adapt(function, changes):
    """只建立独立的函数环境，不改动任何已冻结模块全局变量。"""
    return FunctionType(function.__code__, {**function.__globals__, **changes}, function.__name__, function.__defaults__, function.__closure__)


FIT = adapt(primitives.fit_day, {"parent": SimpleNamespace(BASE=[*PRICE, CREDIT]),
                               "FEATURE": GOVERNMENT, "MODELS": MODELS, "PRIMARY": COMPOSITION})


def information():
    x = pd.read_parquet(primitives.funding.OUT / "inputs/decision_information.parquet")
    x["date"] = pd.to_datetime(x.date).astype("datetime64[ns]")
    r = pd.read_parquet(source.OUT / "results/released_composition.parquet")
    assert r.status.eq("EXTRACTED_FROM_SAVED_PBC_RELEASE").all()
    original = pd.read_parquet(source.SOURCE).set_index("reference_period")
    differences = []
    for row in r.itertuples():
        old_period = str(pd.Period(row.reference_period, freq="M") - 3)
        earlier = original.loc[old_period]
        assert pd.Timestamp(earlier.available_at) < pd.Timestamp(row.conservative_known_at)
        differences.append(float(row.total_tsf_yoy_percent) - float(earlier.first_release_value))
    r[CREDIT] = differences
    r = r.rename(columns={"reference_period": "credit_reference_period", "conservative_known_at": "credit_known_at"})
    r["credit_known_at"] = pd.to_datetime(r.credit_known_at).dt.tz_convert("Asia/Shanghai").dt.as_unit("ns")
    columns = ["credit_reference_period", "credit_known_at", GOVERNMENT, CREDIT, "total_tsf_yoy_percent", "share_percent"]
    x = pd.merge_asof(x.sort_values("decision_time"), r[columns].sort_values("credit_known_at"),
                      left_on="decision_time", right_on="credit_known_at", direction="backward")
    x["funding_information_known"] = x.common_known
    age = (x.decision_time - x.credit_known_at).dt.total_seconds() / 86400.
    x["credit_source_age_days"] = age
    x["credit_composition_known"] = x[[GOVERNMENT, CREDIT]].notna().all(axis=1) & age.between(0, 31)
    x["common_known"] &= x.credit_composition_known
    assert x.idx.tolist() == list(range(len(x)))
    assert x.loc[x.common_known, "credit_known_at"].le(x.loc[x.common_known, "decision_time"]).all()
    return x


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("社融构成的账户规则已固定，不能覆盖。")
    assert read(source.OUT / "result.json")["status"] == "SAVED_RELEASE_COMPOSITION_READY"
    x = information()
    for folder in ["code", "inputs", "results", "accounts"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY, "primary": PRIMARY,
        "question": "在相同价格状态、资金意外和总社融增速变化下，政府债券占比的同比变化能否改善五日分布与完整账户？",
        "mechanism_hypothesis": "总量信贷扩张可来自不同融资部门；政府融资占比变化可能伴随不同财政支持、实体融资需求和利率环境。仅凭构成不能断言财政资金进入股票，也不预设高占比利多或利空。",
        "government_composition": "使用当期央行原文直接公布的政府债券余额占社融比重同比变化，单位百分点，不代替私人信用或中性利率。",
        "credit_control": "当前已公布总社融存量同比减三个月前当期已公布同比；保留两个发布时间，不用后来修订年表重写过去。",
        "funding": "复用每一决策日严格两年窗内重建的FDR007日历残差，不新增资金回归。",
        "models": MODELS, "training": "所有模型同一共同成熟池，最近两日历年、五日退出开盘早于决策日、至少252原点；固定126近邻，训练内标准化截断5、距离同分按原点顺序。每日重复月值不当独立宏观公布，另存各训练池独立月份数。",
        "clock": "月报保守日末可用，最早下一交易日；从该日末起31自然日过期。原来源发布时点、历史首次实际送达边界保持原记录。",
        "execution": "主方案FIXED5_NO_ADD：空仓入场、五日到期退出、期间只按原风险预算缩量；对照ROLL_NO_ADD：已有库存不加仓，每天按原五日分布选择。两者均复用已有执行引擎，不搜索退出期。",
        "economic_entry": "仅已知五日价格弱势，持仓大小由正期望、估计成本、方差及账户尾部预算决定；股息、T+1、100份、tick0.001、涨跌停与现金限制全保留。",
        "capital": 200000, "assets": ["510300.SH", "CASH_CNY"], "period": [START, END], "annual_days": 242,
        "risk": {"position_max": .5, "ES95_5day_budget": .025, "gap_return": -.1, "gap_budget": .05, "drawdown_reserve_share": .5, "drawdown_stop": .1},
        "costs": primitives.distribution.COSTS, "cash_and_risk_free_rate": 0,
        "targets": {"stress_sharpe": 1.2, "stress_cagr": .1, "max_drawdown": .1},
        "comparison": "主方案对同期限总社融对照和价格资金对照，及同一构成信息ROLL_NO_ADD执行对照；20日循环区块2000次seed2026092605，固定2020-2023和2024-末端、全部滚动两年；不事后选择有利时期。",
        "development_gate": "主压力三目标通过，两个同期限信息对照的完整配对差95%下界均正、两个固定时期增量均正，才列开发候选；仍需独立验证。",
        "monitor": "五日固定相位、60成熟原点q05跌穿率超过15%记警报，诊断不回改账户；月度重复及样本不足另存。",
        "source_coverage": {"common_dates": int(x.common_known.sum()), "first": x.loc[x.common_known, "date"].min(), "last": x.loc[x.common_known, "date"].max()},
        "new_accounts": 12, "new_source_months": 80, "new_funding_regression_fits": 0, "new_parameter_grid": 0,
        "historical_prices_previously_researched": True, "historical_first_delivery_authenticated": False,
        "new_independent_forward_observations": 0, "goal_achieved": False, "orders_authorized": False}, True)
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    paths = [Path(__file__), Path(primitives.__file__), Path(primitives.parent.__file__), Path(primitives.funding.__file__),
             Path(primitives.engine.__file__), Path(primitives.distribution.__file__), Path(maturity.__file__), Path(source.__file__),
             ROOT / "research/selected_mix_daily_two_year_v1.py", ROOT / "research/strategy_review_diagnostics_v1.py",
             source.SOURCE, source.OUT / "protocol.json", source.OUT / "result.json", source.OUT / "results/released_composition.parquet",
             primitives.funding.OUT / "inputs/decision_information.parquet", primitives.funding.OUT / "inputs/funding_features.parquet",
             primitives.funding.OUT / "results/nested_funding_predictions.parquet", primitives.parent.OUT / "inputs/market.parquet",
             primitives.parent.OUT / "inputs/mature_labels.parquet", primitives.parent.DIVIDENDS]
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
                               "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths},
                               "before_new_composition_predictions_and_accounts": True}, True)
    print(f"社融融资构成的增量已固定，共同信息日{x.common_known.sum()}个，准备12个完整账户。", flush=True)


def run():
    frozen = read(OUT / "freeze.json")
    assert digest(OUT / "protocol.json") == frozen["protocol_sha256"]
    for path, sha in frozen["sources"].items():
        assert digest(ROOT / path) == sha, path
    save(OUT / "RUN_STARTED.json", {"at": now()}, True)
    x = information()
    market = pd.read_parquet(primitives.parent.OUT / "inputs/market.parquet")
    labels = pd.read_parquet(primitives.parent.OUT / "inputs/mature_labels.parquet")
    dividends = primitives.engine.normalize_dividends(pd.read_csv(primitives.parent.DIVIDENDS))
    funding_source = pd.read_parquet(primitives.funding.OUT / "inputs/funding_features.parquet", columns=["source_idx"])
    nested = pd.read_parquet(primitives.funding.OUT / "results/nested_funding_predictions.parquet")
    groups = {int(i): rows for i, rows in nested.groupby("decision_idx", sort=False)}
    x.to_parquet(OUT / "inputs/decision_information.parquet", index=False)
    predictions, models, receipts = [], [], []
    for i in np.flatnonzero(x.date.ge(START)):
        p, m, r = FIT(int(i), x, groups.get(int(i)), len(funding_source), labels)
        r["distinct_training_reference_months"] = int(x.loc[m[0]["training_indices"], "credit_reference_period"].nunique()) if m else 0
        predictions.extend(p)
        models.extend(m)
        receipts.append(r)
        if len(receipts) % 300 == 0:
            print(f"社融融资构成条件分布已更新至{x.date.iloc[i].date()}。", flush=True)
    pred, schedule = pd.DataFrame(predictions), pd.DataFrame(receipts)
    pred.to_parquet(OUT / "results/predictions.parquet", index=False)
    schedule.to_parquet(OUT / "results/update_receipts.parquet", index=False)
    save(OUT / "results/saved_distributions.json", models, True)
    verifier = adapt(primitives.verify_predictions, {"fit_day": FIT})
    verification = verifier(pred, models, x, labels, groups, len(funding_source))
    save(OUT / "results/prediction_verification.json", verification, True)
    evaluate = adapt(primitives.forecast_evaluation, {"OUT": OUT, "PRIMARY": COMPOSITION, "MODELS": MODELS})
    forecast, scored, monitor = evaluate(pred, labels, x)
    accounts, checks, cycle_counts = {}, [], []
    for model in TRADED:
        for variant in VARIANTS:
            for cost in primitives.distribution.COSTS:
                policy = model + "__" + variant
                ledger, decisions = maturity.simulate(market, x, dividends, pred, model, variant, cost)
                check = primitives.parent.account_verification(ledger, decisions)
                assert not (ledger.shares_before.gt(0) & ledger.filled_quantity.gt(0)).any()
                if variant == "FIXED5_NO_ADD":
                    exits = decisions[decisions.reason.eq("FIVE_DAY_MATURITY")]
                    assert exits.idx.ge(exits.deadline_idx).all()
                folder = OUT / "accounts" / cost / policy
                folder.mkdir(parents=True, exist_ok=True)
                ledger.to_parquet(folder / "ledger.parquet", index=False)
                decisions.to_parquet(folder / "decisions.parquet", index=False)
                closed, pending = cycles(ledger)
                closed.to_parquet(folder / "cycles.parquet", index=False)
                save(folder / "pending_cycle.json", pending, True)
                accounts[policy, cost] = ledger
                checks.append({"policy": policy, "cost": cost, **check, "no_add_to_open_cycle": True})
                cycle_counts.append({"policy": policy, "cost": cost, "closed": len(closed), "open": pending is not None})
                m = metrics(ledger)
                sh = "未定义" if m["sharpe"] is None else f"{m['sharpe']:.6f}"
                print(f"{policy}/{cost}：夏普{sh}、年化{m['annual_return']:.3%}、回撤{abs(m['max_drawdown']):.3%}。", flush=True)
    measures, rolling = summarize_accounts(OUT, market, accounts)
    comparisons = []
    controls = [(AGGREGATE + "__FIXED5_NO_ADD", "COMPOSITION_INCREMENT"),
                (FUNDING + "__FIXED5_NO_ADD", "TOTAL_AND_COMPOSITION"), (COMPOSITION + "__ROLL_NO_ADD", "EXECUTION")]
    rng = np.random.default_rng(2026092605)
    for cost in primitives.distribution.COSTS:
        left = accounts[PRIMARY, cost]
        for control, purpose in controls:
            right = accounts[control, cost]
            eras = []
            for lo, hi in [(START, "2023-12-31"), ("2024-01-01", END)]:
                mask = left.date.between(lo, hi)
                eras.append({"start": lo, "end": hi, "annual_arithmetic_difference": float((left.loc[mask, "net_return"] - right.loc[mask, "net_return"]).mean() * 242)})
            comparisons.append({"cost": cost, "control": control, "purpose": purpose,
                                **paired_interval(left, right, rng), "fixed_periods": eras})
    main = next(m for m in measures if m["policy"] == PRIMARY and m["cost"] == "STRESS")
    information_pass = all(r["lower_95"] > 0 and all(v["annual_arithmetic_difference"] > 0 for v in r["fixed_periods"])
                           for r in comparisons if r["cost"] == "STRESS" and r["purpose"] != "EXECUTION")
    ready = pred[["idx", "date"]].drop_duplicates()
    updated = schedule[schedule.status.eq("UPDATED")]
    ledger = accounts[PRIMARY, "STRESS"]
    gross = float((ledger.price_pnl + ledger.dividend_recognized).sum())
    expense = float((ledger.commission + ledger.slippage_cost).sum() + ledger.terminal_exit_reserve.iloc[-1])
    np.testing.assert_allclose(gross - expense, ledger.equity.iloc[-1] - 200000., atol=1e-6, rtol=0)
    candidate = main["historical_point_targets_met"] and information_pass
    result = {"at": now(), "study_id": STUDY,
        "status": "DEVELOPMENT_CANDIDATE_REQUIRES_INDEPENDENT_VALIDATION" if candidate else "FROZEN_NO_QUALIFIED_TSF_COMPOSITION_STRATEGY",
        "primary": main, "all_accounts": measures, "comparisons": comparisons, "forecast_evaluation": forecast,
        "information_increment_gate": information_pass, "new_full_accounts": len(accounts), "new_conditional_distributions": len(pred),
        "new_funding_regression_fits": 0, "new_parameter_searches": 0, "new_source_months": 80,
        "unique_prediction_days": len(ready), "first_prediction_date": ready.date.min(), "last_prediction_date": ready.date.max(),
        "mature_prediction_days": int(scored[scored.gross_return5.notna()].idx.nunique()), "schedule_statuses": schedule.status.value_counts().to_dict(),
        "training_rows_min": int(updated.n_train.min()), "training_rows_max": int(updated.n_train.max()),
        "distinct_training_reference_months_min": int(updated.distinct_training_reference_months.min()),
        "distinct_training_reference_months_max": int(updated.distinct_training_reference_months.max()),
        "distribution_checks": verification, "account_checks": checks, "completed_and_pending_cycles": cycle_counts,
        "primary_cash_attribution": {"gross_pnl": gross, "cost_and_reserve": expense, "net_profit": float(ledger.equity.iloc[-1] - 200000.)},
        "primary_rolling_two_year_joint_passes": sum(r["joint_point_pass"] for r in rolling if r["policy"] == PRIMARY and r["cost"] == "STRESS"),
        "mature_monitor": [{"model": name, "mature_nonoverlap_rows": len(part), "assessable_rows": int(part.monitor_assessable.sum()), "alerts": int(part.alert.sum())}
                           for name, part in monitor.groupby("model")],
        "historical_first_vintage_authenticated": False, "new_independent_forward_observations": 0,
        "current_market_view": "NO_VIEW", "goal_status": "active", "goal_achieved": False, "orders_authorized": False,
        "review_package_created": False}
    save(OUT / "result.json", result, True)
    print("政府债券融资构成的12个完整账户和固定增量比较已完成。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="社融总量与融资构成的两年日更账户")
    parser.add_argument("command", choices=["freeze", "run"])
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    else:
        try:
            run()
        except Exception as exc:
            if not (OUT / "RUN_FAILURE.json").exists():
                save(OUT / "RUN_FAILURE.json", {"at": now(), "error": f"{type(exc).__name__}: {exc}"}, True)
            raise
