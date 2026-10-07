"""住户银行存贷行为的单项增量研究：两年日更、完整账户、公布间隔比较。"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.household_bank_balance_source_v1 as source
import research.loan_maturity_composition_daily_v1 as loan
from research.selected_mix_reappraisal_v1 import read, save, now, digest
from research.selected_mix_daily_two_year_v1 import summarize_accounts, paired_interval
from research.strategy_review_diagnostics_v1 import metrics, cycles

shared, prior, maturity = loan.shared, loan.prior, loan.maturity
OUT = ROOT / "reports/research/510300_household_bank_balance_daily_v1"
STUDY = "510300_HOUSEHOLD_BANK_BALANCE_DAILY_V1"
START, END = loan.START, loan.END
FEATURE = source.FEATURE
LOAN_GROWTH, DEPOSIT_GROWTH = "rmb_loan_stock_yoy_percent", "rmb_deposit_stock_yoy_percent"
CALENDAR = loan.CALENDAR
CONTROL = "PRICE_BANK_CALENDAR_AND_LOAN_GROWTH"
TOTAL = "PRICE_BANK_CALENDAR_LOAN_AND_DEPOSIT_GROWTH"
SIGNAL = "PRICE_BANK_CALENDAR_TOTALS_AND_HOUSEHOLD_BANK_BALANCE"
BASE = [*prior.parent.BASE, "funding_innovation", *CALENDAR, LOAN_GROWTH]
MODELS = {"HISTORY": [], CONTROL: BASE, TOTAL: [*BASE, DEPOSIT_GROWTH],
          SIGNAL: [*BASE, DEPOSIT_GROWTH, FEATURE]}
TRADED, VARIANTS = [CONTROL, TOTAL, SIGNAL], shared.VARIANTS
PRIMARY = SIGNAL + "__FIXED5_NO_ADD"
FIT_BASE = shared.adapt(prior.fit_day, {
    "parent": SimpleNamespace(BASE=[*prior.parent.BASE, *CALENDAR, LOAN_GROWTH, DEPOSIT_GROWTH]),
    "FEATURE": FEATURE, "MODELS": MODELS, "PRIMARY": SIGNAL})
fit_day = shared.adapt(loan.fit_day, {"FIT_BASE": FIT_BASE})


def information(releases=None):
    x = pd.read_parquet(prior.funding.OUT / "inputs/decision_information.parquet")
    q = (pd.read_parquet(source.OUT / "released_household_bank_balance.parquet") if releases is None else releases).copy()
    q["conservative_known_at"] = pd.to_datetime(q.conservative_known_at).dt.tz_convert("Asia/Shanghai").dt.as_unit("ns")
    lookup = q.set_index("stat_month").conservative_known_at.to_dict()
    q["earliest_component_release_at"] = pd.to_datetime([
        min(lookup[m] for m in months) for months in q.cumulative_source_months]).tz_convert("Asia/Shanghai").as_unit("ns")
    month = pd.PeriodIndex(q.stat_month, freq="M").month
    q["stat_month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    q["stat_month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    cols = ["stat_month", "conservative_known_at", "earliest_component_release_at", "statistical_regime",
            "ratio_available", FEATURE, LOAN_GROWTH, DEPOSIT_GROWTH, *CALENDAR]
    q = q[cols].rename(columns={"stat_month": "loan_stat_month", "conservative_known_at": "loan_known_at"})
    x = pd.merge_asof(x.sort_values("decision_time"), q.sort_values("loan_known_at"),
                      left_on="decision_time", right_on="loan_known_at", direction="backward")
    x["bank_information_known"] = x.common_known
    x["monthly_information_age_days"] = (x.decision_time - x.loan_known_at).dt.total_seconds() / 86400
    x["common_known"] &= x.ratio_available.fillna(False).astype(bool) & x.monthly_information_age_days.between(0, 31)
    x["common_known"] &= x[[FEATURE, LOAN_GROWTH, DEPOSIT_GROWTH, *CALENDAR]].notna().all(axis=1)
    assert x.idx.tolist() == list(range(len(x)))
    assert x.loc[x.common_known, "loan_known_at"].lt(x.loc[x.common_known, "decision_time"]).all()
    assert np.isfinite(x.loc[x.common_known, [FEATURE, LOAN_GROWTH, DEPOSIT_GROWTH]]).all().all()
    return x


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("住户银行行为实验已固定，不覆盖。")
    x = information()
    q = pd.read_parquet(source.OUT / "released_household_bank_balance.parquet")
    cutoff = pd.Timestamp("2024-06-28", tz="Asia/Shanghai")
    changed = q.copy()
    later = pd.to_datetime(changed.conservative_known_at).ge(cutoff)
    changed.loc[later, [FEATURE, LOAN_GROWTH, DEPOSIT_GROWTH]] += 700.
    altered = information(changed)
    cols = [FEATURE, LOAN_GROWTH, DEPOSIT_GROWTH, *CALENDAR, "common_known"]
    pd.testing.assert_frame_equal(x.loc[x.decision_time.lt(cutoff), cols],
                                  altered.loc[altered.decision_time.lt(cutoff), cols], check_exact=True)
    for folder in ["inputs", "code", "results", "accounts"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY, "primary": PRIMARY,
        "question": "在同样价格、资金、贷款和存款总量增长下，住户银行存贷净增加差额是否改善五日修复预测与账户？",
        "new_evidence": read(source.OUT / "result.json"),
        "feature": "100*(住户存款年初累计净增-住户贷款年初累计净增)/当期人民币存款余额。只增加这一项住户行为指标，不从企业、财政或非银部门挑选表现好的指标。",
        "mechanism": "住户在银行端积累存款同时少借款或偿还贷款，可能对应风险承受能力或资金配置变化；其股票收益方向未知，并受资产交易、存款利息等其他机制影响。不能称为净储蓄或股市买盘。",
        "duplicate_boundary": "旧贷款期限、M1/M2、社融政府债、资金政策差与IF基差结果保留。新变量来自住户存款与贷款总额的同统计区间差额，不复用旧失败规则。既有历史仍受大量选择影响，不把该用途称独立验证。",
        "controls": "三组同日期同训练池。CONTROL已包含贷款余额同比，TOTAL加存款余额同比，SIGNAL再加住户差额。所有模型共有统计月份sin/cos，避免累计投放季节性充当信号。",
        "models": MODELS, "traded_models": TRADED, "variants": VARIANTS,
        "training": "每日最近两个日历年的成熟五日原点；当前与训练原点的金融统计范围一致，所有累计依赖月报也不早于当前两年下界；至少252原点，固定126近邻，训练均值/样本标准差标准化后截断5。",
        "known_at": "月报公布日结束之后最近的A股09:00，最长31自然日。2023年新统计范围重新积累样本。2022-04与2022-05住户总贷款已知，不继承旧期限细分的缺口。",
        "funding": "复用每个当前两年窗嵌套重建的FDR007日历残差，无新增资金参数或回归。",
        "account": "3信息x2期限x2成本共12新完整账户。只有此前五日弱势才新增，持仓期间不加仓；主五个后续开盘到期，每日尾部约束可提前减仓。",
        "period": [START, END], "capital": 200000, "assets": ["510300.SH", "CASH_CNY"],
        "annual_days": 242, "cash_and_risk_free": 0, "costs": prior.distribution.COSTS,
        "risk_contract": read(ROOT / "config/510300_existing_data_training_mandate_v1.json")["account_tail_risk_contract"],
        "targets": {"stress_sharpe": 1.2, "stress_cagr": .1, "max_drawdown": .1},
        "comparisons": "主FIXED5对两个信息对照及同信息ROLL；20日循环区块2000次以及完整月报公布间隔区块2000次，seed2026092609。固定2020-2023、2024-末端、年度、滚动两年全部保留。",
        "development_gate": "主压力三项目标通过，住户指标对两个信息对照的两种95%区块下界均正，两个固定时期增量均正；依然需要独立前向和首版认证。",
        "monitor": "固定每5日相位，使用已成熟的q05结果；最近60条突破率>15%记警报，零警报不证明正期望。",
        "future_source_perturbation_pass": True, "common_information_days": int(x.common_known.sum()),
        "new_full_accounts": 12, "new_parameter_searches": 0, "new_funding_regression_fits": 0,
        "new_independent_forward_observations": 0, "goal_achieved": False, "current_market_view": "NO_VIEW",
        "orders_authorized": False, "review_package_created": False}, True)
    paths = [Path(__file__), Path(source.__file__), Path(loan.__file__), Path(shared.__file__), Path(prior.__file__),
             Path(prior.parent.__file__), Path(prior.distribution.__file__), Path(prior.engine.__file__), Path(maturity.__file__),
             ROOT / "research/selected_mix_daily_two_year_v1.py", ROOT / "research/strategy_review_diagnostics_v1.py",
             source.OUT / "result.json", source.OUT / "released_household_bank_balance.parquet",
             source.OUT / "extracted_originals.json", prior.parent.DIVIDENDS,
             prior.parent.OUT / "inputs/market.parquet", prior.parent.OUT / "inputs/mature_labels.parquet",
             prior.funding.OUT / "inputs/decision_information.parquet", prior.funding.OUT / "inputs/funding_features.parquet",
             prior.funding.OUT / "results/nested_funding_predictions.parquet"]
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
        "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}, "before_new_strategy_returns": True}, True)
    save(OUT / "inputs/mandate.json", read(ROOT / "config/510300_existing_data_training_mandate_v1.json"), True)
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    print(f"住户银行行为12个账户已固定，共同信息日{x.common_known.sum()}。", flush=True)


def run():
    frozen = read(OUT / "freeze.json")
    assert digest(OUT / "protocol.json") == frozen["protocol_sha256"]
    for path, sha in frozen["sources"].items():
        assert digest(ROOT / path) == sha, path
    save(OUT / "RUN_STARTED.json", {"at": now()}, True)
    x = information()
    x.to_parquet(OUT / "inputs/decision_information.parquet", index=False)
    market = pd.read_parquet(prior.parent.OUT / "inputs/market.parquet")
    labels = pd.read_parquet(prior.parent.OUT / "inputs/mature_labels.parquet")
    dividends = prior.engine.normalize_dividends(pd.read_csv(prior.parent.DIVIDENDS))
    count = len(pd.read_parquet(prior.funding.OUT / "inputs/funding_features.parquet", columns=["source_idx"]))
    nested = pd.read_parquet(prior.funding.OUT / "results/nested_funding_predictions.parquet")
    groups = {int(i): part for i, part in nested.groupby("decision_idx", sort=False)}
    predictions, models, receipts = [], [], []
    for i in np.flatnonzero(x.date.ge(START)):
        p, m, r = fit_day(int(i), x, groups.get(int(i)), count, labels)
        predictions.extend(p)
        models.extend(m)
        receipts.append(r)
        if len(receipts) % 300 == 0:
            print(f"住户银行行为分布已计算至{x.date.iloc[i].date()}。", flush=True)
    pred, schedule = pd.DataFrame(predictions), pd.DataFrame(receipts)
    pred.to_parquet(OUT / "results/predictions.parquet", index=False)
    schedule.to_parquet(OUT / "results/update_receipts.parquet", index=False)
    save(OUT / "results/saved_distributions.json", models, True)
    verification = shared.adapt(prior.verify_predictions, {"fit_day": fit_day})(pred, models, x, labels, groups, count)
    for m in models:
        ids, i = m["training_indices"], m["idx"]
        assert x.loc[ids, "statistical_regime"].eq(x.statistical_regime.iloc[i]).all()
        assert x.loc[ids, "earliest_component_release_at"].ge((x.date.iloc[i] - pd.DateOffset(years=2)).tz_localize("Asia/Shanghai")).all()
    verification.update(statistical_regime_never_pooled=True, cumulative_release_dependencies_inside_two_years=True)
    save(OUT / "results/prediction_verification.json", verification, True)
    evaluation, scored, monitor = shared.adapt(prior.forecast_evaluation, {"OUT": OUT, "PRIMARY": SIGNAL, "MODELS": MODELS})(pred, labels, x)
    accounts, checks, cycle_counts = {}, [], []
    for model in TRADED:
        for variant in VARIANTS:
            policy = model + "__" + variant
            for cost in prior.distribution.COSTS:
                ledger, decisions = maturity.simulate(market, x, dividends, pred, model, variant, cost)
                check = prior.parent.account_verification(ledger, decisions)
                assert not (ledger.shares.shift(1, fill_value=0).gt(0) & ledger.filled_quantity.gt(0)).any()
                folder = OUT / "accounts" / cost / policy
                folder.mkdir(parents=True, exist_ok=True)
                ledger.to_parquet(folder / "ledger.parquet", index=False)
                decisions.to_parquet(folder / "decisions.parquet", index=False)
                finished, pending = cycles(ledger)
                finished.to_parquet(folder / "cycles.parquet", index=False)
                save(folder / "pending_cycle.json", pending, True)
                accounts[policy, cost] = ledger
                checks.append({"policy": policy, "cost": cost, **check})
                cycle_counts.append({"policy": policy, "cost": cost, "closed": len(finished), "open": pending is not None})
                m = metrics(ledger)
                print(f"{policy}/{cost}：夏普{m['sharpe']:.6f}，年化{m['annual_return']:.3%}，回撤{abs(m['max_drawdown']):.3%}。", flush=True)
    measures, rolling = summarize_accounts(OUT, market, accounts)
    comparisons, rng = [], np.random.default_rng(2026092609)
    for cost in prior.distribution.COSTS:
        left = accounts[PRIMARY, cost]
        for name, purpose in [(CONTROL, "DEPOSITS_AND_HOUSEHOLD_INCREMENT"), (TOTAL, "HOUSEHOLD_INCREMENT")]:
            right = accounts[name + "__FIXED5_NO_ADD", cost]
            periods = []
            for lo, hi in [(START, "2023-12-31"), ("2024-01-01", END)]:
                mask = left.date.between(lo, hi)
                periods.append({"start": lo, "end": hi, "annual_arithmetic_difference": float((left.loc[mask, "net_return"] - right.loc[mask, "net_return"]).mean() * 242)})
            comparisons.append({"cost": cost, "control": name, "purpose": purpose,
                "daily_blocks": paired_interval(left, right, rng),
                "release_blocks": loan.release_block_interval(left, right, x, rng), "fixed_periods": periods})
        comparisons.append({"cost": cost, "control": SIGNAL + "__ROLL_NO_ADD", "purpose": "EXECUTION",
            "daily_blocks": paired_interval(left, accounts[SIGNAL + "__ROLL_NO_ADD", cost], rng)})
    main = next(m for m in measures if m["policy"] == PRIMARY and m["cost"] == "STRESS")
    increment = all(r["daily_blocks"]["lower_95"] > 0 and r["release_blocks"]["lower_95"] > 0
        and all(p["annual_arithmetic_difference"] > 0 for p in r["fixed_periods"])
        for r in comparisons if r["cost"] == "STRESS" and r["purpose"] != "EXECUTION")
    ledger = accounts[PRIMARY, "STRESS"]
    gross = float((ledger.price_pnl + ledger.dividend_recognized).sum())
    expense = float((ledger.commission + ledger.slippage_cost).sum() + ledger.terminal_exit_reserve.iloc[-1])
    np.testing.assert_allclose(gross - expense, ledger.equity.iloc[-1] - 200000, atol=1e-6, rtol=0)
    ready, updated = pred[["idx", "date"]].drop_duplicates(), schedule[schedule.status.eq("UPDATED")]
    save(OUT / "result.json", {"at": now(), "study_id": STUDY,
        "status": "DEVELOPMENT_CANDIDATE_REQUIRES_INDEPENDENT_VALIDATION" if main["historical_point_targets_met"] and increment else "FROZEN_NO_QUALIFIED_HOUSEHOLD_BANK_BALANCE_STRATEGY",
        "primary": main, "all_accounts": measures, "comparisons": comparisons,
        "new_full_accounts": len(accounts), "joint_target_pass_accounts": sum(bool(m["historical_point_targets_met"]) for m in measures),
        "new_conditional_distributions": len(pred), "unique_prediction_days": len(ready),
        "first_prediction_date": ready.date.min(), "last_prediction_date": ready.date.max(),
        "training_rows_min": int(updated.n_train.min()), "training_rows_max": int(updated.n_train.max()),
        "distinct_training_release_months_min": int(updated.distinct_training_release_months.min()),
        "distinct_training_release_months_max": int(updated.distinct_training_release_months.max()),
        "schedule_statuses": schedule.status.value_counts().to_dict(),
        "forecast_evaluation": evaluation, "development_increment_gate": increment,
        "primary_cash_attribution": {"gross_pnl": gross, "cost_and_reserve": expense, "net_profit": float(ledger.equity.iloc[-1] - 200000)},
        "primary_rolling_two_year_joint_passes": sum(r["joint_point_pass"] for r in rolling if r["policy"] == PRIMARY and r["cost"] == "STRESS"),
        "mature_monitor": [{"model": name, "mature_nonoverlap_rows": len(part), "assessable_rows": int(part.monitor_assessable.sum()), "alerts": int(part.alert.sum())} for name, part in monitor.groupby("model")],
        "prediction_checks": verification, "account_checks": checks, "cycle_counts": cycle_counts,
        "new_parameter_searches": 0, "new_funding_regression_fits": 0, "new_independent_forward_observations": 0,
        "source_months": 104, "joint_ratio_months": 104, "historical_first_vintage_verified": False,
        "current_market_view": "NO_VIEW", "goal_status": "active", "goal_achieved": False,
        "orders_authorized": False, "review_package_created": False}, True)
    print("住户银行行为12个完整账户及公布间隔增量比较已完成。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="住户存贷差额的每日两年滚动研究")
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
