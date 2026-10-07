"""财务口径收款期限的单项增量，逐日只训练最近两年。"""
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

import research.industrial_receivable_legacy_completion_v1 as source
import research.loan_maturity_composition_daily_v1 as loan
from research.selected_mix_reappraisal_v1 import read, save, now, digest
from research.selected_mix_daily_two_year_v1 import summarize_accounts, paired_interval
from research.strategy_review_diagnostics_v1 import metrics, cycles

shared, prior, maturity = loan.shared, loan.prior, loan.maturity
OUT = ROOT / "reports/research/510300_industrial_collection_delay_daily_v1"
STUDY = "510300_INDUSTRIAL_COLLECTION_DELAY_DAILY_V1"
START, END = loan.START, loan.END
FEATURE = "receivable_days_published_yoy_change"
CALENDAR = ["industrial_month_sin", "industrial_month_cos"]
CONTROL = "PRICE_FUNDING_AND_INDUSTRIAL_MONTH"
SIGNAL = "PRICE_FUNDING_AND_RECEIVABLE_DELAY"
BASE = [*prior.parent.BASE, "funding_innovation", *CALENDAR]
MODELS = {"HISTORY": [], CONTROL: BASE, SIGNAL: [*BASE, FEATURE]}
TRADED, VARIANTS = [CONTROL, SIGNAL], shared.VARIANTS
PRIMARY = SIGNAL + "__FIXED5_NO_ADD"
FIT_BASE = shared.adapt(prior.fit_day, {"parent": SimpleNamespace(BASE=[*prior.parent.BASE, *CALENDAR]),
                                      "FEATURE": FEATURE, "MODELS": MODELS, "PRIMARY": SIGNAL})


def information(releases=None):
    x = pd.read_parquet(prior.funding.OUT / "inputs/decision_information.parquet")
    q = (pd.read_parquet(source.PANEL) if releases is None else releases).copy()
    q["known_at"] = pd.to_datetime(q.known_at).dt.tz_convert("Asia/Shanghai").dt.as_unit("ns")
    month = pd.PeriodIndex(q.stat_month, freq="M").month
    q["industrial_month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    q["industrial_month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    q = q[["known_at", "stat_month", "method_group", FEATURE, *CALENDAR]].rename(columns={"known_at": "industrial_known_at", "stat_month": "industrial_stat_month"})
    assert not q.industrial_known_at.duplicated().any()
    x = pd.merge_asof(x.sort_values("decision_time"), q.sort_values("industrial_known_at"), left_on="decision_time", right_on="industrial_known_at", direction="backward")
    x["industrial_age_days"] = (x.decision_time - x.industrial_known_at).dt.total_seconds() / 86400
    x["bank_information_known"] = x.common_known
    x["common_known"] &= x.industrial_age_days.between(0, 70) & x[[FEATURE, *CALENDAR]].notna().all(axis=1)
    assert x.loc[x.common_known, "industrial_known_at"].lt(x.loc[x.common_known, "decision_time"]).all()
    assert x.idx.tolist() == list(range(len(x)))
    return x


def fit_day(i, x, nested, count, labels):
    lower = x.date.iloc[i] - pd.DateOffset(years=2)
    pool = labels[labels.date.ge(lower) & labels.exit_idx.lt(i)]
    ids = pool.idx.to_numpy(int)
    inside = x.industrial_known_at.iloc[ids].ge(lower.tz_localize("Asia/Shanghai")).to_numpy(bool)
    same_method = x.method_group.iloc[ids].eq(x.method_group.iloc[i]).to_numpy(bool)
    rows, models, receipt = FIT_BASE(i, x, nested, count, pool.loc[inside & same_method])
    receipt["distinct_training_release_months"] = int(x.loc[models[0]["training_indices"], "industrial_stat_month"].nunique()) if models else 0
    return rows, models, receipt


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("工业收款期限试验已固定。")
    sr = read(source.OUT / "result.json")
    assert sr["status"] == "COMPLETE_SAME_DEFINITION_MONTHLY_RECEIVABLE_SOURCES_READY" and sr["admitted_months"] == 72
    q, x = pd.read_parquet(source.PANEL), information()
    cutoff = pd.Timestamp("2024-06-28", tz="Asia/Shanghai")
    changed = q.copy()
    changed.loc[changed.known_at.ge(cutoff), FEATURE] += 700.
    after = information(changed)
    cols = [FEATURE, *CALENDAR, "common_known"]
    pd.testing.assert_frame_equal(x.loc[x.decision_time.lt(cutoff), cols], after.loc[after.decision_time.lt(cutoff), cols], check_exact=True)
    for folder in ["inputs", "code", "results", "accounts"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY, "primary": PRIMARY,
        "question": "企业财务汇总的收款期限同比变化，能否帮助区分同样价格压力与银行资金条件下的后续修复？",
        "source": sr, "feature": "仅用当期原报告直接公布的应收账款平均回收期同比增减天数，不用两份不同年份原值相减代替可比同比。",
        "mechanism": "营业活动占用的收款资金与问卷信心不同，回收期拉长可能体现更持久的经营现金压力。该指标同时受收入分母、债权余额和样本构成影响，不能单独识别违约或流动性冲击；对股票方向不预设。",
        "universe": "全国规模以上工业企业财务汇总，不是全部宏观部门，也不是沪深300成分股财务的加权合计。",
        "method": "只纳入2020年后的应收账款/营业收入口径；不拼2018主营业务收入或2019含应收票据口径。调查单位及比较基数调整仍可能影响同比，不把同名公式当恒定样本。",
        "information_clock": "54个月的原页与目录公布日核对，18份旧档案只能据原页时间，统一保守当日日末可用。公布年龄上限70自然日，以容纳1月免报的年度间隔；此上限事前固定，不按收益搜索。",
        "training": "每天最近两日历年、五日退出开盘严格早于当前日的共同原点；月报公布时间也在两年内、口径一致，至少252行、固定126近邻、训练内均值标准差并截断5。",
        "calendar_control": "双方都包括所用统计月份的sin/cos，减少年初累计口径季节差异；唯一新增财务变量是原文公布回收期同比变化。",
        "funding": "只读复用每个当前两年窗内嵌套重建的FDR007日历残差，不新增资金回归。",
        "models": MODELS, "traded_models": TRADED, "variants": VARIANTS,
        "account": "2信息模型、2期限、2费用共8个新完整账户。因资料覆盖与旧模型不同，同池基准重新计算，不直接借用旧账户。20万元、只510300与现金，弱势时允许新增、持仓不加仓；主固定第五个后续开盘到期。",
        "period": [START, END], "annual_days": 242, "cash_and_risk_free": 0,
        "costs": prior.distribution.COSTS, "risk_contract": read(ROOT / "config/510300_existing_data_training_mandate_v1.json")["account_tail_risk_contract"],
        "targets": {"stress_sharpe": 1.2, "stress_cagr": .1, "max_drawdown": .1},
        "comparisons": "主固定五日相对同期限价格资金对照与同信息滚动期限；20日循环及完整月报公布间隔区块各2000次，seed2026092612，固定两时期及全部年度和滚动两年保留。",
        "development_gate": "主压力三目标同时通过，信息增量在两类区块的95%下界及两个固定时期均正；仍需独立前向验证。",
        "monitor": "固定每5个原点的成熟q05结果，60条窗口跌穿率超过15%记录警报；没有警报不证明优势。",
        "common_information_days": int(x.common_known.sum()), "future_source_perturbation_pass": True,
        "new_accounts": 8, "new_parameter_searches": 0, "new_funding_regression_fits": 0,
        "historical_first_vintage_verified": False, "new_independent_forward_observations": 0,
        "goal_achieved": False, "current_market_view": "NO_VIEW", "orders_authorized": False}, True)
    paths = [Path(__file__), Path(source.__file__), source.PANEL, source.OUT / "result.json", source.OUT / "released_records.json",
             Path(loan.__file__), Path(shared.__file__), Path(prior.__file__), Path(prior.parent.__file__),
             Path(prior.distribution.__file__), Path(prior.engine.__file__), Path(maturity.__file__),
             ROOT / "research/selected_mix_daily_two_year_v1.py", ROOT / "research/strategy_review_diagnostics_v1.py",
             prior.parent.DIVIDENDS, prior.parent.OUT / "inputs/market.parquet", prior.parent.OUT / "inputs/mature_labels.parquet",
             prior.funding.OUT / "inputs/decision_information.parquet", prior.funding.OUT / "inputs/funding_features.parquet",
             prior.funding.OUT / "results/nested_funding_predictions.parquet"]
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
        "sources": {path.relative_to(ROOT).as_posix(): digest(path) for path in paths}, "before_new_strategy_returns": True}, True)
    save(OUT / "inputs/mandate.json", read(ROOT / "config/510300_existing_data_training_mandate_v1.json"), True)
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    print(f"财务收款期限的8个完整账户已固定，共同信息日{x.common_known.sum()}。", flush=True)


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
            print(f"财务收款期限分布已计算至{x.date.iloc[i].date()}。", flush=True)
    pred, schedule = pd.DataFrame(predictions), pd.DataFrame(receipts)
    pred.to_parquet(OUT / "results/predictions.parquet", index=False)
    schedule.to_parquet(OUT / "results/update_receipts.parquet", index=False)
    save(OUT / "results/saved_distributions.json", models, True)
    verification = shared.adapt(prior.verify_predictions, {"fit_day": fit_day})(pred, models, x, labels, groups, count)
    for model in models:
        lower = (x.date.iloc[model["idx"]] - pd.DateOffset(years=2)).tz_localize("Asia/Shanghai")
        assert x.loc[model["training_indices"], "industrial_known_at"].ge(lower).all()
    verification["source_clock_inside_two_years"] = True
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
                print(f"{policy}/{cost}：夏普{m['sharpe']:.6f}、年化{m['annual_return']:.3%}、回撤{abs(m['max_drawdown']):.3%}。", flush=True)
    measures, rolling = summarize_accounts(OUT, market, accounts)
    comparisons, rng = [], np.random.default_rng(2026092612)
    for cost in prior.distribution.COSTS:
        left, right = accounts[PRIMARY, cost], accounts[CONTROL + "__FIXED5_NO_ADD", cost]
        periods = []
        for lo, hi in [(START, "2023-12-31"), ("2024-01-01", END)]:
            mask = left.date.between(lo, hi)
            periods.append({"start": lo, "end": hi, "annual_arithmetic_difference": float((left.loc[mask, "net_return"] - right.loc[mask, "net_return"]).mean() * 242)})
        comparisons.append({"cost": cost, "control": CONTROL, "purpose": "RECEIVABLE_DELAY_INCREMENT",
            "daily_blocks": paired_interval(left, right, rng),
            "release_blocks": loan.release_block_interval(left, right, x.assign(loan_stat_month=x.industrial_stat_month), rng), "fixed_periods": periods})
        comparisons.append({"cost": cost, "control": SIGNAL + "__ROLL_NO_ADD", "purpose": "EXECUTION",
            "daily_blocks": paired_interval(left, accounts[SIGNAL + "__ROLL_NO_ADD", cost], rng)})
    main = next(m for m in measures if m["policy"] == PRIMARY and m["cost"] == "STRESS")
    increment = next(r for r in comparisons if r["cost"] == "STRESS" and r["purpose"] == "RECEIVABLE_DELAY_INCREMENT")
    gate = increment["daily_blocks"]["lower_95"] > 0 and increment["release_blocks"]["lower_95"] > 0 and all(p["annual_arithmetic_difference"] > 0 for p in increment["fixed_periods"])
    ledger = accounts[PRIMARY, "STRESS"]
    gross = float((ledger.price_pnl + ledger.dividend_recognized).sum())
    expense = float((ledger.commission + ledger.slippage_cost).sum() + ledger.terminal_exit_reserve.iloc[-1])
    np.testing.assert_allclose(gross - expense, ledger.equity.iloc[-1] - 200000, atol=1e-6, rtol=0)
    ready, updated = pred[["idx", "date"]].drop_duplicates(), schedule[schedule.status.eq("UPDATED")]
    save(OUT / "result.json", {"at": now(), "study_id": STUDY,
        "status": "DEVELOPMENT_CANDIDATE_REQUIRES_INDEPENDENT_VALIDATION" if main["historical_point_targets_met"] and gate else "FROZEN_NO_QUALIFIED_INDUSTRIAL_COLLECTION_STRATEGY",
        "primary": main, "all_accounts": measures, "comparisons": comparisons, "new_full_accounts": len(accounts),
        "joint_target_pass_accounts": sum(bool(m["historical_point_targets_met"]) for m in measures),
        "new_conditional_distributions": len(pred), "unique_prediction_days": len(ready),
        "first_prediction_date": ready.date.min(), "last_prediction_date": ready.date.max(),
        "training_rows_min": int(updated.n_train.min()), "training_rows_max": int(updated.n_train.max()),
        "distinct_training_release_months_min": int(updated.distinct_training_release_months.min()),
        "distinct_training_release_months_max": int(updated.distinct_training_release_months.max()),
        "schedule_statuses": schedule.status.value_counts().to_dict(), "forecast_evaluation": evaluation,
        "development_increment_gate": gate,
        "primary_cash_attribution": {"gross_pnl": gross, "cost_and_reserve": expense, "net_profit": float(ledger.equity.iloc[-1] - 200000)},
        "primary_rolling_two_year_joint_passes": sum(r["joint_point_pass"] for r in rolling if r["policy"] == PRIMARY and r["cost"] == "STRESS"),
        "mature_monitor": [{"model": name, "mature_nonoverlap_rows": len(part), "assessable_rows": int(part.monitor_assessable.sum()), "alerts": int(part.alert.sum())} for name, part in monitor.groupby("model")],
        "prediction_checks": verification, "account_checks": checks, "cycle_counts": cycle_counts,
        "source_months": 72, "new_parameter_searches": 0, "new_funding_regression_fits": 0, "new_independent_forward_observations": 0,
        "historical_first_vintage_verified": False, "current_market_view": "NO_VIEW", "goal_status": "active", "goal_achieved": False,
        "orders_authorized": False, "review_package_created": False}, True)
    print("财务收款期限8个完整账户及月报区块比较已完成。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="工业企业收款期限的两年日更研究")
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
