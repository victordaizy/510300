"""贷款总量与期限构成的增量：同口径、同日历对照和两年日更完整账户。"""
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

import research.rmb_loan_composition_completion_v1 as source
import research.exchange_bank_funding_gap_daily_v1 as shared
from research.selected_mix_reappraisal_v1 import read, save, now, digest
from research.selected_mix_daily_two_year_v1 import summarize_accounts, paired_interval
from research.strategy_review_diagnostics_v1 import metrics, cycles

prior, maturity = shared.prior, shared.maturity
OUT = ROOT / "reports/research/510300_loan_maturity_composition_daily_v1"
STUDY = "510300_LOAN_MATURITY_COMPOSITION_DAILY_V1"
START, END = shared.START, shared.END
FEATURE = "long_term_net_loan_share_percent"
AGGREGATE = "rmb_stock_yoy_percent"
CALENDAR = ["stat_month_sin", "stat_month_cos"]
CONTROL = "PRICE_BANK_AND_STAT_CALENDAR"
TOTAL = "PRICE_BANK_CALENDAR_AND_LOAN_GROWTH"
COMPOSITION = "PRICE_BANK_CALENDAR_GROWTH_AND_LOAN_MATURITY"
BASE_FEATURES = [*prior.parent.BASE, "funding_innovation", *CALENDAR]
MODELS = {"HISTORY": [], CONTROL: BASE_FEATURES, TOTAL: [*BASE_FEATURES, AGGREGATE],
          COMPOSITION: [*BASE_FEATURES, AGGREGATE, FEATURE]}
TRADED, VARIANTS = [CONTROL, TOTAL, COMPOSITION], shared.VARIANTS
PRIMARY = COMPOSITION + "__FIXED5_NO_ADD"
FIT_BASE = shared.adapt(prior.fit_day, {"parent": SimpleNamespace(BASE=[*prior.parent.BASE, *CALENDAR, AGGREGATE]),
                                     "FEATURE": FEATURE, "MODELS": MODELS, "PRIMARY": COMPOSITION})


def information(releases=None):
    x = pd.read_parquet(prior.funding.OUT / "inputs/decision_information.parquet")
    if releases is None:
        releases = pd.read_parquet(source.OUT / "released_loan_composition.parquet")
    q = releases.copy()
    q["conservative_known_at"] = pd.to_datetime(q.conservative_known_at).dt.tz_convert("Asia/Shanghai").dt.as_unit("ns")
    lookup = q.set_index("stat_month").conservative_known_at.to_dict()
    q["earliest_component_release_at"] = [min(lookup[m] for m in months) for months in q.cumulative_source_months]
    q["earliest_component_release_at"] = pd.to_datetime(q.earliest_component_release_at).dt.tz_convert("Asia/Shanghai").dt.as_unit("ns")
    month = pd.PeriodIndex(q.stat_month, freq="M").month
    q["stat_month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    q["stat_month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    cols = ["stat_month", "conservative_known_at", "earliest_component_release_at", "statistical_regime", "ratio_available",
            FEATURE, AGGREGATE, *CALENDAR, "corporate_long_contribution_percent", "household_long_contribution_percent", "bill_contribution_percent"]
    q = q[cols].rename(columns={"stat_month": "loan_stat_month", "conservative_known_at": "loan_known_at"})
    x = pd.merge_asof(x.sort_values("decision_time"), q.sort_values("loan_known_at"),
                      left_on="decision_time", right_on="loan_known_at", direction="backward")
    x["bank_information_known"] = x.common_known
    age = (x.decision_time - x.loan_known_at).dt.total_seconds() / 86400
    x["loan_information_age_days"] = age
    x["common_known"] &= x.ratio_available.fillna(False).astype(bool) & age.between(0, 31) & x[[FEATURE, AGGREGATE, *CALENDAR]].notna().all(axis=1)
    assert x.loc[x.common_known, "loan_known_at"].lt(x.loc[x.common_known, "decision_time"]).all()
    # 未公布期限的月报到来时覆盖旧记录为未知，不跳过它继续沿用先前有效数值。
    assert not x.loc[x.loan_stat_month.isin(["2022-04", "2022-05"]), "common_known"].any()
    return x


def fit_day(i, x, nested, count, labels):
    lower = x.date.iloc[i] - pd.DateOffset(years=2)
    regime = x.statistical_regime.iloc[i]
    # 先筛掉不成熟或过期标签，使截断到当前日的重算也不读取未来行。
    labels = labels[labels.date.ge(lower) & labels.exit_idx.lt(i)]
    ids = labels.idx.to_numpy(int)
    same = x.statistical_regime.iloc[ids].eq(regime).to_numpy(bool)
    dates = x.earliest_component_release_at.iloc[ids]
    inside = dates.ge(lower.tz_localize("Asia/Shanghai")).to_numpy(bool)
    rows, saved, receipt = FIT_BASE(i, x, nested, count, labels.loc[same & inside])
    receipt["statistical_regime"] = regime
    receipt["distinct_training_release_months"] = int(x.loc[saved[0]["training_indices"], "loan_stat_month"].nunique()) if saved else 0
    return rows, saved, receipt


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("贷款期限构成试验已经固定。")
    x = information()
    q = pd.read_parquet(source.OUT / "released_loan_composition.parquet")
    cutoff = pd.Timestamp("2024-06-28", tz="Asia/Shanghai")
    mutated = q.copy()
    later = pd.to_datetime(mutated.conservative_known_at).ge(cutoff)
    mutated.loc[later, FEATURE] += 700.
    mutated.loc[later, AGGREGATE] += 700.
    after = information(mutated)
    cols = [FEATURE, AGGREGATE, *CALENDAR, "common_known"]
    pd.testing.assert_frame_equal(x.loc[x.decision_time.lt(cutoff), cols], after.loc[after.decision_time.lt(cutoff), cols], check_exact=True)
    for directory in ["code", "inputs", "results", "accounts"]:
        (OUT / directory).mkdir(parents=True, exist_ok=True)
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY, "primary": PRIMARY,
        "question": "在同样的价格、银行资金、统计月份和人民币贷款总量增速下，新增贷款偏向企业/居民中长期的程度能否改善后续修复判断与账户？",
        "new_source_evidence": read(source.OUT / "result.json"),
        "feature": "(企业中长期贷款年初累计净增+居民中长期贷款年初累计净增)/人民币贷款年初累计净增，乘100。原文亿元统一后先统一统计区间；负净增保留。净增分项比率不强制介于0和100%。",
        "mechanism_hypothesis": "中长期融资承诺可能区别于短期票据资金需求，但也可能来自政策性融资、续贷或还款变化。报告无法拆出纯民企投资、私人自发需求或直接股票买盘，不预先把比率高判为利好。",
        "components": "企业、居民中长期及票据分别保留用于解释；本次入模只加合并中长期比率，不根据账户挑选某部门。",
        "controls": "所有模型共有统计月份的sin/cos，防止年初投放季节性被当成构成优势；第二对照再加人民币贷款余额官方同比，主方案最后加一个期限比率。不是社融政府债比重的同一用途，也不恢复旧M1/M2或信用利差规则。",
        "known_at": "公布日结束后最近的A股09:00决策，信息年龄不超过31自然日；没有把统计月末视为可用日。未知月报覆盖旧值为未知。历史初版不可变记录仍不充分。",
        "methodology_break": "2023年1月起扩展统计机构范围。每个当时训练池仅保留与当前最新已公布月报相同的统计范围版本；新版本不足252个成熟原点时NO_VIEW，不混入旧口径救样本。",
        "training": "每天只取最近两个日历年的原点，五日退出开盘严格早于当前决策；用于累计构造的每份月报公布时间也不得早于当前两年下界。至少252原点，126近邻，训练内标准化截断5。月报反复用于日预测不能算新的独立月度样本。",
        "funding": "复用既有每个当前两年窗内重新嵌套的FDR007日历残差；不携两年前参数，不重新估计或搜索资金模型。",
        "models": MODELS, "traded_models": TRADED, "variants": VARIANTS,
        "account": "3信息模型x2持仓期限x2成本，共12新完整账户；20万元、510300.SH和人民币现金、T+1、100份、tick0.001，只有此前五日价格弱势可新增仓位，持仓期间不加仓。主固定第五个后续开盘退出，风险约束可提前缩量。",
        "risk_contract": read(ROOT / "config/510300_existing_data_training_mandate_v1.json")["account_tail_risk_contract"],
        "period": [START, END], "annual_days": 242, "cash_and_riskfree": 0,
        "costs": prior.distribution.COSTS, "targets": {"stress_sharpe": 1.2, "stress_cagr": .1, "max_drawdown": .1},
        "comparisons": "主固定五日相对两个同期限信息对照、相同信息滚动期限；20交易日循环区块2000次seed2026092608，全部年度/滚动两年。宏观共同冲击另按完整公布间隔区块重采样，不能把20个日预测当20条独立宏观事实。",
        "development_gate": "主压力三目标通过，相对两个信息对照的20日区块和公布间隔区块95%下界均正，固定2020-2023与2024-末端增量均正；仍需独立前向和首版时钟验证。",
        "monitor": "固定每5个A股原点的已成熟q05预测，最近60条突破率>15%记录。没有尾部警报不证明收益优势有效。",
        "history_previously_researched": True, "new_parameter_searches": 0, "new_full_accounts": 12,
        "new_funding_regression_fits": 0, "new_independent_forward_observations": 0,
        "future_release_perturbation_pass": True, "common_information_days": int(x.common_known.sum()),
        "goal_achieved": False, "current_market_view": "NO_VIEW", "orders_authorized": False}, True)
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    save(OUT / "inputs/mandate.json", read(ROOT / "config/510300_existing_data_training_mandate_v1.json"), True)
    paths = [Path(__file__), Path(source.__file__), Path(prior.__file__), Path(shared.__file__), Path(maturity.__file__),
             Path(prior.parent.__file__), Path(prior.engine.__file__), Path(prior.distribution.__file__), prior.parent.DIVIDENDS,
             ROOT / "research/selected_mix_daily_two_year_v1.py", ROOT / "research/strategy_review_diagnostics_v1.py",
             source.OUT / "result.json", source.OUT / "released_loan_composition.parquet", source.OUT / "extracted_originals.json",
             prior.parent.OUT / "inputs/market.parquet", prior.parent.OUT / "inputs/mature_labels.parquet",
             prior.funding.OUT / "inputs/decision_information.parquet", prior.funding.OUT / "inputs/funding_features.parquet",
             prior.funding.OUT / "results/nested_funding_predictions.parquet"]
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
         "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}, "before_new_candidate_returns": True}, True)
    print(f"贷款期限构成12账户已固定，共同信息日{x.common_known.sum()}；统计范围变化后重新积累训练。", flush=True)


def release_block_interval(left, right, x, rng):
    """按连续公布间隔整体重采样，保留每个月报覆盖日的共同性。"""
    a = left.net_return.to_numpy(float) - right.net_return.to_numpy(float)
    months = x.set_index("idx").loc[left.idx, "loan_stat_month"].fillna("NO_RELEASE").to_numpy()
    changes = np.r_[True, months[1:] != months[:-1]]
    intervals = np.split(np.arange(len(a)), np.flatnonzero(changes)[1:])
    sums = np.array([a[ids].sum() for ids in intervals])
    counts = np.array([len(ids) for ids in intervals])
    draws = rng.integers(0, len(intervals), size=(2000, len(intervals)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1) * 242
    return {"annual_arithmetic_difference": float(a.mean() * 242), "lower_95": float(np.quantile(means, .025)),
            "upper_95": float(np.quantile(means, .975)), "release_intervals": len(intervals),
            "replications": 2000, "selection_adjusted": False,
            "limitation": "逐公布间隔重采样不保留相邻月报之间全部持久性，仍只是开发不确定性诊断。"}


def run():
    execution_freeze = OUT / "execution_freeze.json"
    frozen = read(execution_freeze if execution_freeze.exists() else OUT / "freeze.json")
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
            print(f"贷款期限构成分布已更新至{x.date.iloc[i].date()}。", flush=True)
    pred, schedule = pd.DataFrame(predictions), pd.DataFrame(receipts)
    pred.to_parquet(OUT / "results/predictions.parquet", index=False)
    schedule.to_parquet(OUT / "results/update_receipts.parquet", index=False)
    save(OUT / "results/saved_distributions.json", models, True)
    verification = shared.adapt(prior.verify_predictions, {"fit_day": fit_day})(pred, models, x, labels, groups, count)
    for m in models:
        indices, i = m["training_indices"], m["idx"]
        assert x.loc[indices, "statistical_regime"].eq(x.statistical_regime.iloc[i]).all()
        assert x.loc[indices, "earliest_component_release_at"].ge((x.date.iloc[i] - pd.DateOffset(years=2)).tz_localize("Asia/Shanghai")).all()
    verification.update(statistical_regime_never_pooled=True, cumulative_release_dependencies_inside_two_years=True)
    save(OUT / "results/prediction_verification.json", verification, True)
    evaluation, scored, monitor = shared.adapt(prior.forecast_evaluation, {"OUT": OUT, "PRIMARY": COMPOSITION, "MODELS": MODELS})(pred, labels, x)
    accounts, checks, counts = {}, [], []
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
                counts.append({"policy": policy, "cost": cost, "closed": len(finished), "open": pending is not None})
                m = metrics(ledger)
                sh = "未定义" if m["sharpe"] is None else f"{m['sharpe']:.6f}"
                print(f"{policy}/{cost}：夏普{sh}、年化{m['annual_return']:.3%}、回撤{abs(m['max_drawdown']):.3%}。", flush=True)
    measures, rolling = summarize_accounts(OUT, market, accounts)
    comparisons, rng = [], np.random.default_rng(2026092608)
    for cost in prior.distribution.COSTS:
        left = accounts[PRIMARY, cost]
        for name, purpose in [(CONTROL, "TOTAL_AND_COMPOSITION"), (TOTAL, "COMPOSITION_INCREMENT")]:
            right = accounts[name + "__FIXED5_NO_ADD", cost]
            eras = []
            for lo, hi in [(START, "2023-12-31"), ("2024-01-01", END)]:
                mask = left.date.between(lo, hi)
                eras.append({"start": lo, "end": hi, "annual_arithmetic_difference": float((left.loc[mask, "net_return"] - right.loc[mask, "net_return"]).mean() * 242)})
            comparisons.append({"cost": cost, "control": name, "purpose": purpose,
                                "daily_blocks": paired_interval(left, right, rng), "release_blocks": release_block_interval(left, right, x, rng), "fixed_periods": eras})
        comparisons.append({"cost": cost, "control": COMPOSITION + "__ROLL_NO_ADD", "purpose": "EXECUTION",
                            "daily_blocks": paired_interval(left, accounts[COMPOSITION + "__ROLL_NO_ADD", cost], rng)})
    main = next(m for m in measures if m["policy"] == PRIMARY and m["cost"] == "STRESS")
    increment = all(r["daily_blocks"]["lower_95"] > 0 and r["release_blocks"]["lower_95"] > 0
                    and all(e["annual_arithmetic_difference"] > 0 for e in r["fixed_periods"])
                    for r in comparisons if r["cost"] == "STRESS" and r["purpose"] != "EXECUTION")
    ledger = accounts[PRIMARY, "STRESS"]
    gross = float((ledger.price_pnl + ledger.dividend_recognized).sum())
    expense = float((ledger.commission + ledger.slippage_cost).sum() + ledger.terminal_exit_reserve.iloc[-1])
    np.testing.assert_allclose(gross - expense, ledger.equity.iloc[-1] - 200000, atol=1e-6, rtol=0)
    ready, updated = pred[["idx", "date"]].drop_duplicates(), schedule[schedule.status.eq("UPDATED")]
    save(OUT / "result.json", {"at": now(), "study_id": STUDY,
        "status": "DEVELOPMENT_CANDIDATE_REQUIRES_INDEPENDENT_VALIDATION" if main["historical_point_targets_met"] and increment else "FROZEN_NO_QUALIFIED_LOAN_MATURITY_COMPOSITION_STRATEGY",
        "primary": main, "all_accounts": measures, "comparisons": comparisons,
        "new_full_accounts": len(accounts), "new_conditional_distributions": len(pred), "unique_prediction_days": len(ready),
        "first_prediction_date": ready.date.min(), "last_prediction_date": ready.date.max(),
        "training_rows_min": int(updated.n_train.min()), "training_rows_max": int(updated.n_train.max()),
        "distinct_training_release_months_min": int(updated.distinct_training_release_months.min()),
        "distinct_training_release_months_max": int(updated.distinct_training_release_months.max()),
        "schedule_statuses": schedule.status.value_counts().to_dict(),
        "updated_days_by_statistical_regime": updated.statistical_regime.value_counts().to_dict(),
        "forecast_evaluation": evaluation, "development_increment_gate": increment,
        "primary_cash_attribution": {"gross_pnl": gross, "cost_and_reserve": expense, "net_profit": float(ledger.equity.iloc[-1] - 200000)},
        "primary_rolling_two_year_joint_passes": sum(r["joint_point_pass"] for r in rolling if r["policy"] == PRIMARY and r["cost"] == "STRESS"),
        "mature_monitor": [{"model": name, "mature_nonoverlap_rows": len(part), "assessable_rows": int(part.monitor_assessable.sum()), "alerts": int(part.alert.sum())}
                           for name, part in monitor.groupby("model")],
        "prediction_checks": verification, "account_checks": checks, "cycle_counts": counts,
        "new_parameter_searches": 0, "new_funding_regression_fits": 0, "new_independent_forward_observations": 0,
        "source_months": 104, "joint_ratio_months": 102, "historical_first_vintage_verified": False,
        "current_market_view": "NO_VIEW", "goal_status": "active", "goal_achieved": False,
        "orders_authorized": False, "review_package_created": False}, True)
    print("贷款期限构成12个完整账户及两种区块增量检验已完成。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="人民币贷款期限构成的每日两年滚动研究")
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
