"""同一贷款需求判断下的审批条件增量，只操作510300与现金。"""
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

import research.banker_survey_availability_completion_v1 as source
import research.loan_maturity_composition_daily_v1 as loan
from research.selected_mix_reappraisal_v1 import read, save, now, digest
from research.selected_mix_daily_two_year_v1 import summarize_accounts, paired_interval
from research.strategy_review_diagnostics_v1 import metrics, cycles

shared, prior, maturity = loan.shared, loan.prior, loan.maturity
OUT = ROOT / "reports/research/510300_banker_loan_approval_daily_v1"
STUDY = "510300_BANKER_LOAN_APPROVAL_DAILY_V1"
START, END = loan.START, loan.END
FEATURE, DEMAND = "loan_approval_index", "loan_demand_index"
CALENDAR = ["survey_quarter_sin", "survey_quarter_cos"]
CONTROL = "PRICE_FUNDING_AND_SURVEY_QUARTER"
TOTAL = "PRICE_FUNDING_QUARTER_AND_LOAN_DEMAND"
SIGNAL = "PRICE_FUNDING_SURVEY_DEMAND_AND_APPROVAL"
BASE = [*prior.parent.BASE, "funding_innovation", *CALENDAR]
MODELS = {"HISTORY": [], CONTROL: BASE, TOTAL: [*BASE, DEMAND], SIGNAL: [*BASE, DEMAND, FEATURE]}
TRADED, VARIANTS = [CONTROL, TOTAL, SIGNAL], shared.VARIANTS
PRIMARY = SIGNAL + "__FIXED5_NO_ADD"
FIT_BASE = shared.adapt(prior.fit_day, {"parent": SimpleNamespace(BASE=[*prior.parent.BASE, *CALENDAR, DEMAND]),
                                      "FEATURE": FEATURE, "MODELS": MODELS, "PRIMARY": SIGNAL})


def information(releases=None):
    x = pd.read_parquet(prior.funding.OUT / "inputs/decision_information.parquet")
    q = (pd.read_parquet(source.PANEL) if releases is None else releases).copy()
    q["known_at"] = pd.to_datetime(q.known_at).dt.tz_convert("Asia/Shanghai").dt.as_unit("ns")
    q = q.sort_values(["known_at", "quarter"]).drop_duplicates("known_at", keep="last")
    quarter = pd.PeriodIndex(q.quarter, freq="Q").quarter
    q["survey_quarter_sin"] = np.sin(2 * np.pi * (quarter - 1) / 4)
    q["survey_quarter_cos"] = np.cos(2 * np.pi * (quarter - 1) / 4)
    cols = ["known_at", "quarter", DEMAND, FEATURE, *CALENDAR]
    q = q[cols].rename(columns={"quarter": "survey_quarter", "known_at": "survey_known_at"})
    x = pd.merge_asof(x.sort_values("decision_time"), q, left_on="decision_time", right_on="survey_known_at", direction="backward")
    x["survey_age_days"] = (x.decision_time - x.survey_known_at).dt.total_seconds() / 86400
    x["bank_information_known"] = x.common_known
    x["common_known"] &= x.survey_age_days.between(0, 120) & x[[FEATURE, DEMAND, *CALENDAR]].notna().all(axis=1)
    assert x.loc[x.common_known, "survey_known_at"].lt(x.loc[x.common_known, "decision_time"]).all()
    assert x.idx.tolist() == list(range(len(x)))
    return x


def fit_day(i, x, nested, count, labels):
    lower = x.date.iloc[i] - pd.DateOffset(years=2)
    pool = labels[labels.date.ge(lower) & labels.exit_idx.lt(i)]
    ids = pool.idx.to_numpy(int)
    inside = x.survey_known_at.iloc[ids].ge(lower.tz_localize("Asia/Shanghai")).to_numpy(bool)
    rows, models, receipt = FIT_BASE(i, x, nested, count, pool.loc[inside])
    receipt["distinct_training_source_quarters"] = int(x.loc[models[0]["training_indices"], "survey_quarter"].nunique()) if models else 0
    return rows, models, receipt


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("贷款审批实验已固定。")
    x, q = information(), pd.read_parquet(source.PANEL)
    cutoff = pd.Timestamp("2024-06-28", tz="Asia/Shanghai")
    changed = q.copy()
    changed.loc[changed.known_at.ge(cutoff), [FEATURE, DEMAND]] += 700.
    after = information(changed)
    cols = [FEATURE, DEMAND, *CALENDAR, "common_known"]
    pd.testing.assert_frame_equal(x.loc[x.decision_time.lt(cutoff), cols], after.loc[after.decision_time.lt(cutoff), cols], check_exact=True)
    for folder in ["inputs", "code", "results", "accounts"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY, "primary": PRIMARY,
        "question": "银行家对贷款需求的判断相同或相近时，审批条件的松紧能否区别后续的价格修复分布？",
        "source": read(source.OUT / "result.json"),
        "feature": "直接采用当季原报告贷款审批扩散指数，定义为放松比例加0.5乘基本不变比例；不是贷款批准率、放贷量或中性利率。",
        "mechanism": "贷款数量同时受需求与供给影响，本轮尝试在需求判断之外加入审批条件。调查为银行家主观扩散判断，不能证明供给冲击的因果性，也不预设宽松必然利好股票。",
        "controls": "CONTROL为价格、银行资金和调查季度sin/cos，TOTAL增加需求扩散指数，SIGNAL最后只加审批指数。政策感受指数不入模，避免再加第三项挑选。",
        "known_at": "32份原报告产生29个不同保守公布日；同日补发取最新统计季度作为之后状态，较旧季度不回填其历史日期。最长120自然日，逾期未知。该上限只是事前选定新研究条件。",
        "statistics": "删除宏观信心列及行业表头变化不影响本次两项目标的定义；按各原报告表头提取，编制编号不充当列号。受访机构历史构成不可变证明仍不充分。",
        "training": "每天仅最近两日历年且五日标签已成熟的共同原点；原报告公布时间本身也不早于两年下界；至少252原点，固定126近邻，训练内标准化及截断5。季度填充日不算独立宏观事实。",
        "funding": "复用每个当前两年窗内嵌套重建的FDR007日历残差，无新增资金回归或参数。",
        "models": MODELS, "traded_models": TRADED, "variants": VARIANTS,
        "account": "三组信息、两种期限、两种成本共12新账户；20万元、只510300和现金，五日弱势时才新增，持仓不加仓。主固定第五个后续开盘到期，每日尾部约束可提前减仓。",
        "period": [START, END], "annual_days": 242, "cash_and_risk_free": 0, "costs": prior.distribution.COSTS,
        "risk_contract": read(ROOT / "config/510300_existing_data_training_mandate_v1.json")["account_tail_risk_contract"],
        "targets": {"stress_sharpe": 1.2, "stress_cagr": .1, "max_drawdown": .1},
        "comparisons": "主固定五日对两个信息对照、同信息滚动期限；20日循环区块及完整季度公布间隔区块各2000次，seed2026092610，固定两时期及年度、滚动两年全部披露。",
        "development_gate": "主压力三项目标通过，两类区块下界及两个固定时期的信息增量均正；仍需独立前向，不把季度变量的日频重复当独立验证。",
        "monitor": "每5个固定相位的成熟q05结果，60条窗口突破率>15%记录；无报警不证明优势。",
        "common_information_days": int(x.common_known.sum()), "future_source_perturbation_pass": True,
        "new_accounts": 12, "new_parameter_searches": 0, "new_funding_regression_fits": 0,
        "new_independent_forward_observations": 0, "historical_first_vintage_verified": False,
        "goal_achieved": False, "current_market_view": "NO_VIEW", "orders_authorized": False}, True)
    paths = [Path(__file__), Path(source.__file__), source.PANEL, source.OUT / "result.json",
             source.fields.OUT / "source_identity_completion.json", Path(loan.__file__), Path(shared.__file__), Path(prior.__file__),
             Path(prior.parent.__file__), Path(prior.distribution.__file__), Path(prior.engine.__file__), Path(maturity.__file__),
             ROOT / "research/selected_mix_daily_two_year_v1.py", ROOT / "research/strategy_review_diagnostics_v1.py",
             prior.parent.DIVIDENDS, prior.parent.OUT / "inputs/market.parquet", prior.parent.OUT / "inputs/mature_labels.parquet",
             prior.funding.OUT / "inputs/decision_information.parquet", prior.funding.OUT / "inputs/funding_features.parquet",
             prior.funding.OUT / "results/nested_funding_predictions.parquet"]
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
        "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}, "before_new_strategy_returns": True}, True)
    save(OUT / "inputs/mandate.json", read(ROOT / "config/510300_existing_data_training_mandate_v1.json"), True)
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    print(f"需求与审批增量的12个账户已固定，共同信息日{x.common_known.sum()}。", flush=True)


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
            print(f"贷款审批分布已计算至{x.date.iloc[i].date()}。", flush=True)
    pred, schedule = pd.DataFrame(predictions), pd.DataFrame(receipts)
    pred.to_parquet(OUT / "results/predictions.parquet", index=False)
    schedule.to_parquet(OUT / "results/update_receipts.parquet", index=False)
    save(OUT / "results/saved_distributions.json", models, True)
    verification = shared.adapt(prior.verify_predictions, {"fit_day": fit_day})(pred, models, x, labels, groups, count)
    for m in models:
        assert x.loc[m["training_indices"], "survey_known_at"].ge((x.date.iloc[m["idx"]] - pd.DateOffset(years=2)).tz_localize("Asia/Shanghai")).all()
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
    comparisons, rng = [], np.random.default_rng(2026092610)
    for cost in prior.distribution.COSTS:
        left = accounts[PRIMARY, cost]
        for name, purpose in [(CONTROL, "DEMAND_AND_APPROVAL_INCREMENT"), (TOTAL, "APPROVAL_INCREMENT")]:
            right = accounts[name + "__FIXED5_NO_ADD", cost]
            periods = []
            for lo, hi in [(START, "2023-12-31"), ("2024-01-01", END)]:
                mask = left.date.between(lo, hi)
                periods.append({"start": lo, "end": hi, "annual_arithmetic_difference": float((left.loc[mask, "net_return"] - right.loc[mask, "net_return"]).mean() * 242)})
            interval = loan.release_block_interval(left, right, x.assign(loan_stat_month=x.survey_quarter), rng)
            interval["limitation"] = "按完整季度公布间隔重采样，未保留相邻季度全部持续性，且未调整历史反复选择；不是独立验证。"
            comparisons.append({"cost": cost, "control": name, "purpose": purpose,
                "daily_blocks": paired_interval(left, right, rng), "quarter_release_blocks": interval, "fixed_periods": periods})
        comparisons.append({"cost": cost, "control": SIGNAL + "__ROLL_NO_ADD", "purpose": "EXECUTION",
            "daily_blocks": paired_interval(left, accounts[SIGNAL + "__ROLL_NO_ADD", cost], rng)})
    main = next(m for m in measures if m["policy"] == PRIMARY and m["cost"] == "STRESS")
    increment = all(r["daily_blocks"]["lower_95"] > 0 and r["quarter_release_blocks"]["lower_95"] > 0
        and all(p["annual_arithmetic_difference"] > 0 for p in r["fixed_periods"])
        for r in comparisons if r["cost"] == "STRESS" and r["purpose"] != "EXECUTION")
    ledger = accounts[PRIMARY, "STRESS"]
    gross = float((ledger.price_pnl + ledger.dividend_recognized).sum())
    expense = float((ledger.commission + ledger.slippage_cost).sum() + ledger.terminal_exit_reserve.iloc[-1])
    np.testing.assert_allclose(gross - expense, ledger.equity.iloc[-1] - 200000, atol=1e-6, rtol=0)
    ready, updated = pred[["idx", "date"]].drop_duplicates(), schedule[schedule.status.eq("UPDATED")]
    save(OUT / "result.json", {"at": now(), "study_id": STUDY,
        "status": "DEVELOPMENT_CANDIDATE_REQUIRES_INDEPENDENT_VALIDATION" if main["historical_point_targets_met"] and increment else "FROZEN_NO_QUALIFIED_BANKER_APPROVAL_STRATEGY",
        "primary": main, "all_accounts": measures, "comparisons": comparisons, "new_full_accounts": len(accounts),
        "joint_target_pass_accounts": sum(bool(m["historical_point_targets_met"]) for m in measures),
        "new_conditional_distributions": len(pred), "unique_prediction_days": len(ready),
        "first_prediction_date": ready.date.min(), "last_prediction_date": ready.date.max(),
        "training_rows_min": int(updated.n_train.min()), "training_rows_max": int(updated.n_train.max()),
        "distinct_training_source_quarters_min": int(updated.distinct_training_source_quarters.min()),
        "distinct_training_source_quarters_max": int(updated.distinct_training_source_quarters.max()),
        "schedule_statuses": schedule.status.value_counts().to_dict(), "forecast_evaluation": evaluation,
        "development_increment_gate": increment,
        "primary_cash_attribution": {"gross_pnl": gross, "cost_and_reserve": expense, "net_profit": float(ledger.equity.iloc[-1] - 200000)},
        "primary_rolling_two_year_joint_passes": sum(r["joint_point_pass"] for r in rolling if r["policy"] == PRIMARY and r["cost"] == "STRESS"),
        "mature_monitor": [{"model": name, "mature_nonoverlap_rows": len(part), "assessable_rows": int(part.monitor_assessable.sum()), "alerts": int(part.alert.sum())} for name, part in monitor.groupby("model")],
        "prediction_checks": verification, "account_checks": checks, "cycle_counts": cycle_counts,
        "new_parameter_searches": 0, "new_funding_regression_fits": 0, "new_independent_forward_observations": 0,
        "source_quarters": 32, "effective_release_days": 29, "historical_first_vintage_verified": False,
        "current_market_view": "NO_VIEW", "goal_status": "active", "goal_achieved": False,
        "orders_authorized": False, "review_package_created": False}, True)
    print("贷款需求与审批的12个完整账户及季度区块比较已完成。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="银行家贷款审批条件的每日两年研究")
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
