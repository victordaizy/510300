"""企业销货款回笼对银行条件与价格修复的单项增量研究。"""
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

import research.entrepreneur_cash_field_admission_v1 as source
import research.banker_loan_approval_daily_v1 as banking
from research.selected_mix_reappraisal_v1 import read, save, now, digest
from research.selected_mix_daily_two_year_v1 import summarize_accounts, paired_interval
from research.strategy_review_diagnostics_v1 import metrics, cycles

shared, prior, maturity, loan = banking.shared, banking.prior, banking.maturity, banking.loan
OUT = ROOT / "reports/research/510300_entrepreneur_sales_collection_daily_v1"
STUDY = "510300_ENTREPRENEUR_SALES_COLLECTION_DAILY_V1"
START, END = banking.START, banking.END
FEATURE = "sales_revenue_collection_index"
CONTROL, BANK = banking.CONTROL, banking.SIGNAL
SIGNAL = "PRICE_FUNDING_BANKING_AND_SALES_COLLECTION"
MODELS = {"HISTORY": [], CONTROL: banking.BASE, BANK: banking.MODELS[BANK],
          SIGNAL: [*banking.MODELS[BANK], FEATURE]}
TRADED, VARIANTS = [CONTROL, BANK, SIGNAL], shared.VARIANTS
PRIMARY = SIGNAL + "__FIXED5_NO_ADD"
FIT_BASE = shared.adapt(prior.fit_day, {
    "parent": SimpleNamespace(BASE=[*prior.parent.BASE, *banking.CALENDAR, banking.DEMAND, banking.FEATURE]),
    "FEATURE": FEATURE, "MODELS": MODELS, "PRIMARY": SIGNAL})


def information(enterprise_releases=None, bank_releases=None):
    x = banking.information(bank_releases)
    q = (pd.read_parquet(source.PANEL) if enterprise_releases is None else enterprise_releases).copy()
    q["known_at"] = pd.to_datetime(q.known_at).dt.tz_convert("Asia/Shanghai").dt.as_unit("ns")
    q = q.sort_values(["known_at", "quarter"]).drop_duplicates("known_at", keep="last")
    q = q[["known_at", "quarter", FEATURE]].rename(columns={"known_at": "enterprise_known_at", "quarter": "enterprise_quarter"})
    x = pd.merge_asof(x.sort_values("decision_time"), q, left_on="decision_time", right_on="enterprise_known_at", direction="backward")
    x["enterprise_age_days"] = (x.decision_time - x.enterprise_known_at).dt.total_seconds() / 86400
    x["bank_survey_information_known"] = x.common_known
    x["common_known"] &= x.enterprise_age_days.between(0, 120) & x[FEATURE].notna() & x.enterprise_quarter.eq(x.survey_quarter)
    assert x.loc[x.common_known, "enterprise_known_at"].lt(x.loc[x.common_known, "decision_time"]).all()
    assert x.idx.tolist() == list(range(len(x)))
    return x


def fit_day(i, x, nested, count, labels):
    lower = x.date.iloc[i] - pd.DateOffset(years=2)
    pool = labels[labels.date.ge(lower) & labels.exit_idx.lt(i)]
    ids = pool.idx.to_numpy(int)
    lower_clock = lower.tz_localize("Asia/Shanghai")
    inside = (x.survey_known_at.iloc[ids].ge(lower_clock) & x.enterprise_known_at.iloc[ids].ge(lower_clock)).to_numpy(bool)
    rows, models, receipt = FIT_BASE(i, x, nested, count, pool.loc[inside])
    receipt["distinct_training_source_quarters"] = int(x.loc[models[0]["training_indices"], "enterprise_quarter"].nunique()) if models else 0
    return rows, models, receipt


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("企业收款增量已固定，不能覆盖。")
    x = information()
    q = pd.read_parquet(source.PANEL)
    b = pd.read_parquet(banking.source.PANEL)
    cutoff = pd.Timestamp("2024-06-28", tz="Asia/Shanghai")
    changed_q, changed_b = q.copy(), b.copy()
    changed_q.loc[changed_q.known_at.ge(cutoff), FEATURE] += 700.
    changed_b.loc[changed_b.known_at.ge(cutoff), [banking.DEMAND, banking.FEATURE]] += 700.
    after = information(changed_q, changed_b)
    cols = [FEATURE, banking.DEMAND, banking.FEATURE, *banking.CALENDAR, "common_known"]
    pd.testing.assert_frame_equal(x.loc[x.decision_time.lt(cutoff), cols], after.loc[after.decision_time.lt(cutoff), cols], check_exact=True)
    old_information = pd.read_parquet(banking.OUT / "inputs/decision_information.parquet")
    control_cols = ["idx", "date", "decision_time", "common_known", "survey_known_at", "survey_quarter", "source_idx", "available_at",
                    *prior.parent.BASE, *banking.CALENDAR, banking.DEMAND, banking.FEATURE]
    pd.testing.assert_frame_equal(x[control_cols], old_information[control_cols], check_exact=True)
    both = x.survey_known_at.notna() | x.enterprise_known_at.notna()
    pd.testing.assert_series_equal(x.loc[both, "survey_known_at"], x.loc[both, "enterprise_known_at"], check_names=False, check_exact=True)
    for folder in ["inputs", "code", "results", "accounts"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY, "primary": PRIMARY,
        "question": "银行需求与审批条件之外，企业自身收款状况能否区别相似价格压力之后的五日修复分布？",
        "hypothesis": "信用条件宽松与企业回款良好不是同一事实。回款改善可能减少被迫现金需求，但也可能滞后经济或市场价格，方向与效果只由固定训练规则检验。",
        "feature": "销货款回笼指数原季值：企业判断良好的比例加0.5乘一般比例。全国工业企业问卷，不是510300成分股财务现金流或股票资金流。",
        "unused_field": "资金周转指数只做原文定义核对，不参与策略、筛选或事后替换。",
        "source": read(source.OUT / "result.json"),
        "controls": "共同样本下的价格/银行资金/调查季度对照，及增加贷款需求与审批的对照。主模型再加一项企业收款，不视银行对照为已获验证的优势。",
        "clock": "银行与企业各按本报告公布日末可用，均不超过120自然日、且最新可见统计季度相同。原季度每份只取当季行，补发不得回填。",
        "training": "每天最近两日历年且五日退出开盘已过去的共同成熟原点。银行和企业原文的公布时间均不早于两年下界；至少252行、固定126近邻、训练内标准化并截断5。",
        "quarter_dependence": "一个季度反复填充的交易日不是独立的宏观观察；同时报告实际不同季度数与整季公布间隔区块。",
        "funding": "复用每个当前两年窗内嵌套重建的FDR007日历残差，不新增资金回归。",
        "models": MODELS, "traded_models": TRADED, "variants": VARIANTS,
        "account": "仅主新增信息的BASE/STRESS及固定/滚动期限共4个新账户；8个对照在时点、输入、训练池及预测逐值一致后复用原账本。20万元、510300与现金、弱势新增、持仓不加仓。",
        "control_reuse": "原银行实验已保存的价格资金对照与需求审批对照；共同信息表已逐值一致，执行前还要逐模型核对完整预测，失败即停止复用。",
        "period": [START, END], "annual_days": 242, "cash_and_risk_free": 0, "costs": prior.distribution.COSTS,
        "risk_contract": read(ROOT / "config/510300_existing_data_training_mandate_v1.json")["account_tail_risk_contract"],
        "targets": {"stress_sharpe": 1.2, "stress_cagr": .1, "max_drawdown": .1},
        "comparisons": "主固定五日对两个信息对照及同信息滚动期限。20日循环区块和季度公布间隔区块各2000次，seed2026092611；两个固定时期及所有年度、滚动两年保留。",
        "development_gate": "主压力三目标同时通过；对两个信息对照的两类区块下界和两个固定时期增量均正，才列开发候选，仍需独立前向。",
        "monitor": "每5日固定相位的成熟q05结果，60条窗口跌穿率超过15%记录警报；无报警不表示有收益优势。",
        "common_information_days": int(x.common_known.sum()), "future_source_perturbation_pass": True,
        "control_information_exact_match": True, "new_accounts": 4, "reused_control_accounts": 8,
        "new_parameter_searches": 0, "new_funding_regression_fits": 0, "new_independent_forward_observations": 0,
        "historical_first_vintage_verified": False, "current_market_view": "NO_VIEW", "goal_achieved": False, "orders_authorized": False}, True)
    paths = [Path(__file__), Path(source.__file__), source.PANEL, source.OUT / "result.json", source.OUT / "extracted_current_quarters.json",
             Path(banking.__file__), banking.OUT / "result.json", banking.OUT / "inputs/decision_information.parquet",
             banking.OUT / "results/predictions.parquet", banking.source.PANEL,
             Path(loan.__file__), Path(shared.__file__), Path(prior.__file__), Path(prior.parent.__file__),
             Path(prior.distribution.__file__), Path(prior.engine.__file__), Path(maturity.__file__),
             ROOT / "research/selected_mix_daily_two_year_v1.py", ROOT / "research/strategy_review_diagnostics_v1.py",
             prior.parent.DIVIDENDS, prior.parent.OUT / "inputs/market.parquet", prior.parent.OUT / "inputs/mature_labels.parquet",
             prior.funding.OUT / "inputs/decision_information.parquet", prior.funding.OUT / "inputs/funding_features.parquet",
             prior.funding.OUT / "results/nested_funding_predictions.parquet"]
    for model in [CONTROL, BANK]:
        for variant in VARIANTS:
            for cost in prior.distribution.COSTS:
                folder = banking.OUT / "accounts" / cost / (model + "__" + variant)
                paths.extend(folder / name for name in ["ledger.parquet", "decisions.parquet", "cycles.parquet", "pending_cycle.json"])
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
        "sources": {path.relative_to(ROOT).as_posix(): digest(path) for path in paths}, "before_new_strategy_returns": True}, True)
    save(OUT / "inputs/mandate.json", read(ROOT / "config/510300_existing_data_training_mandate_v1.json"), True)
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    print(f"企业收款的4个新账户与8个复用对照已固定，共同信息日{x.common_known.sum()}。", flush=True)


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
            print(f"企业收款分布已计算至{x.date.iloc[i].date()}。", flush=True)
    pred, schedule = pd.DataFrame(predictions), pd.DataFrame(receipts)
    pred.to_parquet(OUT / "results/predictions.parquet", index=False)
    schedule.to_parquet(OUT / "results/update_receipts.parquet", index=False)
    save(OUT / "results/saved_distributions.json", models, True)
    verification = shared.adapt(prior.verify_predictions, {"fit_day": fit_day})(pred, models, x, labels, groups, count)
    for model in models:
        lower = (x.date.iloc[model["idx"]] - pd.DateOffset(years=2)).tz_localize("Asia/Shanghai")
        for clock in ["survey_known_at", "enterprise_known_at"]:
            assert x.loc[model["training_indices"], clock].ge(lower).all()
    verification["both_source_clocks_inside_two_years"] = True
    old = pd.read_parquet(banking.OUT / "results/predictions.parquet")
    columns = ["idx", "date", "model", "n_train", "n_selected", "latest_exit_idx", "mu5", "variance5", "q05", "es95"]
    reused_models = ["HISTORY", CONTROL, BANK]
    before = old.loc[old.model.isin(reused_models), columns].sort_values(["idx", "model"]).reset_index(drop=True)
    rebuilt = pred.loc[pred.model.isin(reused_models), columns].sort_values(["idx", "model"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(before, rebuilt, check_exact=True)
    verification["exactly_recomputed_control_distributions"] = len(rebuilt)
    save(OUT / "results/prediction_verification.json", verification, True)
    evaluation, scored, monitor = shared.adapt(prior.forecast_evaluation, {"OUT": OUT, "PRIMARY": SIGNAL, "MODELS": MODELS})(pred, labels, x)
    accounts, checks, cycle_counts, reuse = {}, [], [], []
    for model in TRADED:
        for variant in VARIANTS:
            policy = model + "__" + variant
            for cost in prior.distribution.COSTS:
                folder = OUT / "accounts" / cost / policy
                folder.mkdir(parents=True, exist_ok=True)
                if model == SIGNAL:
                    ledger, decisions = maturity.simulate(market, x, dividends, pred, model, variant, cost)
                    ledger.to_parquet(folder / "ledger.parquet", index=False)
                    decisions.to_parquet(folder / "decisions.parquet", index=False)
                    finished, pending = cycles(ledger)
                    finished.to_parquet(folder / "cycles.parquet", index=False)
                    save(folder / "pending_cycle.json", pending, True)
                    m = metrics(ledger)
                    print(f"企业收款{variant}/{cost}：夏普{m['sharpe']:.6f}、年化{m['annual_return']:.3%}、回撤{abs(m['max_drawdown']):.3%}。", flush=True)
                else:
                    original = banking.OUT / "accounts" / cost / policy
                    for name in ["ledger.parquet", "decisions.parquet", "cycles.parquet", "pending_cycle.json"]:
                        shutil.copy2(original / name, folder / name)
                        assert digest(folder / name) == digest(original / name)
                    ledger = pd.read_parquet(folder / "ledger.parquet")
                    decisions = pd.read_parquet(folder / "decisions.parquet")
                    finished = pd.read_parquet(folder / "cycles.parquet")
                    pending = read(folder / "pending_cycle.json")
                    reuse.append({"policy": policy, "cost": cost, "source": original.relative_to(ROOT).as_posix(),
                                  "ledger_sha256": digest(folder / "ledger.parquet"), "exact_prediction_match": True})
                check = prior.parent.account_verification(ledger, decisions)
                assert not (ledger.shares.shift(1, fill_value=0).gt(0) & ledger.filled_quantity.gt(0)).any()
                accounts[policy, cost] = ledger
                checks.append({"policy": policy, "cost": cost, "account_origin": "NEW" if model == SIGNAL else "REUSED_MATCHED_CONTROL", **check})
                cycle_counts.append({"policy": policy, "cost": cost, "closed": len(finished), "open": pending is not None})
    measures, rolling = summarize_accounts(OUT, market, accounts)
    comparisons, rng = [], np.random.default_rng(2026092611)
    for cost in prior.distribution.COSTS:
        left = accounts[PRIMARY, cost]
        for name, purpose in [(CONTROL, "BANK_AND_COLLECTION_INCREMENT"), (BANK, "SALES_COLLECTION_INCREMENT")]:
            right = accounts[name + "__FIXED5_NO_ADD", cost]
            periods = []
            for lo, hi in [(START, "2023-12-31"), ("2024-01-01", END)]:
                mask = left.date.between(lo, hi)
                periods.append({"start": lo, "end": hi, "annual_arithmetic_difference": float((left.loc[mask, "net_return"] - right.loc[mask, "net_return"]).mean() * 242)})
            interval = loan.release_block_interval(left, right, x.assign(loan_stat_month=x.survey_quarter), rng)
            interval["limitation"] = "按完整季度公布间隔重采样，未保留相邻季度全部持续性，未校正反复研究；不是独立验证。"
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
        "status": "DEVELOPMENT_CANDIDATE_REQUIRES_INDEPENDENT_VALIDATION" if main["historical_point_targets_met"] and increment else "FROZEN_NO_QUALIFIED_SALES_COLLECTION_STRATEGY",
        "primary": main, "all_accounts": measures, "comparisons": comparisons,
        "new_full_accounts": 4, "reused_control_accounts": reuse, "compared_full_accounts": len(accounts),
        "joint_target_pass_new_accounts": sum(bool(m["historical_point_targets_met"]) for m in measures if m["policy"].startswith(SIGNAL)),
        "joint_target_pass_accounts": sum(bool(m["historical_point_targets_met"]) for m in measures),
        "new_conditional_distributions": int(pred.model.eq(SIGNAL).sum()), "recomputed_control_distributions": len(rebuilt),
        "total_compared_distributions": len(pred), "unique_prediction_days": len(ready),
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
    print("企业收款4个新完整账户与8个匹配对照的比较已完成。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="企业收款的两年日更单项增量")
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
