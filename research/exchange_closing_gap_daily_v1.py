"""两个来源一致的GC007收盘报价增量：两年日更、次日使用及延迟对照。"""
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

import research.exchange_bank_funding_gap_daily_v1 as fixing
import research.exchange_repo_quote_crosscheck_v1 as quote
from research.selected_mix_reappraisal_v1 import read, save, now, digest
from research.selected_mix_daily_two_year_v1 import summarize_accounts, paired_interval
from research.strategy_review_diagnostics_v1 import metrics, cycles

prior, maturity = fixing.prior, fixing.maturity
OUT = ROOT / "reports/research/510300_exchange_closing_gap_daily_v1"
STUDY = "510300_EXCHANGE_CLOSING_GAP_DAILY_V1"
START, END = fixing.START, fixing.END
CONTROL = "PRICE_AND_BANK_FUNDING"
SIGNAL = "PRICE_BANK_AND_CLOSING_GAP_LAG1"
DELAYED = "PRICE_BANK_AND_CLOSING_GAP_LAG2"
COMMON = fixing.COMMON
MODELS = {"HISTORY": [], CONTROL: COMMON, SIGNAL: [*COMMON, "closing_gap_lag1_pp"],
          DELAYED: [*COMMON, "closing_gap_lag2_pp"]}
TRADED, VARIANTS = [CONTROL, SIGNAL, DELAYED], fixing.VARIANTS
PRIMARY = SIGNAL + "__FIXED5_NO_ADD"


def information(q=None):
    x = pd.read_parquet(prior.funding.OUT / "inputs/decision_information.parquet")
    if q is None:
        q = pd.read_parquet(quote.OUT / "crosschecked_closing_quotes.parquet", columns=["date", "gc_close_percent"])
    bank = pd.read_parquet(prior.funding.SOURCE / "fixing_segmentation.parquet", columns=["date", "fdr007_percent"])
    both = q.merge(bank, on="date", how="inner", validate="one_to_one")
    both["gap"] = both.gc_close_percent - both.fdr007_percent
    aligned = x[["date", "idx"]].merge(both, on="date", how="left", validate="one_to_one")
    for lag in [1, 2]:
        x[f"closing_gap_lag{lag}_pp"] = aligned.gap.shift(lag)
        x[f"quote_source_date_lag{lag}"] = aligned.date.shift(lag)
        x[f"quote_source_idx_lag{lag}"] = aligned.idx.shift(lag)
        x[f"gc_close_lag{lag}"] = aligned.gc_close_percent.shift(lag)
        x["common_known"] &= x[f"closing_gap_lag{lag}_pp"].notna()
    x["assumed_close_available_at"] = (x.date.shift(1) + pd.Timedelta(hours=23, minutes=59, seconds=59)).dt.tz_localize("Asia/Shanghai")
    x["historical_first_publication_verified"] = False
    assert x.loc[x.common_known, "assumed_close_available_at"].lt(x.loc[x.common_known, "decision_time"]).all()
    for lag in [1, 2]:
        assert (x.loc[x.common_known, "idx"] - x.loc[x.common_known, f"quote_source_idx_lag{lag}"]).eq(lag).all()
    return x


def fit_day(i, x, nested, source_count, labels):
    day = x.date.iloc[i]
    lower = day - pd.DateOffset(years=2)
    receipt = {"idx": i, "date": day, "lower_bound": lower, "n_train": 0,
               "latest_exit_idx": None, "status": "NO_VIEW_SOURCE"}
    if not bool(x.common_known.iloc[i]) or nested is None:
        return [], [], receipt
    innovations = prior.innovation_series(i, x, nested, source_count)
    pool = labels[labels.date.ge(lower) & labels.exit_idx.lt(i)].copy()
    ids = pool.idx.to_numpy(int)
    keep = (x.common_known.iloc[ids].to_numpy(bool) & np.isfinite(innovations[ids])
            & x.quote_source_date_lag2.iloc[ids].ge(lower).to_numpy(bool))
    pool = pool.loc[keep]
    receipt.update(n_train=len(pool), latest_exit_idx=int(pool.exit_idx.max()) if len(pool) else None)
    if len(pool) < 252 or not np.isfinite(innovations[i]):
        return [], [], {**receipt, "status": "NO_VIEW_TRAINING"}
    receipt["status"] = "UPDATED"
    ids = pool.idx.to_numpy(int)
    values = x.loc[ids, [*prior.parent.BASE, "closing_gap_lag1_pp", "closing_gap_lag2_pp"]].copy()
    values["funding_innovation"] = innovations[ids]
    current = {**x.iloc[i].to_dict(), "funding_innovation": innovations[i]}
    predictions, saved = [], []
    for name, columns in MODELS.items():
        mean, scale = np.array([]), np.array([])
        if columns:
            raw = values[columns].to_numpy(float)
            assert np.isfinite(raw).all()
            mean, scale = raw.mean(axis=0), raw.std(axis=0, ddof=1)
            scale[scale < 1e-12] = 1.
            z = np.clip((raw - mean) / scale, -5, 5)
            live = np.clip((np.array([current[c] for c in columns]) - mean) / scale, -5, 5)
            selected = np.lexsort((ids, np.sum((z - live)**2, axis=1)))[:126]
        else:
            selected = np.arange(len(pool))
        chosen = pool.iloc[selected]
        predictions.append({**receipt, "model": name, "n_selected": len(chosen),
                            **prior.distribution.empirical_statistics(chosen.gross_return5)})
        saved.append({"idx": i, "model": name, "features": columns, "mean": mean.tolist(), "scale": scale.tolist(),
                      "training_indices": ids.tolist(), "selected_indices": chosen.idx.tolist(),
                      "current_values": {c: float(current[c]) for c in [*COMMON, "closing_gap_lag1_pp", "closing_gap_lag2_pp"]}})
    return predictions, saved, receipt


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("收盘报价试验已经固定。")
    for folder in ["code", "inputs", "results", "accounts"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    x = information()
    q = pd.read_parquet(quote.OUT / "crosschecked_closing_quotes.parquet", columns=["date", "gc_close_percent"])
    q.loc[q.date.ge("2024-06-28"), "gc_close_percent"] += 70.
    changed = information(q)
    columns = ["closing_gap_lag1_pp", "closing_gap_lag2_pp", "common_known"]
    pd.testing.assert_frame_equal(x.loc[x.date.le("2024-06-28"), columns], changed.loc[changed.date.le("2024-06-28"), columns], check_exact=True)
    source_result = read(quote.OUT / "crosscheck_result.json")
    previous = read(fixing.OUT / "result.json")
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY, "primary": PRIMARY,
         "question": "在实际可观察的收盘报价层面，交易所相对银行的资金价格差能否改善次日以后510300的成本后表现？",
         "new_evidence": source_result,
         "previous_result": "此前带代码的全天定盘试验12账户均失败，主压力夏普" + str(previous["primary"]["sharpe"]),
         "why_distinct": "新增收盘报价观测，不是把上一轮全天定盘提前一天。2017-05-22后官方规则收盘为最后一小时成交量加权，定盘为全天加权，两者保持分开。",
         "feature": "两个提供方一致的原始GC007收盘百分数减同经济日FDR007上午定盘百分数。时段、参与者、抵押品及算法不同；只能称跨市场价格差，不能直接当作纯融资压力或保险费。",
         "clock": "源交易日末假设可取得该日收盘，最早下一A股交易日09:00决策；额外延迟一交易日为敏感性对照。历史首版/送达仍未认证，不能据此晋升真实当前观点。",
         "missing": "2273经济日中2250收盘一致，23个冲突保持未知，两种延迟共用完整信息原点，不择源、不填补。开高低字段亦有不同，但本轮不使用。",
         "calendar": "不沿用上一轮针对定盘年末顺延的黑名单；本轮使用实际收盘报价。所有经济日期执行同一可用规则，并受同池缺资料限制。",
         "models": MODELS, "traded_models": TRADED, "variants": VARIANTS,
         "training": "最近两日历年每天重建；五日标签退出严格早于本次09:00；原始报价日期也在当前两年内；252行起算，固定126近邻，训练内标准化和截断5。银行资金意外只读复用既有严格嵌套结果，不重选参数。",
         "account": "3模型、两期限、两成本共12新账户；20万元510300.SH/CASH_CNY；持仓不加仓；主固定第五个后续开盘退出，每日尾部可缩量，另保留滚动期限对照。",
         "risk_contract": read(ROOT / "config/510300_existing_data_training_mandate_v1.json")["account_tail_risk_contract"],
         "period": [START, END], "annual_days": 242, "cash_and_riskfree_rate": 0,
         "costs": prior.distribution.COSTS, "targets": {"stress_sharpe": 1.2, "stress_cagr": .1, "max_drawdown": .1},
         "comparisons": "主及延迟固定五日各对同池银行资金账户；20日循环区块2000次seed2026092607，固定2020-2023/2024-末端、所有滚动两年。旧定盘账户不直接充当共同样本对照。",
         "development_gate": "主和额外延迟压力版本都过1.2/10%/10%；各自增量95%下界和两个固定时期均正。即使过关还需首版时钟和独立前向。",
         "monitor": "固定每五日一相位，已成熟60个q05突破率大于15%记录警报。零警报不证明正期望。",
         "historical_selection_boundary": "在已见定盘失败后提出，属于开发研究，不是未触碰验证；全历史已经反复被研究。",
         "source_future_perturbation_check": True, "new_full_accounts": 12, "new_parameter_searches": 0,
         "new_funding_regression_fits": 0, "independent_forward_observations": 0,
         "goal_achieved": False, "current_market_view": "NO_VIEW", "orders_authorized": False}, True)
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    save(OUT / "inputs/mandate.json", read(ROOT / "config/510300_existing_data_training_mandate_v1.json"), True)
    dependencies = read(fixing.OUT / "freeze.json")["sources"]
    paths = [ROOT / p for p in dependencies if "exchange_repo_fixing_source_v1" not in p]
    paths += [Path(__file__), quote.OUT / "crosscheck_result.json", quote.OUT / "crosschecked_closing_quotes.parquet", fixing.OUT / "result.json"]
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
         "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}, "before_new_closing_quote_returns": True}, True)
    print(f"收盘报价12账户已固定，共同信息日{x.common_known.sum()}；未复用或提前全天定盘。", flush=True)


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
            print(f"收盘资金价格差分布更新至{x.date.iloc[i].date()}。", flush=True)
    pred, schedule = pd.DataFrame(predictions), pd.DataFrame(receipts)
    pred.to_parquet(OUT / "results/predictions.parquet", index=False)
    schedule.to_parquet(OUT / "results/update_receipts.parquet", index=False)
    save(OUT / "results/saved_distributions.json", models, True)
    verification = fixing.adapt(prior.verify_predictions, {"fit_day": fit_day})(pred, models, x, labels, groups, count)
    for model in models:
        assert x.loc[model["training_indices"], "quote_source_date_lag2"].ge(x.date.iloc[model["idx"]] - pd.DateOffset(years=2)).all()
    verification["quote_raw_dates_inside_current_two_year_window"] = True
    save(OUT / "results/prediction_verification.json", verification, True)
    evaluation, scored, monitor = fixing.adapt(prior.forecast_evaluation, {"OUT": OUT, "PRIMARY": SIGNAL, "MODELS": MODELS})(pred, labels, x)
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
                sh = "未定义" if m["sharpe"] is None else f"{m['sharpe']:.6f}"
                print(f"{policy}/{cost}：夏普{sh}、年化{m['annual_return']:.3%}、回撤{abs(m['max_drawdown']):.3%}。", flush=True)
    measures, rolling = summarize_accounts(OUT, market, accounts)
    comparisons = []
    rng = np.random.default_rng(2026092607)
    for cost in prior.distribution.COSTS:
        for name in [SIGNAL, DELAYED]:
            left, right = accounts[name + "__FIXED5_NO_ADD", cost], accounts[CONTROL + "__FIXED5_NO_ADD", cost]
            eras = []
            for lo, hi in [(START, "2023-12-31"), ("2024-01-01", END)]:
                mask = left.date.between(lo, hi)
                eras.append({"start": lo, "end": hi, "annual_arithmetic_difference": float((left.loc[mask, "net_return"] - right.loc[mask, "net_return"]).mean() * 242)})
            comparisons.append({"cost": cost, "candidate": name, "control": CONTROL,
                                **paired_interval(left, right, rng), "fixed_periods": eras})
    main = next(m for m in measures if m["policy"] == PRIMARY and m["cost"] == "STRESS")
    delayed = next(m for m in measures if m["policy"] == DELAYED + "__FIXED5_NO_ADD" and m["cost"] == "STRESS")
    increment = all(r["lower_95"] > 0 and all(e["annual_arithmetic_difference"] > 0 for e in r["fixed_periods"])
                    for r in comparisons if r["cost"] == "STRESS")
    development_pass = main["historical_point_targets_met"] and delayed["historical_point_targets_met"] and increment
    ledger = accounts[PRIMARY, "STRESS"]
    gross = float((ledger.price_pnl + ledger.dividend_recognized).sum())
    expense = float((ledger.commission + ledger.slippage_cost).sum() + ledger.terminal_exit_reserve.iloc[-1])
    np.testing.assert_allclose(gross - expense, ledger.equity.iloc[-1] - 200000, atol=1e-6, rtol=0)
    ready, updated = pred[["idx", "date"]].drop_duplicates(), schedule[schedule.status.eq("UPDATED")]
    save(OUT / "result.json", {"at": now(), "study_id": STUDY,
         "status": "DEVELOPMENT_CANDIDATE_REQUIRES_CLOCK_AND_FORWARD_VALIDATION" if development_pass else "FROZEN_NO_QUALIFIED_CLOSING_GAP_STRATEGY",
         "primary": main, "extra_delay_diagnostic": delayed, "all_accounts": measures, "comparisons": comparisons,
         "forecast_evaluation": evaluation, "development_increment_gate": increment,
         "new_full_accounts": len(accounts), "new_conditional_distributions": len(pred), "new_agreed_source_dates": 2250,
         "unique_prediction_days": len(ready), "first_prediction_date": ready.date.min(), "last_prediction_date": ready.date.max(),
         "mature_prediction_days": int(scored[scored.gross_return5.notna()].idx.nunique()),
         "training_rows_min": int(updated.n_train.min()), "training_rows_max": int(updated.n_train.max()),
         "schedule_statuses": schedule.status.value_counts().to_dict(),
         "primary_cash_attribution": {"gross_pnl": gross, "cost_and_reserve": expense, "net_profit": float(ledger.equity.iloc[-1] - 200000)},
         "primary_rolling_two_year_joint_passes": sum(r["joint_point_pass"] for r in rolling if r["policy"] == PRIMARY and r["cost"] == "STRESS"),
         "mature_monitor": [{"model": name, "mature_nonoverlap_rows": len(part), "assessable_rows": int(part.monitor_assessable.sum()), "alerts": int(part.alert.sum())}
                            for name, part in monitor.groupby("model")],
         "prediction_checks": verification, "account_checks": checks, "cycle_counts": cycle_counts,
         "new_parameter_searches": 0, "new_funding_regression_fits": 0,
         "historical_first_publication_verified": False, "new_independent_forward_observations": 0,
         "current_market_view": "NO_VIEW", "goal_status": "active", "goal_achieved": False,
         "orders_authorized": False, "review_package_created": False}, True)
    print("收盘报价12个完整账户及延迟增量比较已完成。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GC007真实收盘价差条件研究")
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
