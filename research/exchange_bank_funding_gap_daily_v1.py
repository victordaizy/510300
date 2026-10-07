"""交易所与银行资金价格差：同池、两年日更和延迟敏感性的12个账户。"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from types import FunctionType

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.exchange_repo_fixing_source_v1 as source
import research.funding_repurchase_demand_daily_v1 as prior
import research.funding_forecast_maturity_account_v1 as maturity
from research.selected_mix_reappraisal_v1 import read, save, now, digest
from research.selected_mix_daily_two_year_v1 import summarize_accounts, paired_interval
from research.strategy_review_diagnostics_v1 import metrics, cycles

OUT = ROOT / "reports/research/510300_exchange_bank_funding_gap_daily_v1"
STUDY = "510300_EXCHANGE_BANK_FUNDING_GAP_DAILY_V1"
START, END = prior.START, prior.END
CONTROL = "PRICE_AND_BANK_FUNDING"
SIGNAL = "PRICE_BANK_AND_EXCHANGE_GAP_LAG2"
DELAYED = "PRICE_BANK_AND_EXCHANGE_GAP_LAG5"
COMMON = [*prior.parent.BASE, "funding_innovation"]
MODELS = {"HISTORY": [], CONTROL: COMMON, SIGNAL: [*COMMON, "gap_lag2_pp"],
          DELAYED: [*COMMON, "gap_lag5_pp"]}
TRADED = [CONTROL, SIGNAL, DELAYED]
VARIANTS = ["ROLL_NO_ADD", "FIXED5_NO_ADD"]
PRIMARY = SIGNAL + "__FIXED5_NO_ADD"


def adapt(function, changes):
    return FunctionType(function.__code__, {**function.__globals__, **changes}, function.__name__,
                        function.__defaults__, function.__closure__)


def information(gc=None):
    x = pd.read_parquet(prior.funding.OUT / "inputs/decision_information.parquet")
    if gc is None:
        gc = pd.read_parquet(source.OUT / "results/gc007_fixing.parquet", columns=["date", "gc007_percent"])
    bank = pd.read_parquet(prior.funding.SOURCE / "fixing_segmentation.parquet", columns=["date", "fdr007_percent"])
    paired = gc.merge(bank, on="date", how="inner", validate="one_to_one")
    paired["exchange_bank_gap_pp"] = paired.gc007_percent - paired.fdr007_percent
    aligned = x[["date", "idx"]].merge(paired, on="date", how="left", validate="one_to_one")
    x["bank_information_known"] = x.common_known
    for lag in [2, 5]:
        x[f"gap_lag{lag}_pp"] = aligned.exchange_bank_gap_pp.shift(lag)
        x[f"gc_source_date_lag{lag}"] = aligned.date.shift(lag)
        x[f"gc_source_idx_lag{lag}"] = aligned.idx.shift(lag)
        x[f"gc_percent_lag{lag}"] = aligned.gc007_percent.shift(lag)
        x[f"paired_fdr_percent_lag{lag}"] = aligned.fdr007_percent.shift(lag)
        day = x[f"gc_source_date_lag{lag}"]
        year_edge = (day.dt.month.eq(12) & day.dt.day.ge(24)) | (day.dt.month.eq(1) & day.dt.day.le(7))
        x[f"gc_year_edge_excluded_lag{lag}"] = year_edge
        x["common_known"] &= x[f"gap_lag{lag}_pp"].notna() & ~year_edge
    # 这是研究假设的最迟读取时点，绝非历史实际发布时间或不可变送达记录。
    x["assumed_readiness_at"] = (x.date.shift(1) + pd.Timedelta(hours=23, minutes=59, seconds=59)).dt.tz_localize("Asia/Shanghai")
    x["historical_publication_verified"] = False
    assert x.idx.tolist() == list(range(len(x)))
    assert x.loc[x.common_known, "assumed_readiness_at"].lt(x.loc[x.common_known, "decision_time"]).all()
    for lag in [2, 5]:
        assert (x.loc[x.common_known, "idx"] - x.loc[x.common_known, f"gc_source_idx_lag{lag}"]).eq(lag).all()
        np.testing.assert_allclose(x[f"gap_lag{lag}_pp"], x[f"gc_percent_lag{lag}"] - x[f"paired_fdr_percent_lag{lag}"], equal_nan=True)
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
            & x.gc_source_date_lag5.iloc[ids].ge(lower).to_numpy(bool))
    pool = pool.loc[keep]
    receipt.update(n_train=len(pool), latest_exit_idx=int(pool.exit_idx.max()) if len(pool) else None)
    if len(pool) < 252 or not np.isfinite(innovations[i]):
        return [], [], {**receipt, "status": "NO_VIEW_TRAINING"}
    receipt["status"] = "UPDATED"
    ids = pool.idx.to_numpy(int)
    values = x.loc[ids, [*prior.parent.BASE, "gap_lag2_pp", "gap_lag5_pp"]].copy()
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
                      "current_values": {c: float(current[c]) for c in [*COMMON, "gap_lag2_pp", "gap_lag5_pp"]}})
    return predictions, saved, receipt


def verify_mapping():
    raw = pd.read_parquet(source.OUT / "results/gc007_fixing.parquet", columns=["date", "gc007_percent"])
    x = information(raw)
    cutoff = pd.Timestamp("2024-06-28")
    changed = raw.copy()
    changed.loc[changed.date.ge(cutoff), "gc007_percent"] += 70.
    after = information(changed)
    # 当前日和未来源数据都不影响截至当天已经形成的输入。
    cols = ["gap_lag2_pp", "gap_lag5_pp", "common_known"]
    pd.testing.assert_frame_equal(x.loc[x.date.le(cutoff), cols], after.loc[after.date.le(cutoff), cols], check_exact=True)
    return {"current_and_future_source_mutation_does_not_change_past_inputs": True,
            "exact_prior_session_mapping_checked": [2, 5], "no_source_forward_fill": True,
            "common_information_days": int(x.common_known.sum())}


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("本轮规则已固定，禁止覆盖。")
    checks = verify_mapping()
    source_result = read(source.OUT / "result.json")
    assert source_result["rows"] == 2272 and not source_result["missing_market_dates"]
    for folder in ["code", "inputs", "results", "accounts"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY, "primary": PRIMARY,
         "question": "相同价格背景和银行资金意外下，交易所相对银行的资金价格差，能否提供新的修复或持续下跌信息？",
         "single_new_information": "同经济日GC007定盘减FDR007定盘，百分点；GC007为全天成交量加权、FDR007为上午子样本基准，参与者、抵押品、时段和算法均不同。不能称纯非银成本、纯流动性供给冲击或中性利率差。",
         "hypothesis": "更高的跨市场利差可能反映交易所融资压力，也可能是日历需求和交易活跃的内生结果；不预定股票收益方向，由共同成熟训练池估计。",
         "source": source_result,
         "clock_scenarios": {"LAG2": "第s个交易日的定盘假设在s+1交易日末可取得，最早s+2开盘决策；这是未认证的时点重建。",
                             "LAG5": "再推迟三个交易日，在s+5开盘才使用；属于延迟敏感性，不认证逐日首版。",
                             "year_edge": "经济日期在12月24日至次年1月7日的定盘两种版本都不使用，以排除旧公布规则的跨年延迟边界；自然日条件固定，未查看收益。",
                             "common_pool": "两种延迟的输入都存在才允许比较；不向前填缺值。滞后使用截至当前已发生的A股交易日。源时间早于当前两年下界的训练原点再排除。"},
         "funding_control": "只读复用FDR007日历意外；每个当前两年窗内曾重新嵌套拟合，只用此前利率，不携两年前参数。没有新增资金回归或修改旧失败账户。",
         "models": MODELS, "traded_models": TRADED, "variants": VARIANTS,
         "training": "最近两日历年、五日标签退出严格早于当前开盘、至少252共同原点、训练内均值与样本标准差标准化和截断5、固定126近邻；每天更新。HISTORY为相同训练池。",
         "execution": "空仓时仅价格前五日下跌允许买入，全部账户持仓期间不加仓。主方案第五个后续交易日开盘退出、每日尾部可提前减仓；ROLL_NO_ADD为原滚动均值对照。复用成熟账户引擎。",
         "account": "20万元、510300.SH/CASH_CNY、100份、tick0.001、T+1、现金股息分账。最大仓位50%、ES95五日2.5%、-10%跳空损失预算不超过5%权益且只用一半回撤余量，调整/退出费用计入；10%回撤后本轮不恢复。",
         "period": [START, END], "annual_days": 242, "cash_and_riskfree_rate": 0,
         "costs": prior.distribution.COSTS, "targets": {"stress_sharpe": 1.2, "stress_cagr": .1, "max_drawdown": .1},
         "comparisons": "两个延迟分别对同期限银行资金对照；主方案再对滚动期限比较。20日循环区块2000次seed2026092606、完整共同账户、固定2020-2023/2024-末端和全部滚动两年。",
         "development_gate": "主压力和更迟版本均过三目标；各自相对同期限银行资金对照的增量95%下界正、两个固定时期均正。即使通过仍须补时钟来源和独立前向。",
         "monitor": "五个交易日固定相位的成熟预测，最近60条q05跌穿率大于15%记录；不足60条不可评价。零警报不等于优势存在。",
         "history_previously_used": True, "historical_first_publication_verified": False,
         "new_accounts": 12, "new_parameter_searches": 0, "new_funding_fits": 0,
         "independent_forward_observations": 0, "implementation_checks": checks,
         "current_market_view": "NO_VIEW", "goal_achieved": False, "orders_authorized": False}, True)
    save(OUT / "inputs/mandate.json", read(ROOT / "config/510300_existing_data_training_mandate_v1.json"), True)
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    paths = [Path(__file__), Path(prior.__file__), Path(prior.parent.__file__), Path(prior.funding.__file__),
             Path(prior.engine.__file__), Path(prior.distribution.__file__), Path(maturity.__file__), prior.parent.DIVIDENDS,
             ROOT / "research/selected_mix_daily_two_year_v1.py", ROOT / "research/strategy_review_diagnostics_v1.py",
             source.OUT / "result.json", source.OUT / "results/gc007_fixing.parquet",
             prior.parent.OUT / "inputs/market.parquet", prior.parent.OUT / "inputs/mature_labels.parquet",
             prior.funding.OUT / "inputs/decision_information.parquet", prior.funding.OUT / "inputs/funding_features.parquet",
             prior.funding.OUT / "results/nested_funding_predictions.parquet", prior.funding.SOURCE / "fixing_segmentation.parquet"]
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
         "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}, "before_new_strategy_returns": True}, True)
    print(f"跨市场资金差12账户已固定；共同信息日{checks['common_information_days']}，不搜索方向或参数。", flush=True)


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
    funding_count = len(pd.read_parquet(prior.funding.OUT / "inputs/funding_features.parquet", columns=["source_idx"]))
    nested = pd.read_parquet(prior.funding.OUT / "results/nested_funding_predictions.parquet")
    groups = {int(i): part for i, part in nested.groupby("decision_idx", sort=False)}
    predictions, models, receipts = [], [], []
    for i in np.flatnonzero(x.date.ge(START)):
        p, m, r = fit_day(int(i), x, groups.get(int(i)), funding_count, labels)
        predictions.extend(p)
        models.extend(m)
        receipts.append(r)
        if len(receipts) % 300 == 0:
            print(f"跨市场资金差分布更新至{x.date.iloc[i].date()}。", flush=True)
    pred, schedule = pd.DataFrame(predictions), pd.DataFrame(receipts)
    pred.to_parquet(OUT / "results/predictions.parquet", index=False)
    schedule.to_parquet(OUT / "results/update_receipts.parquet", index=False)
    save(OUT / "results/saved_distributions.json", models, True)
    verification = adapt(prior.verify_predictions, {"fit_day": fit_day})(pred, models, x, labels, groups, funding_count)
    for model in models:
        lower = x.date.iloc[model["idx"]] - pd.DateOffset(years=2)
        assert x.loc[model["training_indices"], "gc_source_date_lag5"].ge(lower).all()
    verification["all_gc_source_dates_inside_current_two_year_window"] = True
    save(OUT / "results/prediction_verification.json", verification, True)
    evaluation, scored, monitor = adapt(prior.forecast_evaluation, {"OUT": OUT, "PRIMARY": SIGNAL, "MODELS": MODELS})(pred, labels, x)
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
                closed, pending = cycles(ledger)
                closed.to_parquet(folder / "cycles.parquet", index=False)
                save(folder / "pending_cycle.json", pending, True)
                accounts[policy, cost] = ledger
                checks.append({"policy": policy, "cost": cost, **check})
                cycle_counts.append({"policy": policy, "cost": cost, "closed": len(closed), "open": pending is not None})
                m = metrics(ledger)
                sh = "未定义" if m["sharpe"] is None else f"{m['sharpe']:.6f}"
                print(f"{policy}/{cost}：夏普{sh}，年化{m['annual_return']:.3%}，回撤{abs(m['max_drawdown']):.3%}。", flush=True)
    measures, rolling = summarize_accounts(OUT, market, accounts)
    comparisons = []
    rng = np.random.default_rng(2026092606)
    for cost in prior.distribution.COSTS:
        for name in [SIGNAL, DELAYED]:
            left, right = accounts[name + "__FIXED5_NO_ADD", cost], accounts[CONTROL + "__FIXED5_NO_ADD", cost]
            eras = []
            for lo, hi in [(START, "2023-12-31"), ("2024-01-01", END)]:
                mask = left.date.between(lo, hi)
                eras.append({"start": lo, "end": hi, "annual_arithmetic_difference": float((left.loc[mask, "net_return"] - right.loc[mask, "net_return"]).mean() * 242)})
            comparisons.append({"cost": cost, "candidate": name, "control": CONTROL,
                                "purpose": "INFORMATION", **paired_interval(left, right, rng), "fixed_periods": eras})
        comparisons.append({"cost": cost, "candidate": PRIMARY, "control": SIGNAL + "__ROLL_NO_ADD",
                            "purpose": "EXECUTION", **paired_interval(accounts[PRIMARY, cost], accounts[SIGNAL + "__ROLL_NO_ADD", cost], rng)})
    main = next(m for m in measures if m["policy"] == PRIMARY and m["cost"] == "STRESS")
    delayed = next(m for m in measures if m["policy"] == DELAYED + "__FIXED5_NO_ADD" and m["cost"] == "STRESS")
    increment = all(r["lower_95"] > 0 and all(e["annual_arithmetic_difference"] > 0 for e in r["fixed_periods"])
                    for r in comparisons if r["cost"] == "STRESS" and r["purpose"] == "INFORMATION")
    development_pass = main["historical_point_targets_met"] and delayed["historical_point_targets_met"] and increment
    ledger = accounts[PRIMARY, "STRESS"]
    gross = float((ledger.price_pnl + ledger.dividend_recognized).sum())
    expense = float((ledger.commission + ledger.slippage_cost).sum() + ledger.terminal_exit_reserve.iloc[-1])
    np.testing.assert_allclose(gross - expense, ledger.equity.iloc[-1] - 200000, atol=1e-6, rtol=0)
    ready, updated = pred[["idx", "date"]].drop_duplicates(), schedule[schedule.status.eq("UPDATED")]
    result = {"at": now(), "study_id": STUDY,
              "status": "DEVELOPMENT_CANDIDATE_REQUIRES_CLOCK_AND_FORWARD_VALIDATION" if development_pass else "FROZEN_NO_QUALIFIED_EXCHANGE_BANK_GAP_STRATEGY",
              "primary": main, "extra_delay_diagnostic": delayed, "all_accounts": measures, "comparisons": comparisons,
              "new_source_rows": 2272, "new_full_accounts": len(accounts), "new_conditional_distributions": len(pred),
              "unique_prediction_days": len(ready), "first_prediction_date": ready.date.min(), "last_prediction_date": ready.date.max(),
              "mature_prediction_days": int(scored[scored.gross_return5.notna()].idx.nunique()),
              "training_rows_min": int(updated.n_train.min()), "training_rows_max": int(updated.n_train.max()),
              "common_information_days": int(x.common_known.sum()), "schedule_statuses": schedule.status.value_counts().to_dict(),
              "development_increment_gate": increment, "forecast_evaluation": evaluation,
              "primary_cash_attribution": {"gross_pnl": gross, "cost_and_reserve": expense, "net_profit": float(ledger.equity.iloc[-1] - 200000)},
              "primary_rolling_two_year_joint_passes": sum(r["joint_point_pass"] for r in rolling if r["policy"] == PRIMARY and r["cost"] == "STRESS"),
              "mature_monitor": [{"model": name, "mature_nonoverlap_rows": len(part), "assessable_rows": int(part.monitor_assessable.sum()), "alerts": int(part.alert.sum())}
                                 for name, part in monitor.groupby("model")],
              "prediction_checks": verification, "account_checks": checks, "cycle_counts": cycle_counts,
              "new_parameter_searches": 0, "new_funding_regression_fits": 0,
              "historical_first_publication_verified": False, "new_independent_forward_observations": 0,
              "current_market_view": "NO_VIEW", "goal_status": "active", "goal_achieved": False,
              "orders_authorized": False, "review_package_created": False}
    save(OUT / "result.json", result, True)
    print("跨市场资金价格差12个账户、延迟比较和成熟预测监控已完成。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="交易所与银行资金价差的两年日更账户")
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
