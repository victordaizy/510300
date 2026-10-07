"""成本传导环境的一项新增信息，拆分收益与五日尾部预算的账户作用。"""
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

import research.manufacturing_price_transmission_source_v1 as source
import research.housing_collateral_daily_v1 as mechanics
from research.selected_mix_reappraisal_v1 import read, save, now, digest
from research.selected_mix_daily_two_year_v1 import summarize_accounts, paired_interval
from research.strategy_review_diagnostics_v1 import metrics, cycles

prior, shared, loan = mechanics.prior, mechanics.shared, mechanics.loan
parent, engine, distribution = mechanics.parent, mechanics.engine, mechanics.distribution
OUT = ROOT / "reports/research/510300_manufacturing_price_transmission_daily_v1"
STUDY = "510300_MANUFACTURING_PRICE_TRANSMISSION_DAILY_V1"
START, END = mechanics.START, mechanics.END
FEATURE = "output_minus_input_diffusion_pp"
CONTROL, SIGNAL = "PRICE_AND_FUNDING", "PRICE_FUNDING_AND_PRICE_TRANSMISSION"
MODELS = {"HISTORY": [], CONTROL: [*parent.BASE, "funding_innovation"], SIGNAL: [*parent.BASE, "funding_innovation", FEATURE]}
MAIN_SCOPE, METHOD_SCOPE = "PUBLISHED_MEASUREMENT", "SAME_METHOD_ONLY"
SCOPES = [MAIN_SCOPE, METHOD_SCOPE]
ARMS = {"BASE": (CONTROL, CONTROL), "MEAN_ONLY": (SIGNAL, CONTROL), "TAIL_ONLY": (CONTROL, SIGNAL), "JOINT": (SIGNAL, SIGNAL)}
HORIZONS = [5, 20]
PRIMARY = "JOINT__FIXED20_NO_ADD"


def policy_name(model, scope, horizon):
    assert scope == MAIN_SCOPE and horizon == 20
    return f"{model}__FIXED20_NO_ADD"


simulate = shared.adapt(mechanics.simulate, {"policy_name": policy_name})


def information(releases=None):
    x = pd.read_parquet(prior.funding.OUT / "inputs/decision_information.parquet")
    q = (pd.read_parquet(source.PANEL) if releases is None else releases).copy()
    q["known_at"] = pd.to_datetime(q.known_at).dt.tz_convert("Asia/Shanghai").dt.as_unit("ns")
    q = q[["known_at", "stat_month", "method_group", FEATURE]].rename(columns={"known_at": "pricing_known_at", "stat_month": "pricing_stat_month"})
    assert not q.pricing_known_at.duplicated().any()
    x = pd.merge_asof(x.sort_values("decision_time"), q.sort_values("pricing_known_at"), left_on="decision_time", right_on="pricing_known_at", direction="backward")
    x["pricing_age_days"] = (x.decision_time - x.pricing_known_at).dt.total_seconds() / 86400
    x["bank_information_known"] = x.common_known
    x["common_known"] &= x.pricing_age_days.between(0, 45) & x[FEATURE].notna()
    assert x.loc[x.common_known, "pricing_known_at"].lt(x.loc[x.common_known, "decision_time"]).all()
    assert x.idx.tolist() == list(range(len(x)))
    return x


def fit_day(i, x, nested, count, labels, scope):
    day, lower = x.date.iloc[i], x.date.iloc[i] - pd.DateOffset(years=2)
    receipt = {"idx": i, "date": day, "scope": scope, "lower_bound": lower, "status": "NO_VIEW_SOURCE", "n_train": 0}
    if not bool(x.common_known.iloc[i]) or nested is None:
        return [], [], receipt
    innovations = prior.innovation_series(i, x, nested, count)
    pool = labels[labels.date.ge(lower) & labels.exit_idx.lt(i)].copy()
    ids = pool.idx.to_numpy(int)
    ok = x.common_known.iloc[ids].to_numpy(bool) & np.isfinite(innovations[ids])
    ok &= x.pricing_known_at.iloc[ids].ge(lower.tz_localize("Asia/Shanghai")).to_numpy(bool)
    if scope == METHOD_SCOPE:
        ok &= x.method_group.iloc[ids].eq(x.method_group.iloc[i]).to_numpy(bool)
    pool = pool.loc[ok]
    ids = pool.idx.to_numpy(int)
    receipt.update(n_train=len(pool), latest_exit_idx=int(pool.exit_idx.max()) if len(pool) else None,
        release_months=int(x.pricing_stat_month.iloc[ids].nunique()), training_methods=sorted(x.method_group.iloc[ids].unique().tolist()))
    if len(pool) < 252 or not np.isfinite(innovations[i]):
        receipt["status"] = "NO_VIEW_TRAINING"
        return [], [], receipt
    receipt["status"] = "UPDATED"
    values = x.loc[ids, [*parent.BASE, FEATURE]].copy()
    values["funding_innovation"] = innovations[ids]
    current = {**x.iloc[i].to_dict(), "funding_innovation": innovations[i]}
    rows, saved = [], []
    for model, columns in MODELS.items():
        mean, scale = np.array([]), np.array([])
        if columns:
            raw = values[columns].to_numpy(float)
            assert np.isfinite(raw).all()
            mean, scale = raw.mean(axis=0), raw.std(axis=0, ddof=1)
            scale[scale < 1e-12] = 1.
            z = np.clip((raw - mean) / scale, -5, 5)
            live = np.clip((np.array([current[c] for c in columns]) - mean) / scale, -5, 5)
            chosen = pool.iloc[np.lexsort((ids, np.sum((z - live) ** 2, axis=1)))[:126]]
        else:
            chosen = pool
        record = {**receipt, "model": model, "n_selected": len(chosen)}
        for h in HORIZONS:
            stats = distribution.empirical_statistics(chosen[f"gross_return{h}"])
            record.update({f"mu_{h}": stats["mu5"], f"variance_{h}": stats["variance5"], f"q05_{h}": stats["q05"], f"es95_{h}": stats["es95"]})
        rows.append(record)
        saved.append({"idx": i, "scope": scope, "model": model, "features": columns,
            "training_indices": ids.tolist(), "selected_indices": chosen.idx.tolist(), "mean": mean.tolist(), "scale": scale.tolist(),
            "current_values": {c: float(current[c]) for c in MODELS[SIGNAL]}})
    return rows, saved, receipt


def account_predictions(pred):
    """收益均值/方差与五日尾部模块按事前固定2×2组合，不拼收益路径。"""
    active = pred[pred.scope.eq(MAIN_SCOPE)].copy()
    by_model = {name: frame.set_index("idx", drop=False).sort_index() for name, frame in active.groupby("model")}
    assert by_model[CONTROL].index.equals(by_model[SIGNAL].index)
    rows = []
    for arm, (mean_model, tail_model) in ARMS.items():
        frame = by_model[mean_model].copy()
        frame["model"], frame["mean_source"], frame["tail_source"] = arm, mean_model, tail_model
        for h in HORIZONS:
            frame[f"q05_{h}"] = by_model[tail_model][f"q05_{h}"]
            frame[f"es95_{h}"] = by_model[tail_model][f"es95_{h}"]
        rows.append(frame.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True)


def implementation_checks():
    dates = pd.bdate_range("2021-01-04", periods=900)
    n, i = len(dates), len(dates) - 1
    x = pd.DataFrame({"idx": np.arange(n), "date": dates, "source_idx": np.arange(n), "common_known": True,
        "pricing_known_at": dates.tz_localize("Asia/Shanghai") - pd.Timedelta(days=1), "pricing_stat_month": dates.strftime("%Y-%m"),
        "method_group": np.where(np.arange(n) >= 700, "NEW", "OLD")})
    for j, column in enumerate([*parent.BASE, FEATURE]):
        x[column] = np.sin(np.arange(n) / (j + 11))
    labels = pd.DataFrame({"idx": np.arange(n - 20), "date": dates[:-20], "exit_idx": np.arange(n - 20) + 20,
        "gross_return5": np.sin(np.arange(n - 20)) / 100, "gross_return20": np.cos(np.arange(n - 20)) / 20})
    nested = pd.DataFrame({"decision_idx": i, "source_idx": np.arange(n), "actual": np.sin(np.arange(n) / 17), "CALENDAR": 0.})
    before, saved, receipt = fit_day(i, x, nested, n, labels, MAIN_SCOPE)
    assert receipt["status"] == "UPDATED"
    changed = labels.copy()
    invalid = changed.exit_idx.ge(i) | changed.date.lt(dates[-1] - pd.DateOffset(years=2))
    changed.loc[invalid, ["gross_return5", "gross_return20"]] = -9999.
    after, rebuilt, _ = fit_day(i, x, nested, n, changed, MAIN_SCOPE)
    assert before == after and saved == rebuilt
    strict, _, r = fit_day(i, x, nested, n, labels, METHOD_SCOPE)
    assert not strict and r["status"] == "NO_VIEW_TRAINING"
    composed = account_predictions(pd.DataFrame(before)).set_index("model")
    original = {r["model"]: r for r in before}
    for arm, (mean_model, tail_model) in ARMS.items():
        for h in HORIZONS:
            assert composed.loc[arm, f"mu_{h}"] == original[mean_model][f"mu_{h}"]
            assert composed.loc[arm, f"variance_{h}"] == original[mean_model][f"variance_{h}"]
            assert composed.loc[arm, f"es95_{h}"] == original[tail_model][f"es95_{h}"]
    return {"future_and_expired_labels_ignored": True, "same_method_insufficient_returns_no_view": True,
        "four_account_input_combinations_exact": True, "reused_execution_checks": mechanics.implementation_checks()}


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("成本传导试验已固定，不覆盖。")
    sr = read(source.OUT / "result.json")
    assert sr["status"] == "COMPLETE_MANUFACTURING_PRICE_TRANSMISSION_SOURCE_READY" and sr["admitted_months"] == 116
    checks, x = implementation_checks(), information()
    q = pd.read_parquet(source.PANEL)
    cutoff = pd.Timestamp("2024-06-28", tz="Asia/Shanghai")
    changed = q.copy()
    changed.loc[changed.known_at.ge(cutoff), FEATURE] += 700
    pd.testing.assert_frame_equal(x.loc[x.decision_time.lt(cutoff), [FEATURE, "common_known"]],
        information(changed).loc[x.decision_time.lt(cutoff), [FEATURE, "common_known"]], check_exact=True)
    for folder in ["inputs", "results", "code", "accounts"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY, "primary": PRIMARY,
        "question": "采购成本涨价普遍程度相对出厂涨价普遍程度，是否改变价格压力后的20日收益与五日尾部风险？新增宏观信息的账户作用来自收益判断、风险预算，还是二者的相互影响？",
        "mechanism": "出厂减购进的扩散指数差可能表示成本传导环境；它不等于实际利润率、通胀幅度、市场预期差或制造业对沪深300的贡献。正负方向由当时条件分布估计，不事后反号。",
        "feature": FEATURE, "source": sr, "source_clock": "只用当期原报告本月行，统一原公布日日末可用，45自然日有效。2015—2016未发布出厂价格，不从后来报告回填。",
        "method_sensitivity": "主研究使用实际公布同名季调指标；2019行业分类、2022样本扩充全部保留。另每天计算只用同方法组历史的预测，少于252时NO_VIEW；敏感性只评估预测，不产生额外账户。",
        "training": "最近两日历年，来源公布也在两年内；20日退出开盘严格早于决策日。5日/20日共用同一成熟池，252行最低，126近邻，训练内标准化clip5；不搜参数。",
        "models": MODELS, "account_arms": ARMS,
        "factorial": "BASE全部价格资金预测；MEAN_ONLY只把20日均值/方差换为新增信息预测；TAIL_ONLY只把每日五日ES预算换为新增信息预测；JOINT两者都换。每个账户独立连续执行，不合并盈利路径。",
        "horizon": "所有账户固定第20个后续开盘到期；持仓不加仓，只因当日五日风险缩量。缺预测满5日退出。期限按慢速经营渠道事前固定，本轮不搜索其他期限。",
        "risk": read(ROOT / "config/510300_existing_data_training_mandate_v1.json")["account_tail_risk_contract"],
        "account": "四组合各BASE/STRESS费用，共8个完整账户；20万元，只510300/CASH_CNY，T+1、100份、开盘只能缩量、现金分红分账。",
        "period": [START, END], "annual_days": 242, "cash_and_risk_free": 0, "costs": distribution.COSTS,
        "targets": {"stress_sharpe": 1.2, "stress_cagr": .1, "max_drawdown": .1},
        "comparisons": "MEAN_ONLY、TAIL_ONLY、JOINT各减BASE，并报告JOINT-MEAN_ONLY-TAIL_ONLY+BASE的非可交易交互诊断；20日循环和完整月报公布间隔各2000次seed2026092614；固定2020-2023/2024-末端。",
        "gate": "主JOINT压力账户三目标同时通过；JOINT-BASE两种区块95%下界均正且两个固定时期均正；相同方法预测敏感性的20日MSE和五日分位损失均不差于其价格资金对照。之后仍需独立前向。",
        "monitor": "固定五日及20日相位，只纳入成熟结果；60个非重叠样本，q05跌穿率>15%记警报。未满60明确不可评估；不依据本轮警报修改账户。",
        "information_limits": "相邻日标签重叠，同月信息重复；区块未校正全部既有研究的选择，宏观有效样本按实际公布月数理解。",
        "common_information_days": int(x.common_known.sum()), "implementation_checks": checks,
        "new_accounts": 8, "new_parameter_searches": 0, "new_funding_regression_fits": 0,
        "new_independent_forward_observations": 0, "historical_first_vintage_verified": False,
        "goal_achieved": False, "orders_authorized": False, "current_market_view": "NO_VIEW"}, True)
    paths = [Path(__file__), Path(source.__file__), source.PANEL, source.OUT / "result.json", source.OUT / "released_records.json",
        Path(mechanics.__file__), Path(mechanics.maturity.__file__), Path(prior.__file__), Path(parent.__file__),
        Path(engine.__file__), Path(distribution.__file__), Path(loan.__file__), Path(shared.__file__),
        ROOT / "research/selected_mix_daily_two_year_v1.py", ROOT / "research/strategy_review_diagnostics_v1.py",
        parent.DIVIDENDS, parent.OUT / "inputs/market.parquet", parent.OUT / "inputs/mature_labels.parquet",
        prior.funding.OUT / "inputs/decision_information.parquet", prior.funding.OUT / "inputs/funding_features.parquet",
        prior.funding.OUT / "results/nested_funding_predictions.parquet"]
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
        "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}, "before_new_strategy_returns": True}, True)
    save(OUT / "inputs/mandate.json", read(ROOT / "config/510300_existing_data_training_mandate_v1.json"), True)
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    print(f"成本传导信息的8账户与作用拆分已固定，共同信息日{x.common_known.sum()}。", flush=True)


def verify_predictions(pred, saved, x, labels, groups, count):
    lookup, sample, pools = pred.set_index(["idx", "scope", "model"]), labels.set_index("idx", drop=False), {}
    for m in saved:
        i, scope = m["idx"], m["scope"]
        pool = sample.loc[m["training_indices"]]
        lower = x.date.iloc[i] - pd.DateOffset(years=2)
        assert pool.date.ge(lower).all() and pool.exit_idx.lt(i).all()
        assert x.loc[pool.idx, "pricing_known_at"].ge(lower.tz_localize("Asia/Shanghai")).all()
        assert x.common_known.iloc[pool.idx.to_numpy(int)].all()
        assert pools.setdefault((i, scope), m["training_indices"]) == m["training_indices"]
        if scope == METHOD_SCOPE:
            assert x.method_group.iloc[pool.idx.to_numpy(int)].eq(x.method_group.iloc[i]).all()
        for h in HORIZONS:
            stats = distribution.empirical_statistics(sample.loc[m["selected_indices"], f"gross_return{h}"])
            for old, field in [("mu5", "mu"), ("variance5", "variance"), ("q05", "q05"), ("es95", "es95")]:
                np.testing.assert_allclose(stats[old], lookup.loc[(i, scope, m["model"]), f"{field}_{h}"], atol=1e-14, rtol=0)
    dates = sorted({key[0] for key in pools})
    checked = []
    for i in sorted({dates[0], dates[len(dates) // 2], dates[-1]}):
        altered = labels.copy()
        forbidden = altered.exit_idx.ge(i) | altered.date.lt(x.date.iloc[i] - pd.DateOffset(years=2))
        altered.loc[forbidden, ["gross_return5", "gross_return20"]] = -9999.
        for scope in SCOPES:
            a, _, _ = fit_day(i, x.iloc[:i + 1], groups[i], count, labels, scope)
            b, _, _ = fit_day(i, x.iloc[:i + 1], groups[i], count, altered, scope)
            assert a == b
        checked.append(x.date.iloc[i])
    return {"saved_distributions_recomputed": len(saved) * 2, "common_pool_and_source_clock": True,
        "future_and_expired_label_perturbation_dates": checked}


def diagnostics(pred, labels, x):
    scored = pred.merge(labels[["idx", "gross_return20"]], on="idx", how="left", validate="many_to_one")
    full5 = pd.read_parquet(parent.OUT / "inputs/mature_labels.parquet")[["idx", "gross_return5"]]
    scored = scored.merge(full5, on="idx", how="left", validate="many_to_one")
    scored = scored.merge(x[["idx", "pressure5", "method_group"]], on="idx", how="left", validate="many_to_one")
    scored.to_parquet(OUT / "results/scored_predictions.parquet", index=False)
    evaluation, monitors = [], []
    first = int(x.index[x.date.ge(START)][0])
    for (scope, model), rows in scored.groupby(["scope", "model"]):
        for h in HORIZONS:
            for subset in ["ALL", "PRICE_WEAK"]:
                mature = rows[rows[f"gross_return{h}"].notna()].copy()
                if subset == "PRICE_WEAK":
                    mature = mature[mature.pressure5.gt(0)]
                e = mature[f"gross_return{h}"] - mature[f"q05_{h}"]
                evaluation.append({"scope": scope, "model": model, "horizon": h, "subset": subset, "n": len(mature),
                    "MSE": float(((mature[f"gross_return{h}"] - mature[f"mu_{h}"]) ** 2).mean()),
                    "quantile_loss": float(np.maximum(.05 * e, -.95 * e).mean()), "q05_breach_fraction": float(e.lt(0).mean())})
            sample = rows[rows[f"gross_return{h}"].notna() & ((rows.idx - first) % h == 0)].copy()
            rate = sample[f"gross_return{h}"].lt(sample[f"q05_{h}"]).rolling(60, min_periods=60).mean()
            monitors.append({"scope": scope, "model": model, "horizon": h, "nonoverlap_rows": len(sample),
                "assessable_rows": int(rate.notna().sum()), "alerts": int(rate.gt(.15).sum()),
                "status": "ASSESSABLE" if rate.notna().any() else "NOT_ASSESSABLE"})
    return evaluation, monitors


def run():
    frozen = read(OUT / "freeze.json")
    assert digest(OUT / "protocol.json") == frozen["protocol_sha256"]
    for path, sha in frozen["sources"].items():
        assert digest(ROOT / path) == sha, path
    save(OUT / "RUN_STARTED.json", {"at": now()}, True)
    market, x = pd.read_parquet(parent.OUT / "inputs/market.parquet"), information()
    dividends = engine.normalize_dividends(pd.read_csv(parent.DIVIDENDS))
    labels = mechanics.labels_for_horizons(market, dividends)
    x.to_parquet(OUT / "inputs/decision_information.parquet", index=False)
    labels.to_parquet(OUT / "inputs/mature_labels.parquet", index=False)
    count = len(pd.read_parquet(prior.funding.OUT / "inputs/funding_features.parquet", columns=["source_idx"]))
    nested = pd.read_parquet(prior.funding.OUT / "results/nested_funding_predictions.parquet")
    groups = {int(i): part for i, part in nested.groupby("decision_idx", sort=False)}
    rows, saved, receipts = [], [], []
    for i in np.flatnonzero(x.date.ge(START)):
        for scope in SCOPES:
            p, m, r = fit_day(int(i), x, groups.get(int(i)), count, labels, scope)
            rows.extend(p)
            saved.extend(m)
            receipts.append(r)
        if len(receipts) % 600 == 0:
            print(f"成本传导分布已计算至{x.date.iloc[i].date()}。", flush=True)
    pred, schedule = pd.DataFrame(rows), pd.DataFrame(receipts)
    pred.to_parquet(OUT / "results/predictions.parquet", index=False)
    schedule.to_parquet(OUT / "results/update_receipts.parquet", index=False)
    save(OUT / "results/saved_distributions.json", saved, True)
    checks = verify_predictions(pred, saved, x, labels, groups, count)
    evaluation, monitors = diagnostics(pred, labels, x)
    allocated = account_predictions(pred)
    allocated.to_parquet(OUT / "results/account_prediction_inputs.parquet", index=False)
    accounts, account_checks, cycle_rows = {}, [], []
    for arm in ARMS:
        policy = policy_name(arm, MAIN_SCOPE, 20)
        for cost in distribution.COSTS:
            ledger, decisions = simulate(market, x, dividends, allocated, arm, MAIN_SCOPE, 20, cost)
            check = parent.account_verification(ledger, decisions)
            assert not (ledger.shares.shift(1, fill_value=0).gt(0) & ledger.filled_quantity.gt(0)).any()
            folder = OUT / "accounts" / cost / policy
            folder.mkdir(parents=True, exist_ok=True)
            ledger.to_parquet(folder / "ledger.parquet", index=False)
            decisions.to_parquet(folder / "decisions.parquet", index=False)
            completed, pending = cycles(ledger)
            completed.to_parquet(folder / "cycles.parquet", index=False)
            save(folder / "pending_cycle.json", pending, True)
            accounts[policy, cost] = ledger
            account_checks.append({"policy": policy, "cost": cost, **check})
            cycle_rows.append({"policy": policy, "cost": cost, "closed": len(completed), "open": pending is not None})
            m = metrics(ledger)
            print(f"{policy}/{cost}：夏普{mechanics.sharpe_text(m['sharpe'])}、年化{m['annual_return']:.3%}、回撤{abs(m['max_drawdown']):.3%}。", flush=True)
    measures, rolling = summarize_accounts(OUT, market, accounts)
    comparisons, rng = [], np.random.default_rng(2026092614)
    block_x = x.assign(loan_stat_month=x.pricing_stat_month)
    for cost in distribution.COSTS:
        base = accounts[policy_name("BASE", MAIN_SCOPE, 20), cost]
        for arm in ["MEAN_ONLY", "TAIL_ONLY", "JOINT"]:
            left = accounts[policy_name(arm, MAIN_SCOPE, 20), cost]
            periods = []
            for lo, hi in [(START, "2023-12-31"), ("2024-01-01", END)]:
                mask = left.date.between(lo, hi)
                periods.append({"start": lo, "end": hi, "annual_arithmetic_difference": float((left.loc[mask, "net_return"] - base.loc[mask, "net_return"]).mean() * 242)})
            comparisons.append({"cost": cost, "arm": arm, "purpose": "INCREMENT_VS_BASE", "daily_blocks": paired_interval(left, base, rng),
                "release_blocks": loan.release_block_interval(left, base, block_x, rng), "fixed_periods": periods})
        interaction = base[["date", "idx", "net_return"]].copy()
        interaction["net_return"] += accounts[PRIMARY, cost].net_return - accounts[policy_name("MEAN_ONLY", MAIN_SCOPE, 20), cost].net_return - accounts[policy_name("TAIL_ONLY", MAIN_SCOPE, 20), cost].net_return
        zero = interaction.copy()
        zero["net_return"] = 0.
        comparisons.append({"cost": cost, "purpose": "FACTORIAL_INTERACTION_NOT_A_TRADABLE_ACCOUNT", "daily_blocks": paired_interval(interaction, zero, rng),
            "release_blocks": loan.release_block_interval(interaction, zero, block_x, rng)})
    primary = next(m for m in measures if m["policy"] == PRIMARY and m["cost"] == "STRESS")
    increment = next(r for r in comparisons if r["purpose"] == "INCREMENT_VS_BASE" and r["arm"] == "JOINT" and r["cost"] == "STRESS")
    e = {(r["scope"], r["model"], r["horizon"], r["subset"]): r for r in evaluation}
    sensitivity = e[METHOD_SCOPE, SIGNAL, 20, "ALL"]["MSE"] <= e[METHOD_SCOPE, CONTROL, 20, "ALL"]["MSE"] and e[METHOD_SCOPE, SIGNAL, 5, "ALL"]["quantile_loss"] <= e[METHOD_SCOPE, CONTROL, 5, "ALL"]["quantile_loss"]
    gate = sensitivity and increment["daily_blocks"]["lower_95"] > 0 and increment["release_blocks"]["lower_95"] > 0 and all(r["annual_arithmetic_difference"] > 0 for r in increment["fixed_periods"])
    ledger = accounts[PRIMARY, "STRESS"]
    gross = float((ledger.price_pnl + ledger.dividend_recognized).sum())
    expense = float((ledger.commission + ledger.slippage_cost).sum() + ledger.terminal_exit_reserve.iloc[-1])
    np.testing.assert_allclose(gross - expense, primary["profit"], atol=1e-6, rtol=0)
    schedules = []
    for scope, group in schedule.groupby("scope"):
        updated = group[group.status.eq("UPDATED")]
        schedules.append({"scope": scope, "statuses": group.status.value_counts().to_dict(), "first_prediction": updated.date.min(), "last_prediction": updated.date.max(),
            "training_rows_min": int(updated.n_train.min()), "training_rows_max": int(updated.n_train.max()),
            "release_months_min": int(updated.release_months.min()), "release_months_max": int(updated.release_months.max()), "last_status": group.iloc[-1].to_dict()})
    save(OUT / "result.json", {"at": now(), "study_id": STUDY,
        "status": "DEVELOPMENT_CANDIDATE_REQUIRES_INDEPENDENT_VALIDATION" if primary["historical_point_targets_met"] and gate else "FROZEN_NO_QUALIFIED_MANUFACTURING_PRICE_TRANSMISSION_STRATEGY",
        "primary": primary, "all_accounts": measures, "comparisons": comparisons, "method_prediction_sensitivity_pass": sensitivity,
        "development_increment_gate": gate, "new_full_accounts": len(accounts), "joint_target_pass_accounts": sum(bool(m["historical_point_targets_met"]) for m in measures),
        "prediction_records": len(pred), "conditional_distributions": len(pred) * 2, "unique_prediction_days": int(pred.idx.nunique()), "schedules": schedules,
        "forecast_evaluation": evaluation, "mature_monitors": monitors, "prediction_checks": checks, "account_checks": account_checks, "cycle_counts": cycle_rows,
        "primary_cash_attribution": {"gross_pnl": gross, "cost_and_reserve": expense, "net_profit": primary["profit"]},
        "primary_rolling_two_year_joint_passes": sum(r["joint_point_pass"] for r in rolling if r["policy"] == PRIMARY and r["cost"] == "STRESS"),
        "source_months": 116, "new_parameter_searches": 0, "new_funding_regression_fits": 0, "new_independent_forward_observations": 0,
        "historical_first_vintage_verified": False, "goal_status": "active", "goal_achieved": False, "current_market_view": "NO_VIEW", "orders_authorized": False}, True)
    print("成本传导信息的8个完整账户及收益/尾部作用比较已经完成。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="制造业成本传导环境的账户增量研究")
    parser.add_argument("command", choices=["check", "freeze", "run"])
    action = parser.parse_args().command
    if action == "check":
        print(implementation_checks())
    elif action == "freeze":
        freeze()
    else:
        try:
            run()
        except Exception as exc:
            if not (OUT / "RUN_FAILURE.json").exists():
                save(OUT / "RUN_FAILURE.json", {"at": now(), "error": f"{type(exc).__name__}: {exc}"}, True)
            raise
