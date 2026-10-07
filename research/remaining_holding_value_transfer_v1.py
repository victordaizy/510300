"""固定三形态的剩余清算价值与逐期资金误差修正；旧退出研究的有限迁移比较。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE / "reports/research/510300_remaining_holding_value_transfer_v1"
SOURCE = WORKSPACE / "reports/research/510300_holiday_event_capital_risk_v1"
PREDICTION = WORKSPACE / "reports/research/510300_integrated_macro_micro_prediction_v1"
TECH = ["technical_trend", "technical_relative_volatility", "technical_trend_change3", "technical_log_volume"]
EXTRA = ["pmi_level", "pmi_change3", "funding_gap_pp", "funding_change5_pp", "flow5", "flow_breadth5", "if_basis_rank", "if_oi_rank", "GSPC_z", "VIX_z"]
POLICIES = ("A", "FIVE_DAY_REENTRY_VALUE", "REMAINING_PRICE", "REMAINING_MEAN_CORRECTION", "REMAINING_MACRO_CORRECTION")
PERIODS = {"PRIMARY": ("2021-01-04", "2026-08-14"), "RECENT_DIAGNOSTIC": ("2024-08-20", "2026-08-14")}


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def engines(root):
    return (imported("remaining_value_exit_engine", root / "code/exit_engine.py"),
            imported("remaining_value_parent", root / "code/parent_engine.py"))


def fit(x, y, columns, indices, fit_idx, maturity_idx, name):
    values = x.loc[indices, columns].to_numpy(float)
    means, scales = values.mean(axis=0), values.std(axis=0, ddof=1)
    scales[scales < 1e-12] = 1.
    z = np.clip((values - means) / scales, -5., 5.)
    center, ymean = z.mean(axis=0), float(y.mean())
    beta = np.linalg.solve((z - center).T @ (z - center) + 10. * np.eye(len(columns)), (z - center).T @ (y - ymean))
    return {"model": name, "fit_idx": int(fit_idx), "fit_date": str(x.date.iloc[fit_idx]),
            "latest_label_maturity_idx": int(maturity_idx), "columns": columns, "training_indices": indices.tolist(),
            "n_train": len(indices), "mean": means.tolist(), "scale": scales.tolist(), "beta": beta.tolist(),
            "intercept": ymean - float(center @ beta), "training_target_mean": ymean}


def predict(model, row):
    z = np.clip((row[model["columns"]].to_numpy(float) - model["mean"]) / model["scale"], -5., 5.)
    return float(model["intercept"] + z @ model["beta"])


def first_sellable(p, d, i):
    while i < len(d):
        if p.tradable(d, i, "SELL"):
            return i
        i += 1
    return None


def liquidation_increment(p, d, origin, horizon):
    """旧持仓两种卖出方案；共同买入成本及原有分红抵消，仅计新增权益。"""
    immediate, later = first_sellable(p, d, origin), first_sellable(p, d, origin + horizon)
    if immediate is None or later is None:
        return np.nan, None
    c = p.COSTS["STRESS"][0]
    immediate_value = p.fill_price(float(d.open.iloc[immediate]), "SELL", "STRESS") * (1 - c)
    later_value = p.fill_price(float(d.open.iloc[later]), "SELL", "STRESS") * (1 - c)
    dividend = float(d.dividend.iloc[immediate + 1:later + 1].sum())
    target = (later_value - immediate_value + dividend) / float(d.open.iloc[immediate])
    return float(target), int(later)


def labels(root, p, d):
    controls = pd.read_parquet(root / "inputs/controls.parquet")
    result = pd.DataFrame({"idx": np.arange(len(d)), "date": d.date, "RAW5": np.nan, "maturity_idx": np.nan,
                           **{f"H{h}": np.nan for h in range(1, 5)}})
    for control in controls.to_dict("records"):
        i, finish = int(control["entry_idx"]), int(control["exit_idx"])
        if finish >= len(d):
            continue
        target = {"RAW5": float(control["net_return"])}
        mature = finish
        for h in range(1, 5):
            value, end = liquidation_increment(p, d, i, h)
            assert end is not None and np.isfinite(value)
            target[f"H{h}"] = value
            mature = max(mature, end)
        result.loc[i, list(target)] = list(target.values())
        result.loc[i, "maturity_idx"] = mature
    return result


def learn(d, x, targets, start, end):
    out = pd.DataFrame({"idx": np.arange(len(d)), "date": d.date, "RAW5": np.nan,
                        "price_fit_idx": np.nan, "correction_fit_idx": np.nan,
                        **{f"H{h}_{kind}": np.nan for h in range(1, 5) for kind in ("PRICE", "MEAN", "MACRO")}})
    records, training_residuals, schedule = [], [], []
    price, corrections = {}, {}
    tech_known = np.isfinite(x[TECH].to_numpy(float)).all(axis=1)
    full_known = x.model_features_known.to_numpy(bool)
    target_cols = ["RAW5", "H1", "H2", "H3", "H4"]
    for i in range(start, end + 1):
        if (i - start) % 21 == 0:
            mature = targets.maturity_idx.lt(i) & targets.idx.ge(i - 504) & tech_known
            mature &= np.isfinite(targets[target_cols].to_numpy(float)).all(axis=1)
            indices = np.flatnonzero(mature)
            price, corrections = {}, {}
            record = {"idx": i, "date": d.date.iloc[i], "price_training_rows": len(indices), "correction_training_rows": 0}
            if len(indices) >= 252:
                last = int(targets.maturity_idx.iloc[indices].max())
                for name in target_cols:
                    model = fit(x, targets[name].iloc[indices].to_numpy(float), TECH, indices, i, last, f"PRICE_{name}")
                    price[name] = model
                    records.append(model)
                # 修正目标必须减去当年当日实际生成的价格预测，不能减本次回拟合值。
                past_predictions_known = out[[f"H{h}_PRICE" for h in range(1, 5)]].notna().all(axis=1).to_numpy(bool)
                residue_ids = np.flatnonzero(mature & full_known & past_predictions_known)
                record["correction_training_rows"] = len(residue_ids)
                if len(residue_ids) >= 252:
                    for h in range(1, 5):
                        y = targets[f"H{h}"].iloc[residue_ids].to_numpy(float) - out[f"H{h}_PRICE"].iloc[residue_ids].to_numpy(float)
                        model = fit(x, y, EXTRA, residue_ids, i, int(targets.maturity_idx.iloc[residue_ids].max()), f"CORRECTION_H{h}")
                        corrections[h] = model
                        records.append(model)
                        for k, origin in enumerate(residue_ids):
                            training_residuals.append({"fit_idx": i, "horizon": h, "origin_idx": int(origin),
                                "label_maturity_idx": int(targets.maturity_idx.iloc[origin]),
                                "original_price_fit_idx": int(out.price_fit_idx.iloc[origin]),
                                "actual_liquidation_increment": float(targets[f"H{h}"].iloc[origin]),
                                "prequential_price_prediction": float(out[f"H{h}_PRICE"].iloc[origin]), "residual": float(y[k])})
            record["price_status"] = "FIT" if price else "NO_VIEW"
            record["correction_status"] = "FIT" if corrections else "NO_VIEW_USE_PRICE_BASELINE"
            schedule.append(record)
        if price and tech_known[i]:
            out.loc[i, "price_fit_idx"] = price["RAW5"]["fit_idx"]
            out.loc[i, "RAW5"] = predict(price["RAW5"], x.iloc[i])
            for h in range(1, 5):
                base = predict(price[f"H{h}"], x.iloc[i])
                out.loc[i, [f"H{h}_PRICE", f"H{h}_MEAN", f"H{h}_MACRO"]] = base
                if corrections and full_known[i]:
                    correction = corrections[h]
                    out.loc[i, f"H{h}_MEAN"] = base + correction["training_target_mean"]
                    out.loc[i, f"H{h}_MACRO"] = base + predict(correction, x.iloc[i])
                    out.loc[i, "correction_fit_idx"] = correction["fit_idx"]
    return out, records, pd.DataFrame(training_residuals), pd.DataFrame(schedule)


def exit_function(policy, forecasts, store):
    def decide(i, active):
        if policy == "A" or active is None:
            return False
        remaining = int(active["entry_idx"]) + 5 - i
        if i <= int(active["entry_idx"]) or remaining not in (1, 2, 3, 4):
            return False
        if policy == "FIVE_DAY_REENTRY_VALUE":
            column = "RAW5"
        else:
            kind = {"REMAINING_PRICE": "PRICE", "REMAINING_MEAN_CORRECTION": "MEAN", "REMAINING_MACRO_CORRECTION": "MACRO"}[policy]
            column = f"H{remaining}_{kind}"
        value = float(forecasts[column].iloc[i])
        ready = np.isfinite(value)
        store.append({"idx": i, "date": forecasts.date.iloc[i], "signal_id": active["signal_id"], "remaining_sessions": remaining,
                      "forecast_column": column, "value": value, "status": "PREDICTED" if ready else "NO_VIEW",
                      "price_fit_idx": forecasts.price_fit_idx.iloc[i], "correction_fit_idx": forecasts.correction_fit_idx.iloc[i],
                      "extra_exit_requested": bool(ready and value <= 0)})
        return bool(ready and value <= 0)
    return decide


def mechanism_checks(m, p):
    dates = pd.bdate_range("2020-01-01", periods=12).strftime("%Y-%m-%d")
    close = np.array([4., 4.02, 4.04, 4.01, 4.06, 4.1, 4.12, 4.09, 4.13, 4.15, 4.17, 4.2])
    d = pd.DataFrame({"date": dates, "open": close - .01, "high": close + .02, "low": close - .03, "close": close,
                      "ao": close - .01, "ac": close, "dividend": 0., "volume": 100000.})
    d.loc[4, "dividend"] = .03
    dv = pd.DataFrame([{"ex_date": dates[4], "payment_date": dates[6], "cash_dividend_per_share": .03}])
    sig = pd.DataFrame([{"signal_id": "人工形态", "family": "BREAKOUT", "signal_idx": 1, "signal_date": dates[1], "stop_index": 3.}])
    views = pd.DataFrame({"common_known": [True] * 12})
    original_dec = pd.DataFrame([{"idx": i, "family": "BREAKOUT", "PATTERN_ONLY": True} for i in range(12)])
    tests = []
    for cost in p.COSTS:
        original = p.account(d, dv, sig, original_dec, 0, 11, "PATTERN_ONLY", cost)
        current = m.simulate(p, d, dv, sig, views, 0, 11, "A", cost, extra_exit=lambda i, active: False)
        pd.testing.assert_frame_equal(original[0], current[0][original[0].columns])
        tests.append(f"{cost}母账户与原引擎逐日相同")
    forecasts = pd.DataFrame({"date": dates, "price_fit_idx": 0., "correction_fit_idx": np.nan,
                              "RAW5": -.01, **{f"H{h}_PRICE": -.01 for h in range(1, 5)}})
    logs = []
    early = m.simulate(p, d, dv, sig, views, 0, 11, "REMAINING_PRICE", "STRESS", extra_exit=exit_function("REMAINING_PRICE", forecasts, logs))
    assert int(early[1].entry_idx.iloc[0]) == 2 and int(early[1].exit_idx.iloc[0]) == 3
    assert logs[0]["remaining_sessions"] == 4 and logs[0]["forecast_column"] == "H4_PRICE"
    tests.append("新买当日不卖，首个可卖日只读剩余四日期限")
    value, mature = liquidation_increment(p, d, 3, 1)
    c = p.COSTS["STRESS"][0]
    expected = ((p.fill_price(float(d.open.iloc[4]), "SELL", "STRESS") - p.fill_price(float(d.open.iloc[3]), "SELL", "STRESS")) * (1 - c) + .03) / float(d.open.iloc[3])
    assert mature == 4 and np.isclose(value, expected)
    tests.append("清算目标不重复扣买入费用并计新增应得股息")
    return tests


def freeze(root):
    if (root / "freeze.json").exists():
        raise RuntimeError("本次有限迁移比较已冻结。")
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    (root / "code").mkdir(exist_ok=True)
    for name in ("features.parquet", "signals.parquet", "dividends.csv"):
        shutil.copy2(SOURCE / "inputs" / name, root / "inputs" / name)
    shutil.copy2(PREDICTION / "inputs/controls.parquet", root / "inputs/controls.parquet")
    shutil.copy2(PREDICTION / "results/model_inputs.parquet", root / "inputs/model_inputs.parquet")
    shutil.copy2(SOURCE / "code/parent_engine.py", root / "code/parent_engine.py")
    original_engine = (SOURCE / "code/holiday_event_capital_risk_v1.py").read_text(encoding="utf-8")
    transforms = {
        "def simulate(p, d, dividends, signals, views, start, end, policy, cost):": "def simulate(p, d, dividends, signals, views, start, end, policy, cost, extra_exit=None):",
        "risk = bool(v[policy]) if policy in CANDIDATES else False": "risk = bool(extra_exit(i, active)) if extra_exit is not None and active is not None else False",
        "missing = policy != \"A\" and not bool(v.common_known)": "missing = False",
    }
    derived = original_engine
    for old, new in transforms.items():
        assert derived.count(old) == 1
        derived = derived.replace(old, new)
    (root / "code/exit_engine.py").write_text(derived, encoding="utf-8")
    shutil.copy2(Path(__file__), root / "code" / Path(__file__).name)
    m, p = engines(root)
    checks = mechanism_checks(m, p)
    protocol = {
        "study_id": "510300_REMAINING_HOLDING_VALUE_TRANSFER_V1", "frozen_at": p.now(), "periods": PERIODS,
        "question": "固定三形态内，已持有份额的剩余清算价值是否比重新买入五日净收益更适合作为退出判断；宏观资金能否在真实逐期误差上提供增量。",
        "novelty_boundary": "继续价值及期限对齐是旧44、52、118、121轮已研究方法，不声称新方法。本轮是新三形态、普通日训练池、09:00信息时钟、按原五日期限自然缩短的有限迁移。旧D60研究和85/15策略不恢复。",
        "capacity_preflight": "共同资料齐备后才启动额外退出的压力上限8.745%，已排除该架构；技术基线拥有更早价格支持，完整原入场的提前退出事后上限11.391%。上限均使用未来信息，不是策略。",
        "policies": POLICIES, "entry": "原形态及优先级全部沿用，不新增预测入场门；原失效和五日退出保持。",
        "decision_clock": "已有持仓在执行日09:00读取预测，当日09:30尝试卖出；买入次日可卖，受阻请求保留。",
        "RAW5": "原普通日期的新买入五日压力净收益，用作期限/现金流尚未对齐的退出对照。",
        "remaining_target": "第一个可卖开盘与h个交易日后的第一个可卖开盘的净卖出每份所得之差，加仅因继续持有新增的股息权益，除以立即可卖开盘原价。买入成本为双方共同历史成本，不重复扣。",
        "remaining_horizons": "h=1,2,3,4对应原第五日期限剩余交易日数；非择优窗口，不能替换或延长。",
        "training_cost": "STRESS比例卖出佣金及原报价滑点取整；标签是每份经济差，实际20万元账户仍计最低5元佣金及整手。",
        "training": {"lookback_sessions": 504, "min_price_rows": 252, "min_correction_rows": 252, "refit_every_sessions": 21,
                     "ridge_alpha": 10., "clip_standard_deviations": 5., "common_label_maturity": "四个剩余标签与原五日标签全部成熟且退出索引严格早于当前09:00拟合日。"},
        "technical_features": TECH, "correction_features": EXTRA,
        "residual_target": "实际剩余清算差减该普通日期原本实际生成的价格预测。禁止以当前模型回拟合过去形成虚假的小残差。",
        "mean_control": "相同残差样本的平均误差修正；与有宏观资金特征的误差修正使用同一训练日期及同一可用时点。",
        "fallback": "宏观资金缺失或成熟残差不够时，两修正版本均完整沿用价格基线，不填零因子，不跳过本金损益。",
        "exit": "本策略相应预测<=0时额外请求退出；没有预测则不新增退出请求。未关闭原失效和期限规则，不重新买回旧信号。",
        "primary_increment": "REMAINING_MACRO_CORRECTION相对REMAINING_MEAN_CORRECTION",
        "secondary_mechanism": "REMAINING_PRICE相对FIVE_DAY_REENTRY_VALUE",
        "inference": "主压力账户20日共同循环区块2000次；两项预定比较用单侧.05/2；四期限预测误差仅描述，不将重复日期当四倍独立样本。",
        "targets": {"net_sharpe": 1.2, "net_cagr": .1, "max_drawdown": .1}, "capital_cny": 200000,
        "costs": p.COSTS, "annual_days": 252, "parameters_searched": 0, "new_collection": False,
        "review_package": False, "independent_validation": False, "orders": False, "current_view": "NO_VIEW",
        "derived_engine_exact_replacements": transforms, "pre_freeze_mechanism_checks": checks,
    }
    p.save_json(root / "protocol.json", protocol)
    files = [root / "protocol.json", *[f for f in (root / "inputs").iterdir() if f.is_file()], *[f for f in (root / "code").iterdir() if f.is_file()]]
    p.save_json(root / "freeze.json", {"frozen_at": p.now(), "before_fit": True,
                "files": [{"path": f.relative_to(root).as_posix(), "sha256": p.digest(f)} for f in files]})
    print("已冻结剩余清算价值与两种误差修正；不改变入场，不搜索期限，不重复扣除历史买入费用。")


def run(root):
    m, p = engines(root)
    frozen = json.loads((root / "freeze.json").read_text(encoding="utf-8"))
    for item in frozen["files"]:
        assert p.digest(root / item["path"]) == item["sha256"]
    if (root / "RUN_STARTED.json").exists():
        raise RuntimeError("已开始本次固定研究，禁止覆盖重跑。")
    p.save_json(root / "RUN_STARTED.json", {"started_at": p.now()})
    out = root / "results"
    out.mkdir(exist_ok=True)
    d, signals, dividends = m.load(root)
    d = d.loc[d.date.le(PERIODS["PRIMARY"][1])].copy()
    signals = signals.loc[signals.signal_idx.lt(len(d))].copy()
    x = pd.read_parquet(root / "inputs/model_inputs.parquet")
    assert x.date.tolist() == d.date.tolist()
    target = labels(root, p, d)
    target.to_parquet(out / "liquidation_targets.parquet", index=False)
    start = int(d.index[d.date.ge(PERIODS["PRIMARY"][0])][0])
    forecasts, models, residuals, schedule = learn(d, x, target, start, len(d) - 1)
    forecasts.to_parquet(out / "forecasts.parquet", index=False)
    residuals.to_parquet(out / "correction_training_residuals.parquet", index=False)
    schedule.to_parquet(out / "fit_schedule.parquet", index=False)
    p.save_json(out / "fitted_models.json", models)
    print(f"已完成{len(models)}次固定回归拟合；价格与资金修正分别保留时钟。", flush=True)
    views = pd.DataFrame({"common_known": [True] * len(d)})
    accounts, metrics = {}, []
    for period, (lo, hi) in PERIODS.items():
        first, last = int(d.index[d.date.ge(lo)][0]), int(d.index[d.date.le(hi)][-1])
        for cost in p.COSTS:
            for policy in POLICIES:
                decision_forecasts = []
                callback = exit_function(policy, forecasts, decision_forecasts)
                result = m.simulate(p, d, dividends, signals, views, first, last, policy, cost, extra_exit=callback)
                ledger, trades, decisions, terminal = result
                accounts[(period, cost, policy)] = result
                folder = out / "accounts" / period / cost / policy
                folder.mkdir(parents=True, exist_ok=True)
                for name, frame in zip(("ledger", "trades", "decisions"), result[:3]):
                    frame.to_parquet(folder / f"{name}.parquet", index=False)
                pd.DataFrame(decision_forecasts).to_parquet(folder / "holding_decision_forecasts.parquet", index=False)
                metric = {"period": period, "cost": cost, "policy": policy, **p.metrics(ledger, trades, terminal)}
                if len(trades):
                    metric["largest_cycle_cny"] = float(trades.net_pnl.max())
                    metric["profit_without_largest_cycle_cny"] = metric["net_profit_cny"] - metric["largest_cycle_cny"]
                    assert (trades.exit_idx > trades.entry_idx).all()
                if not terminal["open_position"]:
                    assert np.isclose(trades.net_pnl.sum() if len(trades) else 0., metric["net_profit_cny"], atol=1e-6)
                assert np.allclose(ledger.cash_cny + ledger.shares * ledger.close + ledger.receivable_cny, ledger.equity_cny, atol=1e-7)
                if policy == "A":
                    original = pd.read_parquet(SOURCE / f"results/accounts/{period}/{cost}/A/ledger.parquet")
                    pd.testing.assert_frame_equal(original.drop(columns=["common_known"]), ledger.drop(columns=["common_known"]))
                p.save_json(folder / "metrics.json", metric)
                p.save_json(folder / "terminal.json", terminal)
                metrics.append(metric)
            print(f"已完成{period}、{cost}五个退出账户。", flush=True)
    p.save_json(out / "account_metrics.json", metrics)
    assess(root, p, d, x, target, forecasts, models, residuals, accounts, metrics, start)


def assess(root, p, d, x, target, forecasts, models, residuals, accounts, metrics, start):
    out = root / "results"
    n = len(d) - start
    rng = np.random.default_rng(202609244)
    starts = rng.integers(0, n, size=(2000, math.ceil(n / 20)))
    indices = ((starts[:, :, None] + np.arange(20)) % n).reshape(2000, -1)[:, :n]
    np.savez_compressed(out / "bootstrap_indices.npz", indices=indices)
    comparisons = []
    pairs = [("REMAINING_MACRO_CORRECTION", "REMAINING_MEAN_CORRECTION", "PRIMARY_INFORMATION"),
             ("REMAINING_PRICE", "FIVE_DAY_REENTRY_VALUE", "SECONDARY_HORIZON_AND_CASHFLOW")]
    for a, b, role in pairs:
        first, second = accounts[("PRIMARY", "STRESS", a)], accounts[("PRIMARY", "STRESS", b)]
        delta = p.reserved_returns(first[0], first[3]) - p.reserved_returns(second[0], second[3])
        boot = delta[indices].mean(axis=1) * 252
        lower = float(np.quantile(boot, .025))
        comparisons.append({"candidate": a, "control": b, "role": role, "annual_mean_increment": float(delta.mean() * 252),
                            "ci95": np.quantile(boot, [.025, .975]).tolist(), "familywise_one_sided_lower": lower,
                            "supports_positive_increment": lower > 0})
    p.save_json(out / "paired_increment.json", comparisons)
    scored = []
    for h in range(1, 5):
        eligible = forecasts.correction_fit_idx.notna() & target[f"H{h}"].notna()
        for kind in ("PRICE", "MEAN", "MACRO"):
            prediction = forecasts.loc[eligible, f"H{h}_{kind}"]
            actual = target.loc[eligible, f"H{h}"]
            scored.append({"remaining_horizon": h, "model": kind, "same_dates": int(eligible.sum()),
                           "mse": float(((prediction - actual) ** 2).mean()), "interpretation": "四期限共享日期，不能相加为独立样本。"})
    p.save_json(out / "matched_prediction_errors.json", scored)
    model_lookup = {(m["fit_idx"], m["model"]): m for m in models}
    checks = 0
    for row in forecasts.loc[forecasts.RAW5.notna()].itertuples():
        i, k = int(row.idx), int(row.price_fit_idx)
        original = model_lookup[(k, "PRICE_RAW5")]
        assert original["latest_label_maturity_idx"] < k <= i
        assert np.isclose(predict(original, x.iloc[i]), row.RAW5, atol=1e-12)
        checks += 1
        for h in range(1, 5):
            price = model_lookup[(k, f"PRICE_H{h}")]
            value = predict(price, x.iloc[i])
            assert np.isclose(value, getattr(row, f"H{h}_PRICE"), atol=1e-12)
            if pd.notna(row.correction_fit_idx):
                correction = model_lookup[(int(row.correction_fit_idx), f"CORRECTION_H{h}")]
                assert correction["latest_label_maturity_idx"] < correction["fit_idx"] <= i
                assert np.isclose(value + predict(correction, x.iloc[i]), getattr(row, f"H{h}_MACRO"), atol=1e-12)
                assert np.isclose(value + correction["training_target_mean"], getattr(row, f"H{h}_MEAN"), atol=1e-12)
            else:
                assert value == getattr(row, f"H{h}_MEAN") == getattr(row, f"H{h}_MACRO")
            checks += 3
    if len(residuals):
        assert (residuals.label_maturity_idx < residuals.fit_idx).all()
        assert (residuals.original_price_fit_idx <= residuals.origin_idx).all()
        assert np.allclose(residuals.actual_liquidation_increment - residuals.prequential_price_prediction, residuals.residual)
        for h, group in residuals.groupby("horizon"):
            origin = group.origin_idx.to_numpy(int)
            assert np.allclose(forecasts[f"H{h}_PRICE"].iloc[origin], group.prequential_price_prediction)
    for model in models:
        ids = np.asarray(model["training_indices"], dtype=int)
        assert (target.maturity_idx.iloc[ids] < model["fit_idx"]).all()
        assert (ids >= model["fit_idx"] - 504).all() and len(ids) >= 252
    p.save_json(root / "verification.json", {"status": "PASS", "model_fits": len(models), "saved_prediction_checks": checks,
                "prequential_residual_rows_checked": len(residuals), "account_count": len(metrics),
                "account_rows": sum(len(a[0]) for a in accounts.values()), "A_equals_original": True,
                "maturity_before_0900": True, "missing_correction_uses_same_price_baseline": True, "independent_validation": False})
    passes = [policy for policy in POLICIES if policy != "A" and all(next(v for v in metrics if v["period"] == "PRIMARY" and v["cost"] == c and v["policy"] == policy)["numerical_target_pass"] for c in p.COSTS)]
    p.save_json(root / "result.json", {"status": "HISTORICAL_POINT_CANDIDATE_NOT_INDEPENDENTLY_VALIDATED" if passes else "FROZEN_REMAINING_VALUE_TRANSFER_TARGET_NOT_MET",
                "completed_at": p.now(), "dual_cost_primary_numeric_pass": passes, "predeclared_comparisons": comparisons,
                "model_fits": len(models), "price_supported_days": int(forecasts.RAW5.notna().sum()),
                "correction_supported_days": int(forecasts.correction_fit_idx.notna().sum()),
                "goal_achieved": False, "independent_validation": False, "new_collection": False, "review_package_created": False})
    print("剩余持有价值迁移比较完成；主信息增量与期限对齐增量分别保留。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["freeze", "run"])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    (freeze if args.command == "freeze" else run)(args.root.resolve())


if __name__ == "__main__":
    main()
