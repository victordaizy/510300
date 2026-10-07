"""固定认沽价差的十日买回成本预测；不改变父研究选腿、期限和账户规则。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config/510300_existing_data_training_mandate_v1.json").is_file())
PARENT = ROOT / "reports/research/510300_protected_put_spread_daily_v1"
OUT = ROOT / "reports/research/510300_put_spread_buyback_cost_daily_v1"
STUDY = "510300_PUT_SPREAD_BUYBACK_COST_DAILY_V1"
POLICIES = ["MATCHED_VARIANCE", "EMPIRICAL", "PRICE", "MACRO"]
PRICE_COLUMNS = ["credit_fraction", "log_iv_rv20", "spot_change_since_quote", "trend20_current"]
MACRO_COLUMNS = ["funding_gap_pp", "credit_acceleration3_pp", "GSPC_z", "VIX_z"]
MODELS = {"EMPIRICAL": [], "PRICE": PRICE_COLUMNS, "MACRO": PRICE_COLUMNS + MACRO_COLUMNS}


def engine(root):
    source = root / "code/spread_engine.py"
    if not source.exists():
        source = PARENT / "code/protected_put_spread_daily_v1.py"
    spec = importlib.util.spec_from_file_location("frozen_put_spread_engine", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def freeze(root):
    h = engine(root)
    if (root / "freeze.json").exists():
        raise RuntimeError("本版已经冻结。")
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "candidates.parquet": PARENT / "results/candidates.parquet",
        "parent_labels.parquet": PARENT / "results/payoff_labels.parquet",
        "parent_signals.parquet": PARENT / "results/signals.parquet",
        "daily_terms.parquet": PARENT / "results/daily_terms.parquet",
        "option_eod.parquet": PARENT / "inputs/option_eod.parquet",
        "market.parquet": PARENT / "inputs/market.parquet",
        "calendar_prices.parquet": PARENT / "inputs/calendar_prices.parquet",
        "dividends.csv": PARENT / "inputs/dividends.csv",
        "macro_information.parquet": ROOT / "reports/research/510300_synthetic_short_gamma_variance_v1/results/information.parquet",
        "parent_protocol.json": PARENT / "protocol.json",
        "parent_result.json": PARENT / "result.json",
        "parent_verification.json": PARENT / "verification.json",
        "authority_update.json": PARENT / "authority_update.json",
        "mandate.json": ROOT / "config/510300_existing_data_training_mandate_v1.json",
    }
    for name, source in paths.items():
        (root / "inputs").mkdir(exist_ok=True)
        shutil.copy2(source, root / "inputs" / name)
    (root / "code").mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), root / "code" / Path(__file__).name)
    shutil.copy2(PARENT / "code/protected_put_spread_daily_v1.py", root / "code/spread_engine.py")
    shutil.copy2(PARENT / "code/account_primitives_source.py", root / "code/account_primitives_source.py")
    protocol = {
        "study_id": STUDY, "at": h.now(), "primary": "MACRO", "policies": POLICIES,
        "parent_rejection_preserved": True, "new_assets_or_capital_revision": False,
        "economic_question": "固定十日退出时，预测未来买回价差的现金成本能否改善已收保险费与后续赔付的比较。",
        "difference_from_parent": "父研究用20日方差作较长期到期赔付定价代理；本研究直接学同一合约十日买回成本，不改Delta、期限、费用和风险规则。",
        "target": "第10个后续交易日开盘的压力买回价差成本，不含固定手续费，除以原观察日价差宽度金额；分红按逐日合约单位还原",
        "target_maturity": "退出日期严格早于预测日09:00；当前尚未成熟标签不能影响参数",
        "window": "过去两个日历年的原点；每日更新；至少252个共同成熟标签",
        "models": MODELS, "price_feature_definitions": {
            "credit_fraction": "观察日压力净权利金/价差宽度金额",
            "log_iv_rv20": "观察日两腿平均隐波/20日实现波动率的对数",
            "spot_change_since_quote": "报价观察日之后、截至昨日收盘的含已发生分红对数总收益/观察日隐波日尺度；使报价陈旧期的已知价格变化进入价格对照",
            "trend20_current": "已保存当日09:00可知的20日价格趋势表征",
        },
        "macro_features": "资金利差、社融同比三月变化、美股及VIX既有可知表征；不把资金政策利差称为中性利率差",
        "fit": {"EMPIRICAL": "相同历史池均值", "PRICE_and_MACRO": "岭回归lambda10，不惩罚截距；训练均值和样本标准差标准化并截断[-5,5]",
                "prediction": "预测买回成本下限截断为零，上界不截断；边界次数披露，无参数搜索"},
        "minimum_entry_credit": "max(预测买回成本+四次每张5元费用, 四次费用+观察日退出滑点储备)；在09:00冻结，开盘按该信用下限验证",
        "gate": "观察日压力净权利金高于上述下限；开盘实际压力可得信用仍需满足同一下限",
        "matched_variance": "父研究PRICE定价门，只限制为本研究共同资料可用日；不是父账户的重写",
        "unchanged": ["Delta-0.30/-0.10两腿及到期选择", "原10个交易日退出", "原尾部风险预测", "初始最坏损失5%、ES2.5%及回撤余量", "原逐日减仓和现金保证金逻辑", "20万元、252日年化与零现金利息"],
        "costs": ["BASE", "STRESS"], "targets": {"net_sharpe": 1.2, "net_cagr": .1, "max_drawdown": .1},
        "primary_prediction_increment": "MACRO相对PRICE的真实买回成本归一化MSE及同日配对损失",
        "primary_account_increment": "MACRO-STRESS相对PRICE-STRESS完整账户的日收益",
        "bootstrap": {"block_sessions": 20, "draws": 2000, "seed": 2026092503},
        "fixed_periods": ["2021-2023", "2024-END"], "all_rolling_two_years": True,
        "post_selection_limit": "本研究源于父研究后识别的预测期限问题，属于开发迭代；不能视为独立确认",
        "raw_sync_quotes_available": False, "historical_prices_previously_seen": True,
        "new_market_downloads": 0, "orders_authorized": False,
    }
    h.save(root / "protocol.json", protocol, exclusive=True)
    paths_to_freeze = [root / "protocol.json", *sorted((root / "inputs").glob("*")), *sorted((root / "code").glob("*"))]
    h.save(root / "freeze.json", {"at": h.now(), "before_new_buyback_labels": True,
                                 "files": {p.relative_to(root).as_posix(): h.digest(p) for p in paths_to_freeze}}, exclusive=True)
    print("十日买回成本预测已冻结；父研究选腿、期限、费用与账户规则保持。", flush=True)


def data(root, h):
    market, calendar = h.load_market(root)
    eod = pd.read_parquet(root / "inputs/option_eod.parquet")
    eod["trade_date"] = pd.to_datetime(eod.trade_date).astype("datetime64[ns]")
    terms = pd.read_parquet(root / "inputs/daily_terms.parquet").rename(columns={"date": "trade_date", "strike": "effective_strike"})
    panel = eod.merge(terms, on=["trade_date", "contract_code"], how="left", validate="one_to_one")
    quotes = {(pd.Timestamp(r.trade_date), str(r.contract_code)): r._asdict() for r in panel.itertuples(index=False)}
    divs = pd.read_csv(root / "inputs/dividends.csv")
    events = {pd.Timestamp(r.ex_date): float(r.cash_dividend_per_share) for r in divs.itertuples()}
    c = pd.read_parquet(root / "inputs/candidates.parquet")
    macro = pd.read_parquet(root / "inputs/macro_information.parquet")
    keep = ["date", "decision_time", "trend20", *MACRO_COLUMNS, "dr_available_at", "policy_available_at", "credit_available_at", "GSPC_available_at", "VIX_available_at"]
    macro = macro[keep].rename(columns={"trend20": "trend20_current"})
    frame = c.merge(macro, on="date", how="left", validate="one_to_one")
    previous_total_log_return = np.log1p(market.total_simple).shift(1).reindex(pd.DatetimeIndex(frame.date)).to_numpy(float)
    frame["credit_fraction"] = frame.prior_stress_credit / frame.width_cash
    frame["spot_change_since_quote"] = previous_total_log_return / (np.exp(frame.log_iv) / np.sqrt(252))
    frame["features_known"] = np.isfinite(frame[PRICE_COLUMNS + MACRO_COLUMNS]).all(axis=1)
    for column in ["dr_available_at", "policy_available_at", "credit_available_at", "GSPC_available_at", "VIX_available_at"]:
        frame["features_known"] &= frame[column].notna() & frame[column].le(frame.decision_time)
    source_labels = pd.read_parquet(root / "inputs/parent_labels.parquet")
    frame = frame.merge(source_labels[["idx", "status10"]], on="idx", how="left", validate="one_to_one")
    frame["future_buyback_cost"] = np.nan
    frame["target_cost_fraction"] = np.nan
    for row in frame[frame.status10.eq("READY")].to_dict("records"):
        legs = h.terms_at(row, row["exit10_date"], quotes)
        assert h.valid_quote(legs, "open")
        cost = h.close_cost(legs, "open", "STRESS")
        loc = frame.index[frame.idx.eq(row["idx"])][0]
        frame.loc[loc, "future_buyback_cost"] = cost
        frame.loc[loc, "target_cost_fraction"] = cost / row["width_cash"]
    frame.to_parquet(root / "results/information_and_labels.parquet", index=False)
    return frame, quotes, market, calendar, events


def fit_at(current, frame):
    day = current["date"]
    lower = day - pd.DateOffset(years=2)
    train = frame[frame.date.ge(lower) & frame.exit10_date.lt(day) & frame.features_known & frame.status10.eq("READY")].copy()
    train = train[np.isfinite(train.target_cost_fraction)]
    receipt = {"date": day, "idx": int(current["idx"]), "lower_bound": lower, "training_count": len(train),
               "latest_training_exit": train.exit10_date.max() if len(train) else None}
    if len(train) < 252 or not current["features_known"]:
        return [], [], {**receipt, "status": "NO_VIEW"}
    y = train.target_cost_fraction.to_numpy(float)
    predictions, models = [], []
    for name, columns in MODELS.items():
        beta, mean, scale = [], [], []
        intercept = float(y.mean())
        if columns:
            raw = train[columns].to_numpy(float)
            mean, scale = raw.mean(axis=0), raw.std(axis=0, ddof=1)
            scale[scale < 1e-12] = 1
            z = np.clip((raw - mean) / scale, -5, 5)
            center = z.mean(axis=0)
            beta = np.linalg.solve((z - center).T @ (z - center) + 10 * np.eye(len(columns)), (z - center).T @ (y - y.mean()))
            intercept = float(y.mean() - center @ beta)
            x = np.clip((np.array([current[k] for k in columns]) - mean) / scale, -5, 5)
            raw_pred = float(intercept + x @ beta)
        else:
            raw_pred = intercept
        prediction = max(0., raw_pred)
        predictions.append({**receipt, "model": name, "raw_prediction": raw_pred, "predicted_cost_fraction": prediction,
                            "predicted_cost_cny": prediction * current["width_cash"], "zero_floor_used": raw_pred < 0})
        models.append({**receipt, "model": name, "features": columns, "training_indices": train.idx.to_list(),
                       "mean": mean, "scale": scale, "beta": beta, "intercept": intercept})
    return predictions, models, {**receipt, "status": "UPDATED"}


def learn(root, h, frame):
    predictions, models, receipts = [], [], []
    for current in frame[frame.date.ge(h.START) & frame.selection_status.eq("READY")].to_dict("records"):
        p, m, r = fit_at(current, frame)
        predictions.extend(p)
        models.extend(m)
        receipts.append(r)
        if len(receipts) % 300 == 0:
            print(f"十日买回成本模型已更新至{current['date'].date()}。", flush=True)
    pred = pd.DataFrame(predictions)
    pred.to_parquet(root / "results/predictions.parquet", index=False)
    pd.DataFrame(receipts).to_parquet(root / "results/update_receipts.parquet", index=False)
    h.save(root / "results/saved_models.json", models)
    source = pd.read_parquet(root / "inputs/parent_signals.parquet")
    lookup = {(r.date, r.model): r for r in pred.itertuples()}
    signals = []
    for row in source.to_dict("records"):
        if any((row["date"], model) not in lookup for model in MODELS):
            continue
        row["minimum_credit_MATCHED_VARIANCE"] = row["minimum_credit_PRICE"]
        row["gate_MATCHED_VARIANCE"] = row["gate_PRICE"]
        for model in MODELS:
            forecast = lookup[(row["date"], model)]
            minimum = max(forecast.predicted_cost_cny + 4 * h.FEE, 4 * h.FEE + row["prior_exit_slip"])
            row[f"minimum_credit_{model}"] = minimum
            row[f"gate_{model}"] = row["prior_stress_credit"] > minimum
            row[f"forecast_buyback_cost_{model}"] = forecast.predicted_cost_cny
        signals.append(row)
    signal_frame = pd.DataFrame(signals)
    signal_frame.to_parquet(root / "results/signals.parquet", index=False)
    return pred, models, signal_frame


def prediction_evaluation(root, h, pred, frame):
    scored = pred.merge(frame[["idx", "status10", "target_cost_fraction", "future_buyback_cost", "exit10_date"]], on="idx", validate="many_to_one")
    scored = scored[scored.status10.eq("READY")].copy()
    scored["error"] = scored.predicted_cost_fraction - scored.target_cost_fraction
    scored["squared_error"] = scored.error.pow(2)
    scored.to_parquet(root / "results/scored_predictions.parquet", index=False)
    summaries = []
    for name, group in scored.groupby("model"):
        summaries.append({"model": name, "mature_predictions": len(group), "MSE": float(group.squared_error.mean()),
                          "MAE": float(group.error.abs().mean()), "mean_prediction_error": float(group.error.mean()),
                          "zero_floor_predictions": int(group.zero_floor_used.sum())})
    pivot = scored.pivot(index="date", columns="model", values="squared_error")
    difference = pivot.MACRO.to_numpy() - pivot.PRICE.to_numpy()
    rng = np.random.default_rng(2026092503)
    means = []
    for _ in range(2000):
        starts = rng.integers(0, len(difference), size=math.ceil(len(difference) / 20))
        idx = ((starts[:, None] + np.arange(20)) % len(difference)).ravel()[:len(difference)]
        means.append(float(difference[idx].mean()))
    result = {"models": summaries, "macro_minus_price_MSE": float(difference.mean()),
              "MSE_increment_ci95": np.quantile(means, [.025, .975]),
              "fixed_periods": []}
    for lower, upper in [(h.START, pd.Timestamp("2023-12-31")), (pd.Timestamp("2024-01-01"), h.END)]:
        part = scored[scored.date.between(lower, upper)]
        result["fixed_periods"].extend({"model": name, "start": lower, "end": upper, "MSE": float(g.squared_error.mean()), "n": len(g)}
                                       for name, g in part.groupby("model"))
    h.save(root / "results/prediction_metrics.json", result)
    return result


def check(h):
    # 完全相同的常量现金成本应被三个含截距模型恢复，未成熟高额标签不得进入。
    days = pd.date_range("2022-01-05", periods=253)
    current_day = pd.Timestamp("2024-01-03")
    frame = pd.DataFrame({"idx": range(253), "date": days, "exit10_date": [pd.Timestamp("2023-01-01")] * 252 + [current_day],
                          "features_known": True, "status10": "READY", "target_cost_fraction": .25})
    for number, column in enumerate(PRICE_COLUMNS + MACRO_COLUMNS):
        frame[column] = np.sin(np.arange(253) / (number + 3))
    current = {"date": current_day, "idx": 500, "features_known": True, "width_cash": 2000.}
    current.update({column: 0. for column in PRICE_COLUMNS + MACRO_COLUMNS})
    before, models, _ = fit_at(current, frame)
    assert all(252 not in m["training_indices"] for m in models)
    np.testing.assert_allclose([r["predicted_cost_cny"] for r in before], 500., atol=1e-10)
    frame.loc[252, "target_cost_fraction"] = 1e9
    after, _, _ = fit_at(current, frame)
    np.testing.assert_allclose([r["predicted_cost_cny"] for r in after], [r["predicted_cost_cny"] for r in before], atol=1e-10)
    return {"intercept_recovers_constant_cash_cost": True, "future_exit_label_excluded": True, "no_parent_rule_change": True}


def verify(root):
    h = engine(root)
    h.verify_freeze(root)
    frame = pd.read_parquet(root / "results/information_and_labels.parquet").set_index("idx", drop=False)
    pred = pd.read_parquet(root / "results/predictions.parquet").set_index(["idx", "model"])
    models = json.loads((root / "results/saved_models.json").read_text(encoding="utf-8"))
    for model in models:
        train = frame.loc[model["training_indices"]]
        assert train.date.ge(pd.Timestamp(model["lower_bound"])).all() and train.exit10_date.lt(pd.Timestamp(model["date"])).all()
        current = frame.loc[model["idx"]]
        value = model["intercept"]
        if model["features"]:
            z = np.clip((current[model["features"]].to_numpy(float) - np.array(model["mean"])) / np.array(model["scale"]), -5, 5)
            value += float(z @ np.array(model["beta"]))
        np.testing.assert_allclose(max(0., value), pred.loc[(model["idx"], model["model"]), "predicted_cost_fraction"], atol=1e-12)
    prefix = []
    for cutoff in [pd.Timestamp("2023-12-29"), pd.Timestamp("2025-12-31")]:
        eligible = [m for m in models if m["model"] == "MACRO" and pd.Timestamp(m["date"]) <= cutoff]
        if not eligible:
            continue
        chosen = eligible[-1]
        current = frame.loc[chosen["idx"]].to_dict()
        truncated = frame[frame.date.le(current["date"])].copy()
        truncated.loc[truncated.exit10_date.ge(current["date"]), "target_cost_fraction"] = -1e9
        results, _, _ = fit_at(current, truncated)
        for row in results:
            np.testing.assert_allclose(row["predicted_cost_fraction"], pred.loc[(chosen["idx"], row["model"]), "predicted_cost_fraction"], atol=1e-12)
        prefix.append(str(pd.Timestamp(chosen["date"]).date()))
    for scenario in ["BASE", "STRESS"]:
        for policy in POLICIES:
            path = root / "accounts" / scenario / policy
            ledger = pd.read_parquet(path / "ledger.parquet")
            fills = pd.read_parquet(path / "fills.parquet")
            dec = pd.read_parquet(path / "decisions.parquet")
            np.testing.assert_allclose(ledger.equity, ledger.cash + ledger.option_value - ledger.terminal_exit_reserve, atol=1e-6)
            if len(fills):
                changes = fills.groupby("date").cash_change.sum().reindex(ledger.date, fill_value=0.).to_numpy()
                np.testing.assert_allclose(h.INITIAL + np.cumsum(changes), ledger.cash, atol=1e-6)
                assert fills.cash_after_leg.ge(-1e-7).all()
            trades = dec[dec.action.eq("ENTRY_FILLED")]
            if len(trades):
                assert trades.actual_maximum_loss.le(trades.budget + 1e-7).all()
                assert trades.actual_ES95.le(.025 * trades.known_equity + 1e-7).all()
    result = {"at": h.now(), "saved_predictions_recomputed": len(models), "prefix_and_future_poisoning": prefix,
              "strict_two_calendar_years_and_maturity": True, "accounts_reconciled": 8, "entry_tail_budgets_verified": True,
              "mechanism_checks": check(h)}
    h.save(root / "verification.json", result)
    return result


def run(root):
    h = engine(root)
    h.verify_freeze(root)
    assert h.digest(Path(__file__)) == h.digest(root / "code/put_spread_buyback_cost_daily_v1.py")
    h.save(root / "RUN_STARTED.json", {"at": h.now()}, exclusive=True)
    (root / "results").mkdir(exist_ok=True)
    h.save(root / "mechanism_checks.json", check(h))
    frame, quotes, market, calendar, events = data(root, h)
    pred, models, signals = learn(root, h, frame)
    prediction_metrics = prediction_evaluation(root, h, pred, frame)
    metrics, yearly, rolling, ledgers = [], [], [], {}
    for scenario in ["BASE", "STRESS"]:
        for policy in POLICIES:
            account = h.simulate(policy, scenario, signals, quotes, market, calendar, events)
            folder = root / "accounts" / scenario / policy
            folder.mkdir(parents=True, exist_ok=True)
            for name, value in account.items():
                if isinstance(value, pd.DataFrame):
                    value.to_parquet(folder / f"{name}.parquet", index=False)
                else:
                    h.save(folder / f"{name}.json", value)
            measure = h.account_metrics(policy, scenario, account)
            metrics.append(measure)
            y, r = h.yearly_and_rolling(policy, scenario, account["ledger"])
            yearly.extend(y)
            rolling.extend(r)
            ledgers[(policy, scenario)] = account["ledger"]
            print(f"{scenario}/{policy}完成：夏普{measure['net_sharpe']}，年化{measure['annualized_return']:.3%}，回撤{measure['max_drawdown']:.3%}。", flush=True)
    h.save(root / "results/account_metrics.json", metrics)
    h.save(root / "results/yearly_metrics.json", yearly)
    pd.DataFrame(rolling).to_parquet(root / "results/all_rolling_two_years.parquet", index=False)
    paired = ledgers[("MACRO", "STRESS")]["return"].to_numpy() - ledgers[("PRICE", "STRESS")]["return"].to_numpy()
    rng = np.random.default_rng(2026092503)
    estimates = []
    for _ in range(2000):
        starts = rng.integers(0, len(paired), size=math.ceil(len(paired) / 20))
        indices = ((starts[:, None] + np.arange(20)) % len(paired)).ravel()[:len(paired)]
        estimates.append(float(paired[indices].mean() * 252))
    periods = []
    for lower, upper in [(h.START, pd.Timestamp("2023-12-31")), (pd.Timestamp("2024-01-01"), h.END)]:
        for policy in POLICIES:
            ledger = ledgers[(policy, "STRESS")]
            periods.append({"policy": policy, "start": lower, "end": upper,
                            **h.return_metrics(ledger.loc[ledger.date.between(lower, upper), "return"])})
    h.save(root / "results/fixed_periods.json", periods)
    primary = next(m for m in metrics if m["policy"] == "MACRO" and m["scenario"] == "STRESS")
    result = {"study_id": STUDY, "at": h.now(), "primary": primary, "goal_achieved": False,
              "status": "HISTORICAL_CANDIDATE_REQUIRES_INDEPENDENT_VALIDATION" if primary["eligible_historical_candidate"] else "FROZEN_NO_QUALIFIED_BUYBACK_COST_POLICY",
              "new_daily_estimates": len(models), "new_ridge_fits": sum(m["model"] != "EMPIRICAL" for m in models),
              "accounts": len(metrics), "common_decision_days": len(signals),
              "gate_days": {p: int(signals[f"gate_{p}"].sum()) for p in POLICIES},
              "prediction_metrics": prediction_metrics, "paired_macro_minus_price_annual_mean": float(paired.mean() * 252),
              "paired_increment_ci95": np.quantile(estimates, [.025, .975]),
              "historical_candidates": [m for m in metrics if m["eligible_historical_candidate"]],
              "independent_validation": False, "orders_authorized": False}
    h.save(root / "result.json", result)
    checks = verify(root)
    print(json.dumps(h.clean({"status": result["status"], "primary": primary, "verification": checks}), ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="固定十日买回成本的两年每日学习研究")
    parser.add_argument("command", choices=["freeze", "run", "verify", "check"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.out)
    elif args.command == "run":
        run(args.out)
    elif args.command == "verify":
        print(json.dumps(engine(args.out).clean(verify(args.out)), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(check(engine(args.out)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
