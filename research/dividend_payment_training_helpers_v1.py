"""复用已检验的成熟事件训练和分红到账账户函数；没有采集入口。"""
from __future__ import annotations
import copy
import math
import numpy as np
import pandas as pd
import sparse_event_training_core_v3 as core

def train(events, market, cfg):
    predictions, models, final = [], [], {}
    for name, spec in cfg["models"].items():
        columns = sum([cfg["features"][g] for g in spec["groups"]], [])
        previous = []
        for _, row in events.iterrows():
            mature = events[events.label_end_at < row.decision_at]
            if len(mature) < cfg["min_train"]:
                continue
            model = core.fit_ridge(mature, columns, cfg["ridge_mean_loss_penalty"], target_column="target5")
            model.update({"model": name, "model_type": "RIDGE" if columns else "MATURE_MEAN",
                "prediction_month": row.month, "decision_at": row.decision_at, "target_column": "target5"})
            prediction = core.predict(model, row)
            errors = [r for r in previous if r["label_end_at"] < row.decision_at]
            rmse = math.sqrt(float(np.mean([(r["prediction"] - r["target5"]) ** 2 for r in errors]))) if errors else None
            record = {field: row[field] for field in ["month", "decision_at", "entry_date", "exit_date", "entry_i", "exit_i",
                "label_end_at", "reference_close", "vol20", "target5", "dividend_yield"]}
            record.update({"model": name, "prediction": prediction, "mean_prediction": model["intercept"], "n_train": len(mature),
                "train_max_label_end_at": model["maximum_training_label_end_at"],
                "mature_prequential_error_count": len(errors), "prior_prediction_rmse": rmse,
                "prior_error_months": "|".join(r["month"] for r in errors)})
            predictions.append(record)
            previous.append(record)
            models.append(model)
        cutoff = (pd.Timestamp(market.date.iloc[-1]).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15, minutes=1)).isoformat()
        mature = events[events.label_end_at < cutoff]
        model = core.fit_ridge(mature, columns, cfg["ridge_mean_loss_penalty"], target_column="target5")
        model.update({"model": name, "model_type": "RIDGE" if columns else "MATURE_MEAN", "as_of": cutoff,
            "target_column": "target5", "status": "EXPLORATORY_TRAINED_NO_CURRENT_SIGNAL"})
        final[name] = model
    return pd.DataFrame(predictions), models, final

def score(predictions):
    metrics = []
    for name, sample in predictions.groupby("model", sort=True):
        mse = float((sample.target5 - sample.prediction).pow(2).mean())
        mean_mse = float((sample.target5 - sample.mean_prediction).pow(2).mean())
        zero_mse = float(sample.target5.pow(2).mean())
        metrics.append({"model": name, "n_predictions": len(sample), "rmse": math.sqrt(mse), "mse": mse,
            "direction_hit": float(((sample.target5 > 0) == (sample.prediction > 0)).mean()),
            "mse_improvement_vs_mean": 1 - mse / mean_mse, "mse_improvement_vs_zero": 1 - mse / zero_mse})
    matched = predictions[predictions.model == "B_DIVIDEND"].merge(
        predictions[predictions.model == "A_PRICE"][["month", "prediction"]], on="month", suffixes=("", "_reference")).sort_values("decision_at")
    comparisons = []
    for name, sample in [("ALL", matched), ("FIRST_HALF", matched.iloc[:len(matched) // 2]), ("SECOND_HALF", matched.iloc[len(matched) // 2:])]:
        candidate_mse = float((sample.target5 - sample.prediction).pow(2).mean())
        reference_mse = float((sample.target5 - sample.prediction_reference).pow(2).mean())
        comparisons.append({"segment": name, "n": len(sample), "candidate_mse": candidate_mse,
            "reference_mse": reference_mse, "mse_improvement": 1 - candidate_mse / reference_mse})
    return pd.DataFrame(metrics), pd.DataFrame(comparisons)

def run_accounts(out, cfg, events, market, predictions, start_i):
    dividends = pd.read_csv(out / "inputs/dividends.csv")
    dividends = dividends[dividends.symbol == "510300.SH"]
    dividend_map = dividends.assign(date=pd.to_datetime(dividends.ex_date)).set_index("date").cash_dividend_per_share
    for row in market[market.dividend > 0].itertuples():
        core.require(row.date in dividend_map.index and abs(dividend_map.loc[row.date] - row.dividend) < 1e-12, "付款表与市场分红权益不同。")
    navs, trades, decisions, account_rows = [], [], [], []

    def account(name, scenario, policy, forecasts, settings, event_only=False):
        nav, ledger, decision, metric = core.account_run(name, scenario, forecasts, market, start_i, settings, dividends, event_only=event_only)
        nav["policy"] = policy
        for row in [*ledger, *decision, metric]:
            row["policy"] = policy
        navs.append(nav)
        trades.extend(ledger)
        decisions.extend(decision)
        account_rows.append(metric)

    for policy, options in cfg["policies"].items():
        effective = copy.deepcopy(cfg)
        effective["account"].update(options)
        for name in cfg["models"]:
            for scenario in cfg["account"]["slippage_per_side"]:
                account(name, scenario, policy, predictions, effective)
    unfiltered = events.assign(model="EVENT_ONLY", prediction=0.0, prior_prediction_rmse=0.0, mature_prequential_error_count=0)
    effective = copy.deepcopy(cfg)
    effective["models"]["EVENT_ONLY"] = {"type": "FIXED_EVENT_RULE_WITHOUT_FORECAST"}
    effective["account"].update({"prediction_error_multiplier": 0, "min_mature_prequential_errors": 0})
    for scenario in cfg["account"]["slippage_per_side"]:
        account("EVENT_ONLY", scenario, "EVENT_RULE_DIAGNOSTIC", unfiltered, effective, event_only=True)
    for name in ["CASH", "BUY_AND_HOLD"]:
        for scenario in cfg["account"]["slippage_per_side"]:
            account(name, scenario, "BENCHMARK", predictions, cfg)
    accounts = pd.DataFrame(account_rows)
    for name, frame in {"完整逐日账户": pd.concat(navs, ignore_index=True), "机会成交账簿": pd.DataFrame(trades),
            "逐事件交易判断": pd.DataFrame(decisions), "账户指标": accounts}.items():
        core.export(out / "results" / f"{name}.csv", frame)
    return accounts


def verify_accounts(out, cfg, market):
    nav = pd.read_csv(out / "results/完整逐日账户.csv")
    trades = pd.read_csv(out / "results/机会成交账簿.csv")
    accounts = pd.read_csv(out / "results/账户指标.csv")
    decisions = pd.read_csv(out / "results/逐事件交易判断.csv")
    prices = market.assign(day=market.date.dt.strftime("%Y-%m-%d")).set_index("day").close
    dividend_table = pd.read_csv(out / "inputs/dividends.csv")
    payments = {r.ex_date: r.payment_date for r in dividend_table[dividend_table.symbol == "510300.SH"].itertuples()}
    for (model, policy, scenario), daily in nav.groupby(["model", "policy", "scenario"]):
        metric = accounts[(accounts.model == model) & (accounts.policy == policy) & (accounts.scenario == scenario)].iloc[0]
        equity = daily.equity_cny.to_numpy(float)
        returns = equity / np.r_[cfg["capital_cny"], equity[:-1]] - 1
        peak = np.maximum.accumulate(np.r_[cfg["capital_cny"], equity])[1:]
        core.require(np.allclose(daily.cash_cny + daily.shares * daily.date.map(prices) + daily.dividend_receivable_cny, equity, atol=1e-7, rtol=0), "账户现金持仓不平。")
        core.require(np.allclose(returns, daily.daily_return, atol=1e-12, rtol=0), "逐日收益不同。")
        core.require(abs(-(equity / peak - 1).min() - metric.max_drawdown) < 1e-12, "最大回撤不同。")
        if returns.std(ddof=1) > 1e-14:
            core.require(abs(returns.mean() / returns.std(ddof=1) * math.sqrt(cfg["annual_days"]) - metric.net_sharpe) < 1e-10, "含空仓日的夏普不同。")
        else:
            core.require(pd.isna(metric.net_sharpe), "空仓夏普必须未定义。")
        selected = trades[(trades.model == model) & (trades.policy == policy) & (trades.scenario == scenario)]
        core.require(abs(selected.net_pnl_cny.sum() - (equity[-1] - cfg["capital_cny"])) < 1e-7, "全部成交损益不能对账。")
        obligations = []
        for row in daily.itertuples():
            if row.dividends_cny:
                obligations.append((payments[row.date], row.dividends_cny))
            core.require(abs(sum(v for d, v in obligations if d > row.date) - row.dividend_receivable_cny) < 1e-7, "未到账权益不同。")
            core.require(abs(sum(v for d, v in obligations if d == row.date) - row.dividend_paid_cny) < 1e-7, "分红付款不同。")
        for t in selected.itertuples():
            core.require(t.exit_i > t.entry_i, "T+1失效。")
            dividend = float(market.iloc[int(t.entry_i) + 1:int(t.exit_i) + 1].dividend.sum()) * t.shares
            pnl = t.shares * t.exit_fill_price - t.exit_commission_cny + dividend - t.entry_cash_debit
            core.require(abs(pnl - t.net_pnl_cny) < 1e-7, "单笔净损益不同。")
    unconditional = decisions[decisions.model == "EVENT_ONLY"]
    core.require(unconditional.prediction.isna().all(), "固定事件对照不得假装模型预测。")
    core.require(unconditional.decision.isin(["BUY", "HALTED_AFTER_DRAWDOWN", "SKIP_ALREADY_HOLDING", "BUDGET_BELOW_ONE_LOT", "OPEN_PRICE_EXCEEDS_BUDGET"]).all(), "固定事件对照误用了预测门槛。")
    return len(accounts)

