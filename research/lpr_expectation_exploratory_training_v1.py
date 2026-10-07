"""按用户指令直接训练LPR预期模型，保存逐期预测和完整账户诊断。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports/research/510300_lpr_expectation_exploratory_training_v1"
PARENT = ROOT / "reports/research/510300_lpr_expectation_source_extension_v2"
PARENT_ZIP = ROOT / "deliverables/510300_LPR事前调查扩样_V2_GPT审阅_20260922.zip"
PARENT_HASH = "1a722741b1823adc93ead9c458cd93bac64ada5f9d5dd26166fc9b532ed377f3"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def export(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g")


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def freeze(out):
    require(not (out / "freeze.json").exists(), "已有冻结记录，请使用原记录训练，不能覆盖。")
    require(digest(PARENT_ZIP) == PARENT_HASH, "父级调查资料包身份不符。")
    inputs = out / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for name, source in {
        "ledger.json": PARENT / "ledger.json",
        "market.parquet": ROOT / "reports/research/510300_lpr_joint_response_20d_v1/inputs/market.parquet",
        "source_extension_v2.zip": PARENT_ZIP,
        "mandate.json": ROOT / "config/510300_sparse_opportunity_mandate_v1.json",
    }.items():
        shutil.copy2(source, inputs / name)
    (out / "code").mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), out / "code" / Path(__file__).name)
    paths = [out / "protocol.json", *inputs.iterdir(), *list((out / "code").iterdir())]
    save(out / "freeze.json", {
        "frozen_at": now(), "before_this_round_equity_labels_and_fits": True,
        "user_override": "不用那么严格，直接进入训练",
        "old_36_train_24_eval_gate_applied": False,
        "identities": {p.relative_to(out).as_posix(): digest(p) for p in paths},
        "prior_history_already_seen": True,
    })
    print("训练设定及直接输入已固定；开始生成标签和逐期拟合。", flush=True)


def read_market(out):
    market = pd.read_parquet(out / "inputs/market.parquet").sort_values("date").reset_index(drop=True)
    market["date"] = pd.to_datetime(market.date).astype("datetime64[ns]")
    require(not market.date.duplicated().any(), "市场日期重复。")
    require((market[["open", "close"]] > 0).all().all(), "价格必须为正。")
    require((market.dividend >= 0).all(), "现金分红不能为负。")
    market["return1"] = market.total_simple.astype(float)
    market["past20"] = np.expm1(np.log1p(market.return1).rolling(20).sum())
    market["vol20"] = market.return1.rolling(20).std(ddof=1)
    market["logvol20"] = np.log(market.vol20.clip(lower=1e-8))
    return market


def build_events(out, cfg, market):
    index = pd.DatetimeIndex(market.date)
    openings = index.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=30)
    events, excluded = [], []
    amount_cols = cfg["feature_groups"]["amount"]
    for original in load(out / "inputs/ledger.json"):
        if original["status"] != "ADMITTED_SAVED_HISTORICAL_SURVEY":
            excluded.append({"month": original["month"], "reason": original["status"]})
            continue
        event = dict(original)
        announcement = pd.Timestamp(event["announcement_at"])
        require(pd.Timestamp(event["survey_published_at"]) < announcement, "调查发布时间晚于公告。")
        require(pd.Timestamp(event["survey_modified_at"]) < announcement, "调查修改上界晚于公告。")
        review_i = int(openings.searchsorted(announcement, side="right"))
        entry_i = review_i + 1
        exit_i = entry_i + cfg["horizon_sessions"] - 1
        require(exit_i < len(market), "保存行情不足20个交易日标签。")
        review, block = market.iloc[review_i], market.iloc[entry_i:exit_i + 1]
        dividends = block.dividend.to_numpy(dtype=float).copy()
        dividends[0] = 0.0
        target = (float(block.close.iloc[-1]) + float(dividends.sum())) / float(block.open.iloc[0]) - 1
        decision = openings[entry_i] - pd.Timedelta(minutes=30)
        event.update({
            "review_date": str(index[review_i].date()), "decision_at": decision.isoformat(),
            "entry_date": str(index[entry_i].date()), "exit_date": str(index[exit_i].date()),
            "label_end_at": (index[exit_i].tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)).isoformat(),
            "review_i": review_i, "entry_i": entry_i, "exit_i": exit_i,
            "reference_close": float(review.close), "vol20": float(review.vol20),
            "past20": float(review.past20), "logvol20": float(review.logvol20),
            "return1": float(review.return1), "target20": target,
            "amount_cohort": all(event.get(col) is not None and np.isfinite(event[col]) for col in amount_cols),
        })
        cols = sum((cfg["feature_groups"][g] for g in ["price", "actual", "votes"]), [])
        require(all(event.get(col) is not None and np.isfinite(event[col]) for col in cols), "主样本必要特征缺失。")
        events.append(event)
    events.sort(key=lambda x: x["month"])
    require(len(events) == 56 and len(excluded) == 28, "56调查月及28缺失月边界改变。")
    return pd.DataFrame(events), pd.DataFrame(excluded)


def fit_ridge(frame, columns, penalty):
    x = frame[columns].to_numpy(dtype=float)
    y = frame.target20.to_numpy(dtype=float)
    center, raw_scale = x.mean(axis=0), x.std(axis=0, ddof=0)
    active = raw_scale >= 1e-12
    scale = np.where(active, raw_scale, 1.0)
    z = (x - center) / scale
    z[:, ~active] = 0.0
    intercept = float(y.mean())
    coefficients = np.linalg.solve(z.T @ z / len(frame) + penalty * np.eye(len(columns)), z.T @ (y - intercept) / len(frame))
    coefficients[~active] = 0.0
    return {
        "columns": columns, "center": center.tolist(), "scale": scale.tolist(),
        "active": active.tolist(), "coefficients_standardized": coefficients.tolist(),
        "intercept": intercept, "train_months": frame.month.tolist(), "n_train": len(frame),
        "maximum_training_label_end_at": frame.label_end_at.max(),
    }


def predict(model, row):
    x = np.array([row[col] for col in model["columns"]], dtype=float)
    z = (x - np.asarray(model["center"])) / np.asarray(model["scale"])
    z[~np.asarray(model["active"], dtype=bool)] = 0.0
    return float(model["intercept"] + z @ np.asarray(model["coefficients_standardized"]))


def walk_train(cfg, events, market):
    predictions, fitted, final_models = [], [], {}
    for name, spec in cfg["models"].items():
        cohort = events if spec["cohort"] == "ALL" else events[events.amount_cohort]
        columns = sum((cfg["feature_groups"][group] for group in spec["groups"]), [])
        past_predictions = []
        for _, row in cohort.iterrows():
            train = cohort[cohort.label_end_at < row.decision_at]
            if len(train) < cfg["min_train"]:
                continue
            model = fit_ridge(train, columns, cfg["ridge_lambda_on_mean_squared_loss"])
            model.update({"model": name, "prediction_month": row.month, "decision_at": row.decision_at})
            prediction = predict(model, row)
            old_errors = [p for p in past_predictions if p["label_end_at"] < row.decision_at]
            prior_rmse = math.sqrt(float(np.mean([(p["prediction"] - p["target20"]) ** 2 for p in old_errors]))) if old_errors else None
            result = {k: row[k] for k in ["month", "decision_at", "entry_date", "exit_date", "label_end_at", "entry_i", "exit_i", "reference_close", "vol20", "target20"]}
            result.update({
                "model": name, "cohort": spec["cohort"], "prediction": prediction,
                "mean_prediction": model["intercept"], "n_train": len(train),
                "train_max_label_end_at": model["maximum_training_label_end_at"],
                "mature_prequential_error_count": len(old_errors), "prior_prediction_rmse": prior_rmse,
                "prior_error_months": "|".join(p["month"] for p in old_errors),
            })
            require(model["maximum_training_label_end_at"] < row.decision_at, "训练标签尚未成熟。")
            predictions.append(result)
            past_predictions.append(result)
            fitted.append(model)
        as_of = (pd.Timestamp(market.date.iloc[-1]).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15, minutes=1)).isoformat()
        mature = cohort[cohort.label_end_at < as_of]
        final_model = fit_ridge(mature, columns, cfg["ridge_lambda_on_mean_squared_loss"])
        final_model.update({"model": name, "as_of": as_of, "status": "EXPLORATORY_TRAINED_NOT_LIVE_SIGNAL"})
        final_models[name] = final_model
    return pd.DataFrame(predictions), fitted, final_models


def prediction_metrics(predictions):
    rows = []
    for name, group in predictions.groupby("model", sort=True):
        group = group.sort_values("month")
        y, p, mean = (group[c].to_numpy(dtype=float) for c in ["target20", "prediction", "mean_prediction"])
        mse, mse_mean, mse_zero = [float(np.mean(v * v)) for v in [y - p, y - mean, y]]
        rows.append({"model": name, "n_predictions": len(group), "first_month": group.month.iloc[0],
                     "last_month": group.month.iloc[-1], "mse": mse, "rmse": math.sqrt(mse),
                     "mae": float(np.mean(np.abs(y - p))), "direction_hit": float(np.mean(np.sign(y) == np.sign(p))),
                     "mse_mean": mse_mean, "mse_zero": mse_zero,
                     "mse_improvement_vs_mean": 1 - mse / mse_mean,
                     "mse_improvement_vs_zero": 1 - mse / mse_zero})
    comparisons = []
    for candidate, reference in [("A_ACTUAL", "P_PRICE"), ("B_VOTES", "A_ACTUAL"), ("B_VOTES", "P_PRICE"), ("C_AMOUNT", "A_AMOUNT_MATCH")]:
        left = predictions[predictions.model == candidate]
        right = predictions[predictions.model == reference]
        match = left.merge(right[["month", "prediction"]], on="month", suffixes=("", "_reference")).sort_values("month")
        for segment, sample in [("ALL", match), ("FIRST_HALF", match.iloc[:len(match) // 2]), ("SECOND_HALF", match.iloc[len(match) // 2:])]:
            e = (sample.target20 - sample.prediction).pow(2)
            ref_e = (sample.target20 - sample.prediction_reference).pow(2)
            comparisons.append({"candidate": candidate, "reference": reference, "segment": segment,
                                "n": len(sample), "mse": float(e.mean()), "reference_mse": float(ref_e.mean()),
                                "mse_improvement": float(1 - e.mean() / ref_e.mean()),
                                "mean_squared_error_reduction": float((ref_e - e).mean())})
    return pd.DataFrame(rows), pd.DataFrame(comparisons)


def commission(value, cfg):
    return max(cfg["account"]["minimum_commission_cny"], abs(value) * cfg["account"]["commission_per_side"])


def quantity_for_budget(budget, cash, reference, slippage, cfg):
    lot = cfg["account"]["lot_size"]
    unit_price = reference * (1 + slippage)
    quantity = int(max(0.0, min(budget, cash)) / unit_price / lot) * lot
    while quantity > 0 and quantity * unit_price + commission(quantity * unit_price, cfg) > min(budget, cash) + 1e-9:
        quantity -= lot
    return quantity


def account_run(name, scenario, predictions, market, start_i, cfg):
    annual, params = cfg["annual_days"], cfg["account"]
    slip = params["slippage_per_side"][scenario]
    signals = {int(row.entry_i): row for row in predictions[predictions.model == name].itertuples()} if name in cfg["models"] else {}
    cash = peak = previous_equity = float(cfg["capital_cny"])
    quantity, position, halted, pending_brake = 0, None, False, False
    daily, trades, decisions = [], [], []
    cumulative_cost = 0.0
    for i in range(start_i, len(market)):
        day = market.iloc[i]
        held_at_start = quantity
        dividend_cash = quantity * float(day.dividend)
        cash += dividend_cash
        if position is not None:
            position["dividends_cny"] += dividend_cash
        day_cost, had_exposure = 0.0, quantity > 0

        def sell(raw_price, reason):
            nonlocal cash, quantity, position, day_cost, cumulative_cost
            require(position is not None and i > position["entry_i"], "卖出违反T+1。")
            proceeds = quantity * raw_price * (1 - slip)
            fee = commission(proceeds, cfg)
            slippage_cash = quantity * raw_price * slip
            cash += proceeds - fee
            day_cost += fee + slippage_cash
            cumulative_cost += fee + slippage_cash
            position.update({"exit_date": str(day.date.date()), "exit_i": i, "exit_raw_price": raw_price,
                             "exit_fill_price": raw_price * (1 - slip), "exit_commission_cny": fee,
                             "exit_reason": reason, "net_pnl_cny": proceeds - fee + position["dividends_cny"] - position["entry_cash_debit"],
                             "holding_sessions": i - position["entry_i"] + 1})
            trades.append(position)
            quantity, position = 0, None

        if quantity > 0 and pending_brake and i > position["entry_i"]:
            sell(float(day.open), "ACCOUNT_DRAWDOWN_BRAKE_NEXT_OPEN")
            pending_brake = False

        signal = signals.get(i)
        if signal is not None:
            decision = {"model": name, "scenario": scenario, "month": signal.month, "entry_date": signal.entry_date,
                        "prediction": signal.prediction, "prior_prediction_rmse": signal.prior_prediction_rmse,
                        "mature_error_count": signal.mature_prequential_error_count, "decision": "CASH"}
            if halted:
                decision["decision"] = "HALTED_AFTER_DRAWDOWN"
            elif quantity > 0:
                decision["decision"] = "SKIP_ALREADY_HOLDING"
            elif signal.mature_prequential_error_count < params["min_mature_prequential_errors"]:
                decision["decision"] = "PREQUENTIAL_ERROR_WARMUP"
            else:
                weight = min(1.0, params["vol_target_annual"] / (signal.vol20 * math.sqrt(annual)))
                budget = cash * weight
                stress_slip = params["slippage_per_side"]["STRESS"]
                estimate_q = quantity_for_budget(budget, cash, signal.reference_close, stress_slip, cfg)
                if estimate_q <= 0:
                    decision["decision"] = "BUDGET_BELOW_ONE_LOT"
                else:
                    base_value = estimate_q * signal.reference_close
                    estimated_cost = (2 * base_value * stress_slip + commission(base_value * (1 + stress_slip), cfg) + commission(base_value * (1 - stress_slip), cfg)) / base_value
                    threshold = estimated_cost + signal.prior_prediction_rmse
                    decision.update({"weight": weight, "estimated_stress_roundtrip_cost": estimated_cost, "entry_threshold": threshold})
                    if signal.prediction > threshold:
                        # 委托量于前收盘确定；开盘只允许减少至现金可承担数量。
                        affordable_q = quantity_for_budget(budget, cash, float(day.open), slip, cfg)
                        quantity = min(estimate_q, affordable_q)
                        if quantity > 0:
                            decision["decision"] = "BUY"
                            decision["shares"] = quantity
                        else:
                            decision["decision"] = "OPEN_PRICE_EXCEEDS_BUDGET"
                    else:
                        decision["decision"] = "PREDICTION_BELOW_ERROR_AND_COST"
            decisions.append(decision)
        elif name == "BUY_AND_HOLD" and i == start_i:
            quantity = quantity_for_budget(cash, cash, float(day.open), slip, cfg)

        if quantity > 0 and position is None:
            fill = float(day.open) * (1 + slip)
            value = quantity * fill
            fee = commission(value, cfg)
            cash -= value + fee
            day_cost += fee + quantity * float(day.open) * slip
            cumulative_cost += fee + quantity * float(day.open) * slip
            planned_exit = len(market) - 1 if name == "BUY_AND_HOLD" else int(signal.exit_i)
            position = {"model": name, "scenario": scenario, "event_month": "BENCHMARK" if signal is None else signal.month,
                        "entry_date": str(day.date.date()), "entry_i": i, "entry_raw_price": float(day.open),
                        "entry_fill_price": fill, "shares": quantity, "entry_commission_cny": fee,
                        "entry_cash_debit": value + fee, "dividends_cny": 0.0, "planned_exit_i": planned_exit}
            had_exposure = True
        if quantity > 0 and i == position["planned_exit_i"]:
            sell(float(day.close), "FIXED_HORIZON_CLOSE" if name != "BUY_AND_HOLD" else "BENCHMARK_END_CLOSE")
        equity = cash + quantity * float(day.close)
        peak = max(peak, equity)
        drawdown = equity / peak - 1
        if name in cfg["models"] and drawdown <= -0.08 and not halted:
            halted = True
            pending_brake = quantity > 0
        require(cash >= -1e-7 and quantity >= 0 and quantity % params["lot_size"] == 0, "账户现金、持仓或整手约束失效。")
        daily.append({"model": name, "scenario": scenario, "date": str(day.date.date()),
                      "cash_cny": cash, "shares": quantity, "equity_cny": equity,
                      "daily_return": equity / previous_equity - 1, "drawdown": drawdown,
                      "dividends_cny": dividend_cash, "cost_cny": day_cost, "cumulative_cost_cny": cumulative_cost,
                      "held_at_start": held_at_start, "any_exposure": had_exposure, "halted": halted})
        previous_equity = equity
    require(quantity == 0, "期末仍有未结持仓。")
    nav = pd.DataFrame(daily)
    returns = nav.daily_return.to_numpy(dtype=float)
    volatility = float(returns.std(ddof=1))
    years = len(nav) / annual
    metric = {"model": name, "scenario": scenario, "start": nav.date.iloc[0], "end": nav.date.iloc[-1],
              "n_trading_days": len(nav), "initial_capital_cny": cfg["capital_cny"], "ending_equity_cny": float(nav.equity_cny.iloc[-1]),
              "total_return": float(nav.equity_cny.iloc[-1] / cfg["capital_cny"] - 1),
              "annualized_return": float((nav.equity_cny.iloc[-1] / cfg["capital_cny"]) ** (1 / years) - 1),
              "net_sharpe": float(returns.mean() / volatility * math.sqrt(annual)) if volatility > 1e-14 else None,
              "max_drawdown": float(-nav.drawdown.min()), "opportunities": len(trades),
              "opportunities_per_year": len(trades) / years, "exposure_day_fraction": float(nav.any_exposure.mean()),
              "cost_cny": cumulative_cost, "cash_dividends_cny": float(nav.dividends_cny.sum()),
              "profitable_opportunities": sum(t["net_pnl_cny"] > 0 for t in trades),
              "drawdown_brake_triggered": halted}
    metric["meets_numeric_targets"] = metric["net_sharpe"] is not None and metric["net_sharpe"] >= cfg["target_net_sharpe"] and metric["max_drawdown"] <= cfg["target_max_drawdown_magnitude"]
    return nav, trades, decisions, metric


def render_report(out, cfg, summary, scores, comparisons, accounts):
    lines = ["# 510300 LPR预期探索训练结果", "", "本轮已按用户最新指令完成实际训练，取消36个训练月、24个评价月的旧数量门。训练完成不代表夏普目标已达到。", "",
             f"使用{summary['event_months']}个事件月，其中幅度上下界完整的月份为{summary['amount_months']}个；产生{summary['prequential_fit_count']}次逐期拟合和5个期末模型。行情截止{summary['market_end']}，调查截止{summary['last_event_month']}。", "",
             "## 固定训练方法", "", "至少12个已成熟事件开始扩展训练；预测下一开盘至第20个收盘点的现金分红回报。岭回归惩罚为均方损失口径1.0，没有窗口或参数搜索。56月完整保留票数区间；幅度上下界完整的子集另设同样本基准。模型只使用当时已经结束的标签。", "",
             "P_PRICE=价格；A_ACTUAL=价格与实际LPR变化；B_VOTES=再加调查票数落差上下界；A_AMOUNT_MATCH=幅度子集的价格及实际变化基准；C_AMOUNT=该子集再加实际变化相对调查中位数的幅度上下界。", "",
             "## 逐期预测", "", "|模型|预测月数|20日收益RMSE|方向命中率|较成熟历史均值MSE改善|", "|---|---:|---:|---:|---:|"]
    for r in scores.itertuples():
        lines.append(f"|{r.model}|{r.n_predictions}|{r.rmse:.2%}|{r.direction_hit:.2%}|{r.mse_improvement_vs_mean:.2%}|")
    lines += ["", "|增量模型/同月基准|全部月份MSE改善|前半段|后半段|", "|---|---:|---:|---:|"]
    for candidate, reference in comparisons[["candidate", "reference"]].drop_duplicates().itertuples(index=False):
        group = comparisons[(comparisons.candidate == candidate) & (comparisons.reference == reference)].set_index("segment")
        values = [float(group.loc[s, "mse_improvement"]) for s in ["ALL", "FIRST_HALF", "SECOND_HALF"]]
        lines.append(f"|{candidate} / {reference}|{values[0]:.2%}|{values[1]:.2%}|{values[2]:.2%}|")
    lines += ["", "## 20万元完整账户", "", "现金收益及无风险收益均为0，242日年化。信号必须超过压力往返费用及至少6个已成熟逐期预测误差的RMSE；这不是经过校准的胜率保证。仓位按10%年化波动目标限制且不超过本金，固定20日退出；账户回撤8%后次开盘清仓并停止。基准买入持有不采用该停止规则。", "", "|模型|成本|净夏普|最大回撤|年化收益|机会次数|在场天数占比|", "|---|---|---:|---:|---:|---:|---:|"]
    for r in accounts.itertuples():
        sharpe = "未定义（无波动）" if pd.isna(r.net_sharpe) else f"{r.net_sharpe:.3f}"
        lines.append(f"|{r.model}|{r.scenario}|{sharpe}|{r.max_drawdown:.2%}|{r.annualized_return:.2%}|{r.opportunities}|{r.exposure_day_fraction:.2%}|")
    lines += ["", "基础成本为单边万2佣金、最低5元及万5滑点；压力成本为相同佣金及单边千1滑点。两种费用均进入现金、成交、逐日净值。整手100份，持仓中不重复入场，空仓日完整计入。", "",
              "## 结论与范围", "", summary["conclusion"], "", "参数已保存，可以重现本轮逐期预测。该历史行情曾在其他研究中使用，新闻首版未独立认证，本轮没有新增真实前瞻样本。不能把顺序预测称作完全独立样本，更不能以训练内拟合或少量成交宣称确定获利。", "",
              "本轮按用户要求直接训练并完成账户诊断，不用原有月份数量门拦截。28个资料缺失月没有被填成零；已终止的SELECTED_MIX_BAND10_SIMPLE2及其他旧裁决保持原状。10%为账户回撤目标，隔夜跳空及执行偏差使8%停止规则也不能保证未来不超过10%。", "",
              "最新模型对应保存行情截止日；没有2026-09-22当日完整输入，因此本报告不生成当日买入信号。当前真实持仓未知，训练状态不等于账户空仓。", ""]
    (out / "训练结果.md").write_text("\n".join(lines), encoding="utf-8")


def run(out):
    require(not (out / "summary.json").exists(), "已有训练结果，不覆盖。")
    for relative, expected in load(out / "freeze.json")["identities"].items():
        require(digest(out / relative) == expected, f"固定输入或代码变化：{relative}")
    cfg = load(out / "protocol.json")
    market = read_market(out)
    events, excluded = build_events(out, cfg, market)
    print(f"已生成{len(events)}个月的20日标签；幅度区间完整{int(events.amount_cohort.sum())}个月。", flush=True)
    predictions, fitted, final_models = walk_train(cfg, events, market)
    scores, comparisons = prediction_metrics(predictions)
    export(out / "results/事件特征与20日标签.csv", events)
    export(out / "results/缺失月份.csv", excluded)
    export(out / "results/逐期预测.csv", predictions)
    export(out / "results/预测指标.csv", scores)
    export(out / "results/同月增量比较.csv", comparisons)
    save(out / "models/逐期模型.json", fitted)
    save(out / "models/期末已训练模型.json", final_models)
    print(f"完成{len(fitted)}次逐期拟合，已保存5个期末模型；开始计算完整账户。", flush=True)
    start_i = int(predictions[predictions.cohort == "ALL"].entry_i.min())
    all_nav, all_trades, all_decisions, account_metrics = [], [], [], []
    for name in [*cfg["models"], "CASH", "BUY_AND_HOLD"]:
        for scenario in cfg["account"]["slippage_per_side"]:
            nav, trades, decisions, metric = account_run(name, scenario, predictions, market, start_i, cfg)
            all_nav.append(nav)
            all_trades.extend(trades)
            all_decisions.extend(decisions)
            account_metrics.append(metric)
    accounts = pd.DataFrame(account_metrics)
    export(out / "results/完整逐日账户.csv", pd.concat(all_nav, ignore_index=True))
    export(out / "results/机会成交账簿.csv", pd.DataFrame(all_trades))
    export(out / "results/逐事件交易判断.csv", pd.DataFrame(all_decisions))
    export(out / "results/账户指标.csv", accounts)
    candidate_metrics = accounts[accounts.model.isin(cfg["models"])]
    passing = [name for name, group in candidate_metrics.groupby("model") if bool(group.meets_numeric_targets.all())]
    summary = {
        "study_id": cfg["study_id"], "completed_at": now(), "status": "EXPLORATORY_TRAINED",
        "event_months": len(events), "amount_months": int(events.amount_cohort.sum()), "missing_months": len(excluded),
        "first_event_month": events.month.iloc[0], "last_event_month": events.month.iloc[-1],
        "market_end": str(market.date.iloc[-1].date()), "prequential_fit_count": len(fitted), "final_model_count": len(final_models),
        "prediction_rows": len(predictions), "account_scenarios_including_benchmarks": len(accounts),
        "candidate_roundtrips_across_costs": int(candidate_metrics.opportunities.sum()),
        "models_meeting_both_cost_numeric_targets": passing,
        "formal_independent_validation": False, "goal_achieved": False, "orders_authorized": False,
        "conclusion": ("本轮探索历史中有模型同时满足两种费用的数值目标，但少量历史样本不建立独立有效性。" if passing else "本轮已完成实际训练；固定的高门槛事件账户没有模型同时达到两种成本下净夏普1.5、回撤不超过10%的目标。") + "保留全部预测、参数及账户，不为制造交易或漂亮夏普回调阈值。",
    }
    save(out / "summary.json", summary)
    render_report(out, cfg, summary, scores, comparisons, accounts)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def verify(out):
    """只复算保存输出：不下载、不拟合、不重新生成账户。"""
    cfg, frozen = load(out / "protocol.json"), load(out / "freeze.json")
    for name, expected in frozen["identities"].items():
        require(digest(out / name) == expected, f"输入身份不同：{name}")
    predictions = pd.read_csv(out / "results/逐期预测.csv")
    events = pd.read_csv(out / "results/事件特征与20日标签.csv").set_index("month")
    models = load(out / "models/逐期模型.json")
    saved = {(r.model, r.month): r for r in predictions.itertuples()}
    market = read_market(out)
    for model in models:
        row = saved[(model["model"], model["prediction_month"])]
        value = predict(model, events.loc[row.month])
        require(np.isclose(value, row.prediction, atol=1e-12, rtol=0), "保存参数不能重现预测。")
        train_events = events.loc[model["train_months"]]
        require((train_events.label_end_at < row.decision_at).all(), "发现未来标签进入训练。")
        previous = predictions[(predictions.model == row.model) & (predictions.label_end_at < row.decision_at)]
        require(len(previous) == row.mature_prequential_error_count, "误差样本成熟计数不符。")
        if len(previous):
            rmse = math.sqrt(float((previous.prediction - previous.target20).pow(2).mean()))
            require(np.isclose(rmse, row.prior_prediction_rmse, atol=1e-12, rtol=0), "信号误差阈值使用错误。")
    for _, event in events.iterrows():
        block = market.iloc[int(event.entry_i):int(event.exit_i) + 1]
        target = (float(block.close.iloc[-1]) + float(block.dividend.iloc[1:].sum())) / float(block.open.iloc[0]) - 1
        require(np.isclose(target, event.target20, atol=1e-12, rtol=0), "20日标签或分红口径不同。")
    scores, comparisons = prediction_metrics(predictions)
    for computed, filename, keys in [(scores, "预测指标.csv", ["model"]), (comparisons, "同月增量比较.csv", ["candidate", "reference", "segment"])]:
        old = pd.read_csv(out / "results" / filename)
        pd.testing.assert_frame_equal(computed.sort_values(keys).reset_index(drop=True), old.sort_values(keys).reset_index(drop=True), check_dtype=False, atol=1e-12, rtol=1e-10)
    nav = pd.read_csv(out / "results/完整逐日账户.csv")
    trades = pd.read_csv(out / "results/机会成交账簿.csv")
    metrics = pd.read_csv(out / "results/账户指标.csv")
    closes = market.assign(day=market.date.dt.strftime("%Y-%m-%d")).set_index("day").close
    for (name, scenario), daily in nav.groupby(["model", "scenario"], sort=False):
        require(np.allclose(daily.cash_cny + daily.shares * daily.date.map(closes), daily.equity_cny, atol=1e-7, rtol=0), "现金持仓与净值不一致。")
        r = daily.equity_cny.to_numpy() / np.r_[cfg["capital_cny"], daily.equity_cny.to_numpy()[:-1]] - 1
        require(np.allclose(r, daily.daily_return, atol=1e-12, rtol=0), "日收益无法复算。")
        peaks = np.maximum.accumulate(np.r_[cfg["capital_cny"], daily.equity_cny.to_numpy()])[1:]
        dd = daily.equity_cny.to_numpy() / peaks - 1
        metric = metrics[(metrics.model == name) & (metrics.scenario == scenario)].iloc[0]
        require(np.isclose(-dd.min(), metric.max_drawdown, atol=1e-12, rtol=0), "最大回撤不同。")
        if r.std(ddof=1) > 1e-14:
            sharpe = r.mean() / r.std(ddof=1) * math.sqrt(cfg["annual_days"])
            require(np.isclose(sharpe, metric.net_sharpe, atol=1e-9, rtol=0), "全日历夏普不同。")
        else:
            require(pd.isna(metric.net_sharpe), "无波动账户夏普应未定义。")
        selected = trades[(trades.model == name) & (trades.scenario == scenario)]
        require(np.isclose(selected.net_pnl_cny.sum(), daily.equity_cny.iloc[-1] - cfg["capital_cny"], atol=1e-6, rtol=0), "成交损益与期末权益不一致。")
        require(len(selected) == metric.opportunities, "完整机会数不同。")
        require((selected.exit_i > selected.entry_i).all(), "交易不满足T+1。")
        require(np.isclose(daily.cost_cny.sum(), metric.cost_cny, atol=1e-7, rtol=0), "费用汇总不同。")
    return {"status": "PASS_SAVED_OUTPUT_RECOMPUTATION", "checked_prediction_rows": len(predictions),
            "checked_event_labels": len(events), "checked_account_scenarios": len(metrics),
            "new_fits": 0, "new_accounts": 0, "downloads": 0,
            "scope": "输入身份、已保存参数预测、标签成熟、历史误差阈值、分红标签、完整日收益夏普回撤及成交损益。"}


def main():
    parser = argparse.ArgumentParser(description="LPR探索训练及保存结果复算")
    parser.add_argument("command", choices=["freeze", "train", "verify"])
    parser.add_argument("--root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.root)
    elif args.command == "train":
        run(args.root)
    else:
        result = verify(args.root)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
