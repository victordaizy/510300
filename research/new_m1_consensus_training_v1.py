"""新版M1口径实际训练：固定模型、五日目标及完整现金账户。"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import sparse_event_training_core_v1 as core

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_new_m1_consensus_training_v1"
PARENT = ROOT / "reports/research/510300_money_consensus_increment_v2"
SOURCE = ROOT / "reports/research/510300_money_consensus_source_extension_v1"


def freeze(out):
    core.require(not (out / "freeze.json").exists(), "本轮已保存输入，不覆盖。")
    cfg = core.load(out / "protocol.json")
    for record in core.load(PARENT / "freeze_receipt.json")["files"]:
        core.require(core.digest(PARENT / record["path"]) == record["sha256"], f"父级输入变化：{record['path']}")
    for folder in ["inputs/forecasts", "inputs/official", "code"]:
        (out / folder).mkdir(parents=True, exist_ok=True)
    copies = {
        "market.parquet": PARENT / "inputs/market_daily.parquet",
        "all_consensus.csv": PARENT / "inputs/104个月共识选择.csv",
        "official_releases.csv": PARENT / "inputs/official_releases.csv",
        "parent_events.csv": PARENT / "results/全部准入事件_固定五日.csv",
        "parent_pending.csv": PARENT / "results/已准入但行情未成熟.csv",
        "parent_protocol.json": PARENT / "protocol.json",
        "parent_status.json": PARENT / "status.json",
        "parent_freeze_receipt.json": PARENT / "freeze_receipt.json",
        "mandate.json": ROOT / "config/510300_existing_data_training_mandate_v1.json",
    }
    for name, path in copies.items():
        shutil.copy2(path, out / "inputs" / name)
    all_rows = pd.read_csv(out / "inputs/all_consensus.csv")
    coverage = all_rows[all_rows.training_regime == cfg["regime"]].copy()
    core.export(out / "inputs/新版全部月份覆盖.csv", coverage)
    selected = [r for r in core.load(PARENT / "sources/selected_numeric_sources.json") if r["stat_month"] in coverage.stat_month.tolist()]
    core.require(len(selected) == 17, "新版配对资料应为17月。")
    for row in selected:
        core.require(row["source_family"] == "Bualuang", "本轮已有预期来源范围改变。")
        source = SOURCE / "raw" / row["filename"]
        core.require(core.digest(source) == row["sha256"], f"预期原报告不同：{row['stat_month']}")
        shutil.copy2(source, out / "inputs/forecasts" / source.name)
    core.save(out / "inputs/selected_forecasts.json", selected)
    locator = pd.read_csv(PARENT / "sources/104个月原始页面定位.csv")
    official_paths = []
    for row in locator[locator.stat_month.isin(coverage.stat_month)].to_dict("records"):
        source = PARENT / row["package_source_path"]
        core.require(core.digest(source) == row["source_sha256"], "官方原始页面不同。")
        destination = out / "inputs/official" / source.name
        shutil.copy2(source, destination)
        row["local_source_path"] = destination.relative_to(out).as_posix()
        official_paths.append(row)
    core.export(out / "inputs/official_source_paths.csv", pd.DataFrame(official_paths))
    for source in [Path(__file__), Path(core.__file__)]:
        shutil.copy2(source, out / "code" / source.name)
    files = [out / "protocol.json", *[p for p in (out / "inputs").rglob("*") if p.is_file()], *list((out / "code").iterdir())]
    core.save(out / "freeze.json", {"frozen_at": core.now(), "before_this_round_fits_and_accounts": True,
        "parent_historical_labels_already_existed": True, "new_downloads": 0,
        "new_regime_months": len(coverage), "paired_forecast_months": len(selected),
        "user_override": "原新口径样本门未运行；按直接训练指令从6个成熟事件启动，不改旧口径拒绝。",
        "identities": {p.relative_to(out).as_posix(): core.digest(p) for p in files}})
    print("新版17个月预期、官方原文与固定参数已保存；开始实际训练。", flush=True)


def make_events(out, cfg):
    market = core.read_market(out)
    dates = pd.DatetimeIndex(market.date)
    opens = dates.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=30)
    closes = dates.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
    coverage = pd.read_csv(out / "inputs/新版全部月份覆盖.csv")
    rows, pending = [], []
    for raw in coverage[coverage.consensus_admitted].to_dict("records"):
        published = pd.Timestamp(raw["available_at_upper_bound"])
        forecast_upper = pd.Timestamp(raw["forecast_date"]).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=23, minutes=59, seconds=59)
        core.require(forecast_upper < published, "预期报告必须在公布之前。")
        observation_i = int(opens.searchsorted(published, side="right"))
        entry_i, exit_i = observation_i + 1, observation_i + cfg["horizon_sessions"]
        if exit_i >= len(market):
            pending.append(dict(raw, label_status="PENDING_MARKET_DATA", prediction_status="PENDING_OBSERVATION_PRICE_INPUT"))
            continue
        pre_i = int(closes.searchsorted(published, side="left")) - 1
        core.require(pre_i >= 20 and pre_i < observation_i, "公告之前价格不足。")
        pre, observed = market.iloc[pre_i], market.iloc[observation_i]
        block = market.iloc[entry_i:exit_i + 1]
        sigma = float(market.iloc[observation_i - 19:observation_i + 1].total_simple.std(ddof=1))
        target = (float(block.close.iloc[-1]) + float(block.dividend.iloc[1:].sum())) / float(block.open.iloc[0]) - 1
        row = dict(raw, month=raw["stat_month"], available_at=published.isoformat(),
            forecast_available_upper=forecast_upper.isoformat(), observation_i=observation_i,
            decision_at=closes[observation_i].isoformat(), entry_i=entry_i, exit_i=exit_i,
            entry_date=str(dates[entry_i].date()), exit_date=str(dates[exit_i].date()),
            label_end_at=closes[exit_i].isoformat(), reference_close=float(observed.close), vol20=sigma,
            price_pre20_return=float(pre.wealth / market.iloc[pre_i - 20].wealth - 1),
            initial_response=float((observed.close + market.iloc[pre_i + 1:observation_i + 1].dividend.sum()) / pre.close - 1),
            price_log_sigma20=math.log(sigma), target5=target)
        core.require(abs(row["spread_surprise_proxy_pp"] - (raw["m1_yoy_pp"] - raw["m2_yoy_pp"] - raw["expected_spread_proxy_pp"])) < 1e-12, "预期差公式不一致。")
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("decision_at").reset_index(drop=True)
    core.require(len(frame) == 16 and len(pending) == 1, "既有16成熟月、1待行情月范围不同。")
    core.require((frame.entry_i.iloc[1:].to_numpy() > frame.exit_i.iloc[:-1].to_numpy()).all(), "新版事件窗口存在重叠。")
    parent = pd.read_csv(out / "inputs/parent_events.csv").set_index("stat_month")
    for row in frame.itertuples():
        reference = parent.loc[row.month]
        for field in ["price_pre20_return", "initial_response", "price_log_sigma20", "spread_surprise_proxy_pp"]:
            core.require(abs(getattr(row, field) - reference[field]) < 1e-12, f"父级价格或消息特征变化：{row.month}/{field}")
        core.require(abs(row.target5 - reference.residual_5d_gross_return) < 1e-12, "五日目标不同。")
        core.require(row.entry_date == reference.entry_date and row.exit_date == reference.exit_date, "父级执行日期不同。")
    return frame, pd.DataFrame(pending), market


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
                "label_end_at", "reference_close", "vol20", "target5", "spread_surprise_proxy_pp"]}
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
    matched = predictions[predictions.model == "B_PRICE_SURPRISE"].merge(
        predictions[predictions.model == "A_PRICE"][["month", "prediction"]], on="month", suffixes=("", "_reference")).sort_values("decision_at")
    comparisons = []
    for name, sample in [("ALL", matched), ("FIRST_HALF", matched.iloc[:len(matched) // 2]), ("SECOND_HALF", matched.iloc[len(matched) // 2:])]:
        candidate_mse = float((sample.target5 - sample.prediction).pow(2).mean())
        reference_mse = float((sample.target5 - sample.prediction_reference).pow(2).mean())
        comparisons.append({"segment": name, "n": len(sample), "candidate_mse": candidate_mse,
            "reference_mse": reference_mse, "mse_improvement": 1 - candidate_mse / reference_mse})
    return pd.DataFrame(metrics), pd.DataFrame(comparisons)


def write_report(out, summary, metrics, comparisons, accounts):
    lines = ["# 新版M1/M2预期差：实际训练与账户", "", summary["conclusion"], "",
        "仅用已有本地资料：新版20个月母集中17个月有配对预期，16个月已成熟，2026年8月所属期缺后续行情与观察日价格。2025年1月、7月、11月没有配对来源，未补值。旧口径不混入，也未改变其冻结拒绝。", "",
        "按用户直接训练指令，从6个成熟事件开始。保留原5个交易日目标、公告后首个完整交易日收盘观察及次日开盘时钟；标准化仅用成熟训练样本，岭回归平均MSE惩罚固定为1。A用公告前20日回报、公布后的初始反应、20日波动；B再加M1与M2增速差的预期偏差。另列成熟训练均值对照。预期为已存原报告代理，两项预期之差不等同于直接调查得到的剪刀差预期。", "",
        "完成20次逐期岭回归拟合、10次均值更新；期末另保存2个全成熟样本岭模型与1个均值基准。每组10次历史顺序预测，期末模型没有回填历史决策。训练时期为2025年3月起，账户评价从2025年10月17日开始，至已有行情2026年9月11日。", "",
        "|模型|预测数|RMSE|方向命中率|相对成熟均值MSE改善|", "|---|---:|---:|---:|---:|"]
    for r in metrics.itertuples():
        lines.append(f"|{r.model}|{r.n_predictions}|{r.rmse:.2%}|{r.direction_hit:.2%}|{r.mse_improvement_vs_mean:.2%}|")
    lines += ["", "|B相对A|事件数|MSE改善|", "|---|---:|---:|"]
    for r in comparisons.itertuples():
        lines.append(f"|{r.segment}|{r.n}|{r.mse_improvement:.2%}|")
    lines += ["", "主规则ERROR_BUFFER_PRIMARY要求预测覆盖压力往返成本和至少6个成熟逐期误差的RMSE；NET_POSITIVE_DIAGNOSTIC仅要求覆盖压力成本，用于检查经济意义。两条规则均在本轮拟合前指定，不因结果换门槛。", "",
        "全部账户20万元、510300与现金、100份整手、T+1；佣金单边万2且最低5元，基础/压力滑点万5/千1。按观察收盘20日波动控制10%目标年化波动，仓位不超过100%。固定五日收盘退出；收盘账户回撤8%时次开盘清仓并停止。每天包括空仓，242日年化，现金和无风险收益均设0。", "",
        "|模型|规则|成本|净夏普|最大回撤|年化收益|机会数|期末资金|", "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in accounts.itertuples():
        sh = "未定义" if pd.isna(r.net_sharpe) else f"{r.net_sharpe:.3f}"
        lines.append(f"|{r.model}|{r.policy}|{r.scenario}|{sh}|{r.max_drawdown:.2%}|{r.annualized_return:.2%}|{r.opportunities}|{r.ending_equity_cny:,.2f}|")
    lines += ["", "模型决策使用预期与官方公布的历史时钟，网页和PDF均为后续保存，历史首版没有认证；这仍是历史探索，不是新前瞻样本。只有10次评价且不到一年账户，不能凭单个指标点值认定高置信度。费用对照与均值对照不自动晋升主策略。", "",
        "2026-01-19入场发生在除息日，当日买入不享有0.123元分红，目标和策略账户均未计入；买入持有基准享有除息权益，现金到账时差没有单独模拟，但该基准不再投资或使用分红买入。滑点使用连续价格近似，未模拟最小价位和涨跌停队列。8%警戒并不保证未来跳空时回撤不超过10%。", "",
        "最新已存行情为2026-09-11，未用9月14日发布的8月数据生成缺少观察价格的当前信号。没有采集、订单或已终止策略恢复。", ""]
    (out / "训练结果.md").write_text("\n".join(lines), encoding="utf-8")


def run(out):
    core.require(not (out / "summary.json").exists(), "训练已完成，不覆盖。")
    for name, expected in core.load(out / "freeze.json")["identities"].items():
        core.require(core.digest(out / name) == expected, f"本轮固定输入变化：{name}")
    cfg = core.load(out / "protocol.json")
    events, pending, market = make_events(out, cfg)
    predictions, models, final = train(events, market, cfg)
    metrics, comparisons = score(predictions)
    for name, frame in {"事件特征与五日标签": events, "待行情事件": pending,
            "逐期预测": predictions, "预测指标": metrics, "消息增量比较": comparisons}.items():
        core.export(out / "results" / f"{name}.csv", frame)
    core.save(out / "models/逐期模型.json", models)
    core.save(out / "models/期末已训练模型.json", final)
    print("已完成20次逐期岭拟合和10次均值更新，开始完整账户。", flush=True)
    start_i = int(predictions.entry_i.min())
    paths, trades, decisions, account_metrics = [], [], [], []
    for policy, settings in cfg["policies"].items():
        effective = copy.deepcopy(cfg)
        effective["account"].update(settings)
        for name in cfg["models"]:
            for scenario in cfg["account"]["slippage_per_side"]:
                nav, trade, decision, metric = core.account_run(name, scenario, predictions, market, start_i, effective)
                nav["policy"] = policy
                for row in [*trade, *decision, metric]:
                    row["policy"] = policy
                paths.append(nav)
                trades.extend(trade)
                decisions.extend(decision)
                account_metrics.append(metric)
    for name in ["CASH", "BUY_AND_HOLD"]:
        for scenario in cfg["account"]["slippage_per_side"]:
            nav, trade, decision, metric = core.account_run(name, scenario, predictions, market, start_i, cfg)
            nav["policy"] = "BENCHMARK"
            for row in [*trade, metric]:
                row["policy"] = "BENCHMARK"
            paths.append(nav)
            trades.extend(trade)
            account_metrics.append(metric)
    accounts = pd.DataFrame(account_metrics)
    for name, frame in {"完整逐日账户": pd.concat(paths, ignore_index=True), "机会成交账簿": pd.DataFrame(trades),
            "逐事件交易判断": pd.DataFrame(decisions), "账户指标": accounts}.items():
        core.export(out / "results" / f"{name}.csv", frame)
    passing = []
    for (model, policy), group in accounts[accounts.policy != "BENCHMARK"].groupby(["model", "policy"]):
        if group.meets_numeric_targets.all():
            passing.append({"model": model, "policy": policy})
    summary = {"study_id": cfg["study_id"], "completed_at": core.now(), "status": "EXPLORATORY_TRAINED",
        "new_regime_months": 20, "paired_months": 17, "mature_months": len(events), "pending_months": len(pending),
        "prequential_ridge_fits": sum(m["model_type"] == "RIDGE" for m in models),
        "prequential_mean_updates": sum(m["model_type"] == "MATURE_MEAN" for m in models),
        "final_ridge_models": 2, "final_mean_references": 1, "prediction_rows": len(predictions),
        "account_scenarios": len(accounts), "account_start": str(market.date.iloc[start_i].date()),
        "market_end": str(market.date.iloc[-1].date()), "numeric_targets_pass_both_costs": passing,
        "goal_achieved": False, "true_forward_events": 0, "new_downloads": 0, "orders_authorized": False,
        "previous_goal_turn_classification": "PROGRESS_POLICY_ACCOUNTS_COMPLETED",
        "conclusion": "新版M1/M2预期差已完成实际训练和完整账户。" + ("存在历史数值达标组合，但样本短且没有独立验证。" if passing else "没有组合同时达到两档成本下净夏普1.5、最大回撤不超过10%。")}
    core.save(out / "summary.json", summary)
    write_report(out, summary, metrics, comparisons, accounts)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(metrics.to_string(index=False), flush=True)
    print(comparisons.to_string(index=False), flush=True)
    print(accounts[["model", "policy", "scenario", "net_sharpe", "max_drawdown", "opportunities", "ending_equity_cny"]].to_string(index=False), flush=True)


def verify(out):
    cfg = core.load(out / "protocol.json")
    for name, expected in core.load(out / "freeze.json")["identities"].items():
        core.require(core.digest(out / name) == expected, f"保存输入身份不同：{name}")
    events, pending, market = make_events(out, cfg)
    saved_events = pd.read_csv(out / "results/事件特征与五日标签.csv")
    pd.testing.assert_frame_equal(events, saved_events, check_dtype=False, check_like=True, rtol=1e-10, atol=1e-12)
    indexed = events.set_index("month")
    predictions = pd.read_csv(out / "results/逐期预测.csv")
    models = core.load(out / "models/逐期模型.json")
    final = core.load(out / "models/期末已训练模型.json")
    for model in [*models, *final.values()]:
        data = indexed.loc[model["train_months"]]
        limit = model.get("decision_at", model.get("as_of"))
        core.require((data.label_end_at < limit).all(), "模型使用了未成熟标签。")
        columns = model["columns"]
        x, y = data[columns].to_numpy(float), data.target5.to_numpy(float)
        center, scale, beta = [np.asarray(model[k]) for k in ["center", "scale", "coefficients_standardized"]]
        core.require(np.allclose(center, x.mean(axis=0), rtol=0, atol=1e-12), "训练中心不同。")
        actual_scale = np.where(x.std(axis=0, ddof=0) >= 1e-12, x.std(axis=0, ddof=0), 1.0)
        core.require(np.allclose(scale, actual_scale, rtol=0, atol=1e-12), "训练标准化尺度不同。")
        z = (x - center) / scale
        z[:, ~np.asarray(model["active"], bool)] = 0.0
        left = (z.T @ z / len(data) + cfg["ridge_mean_loss_penalty"] * np.eye(len(columns))) @ beta
        right = z.T @ (y - model["intercept"]) / len(data)
        core.require(np.allclose(left, right, rtol=0, atol=1e-12), "保存参数不满足固定岭回归方程。")
        core.require(abs(model["intercept"] - y.mean()) < 1e-12, "训练均值不同。")
        if "prediction_month" in model:
            row = predictions[(predictions.model == model["model"]) & (predictions.month == model["prediction_month"])].iloc[0]
            core.require(abs(core.predict(model, indexed.loc[row.month]) - row.prediction) < 1e-12, "保存参数不能重现预测。")
            errors = predictions[(predictions.model == row.model) & (predictions.label_end_at < row.decision_at)]
            core.require(len(errors) == row.mature_prequential_error_count, "成熟误差数量不同。")
            if len(errors):
                rmse = math.sqrt(float((errors.target5 - errors.prediction).pow(2).mean()))
                core.require(abs(rmse - row.prior_prediction_rmse) < 1e-12, "误差缓冲不同。")
    metrics, comparisons = score(predictions)
    for actual, name in [(metrics, "预测指标"), (comparisons, "消息增量比较")]:
        pd.testing.assert_frame_equal(actual, pd.read_csv(out / "results" / f"{name}.csv"), check_dtype=False, rtol=1e-10, atol=1e-12)
    nav = pd.read_csv(out / "results/完整逐日账户.csv")
    trades = pd.read_csv(out / "results/机会成交账簿.csv")
    accounts = pd.read_csv(out / "results/账户指标.csv")
    prices = market.assign(day=market.date.dt.strftime("%Y-%m-%d")).set_index("day").close
    for (model, policy, scenario), daily in nav.groupby(["model", "policy", "scenario"]):
        metric = accounts[(accounts.model == model) & (accounts.policy == policy) & (accounts.scenario == scenario)].iloc[0]
        equity = daily.equity_cny.to_numpy(float)
        returns = equity / np.r_[cfg["capital_cny"], equity[:-1]] - 1
        peak = np.maximum.accumulate(np.r_[cfg["capital_cny"], equity])[1:]
        core.require(np.allclose(daily.cash_cny + daily.shares * daily.date.map(prices), equity, rtol=0, atol=1e-7), "账户市值不一致。")
        core.require(np.allclose(returns, daily.daily_return, rtol=0, atol=1e-12), "日收益不同。")
        core.require(abs(-(equity / peak - 1).min() - metric.max_drawdown) < 1e-12, "最大回撤不同。")
        if returns.std(ddof=1) > 1e-14:
            core.require(abs(returns.mean() / returns.std(ddof=1) * math.sqrt(cfg["annual_days"]) - metric.net_sharpe) < 1e-10, "全账户夏普不同。")
        else:
            core.require(pd.isna(metric.net_sharpe), "空仓夏普应未定义。")
        selected = trades[(trades.model == model) & (trades.policy == policy) & (trades.scenario == scenario)]
        core.require(len(selected) == metric.opportunities, "完整交易数不同。")
        core.require(abs(selected.net_pnl_cny.sum() - (equity[-1] - cfg["capital_cny"])) < 1e-7, "账户总损益不能对账。")
        for t in selected.itertuples():
            core.require(t.exit_i > t.entry_i, "T+1不满足。")
            dividend = float(market.iloc[int(t.entry_i) + 1:int(t.exit_i) + 1].dividend.sum()) * t.shares
            pnl = t.shares * t.exit_fill_price - t.exit_commission_cny + dividend - t.entry_cash_debit
            core.require(abs(pnl - t.net_pnl_cny) < 1e-7, "交易现金流不同。")
    return {"status": "PASS_SAVED_MODELS_LABELS_AND_ACCOUNTS", "saved_models_and_mean_snapshots": len(models) + len(final),
        "prediction_rows": len(predictions), "mature_events": len(events), "pending_events": len(pending),
        "account_scenarios": len(accounts), "new_model_fits": 0, "new_accounts": 0, "new_downloads": 0}


def main():
    parser = argparse.ArgumentParser(description="新版M1/M2现有资料实际训练。")
    parser.add_argument("command", choices=["freeze", "train", "verify"])
    parser.add_argument("--root", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.root)
    elif args.command == "train":
        run(args.root)
    else:
        print(json.dumps(verify(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
