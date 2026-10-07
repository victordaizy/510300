"""IF 压力耗竭的本地探索训练：固定事件、十日期限、逐期成熟标签。"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import yaml

import sparse_event_training_core_v3 as core
import if_exhaustion_local_features_v1 as local

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_if_exhaustion_existing_training_v1"


def freeze(out):
    core.require(not (out / "freeze.json").exists(), "本轮已冻结，不能覆盖。")
    cfg = core.load(out / "protocol.json")
    parent = yaml.safe_load((ROOT / "config/510300_direction_switch_v1.yaml").read_text(encoding="utf-8"))
    copies = {
        "market.parquet": ROOT / "reports/research/510300_private_manager_intent_v1/inputs/market.parquet",
        "dividends.csv": ROOT / "data/reference/510300_dividends.csv",
        "if_saved_features.parquet": ROOT / "data/research/510300_if_forced_flow_state_v1/features.parquet",
        "parent_protocol.yaml": ROOT / "config/510300_direction_switch_v1.yaml",
        "parent_result.json": ROOT / "reports/research/510300_if_forced_flow_state_v1.json",
        "parent_umbrella_result.json": ROOT / "reports/research/510300_direction_switch_v1_result.json",
        "prior_oi_increment_rejection.json": ROOT / "reports/research/510300_if_open_interest_increment_v1/result.json",
        "mandate.json": ROOT / "config/510300_existing_data_training_mandate_v1.json",
    }
    for field in ["if_contract_daily", "if_contract_expiry", "if_spot_index", "if_acquisition_receipt", "if_execution_etf"]:
        source = ROOT / parent["inputs"][field]["file"]
        core.require(core.digest(source) == parent["inputs"][field]["sha256"], "原 IF 输入身份不同。")
        copies[field + source.suffix] = source
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    (out / "code").mkdir(exist_ok=True)
    for name, source in copies.items():
        shutil.copy2(source, out / "inputs" / name)
    old = core.load(out / "inputs/parent_umbrella_result.json")
    expected = old["artifact_hashes"]["data/research/510300_if_forced_flow_state_v1/features.parquet"]
    core.require(core.digest(out / "inputs/if_saved_features.parquet") == expected, "旧保存状态发生变化。")
    for source in [Path(__file__), Path(core.__file__), Path(local.__file__)]:
        shutil.copy2(source, out / "code" / source.name)
    for source in [ROOT / "research/if_forced_flow_state_v1.py", ROOT / "research/direction_switch_common_v1.py"]:
        shutil.copy2(source, out / "code" / ("parent_reference_" + source.name))
    files = [out / "protocol.json", *list((out / "inputs").iterdir()), *list((out / "code").iterdir())]
    core.save(out / "freeze.json", {"frozen_at": core.now(), "before_this_round_labels_and_fits": True,
        "historical_prices_seen": True, "event_dates_seen": True, "independent_holdout": False,
        "old_failure_preserved": True, "user_instruction": cfg["user_instruction"],
        "identities": {p.relative_to(out).as_posix(): core.digest(p) for p in files}})
    print("固定事件、月末训练样本、十日期限及账户设定已保存；开始实际训练。", flush=True)


def check_inputs(out):
    for name, expected in core.load(out / "freeze.json")["identities"].items():
        core.require(core.digest(out / name) == expected, f"输入或代码变化：{name}")


def rebuild_features(out):
    """从已存 IF 合约重算因子和状态，不访问外部接口。"""
    cfg = yaml.safe_load((out / "inputs/parent_protocol.yaml").read_text(encoding="utf-8"))
    saved = pd.read_parquet(out / "inputs/if_saved_features.parquet")
    contracts = pd.read_parquet(out / "inputs/if_contract_daily.parquet")
    expiry = pd.read_parquet(out / "inputs/if_contract_expiry.parquet")
    contracts["date"] = pd.to_datetime(contracts.date).dt.normalize()
    expiry["expiry_date"] = pd.to_datetime(expiry.expiry_date).dt.normalize()
    joined = contracts.merge(expiry[["symbol", "expiry_date"]], on="symbol", how="left", validate="many_to_one")
    joined["calendar_days_to_expiry"] = (joined.expiry_date - joined.date).dt.days
    spot = pd.read_parquet(out / "inputs/if_spot_index.parquet")
    spot["date"] = pd.to_datetime(spot.date).dt.normalize()
    spot = spot[spot.date.between(cfg["data_scope"]["if_feature_warmup_start"], cfg["data_scope"]["if_evaluation_end"])].sort_values("date").reset_index(drop=True)
    market_columns = list(saved.columns[:saved.columns.get_loc("etf_past_3d_total_return")])
    market = saved[market_columns].copy()
    previous = market.close.shift(1)
    total = (market.close + market.cash_dividend_per_share) / previous - 1
    core.require(np.allclose(total, market.total_return, equal_nan=True, atol=1e-12, rtol=0), "旧市场分红收益不一致。")
    _, rebuilt, audit = local.build_if_features(cfg, {"joined_contracts": joined, "spot": spot, "market": market})
    rebuilt = local.build_state_machine(cfg, rebuilt)
    for col in ["basis_residual", "aggregate_open_interest", "open_interest_shock", "basis_residual_percentile",
                "open_interest_shock_percentile", "etf_past_5d_total_return", "etf_realized_volatility_20d",
                "etf_same_day_total_return", "etf_log_amount_vs_trailing_60d_median"]:
        core.require(np.allclose(saved[col], rebuilt[col], equal_nan=True, atol=1e-12, rtol=0), f"原因子重算不同：{col}")
    for col in ["state", "data_eligible", "pressure_start_event", "pressure_exhaustion_event"]:
        core.require(saved[col].equals(rebuilt[col]), f"原状态重算不同：{col}")
    return saved, audit


def make_rows(out, cfg):
    full = core.read_market(out)
    market = full[full.date <= cfg["account_end"]].reset_index(drop=True)
    features = pd.read_parquet(out / "inputs/if_saved_features.parquet").sort_values("date").reset_index(drop=True)
    features["basis_oi_interaction"] = features.basis_residual_percentile * features.open_interest_shock_percentile
    calendar = pd.DatetimeIndex(full.date)
    monthends = set(full.groupby(full.date.dt.to_period("M"), sort=True).tail(1).date)
    feature_columns = sum(cfg["features"].values(), [])
    common = features.data_eligible & (features.date <= cfg["signal_end"])
    groups = [features[common & features.date.isin(monthends)], features[common & features.pressure_exhaustion_event]]
    outputs = []
    for candidates in groups:
        rows = []
        for f in candidates.itertuples():
            signal_i = int(calendar.get_loc(f.date))
            entry_i, exit_i = signal_i + 1, signal_i + cfg["horizon_sessions"]
            core.require(exit_i < len(market), "已有行情不足以完成十日标签。")
            block = market.iloc[entry_i:exit_i + 1]
            target = (float(block.close.iloc[-1]) + float(block.dividend.iloc[1:].sum())) / float(block.open.iloc[0]) - 1
            known = market.iloc[signal_i]
            row = {"month": str(f.date.date()), "event_date": str(f.date.date()),
                "feature_source_date": str(f.feature_source_date.date()),
                "decision_at": (f.date.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15, minutes=1)).isoformat(),
                "entry_i": entry_i, "exit_i": exit_i, "entry_date": str(calendar[entry_i].date()), "exit_date": str(calendar[exit_i].date()),
                "label_end_at": (calendar[exit_i].tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)).isoformat(),
                "reference_close": float(known.close), "vol20": float(known.vol20), "target10": target, "label_status": "MATURE"}
            row.update({col: float(getattr(f, col)) for col in feature_columns})
            rows.append(row)
        frame = pd.DataFrame(rows)
        core.require(np.isfinite(frame[feature_columns + ["vol20", "target10"]].to_numpy()).all(), "训练样本存在缺失或无穷值。")
        outputs.append(frame)
    return outputs[0], outputs[1], market


def train(monthly, events, cfg):
    snapshots, predictions, calibrations, final = [], [], [], {}
    cutoff = pd.Timestamp(cfg["signal_end"], tz="Asia/Shanghai").replace(hour=15, minute=2).isoformat()
    for name, spec in cfg["models"].items():
        columns = sum([cfg["features"][g] for g in spec["groups"]], [])
        prior = []
        for role, rows in [("MONTHLY_CALIBRATION", monthly), ("ENTRY_EVENT", events)]:
            for _, row in rows.iterrows():
                sample = monthly[monthly.label_end_at < row.decision_at]
                if len(sample) < cfg["min_train"]:
                    continue
                model = core.fit_ridge(sample, columns, cfg["ridge_mean_loss_penalty"], target_column="target10")
                model.update({"model": name, "model_type": "RIDGE" if columns else "MATURE_MEAN",
                    "prediction_month": row.month, "decision_at": row.decision_at, "prediction_role": role})
                errors = [r for r in prior if r["label_end_at"] < row.decision_at]
                rmse = math.sqrt(float(np.mean([(r["prediction"] - r["target10"]) ** 2 for r in errors]))) if errors else None
                record = {field: row[field] for field in ["month", "decision_at", "entry_date", "exit_date", "entry_i", "exit_i",
                    "label_end_at", "reference_close", "vol20", "target10", "label_status"]}
                record.update({"model": name, "prediction": core.predict(model, row), "mean_prediction": model["intercept"],
                    "n_train": len(sample), "train_max_label_end_at": model["maximum_training_label_end_at"],
                    "prediction_role": role, "mature_prequential_error_count": len(errors), "prior_prediction_rmse": rmse})
                snapshots.append(model)
                if role == "MONTHLY_CALIBRATION":
                    prior.append(record)
                    calibrations.append(record)
                else:
                    predictions.append(record)
        mature = monthly[monthly.label_end_at < cutoff]
        model = core.fit_ridge(mature, columns, cfg["ridge_mean_loss_penalty"], target_column="target10")
        model.update({"model": name, "model_type": "RIDGE" if columns else "MATURE_MEAN", "as_of": cutoff,
            "status": "EXPLORATORY_TRAINED_NO_CURRENT_SIGNAL"})
        final[name] = model
    return pd.DataFrame(predictions), pd.DataFrame(calibrations), snapshots, final


def score(predictions):
    metrics = []
    for model, sample in predictions.groupby("model", sort=True):
        mse = float((sample.target10 - sample.prediction).pow(2).mean())
        mean_mse = float((sample.target10 - sample.mean_prediction).pow(2).mean())
        metrics.append({"model": model, "n": len(sample), "mse": mse, "rmse": math.sqrt(mse),
            "direction_hit": float(((sample.target10 > 0) == (sample.prediction > 0)).mean()),
            "mse_improvement_vs_mean": 1 - mse / mean_mse})
    paired = predictions[predictions.model == "B_IF_EXHAUSTION"].merge(
        predictions[predictions.model == "A_PRICE"][["month", "prediction"]], on="month", suffixes=("", "_price")).sort_values("decision_at")
    comparisons = []
    for segment, sample in [("ALL", paired), ("FIRST_HALF", paired.iloc[:len(paired) // 2]), ("SECOND_HALF", paired.iloc[len(paired) // 2:])]:
        mse = float((sample.target10 - sample.prediction).pow(2).mean())
        reference = float((sample.target10 - sample.prediction_price).pow(2).mean())
        comparisons.append({"segment": segment, "n": len(sample), "mse_improvement_vs_price": 1 - mse / reference})
    return pd.DataFrame(metrics), pd.DataFrame(comparisons)


def run(out):
    core.require(not (out / "summary.json").exists(), "本轮已完成，不重复拟合或回测。")
    check_inputs(out)
    cfg = core.load(out / "protocol.json")
    features, audit = rebuild_features(out)
    core.save(out / "feature_recomputation.json", audit)
    monthly, events, market = make_rows(out, cfg)
    predictions, calibrations, snapshots, final = train(monthly, events, cfg)
    for name, frame in {"月末训练特征与十日标签": monthly, "候选事件与十日标签": events,
            "月末逐期预测": calibrations, "事件逐期预测": predictions}.items():
        core.export(out / "results" / f"{name}.csv", frame)
    core.save(out / "models/逐期模型.json", snapshots)
    core.save(out / "models/期末已训练模型.json", final)
    for prefix, frame in [("月末", calibrations), ("事件", predictions)]:
        metrics, comparisons = score(frame)
        core.export(out / "results" / f"{prefix}预测指标.csv", metrics)
        core.export(out / "results" / f"{prefix}增量比较.csv", comparisons)
    print(f"已完成 {len(snapshots)} 份逐期参数和均值快照，开始连续账户。", flush=True)
    dividends = pd.read_csv(out / "inputs/dividends.csv")
    dividends = dividends[dividends.symbol == "510300.SH"]
    dividend_map = dividends.assign(date=pd.to_datetime(dividends.ex_date)).set_index("date").cash_dividend_per_share
    for row in market[market.dividend > 0].itertuples():
        core.require(row.date in dividend_map.index and abs(dividend_map.loc[row.date] - row.dividend) < 1e-12, "付款表与市场分红权益不同。")
    start_i = int(calibrations.entry_i.min())
    core.require(int(events.entry_i.min()) >= start_i, "事件发生在模型预热完成前，须保留未预测状态。")
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
    passing = []
    for (model, policy), group in accounts[accounts.policy != "BENCHMARK"].groupby(["model", "policy"]):
        if group.meets_numeric_targets.all():
            passing.append({"model": model, "policy": policy})
    summary = {"study_id": cfg["study_id"], "completed_at": core.now(), "status": "EXPLORATORY_EXISTING_IF_EXHAUSTION_TRAINED",
        "candidate_events": len(events), "mature_event_labels": len(events), "monthly_training_rows": len(monthly),
        "monthly_prediction_rows": len(calibrations), "event_prediction_rows": len(predictions),
        "prequential_ridge_fits": sum(m["model_type"] == "RIDGE" for m in snapshots),
        "prequential_mean_updates": sum(m["model_type"] == "MATURE_MEAN" for m in snapshots),
        "final_ridge_models": 2, "final_mean_references": 1, "account_scenarios": len(accounts),
        "account_start": str(market.date.iloc[start_i].date()), "signal_end": cfg["signal_end"], "account_end": cfg["account_end"],
        "complete_calendar_sessions": len(market) - start_i, "numeric_targets_pass_both_costs": passing,
        "independent_forward_events": 0, "goal_achieved": False, "new_downloads": 0, "orders_authorized": False,
        "conclusion": "已完成 IF 压力耗竭事件的实际训练和连续账户。" + ("历史上存在数值达线组合，尚不构成独立有效证据。" if passing else "没有组合同时达到两档成本下净夏普1.5、最大回撤不超过10%。")}
    core.save(out / "summary.json", summary)
    report(out, cfg, summary, accounts)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(accounts[["model", "policy", "scenario", "net_sharpe", "max_drawdown", "opportunities", "ending_equity_cny"]].to_string(index=False))


def report(out, cfg, summary, accounts):
    lines = ["# IF 卖压缓解后的稀疏机会训练", "", summary["conclusion"], "",
        "沿用旧 IF 基差压力和持仓冲击状态机的10个耗竭事件。原研究因每类至少20个独立事件而在收益检验前关闭；用户要求直接训练、放宽限制后，本轮单独做探索，旧关闭记录不改。此前一般 IF 持仓增量模型已失败，因此这轮属于同一家族后续尝试，不能重置试验次数或称作独立样本。", "",
        "原状态：ETF过去3日收益为负，季节调整基差的历史分位不超过5%，三日总持仓变化的分位至少70%，进入压力状态；随后连续两天基差分位至少25%、持仓变化分位不超过50%，才记一次缓解。分位仅用之前756个交易日、至少504个有效观察；同日公开时点无法证明，原IF数据整体滞后一个交易日。这里只能称市场压力代理，不能识别真实卖方或保证强制卖出已经结束。", "",
        "训练样本为固定日历月末的所有合格状态，从24个已成熟十日标签起扩展训练。A使用原先的过去5日回报、20日波动、当日回报和成交额相对过去60日中位数；B再加入滞后基差分位、持仓冲击分位及其乘积；另存成熟训练均值。惩罚固定1，标准化只用成熟训练样本，不搜索窗口或门槛。每次训练标签结束时刻严格早于预测时刻。", "",
        f"保存{summary['prequential_ridge_fits']}次逐期岭拟合、{summary['prequential_mean_updates']}次逐期均值更新，期末另存2个岭模型及1个均值。{summary['monthly_training_rows']}个月末样本，{summary['monthly_prediction_rows']}条月末预测、{summary['event_prediction_rows']}条事件预测。事件只有10个，误差缓冲来自过去成熟月末预测，不是10个事件的独立置信界。", ""]
    for prefix in ["月末", "事件"]:
        comparison = pd.read_csv(out / "results" / f"{prefix}增量比较.csv")
        lines += [f"{prefix}预测中B相对A的MSE改善：", "", "|区间|样本数|MSE改善|", "|---|---:|---:|"]
        for r in comparison.itertuples():
            lines.append(f"|{r.segment}|{r.n}|{r.mse_improvement_vs_price:.2%}|")
        lines.append("")
    lines += [f"账户从{summary['account_start']}模型完成预热后的首个交易日开始，到{summary['account_end']}，完整{summary['complete_calendar_sessions']}个交易日，计入全部空仓日。IF信号只截止{cfg['signal_end']}，其后仅给已进入的最后事件完成预定10日持有，不评估新信号；期末模型仍按信号截止日已成熟标签训练。", "",
        "仅交易510300和现金，本金20万元，预测超过压力往返费用加历史预测RMSE才进入主规则；另保留仅覆盖费用的对照。不筛选事件的固定规则只作诊断，没有拟造预测概率。次开盘入场，固定第10个持有交易日收盘退出；10%年化波动目标、最大仓位100%、100份整手、不融资、T+1。收盘回撤8%后次开盘清仓并停止，不能保证跳空情况下回撤一定小于10%。", "",
        "佣金单边万2最低5元；基础/压力滑点分别单边万5/千1；现金和无风险收益设为0，242日年化。分红除息记应收，实际付款日才加入可用现金。最小报价单位、涨跌停排队与实际冲击未模拟，20万元不自动消除成交风险。", "",
        "|模型|规则|成本|净夏普|最大回撤|完整机会数|期末资金|", "|---|---|---|---:|---:|---:|---:|"]
    for r in accounts.itertuples():
        sharpe = "未定义" if pd.isna(r.net_sharpe) else f"{r.net_sharpe:.3f}"
        lines.append(f"|{r.model}|{r.policy}|{r.scenario}|{sharpe}|{r.max_drawdown:.2%}|{r.opportunities}|{r.ending_equity_cny:,.2f}|")
    lines += ["", "原IF状态输入保留旧价格版本；账户和十日标签统一用最新已有本地行情，重叠3454天开高低一致，收盘最大差0.002元。未据收益择价，未修改原状态。完整原合约、原状态、两版行情、来源回执、代码及逐日账簿随包保留。", "",
        "多年历史已在之前多轮研究中观察，不是全新保留样本；没有当日新资料或独立前向事件。本轮完成探索训练，不构成当前买入信号。", ""]
    (out / "训练结果.md").write_text("\n".join(lines), encoding="utf-8")


def verify(out):
    """复核保存结果；不重新训练或回测账户。"""
    check_inputs(out)
    cfg = core.load(out / "protocol.json")
    features, audit = rebuild_features(out)
    monthly, events, market = make_rows(out, cfg)
    for name, frame in [("月末训练特征与十日标签", monthly), ("候选事件与十日标签", events)]:
        pd.testing.assert_frame_equal(frame, pd.read_csv(out / "results" / f"{name}.csv"), check_dtype=False, rtol=1e-10, atol=1e-12)
    predictions = pd.read_csv(out / "results/事件逐期预测.csv")
    calibrations = pd.read_csv(out / "results/月末逐期预测.csv")
    joined_predictions = pd.concat([predictions, calibrations])
    indexed = monthly.set_index("month")
    all_features = pd.concat([monthly, events]).drop_duplicates("month").set_index("month")
    models = core.load(out / "models/逐期模型.json")
    final = core.load(out / "models/期末已训练模型.json")
    for model in [*models, *final.values()]:
        sample = indexed.loc[model["train_months"]]
        limit = model.get("decision_at", model.get("as_of"))
        core.require((sample.label_end_at < limit).all(), "训练包含未成熟标签。")
        x, y = sample[model["columns"]].to_numpy(float), sample.target10.to_numpy(float)
        center, scale, beta = [np.asarray(model[k]) for k in ["center", "scale", "coefficients_standardized"]]
        core.require(np.allclose(center, x.mean(axis=0), atol=1e-12, rtol=0), "训练中心不同。")
        core.require(abs(model["intercept"] - y.mean()) < 1e-12, "成熟均值不同。")
        z = (x - center) / scale
        z[:, ~np.asarray(model["active"], bool)] = 0
        expected = z.T @ (y - model["intercept"]) / len(sample)
        actual = (z.T @ z / len(sample) + cfg["ridge_mean_loss_penalty"] * np.eye(len(beta))) @ beta
        core.require(np.allclose(actual, expected, atol=1e-12, rtol=0), "保存参数不满足岭方程。")
        if "prediction_month" in model:
            r = joined_predictions[(joined_predictions.model == model["model"]) & (joined_predictions.month == model["prediction_month"]) & (joined_predictions.prediction_role == model["prediction_role"])].iloc[0]
            core.require(abs(core.predict(model, all_features.loc[r.month]) - r.prediction) < 1e-12, "参数不能复现预测。")
            errors = calibrations[(calibrations.model == r.model) & (calibrations.label_end_at < r.decision_at)]
            core.require(len(errors) == r.mature_prequential_error_count, "预测误差时钟错误。")
            if len(errors):
                core.require(abs(math.sqrt(float((errors.target10 - errors.prediction).pow(2).mean())) - r.prior_prediction_rmse) < 1e-12, "预测误差不同。")
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
    return {"status": "PASS_SAVED_IF_FEATURES_MODELS_AND_ACCOUNTS", "factor_rows": len(features),
        "saved_models_and_means": len(models) + len(final), "event_predictions": len(predictions), "monthly_predictions": len(calibrations),
        "account_scenarios": len(accounts), "new_model_fits": 0, "new_accounts": 0, "new_downloads": 0}


def main():
    parser = argparse.ArgumentParser(description="仅用已有本地数据进行IF耗竭事件训练。")
    parser.add_argument("command", choices=["freeze", "train", "verify"])
    parser.add_argument("--root", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(verify(args.root), ensure_ascii=False, indent=2))
    else:
        {"freeze": freeze, "train": run}[args.command](args.root)


if __name__ == "__main__":
    main()
