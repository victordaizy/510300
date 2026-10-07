"""已有成分股吸收率的探索训练；仅本地输入，保留原覆盖失败记录。"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import yaml

import absorption_local_inputs_v1 as factor_core
import sparse_event_training_core_v2 as core

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_absorption_available_members_training_v1"


def freeze(out):
    core.require(not (out / "freeze.json").exists(), "输入已保存，不覆盖。")
    cfg = core.load(out / "protocol.json")
    parent_path = ROOT / "config/510300_csi300_absorption_ratio_timing_v1.yaml"
    parent = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    (out / "code").mkdir(exist_ok=True)
    copies = {
        "market.parquet": ROOT / "reports/research/510300_private_manager_intent_v1/inputs/market.parquet",
        "dividends.csv": ROOT / "data/reference/510300_dividends.csv",
        "parent_protocol.yaml": parent_path,
        "parent_data_failure.json": ROOT / "reports/research/510300_csi300_absorption_ratio_timing_v1_data_gate_failure.json",
        "mandate.json": ROOT / "config/510300_existing_data_training_mandate_v1.json",
        "constituent_prices.parquet": ROOT / parent["inputs"]["constituent_total_return_close"]["path"],
    }
    for field in ["historical_membership", "external_membership_panel", "current_membership_panel", "csi300_price_index"]:
        name = field + ".parquet"
        copies[name] = ROOT / parent["inputs"][field]["path"]
        parent["inputs"][field]["path"] = "inputs/" + name
    for name, source in copies.items():
        core.require(source.exists(), f"缺少既有输入：{source}")
        shutil.copy2(source, out / "inputs" / name)
    parent["data_contract"]["minimum_valid_member_count"] = cfg["minimum_valid_members"]
    parent["data_contract"]["minimum_valid_member_coverage"] = cfg["minimum_valid_coverage"]
    core.save(out / "inputs/factor_config.json", parent)
    sources = [Path(__file__), Path(core.__file__), Path(factor_core.__file__), ROOT / "scripts/build_510300_csi300_absorption_ratio_inputs_v1.py"]
    for source in sources:
        target = out / "code" / (source.name if source.name != "build_510300_csi300_absorption_ratio_inputs_v1.py" else "parent_builder_reference_only.py")
        shutil.copy2(source, target)
    files = [out / "protocol.json", *list((out / "inputs").iterdir()), *list((out / "code").iterdir())]
    core.save(out / "freeze.json", {"frozen_at": core.now(), "before_factor_and_this_round_targets": True,
        "old_family_rejection_preserved": True, "user_authorized_exploratory_relaxation": True,
        "historical_prices_seen": True, "new_downloads": 0,
        "identities": {p.relative_to(out).as_posix(): core.digest(p) for p in files}})
    print("已保存既有成分股数据、90%覆盖探索设定与固定训练规则。", flush=True)


def local_inputs(out):
    cfg = core.load(out / "inputs/factor_config.json")
    factor_core.ROOT = out
    calendar = factor_core.load_calendar_dates_only(cfg)
    intervals = factor_core.load_sanitized_membership(cfg)
    external = factor_core._load_panel_member_map(out / cfg["inputs"]["external_membership_panel"]["path"], "已有外部成分")
    current = factor_core._load_panel_member_map(out / cfg["inputs"]["current_membership_panel"]["path"], "已有后段成分")
    members, segments = factor_core.build_daily_membership(calendar[500:], intervals, external, current, cfg)
    return cfg, calendar, members, segments


def build_factor(out):
    core.require(not (out / "factor_receipt.json").exists(), "已有吸收率结果，不重复计算。")
    for name, expected in core.load(out / "freeze.json")["identities"].items():
        core.require(core.digest(out / name) == expected, f"本轮固定输入变化：{name}")
    cfg, calendar, members, segments = local_inputs(out)
    prices = pd.read_parquet(out / "inputs/constituent_prices.parquet")
    with threadpool_limits(limits=1):
        factor = factor_core.compute_daily_factor(prices, calendar, members, segments, cfg)
    core.require(factor.absorption_ratio.notna().all(), "90%覆盖下仍有吸收率缺口。")
    output = out / "results/每日吸收率与成分覆盖.parquet"
    output.parent.mkdir(exist_ok=True)
    factor.to_parquet(output, index=False)
    core.export(output.with_suffix(".csv"), factor)
    membership = pd.DataFrame([{"date": date, "con_code": symbol, "segment": segments[date]} for date, symbols in members.items() for symbol in symbols])
    membership.to_parquet(out / "results/本轮使用的逐日成分.parquet", index=False)
    receipt = {"completed_at": core.now(), "factor_rows": len(factor),
        "first_complete_shift": str(factor.loc[factor.standardized_absorption_ratio_shift.notna(), "date"].min().date()),
        "minimum_valid_members": int(factor.valid_member_count.min()), "minimum_coverage": float(factor.valid_member_coverage.min()),
        "active_member_count_values": sorted(factor.active_member_count.unique().tolist()), "new_downloads": 0,
        "factor_sha256": core.digest(output), "membership_sha256": core.digest(out / "results/本轮使用的逐日成分.parquet")}
    core.save(out / "factor_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


def make_events(out, cfg):
    full_market = core.read_market(out)
    market = full_market[full_market.date <= cfg["evaluation_end"]].reset_index(drop=True)
    factor = pd.read_parquet(out / "results/每日吸收率与成分覆盖.parquet").sort_values("date")
    crossing = (factor.standardized_absorption_ratio_shift <= -1) & (factor.standardized_absorption_ratio_shift.shift(1) > -1)
    candidates = factor[crossing & (factor.date >= cfg["evaluation_start"]) & (factor.date <= cfg["evaluation_end"])]
    dates = pd.DatetimeIndex(full_market.date)
    events = []
    for f in candidates.itertuples():
        signal_i = int(dates.get_loc(f.date))
        entry_i, exit_i = signal_i + 1, signal_i + cfg["horizon_sessions"]
        core.require(exit_i < len(full_market), "已有交易日历不足以定位计划退出。")
        known = market.iloc[signal_i]
        mature = exit_i < len(market)
        target = None
        if mature:
            block = market.iloc[entry_i:exit_i + 1]
            target = (float(block.close.iloc[-1]) + float(block.dividend.iloc[1:].sum())) / float(block.open.iloc[0]) - 1
        events.append({"month": str(f.date.date()), "event_date": str(f.date.date()),
            "decision_at": (f.date.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15, minutes=1)).isoformat(),
            "entry_i": entry_i, "exit_i": exit_i, "entry_date": str(dates[entry_i].date()), "exit_date": str(dates[exit_i].date()),
            "label_end_at": (dates[exit_i].tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)).isoformat(),
            "reference_close": float(known.close), "vol20": float(known.vol20),
            "past20": float(known.past20), "logvol20": float(known.logvol20),
            "absorption_ratio": f.absorption_ratio, "standardized_absorption_ratio_shift": f.standardized_absorption_ratio_shift,
            "valid_member_coverage": f.valid_member_coverage, "valid_member_count": f.valid_member_count,
            "target20": target, "label_status": "MATURE" if mature else "RIGHT_CENSORED_AT_EVALUATION_END"})
    frame = pd.DataFrame(events).sort_values("decision_at").reset_index(drop=True)
    core.require(len(frame) > cfg["min_train"], "穿越事件尚不足以启动既定训练。")
    return frame, market


def train(events, market, cfg):
    predictions, models, final = [], [], {}
    for name, spec in cfg["models"].items():
        columns = sum([cfg["features"][g] for g in spec["groups"]], [])
        previous = []
        for _, row in events.iterrows():
            mature = events[(events.label_end_at < row.decision_at) & events.target20.notna()]
            if len(mature) < cfg["min_train"]:
                continue
            model = core.fit_ridge(mature, columns, cfg["ridge_mean_loss_penalty"])
            model.update({"model": name, "model_type": "RIDGE" if columns else "MATURE_MEAN",
                "prediction_month": row.month, "decision_at": row.decision_at})
            prediction = core.predict(model, row)
            errors = [r for r in previous if r["label_end_at"] < row.decision_at and np.isfinite(r["target20"])]
            rmse = math.sqrt(float(np.mean([(r["prediction"] - r["target20"]) ** 2 for r in errors]))) if errors else None
            record = {field: row[field] for field in ["month", "decision_at", "entry_date", "exit_date", "entry_i", "exit_i",
                "label_end_at", "reference_close", "vol20", "target20", "label_status"]}
            record.update({"model": name, "prediction": prediction, "mean_prediction": model["intercept"], "n_train": len(mature),
                "train_max_label_end_at": model["maximum_training_label_end_at"],
                "mature_prequential_error_count": len(errors), "prior_prediction_rmse": rmse})
            predictions.append(record)
            previous.append(record)
            models.append(model)
        cutoff = (market.date.iloc[-1].tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15, minutes=2)).isoformat()
        mature = events[(events.label_end_at < cutoff) & events.target20.notna()]
        model = core.fit_ridge(mature, columns, cfg["ridge_mean_loss_penalty"])
        model.update({"model": name, "model_type": "RIDGE" if columns else "MATURE_MEAN", "as_of": cutoff,
            "status": "EXPLORATORY_TRAINED_NO_CURRENT_SIGNAL"})
        final[name] = model
    return pd.DataFrame(predictions), models, final


def score(predictions):
    complete = predictions[predictions.target20.notna()]
    metrics, comparisons = [], []
    for name, sample in complete.groupby("model", sort=True):
        mse = float((sample.target20 - sample.prediction).pow(2).mean())
        mean_mse = float((sample.target20 - sample.mean_prediction).pow(2).mean())
        metrics.append({"model": name, "n_predictions": len(sample), "rmse": math.sqrt(mse), "mse": mse,
            "direction_hit": float(((sample.target20 > 0) == (sample.prediction > 0)).mean()),
            "mse_improvement_vs_mean": 1 - mse / mean_mse})
    for reference in ["A_PRICE", "C_PRICE_COVERAGE"]:
        matched = complete[complete.model == "B_ABSORPTION"].merge(
            complete[complete.model == reference][["month", "prediction"]], on="month", suffixes=("", "_reference")).sort_values("decision_at")
        for segment, sample in [("ALL", matched), ("FIRST_HALF", matched.iloc[:len(matched) // 2]), ("SECOND_HALF", matched.iloc[len(matched) // 2:])]:
            mse = float((sample.target20 - sample.prediction).pow(2).mean())
            reference_mse = float((sample.target20 - sample.prediction_reference).pow(2).mean())
            comparisons.append({"reference": reference, "segment": segment, "n": len(sample), "mse_improvement": 1 - mse / reference_mse})
    return pd.DataFrame(metrics), pd.DataFrame(comparisons)


def report(out, summary, scores, comparisons, accounts):
    f = core.load(out / "factor_receipt.json")
    lines = ["# 可计算成分股吸收率：探索训练", "", summary["conclusion"], "",
        f"现有714只历史相关成员数据生成{f['factor_rows']}个日度吸收率。每日实际成员300只，可计算成员最少{f['minimum_valid_members']}只，最低覆盖{f['minimum_coverage']:.2%}。原V1因要求285只/95%停止，且未计算市场或组合收益；按用户直接训练、放宽限制要求，本轮单独以270只/90%进入探索，原失败字节保留。属于同一家族后续尝试，不能重新称作独立未见证据。", "",
        "仍用500日协方差、250日指数半衰期、前floor(N/5)个特征值占比；不填上市前价格，不用未来成员。吸收率15日与250日均值之差除以250日标准差，首次从大于-1穿越至不高于-1时形成候选，下一开盘才可入场。该表示衡量共同波动程度，不直接识别资金净买卖或投资者身份。", "",
        f"共{summary['candidate_events']}个候选，其中{summary['mature_event_labels']}个20日标签成熟。从12个成熟事件起逐期训练；完成{summary['prequential_ridge_fits']}次岭拟合和{summary['prequential_mean_updates']}次均值更新，期末另存3个岭模型和1个均值。训练标签在当前决策时刻前成熟，允许事件有重叠，但不把重叠窗口称作独立样本。", "",
        "A为20日价格回报与波动；C再加当日成员覆盖率；B在C基础上增加吸收率水平及标准化变化。加入C是为了识别单纯缺成员比例的影响，但不能完全消除缺失成员行业或上市年龄差异。岭惩罚固定1，训练内标准化，无窗口或阈值搜索。复用账簿字段month在本轮存储完整事件日期。", "",
        "|模型|成熟预测数|RMSE|方向命中|相对成熟均值MSE改善|", "|---|---:|---:|---:|---:|"]
    for r in scores.itertuples():
        lines.append(f"|{r.model}|{r.n_predictions}|{r.rmse:.2%}|{r.direction_hit:.2%}|{r.mse_improvement_vs_mean:.2%}|")
    lines += ["", "|B相对基准|区间|事件数|MSE改善|", "|---|---|---:|---:|"]
    for r in comparisons.itertuples():
        lines.append(f"|{r.reference}|{r.segment}|{r.n}|{r.mse_improvement:.2%}|")
    lines += ["", f"20万元账户从{summary['account_start']}至{summary['market_end']}，计入全部空仓交易日。误差缓冲主规则须超过压力费用及至少6个成熟逐期误差的RMSE；费用覆盖对照仅要求预测为正且超过压力往返费用。10%目标年化波动、仓位上限100%、100份整手、不融资、T+1，佣金单边万2最低5元，基础/压力滑点万5/千1，现金及无风险收益0，242日年化。固定20日退出，收盘回撤8%则次开盘清仓并停止。", "",
        "分红在除息日记应收权益、按已有付款日转可用现金；到账前不能用于买入。期末不足20日的仓位单列清算，不充当完整20日周期。滑点价格未按最小报价单位取整，涨跌停及成交队列未模拟。", "",
        "|模型|规则|成本|净夏普|最大回撤|机会数|期末清算数|期末资金|", "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in accounts.itertuples():
        sh = "未定义" if pd.isna(r.net_sharpe) else f"{r.net_sharpe:.3f}"
        lines.append(f"|{r.model}|{r.policy}|{r.scenario}|{sh}|{r.max_drawdown:.2%}|{r.opportunities}|{r.right_censored_exits}|{r.ending_equity_cny:,.2f}|")
    lines += ["", "数据来自既有回溯重建成分与复权价格，不是原时点逐日归档。旧500日数据覆盖失败没有变成通过；本轮完整来源范围和阈值变化均保留。即使某一历史数值达标，也不能把多轮筛选过的历史称作独立验证或确定盈利。8%警戒不保证未来跳空仍满足10%回撤。", "",
        "本轮没有联网采集、恢复已终止策略或下单。仅有2026-08-14以前吸收率输入，没有2026-09-22当前交易信号。", ""]
    (out / "训练结果.md").write_text("\n".join(lines), encoding="utf-8")


def run(out):
    core.require(not (out / "summary.json").exists(), "本轮训练已完成，不覆盖。")
    for name, expected in core.load(out / "freeze.json")["identities"].items():
        core.require(core.digest(out / name) == expected, f"本轮输入变化：{name}")
    receipt = core.load(out / "factor_receipt.json")
    core.require(core.digest(out / "results/每日吸收率与成分覆盖.parquet") == receipt["factor_sha256"], "计算后因子变化。")
    cfg = core.load(out / "protocol.json")
    events, market = make_events(out, cfg)
    predictions, models, final = train(events, market, cfg)
    scores, comparisons = score(predictions)
    for name, frame in {"候选事件与20日标签": events, "逐期预测": predictions, "预测指标": scores, "增量比较": comparisons}.items():
        core.export(out / "results" / f"{name}.csv", frame)
    core.save(out / "models/逐期模型.json", models)
    core.save(out / "models/期末已训练模型.json", final)
    print(f"完成{len(models)}份逐期模型/均值快照，开始账户。", flush=True)
    dividends = pd.read_csv(out / "inputs/dividends.csv")
    dividends = dividends[dividends.symbol == "510300.SH"]
    known_dividends = dividends.assign(date=pd.to_datetime(dividends.ex_date)).set_index("date").cash_dividend_per_share
    for day in market[market.dividend > 0].itertuples():
        core.require(day.date in known_dividends.index and abs(known_dividends.loc[day.date] - day.dividend) < 1e-12, "已有分红权益与付款表不一致。")
    start_i = int(predictions.entry_i.min())
    paths, trades, decisions, account_rows = [], [], [], []
    for policy, settings in cfg["policies"].items():
        effective = copy.deepcopy(cfg)
        effective["account"].update(settings)
        for name in cfg["models"]:
            for scenario in cfg["account"]["slippage_per_side"]:
                nav, trade, decision, metric = core.account_run(name, scenario, predictions, market, start_i, effective, dividends)
                nav["policy"] = policy
                for row in [*trade, *decision, metric]:
                    row["policy"] = policy
                paths.append(nav)
                trades.extend(trade)
                decisions.extend(decision)
                account_rows.append(metric)
    for name in ["CASH", "BUY_AND_HOLD"]:
        for scenario in cfg["account"]["slippage_per_side"]:
            nav, trade, decision, metric = core.account_run(name, scenario, predictions, market, start_i, cfg, dividends)
            nav["policy"] = "BENCHMARK"
            for row in [*trade, metric]:
                row["policy"] = "BENCHMARK"
            paths.append(nav)
            trades.extend(trade)
            account_rows.append(metric)
    accounts = pd.DataFrame(account_rows)
    for name, frame in {"完整逐日账户": pd.concat(paths, ignore_index=True), "机会成交账簿": pd.DataFrame(trades),
            "逐事件交易判断": pd.DataFrame(decisions), "账户指标": accounts}.items():
        core.export(out / "results" / f"{name}.csv", frame)
    passing = []
    for (model, policy), group in accounts[accounts.policy != "BENCHMARK"].groupby(["model", "policy"]):
        if group.meets_numeric_targets.all():
            passing.append({"model": model, "policy": policy})
    summary = {"study_id": cfg["study_id"], "completed_at": core.now(), "status": "EXPLORATORY_AVAILABLE_MEMBERS_TRAINED",
        "candidate_events": len(events), "mature_event_labels": int(events.target20.notna().sum()),
        "pending_event_labels": int(events.target20.isna().sum()), "prediction_rows": len(predictions),
        "prequential_ridge_fits": sum(m["model_type"] == "RIDGE" for m in models),
        "prequential_mean_updates": sum(m["model_type"] == "MATURE_MEAN" for m in models),
        "final_ridge_models": 3, "final_mean_references": 1, "account_scenarios": len(accounts),
        "account_start": str(market.date.iloc[start_i].date()), "market_end": str(market.date.iloc[-1].date()),
        "numeric_targets_pass_both_costs": passing, "goal_achieved": False, "independent_forward_events": 0,
        "new_downloads": 0, "orders_authorized": False,
        "previous_goal_turn_classification": "PROGRESS_NEW_M1_TRAINING_COMPLETED",
        "conclusion": "已有成分股吸收率完成探索训练与完整账户。" + ("存在历史数值达标组合，仍需检查集中度和跨时期稳定性。" if passing else "没有组合同时达到两档成本下净夏普1.5、最大回撤不超过10%。")}
    core.save(out / "summary.json", summary)
    report(out, summary, scores, comparisons, accounts)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(scores.to_string(index=False), flush=True)
    print(comparisons.to_string(index=False), flush=True)
    print(accounts[["model", "policy", "scenario", "net_sharpe", "max_drawdown", "opportunities", "ending_equity_cny"]].to_string(index=False), flush=True)


def verify(out):
    cfg = core.load(out / "protocol.json")
    for name, expected in core.load(out / "freeze.json")["identities"].items():
        core.require(core.digest(out / name) == expected, f"本轮输入身份变化：{name}")
    f = pd.read_parquet(out / "results/每日吸收率与成分覆盖.parquet")
    expected_shift = (f.absorption_ratio.rolling(15).mean() - f.absorption_ratio.rolling(250).mean()) / f.absorption_ratio.rolling(250).std(ddof=1)
    core.require(np.allclose(f.standardized_absorption_ratio_shift, expected_shift, equal_nan=True, rtol=0, atol=1e-10), "吸收率滚动公式不同。")
    core.require((f.valid_member_count >= 270).all() and (f.valid_member_coverage >= 0.9).all(), "实际覆盖不足当前探索范围。")
    events, market = make_events(out, cfg)
    pd.testing.assert_frame_equal(events, pd.read_csv(out / "results/候选事件与20日标签.csv"), check_dtype=False, rtol=1e-10, atol=1e-12)
    indexed = events.set_index("month")
    predictions = pd.read_csv(out / "results/逐期预测.csv")
    models = core.load(out / "models/逐期模型.json")
    final = core.load(out / "models/期末已训练模型.json")
    for model in [*models, *final.values()]:
        sample = indexed.loc[model["train_months"]]
        limit = model.get("decision_at", model.get("as_of"))
        core.require((sample.label_end_at < limit).all() and sample.target20.notna().all(), "存在未成熟训练标签。")
        x, y = sample[model["columns"]].to_numpy(float), sample.target20.to_numpy(float)
        center, scale, beta = [np.asarray(model[k]) for k in ["center", "scale", "coefficients_standardized"]]
        core.require(np.allclose(center, x.mean(axis=0), rtol=0, atol=1e-12), "训练中心不同。")
        z = (x - center) / scale
        z[:, ~np.asarray(model["active"], bool)] = 0.0
        core.require(np.allclose((z.T @ z / len(sample) + np.eye(len(beta))) @ beta, z.T @ (y - model["intercept"]) / len(sample), rtol=0, atol=1e-12), "模型方程不能对账。")
        if "prediction_month" in model:
            row = predictions[(predictions.model == model["model"]) & (predictions.month == model["prediction_month"])].iloc[0]
            core.require(abs(core.predict(model, indexed.loc[row.month]) - row.prediction) < 1e-12, "保存参数不能重现预测。")
    nav = pd.read_csv(out / "results/完整逐日账户.csv")
    trades = pd.read_csv(out / "results/机会成交账簿.csv")
    accounts = pd.read_csv(out / "results/账户指标.csv")
    prices = market.assign(day=market.date.dt.strftime("%Y-%m-%d")).set_index("day").close
    dividend_table = pd.read_csv(out / "inputs/dividends.csv")
    dividend_table = dividend_table[dividend_table.symbol == "510300.SH"]
    payments = {r.ex_date: r.payment_date for r in dividend_table.itertuples()}
    for (model, policy, scenario), daily in nav.groupby(["model", "policy", "scenario"]):
        metric = accounts[(accounts.model == model) & (accounts.policy == policy) & (accounts.scenario == scenario)].iloc[0]
        equity = daily.equity_cny.to_numpy(float)
        returns = equity / np.r_[cfg["capital_cny"], equity[:-1]] - 1
        peak = np.maximum.accumulate(np.r_[cfg["capital_cny"], equity])[1:]
        core.require(np.allclose(daily.cash_cny + daily.shares * daily.date.map(prices) + daily.dividend_receivable_cny, equity, rtol=0, atol=1e-7), "现金、持仓与应收权益不平。")
        core.require(np.allclose(returns, daily.daily_return, rtol=0, atol=1e-12), "逐日收益不同。")
        core.require(abs(-(equity / peak - 1).min() - metric.max_drawdown) < 1e-12, "最大回撤不同。")
        if returns.std(ddof=1) > 1e-14:
            core.require(abs(returns.mean() / returns.std(ddof=1) * math.sqrt(cfg["annual_days"]) - metric.net_sharpe) < 1e-10, "全账户夏普不同。")
        else:
            core.require(pd.isna(metric.net_sharpe), "空仓夏普必须未定义。")
        selected = trades[(trades.model == model) & (trades.policy == policy) & (trades.scenario == scenario)]
        core.require(abs(selected.net_pnl_cny.sum() - (equity[-1] - cfg["capital_cny"])) < 1e-7, "全部成交损益不能对账。")
        obligations = []
        for row in daily.itertuples():
            if row.dividends_cny:
                obligations.append((payments[row.date], row.dividends_cny))
            receivable = sum(value for date, value in obligations if date > row.date)
            paid = sum(value for date, value in obligations if date == row.date)
            core.require(abs(receivable - row.dividend_receivable_cny) < 1e-7 and abs(paid - row.dividend_paid_cny) < 1e-7, "分红付款或应收时钟不同。")
        for t in selected.itertuples():
            core.require(t.exit_i > t.entry_i, "T+1失效。")
            dividend = float(market.iloc[int(t.entry_i) + 1:int(t.exit_i) + 1].dividend.sum()) * t.shares
            pnl = t.shares * t.exit_fill_price - t.exit_commission_cny + dividend - t.entry_cash_debit
            core.require(abs(pnl - t.net_pnl_cny) < 1e-7, "单笔现金流不一致。")
    return {"status": "PASS_SAVED_FACTORS_MODELS_AND_PAYMENT_ACCOUNTS", "factor_rows": len(f),
        "saved_models_and_means": len(models) + len(final), "prediction_rows": len(predictions),
        "account_scenarios": len(accounts), "new_model_fits": 0, "new_accounts": 0, "new_downloads": 0}


def main():
    parser = argparse.ArgumentParser(description="本地可计算成员吸收率探索训练。")
    parser.add_argument("command", choices=["freeze", "factor", "train", "verify"])
    parser.add_argument("--root", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(verify(args.root), ensure_ascii=False, indent=2))
    else:
        {"freeze": freeze, "factor": build_factor, "train": run}[args.command](args.root)


if __name__ == "__main__":
    main()
