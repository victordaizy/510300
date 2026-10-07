"""直接训练扩样后的私募意向与可加仓空间，比较两种固定入场规则。"""
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
OUT = ROOT / "reports/research/510300_private_manager_exploratory_training_v1"
PARENT = ROOT / "reports/research/510300_private_manager_intent_source_extension_v2"
PARENT_ZIP = ROOT / "deliverables/510300_私募意向资料扩样_V2_GPT审阅_20260922.zip"
PARENT_HASH = "e7319519e24759b1a1c4e3a00f9fa39bfe2441311ad4b0ec4b3f6555746a924c"


def freeze(out):
    core.require(not (out / "freeze.json").exists(), "已有训练输入快照，不覆盖。")
    core.require(core.digest(PARENT_ZIP) == PARENT_HASH, "资料扩样包身份不符。")
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    sources = {
        "sources.json": PARENT / "combined_admitted_sources.json",
        "source_extension_v2.zip": PARENT_ZIP,
        "coverage.csv": PARENT / "全部月份覆盖与缺口.csv",
        "market.parquet": ROOT / "reports/research/510300_private_manager_intent_v1/inputs/market.parquet",
        "mandate.json": ROOT / "config/510300_sparse_opportunity_mandate_v1.json",
    }
    for name, source in sources.items():
        shutil.copy2(source, out / "inputs" / name)
    (out / "code").mkdir(exist_ok=True)
    for source in [Path(__file__), Path(core.__file__)]:
        shutil.copy2(source, out / "code" / source.name)
    paths = [out / "protocol.json", *list((out / "inputs").iterdir()), *list((out / "code").iterdir())]
    core.save(out / "freeze.json", {"frozen_at": core.now(), "before_this_round_labels_and_fits": True,
              "historical_data_seen_in_prior_research": True,
              "user_override": "不用那么严格，直接进入训练",
              "identities": {p.relative_to(out).as_posix(): core.digest(p) for p in paths}})
    print("直接训练设定已保存；开始使用43个月意向资料。", flush=True)


def make_events(out, cfg):
    market = core.read_market(out)
    market["prior_return20"] = market.wealth / market.wealth.shift(20) - 1
    market["logvol20"] = np.log(market.vol20 * math.sqrt(cfg["annual_days"]))
    index = pd.DatetimeIndex(market.date)
    sources = core.load(out / "inputs/sources.json")
    events = []
    for original in sources:
        row = dict(original)
        published = pd.Timestamp(row["available_date"]).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=23, minutes=59, seconds=59)
        entry = int(index.searchsorted(pd.Timestamp(row["available_date"]), side="right"))
        end = entry + cfg["horizon_sessions"] - 1
        core.require(entry >= 21 and end < len(market), "原资料事件缺少价格特征或20日标签。")
        block, last = market.iloc[entry:end + 1], market.iloc[entry - 1]
        decision = index[entry].tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9)
        core.require(published < decision, "调查资料晚于交易决策。")
        exposure = row.get("survey_exposure_pct")
        intent = (row["plan_index"] - 100) / 100
        row.update({"available_at": published.isoformat(), "decision_at": decision.isoformat(),
                    "review_date": str(last.date.date()), "entry_date": str(index[entry].date()),
                    "exit_date": str(index[end].date()), "entry_i": entry, "exit_i": end,
                    "label_end_at": (index[end].tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)).isoformat(),
                    "reference_close": float(last.close), "vol20": float(last.vol20),
                    "prior_return20": float(last.prior_return20), "logvol20": float(last.logvol20),
                    "intent": intent, "headroom": None if exposure is None else 1 - exposure / 100,
                    "intent_headroom": None if exposure is None else intent * (1 - exposure / 100),
                    "space_cohort": exposure is not None,
                    "target20": (float(block.close.iloc[-1]) + float(block.dividend.iloc[1:].sum())) / float(block.open.iloc[0]) - 1})
        events.append(row)
    frame = pd.DataFrame(events).sort_values("decision_at").reset_index(drop=True)
    core.require(len(frame) == 43 and int(frame.space_cohort.sum()) == 34, "43个意向月、34个联合月的资料边界变化。")
    core.require(frame.month.is_unique, "存在重复事件月。")
    return frame, market


def train(cfg, events, market):
    all_predictions, models, final = [], [], {}
    for name, spec in cfg["models"].items():
        frame = events if spec["cohort"] == "ALL" else events[events.space_cohort]
        columns = sum((cfg["features"][g] for g in spec["groups"]), [])
        previous = []
        for _, row in frame.iterrows():
            fit_rows = frame[frame.label_end_at < row.decision_at]
            if len(fit_rows) < cfg["min_train"]:
                continue
            model = core.fit_ridge(fit_rows, columns, cfg["ridge_alpha_on_sum_squared_loss"] / len(fit_rows))
            model.update({"model": name, "prediction_month": row.month, "decision_at": row.decision_at})
            prediction = core.predict(model, row)
            mature = [r for r in previous if r["label_end_at"] < row.decision_at]
            rmse = math.sqrt(float(np.mean([(r["prediction"] - r["target20"]) ** 2 for r in mature]))) if mature else None
            result = {k: row[k] for k in ["month", "decision_at", "entry_date", "exit_date", "entry_i", "exit_i", "label_end_at", "reference_close", "vol20", "target20", "clock"]}
            result.update({"model": name, "cohort": spec["cohort"], "prediction": prediction,
                           "mean_prediction": model["intercept"], "n_train": len(fit_rows),
                           "train_max_label_end_at": model["maximum_training_label_end_at"],
                           "mature_prequential_error_count": len(mature), "prior_prediction_rmse": rmse,
                           "prior_error_months": "|".join(r["month"] for r in mature)})
            previous.append(result)
            all_predictions.append(result)
            models.append(model)
        last_date = (pd.Timestamp(market.date.iloc[-1]).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15, minutes=1)).isoformat()
        fit_rows = frame[frame.label_end_at < last_date]
        model = core.fit_ridge(fit_rows, columns, cfg["ridge_alpha_on_sum_squared_loss"] / len(fit_rows))
        model.update({"model": name, "as_of": last_date, "status": "EXPLORATORY_TRAINED"})
        final[name] = model
    return pd.DataFrame(all_predictions), models, final


def metrics(predictions):
    scores, comparisons = [], []
    for name, group in predictions.groupby("model", sort=True):
        y, p, mean = [group[c].to_numpy(dtype=float) for c in ["target20", "prediction", "mean_prediction"]]
        mse = float(np.mean((y - p) ** 2))
        mse_mean, mse_zero = float(np.mean((y - mean) ** 2)), float(np.mean(y ** 2))
        scores.append({"model": name, "n_predictions": len(group), "first_month": group.month.min(),
                       "last_month": group.month.max(), "rmse": math.sqrt(mse), "mse": mse,
                       "mae": float(np.mean(np.abs(y - p))), "direction_hit": float(np.mean(np.sign(y) == np.sign(p))),
                       "mse_mean": mse_mean, "mse_zero": mse_zero,
                       "mse_improvement_vs_mean": 1 - mse / mse_mean, "mse_improvement_vs_zero": 1 - mse / mse_zero})
    for candidate, reference in [("B_INTENT", "A_PRICE"), ("C_INTENT_SPACE", "AC_SPACE_MATCH")]:
        left, right = predictions[predictions.model == candidate], predictions[predictions.model == reference]
        matched = left.merge(right[["month", "prediction"]], on="month", suffixes=("", "_reference")).sort_values("decision_at")
        subsets = [("ALL", matched), ("FIRST_HALF", matched.iloc[:len(matched) // 2]), ("SECOND_HALF", matched.iloc[len(matched) // 2:]),
                   ("DATED_ARTICLE_DIAGNOSTIC", matched[matched.clock == "DATED_ARTICLE"])]
        for segment, sample in subsets:
            mse = float((sample.target20 - sample.prediction).pow(2).mean())
            reference_mse = float((sample.target20 - sample.prediction_reference).pow(2).mean())
            mean_mse = float((sample.target20 - sample.mean_prediction).pow(2).mean())
            comparisons.append({"candidate": candidate, "reference": reference, "segment": segment, "n": len(sample),
                                "mse": mse, "reference_mse": reference_mse, "mse_improvement": 1 - mse / reference_mse,
                                "mse_improvement_vs_mean": 1 - mse / mean_mse})
    return pd.DataFrame(scores), pd.DataFrame(comparisons)


def report(out, cfg, summary, scores, comparisons, accounts):
    lines = ["# 私募操作意向与可加仓空间：实际训练", "", summary["conclusion"], "",
             f"43个意向月份中34个有同调查仓位；完成{summary['prequential_fit_count']}次逐期拟合和4个期末模型。调查最新月份为{summary['last_event_month']}，行情截至{summary['market_end']}。", "",
             "至少12个成熟事件起步。固定岭回归alpha=10（残差平方和口径），无参数网格；前一日收盘价格特征及已发布调查用于下一开盘。调查日没有时刻的资料按当天23:59:59，20日目标包括入场日之后的分红；所有标签成熟之后才进训练。", "",
             "A_PRICE为价格，B_INTENT增加操作意向；AC_SPACE_MATCH为联合样本的价格基准，C_INTENT_SPACE再加意向、可加仓空间及二者乘积。调查意向不是资金流，空间只是仓位粗略代理。", "",
             "## 逐期预测", "", "|模型|预测月数|RMSE|方向命中率|相对历史均值MSE改善|", "|---|---:|---:|---:|---:|"]
    for r in scores.itertuples():
        lines.append(f"|{r.model}|{r.n_predictions}|{r.rmse:.2%}|{r.direction_hit:.2%}|{r.mse_improvement_vs_mean:.2%}|")
    lines += ["", "|候选/同月基准|样本|事件数|MSE改善|", "|---|---|---:|---:|"]
    for r in comparisons.itertuples():
        lines.append(f"|{r.candidate}/{r.reference}|{r.segment}|{r.n}|{r.mse_improvement:.2%}|")
    lines += ["", "## 20万元全账户结果", "", "NET_POSITIVE_DIAGNOSTIC只要求预测收益超过压力往返费用，用于检查经济增量；ERROR_BUFFER_PRIMARY还要求超过至少6个成熟逐期误差的RMSE，为本轮主方案。两者同时事先指定，不按结果挑阈值；误差缓冲不是概率保证。", "",
              f"共同区间：{summary['account_start']}至{summary['market_end']}。每天含现金；现金和无风险日收益设0，242日年化；100份整手，单边佣金万2且最低5元，基础/压力滑点万5/千1。仓位10%目标波动上限且不超过本金，固定20交易日退出，回撤8%触发最早次开盘清仓及停止。", "",
              "|模型|规则|费用|净夏普|最大回撤|年化收益|完整机会|在场天数|", "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in accounts.itertuples():
        sharpe = "未定义" if pd.isna(r.net_sharpe) else f"{r.net_sharpe:.3f}"
        lines.append(f"|{r.model}|{r.policy}|{r.scenario}|{sharpe}|{abs(r.max_drawdown):.2%}|{r.annualized_return:.2%}|{r.opportunities}|{r.exposure_day_fraction:.2%}|")
    lines += ["", "## 结果范围", "", "用户已明确允许直接训练，旧24训练月/24评价月数量门未用于阻挡。本轮用了旧历史，尚无真正独立前瞻验证；不能将高点值或一两次成交称作确定机会。2022-08分项占比不自洽的来源标记保留，原文计划指数104未反推修改。109个月缺失或原排除月没有补值。", "",
              "同一调查计划指数43月中41月高于100，只有1个月低于100；样本覆盖稀疏且偏向乐观，不能将指数超过100直接解释成罕见机会。仓位变动也可能反映价格或样本构成变化，不能作为真实持续买入的等价标签。", "",
              "当前模型截至保存行情日，调查数据更早。没有当日完整新输入或实盘信号。10%是账户回撤目标，8%警戒退出不保证未来跳空时仍能满足。未细化最小价位取整、涨跌停或成交量约束，小资金成本估算仍非真实成交。", ""]
    (out / "训练结果.md").write_text("\n".join(lines), encoding="utf-8")


def run(out):
    core.require(not (out / "summary.json").exists(), "训练已完成，不覆盖结果。")
    for name, expected in core.load(out / "freeze.json")["identities"].items():
        core.require(core.digest(out / name) == expected, f"输入或代码变化：{name}")
    cfg = core.load(out / "protocol.json")
    events, market = make_events(out, cfg)
    predictions, models, final = train(cfg, events, market)
    scores, comparisons = metrics(predictions)
    for filename, frame in [("事件特征与20日标签", events), ("逐期预测", predictions), ("预测指标", scores), ("同月增量比较", comparisons)]:
        core.export(out / "results" / (filename + ".csv"), frame)
    core.save(out / "models/逐期模型.json", models)
    core.save(out / "models/期末已训练模型.json", final)
    print(f"已完成{len(models)}次逐期拟合及4个期末模型，开始固定规则账户。", flush=True)
    start_i = int(predictions[predictions.cohort == "ALL"].entry_i.min())
    paths, trades, decisions, account_rows = [], [], [], []
    for policy, settings in cfg["policies"].items():
        effective = copy.deepcopy(cfg)
        effective["account"].update({k: settings[k] for k in ["prediction_error_multiplier", "min_mature_prequential_errors"]})
        for name in cfg["models"]:
            for scenario in cfg["account"]["slippage_per_side"]:
                nav, trade, decision, metric = core.account_run(name, scenario, predictions, market, start_i, effective)
                nav["policy"] = policy
                for row in [*trade, *decision, metric]:
                    row["policy"] = policy
                paths.append(nav)
                trades.extend(trade)
                decisions.extend(decision)
                account_rows.append(metric)
    for name in ["CASH", "BUY_AND_HOLD"]:
        for scenario in cfg["account"]["slippage_per_side"]:
            nav, trade, decision, metric = core.account_run(name, scenario, predictions, market, start_i, cfg)
            nav["policy"] = "BENCHMARK"
            for row in [*trade, metric]:
                row["policy"] = "BENCHMARK"
            paths.append(nav)
            trades.extend(trade)
            account_rows.append(metric)
    accounts = pd.DataFrame(account_rows)
    for filename, frame in [("完整逐日账户", pd.concat(paths, ignore_index=True)), ("机会成交账簿", pd.DataFrame(trades)),
                            ("逐事件交易判断", pd.DataFrame(decisions)), ("账户指标", accounts)]:
        core.export(out / "results" / (filename + ".csv"), frame)
    passing = []
    for (model, policy), group in accounts[accounts.policy != "BENCHMARK"].groupby(["model", "policy"]):
        if group.meets_numeric_targets.all():
            passing.append({"model": model, "policy": policy})
    summary = {"study_id": cfg["study_id"], "completed_at": core.now(), "status": "EXPLORATORY_TRAINED",
               "event_months": len(events), "space_months": int(events.space_cohort.sum()),
               "first_event_month": events.month.iloc[0], "last_event_month": events.month.iloc[-1],
               "market_end": str(market.date.iloc[-1].date()), "account_start": str(market.date.iloc[start_i].date()),
               "prequential_fit_count": len(models), "final_model_count": len(final), "account_scenarios": len(accounts),
               "numeric_targets_pass_both_costs": passing, "goal_achieved": False, "true_forward_events": 0,
               "conclusion": "私募意向及可加仓空间已经实际训练，两种固定入场规则的完整账户已计算。" + ("有历史数值达标组合，但不足以建立独立高置信度证据。" if passing else "没有组合同时达到基础及压力成本下净夏普1.5、最大回撤不超过10%的目标。"),
               "orders_authorized": False, "previous_goal_turn_classification": "PROGRESS_LPR_MODELS_AND_ACCOUNTS_COMPLETED"}
    core.save(out / "summary.json", summary)
    report(out, cfg, summary, scores, comparisons, accounts)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def verify(out):
    cfg = core.load(out / "protocol.json")
    for name, expected in core.load(out / "freeze.json")["identities"].items():
        core.require(core.digest(out / name) == expected, f"输入身份改变：{name}")
    events = pd.read_csv(out / "results/事件特征与20日标签.csv").set_index("month")
    events["note"] = events["note"].fillna("")
    predictions = pd.read_csv(out / "results/逐期预测.csv")
    models = core.load(out / "models/逐期模型.json")
    saved = {(r.model, r.month): r for r in predictions.itertuples()}
    for model in models:
        row = saved[(model["model"], model["prediction_month"])]
        core.require(np.isclose(core.predict(model, events.loc[row.month]), row.prediction, rtol=0, atol=1e-12), "保存系数不能重现预测。")
        core.require((events.loc[model["train_months"]].label_end_at < row.decision_at).all(), "标签未成熟。")
        previous = predictions[(predictions.model == row.model) & (predictions.label_end_at < row.decision_at)]
        core.require(len(previous) == row.mature_prequential_error_count, "误差样本成熟计数不同。")
        if len(previous):
            rmse = math.sqrt(float((previous.prediction - previous.target20).pow(2).mean()))
            core.require(np.isclose(rmse, row.prior_prediction_rmse, rtol=0, atol=1e-12), "预测误差缓冲不同。")
    rebuilt, market = make_events(out, cfg)
    pd.testing.assert_frame_equal(events.reset_index(drop=False), rebuilt.set_index("month").reset_index(drop=False), check_dtype=False, check_like=True, atol=1e-12, rtol=1e-10)
    scores, comparisons = metrics(predictions)
    for actual, filename in [(scores, "预测指标.csv"), (comparisons, "同月增量比较.csv")]:
        pd.testing.assert_frame_equal(actual, pd.read_csv(out / "results" / filename), check_dtype=False, atol=1e-12, rtol=1e-10)
    nav = pd.read_csv(out / "results/完整逐日账户.csv")
    trades = pd.read_csv(out / "results/机会成交账簿.csv")
    accounts = pd.read_csv(out / "results/账户指标.csv")
    prices = market.assign(day=market.date.dt.strftime("%Y-%m-%d")).set_index("day").close
    for (model, policy, scenario), daily in nav.groupby(["model", "policy", "scenario"]):
        metric = accounts[(accounts.model == model) & (accounts.policy == policy) & (accounts.scenario == scenario)].iloc[0]
        core.require(np.allclose(daily.cash_cny + daily.shares * daily.date.map(prices), daily.equity_cny, rtol=0, atol=1e-7), "逐日净值不等于现金加持仓。")
        r = daily.equity_cny.to_numpy() / np.r_[cfg["capital_cny"], daily.equity_cny.to_numpy()[:-1]] - 1
        core.require(np.allclose(r, daily.daily_return, rtol=0, atol=1e-12), "日收益不同。")
        peak = np.maximum.accumulate(np.r_[cfg["capital_cny"], daily.equity_cny.to_numpy()])[1:]
        drawdown = daily.equity_cny.to_numpy() / peak - 1
        core.require(np.isclose(-drawdown.min(), metric.max_drawdown, rtol=0, atol=1e-12), "回撤不同。")
        if r.std(ddof=1) > 1e-14:
            core.require(np.isclose(r.mean() / r.std(ddof=1) * math.sqrt(cfg["annual_days"]), metric.net_sharpe, rtol=0, atol=1e-9), "夏普不同。")
        else:
            core.require(pd.isna(metric.net_sharpe), "空仓夏普必须未定义。")
        selected = trades[(trades.model == model) & (trades.policy == policy) & (trades.scenario == scenario)]
        core.require(len(selected) == metric.opportunities, "完整机会数不同。")
        core.require((selected.exit_i > selected.entry_i).all(), "不满足T+1。")
        core.require(np.isclose(selected.net_pnl_cny.sum(), daily.equity_cny.iloc[-1] - cfg["capital_cny"], rtol=0, atol=1e-6), "成交损益不同。")
        core.require(np.isclose(daily.cost_cny.sum(), metric.cost_cny, rtol=0, atol=1e-7), "费用不同。")
    return {"status": "PASS_SAVED_OUTPUT_RECOMPUTATION", "prediction_rows": len(predictions), "event_labels": len(events),
            "account_scenarios": len(accounts), "new_fits": 0, "new_accounts": 0, "downloads": 0}


def main():
    parser = argparse.ArgumentParser(description="私募意向扩样实际训练")
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
