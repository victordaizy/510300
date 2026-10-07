"""固定日线入场与保护，检验训练退出的增量，并给出无训练多空点位对照。"""
from __future__ import annotations

import argparse
from bisect import bisect_right
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import upward_episode_anatomy_v1 as common
from research import historic_cycles_point_translation_v1 as old
from research import point_state_reconstruction_v1 as previous

OUT = ROOT / "reports/research/510300_point_entry_exit_contribution_v1"
CONTEXT = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001"
SOURCE = ROOT / "reports/research/510300_entry_vintage_exit_v1"
MODEL_SOURCE = ROOT / "reports/research/510300_within_cycle_exit_v1"
PERIODS = previous.PERIODS
POLICIES = {
    "LONG_GUARDS": "多头：原价格保护", "LONG_MODEL": "多头：价格保护加固定训练退出",
    "SHORT_GUARDS": "空头：价格保护镜像", "BOTH_GUARDS": "多空：共用一个点位的价格保护",
}
MODEL_FEATURES = ["log_holding_days", "cycle_return", "cycle_drawdown", "entry_mode", "mom5", "mom20", "sma120", "vol20"]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def save_table(name, data):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(path.with_suffix(".parquet"), index=False)
    data.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def make_features(source, dividends):
    data = source[["date", "open", "high", "low", "close", "volume", "feature_valid"]].copy()
    data["dividend"] = data.date.map(dividends.groupby("ex_date").cash_dividend_per_share.sum()).fillna(0.)
    data["overnight_log"] = np.log((data.open + data.dividend) / data.close.shift())
    data["intraday_log"] = np.log((data.close + data.dividend) / (data.open + data.dividend))
    data["total_log"] = data.overnight_log + data.intraday_log
    data["total_simple"] = np.expm1(data.total_log)
    data["wealth"] = (1 + data.total_simple.fillna(0.)).cumprod()
    data["mom5"] = data.total_log.rolling(5).sum()
    data["mom20"] = data.total_log.rolling(20).sum()
    data["sma120"] = data.wealth / data.wealth.rolling(120).mean() - 1
    data["vol20"] = data.total_simple.rolling(20).std(ddof=1) * np.sqrt(242)
    difference = data.intraday_log - data.overnight_log
    scale = difference.rolling(60).std(ddof=1) * np.sqrt(60)
    data["session_score60"] = difference.rolling(60).sum() / scale.replace(0, np.nan)
    for side, label in ((1, "long"), (-1, "short")):
        oriented = side * data.session_score60
        data[f"entry_{label}"] = oriented.gt(1) & oriented.shift().gt(1) & data.feature_valid
        data[f"exit_{label}"] = oriented.lt(0) & oriented.shift().lt(0)
    require(not (data.entry_long & data.entry_short).any(), "同一日同时出现相反的严格入场条件。")
    return data


def selected_model(models, entry_index):
    fits = [r["fit_index"] for r in models]
    index = bisect_right(fits, entry_index) - 1
    if index < 0:
        return None
    record = models[index]
    require(record["fit_index"] <= entry_index, "模型晚于首次持仓收盘。")
    if record["status"] == "FIT_COMPLETE":
        require(record["latest_exit_index"] <= record["fit_index"], "训练标签尚未成熟。")
    return record


def predict(record, values):
    if record is None or record["status"] != "FIT_COMPLETE" or not np.isfinite(values).all():
        return np.nan
    model = record["model"]
    require(model["features"] == MODEL_FEATURES and model["kind"] == "WITHIN_CYCLE_FIXED_INTERCEPT_RIDGE", "退出模型结构改变。")
    normalized = (values - np.asarray(model["mean"])) / np.asarray(model["scale"])
    normalized = np.clip(normalized, -model["feature_clip"], model["feature_clip"])
    return float(model["intercept"] + np.dot(normalized, np.asarray(model["coefficients"])))


def dividend_amount(dividends, entry_date, date, at_open):
    before = dividends.record_date.lt(date) if at_open else dividends.record_date.le(date)
    eligible = dividends.record_date.ge(entry_date) & before & dividends.ex_date.le(date)
    return float(dividends.loc[eligible, "cash_dividend_per_share"].sum())


def return_values(entry_raw, exit_raw, dividend, side, quantity):
    entry_fill, exit_fill = old.adverse_fill(entry_raw, side), old.adverse_fill(exit_raw, -side)
    commission = max(5., quantity * entry_fill * .0004) + max(5., quantity * exit_fill * .0004)
    net = (side * quantity * (exit_fill - entry_fill + dividend) - commission) / (quantity * entry_raw)
    return {"exit_fill": exit_fill, "dividend_per_share": dividend, "total_commission": commission,
            "point_net_return": net, "point_gross_return": side * (exit_raw - entry_raw + dividend) / entry_raw}


def protection_reasons(cycle_return, drawdown, held_closes, original_exit, learned_exit):
    reasons = []
    if original_exit:
        reasons.append("SESSION_STATE_REVERSED_TWO_CLOSES")
    if cycle_return <= -.06:
        reasons.append("LOSS_6_PERCENT")
    if drawdown <= -.08:
        reasons.append("TRAIL_8_PERCENT")
    if held_closes >= 60:
        reasons.append("TIME_60_CLOSES")
    if learned_exit:
        reasons.append("MODEL_NEGATIVE_TWO_CLOSES")
    return reasons


def evaluate_point(data, dividends, models, entry_index, side, learning):
    require(side in (-1, 1) and (side == 1 or not learning), "不允许把多头训练模型用于空头。")
    e = int(entry_index)
    entry_raw = float(data.open.iloc[e])
    quantity = math.floor(100000 / entry_raw / 100) * 100
    entry_fill = old.adverse_fill(entry_raw, side)
    entry_fee = max(5., quantity * entry_fill * .0004)
    raw_notional = quantity * entry_raw
    initial_reference_cost = raw_notional + side * quantity * (entry_fill - entry_raw) + entry_fee
    record = selected_model(models, e) if learning else None
    peak, negative_count, pending, decision = initial_reference_cost, 0, [], None
    rows = []
    base = {"entry_idx": e, "entry_date": data.date.iloc[e], "entry_origin": data.date.iloc[e-1],
            "direction": side, "entry_raw": entry_raw, "entry_fill": entry_fill, "quantity": quantity,
            "entry_fee": entry_fee, "reference_initial_cost": initial_reference_cost,
            "learning_enabled": learning, "model_selected_at": data.date.iloc[e] if learning else pd.NaT,
            "selected_fit_index": record["fit_index"] if record else None,
            "selected_fit_origin": pd.Timestamp(record["fit_origin"]) if record else pd.NaT,
            "selected_model_status": record["status"] if record else "NO_MODEL_SELECTED",
            "model_available_at_entry": bool(record and record["status"] == "FIT_COMPLETE")}
    for t in range(e, len(data)):
        date = data.date.iloc[t]
        if t == len(data) - 1:
            dividend = dividend_amount(dividends, base["entry_date"], date, True)
            mark = return_values(entry_raw, float(data.open.iloc[t]), dividend, side, quantity)
            return {**base, "status": "RIGHT_CENSORED", "exit_date": pd.NaT, "exit_idx": np.nan,
                    "point_net_return": np.nan, "point_gross_return": np.nan,
                    "last_observation_date": date, "terminal_open_mark_return": mark["point_net_return"],
                    "first_exit_decision_idx": decision, "exit_reasons": "|".join(pending)}, pd.DataFrame(rows)
        if pending and t > e:
            blocked = previous.open_blocked(data, t, -side)
            if not blocked:
                dividend = dividend_amount(dividends, base["entry_date"], date, True)
                result = return_values(entry_raw, float(data.open.iloc[t]), dividend, side, quantity)
                return {**base, **result, "status": "COMPLETE", "exit_idx": t, "exit_date": date,
                        "exit_raw": float(data.open.iloc[t]), "exit_origin": data.date.iloc[t-1],
                        "first_exit_decision_idx": decision, "first_exit_decision_date": data.date.iloc[decision],
                        "holding_sessions": t-e, "exit_reasons": "|".join(pending),
                        "blocked_exit_sessions": t-decision-1}, pd.DataFrame(rows)
        dividend = dividend_amount(dividends, base["entry_date"], date, False)
        current = raw_notional + side * quantity * (float(data.close.iloc[t]) - entry_raw + dividend)
        peak = max(peak, current)
        cycle_return, drawdown = current / initial_reference_cost - 1, current / peak - 1
        values = np.asarray([np.log1p(t-e+1), cycle_return, drawdown, 1., data.mom5.iloc[t], data.mom20.iloc[t], data.sma120.iloc[t], data.vol20.iloc[t]])
        prediction = predict(record, values) if learning else np.nan
        negative_count = negative_count + 1 if np.isfinite(prediction) and prediction < 0 else 0
        learned_exit = learning and negative_count >= 2
        if not pending:
            original_exit = bool(data.exit_long.iloc[t] if side == 1 else data.exit_short.iloc[t])
            pending = protection_reasons(cycle_return, drawdown, t-e+1, original_exit, learned_exit)
            if pending:
                decision = t
        rows.append({"entry_idx": e, "date": date, "origin_idx": t, "direction": side,
                     "learning_enabled": learning, "cycle_return": cycle_return, "cycle_drawdown": drawdown,
                     "continuation_prediction": prediction, "negative_count": negative_count,
                     "pending_exit": bool(pending), "exit_reasons": "|".join(pending),
                     "selected_fit_index": record["fit_index"] if record else None})
    raise RuntimeError("点位没有完整或末端状态。")


def sequential(data, dividends, models, policy, start):
    first = int(np.flatnonzero(data.date.ge(start))[0])
    t, last_exit, last_direction, rearmed = first-1, -1000000, 0, True
    points, states, events = [], [], []
    while t < len(data)-1:
        long = bool(data.entry_long.iloc[t]) and policy != "SHORT_GUARDS"
        short = bool(data.entry_short.iloc[t]) and policy in ("SHORT_GUARDS", "BOTH_GUARDS")
        direction = 1 if long else -1 if short else 0
        if direction == 0 or (last_direction and direction != last_direction):
            rearmed = True
        if direction and rearmed and t-last_exit >= 2:
            e = t+1
            if e == len(data)-1:
                events.append({"origin": data.date.iloc[t], "event": "NO_ENTRY_AT_END_BOUNDARY"})
                break
            blocked = previous.open_blocked(data, e, direction)
            if blocked:
                events.append({"origin": data.date.iloc[t], "event": "ENTRY_BLOCKED_" + blocked})
                t += 1
                continue
            point, path = evaluate_point(data, dividends, models, e, direction, policy == "LONG_MODEL")
            point["policy"] = policy
            path["policy"] = policy
            points.append(point)
            states.append(path)
            rearmed, last_direction = False, direction
            if point["status"] != "COMPLETE":
                break
            last_exit = int(point["exit_idx"])
            t = last_exit
            continue
        t += 1
    return pd.DataFrame(points), pd.concat(states, ignore_index=True) if states else pd.DataFrame(), pd.DataFrame(events)


def verify_sources(data, models):
    samples = pd.read_parquet(OUT / "inputs/model_training_samples.parquet")
    rows = []
    for record in models:
        fit = record["fit_index"]
        require(pd.Timestamp(record["fit_origin"]) == data.date.iloc[fit], "模型索引与原日期错位。")
        mature = samples.loc[samples.exit_index.le(fit)]
        cycles = mature[["cycle_id", "exit_index"]].drop_duplicates().sort_values(["exit_index", "cycle_id"])
        ids = cycles.tail(20).cycle_id.to_list()
        chosen = mature.loc[mature.cycle_id.isin(ids)]
        require(ids == record["training_cycles"] and len(chosen) == record["training_rows"], "模型训练成员不符合当时成熟窗口。")
        require(chosen.exit_index.max() == record["latest_exit_index"], "模型最大训练退出时点不同。")
        rows.append({"fit_origin": record["fit_origin"], "status": record["status"], "members": len(ids), "states": len(chosen),
                     "latest_training_exit_index": int(chosen.exit_index.max()), "fit_index": fit})
    prediction_errors = []
    by_date = {pd.Timestamp(r["fit_origin"]): r for r in models}
    for period in PERIODS:
        saved = pd.read_parquet(OUT / f"inputs/{period}/saved_model_decisions.parquet")
        available = saved.loc[saved.learning_status.eq("PREDICTION_AVAILABLE")]
        for row in available.to_dict("records"):
            record = by_date[row["learning_fit_origin"]]
            require(record["fit_index"] <= row["model_selection_index"] <= row["origin_index"], "原预测选择时钟包含未来。")
            values = np.asarray([row[name] for name in MODEL_FEATURES], dtype=float)
            recomputed = predict(record, values)
            prediction_errors.append(abs(recomputed-row["continuation_prediction"]))
    require(max(prediction_errors, default=0) < 1e-12, "原模型预测方程不能重现。")
    return pd.DataFrame(rows), {"recomputed_saved_predictions": len(prediction_errors), "maximum_saved_prediction_error": max(prediction_errors, default=0)}


def paired_comparison(data, dividends, models, anchors, period):
    rows, points = [], []
    for anchor in anchors.to_dict("records"):
        learned, _ = evaluate_point(data, dividends, models, int(anchor["entry_idx"]), 1, True)
        points.append({**learned, "period": period, "analysis": "PAIRED_FIXED_GUARD_ENTRIES"})
        complete = anchor["status"] == learned["status"] == "COMPLETE"
        if complete:
            require(learned["exit_idx"] <= anchor["exit_idx"], "相同入场与保护下增加退出条件却更晚退出。")
        rows.append({"period": period, "entry_date": anchor["entry_date"], "entry_idx": anchor["entry_idx"],
                     "guard_status": anchor["status"], "model_status": learned["status"], "both_complete": complete,
                     "guard_exit_date": anchor.get("exit_date"), "model_exit_date": learned.get("exit_date"),
                     "guard_return": anchor.get("point_net_return"), "model_return": learned.get("point_net_return"),
                     "net_return_increment": learned["point_net_return"]-anchor["point_net_return"] if complete else np.nan,
                     "holding_sessions_saved": anchor["holding_sessions"]-learned["holding_sessions"] if complete else np.nan,
                     "model_available_at_entry": learned["model_available_at_entry"],
                     "model_fit_origin": learned["selected_fit_origin"], "guard_reason": anchor["exit_reasons"], "model_reason": learned["exit_reasons"]})
    return pd.DataFrame(rows), pd.DataFrame(points)


def paired_stats(pairs, period, rng):
    complete = pairs.loc[pairs.both_complete]
    values = complete.net_return_increment.to_numpy()
    start, end = PERIODS[period]
    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year+1)
    blocks = [complete.loc[complete.entry_date.dt.year.eq(y), "net_return_increment"].to_numpy() for y in years]
    boot, empty = [], 0
    for _ in range(5000):
        sample = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        if len(sample):
            boot.append(sample.mean())
        else:
            empty += 1
    return {"period": period, "entry_anchors": len(pairs), "both_complete": len(complete),
            "unresolved_pairs": int((~pairs.both_complete).sum()), "model_available_pairs": int(complete.model_available_at_entry.sum()),
            "mean_increment": float(values.mean()), "median_increment": float(np.median(values)),
            "improved_pairs": int((values > 1e-12).sum()), "worsened_pairs": int((values < -1e-12).sum()),
            "unchanged_pairs": int((np.abs(values) <= 1e-12).sum()),
            "mean_sessions_saved": complete.holding_sessions_saved.mean(),
            "year_block_mean_p025": np.quantile(boot, .025), "year_block_mean_p975": np.quantile(boot, .975),
            "empty_year_bootstraps": empty, "not_independent_validation": True}


def freeze():
    require(not OUT.exists(), "已有研究登记，不覆盖。")
    OUT.mkdir(parents=True)
    files = {}
    def copy(src, relative):
        target = OUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        files[relative] = {"source": src.relative_to(ROOT).as_posix(), "sha256": common.digest(target)}
    copy(previous.OUT / "inputs/source_daily.parquet", "inputs/source_daily.parquet")
    copy(previous.OUT / "inputs/dividends.csv", "inputs/dividends.csv")
    copy(MODEL_SOURCE / "saved_models.json", "inputs/models.json")
    copy(MODEL_SOURCE / "extended_reference_samples.parquet", "inputs/model_training_samples.parquet")
    copy(ROOT / "config/510300_entry_vintage_exit_v1.json", "inputs/original_model_policy.json")
    copy(CONTEXT / "active_goal_effective_requirements.json", "inputs/current_requirements.json")
    copy(ROOT / "docs/510300_ENTRY_VINTAGE_EXIT_V1.md", "inputs/original_rules.md")
    for period in PERIODS:
        copy(SOURCE / f"{period}_entry_exit_conditions.parquet", f"inputs/{period}/saved_signals.parquet")
        copy(SOURCE / f"{period}/STRESS/ENTRY_VINTAGE_EXIT_decisions.parquet", f"inputs/{period}/saved_model_decisions.parquet")
        copy(SOURCE / f"{period}/STRESS/ENTRY_VINTAGE_EXIT_cycles.csv", f"inputs/{period}/saved_model_cycles.csv")
    for source in [Path(__file__), Path(common.__file__), Path(old.__file__), Path(previous.__file__), ROOT / "tests/test_point_entry_exit_contribution_v1.py"]:
        copy(source, f"code/{source.name}")
    protocol = {
        "study": "510300_POINT_ENTRY_EXIT_CONTRIBUTION_V1", "at": common.now(),
        "question": "较高点位盈亏比来自日线入场、价格保护还是训练退出；无训练镜像能否补充有效多空机会？",
        "prior_evidence": "原两条组合线索及既有参考的历史已知；本轮不是未见数据或独立验证。旧研究关闭状态保留。",
        "entry": "60日的每日含息日内对数收益减隔夜对数收益之和，除以该60日差值样本标准差乘sqrt(60)。多头连续两收盘严格>1，空头镜像连续两收盘严格<-1，保留原必要行情完整性。只使用日线开收盘。",
        "guards": "同方向持有参考价值相对入场含费用成本亏损达到6%，或较已观察收盘最高价值回撤达到8%，或60个持仓收盘，或方向化60日指标连续两收盘<0；任一触发，下一可平仓开盘退出。没有日内高低价触发成交。",
        "short_reference_value": "空头以初始原价名义金额加固定份额方向损益构造参考价值，扣股息；只作标的方向标签，不是可融券或期权收益。多头原风险保护口径可精确对应。",
        "model": "只用于多头；买入日首次收盘选择当时最新记录并固定至退出，首次无模型不后补；完整八项持仓/市场状态更新，两次连续负预测加入退出。",
        "model_sources": "只复用原141条记录；核对当时已结束的最近20个训练周期成员和原预测方程，零新训练。模型只训练于多头，不镜像参数。",
        "exit_lock": "退出意图一经产生持续到实际退出；受阻后不因指标转正而取消。",
        "reentry": "退出后在空仓期看到原进入条件消失或方向改变才重新获得资格；距退出至少两个交易日的收盘才可再发下一开盘入场。",
        "continuous_policies": POLICIES, "periods": PERIODS,
        "paired_analysis": "固定LONG_GUARDS连续策略产生的所有入场，逐笔同时评价相同保护与加入训练退出；未完成的任一侧单列。只有两侧完成者用于增量统计。配对结果不等同于独立连续模型策略。",
        "paired_uncertainty": "按所有入场日历年（包括零完成年）分块重采样5000次，随机种子20261001，报告平均净收益差的描述区间；不是选择后显著性结论。",
        "execution": "收盘决定、下一开盘；方向涨跌停或无有效价格/量则拒绝或顺延；同一连续策略只占一个点位。末端开盘仅作未完成标记，不强平计入胜率。",
        "cost": "固定约十万元初始原价名义份额、100份整手、每边佣金万四至少5元、滑点千一及0.001不利取整；净回报除初始原价名义金额。",
        "quality": "实际净胜率p×实际净B严格>1，并报告均值和利润因子；交易次数软目标，逐年按完成退出年计数。",
        "prohibited": "不改变60日、1阈值、两日确认、6%止损、8%回撤、60日持有、两日等待；不增加MACD/量价过滤，不拼接两段资金账户，不把最好子段升级为策略。",
        "new_continuous_point_replays": 8, "new_full_accounts": 0, "new_model_fits": 0,
        "orders_authorized": False, "goal_achieved": False,
    }
    common.save_json(OUT / "protocol.json", protocol)
    files["protocol.json"] = {"sha256": common.digest(OUT / "protocol.json")}
    common.save_json(OUT / "freeze.json", {"at": common.now(), "files": files})
    print("入场与退出贡献研究已固定：四种连续点位规则、两段历史及同入场配对。", flush=True)


def write_report(metrics, paired_summary, annual, reconciliation, summary):
    lines = ["# 日线入场与训练退出的点位收益贡献", "",
             "当前范围为510300标的日线及此前完整周线点位，p×实际净盈亏比>1，频率软目标。本轮将更深层的日线入场与价格保护拆开，检查训练退出的贡献，并加入无训练的空头镜像和共用一个点位的双向对照。", "",
             "## 固定规则", "",
             "以最近60日日内相对隔夜的标准化强弱为入场因子。多头连续两收盘>1，空头镜像连续两收盘<-1；次日开盘进入。保护固定为6%含费用成本亏损、8%已观察持仓价值回撤、60个持仓收盘，以及方向化指标连续两收盘<0。均只在收盘判断、下一可成交开盘退出，没有日内高低点成交。", "",
             "多头训练退出只复用原已保存的141个月度记录，买入日首次收盘选择当时可用版本，固定至退出，当前八项状态逐日更新。连续两次预测继续收益为负时加入退出，原保护始终存在。未成熟模型不补预测，空头不使用多头模型。", "",
             "退出意图锁定至成交。实际退出后须先在空仓期看到原入场条件消失或方向改变，并等至少两个交易日收盘，才能再次发出下一开盘入场。每条连续规则同一时刻只有一个点位，双向规则共享占用。", "",
             "## 连续规则结果", "",
             "| 规则 | 时期 | 完成数 | 胜率 | 实际B | pB | 每笔平均参考净回报 | 完整年均次数 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in metrics.itertuples():
        lines.append(f"| {POLICIES[r.policy]} | {old.PERIOD_NAMES[r.period]} | {r.n} | {r.p:.2%} | {r.b:.3f} | {r.product:.3f} | {r.mean:.2%} | {r.full_year_average:.2f} |")
    lines += ["", "每笔固定约十万元初始名义份额。每边佣金万四且至少5元、滑点千一、0.001元不利取整，净回报以初始原价名义金额归一化。股息按登记资格计入多头、扣减空头。空头只是标的方向损益参考，不包含借券或期权价格与可执行性。", "",
              "## 相同入场下，训练退出增加了什么", "",
              "配对锚点固定为原价格保护多头连续规则的全部实际入场。两种退出从相同日期、价格和份额开始，训练条件只能增加提前退出，不能改变这批锚点的选择。末端任一侧未完成者单列，不用未来补齐。", "",
              "| 时期 | 入场锚点 | 两侧完成 | 入场时有成熟模型 | 平均净回报增量 | 改善/变差/不变 | 平均少持有天数 | 年份区块描述区间 |", "|---|---:|---:|---:|---:|---|---:|---|"]
    for r in paired_summary.itertuples():
        lines.append(f"| {old.PERIOD_NAMES[r.period]} | {r.entry_anchors} | {r.both_complete} | {r.model_available_pairs} | {r.mean_increment:+.2%} | {r.improved_pairs}/{r.worsened_pairs}/{r.unchanged_pairs} | {r.mean_sessions_saved:.2f} | [{r.year_block_mean_p025:+.2%}, {r.year_block_mean_p975:+.2%}] |")
    lines += ["", "区间来自少数年份区块，2026为部分年份；它描述这批既有历史，不代表独立验证或未来收益保证。配对平均增量与连续规则的收益差是不同问题：提前退出还可能改变下一次再入场时点和次数。", "",
              "## 与旧训练退出点位的对应", "",
              "新点位独立计算当前持仓状态和预测，不读取旧策略的实际退出日期作指令。原周期表仅在完成后用于比较，期末强平记录排除。", "",
              "| 时期 | 新生成完成数 | 旧参考完成数 | 日期相同 | 仅新/仅旧 |", "|---|---:|---:|---:|---:|"]
    for r in reconciliation.itertuples():
        lines.append(f"| {old.PERIOD_NAMES[r.period]} | {r.new_count} | {r.old_count} | {r.matched} | {r.new_only}/{r.old_only} |")
    lines += ["", f"核对了{summary['model_records_checked']}条模型的成熟训练成员，并独立重算{summary['recomputed_saved_predictions']}条原已保存预测，最大误差为{summary['maximum_saved_prediction_error']:.3g}。本轮没有重新拟合系数；旧模型训练方式本身与反复使用历史的局限保留。", "",
              "## 频率、末端与范围", "",
              "逐年完成次数保留零笔年份，2026部分年份单列；最长空仓等待单列。八个连续重放有共享点位与同源行情，不能把记录相加当独立样本。末端未完成不以强平收益计入胜率。", "",
              "本轮零新完整资金账户、零新模型拟合、零参数搜索。所有旧研究的关闭状态保留；原价格因子、保护、模型时钟和再进入条件均未根据新结果调整。数据截至2026-08-14，当前市场点位未计算。", "",
              "结果文件包含完整规则逐笔、每日持仓模型状态、同入场配对、配对额外点位、年度频率、退出原因、与旧周期匹配、原模型成熟度及预测复算。当前目标保持进行中，未因历史点值达线自动宣布实现。", ""]
    (OUT / "研究结论.md").write_text("\n".join(lines), encoding="utf-8")


def run():
    require(not (OUT / "summary.json").exists(), "已有完整结果，不重复覆盖。")
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    for name, value in frozen["files"].items():
        require(common.digest(OUT / name) == value["sha256"], f"固定来源改变：{name}")
    require(common.digest(Path(__file__)) == frozen["files"][f"code/{Path(__file__).name}"]["sha256"], "运行代码不是固定版本。")
    source = pd.read_parquet(OUT / "inputs/source_daily.parquet")
    div = pd.read_csv(OUT / "inputs/dividends.csv", parse_dates=["record_date", "ex_date"])
    data = make_features(source, div)
    for column in ["overnight_log", "intraday_log", "mom5", "mom20", "sma120", "vol20"]:
        require(np.allclose(data[column], source[column], atol=1e-11, rtol=1e-11, equal_nan=True), f"原日线因素重建不同：{column}")
    models = json.loads((OUT / "inputs/models.json").read_text(encoding="utf-8"))["models"]
    require(len(models) == 141 and sum(x["status"] == "FIT_COMPLETE" for x in models) == 114, "原模型集合改变。")
    memberships, prediction_check = verify_sources(data, models)
    save_table("原模型成熟训练成员", memberships)
    all_points, all_states, all_events, all_pairs, paired_points = [], [], [], [], []
    metrics, yearly, paired_summaries, reconciliations, comparison_rows = [], [], [], [], []
    rng = np.random.default_rng(20261001)
    for period, (start, end) in PERIODS.items():
        d = data.loc[data.date.le(end)].reset_index(drop=True)
        saved_signals = pd.read_parquet(OUT / f"inputs/{period}/saved_signals.parquet")
        require(np.array_equal(d.entry_long.to_numpy(), saved_signals.entry_condition.to_numpy().astype(bool)), "重建多头入场与原定义不同。")
        require(np.array_equal(d.exit_long.to_numpy(), saved_signals.original_price_exit.to_numpy().astype(bool)), "重建价格状态退出与原定义不同。")
        period_points = {}
        for policy in POLICIES:
            points, states, events = sequential(d, div, models, policy, start)
            require(len(points) > 0, "固定规则没有任何点位，应单独说明不可估计。")
            for frame in (points, states, events):
                frame["period"] = period
                frame["policy"] = policy
            all_points.append(points)
            all_states.append(states)
            all_events.append(events)
            period_points[policy] = points
            complete = points.loc[points.status.eq("COMPLETE")]
            full_years = list(range(pd.Timestamp(start).year, pd.Timestamp(end).year + (pd.Timestamp(end).month == 12)))
            counts = complete.groupby(complete.exit_date.dt.year).size().to_dict()
            first_index = int(np.flatnonzero(d.date.ge(start))[0])
            flat_gaps = [int(points.entry_idx.iloc[0])-first_index]
            for left, right in zip(points.iloc[1:].to_dict("records"), points.iloc[:-1].to_dict("records")):
                flat_gaps.append(int(left["entry_idx"])-int(right["exit_idx"]))
            if points.iloc[-1].status == "COMPLETE":
                flat_gaps.append(len(d)-1-int(points.iloc[-1].exit_idx))
            metrics.append({"policy": policy, "period": period, **old.statistics(complete.point_net_return),
                            "full_year_average": float(np.mean([counts.get(y, 0) for y in full_years])),
                            "zero_full_years": ",".join(str(y) for y in full_years if counts.get(y, 0) == 0),
                            "maximum_flat_session_intervals": max(flat_gaps), "censored": int(points.status.ne("COMPLETE").sum()),
                            "median_holding_sessions": complete.holding_sessions.median()})
            for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year+1):
                yearly.append({"policy": policy, "period": period, "year": year, "full_year": year in full_years, "completed_points": counts.get(year, 0)})
        pairs, additional = paired_comparison(d, div, models, period_points["LONG_GUARDS"], period)
        all_pairs.append(pairs)
        paired_points.append(additional)
        paired_summaries.append(paired_stats(pairs, period, rng))
        old_cycles = pd.read_csv(OUT / f"inputs/{period}/saved_model_cycles.csv", parse_dates=["entry_date", "exit_date"])
        old_cycles = old_cycles.loc[~old_cycles.exit_reasons.str.contains("研究终点", regex=False)]
        rebuilt = period_points["LONG_MODEL"].loc[period_points["LONG_MODEL"].status.eq("COMPLETE")]
        merged = rebuilt.merge(old_cycles[["entry_date", "exit_date", "net_profit_cny", "entry_cost_cny"]], on=["entry_date", "exit_date"], how="outer", indicator=True)
        merged["period"] = period
        comparison_rows.append(merged)
        reconciliations.append({"period": period, "new_count": len(rebuilt), "old_count": len(old_cycles),
                                "matched": int(merged._merge.eq("both").sum()), "new_only": int(merged._merge.eq("left_only").sum()),
                                "old_only": int(merged._merge.eq("right_only").sum())})
    points = pd.concat(all_points, ignore_index=True)
    states = pd.concat(all_states, ignore_index=True)
    metrics, paired_summary, annual, reconciliation = pd.DataFrame(metrics), pd.DataFrame(paired_summaries), pd.DataFrame(yearly), pd.DataFrame(reconciliations)
    exit_reasons = points.loc[points.status.eq("COMPLETE")].groupby(["policy", "period", "exit_reasons"]).size().rename("count").reset_index()
    for name, frame in [("连续规则全部点位", points), ("连续规则每日持仓状态", states), ("入场拒绝及末端事件", pd.concat(all_events, ignore_index=True)),
                        ("连续规则指标", metrics), ("逐年次数", annual), ("同入场配对", pd.concat(all_pairs, ignore_index=True)),
                        ("配对训练退出点位", pd.concat(paired_points, ignore_index=True)), ("同入场增量汇总", paired_summary),
                        ("原模型点位匹配", pd.concat(comparison_rows, ignore_index=True)), ("原模型匹配汇总", reconciliation), ("退出原因", exit_reasons)]:
        save_table(name, frame)
    summary = {"at": common.now(), "study": "510300_POINT_ENTRY_EXIT_CONTRIBUTION_V1", "status": "FIXED_ENTRY_EXIT_CONTRIBUTION_AND_DIRECTIONAL_REPLAYS_COMPLETE",
               "new_point_replays": 8, "new_full_accounts": 0, "new_model_fits": 0, "model_records_checked": len(models),
               "completed_comparison_records": int(points.status.eq("COMPLETE").sum()), "censored_comparison_records": int(points.status.ne("COMPLETE").sum()),
               "numeric_pass_groups": int(metrics.point_estimate_pass.sum()), "groups_total": len(metrics),
               "policies_passing_both_periods": list(metrics.groupby("policy").point_estimate_pass.all().loc[lambda x:x].index),
               **prediction_check, "goal_achieved": False, "orders_authorized": False, "independent_validation": "NOT_ESTABLISHED"}
    write_report(metrics, paired_summary, annual, reconciliation, summary)
    common.save_json(OUT / "summary.json", summary)
    common.save_json(OUT / "verification.json", {"status": "SAVED_MODEL_MATURITY_PREDICTIONS_AND_ENTRY_CONDITIONS_REPRODUCED",
                     **prediction_check, "model_records": len(models), "new_model_fits": 0,
                     "source_signal_identity": True, "paired_earlier_or_equal_exit_invariant": True,
                     "point_reconciliation": reconciliation.to_dict("records")})
    print(json.dumps(common.clean(summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="固定日线点位的入场与训练退出贡献")
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    freeze() if args.action == "freeze" else run()
