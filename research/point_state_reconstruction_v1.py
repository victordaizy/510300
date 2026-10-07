"""从既有收盘状态还原两条历史点位线索，限定组合层复现与量价归因。"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import upward_episode_anatomy_v1 as common
from research import historic_cycles_point_translation_v1 as old

OUT = ROOT / "reports/research/510300_point_state_reconstruction_v1"
CONTEXT = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001"
MODELS = {
    "CORE_AUXILIARY_DRAWDOWN_GATE": "510300_core_auxiliary_drawdown_gate_v1",
    "LAG_CONFIRMED_RUNS_AUXILIARY": "510300_return_confirmation_auxiliary_batch_v1",
}
PARENTS = {
    "CORE": ("510300_trend_noise_reference_blend_v1", "TREND_NOISE_REFERENCE_BLEND"),
    "RUNS": ("510300_return_runs_state_v1", "RETURN_RUNS_STATE"),
    "LAG": ("510300_return_lag_state_v1", "RETURN_LAG_STATE"),
}
PERIODS = {"earlier_diagnostic": ("2015-01-05", "2019-12-31"),
           "evaluation": ("2020-01-02", "2026-08-14")}
FEATURES = {
    "above_ema20": "收盘高于EMA20", "daily_hist_positive": "日MACD柱为正",
    "daily_hist_rising": "日MACD柱回升", "weekly_hist_positive": "上个完整周MACD柱为正",
    "weekly_hist_rising": "上个完整周MACD柱回升", "volume_expansion": "成交量达到此前20日中位数1.5倍",
    "up_volume_dominant": "近五日上涨方向成交量占优", "low_volatility": "20日波动不高于此前一年中位数",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(a, b, message, atol=1e-11):
    av, bv = np.asarray(a, float), np.asarray(b, float)
    require(av.shape == bv.shape and np.allclose(av, bv, rtol=1e-11, atol=atol, equal_nan=True), message)
    finite = np.isfinite(av) & np.isfinite(bv)
    return float(np.max(np.abs(av[finite] - bv[finite]))) if finite.any() else 0.


def save_table(name, frame):
    target = OUT / "results" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    frame.to_parquet(target.with_suffix(".parquet"), index=False)


def confirmed_direction(score, momentum, kind):
    direction = enter = weak = down = 0
    rows = []
    for s, m in zip(score, momentum):
        if not np.isfinite(s) or not np.isfinite(m):
            direction = enter = weak = down = 0
            rows.append(np.nan)
            continue
        strong = s < -1 if kind == "RUNS" else s > 0
        bad = s >= 0 if kind == "RUNS" else s <= 0
        enter = enter + 1 if strong and m > 0 else 0
        weak = weak + 1 if bad else 0
        down = down + 1 if m <= 0 else 0
        if enter >= 2:
            direction = 1
        if weak >= 2 or down >= 2:
            direction = 0
        rows.append(float(direction))
    return np.asarray(rows)


def reconstruct_market(data, dividends):
    result = data[["date", "open", "close", "volume"]].copy()
    ex_map = dividends.groupby("ex_date").cash_dividend_per_share.sum()
    result["dividend"] = result.date.map(ex_map).fillna(0.)
    ratio = (result.close + result.dividend) / result.close.shift()
    result["total_simple"] = ratio - 1
    result["total_log"] = np.log(ratio)
    result["wealth"] = ratio.fillna(1).cumprod()
    result["volatility20"] = result.total_simple.rolling(20).std(ddof=1) * np.sqrt(242)
    result["momentum60"] = result.total_log.rolling(60).sum()
    result["positive_trend120"] = (result.wealth / result.wealth.rolling(120).mean() - 1).clip(lower=0)
    result["noise120"] = result.volatility20 * np.sqrt(120 / 242)
    denominator = result.positive_trend120 + result.noise120
    result["budget131"] = result.positive_trend120 / denominator
    result.loc[denominator.eq(0), "budget131"] = 0.
    result["drawdown60"] = result.wealth / result.wealth.rolling(60).max() - 1
    z, correlation = np.full(len(data), np.nan), np.full(len(data), np.nan)
    values = result.total_log.to_numpy()
    for i in range(59, len(data)):
        sample = values[i-59:i+1]
        if not np.isfinite(sample).all():
            continue
        labels = np.sign(sample - np.median(sample))
        labels = labels[labels != 0]
        n, up, down = len(labels), int((labels > 0).sum()), int((labels < 0).sum())
        product = 2 * up * down
        variance = product * (product - n) / (n * n * (n - 1)) if n > 1 else np.nan
        if up and down and variance > 0:
            runs = 1 + int(np.count_nonzero(labels[1:] != labels[:-1]))
            z[i] = (runs - (1 + product / n)) / np.sqrt(variance)
        earlier, later = sample[:-1], sample[1:]
        if np.ptp(earlier) > 0 and np.ptp(later) > 0:
            correlation[i] = np.corrcoef(earlier, later)[0, 1]
    result["runs_score60"] = z
    result["lag_correlation60"] = correlation
    result["runs_direction"] = confirmed_direction(z, result.momentum60, "RUNS")
    result["lag_direction"] = confirmed_direction(correlation, result.momentum60, "LAG")
    target = np.full(len(result), np.nan)
    target[result.runs_direction.eq(0)] = 0.
    valid = result.runs_direction.eq(1) & result.volatility20.gt(0)
    target[valid] = np.minimum(1., .1 / result.loc[valid, "volatility20"])
    result["aux_target"] = target
    return result


def compose(core, auxiliary, gate):
    core, auxiliary, gate = np.asarray(core, float), np.asarray(auxiliary, float), np.asarray(gate, float)
    known = np.isfinite(core) & np.isfinite(auxiliary) & np.isfinite(gate)
    answer = np.full(core.shape, np.nan)
    effective = np.full(core.shape, np.nan)
    effective[known] = np.where(gate[known] == 1, auxiliary[known], 0)
    answer[known] = np.minimum(1., core[known] + effective[known])
    return answer, effective


def reconstruct_signals(data, market, period):
    parents = {key: pd.read_parquet(OUT / f"inputs/{period}/{key}_decisions.parquet") for key in PARENTS}
    core = parents["CORE"]
    expected_origins = pd.DatetimeIndex(data.date[data.date.ge(pd.Timestamp(PERIODS[period][0]))])
    first = int(np.flatnonzero(data.date.ge(PERIODS[period][0]))[0])
    expected_source = pd.DatetimeIndex(data.date.iloc[first-1:-1])
    for frame in parents.values():
        require(pd.DatetimeIndex(frame.origin).equals(expected_source), "父信号来源日期不按完整日历对齐。")
        require(pd.DatetimeIndex(frame.execution_date).equals(expected_origins), "父信号没有对齐下一开盘。")
    m = market.set_index("date").loc[core.origin].reset_index()
    x = core.VINTAGE_REFERENCE_RISK_parent_target.to_numpy()
    y = core.MODEL_SUPPORT_REFERENCE_ROUTER_parent_target.to_numpy()
    core_target = np.clip(m.budget131.to_numpy() * x + (1 - m.budget131.to_numpy()) * y, 0., 1.)
    errors = [{"period": period, "quantity": "core_budget", "error": same(m.budget131, core.reference_budget131, "核心连续预算不一致。")},
              {"period": period, "quantity": "core_target", "error": same(core_target, core.reference_weight, "核心目标重建不同。")},
              {"period": period, "quantity": "runs_score", "error": same(m.runs_score60, parents["RUNS"].run_score60, "连续段统计重建不同。")},
              {"period": period, "quantity": "runs_target", "error": same(m.aux_target, parents["RUNS"].reference_weight, "辅助目标重建不同。")},
              {"period": period, "quantity": "lag_correlation", "error": same(m.lag_correlation60, parents["LAG"].lag_correlation60, "相邻相关重建不同。")},
              {"period": period, "quantity": "lag_direction", "error": same(m.lag_direction, parents["LAG"].positive_direction, "相关方向状态重建不同。")}]
    same(m.runs_direction, parents["RUNS"].positive_direction, "连续段方向状态重建不同。")
    results = []
    for model in MODELS:
        saved = pd.read_parquet(OUT / f"inputs/{period}/{model}_decisions.parquet")
        require(pd.DatetimeIndex(saved.origin).equals(expected_source), "候选保存收盘时钟不一致。")
        gate = np.where(m.drawdown60.notna(), m.drawdown60.gt(-.05).astype(float), np.nan) if model.startswith("CORE_") else m.lag_direction.to_numpy()
        target, effective = compose(core_target, m.aux_target, gate)
        errors.append({"period": period, "quantity": model, "error": same(target, saved.reference_weight, "组合目标重建不同。")})
        frame = m.copy()
        frame.rename(columns={"date": "origin"}, inplace=True)
        frame["model"], frame["period"] = model, period
        frame["execution_date"] = core.execution_date.to_numpy()
        frame["core_target"], frame["gate"] = core_target, gate
        frame["effective_auxiliary"] = effective
        frame["target"] = target
        frame["known_before_open"] = frame.origin.lt(frame.execution_date)
        results.append(frame)
    return pd.concat(results, ignore_index=True), pd.DataFrame(errors)


def entry_cause(row):
    c, a = row.core_target > 0, row.effective_auxiliary > 0
    return "CORE_AND_AUXILIARY" if c and a else "CORE_ONLY" if c else "AUXILIARY_ONLY" if a else "NONE"


def open_blocked(data, index, action):
    if not np.isfinite(data.open.iloc[index]) or not np.isfinite(data.volume.iloc[index]) or data.volume.iloc[index] <= 0:
        return "NO_VALID_PRICE_OR_VOLUME"
    reference = data.close.iloc[index-1] - data.dividend.iloc[index]
    limit = round(reference * (1 + action * .1), 3)
    bad = data.open.iloc[index] >= limit - .0005 if action > 0 else data.open.iloc[index] <= limit + .0005
    return "DIRECTIONAL_OPEN_LIMIT" if bad else ""


def replay(data, dividends, signals, start):
    require(data.date.is_monotonic_increasing and data.date.is_unique, "价格日期不按时序。")
    daily = signals.set_index("execution_date")
    first = int(np.flatnonzero(data.date.ge(start))[0])
    held = None
    events, points = [], []
    for j in range(first, len(data)):
        date = data.date.iloc[j]
        signal = daily.loc[date]
        require(signal.origin < date, "当日开盘使用了未来收盘。")
        if j == len(data) - 1:
            if held is not None:
                dividend = old.entitled_dividend(dividends, held["entry_date"], date)
                mark = old.point_reference(held["entry_raw"], float(data.open.iloc[j]), dividend)
                points.append({**held, "status": "RIGHT_CENSORED", "exit_date": pd.NaT,
                               "exit_origin": pd.NaT, "point_net_return": np.nan,
                               "last_observation_date": date, "terminal_open_mark_return": mark["point_net_return"]})
            events.append({"date": date, "origin": signal.origin, "event": "END_BOUNDARY_NO_FORCED_TRADE", "target": signal.target})
            break
        if not np.isfinite(signal.target):
            events.append({"date": date, "origin": signal.origin, "event": "UNKNOWN_KEEP_POSITION", "target": np.nan})
            continue
        if held is None and signal.target > 0:
            blocked = open_blocked(data, j, 1)
            if blocked:
                events.append({"date": date, "origin": signal.origin, "event": f"ENTRY_BLOCKED_{blocked}", "target": signal.target})
                continue
            held = {"entry_date": date, "entry_origin": signal.origin, "entry_idx": j,
                    "entry_raw": float(data.open.iloc[j]), "entry_core_target": signal.core_target,
                    "entry_auxiliary_target": signal.effective_auxiliary, "entry_gate": signal.gate,
                    "entry_cause": entry_cause(signal), "entry_target": signal.target}
            events.append({"date": date, "origin": signal.origin, "event": "ENTER", "target": signal.target})
        elif held is not None and signal.target == 0:
            blocked = open_blocked(data, j, -1)
            if blocked:
                events.append({"date": date, "origin": signal.origin, "event": f"EXIT_BLOCKED_{blocked}", "target": signal.target})
                continue
            require(j > held["entry_idx"], "当日进入又退出违反隔夜限制。")
            dividend = old.entitled_dividend(dividends, held["entry_date"], date)
            result = old.point_reference(held["entry_raw"], float(data.open.iloc[j]), dividend)
            points.append({**held, **result, "exit_date": date, "exit_origin": signal.origin, "exit_idx": j,
                           "status": "COMPLETE", "holding_sessions": j - held["entry_idx"],
                           "exit_core_target": signal.core_target, "exit_auxiliary_target": signal.effective_auxiliary,
                           "exit_gate": signal.gate, "exit_raw_auxiliary": signal.aux_target,
                           "exit_reason": "ALL_KNOWN_COMPONENT_TARGETS_ZERO"})
            events.append({"date": date, "origin": signal.origin, "event": "EXIT", "target": signal.target})
            held = None
    return pd.DataFrame(points), pd.DataFrame(events)


def freeze():
    require(not OUT.exists(), "已有登记，不覆盖研究。")
    OUT.mkdir(parents=True)
    files = {}

    def copy(src, dst_name):
        dst = OUT / dst_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        files[dst_name] = {"source": src.relative_to(ROOT).as_posix(), "sha256": common.digest(dst)}

    copy(ROOT / "reports/research/510300_adaptive_allocation_v1/features.parquet", "inputs/source_daily.parquet")
    copy(old.OUT / "inputs/features.parquet", "inputs/technical_features.parquet")
    copy(old.OUT / "inputs/dividends.csv", "inputs/dividends.csv")
    copy(old.OUT / "results/原周期与固定点位逐笔.parquet", "inputs/saved_point_mapping.parquet")
    copy(CONTEXT / "active_goal_effective_requirements.json", "inputs/current_requirements.json")
    for period in PERIODS:
        for key, (study, model) in PARENTS.items():
            copy(ROOT / f"reports/research/{study}/{period}/STRESS/{model}_decisions.parquet", f"inputs/{period}/{key}_decisions.parquet")
        for model, study in MODELS.items():
            copy(ROOT / f"reports/research/{study}/{period}/STRESS/{model}_decisions.parquet", f"inputs/{period}/{model}_decisions.parquet")
    for file in [Path(__file__), Path(old.__file__), Path(common.__file__), ROOT / "tests/test_point_state_reconstruction_v1.py"]:
        copy(file, f"code/{file.name}")
    for filename in ["510300_TREND_NOISE_REFERENCE_BLEND_V1.md", "510300_RETURN_RUNS_STATE_V1.md", "510300_RETURN_LAG_STATE_V1.md", "510300_CORE_AUXILIARY_DRAWNDOWN_GATE_NEXT_20260912.md"]:
        copy(ROOT / "docs" / filename, f"source_rules/{filename}")
    protocol = {
        "study": "510300_POINT_STATE_RECONSTRUCTION_V1", "at": common.now(),
        "question": "两条旧日期线索能否由事前收盘组合状态生成，并在完整成功失败样本中解释量价差异？",
        "selection_known": "两条线索来自上一轮全部十四方案事后数值比较，既有价格历史与收益已知；不称独立验证。",
        "models": list(MODELS), "periods": PERIODS, "gate": "统一参考成本后p×实际B>1；频率软目标",
        "rebuild_market": "从原始收盘和除息重建含息收益、60日连续段分数、60日相邻收益相关、两日确认、20日风险幅度、120日趋势波动预算及60日回撤。",
        "reused_reference_inputs": "核心中的第131与139来源目标读取已保存的当时收盘输出；不重训其历史退出模型，也不声称本轮重建所有模型谱系。",
        "combination": "回撤线索=核心+60日回撤严格>-5%时的辅助；相关线索=核心+相邻相关允许状态下的辅助，上限一；任一必要资料未知则目标未知。",
        "point_state": "收盘目标>0且空仓：下一开盘尝试进入固定约十万元名义份额；目标=0且持有：下一开盘尝试退出；正目标大小变化不加减份额；未知维持已有状态。",
        "execution": "只使用真实日线开盘；无有效价格或成交量、进入方向涨跌停则当日不成交；随后按最新收盘目标重算，不保留过时意图。至少隔夜持有。",
        "cost": "每边佣金万四且至少5元，每边滑点千一，0.001元不利方向取整；100份整手，以初始原价名义金额归一化，按登记资格计股息。",
        "termination": "两段末日开盘仅标记未完成点位，不能强平后算成成功；不在末日开新点位。",
        "fixed_feature_diagnostics": FEATURES,
        "feature_threshold_origin": "沿用既有上涨图谱中的八个单项状态，不调指标周期、放量倍数或波动门槛；不生成组合条件。",
        "feature_assessment": "对每条线索、每个时期比较条件成立与不成立的样本胜率、B、乘积和平均回报；四组平均回报差和乘积差方向是否一致仅作描述。条件下的子集不是重放后的筛选策略。",
        "entry_exit_attribution": "保存入场时核心/辅助参与、退出时归零来源，以及已知技术状态。正负结果均保留。",
        "stop": "收盘目标重建不一致则停止，不微调阈值对齐；新点位与旧日期不一致则完整报告，不回填旧日期。",
        "new_point_replays": 4, "new_full_accounts": 0, "new_model_fits": 0, "parameter_searches": 0,
        "orders_authorized": False, "goal_achieved": False,
    }
    common.save_json(OUT / "protocol.json", protocol)
    files["protocol.json"] = {"sha256": common.digest(OUT / "protocol.json")}
    common.save_json(OUT / "freeze.json", {"at": common.now(), "files": files})
    print("两条历史线索的组合层复现与八项量价诊断已固定。", flush=True)


def feature_diagnostics(points, features):
    columns = ["date", *FEATURES, "daily_hist", "weekly_hist", "weekly_last_date", "weekly_available", "relative_volume", "rv_ratio", "up_volume_balance5", "ac", "ema20"]
    complete = points.loc[points.status.eq("COMPLETE")].merge(features[columns].rename(columns={"date": "feature_date"}),
                             left_on="entry_origin", right_on="feature_date", validate="many_to_one", how="left")
    require(complete.feature_date.notna().all() and complete.feature_date.lt(complete.entry_date).all(), "入场前特征日期不完整。")
    rows, pairs = [], []
    for (model, period), group in complete.groupby(["model", "period"], sort=False):
        for feature, name in FEATURES.items():
            known = group[feature].notna()
            if feature.startswith("weekly_"):
                known &= group.weekly_available
            stats = {}
            for value in (False, True):
                g = group.loc[known & group[feature].eq(value)]
                s = old.statistics(g.point_net_return)
                stats[value] = s
                rows.append({"model": model, "period": period, "feature": feature, "name": name,
                             "condition_true": value, "unknown_points": int((~known).sum()), **s,
                             "post_result_description_not_filter_strategy": True})
            pairs.append({"model": model, "period": period, "feature": feature, "name": name,
                          "n_true": stats[True]["n"], "n_false": stats[False]["n"],
                          "mean_increment_true_minus_false": stats[True]["mean"] - stats[False]["mean"],
                          "product_increment_true_minus_false": stats[True]["product"] - stats[False]["product"]})
    paired = pd.DataFrame(pairs)
    consistency = []
    for feature, group in paired.groupby("feature", sort=False):
        consistency.append({"feature": feature, "name": FEATURES[feature], "comparison_groups": len(group),
                            "positive_mean_difference_groups": int(group.mean_increment_true_minus_false.gt(0).sum()),
                            "positive_product_difference_groups": int(group.product_increment_true_minus_false.gt(0).sum()),
                            "all_four_mean_differences_positive": bool(len(group) == 4 and group.mean_increment_true_minus_false.gt(0).all()),
                            "all_four_product_differences_positive": bool(len(group) == 4 and group.product_increment_true_minus_false.gt(0).all()),
                            "minimum_side_count": int(min(group.n_true.min(), group.n_false.min()))})
    return complete, pd.DataFrame(rows), paired, pd.DataFrame(consistency)


def write_report(metrics, reconciliation, diagnostics, consistency, attribution, summary):
    lines = ["# 两条历史点位线索的组合层复现与量价解释", "",
             "本轮沿用用户最新要求：日线及此前完成周线、多空点位研究、p×实际净盈亏比>1、频率软目标。这里继续核查两条旧多头线索；既有空头失败记录保留，不据此宣称空头已达标。", "",
             "## 本轮究竟重建了什么", "",
             "从原始价格与股息重建60日收益连续段、相邻收益相关和各自的两日确认状态，重建20日波动、120日趋势波动预算及60日市场回撤。核心的两个更深层参考目标读取保存的当时收盘输出，本轮没有重新训练其退出模型，也没有完成底层全部历史模型的独立验证。", "",
             "连续段先按收益高于/低于窗口中位数分类，再计算连续同类段数；它不等于上涨天数，其负分数也不是盈利概率。统计量定义参考[NIST连续段说明](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35d.htm)，交易阈值则是原研究已固定的定义。", "",
             "回撤线索：核心为正可参与；60日市场含息回撤严格高于-5%时，连续段辅助也可参与。相关线索：核心为正可参与；相邻收益相关及累计方向形成原确认允许状态时，连续段辅助可参与。", "",
             "组合目标明确为正且空仓，下一开盘进入固定份额；目标明确为零，下一开盘退出；未知保持原状态。中途正目标比例的变化不加减仓。最后一个观察日仅标记未完成，不以强平结果计入胜率。", "",
             "## 由收盘信号重新生成的完成点位", "",
             "| 线索 | 时期 | 完成数 | 胜率 | 实际B | pB | 每笔平均参考净回报 |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in metrics.itertuples():
        lines.append(f"| {old.NAMES[r.model]} | {old.PERIOD_NAMES[r.period]} | {r.n} | {r.p:.2%} | {r.b:.3f} | {r.product:.3f} | {r.mean:.2%} |")
    lines += ["", f"共{summary['completed_points_with_model_overlap']}条完成记录、{summary['censored_points']}条末端未完成记录。按两条线索分别保存，不将共享点位合并当独立样本。旧日期匹配统计见下表；旧日期没有作为重放输入。", "",
              "| 线索 | 时期 | 新生成完成数 | 旧日期完成数 | 日期相同数 | 新增/缺失日期数 | 最大净回报差 |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in reconciliation.itertuples():
        lines.append(f"| {old.NAMES[r.model]} | {old.PERIOD_NAMES[r.period]} | {r.reconstructed} | {r.saved} | {r.matched} | {r.new_only}/{r.old_only} | {r.max_return_difference:.3g} |")
    lines += ["", "这证明的范围是组合层状态、点位生成与保存来源的对应关系。原参考退出可能包含训练和持仓路径，仍作为明确的来源输入，不能把本轮结果叫做全新样本或完整模型重训验证。", "",
              "## 八项既定技术量价状态的对照", "",
              "每项都在入场决定日观察，比较成立组和不成立组。四组指两条线索×两个时期，彼此共享行情，不能视为四份独立证据。表内次数只数正向差异；它不是统计显著性，也不是四组投票选出新过滤。", "",
              "| 事前条件 | 平均净回报提高的组数/4 | pB提高的组数/4 | 所有分组中较小一侧的最少样本数 |", "|---|---:|---:|---:|"]
    for r in consistency.itertuples():
        lines.append(f"| {r.name} | {r.positive_mean_difference_groups}/4 | {r.positive_product_difference_groups}/4 | {r.minimum_side_count} |")
    lines += ["", "全部64个条件成立/不成立分组均保留。分组仍使用原点位，未重新决定跳过信号后的进入时点，因此分组pB不能当作已经运行的新筛选策略。MACD、量能或波动的图上解释，需要在失败点位上同时接受检验。", "",
              "## 入场来源", "",
              "| 线索 | 时期 | 入场时参与来源 | 完成数 | 胜率 | pB |", "|---|---|---|---:|---:|---:|"]
    for r in attribution.itertuples():
        lines.append(f"| {old.NAMES[r.model]} | {old.PERIOD_NAMES[r.period]} | {r.entry_cause} | {r.n} | {r.p:.1%} | {r.product:.3f} |")
    lines += ["", "入场来源分类只是描述当时哪个分量为正，持有期间分量仍会变化，因此不能将该表称为独立核心策略或独立辅助策略的收益。", "",
              "## 仍未证明的事项", "",
              "两个时期的历史都已反复使用，两条线索本身也由上一轮事后对照识别。早期同一笔2015年大盈利的集中度问题仍然存在。本轮不改变其原关闭结论，不通过新分箱宣布目标实现。", "",
              "本轮四次点位状态重放、零完整账户新回测、零参数搜索、零新训练。参考费用及原价归一化沿用上一轮；它们是标的多头点位回报。原日线截至2026-08-14，不能据此形成当前市场观点。", "",
              "结果目录包含逐日来源状态、进出事件、所有完成及未完成点位、与旧日期的逐笔差异、入场前技术状态、八项全部分组、逐年次数和必要复算记录。", ""]
    (OUT / "研究结论.md").write_text("\n".join(lines), encoding="utf-8")


def run():
    require(not (OUT / "summary.json").exists(), "已有完整研究结果，不覆盖。")
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    for relative, value in frozen["files"].items():
        require(common.digest(OUT / relative) == value["sha256"], f"已固定来源改变：{relative}")
    require(common.digest(Path(__file__)) == frozen["files"][f"code/{Path(__file__).name}"]["sha256"], "运行代码与登记不同。")
    data = pd.read_parquet(OUT / "inputs/source_daily.parquet")
    dividends = pd.read_csv(OUT / "inputs/dividends.csv", parse_dates=["record_date", "ex_date"])
    market = reconstruct_market(data, dividends)
    same(market.total_simple, data.total_simple, "原始价股息重建的日回报不同。")
    same(market.total_log, data.total_log, "原始价股息重建的对数回报不同。")
    same(market.wealth, data.wealth, "原始价股息重建的累计财富不同。")
    signals_list, error_list, point_list, event_list, metric_list = [], [], [], [], []
    for period, (start, cutoff) in PERIODS.items():
        current = data.loc[data.date.le(cutoff)].reset_index(drop=True)
        m = market.loc[market.date.le(cutoff)].reset_index(drop=True)
        signals, errors = reconstruct_signals(current, m, period)
        signals_list.append(signals)
        error_list.append(errors)
        for model in MODELS:
            points, events = replay(m, dividends, signals.loc[signals.model.eq(model)], start)
            points["model"], points["period"] = model, period
            events["model"], events["period"] = model, period
            point_list.append(points)
            event_list.append(events)
            complete = points.loc[points.status.eq("COMPLETE")]
            metric_list.append({"model": model, "period": period, **old.statistics(complete.point_net_return)})
    points, events = pd.concat(point_list, ignore_index=True), pd.concat(event_list, ignore_index=True)
    signals, errors, metrics = pd.concat(signals_list, ignore_index=True), pd.concat(error_list, ignore_index=True), pd.DataFrame(metric_list)
    saved = pd.read_parquet(OUT / "inputs/saved_point_mapping.parquet")
    saved = saved.loc[saved.model.isin(MODELS) & ~saved.terminal]
    complete = points.loc[points.status.eq("COMPLETE")]
    matched = complete.merge(saved[["model", "period", "entry_date", "exit_date", "point_net_return"]], on=["model", "period", "entry_date", "exit_date"], how="outer", suffixes=("_reconstructed", "_saved"), indicator=True)
    matched["return_difference"] = matched.point_net_return_reconstructed - matched.point_net_return_saved
    reconciliation = []
    for (model, period), group in matched.groupby(["model", "period"], sort=False):
        reconciliation.append({"model": model, "period": period, "reconstructed": int(group._merge.ne("right_only").sum()),
                               "saved": int(group._merge.ne("left_only").sum()), "matched": int(group._merge.eq("both").sum()),
                               "new_only": int(group._merge.eq("left_only").sum()), "old_only": int(group._merge.eq("right_only").sum()),
                               "max_return_difference": group.return_difference.abs().max()})
    reconciliation = pd.DataFrame(reconciliation)
    features = pd.read_parquet(OUT / "inputs/technical_features.parquet")
    enriched, diagnostic, pairs, consistency = feature_diagnostics(points, features)
    attribution = []
    for (model, period, cause), group in complete.groupby(["model", "period", "entry_cause"], sort=False):
        attribution.append({"model": model, "period": period, "entry_cause": cause, **old.statistics(group.point_net_return)})
    attribution = pd.DataFrame(attribution)
    annual = []
    for model in MODELS:
        for period, (start, end) in PERIODS.items():
            g = complete.loc[complete.model.eq(model) & complete.period.eq(period)]
            for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
                annual.append({"model": model, "period": period, "year": year, "completed_points": int(g.exit_date.dt.year.eq(year).sum()),
                               "full_year": year < pd.Timestamp(end).year or pd.Timestamp(end).month == 12})
    for name, frame in [("逐日组合来源状态", signals), ("状态进出与拒绝事件", events), ("重建点位", points), ("点位指标", metrics),
                        ("与旧点位逐笔匹配", matched), ("与旧点位匹配汇总", reconciliation), ("入场前状态与结果标签", enriched),
                        ("八项全部分组", diagnostic), ("八项正反条件差异", pairs), ("八项跨期方向一致性", consistency),
                        ("入场来源归因", attribution), ("逐年次数", pd.DataFrame(annual)), ("逐日数值复算误差", errors)]:
        save_table(name, frame)
    summary = {"at": common.now(), "study": "510300_POINT_STATE_RECONSTRUCTION_V1", "status": "COMPOSITE_STATE_REPLAY_AND_FIXED_FEATURE_DIAGNOSTIC_COMPLETE",
               "reconstructed_decision_rows": len(signals), "completed_points_with_model_overlap": len(complete),
               "censored_points": int(points.status.eq("RIGHT_CENSORED").sum()),
               "unique_completed_entry_exit_pairs": len(complete[["entry_date", "exit_date"]].drop_duplicates()),
               "original_point_dates_all_reproduced": bool(reconciliation.new_only.eq(0).all() and reconciliation.old_only.eq(0).all()),
               "maximum_target_or_factor_error": errors.error.max(), "new_point_replays": 4, "new_full_accounts": 0,
               "new_model_fits": 0, "parameter_searches": 0, "numeric_pass_groups": int(metrics.point_estimate_pass.sum()),
               "all_four_positive_mean_feature_count": int(consistency.all_four_mean_differences_positive.sum()),
               "all_four_positive_product_feature_count": int(consistency.all_four_product_differences_positive.sum()),
               "independent_validation": "NOT_ESTABLISHED", "deep_reference_training_rebuilt": False,
               "goal_achieved": False, "orders_authorized": False}
    write_report(metrics, reconciliation, diagnostic, consistency, attribution, summary)
    common.save_json(OUT / "summary.json", summary)
    common.save_json(OUT / "verification.json", {"status": "PASS_COMPOSITE_SIGNAL_AND_POINT_RECONSTRUCTION",
                     "scope": "重建行情统计与组合层；更深层参考收盘输入仍复用原档，不是全谱系重训验证。",
                     "max_target_factor_error": errors.error.max(), "point_reconciliation": reconciliation.to_dict("records"),
                     "entry_features_before_entry": bool(enriched.feature_date.lt(enriched.entry_date).all())})
    print(json.dumps(common.clean(summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="组合状态到固定点位的有限复现")
    parser.add_argument("action", choices=("freeze", "run"))
    arguments = parser.parse_args()
    freeze() if arguments.action == "freeze" else run()
