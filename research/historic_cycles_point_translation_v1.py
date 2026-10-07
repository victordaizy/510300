"""把原十四方案保存的交易周期映射为固定份额点位，仅解释口径差异。"""
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

SOURCE = ROOT / "reports/research/510300_sparse_node_joint_quality_v1"
OUT = ROOT / "reports/research/510300_historic_cycles_point_translation_v1"
PERIODS = ("earlier_diagnostic", "evaluation")
PERIOD_NAMES = {"earlier_diagnostic": "2015—2019", "evaluation": "2020—2026年8月14日"}
NAMES = {
    "CORE_AUXILIARY_DRAWDOWN_GATE": "核心辅助回撤门槛",
    "DRAWDOWN_GATE_TWO_CLOSE_RECOVERY": "连续两日恢复",
    "DUAL_CONFIRMED_RUNS_AUXILIARY": "双方向确认辅助",
    "EITHER_CONFIRMED_RUNS_AUXILIARY": "任一方向确认辅助",
    "EPISODE_START_CONFIRMED_AUXILIARY": "段首确认辅助",
    "EPISODE_WAIT_CONFIRMED_AUXILIARY": "段内等待确认辅助",
    "EXPOSURE_EXPANSION_150": "既定预算一倍半",
    "LAG_CONFIRMED_RUNS_AUXILIARY": "相关方向确认辅助",
    "RUNS_CLOSED_CYCLE_BUDGET": "成熟周期预算",
    "RUNS_COVARIANCE_BUDGET": "协方差预算",
    "RUNS_OPPORTUNITY_CAPPED_SUM": "机会相加封顶",
    "RUNS_OPPORTUNITY_MAX": "机会取较大值",
    "RUNS_REFERENCE_BLEND": "既定各半组合",
    "SIGN_CONFIRMED_RUNS_AUXILIARY": "上涨方向确认辅助",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def boolean(value):
    if str(value).lower() in ("true", "1"):
        return True
    if str(value).lower() in ("false", "0"):
        return False
    raise ValueError(f"未知布尔值：{value}")


def statistics(values):
    values = np.asarray(values, dtype=float)
    require(np.isfinite(values).all(), "统计输入含非有限数值。")
    positive, negative = values[values > 0], values[values < 0]
    n = len(values)
    p = len(positive) / n if n else np.nan
    q = len(negative) / n if n else np.nan
    mean_win = positive.mean() if len(positive) else np.nan
    mean_loss = -negative.mean() if len(negative) else np.nan
    payoff = mean_win / mean_loss if len(positive) and len(negative) else np.nan
    product = p * payoff
    return {"n": n, "wins": len(positive), "losses": len(negative), "flat": int((values == 0).sum()),
            "p": p, "q": q, "b": payoff, "product": product,
            "mean": values.mean() if n else np.nan, "mean_win": mean_win, "mean_loss": mean_loss,
            "standard_expectation_loss_units": product - q,
            "profit_factor": positive.sum() / -negative.sum() if len(negative) else np.nan,
            "point_estimate_pass": bool(np.isfinite(product) and product > 1 and values.mean() > 0)}


def adverse_fill(raw, action):
    require(raw > 0 and action in (-1, 1), "成交价格或方向无效。")
    value = raw * (1 + action * .001) / .001
    return (math.ceil(value - 1e-9) if action == 1 else math.floor(value + 1e-9)) * .001


def saved_fill_exists(row):
    return (row.status in {"FILLED", "PARTIALLY_FILLED_CASH_OR_T_PLUS_ONE"}
            and row.filled_quantity != 0 and np.isfinite(row.fill_price) and row.fill_price > 0)


def entitled_dividend(dividends, entry_date, exit_date):
    # 开盘买入可取得当日登记权益；开盘卖出不能取得当天收盘的登记权益。
    eligible = dividends.record_date.ge(entry_date) & dividends.record_date.lt(exit_date)
    eligible &= dividends.ex_date.le(exit_date)
    return float(dividends.loc[eligible, "cash_dividend_per_share"].sum())


def point_reference(entry_raw, exit_raw, dividend):
    quantity = math.floor(100000 / entry_raw / 100) * 100
    require(quantity > 0, "参考份额必须为正。")
    entry_fill, exit_fill = adverse_fill(entry_raw, 1), adverse_fill(exit_raw, -1)
    fees = max(5., quantity * entry_fill * .0004) + max(5., quantity * exit_fill * .0004)
    denominator = quantity * entry_raw
    net = (quantity * (exit_fill - entry_fill + dividend) - fees) / denominator
    gross = (exit_raw - entry_raw + dividend) / entry_raw
    return {"quantity": quantity, "entry_raw": entry_raw, "exit_raw": exit_raw,
            "entry_fill": entry_fill, "exit_fill": exit_fill, "dividend_per_share": dividend,
            "reference_fees": fees, "reference_notional": denominator,
            "point_net_return": net, "point_gross_return": gross}


def save_table(name, frame):
    dest = OUT / "results" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    frame.to_parquet(dest.with_suffix(".parquet"), index=False)


def freeze():
    require(not OUT.exists(), "已有研究登记，不覆盖。")
    registry = pd.read_csv(SOURCE / "inputs/old_14_scope.csv")
    require(len(registry) == 14 and set(registry.model) == set(NAMES), "原十四方案清单改变。")
    OUT.mkdir(parents=True)
    files = {}

    def copy(src, relative):
        dst = OUT / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        files[relative] = {"source": src.relative_to(ROOT).as_posix(), "sha256": common.digest(dst)}

    copy(SOURCE / "inputs/old_14_scope.csv", "inputs/registry.csv")
    copy(common.OUT / "results/features.parquet", "inputs/features.parquet")
    copy(common.OUT / "inputs/dividends.csv", "inputs/dividends.csv")
    copy(ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/active_goal_effective_requirements.json", "inputs/current_requirements.json")
    copy(Path(__file__), "code/historic_cycles_point_translation_v1.py")
    copy(Path(common.__file__), "code/upward_episode_anatomy_v1.py")
    copy(ROOT / "tests/test_historic_cycles_point_translation_v1.py", "code/test_historic_cycles_point_translation_v1.py")
    for study in registry.source_study.unique():
        folder = SOURCE / "inputs/older" / study
        for filename in ("config.json", "cycles.csv", "original_acceptance_outcome.json"):
            copy(folder / filename, f"inputs/older/{study}/{filename}")
    for candidate in registry.itertuples():
        for period in PERIODS:
            relative = f"inputs/older/{candidate.source_study}/{period}/STRESS/{candidate.model}_ledger.parquet"
            copy(SOURCE / relative, relative)
    protocol = {
        "study": "510300_HISTORIC_CYCLES_POINT_TRANSLATION_V1", "at": common.now(),
        "question": "旧周期包含加减仓，且新门槛与新点位范围已经改变；固定旧进出日期后，较高的胜率乘盈亏比是否仍成立？",
        "known_results_before_registration": "已知旧十四方案2020年以后周期收益口径的p×B均大于1，2015—2019均小于1；本轮不是盲选或未见历史检验。",
        "scope": "原十四方案全部纳入，成本固定STRESS，两个原历史时期分别统计。多头历史点位仅用于解释。",
        "comparison": ["SAVED_CYCLE：原净利润除累计买入支出，包含原加减仓", "FIXED_POINT：原首笔买入开盘至原末笔清仓开盘，固定约十万元名义份额持有"],
        "fixed_reference_cost": {"commission_each_side": .0004, "minimum_each_side_cny": 5, "slippage_each_side": .001, "tick": .001, "lot": 100},
        "dividend": "首笔买入日到最后卖出日前取得登记资格，且退出时已除息的股息计入参考回报。",
        "denominator": "固定点位净收益除初始份额乘原始开盘价，与当轮六组点位一致。旧周期分母仍保留原累计买入支出。两者差异不能只归因于仓位。",
        "prices": "采用原保存账簿的原始开盘价；与当前日线快照逐日比较并保存差异。",
        "terminal": "原期末强平记录单独保存并排除完成点位统计，不拼接两段历史为连续策略。",
        "dependence": "原十四方案高度同源，保存并报告重复日期组合，不能把十四方案当十四个独立检验或把交易相加扩大样本。",
        "sensitivity": "固定报告删除最大盈利一次、逐个日历年删除的乘积范围；只作集中度诊断，不能选择删除后样本。",
        "uncertainty": "逐日历年分块重采样5000次，固定随机种子20261001；仅描述历史不确定性，不建立选择后显著性或独立验证。",
        "bootstrap_partial_year": "2026年保留原部分年度组，与其他年份一同重采样；每段年份组数量少，区间不能视为严格覆盖率保证。",
        "gate": "p×实际净盈亏比>1且平均净回报>0；频率软目标；逐年次数按清仓年份报告。",
        "limitation": "原末次清仓日期可能依赖原账户路径，固定份额映射不能直接作为可执行新策略。没有重新生成信号、优化参数或改写旧关闭状态。",
        "new_strategy_accounts": 0, "orders_authorized": False, "independent_validation": "NOT_ESTABLISHED",
    }
    common.save_json(OUT / "protocol.json", protocol)
    files["protocol.json"] = {"sha256": common.digest(OUT / "protocol.json")}
    common.save_json(OUT / "freeze.json", {"at": common.now(), "files": files})
    print("原十四方案点位映射已固定，不改变原入场或退出日期。", flush=True)


def collect():
    registry = pd.read_csv(OUT / "inputs/registry.csv")
    features = pd.read_parquet(OUT / "inputs/features.parquet").set_index("date")
    div = pd.read_csv(OUT / "inputs/dividends.csv", parse_dates=["record_date", "ex_date"])
    rows, checks, spans = [], [], []
    for candidate in registry.itertuples():
        folder = OUT / "inputs/older" / candidate.source_study
        config = json.loads((folder / "config.json").read_text(encoding="utf-8-sig"))
        cost = config["costs"]["STRESS"]
        require(cost == {"commission": .0004, "minimum": 5., "slippage": .001}, "原成本与映射成本不同。")
        cycles = pd.read_csv(folder / "cycles.csv")
        if "model" in cycles:
            cycles = cycles.loc[cycles.model.eq(candidate.model)]
        else:
            require(config["primary"] == candidate.model and config["candidate_configurations"] == 1, "原周期模型无法唯一确定。")
        for period in PERIODS:
            ledger = pd.read_parquet(folder / period / "STRESS" / f"{candidate.model}_ledger.parquet")
            require(ledger.date.is_monotonic_increasing and ledger.date.is_unique, "原账簿日期不唯一或不按时序。")
            require(ledger.date.isin(features.index).all(), "当前日线未覆盖旧账簿日期。")
            price_error = float(np.max(np.abs(ledger.open.to_numpy() - features.loc[ledger.date, "open"].to_numpy())))
            daily = ledger.set_index("date")
            chosen = cycles.loc[cycles.period.eq(period) & cycles.cost.eq("STRESS")].sort_values("entry_date")
            require(len(chosen) > 0, "原时期没有交易周期。")
            profit_error, debit_error, normalized_error = 0., 0., 0.
            previous_exit = pd.Timestamp.min
            for cycle in chosen.to_dict("records"):
                entry, exit_date = pd.Timestamp(cycle["entry_date"]), pd.Timestamp(cycle["exit_date"])
                require(entry > previous_exit and exit_date > entry, "交易周期重叠或违反隔夜持有。")
                previous_exit = exit_date
                first, last = daily.loc[entry], daily.loc[exit_date]
                require(first.shares_before == 0 and first.filled_quantity > 0, "周期起点不是从空仓买入。")
                require(last.shares_after == 0 and last.filled_quantity < 0, "周期终点不是清仓。")
                require(pd.Timestamp(cycle["entry_origin"]) < entry and pd.Timestamp(cycle["exit_origin"]) < exit_date, "原信号时钟不早于执行。")
                require(saved_fill_exists(first) and saved_fill_exists(last), "原边界没有已保存成交。")
                span = daily.loc[entry:exit_date]
                require(span.iloc[:-1].shares.gt(0).all(), "周期内部出现空仓。")
                buys = span.loc[span.filled_quantity.gt(0)]
                sells = span.loc[span.filled_quantity.lt(0)]
                require(len(buys) == cycle["buy_trades"] and len(sells) == cycle["sell_trades"], "原加减仓次数不一致。")
                recomputed_profit = float(-(span.filled_quantity * span.open).sum() + span.dividend_recognized.sum() - span.commission.sum() - span.slippage_cost.sum())
                recomputed_debit = float((buys.notional + buys.commission).sum())
                profit_error = max(profit_error, abs(recomputed_profit - cycle["net_profit"]))
                debit_error = max(debit_error, abs(recomputed_debit - cycle["buy_debit"]))
                normalized_error = max(normalized_error, abs(recomputed_profit / recomputed_debit - cycle["cycle_net_return"]))
                terminal = boolean(cycle["terminal_exit"])
                require(terminal == (last.mark_clock == "OPEN_TERMINAL"), "期末强平标记不一致。")
                dividend = entitled_dividend(div, entry, exit_date)
                reference = point_reference(float(first.open), float(last.open), dividend)
                rows.append({"model": candidate.model, "name": NAMES[candidate.model], "source_study": candidate.source_study,
                             "period": period, "cycle": int(cycle["cycle"]), "entry_date": entry, "exit_date": exit_date,
                             "entry_origin": pd.Timestamp(cycle["entry_origin"]), "exit_origin": pd.Timestamp(cycle["exit_origin"]),
                             "direction": 1, "terminal": terminal, "holding_sessions": len(span) - 1,
                             "original_entry_fill_status": first.status, "original_exit_fill_status": last.status,
                             "original_buy_trades": len(buys), "original_sell_trades": len(sells),
                             "original_multi_trade": len(buys) > 1 or len(sells) > 1,
                             "saved_cycle_return": float(cycle["cycle_net_return"]),
                             "saved_cycle_profit": float(cycle["net_profit"]), "saved_cycle_buy_debit": float(cycle["buy_debit"]),
                             "entry_exit_pair": f"{entry.date()}|{exit_date.date()}|LONG", **reference})
            require(profit_error < 1e-6 and debit_error < 1e-6 and normalized_error < 1e-10, "原账簿与周期收益无法复算。")
            checks.append({"model": candidate.model, "period": period, "rows": len(ledger), "cycles": len(chosen),
                           "max_saved_profit_error_cny": profit_error, "max_saved_debit_error_cny": debit_error,
                           "max_saved_return_error": normalized_error, "max_open_price_snapshot_difference": price_error})
            spans.append({"model": candidate.model, "period": period, "start": ledger.date.min(), "end": ledger.date.max(), "dates": list(ledger.date)})
    return pd.DataFrame(rows), pd.DataFrame(checks), spans


def uncertainty(g, rng):
    by_year = [part.point_net_return.to_numpy() for _, part in g.groupby(g.exit_date.dt.year)]
    all_years = list(sorted(g.exit_date.dt.year.unique()))
    samples, undefined = [], 0
    for _ in range(5000):
        values = np.concatenate([by_year[i] for i in rng.integers(0, len(by_year), len(by_year))])
        s = statistics(values)
        if np.isfinite(s["product"]):
            samples.append(s["product"])
        else:
            undefined += 1
    minus_largest = g.drop(g.point_net_return.idxmax())
    leave_year = []
    for year in all_years:
        s = statistics(g.loc[g.exit_date.dt.year.ne(year), "point_net_return"])
        leave_year.append({"removed_year": int(year), **s})
    valid = [x["product"] for x in leave_year if np.isfinite(x["product"])]
    return {"bootstrap_valid": len(samples), "bootstrap_undefined": undefined,
            "descriptive_year_bootstrap_p025": np.quantile(samples, .025) if samples else np.nan,
            "descriptive_year_bootstrap_p975": np.quantile(samples, .975) if samples else np.nan,
            "bootstrap_bound_is_conditional_on_finite_payoff": bool(undefined),
            "largest_win_share_of_sum": g.point_net_return.max() / g.point_net_return.sum() if g.point_net_return.sum() > 0 else np.nan,
            "product_without_largest_winner": statistics(minus_largest.point_net_return)["product"],
            "minimum_leave_year_product": min(valid) if valid else np.nan,
            "maximum_leave_year_product": max(valid) if valid else np.nan,
            "leave_year_undefined": len(leave_year) - len(valid)}, leave_year


def summarize(rows, spans):
    completed = rows.loc[~rows.terminal].copy()
    measurements, yearly, sensitivity, leave_years = [], [], [], []
    rng = np.random.default_rng(20261001)
    for span in spans:
        model, period = span["model"], span["period"]
        g = completed.loc[completed.model.eq(model) & completed.period.eq(period)]
        full_years = list(range(span["start"].year, span["end"].year + (1 if span["end"].month == 12 else 0)))
        annual_counts = g.groupby(g.exit_date.dt.year).size().to_dict()
        ordered = g.sort_values("entry_date")
        dates = pd.Index(span["dates"])
        start_indices = [int(dates.get_loc(x)) for x in ordered.entry_date]
        end_indices = [int(dates.get_loc(x)) for x in ordered.exit_date]
        gaps = [start_indices[0], *(left - right for left, right in zip(start_indices[1:], end_indices[:-1])), len(dates) - 1 - end_indices[-1]]
        base = {"model": model, "name": NAMES[model], "period": period,
                "start": span["start"], "end": span["end"], "full_years": len(full_years),
                "full_year_average": float(np.mean([annual_counts.get(y, 0) for y in full_years])),
                "zero_trade_full_years": ",".join(str(y) for y in full_years if not annual_counts.get(y, 0)),
                "maximum_idle_boundary_to_boundary_session_intervals": max(gaps),
                "multi_trade_cycles": int(g.original_multi_trade.sum()),
                "terminal_excluded": int(rows.loc[rows.model.eq(model) & rows.period.eq(period), "terminal"].sum())}
        for label, col in (("SAVED_CYCLE", "saved_cycle_return"), ("FIXED_POINT", "point_net_return")):
            s = statistics(g[col])
            require(abs(s["mean"] - s["mean_loss"] * s["standard_expectation_loss_units"]) < 1e-12, "标准期望恒等式错误。")
            measurements.append({**base, "basis": label, **s})
        for year in range(span["start"].year, span["end"].year + 1):
            yearly.append({"model": model, "name": NAMES[model], "period": period, "year": year,
                           "full_year": year in full_years, "completed_points": annual_counts.get(year, 0),
                           "terminal_excluded": int((rows.model.eq(model) & rows.period.eq(period) & rows.terminal & rows.exit_date.dt.year.eq(year)).sum())})
        extra, leave_year = uncertainty(g, rng)
        sensitivity.append({"model": model, "period": period, **extra})
        leave_years.extend({"model": model, "period": period, **value} for value in leave_year)
    duplicates = completed.groupby(["period", "entry_exit_pair"]).agg(
        model_count=("model", "nunique"), point_net_return=("point_net_return", "first"),
        entry_date=("entry_date", "first"), exit_date=("exit_date", "first"),
        models=("model", lambda x: "|".join(sorted(x)))).reset_index()
    return (pd.DataFrame(measurements), pd.DataFrame(yearly), pd.DataFrame(sensitivity),
            pd.DataFrame(leave_years), duplicates)


def write_report(measurements, sensitivity, summary):
    pivot = measurements.pivot(index="model", columns=["period", "basis"], values="product")
    lines = ["# 旧持仓周期与固定点位的口径对照", "",
             "用户当前要求：只研究510300日线及已完成周线多空点位，频率为软目标，扣参考摩擦后的胜率p×实际盈亏比B严格大于1。", "",
             "本轮将原十四方案保存的首笔买入和末笔清仓日期固定，按约十万元名义份额、双边佣金各万四且至少5元、双边滑点各千一、0.001元不利方向取整，计算等份额持有的标的多头参考收益。每笔以原始入场名义金额为分母，计入持有期间取得资格的股息。", "",
             "原周期口径保留多次加减仓，其分母是累计买入支出。与固定点位的差异同时包含中途加减仓、持有路径和归一化分母差异，不能归因于某一个因素。旧末次退出还可能依赖原账户路径，因此该映射是解释性测量，不是重新实现的可执行策略。", "",
             "## 完整范围", "",
             f"共复算{summary['ledger_groups']}组保存账簿，包含{summary['completed_comparison_rows']}条完成记录和{summary['terminal_excluded_rows']}条单列的期末强平记录。十四方案共享大量点位，完成记录只有{summary['unique_entry_exit_pairs']}组不同进出日期组合，且不同日期组合之间仍可能持仓重叠，不能将这些记录视为独立样本。两个时期分别统计，不拼接为连续账户。", "",
             "| 原方案 | 2015—2019原周期pB | 同日期固定点位pB | 2020以后原周期pB | 同日期固定点位pB |", "|---|---:|---:|---:|---:|"]
    for model in NAMES:
        row = pivot.loc[model]
        values = [row[(p, basis)] for p in PERIODS for basis in ("SAVED_CYCLE", "FIXED_POINT")]
        lines.append(f"| {NAMES[model]} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
    lines += ["", "以上是全部十四方案，未从中挑出最佳者；原关闭状态均保留。旧结果此前已知，不能把这一轮口径转换叫做独立验证。", "",
              "## 固定点位的频率和集中度", "",
              "| 原方案 | 历史时期 | 完成数 | 胜率 | 实际B | pB | 每笔平均参考净回报 | 完整年均次数 | 去最大盈利后pB | 年份重采样描述区间 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---|" ]
    joined = measurements.loc[measurements.basis.eq("FIXED_POINT")].merge(sensitivity, on=["model", "period"])
    for r in joined.itertuples():
        lines.append(f"| {r.name} | {PERIOD_NAMES[r.period]} | {r.n} | {r.p:.1%} | {r.b:.3f} | {r.product:.3f} | {r.mean:.2%} | {r.full_year_average:.2f} | {r.product_without_largest_winner:.3f} | [{r.descriptive_year_bootstrap_p025:.3f}, {r.descriptive_year_bootstrap_p975:.3f}] |")
    lines += ["", "年份分块重采样仅描述历史不确定性，每段仅5或7个年份组，2026年为部分年份；若重采样没有盈利或没有亏损，B无法计算，其次数在敏感性表保留，区间只基于有限值。该描述区间未校正此前反复研究及十四方案选择，不能当作显著性结论。删除最大盈利或单年也只是集中度诊断，不构成新交易过滤。", "",
              "## 数学口径", "",
              "p为盈利笔数占比，q为亏损笔数占比，B为实际平均正净收益率除以实际平均负净收益率的绝对值。标准亏损单位期望=pB−q；无平手时q=1−p。利润因子=正净收益率之和/负净收益率绝对值之和=pB/q。用户指定pB>1是额外、更严格的硬门槛。不同现金仓位收益、价格收益率和风险R单位不可混用分母。", "",
              "逐年次数、所有完成及期末记录、跨模型重复日期、逐年删除、原账簿复算误差均保存在results目录。没有期权定价、期权收益、完整账户新回测或委托。较早与主历史的共同截止日分别为2019-12-31和2026-08-14，不能据此给出当前市场点位。", ""]
    (OUT / "研究结论.md").write_text("\n".join(lines), encoding="utf-8")


def run():
    require((OUT / "freeze.json").exists(), "必须先登记固定范围。")
    require(not (OUT / "summary.json").exists(), "已有研究结果，不重复运行覆盖。")
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    for relative, info in frozen["files"].items():
        require(common.digest(OUT / relative) == info["sha256"], f"固定输入已改变：{relative}")
    expected_code = frozen["files"]["code/historic_cycles_point_translation_v1.py"]["sha256"]
    amendment_path = OUT / "implementation_fix_1.json"
    if amendment_path.exists():
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        require(amendment["original_sha256"] == expected_code, "实现修复不对应原固定代码。")
        expected_code = amendment["corrected_sha256"]
        require(common.digest(OUT / amendment["corrected_code"]) == expected_code, "实现修复快照已改变。")
    require(common.digest(Path(__file__)) == expected_code, "正在运行的研究定义不是固定版本或记录的实现修复。")
    rows, checks, spans = collect()
    measurements, annual, sensitivity, leave_year, duplicates = summarize(rows, spans)
    for name, frame in (("原周期与固定点位逐笔", rows), ("指标对照", measurements), ("逐年次数", annual),
                        ("集中度与描述区间", sensitivity), ("逐年删除诊断", leave_year),
                        ("跨模型重复点位", duplicates), ("原账簿复算", checks)):
        save_table(name, frame)
    summary = {"at": common.now(), "status": "SAVED_CYCLE_POINT_TRANSLATION_COMPLETE_NO_PROMOTION",
               "completed_comparison_rows": int((~rows.terminal).sum()), "terminal_excluded_rows": int(rows.terminal.sum()),
               "unique_entry_exit_pairs": len(duplicates), "ledger_groups": len(checks), "original_models": 14,
               "new_strategy_accounts": 0, "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED",
               "original_closures_preserved": True, "orders_authorized": False,
               "passes_by_basis_and_period": measurements.groupby(["basis", "period"]).point_estimate_pass.sum().astype(int).to_dict()}
    # 元组键转换为明确字符串，避免不同JSON实现的隐式转换。
    summary["passes_by_basis_and_period"] = {"|".join(key): value for key, value in summary["passes_by_basis_and_period"].items()}
    write_report(measurements, sensitivity, summary)
    common.save_json(OUT / "summary.json", summary)
    common.save_json(OUT / "verification.json", {
        "status": "PASS_SAVED_LEDGER_CYCLE_RETURN_AND_BOUNDARY_RECOMPUTATION",
        "maximum_profit_error_cny": checks.max_saved_profit_error_cny.max(),
        "maximum_debit_error_cny": checks.max_saved_debit_error_cny.max(),
        "maximum_cycle_return_error": checks.max_saved_return_error.max(),
        "maximum_raw_open_snapshot_difference": checks.max_open_price_snapshot_difference.max(),
        "matched_signal_before_execution": True, "terminal_exits_separate": True,
        "scope": "仅检验保存数据的数值与边界，不代表独立策略验证"})
    print(json.dumps(common.clean(summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="旧交易周期到固定份额点位的解释性映射")
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    freeze() if args.action == "freeze" else run()
