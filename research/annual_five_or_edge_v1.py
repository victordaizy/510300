"""按每完整年五次、胜率或盈亏比优势，复核已保存的日线研究。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_annual_five_or_edge_v1"
MANDATE = ROOT / "config/510300_existing_data_training_mandate_v1.json"
CONTRACT = ROOT / "config/510300_daily_weekly_frequency_or_contract_v1.json"
USER = "我现在要求一年至少五次交易，要么胜率高，要么盈亏比大"
PERIODS = ["EARLY_FAILURE_DIAGNOSTIC", "LONG_REPLAY_DIAGNOSTIC", "RECENT_504_PRIMARY"]
POINT_NAMES = {"P0_CONFIRM": "周支撑＋日确认", "P1_SPACE": "再加计划净空间至少2", "P2_WEEKLY": "再加周MACD回升", "P3_CONTEXT": "再加日MACD、低波动、相对放量"}


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot():
    if OUT.exists() or CONTRACT.exists():
        raise RuntimeError("本轮文件已存在；只允许verify，不覆盖保存结果。")
    OUT.mkdir(parents=True)
    sources = []

    def copy(source, relative):
        target = OUT / "inputs" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        assert digest(source) == digest(target)
        sources.append({"source": source.relative_to(ROOT).as_posix(), "copy": "inputs/" + relative,
                        "sha256": digest(target)})

    copy(MANDATE, "mandate_before.json")
    p = ROOT / "reports/research/510300_weekly_daily_entry_locations_v1"
    copy(p / "results/四组全部交易与重叠标记.parquet", "point/trades.parquet")
    copy(p / "results/胜率实际盈亏比分组统计.parquet", "point/metrics.parquet")
    copy(p / "inputs/prices.parquet", "calendar_prices.parquet")
    copy(p / "protocol.json", "point/protocol.json")
    p = ROOT / "reports/research/510300_sparse_node_joint_quality_v1"
    copy(p / "results/原14方案跨时期联合指标.csv", "older/metrics.csv")
    copy(p / "results/统一交易周期.csv", "older/cycles.csv")
    p = ROOT / "reports/research/510300_sequential_patterns_regime_v1/results"
    copy(p / "annual.csv", "pattern/saved_annual.csv")
    for period in PERIODS:
        for filename in ["trades.csv", "daily.parquet", "metrics.json"]:
            copy(p / "accounts" / period / "STRESS/PATTERN_ONLY" / filename, "pattern/" + period + "/" + filename)
    copy(Path(__file__).resolve(), "recompute.py")
    previous = load_json(OUT / "inputs/mandate_before.json")
    protocol = {
        "study": "510300_ANNUAL_FIVE_OR_EDGE_V1", "recorded_at": now(), "user_instruction": USER,
        "method": "对既有保存结果的事后重计数与指标比较；不拟合、不重新回测、不把已看历史称为独立验证。",
        "scope": "最近4项日周线点位表达、既定日线顺序模式的3个原区间、原14个已列名历史方案的2个原区间；全部采用原STRESS成本。",
        "assets": ["510300.SH", "CASH_CNY"], "price_resolution": ["DAILY", "COMPLETED_WEEKLY"],
        "minimum_complete_cycles_each_full_calendar_year": 5,
        "cycle": "一次从现金买入至持仓全部卖出算一笔；分批订单合并；按入场年归属；必须在样本截止前真实完成原规则退出。",
        "terminal": "旧回测因样本终点强制清仓的交易另列，不计入自然完成次数；原指标及原裁决不改写。",
        "year": "对照日线交易日历核对完整自然年，零交易年补0；2026以及起止覆盖不完整的年单列，不按全年通过/失败裁决。跨年交易仅计一次。",
        "annual_counting_is_training_admission_gate": False,
        "annual_frequency_is_final_strategy_acceptance_gate": True,
        "force_trades_to_meet_frequency": False,
        "edge_logic": "WIN_RATE_OR_REALIZED_PAYOFF",
        "user_numeric_win_rate_minimum": None, "user_numeric_payoff_minimum": None,
        "working_win_rate_minimum": 0.55, "working_realized_payoff_minimum": 2.0,
        "working_threshold_source": "沿用上一轮冻结的55%与2，仅将AND改为OR；这是研究比较口径，用户尚未指定这两个数值。",
        "net_edge": "正成本后平均净收益，并且平均净利润为正；胜率和盈亏比至少一项达到上述研究比较线。",
        "return_payoff": "每笔净利润先除以该笔累计买入支出，再以盈利笔平均收益率除以亏损笔平均收益率绝对值。",
        "cash_payoff": "平均盈利金额/平均亏损金额绝对值，另列；不与利润因子混淆。",
        "payoff_not_defined": "无交易或没有同时观察到盈利和亏损时，不以无穷盈亏比通过。",
        "costs": "沿用三类输入的STRESS：单边佣金4bp、滑点10bp、最低佣金5元；保持旧成交价、整手及分红。",
        "account_targets": {k: previous[k] for k in ["capital_cny", "target_net_sharpe", "target_net_cagr", "target_max_drawdown"]},
        "account_limit": "旧账户保留其原仓位；不将其视为按当前最高50%仓位及尾部约束重新运行的账户。点位研究无连续账户指标。",
        "original_failures": "原主方案、冻结失败、历史策略停用和独立性状态全部保留；当前复核不恢复任何旧策略。",
        "evidence_date": "2026-09-16；原14方案为2026-08-14",
        "new_accounts": 0, "new_fits": 0, "minute_data_reads": 0, "orders_authorized": False,
        "official_t_plus_one_source": "https://www.sse.com.cn/assortment/fund/etf/question/c/c_20240118_5734755.shtml",
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "freeze.json", {"recorded_at": now(), "outcomes_previously_seen": True,
                                     "protocol_sha256": digest(OUT / "protocol.json"), "sources": sources})


def quality(trades):
    n = len(trades)
    pnl = trades.net_profit_cny.to_numpy(float)
    ret = trades.normalized_cycle_return.to_numpy(float)
    win, loss = pnl > 1e-6, pnl < -1e-6
    wr = float(win.mean()) if n else np.nan
    rp = float(ret[win].mean() / -ret[loss].mean()) if win.any() and loss.any() else np.nan
    cp = float(pnl[win].mean() / -pnl[loss].mean()) if win.any() and loss.any() else np.nan
    mean = float(ret.mean()) if n else np.nan
    z = 1.959963984540054
    center = (wr + z * z / (2 * n)) / (1 + z * z / n) if n else np.nan
    delta = z * math.sqrt(wr * (1 - wr) / n + z * z / (4 * n * n)) / (1 + z * z / n) if n else np.nan
    return {"cycles": n, "wins": int(win.sum()), "losses": int(loss.sum()), "win_rate": wr,
            "win_lower95_descriptive": center - delta, "win_upper95_descriptive": center + delta,
            "realized_return_payoff": rp, "realized_cash_payoff": cp,
            "mean_net_cycle_return": mean, "net_cycle_profit_cny": float(pnl.sum()),
            "net_profit_without_largest_winner_cny": float(pnl.sum() - max(pnl.max(), 0)) if n else np.nan,
            "winner_contribution_to_net_profit": float(pnl.max() / pnl.sum()) if n and pnl.sum() > 0 and win.any() else np.nan,
            "working_win_route": bool(n and wr >= .55 and mean > 0 and pnl.sum() > 0),
            "working_payoff_route": bool(n and np.isfinite(rp) and rp >= 2 and mean > 0 and pnl.sum() > 0)}


def annual_counts(calendar, trades, start, end, label):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    observed = calendar[(calendar >= start) & (calendar <= end)]
    rows = []
    for year in range(start.year, end.year + 1):
        ref = calendar[calendar.dt.year == year]
        obs = observed[observed.dt.year == year]
        complete = bool(len(ref) and len(obs) == len(ref) and obs.iloc[0] == ref.iloc[0]
                        and obs.iloc[-1] == ref.iloc[-1] and year < calendar.iloc[-1].year)
        sub = trades.loc[pd.to_datetime(trades.entry_date).dt.year == year]
        natural = sub.loc[~sub.terminal_or_censored]
        rows.append({"record_id": label, "year": year, "full_calendar_year": complete,
                     "covered_trading_days": len(obs), "saved_complete_cycles": len(sub),
                     "natural_complete_cycles": len(natural), "terminal_cycles_excluded": len(sub) - len(natural),
                     "at_least_five_in_full_year": bool(len(natural) >= 5) if complete else None})
    return rows


def compute():
    inputs = OUT / "inputs"
    protocol = load_json(OUT / "protocol.json")
    target = protocol["account_targets"]
    calendar = pd.to_datetime(pd.read_parquet(inputs / "calendar_prices.parquet").date).sort_values().reset_index(drop=True)
    records, annual, details, checks = [], [], [], []

    def add(label, family, model, period, start, end, trades, account=None, source_metric=None):
        t = trades.copy()
        t["record_id"] = label
        if len(t):
            assert pd.to_datetime(t.exit_date).gt(pd.to_datetime(t.entry_date)).all(), "出现不符合T+1的保存周期。"
            assert (t.buy_debit_cny > 0).all()
            assert np.allclose(t.net_profit_cny / t.buy_debit_cny, t.normalized_cycle_return, atol=1e-10, rtol=1e-10)
        all_quality = quality(t)
        natural_quality = quality(t.loc[~t.terminal_or_censored])
        y = annual_counts(calendar, t, start, end, label)
        full = [v for v in y if v["full_calendar_year"]]
        freq = bool(full and all(v["natural_complete_cycles"] >= 5 for v in full))
        edge = natural_quality["working_win_route"] or natural_quality["working_payoff_route"]
        row = {"record_id": label, "family": family, "model": model, "period": period,
               "start": start, "end": end, "cost": "STRESS", **natural_quality,
               "saved_cycles_including_terminal": len(t), "terminal_cycles_excluded": int(t.terminal_or_censored.sum()),
               "full_years": len(full), "minimum_cycles_full_year": min(v["natural_complete_cycles"] for v in full),
               "passing_full_years": sum(v["natural_complete_cycles"] >= 5 for v in full),
               "frequency_pass": freq, "working_edge_or_pass": edge, "new_requirements_point_screen": freq and edge,
               "current_account_risk_contract_replayed": False, "independent_validation": "NOT_ESTABLISHED",
               "strategy_accepted": False}
        if account:
            row.update(account)
            row["old_account_target_numeric_pass"] = bool(account["net_sharpe"] >= target["target_net_sharpe"]
                and account["net_cagr"] >= target["target_net_cagr"] and account["max_drawdown"] <= target["target_max_drawdown"])
        else:
            row.update(net_sharpe=np.nan, net_cagr=np.nan, max_drawdown=np.nan, old_account_target_numeric_pass=None)
        if source_metric:
            assert all_quality["cycles"] == source_metric["cycles"]
            for key in ["win_rate", "realized_cash_payoff"]:
                assert abs(all_quality[key] - source_metric[key]) < 1e-9
            checks.append({"record_id": label, "saved_cycle_quality_recomputed": True})
        records.append(row)
        annual.extend(y)
        details.append(t)

    point = pd.read_parquet(inputs / "point/trades.parquet")
    point = point.loc[(point.cost == "STRESS") & point.sequential_eligible]
    pm = pd.read_parquet(inputs / "point/metrics.parquet")
    for policy in POINT_NAMES:
        s = point.loc[point.policy == policy].copy()
        t = pd.DataFrame({"entry_date": s.entry_date, "exit_date": s.exit_date,
                          "net_profit_cny": s.net_pnl,
                          "buy_debit_cny": s.quantity * s.buy_price + (s.quantity * s.buy_price * .0004).clip(lower=5),
                          "exit_reason": s.exit_reason})
        t["normalized_cycle_return"] = t.net_profit_cny / t.buy_debit_cny
        t["terminal_or_censored"] = False
        metric = pm.loc[(pm.policy == policy) & (pm.cost == "STRESS") & (pm.scope == "NONOVERLAPPING_CYCLES")].iloc[0]
        assert len(t) == metric.events
        if len(t):
            assert abs(quality(t)["win_rate"] - metric.win_rate) < 1e-10
            assert abs(quality(t)["realized_cash_payoff"] - metric.payoff_ratio) < 1e-10
        checks.append({"record_id": "POINT|" + policy, "saved_cycle_quality_recomputed": True})
        add("POINT|" + policy, "WEEKLY_DAILY_POINTS", policy, "2015_TO_CUTOFF", "2015-01-01", "2026-09-16", t)

    saved_annual = pd.read_csv(inputs / "pattern/saved_annual.csv")
    for period in PERIODS:
        folder = inputs / "pattern" / period
        s = pd.read_csv(folder / "trades.csv")
        t = pd.DataFrame({"entry_date": s.entry_date, "exit_date": s.exit_date, "net_profit_cny": s.net_pnl,
                          "buy_debit_cny": s.quantity * s.entry_price + s.entry_fee, "exit_reason": s.exit_reason})
        t["normalized_cycle_return"] = t.net_profit_cny / t.buy_debit_cny
        t["terminal_or_censored"] = False
        m = load_json(folder / "metrics.json")
        d = pd.read_parquet(folder / "daily.parquet")
        ret = d.daily_return.to_numpy(float)
        nav = d.equity_cny.to_numpy(float)
        initial = 200000.
        assert np.max(np.abs(ret - nav / np.r_[initial, nav[:-1]] + 1)) < 1e-10
        # 该来源原指标使用252日；旧14方案使用242日，分别沿用，不静默混算。
        sh = float(ret.mean() / ret.std(ddof=1) * np.sqrt(252))
        cg = float((nav[-1] / initial) ** (252 / len(nav)) - 1)
        dd = float(-(nav / np.maximum.accumulate(np.r_[initial, nav])[1:] - 1).min())
        for calculated, key in [(sh, "net_sharpe"), (cg, "net_cagr"), (dd, "max_drawdown")]:
            assert abs(calculated - m[key]) < 1e-9
        assert not m["open_position"] and abs(t.net_profit_cny.sum() - (nav[-1] - initial)) < 1e-5
        for row in saved_annual.loc[(saved_annual.period == period) & (saved_annual.cost == "STRESS") & (saved_annual.policy == "PATTERN_ONLY")].itertuples():
            assert int(t.entry_date.str.startswith(str(row.year)).sum()) == row.complete_cycles_by_entry_year
        add("PATTERN|" + period, "DAILY_SEQUENCE_REFERENCE", "PATTERN_ONLY", period, m["start"], m["end"], t,
            {"net_sharpe": sh, "net_cagr": cg, "max_drawdown": dd, "source_annualization_days": 252,
             "maximum_saved_entry_debit_fraction": float((t.buy_debit_cny / s.entry_equity).max())},
            {"cycles": m["completed_cycles"], "win_rate": m["win_rate"], "realized_cash_payoff": m["payoff_ratio"]})

    metrics = pd.read_csv(inputs / "older/metrics.csv")
    cycles = pd.read_csv(inputs / "older/cycles.csv")
    assert metrics.model.nunique() == 14 and len(metrics) == 28
    assert not metrics.model.eq("SELECTED_MIX_BAND10_SIMPLE2").any()
    for m in metrics.itertuples():
        t = cycles.loc[cycles.account_id == m.account_id].copy()
        t["terminal_or_censored"] = t.exit_reason.str.contains("TERMINAL|CENSOR|END_CLOSE", regex=True, na=False)
        t = t[["entry_date", "exit_date", "net_profit_cny", "buy_debit_cny", "normalized_cycle_return", "exit_reason", "terminal_or_censored"]]
        add("OLDER|" + m.period + "|" + m.model, "SAVED_14_REFERENCE", m.model, m.period, m.start, m.end, t,
            {"net_sharpe": m.full_calendar_net_sharpe, "net_cagr": m.annualized_return, "source_annualization_days": 242,
             "max_drawdown": m.max_drawdown_magnitude},
            {"cycles": m.cycles, "win_rate": m.win_rate, "realized_cash_payoff": m.realized_cash_payoff_ratio})
    result, years, trades = pd.DataFrame(records), pd.DataFrame(annual), pd.concat(details, ignore_index=True)
    assert len(result) == 35 and not result.record_id.duplicated().any()
    assert len(checks) == 35
    return result, years, trades, checks


def update_authority():
    protocol = load_json(OUT / "protocol.json")
    # 共享任务书可能被其他任务更新，写入前重新读取，仅合并本次授权的偏好字段。
    before_bytes = MANDATE.read_bytes()
    current = json.loads(before_bytes.decode("utf-8"))
    write_json(OUT / "authority_before_write.json", current)
    updated = dict(current)
    stamp = now()
    receipt = "reports/research/510300_annual_five_or_edge_v1/authority_update.json"
    updated.update({
        "latest_frequency_user_instruction": USER, "latest_frequency_instruction_at": stamp,
        "opportunities_per_year_not_quota": False,
        "minimum_complete_cycles_each_full_calendar_year": 5,
        "annual_frequency_counting": protocol["cycle"] + "各完整自然年逐年满足，不用多年平均替代。",
        "annual_frequency_is_training_admission_gate": False,
        "annual_frequency_is_final_strategy_acceptance_gate": True,
        "annual_frequency_requirement_stage": "最终完整策略每个完整自然年至少5个自然完成周期；单个研究节点可以更少。",
        "annual_frequency_force_trade_enabled": False,
        "annual_frequency_revision_at": stamp, "annual_frequency_revision_receipt": receipt,
        "joint_node_objective": ["高胜率或高实际盈亏比，至少满足一项", "正的成本后单次期望", "完整账户高夏普", "有限尾部损失"],
        "node_edge_objective_logic": "WIN_RATE_OR_REALIZED_PAYOFF",
        "node_direction_revision_at": stamp, "node_direction_revision_receipt": receipt,
        "node_training_contract": "config/510300_daily_weekly_frequency_or_contract_v1.json",
        "permitted_price_observation_resolutions": ["DAILY", "COMPLETED_WEEKLY"],
        "minute_data_research_enabled": False,
        "research_working_edge_thresholds": {"win_rate": .55, "realized_return_payoff": 2.,
             "logic": "OR", "user_specified_numeric_thresholds": False,
             "reason": protocol["working_threshold_source"]},
        "latest_daily_weekly_frequency_reassessment": "reports/research/510300_annual_five_or_edge_v1/result.json",
    })
    assert MANDATE.read_bytes() == before_bytes, "共享任务书在读取后改变，先保留研究结果，避免覆盖。"
    write_json(CONTRACT, protocol)
    write_json(MANDATE, updated)
    changed = [k for k in updated if current.get(k) != updated[k]]
    assert updated["current_round"] == current["current_round"]
    for key in ["target_net_sharpe", "target_net_cagr", "target_max_drawdown", "capital_cny", "account_tail_risk_contract"]:
        assert updated[key] == current[key]
    write_json(OUT / "mandate_after.json", updated)
    write_json(OUT / "authority_update.json", {"at": stamp, "user_instruction": USER, "changed_fields": changed,
        "supersedes": "本任务每年机会无最低次数及胜率与盈亏比必须同时较高的旧要求。",
        "previous_node_contract": current.get("node_training_contract"),
        "other_current_round_preserved": current["current_round"], "old_study_results_modified": False,
        "before_sha256": hashlib.sha256(before_bytes).hexdigest(), "after_sha256": digest(MANDATE)})


def report(result, years):
    def pct(x):
        return "未计算" if pd.isna(x) else f"{x:.2%}"

    def num(x):
        return "未定义" if pd.isna(x) else f"{x:.2f}"

    rows = []
    selected = result.loc[result.family != "SAVED_14_REFERENCE"]
    for s in selected.itertuples():
        name = POINT_NAMES[s.model] if s.family == "WEEKLY_DAILY_POINTS" else {
            PERIODS[0]: "日线顺序模式：2015—2019", PERIODS[1]: "日线顺序模式：2020—2026", PERIODS[2]: "日线顺序模式：最近504日"}[s.period]
        rows.append(f"| {name} | {s.cycles} | {pct(s.win_rate)} | {num(s.realized_return_payoff)} | {pct(s.mean_net_cycle_return)} | {s.passing_full_years}/{s.full_years} | {num(s.net_sharpe)} |")
    p0 = years.loc[years.record_id == "POINT|P0_CONFIRM"]
    pattern = years.loc[years.record_id.isin(["PATTERN|" + PERIODS[0], "PATTERN|" + PERIODS[1]])]
    yearly = []
    for year in range(2015, 2027):
        a = p0.loc[p0.year == year].iloc[0]
        b = pattern.loc[pattern.year == year].iloc[0]
        yearly.append(f"| {year}{'（截至9月16日）' if year == 2026 else ''} | {a.natural_complete_cycles} | {b.natural_complete_cycles} | {'未完整，不裁决' if year == 2026 else ('通过' if b.natural_complete_cycles >= 5 else '不足5次')} |")
    text = "\n".join([
        "# 每年至少五次，胜率或实际盈亏比：现有日周线结果复核", "",
        "用户最新要求已落实：每个完整自然年至少5个完整交易周期，优势来自较高胜率或较大实际盈亏比，二者满足一项即可。此次对35份已有指标记录重新计数，未新增策略或账户。仍没有证据足以认定完整策略达标。", "",
        "## 当前研究标准", "",
        "一次买入到全部卖出计一笔，按入场年归属；分批买卖不拆成多笔。逐年补齐零交易年份，不用多年平均替代。不完整年份单列；未退出和样本终点强制清仓不用于凑次数。此处每年5次是策略验收要求，并非年底补单指令。", "",
        "胜率或盈亏比的数值尚未由用户指定。为保证比较连续，本轮沿用上一轮研究口径：压力成本后胜率≥55%，或平均实际盈亏比≥2，同时要求平均净收益及平均净利润为正。55%和2只是待明确的研究比较线，不是保证。旧的20万元、完整账户净夏普≥1.2、净年化≥10%、最大回撤≤10%及仓位风险约束保留。", "",
        "只用日线与已完成周线。股票ETF按T+1交易：[上交所说明](https://www.sse.com.cn/assortment/fund/etf/question/c/c_20240118_5734755.shtml)。本轮只读取原保存交易记录，没有假设同日买卖。", "",
        "## 与新增要求最相关的结果", "",
        "| 既有方案及原区间 | 自然完成笔数 | 压力净胜率 | 实际盈亏比 | 平均单笔净收益率 | 满5次的完整年/完整年总数 | 原账户净夏普 |",
        "|---|---:|---:|---:|---:|---:|---:|", *rows, "",
        "表中实际盈亏比统一按每笔净利润/该笔买入支出得到的收益率计算；金额盈亏比单独保存在结果表。因仓位与成交支出不同，两者不能混称。最近日周线点位的平均收益是单笔投入收益，连续完整账户尚未计算。三个顺序模式区间沿用原研究边界，彼此存在重叠，不能把次数相加或拼接收益。年化沿用来源口径：日线顺序模式252日，原14方案242日；本轮不据此跨组排名。", "",
        "基础周支撑点位的胜率73.68%可列入胜率路线观察，但19笔样本的胜率描述性95%区间约51.21%—88.19%，上一轮净均值区间也跨零。2015—2025只有2018年达到5笔；增加MACD、低波动、放量条件后次数更少。因此它不符合新的年度频率要求。", "",
        "日线突破/收复/修复对照在原2020年至2026年9月16日期间共47笔，胜率40.43%，收益率盈亏比2.44、金额盈亏比2.35，平均每笔净收益约0.50%；2020—2025每个完整年有5—10笔，符合新增的频率与大盈亏比比较线。但是完整账户净夏普仅0.4416、净年化3.20%、最大回撤11.19%，没有达到保留的账户目标。它只是显示这种优势形态在保存结果中存在，不构成旧策略恢复。", "",
        "该日线模式的2015—2019原对照期有26笔、胜率50%、收益率盈亏比0.77，平均净收益为负，且2016年4笔、2017年3笔。不能只保留2020年以后的有利区间。2020年后最大一笔盈利来自2024年9月25日至10月9日，占总净利润约92.67%；移除该笔后剩余净利润仅约3304元。这是集中度说明，未据此删除该笔或重新回测。", "",
        "原14个历史方案在2020—2025均有年份不足5笔：2021年统一4笔、2022年统一2笔、2023年0—2笔；该组全部不满足每完整年五笔。其较高的保存夏普不能代替频率要求；14项高度相关的变体也不是14份独立证据。旧方案中的样本终点退出已在计数表中另列。", "",
        "## 逐年次数", "",
        "| 年份 | 周支撑＋日确认 | 原日线顺序模式 | 日线顺序模式当年次数判断 |",
        "|---|---:|---:|---|", *yearly, "",
        "日线顺序模式的2015—2019和2020—2026来自原先分开的账户实验；这里只并列逐年次数，不拼接为新账户。2026年没有完整覆盖，6笔可以披露，但不据此宣称全年已经通过所有要求。", "",
        "![逐年交易次数](逐年交易次数.png)", "",
        "## 对下一阶段研究的约束", "",
        "频率不足与质量不足应分别处理：周线支撑低频节点可作为补充观察，不能独自承担年度五笔；日线确认机制能够形成更密集的机会，但仍须证明收益不只依赖少数行情。胜率路线和大盈亏比路线分别报告，不再要求每笔事前目标空间都≥2，也不强制两种优势同时存在。MACD、波动率和成交量继续作为解释条件，任何新增筛选都要同时显示它减少了多少机会；不因次数不足临时松阈值，也不在看到结果后延长持有期。", "",
        "未重新运行当前50%最高仓位和尾部约束下的账户。已有日线顺序模式按原规则接近满仓入场；旧完整账户数字仅是历史对照，不能直接作为当前风险合同的验证。样本均已被使用，独立验证未建立。最新价格仅到2026年9月16日，本报告没有当前进场价。", "",
        "可复核文件：[任务标准](protocol.json)、[35项方案比较](results/方案对照.csv)、[逐年完整次数](results/逐年交易次数.csv)、[原周期统一口径](results/重新计数周期.csv)、[计算结果](result.json)、[验证](verification.json)。", "",
    ])
    (OUT / "标准更新与现有方案结论.md").write_text(text, encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(12.5, 5.8), dpi=160)
    xs = np.arange(12)
    aval = p0.natural_complete_cycles.to_numpy()
    bval = pattern.sort_values("year").natural_complete_cycles.to_numpy()
    ax.bar(xs - .19, aval, .36, label="周支撑＋日确认", color="#64748b")
    ax.bar(xs + .19, bval, .36, label="原日线顺序模式", color="#2563a6")
    ax.axhline(5, color="#b45309", linewidth=1.6, linestyle="--", label="每完整年最低5笔")
    ax.axvline(4.5, color="#cbd5e1", linewidth=1, linestyle=":")
    ax.axvspan(10.5, 11.5, color="#fef3c7", alpha=.45)
    for x, a, b in zip(xs, aval, bval):
        ax.text(x - .19, a + .16, str(a), ha="center", fontsize=9)
        ax.text(x + .19, b + .16, str(b), ha="center", fontsize=9)
    ax.set_xticks(xs, [str(y) if y < 2026 else "2026*" for y in range(2015, 2027)])
    ax.set_ylim(0, 12)
    ax.set_ylabel("自然完成交易周期，按入场年归属")
    ax.set_title("年度五笔要求：频率须逐年满足", loc="left", fontsize=17, pad=18)
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", ncol=3)
    fig.text(.075, .025, "* 2026仅截至9月16日。虚线分隔原日线模式的两个既定研究区间；只并列次数，不拼接账户。", fontsize=10, color="#475569")
    fig.tight_layout(rect=[0, .06, 1, 1])
    fig.savefig(OUT / "逐年交易次数.png", facecolor="white")
    fig.savefig(OUT / "逐年交易次数.svg", facecolor="white")
    plt.close(fig)


def save(result, years, trades, checks):
    folder = OUT / "results"
    folder.mkdir(exist_ok=True)
    for name, table in [("方案对照", result), ("逐年交易次数", years), ("重新计数周期", trades)]:
        table.to_csv(folder / (name + ".csv"), index=False, encoding="utf-8-sig")
    summary = {"study": "510300_ANNUAL_FIVE_OR_EDGE_V1", "status": "REQUIREMENTS_UPDATED_SAVED_RESULTS_REMEASURED",
               "records": len(result), "annual_rows": len(years), "saved_cycle_rows": len(trades),
               "working_new_requirement_pass_records": result.loc[result.new_requirements_point_screen, "record_id"].tolist(),
               "strategy_accepted_count": 0, "goal_achieved": False, "research_round_complete": True,
               "new_accounts": 0, "new_fits": 0, "minute_data_reads": 0,
               "working_numeric_thresholds_user_specified": False, "current_market_view": "NO_VIEW_STALE_LOCAL_DATA",
               "scope": "35条原始指标记录，含重复时期与相关模型，不是35次独立实验。"}
    write_json(OUT / "result.json", summary)
    write_json(OUT / "recomputed_source_metrics.json", checks)


def verify():
    frozen = load_json(OUT / "freeze.json")
    assert digest(OUT / "protocol.json") == frozen["protocol_sha256"]
    for item in frozen["sources"]:
        assert digest(OUT / item["copy"]) == item["sha256"], item["copy"]
    revision = OUT / "code_revision.json"
    if revision.exists():
        meta = load_json(revision)
        assert digest(Path(__file__).resolve()) == meta["active_code_sha256"]
        assert digest(OUT / meta["corrected_copy"]) == meta["active_code_sha256"]
    result, years, trades, checks = compute()
    for name, table in [("方案对照", result), ("逐年交易次数", years), ("重新计数周期", trades)]:
        saved = pd.read_csv(OUT / "results" / (name + ".csv"))
        # CSV空字段还原为NaN；对照内存中的None时统一缺失表示，保留所有有效值。
        expected = table.mask(table.isna(), np.nan)
        for column in ("entry_date", "exit_date"):
            if column in expected:
                saved[column] = pd.to_datetime(saved[column], format="mixed")
                expected[column] = pd.to_datetime(expected[column], format="mixed")
        pd.testing.assert_frame_equal(saved, expected, check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-8)
    p0 = years.loc[(years.record_id == "POINT|P0_CONFIRM") & years.full_calendar_year]
    assert len(p0) == 11 and int(p0.natural_complete_cycles.eq(0).sum()) == 2
    assert p0.loc[p0.at_least_five_in_full_year.eq(True), "year"].tolist() == [2018]
    long = years.loc[(years.record_id == "PATTERN|LONG_REPLAY_DIAGNOSTIC") & years.full_calendar_year]
    assert long.natural_complete_cycles.tolist() == [7, 5, 6, 6, 10, 7]
    assert not years.loc[years.year == 2026, "full_calendar_year"].any()
    assert not result.loc[result.family == "SAVED_14_REFERENCE", "frequency_pass"].any()
    summary = load_json(OUT / "result.json")
    assert summary["working_new_requirement_pass_records"] == result.loc[result.new_requirements_point_screen, "record_id"].tolist()
    verdict = {"status": "PASS_SAVED_CYCLES_ANNUAL_COUNTS_AND_THREE_ACCOUNT_RECOMPUTATIONS",
               "saved_metric_checks": len(checks), "annual_rows": len(years), "cycles": len(trades),
               "zero_years_preserved": True, "partial_2026_not_classified_as_full": True,
               "new_accounts": 0, "authority_snapshot_only": True}
    write_json(OUT / "verification.json", verdict)
    return verdict


def main():
    parser = argparse.ArgumentParser(description="年度五笔及胜率或盈亏比的新要求复核。")
    parser.add_argument("command", choices=["run", "complete", "verify"])
    args = parser.parse_args()
    if args.command in ["run", "complete"]:
        if args.command == "run":
            snapshot()
        else:
            assert (OUT / "code_revision.json").exists(), "补完计算需要明确的实现修正记录。"
            assert not (OUT / "result.json").exists(), "已有结果不重写。"
        result, years, trades, checks = compute()
        save(result, years, trades, checks)
        update_authority()
        report(result, years)
        print(json.dumps(verify(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(verify(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
