"""运行一次固定的本地价格代理研究，先保存规则与来源哈希，再读取新结果。"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from research.volatility_conditioned_intraday_proxy_v1 import (
    block_mean_interval, build_features, detect_episodes, load_config, metrics, simulate_account,
)

CONFIG = ROOT / "config/510300_volatility_conditioned_intraday_proxy_v1.json"
OUTPUT = ROOT / "reports/research/510300_pressure_recovery_v1/volatility_conditioned_proxy_20261001"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def human_report(summary: dict, result: pd.DataFrame, subperiods: pd.DataFrame, intervals: dict) -> str:
    names = {"SWITCH": "按波动切换", "FIXED_RECOVERY": "始终等待恢复信号", "FIXED_CONTINUATION": "始终等待延续信号", "CASH": "始终现金"}
    primary = result.loc[(result.initial_cash_cny == 200000) & (result.cost == "BASE")]
    lines = [
        "# 510300：按近期波动选择机会，固定首轮价格代理结果", "",
        "**本轮完成的是分钟价格与明确成本假设下的对照，不是已验证成交收益。原M1/M2的成交后净期望与夏普仍为NOT_COMPUTED。**", "",
        "用户允许按近期波动调整策略，也允许长时间等待。本轮只用已有本地数据，没有下载新行情，没有交易或仓位指令。", "",
        "## 一次固定检验", "",
        "开盘前用最近20日连续时段的一分钟价格波动，比较此前252个同口径历史值。当天数据不参与当天状态；隔夜跳空、午休跳空和14:57后的变化不参与波动测量。", "",
        "| 波动历史分位 | 切换政策 |", "|---|---|",
        "| ≤30% | 等待，不新开仓 |", "| >30%且≤70% | 只观察下跌后的恢复 |",
        "| >70%且≤95% | 只观察上涨后的延续 |", "| >95% | 等待，不新开仓 |", "",
        "这张映射表是事先固定的待检验假设，不代表高波动天然适合追涨。低/极端波动只约束新买入；此前已有的假设仓位仍依固定退出合同处理。", "",
        "恢复信号：五分钟下跌至少达到max(5bp，过去20日同刻五分钟收益标准差的2倍)，再固定观察五分钟；价格回升且尚未回到冲击前水平。延续信号：同阈值的五分钟上涨后，固定五分钟观察结束价进一步上涨。连续冲击合并，至少连续10个合格分钟不再触发后才能开启新事件。确认失败和收盘/午休截断均保留。", "",
        "收益从确认后的下一分钟close价格代理开始，下一交易日09:36 close退出。每个政策最多每天买入一次，只选当时首个符合条件的事件，不根据事后收益挑选。全账户现金扣费后按100份取整，不借款。最后一个数据日固定禁止新开仓。", "",
        "这只是价格恢复/延续，未使用合格篮子、IOPV、价差或订单流，不能改名为原M1或M2。", "",
        "## 全账户结果", "",
        f"原始分钟资料：{summary['source_rows']:,}条，{summary['source_days']:,}个交易日，{summary['source_start']}至{summary['source_end']}。共同计算区间为{summary['evaluation_start']}至{summary['evaluation_end']}，共{summary['evaluation_days']:,}天。历史已经接触过，不称为独立留出样本。", "",
        "基准费用：每边佣金万二、最低5元，另每边5bp价格摩擦；压力费用：万四、最低5元，另每边10bp。现金利息假设为零。价格摩擦不等于通过验证的滑点模型。下表为20万元基准费用情景；夏普使用所有交易日（包括等待日）和242日年化，回撤为日终净值口径。", "",
        "| 固定政策 | 假设交易笔数 | 无新买入日 | 全天无买卖的现金日 | 净年化收益 | 净夏普 | 最大日终回撤 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in primary.iterrows():
        sharpe = "不适用" if pd.isna(row.sharpe) else f"{row.sharpe:.3f}"
        lines.append(f"| {names[row.policy]} | {int(row.trade_count)} | {int(row.no_entry_days)} | {int(row.no_order_cash_days)} | {row.cagr:.2%} | {sharpe} | {row.maximum_drawdown:.2%} |")
    lines += ["", "‘无新买入日’可能仍有前一日买入份额在早盘退出；‘全天无买卖的现金日’才是该模型全天未持有ETF的日子。现金等待不是缺失数据。初始20万元和2万元、两套费用、四个政策共16个情景均完整保留，没有按结果筛选。", "", "| 切换政策情景 | 净年化收益 | 净夏普 | 日终回撤 | 单笔净收益均值 |", "|---|---:|---:|---:|---:|"]
    for _, row in result.loc[result.policy == "SWITCH"].iterrows():
        sharpe = "不适用" if pd.isna(row.sharpe) else f"{row.sharpe:.3f}"
        lines.append(f"| {int(row.initial_cash_cny):,}元 / {row.cost} | {row.cagr:.2%} | {sharpe} | {row.maximum_drawdown:.2%} | {row.mean_net_trade_bps:.2f}bp |")
    lines += ["", "## 时间稳定性与不确定性", "", "固定区间按日终账户收益拆分；跨年/跨段仓位会在相邻区间分担损益，交易笔数按退出日期统计，不将区间表现重新优化成选择规则。", "", "| 期间 | 切换政策年化 | 固定恢复年化 | 固定延续年化 |", "|---|---:|---:|---:|"]
    for period in subperiods.period.unique():
        group = subperiods.loc[subperiods.period == period].set_index("policy")
        lines.append(f"| {period} | {group.at['SWITCH', 'cagr']:.2%} | {group.at['FIXED_RECOVERY', 'cagr']:.2%} | {group.at['FIXED_CONTINUATION', 'cagr']:.2%} |")
    lines += ["", "20日循环分块、2,000次重抽样，只描述同一历史路径上的不确定性；不作为独立验证，也不声称校正了整个项目此前尝试的多重选择。以下均为每交易日平均收益的基点区间。", "", "| 对象 | 日均收益/差值 | 描述性95%区间 |", "|---|---:|---:|"]
    for name, interval in intervals.items():
        lines.append(f"| {name} | {interval['mean_daily_bps']:.3f}bp | [{interval['lower_95_daily_bps']:.3f}, {interval['upper_95_daily_bps']:.3f}]bp |")
    lines += ["", "## 结果解释边界", "",
        "1. 仅当日历和全部分钟网格完整、历史波动可计算时进入共同样本；不使用9月缺失的近期分钟历史填造9月波动状态。本轮亦不拿三个L2样本充当多年盘口。",
        "2. 买入和退出都假定对应分钟价格可全量成交，分钟时间标签边界、排队、成交选择、涨跌停和实际佣金未独立验证；0笔实际订单。正收益（如有）也不能证明实际可获得。",
        "3. 分红仅用于会计：登记日收盘持仓确定权益，除息日应收、发放日现金，不用事后分红信息构造入场。已存分红完整性记录覆盖本次历史区间。",
        "4. 原研究协议、先前三日17个价格压力事件、旧波动率路由拒绝结论均保留。本轮是用户新提出的价格策略对照，不复活旧失败。",
        "5. 全账户满现金可买数量是统一的研究比较口径，不是建议用户满仓。日终回撤也不等于盘中或实盘最大风险。", "",
        "本轮到固定对照为止。不能因为一个区间较好就改用那个波动档，不能换参数、反向交易或改变退出挽救结果；原盈利目标尚未完成。", "",
        "## 文件", "",
        "- `registration_receipt.json`：计算前登记的规则、程序和输入哈希。",
        "- `01_每日波动状态.csv`：全历史每日状态，包含预热缺失。",
        "- `02_全部固定冲击事件.parquet`：确认与未确认事件全表。",
        "- `03_全账户每日账本.parquet`：全部情景的现金、持仓、分红应收、日收益。",
        "- `04_假设跨日交易.csv`：全部情景的入场、次日退出、损益及费用分解。",
        "- `05_全部情景结果.csv`、`06_固定期间结果.csv`、`07_逐年结果.csv`：共同区间与稳定性。",
        "- `08_事件按波动分组.csv`：候选数量和确认情况，不按未来收益重新分组。",
        "- `bootstrap_intervals.json`、`verification.json`、`summary.json`：不确定性、必要账本核对、明确状态。",
        "- `波动切换_假设账户.png`：净值与交易频率图。", "",
    ]
    return "\n".join(lines)


def plot_result(daily: pd.DataFrame, states: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    plt.rcParams["font.family"] = font.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    names = {"SWITCH": "按波动切换", "FIXED_RECOVERY": "固定恢复", "FIXED_CONTINUATION": "固定延续", "CASH": "现金"}
    colors = {"SWITCH": "#146c94", "FIXED_RECOVERY": "#d18b2c", "FIXED_CONTINUATION": "#b54b5b", "CASH": "#83918b"}
    primary = daily.loc[(daily.initial_cash_cny == 200000) & (daily.cost == "BASE")]
    figure, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1]})
    for policy, group in primary.groupby("policy", sort=False):
        axes[0].plot(pd.to_datetime(group.date), group.equity_cny / 200000, label=names[policy], color=colors[policy], linewidth=1.2)
    axes[0].set_ylabel("假设净值 / 初始净值")
    axes[0].legend(ncol=4, loc="upper left", frameon=False)
    axes[0].set_title("510300：按波动选择机会的固定对照\n20万元、每边万二佣金（最低5元）及5bp摩擦；全量成交未经验证", loc="left", fontsize=14)
    selected = primary.loc[primary.policy == "SWITCH"].copy()
    selected["month"] = pd.to_datetime(selected.date).dt.to_period("M").dt.to_timestamp()
    monthly = selected.assign(entry=(selected.bought_quantity > 0).astype(int)).groupby("month").entry.sum()
    axes[1].bar(monthly.index, monthly.values, width=24, color=colors["SWITCH"])
    axes[1].set_ylabel("切换政策\n每月新买入天数")
    visible = states.loc[states.date >= selected.date.iloc[0]]
    axes[2].plot(pd.to_datetime(visible.date), visible.prior_rank, color="#7267a3", linewidth=.8)
    for rank in (.3, .7, .95):
        axes[2].axhline(rank, color="#aaaaaa", linestyle="--", linewidth=.6)
    axes[2].set_ylabel("开盘前已知\n波动历史分位")
    axes[2].set_ylim(0, 1.03)
    for axis in axes:
        axis.grid(axis="y", alpha=.18)
        axis.spines[["top", "right"]].set_visible(False)
    figure.text(.07, .013, "本地分钟价格代理 · 等待日全部保留 · T+1次日09:36退出 · 不代表原M1/M2收益或实盘业绩", fontsize=10, color="#555555")
    figure.tight_layout(rect=[0, .035, 1, 1])
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise SystemExit("结果目录已经存在且非空，拒绝覆盖固定首轮；复核应读取保存账本。")
    config = load_config(CONFIG)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    inputs = [CONFIG, Path(__file__).resolve(), ROOT / "research/volatility_conditioned_intraday_proxy_v1.py"]
    inputs += [ROOT / config[key] for key in ("minute_file", "calendar_file", "dividend_file", "dividend_coverage_file")]
    inputs += [ROOT / "config/510300_pressure_recovery_v1.json", ROOT / "research/pressure_recovery_v1.py",
               ROOT / "config/510300_volatility_expert_router_v1.json",
               ROOT / "reports/research/510300_volatility_expert_router_v1/acceptance_outcome.json"]
    receipt = {"registered_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "study_id": config["study_id"],
               "saved_before_result_computation": True, "protocol": config,
               "files": [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in inputs]}
    save_json(OUTPUT / "registration_receipt.json", receipt)
    print("已先登记固定规则与来源，开始本地计算。", flush=True)
    bars = pd.read_parquet(ROOT / config["minute_file"])
    if set(bars.symbol.astype(str)) != {"510300"} or set(bars.exchange) != {"SH"}:
        raise ValueError("证券身份不匹配。")
    dates = sorted(pd.to_datetime(bars.timestamp).dt.strftime("%Y-%m-%d").unique())
    calendar = pd.read_csv(ROOT / config["calendar_file"])
    cal_dates = set(calendar.loc[calendar.is_trading_day.astype(str).str.lower().eq("true"), "date"])
    expected = {d for d in cal_dates if dates[0] <= d <= dates[-1]}
    if set(dates) != expected:
        raise ValueError("分钟交易日与本地日历不符，停止计算。")
    coverage = load_config(ROOT / config["dividend_coverage_file"])
    if not coverage["complete_history_confirmed"] or coverage["coverage_end"] < dates[-1] or coverage["distribution_file_sha256"] != sha256(ROOT / config["dividend_file"]):
        raise ValueError("分红会计覆盖不足或账本哈希不一致。")
    dividends = pd.read_csv(ROOT / config["dividend_file"])
    states, prices, threshold = build_features(bars, config)
    episodes = detect_episodes(states, prices, threshold, config)
    print(f"状态与事件计算完成：{len(states):,}日，{len(episodes):,}个合并事件。", flush=True)
    all_daily, all_trades, results, periods, years = [], [], [], [], []
    primary_returns = {}
    for initial in config["hypothetical_account"]["initial_cash_cny"]:
        for cost in config["cost_scenarios"]:
            for policy in config["policies"]:
                daily, trades = simulate_account(states, prices, episodes, dividends, config, policy, initial, cost)
                all_daily.append(daily)
                if not trades.empty:
                    trades = trades.assign(policy=policy, cost=cost, initial_cash_cny=initial)
                    all_trades.append(trades)
                results.append({"policy": policy, "cost": cost, **metrics(daily, trades, config["annual_trading_days"], initial)})
                if initial == 200000 and cost == "BASE":
                    primary_returns[policy] = daily.daily_return.to_numpy()
                    for start, end in config["reporting"]["subperiods"]:
                        part = daily.loc[daily.date.between(start, end)]
                        selected = trades.loc[trades.exit_date.between(start, end)] if not trades.empty else trades
                        if not part.empty:
                            periods.append({"policy": policy, "period": start[:4] + "—" + end[:4], **metrics(part, selected, config["annual_trading_days"])})
                    for year, part in daily.groupby(daily.date.str[:4]):
                        selected = trades.loc[trades.exit_date.str.startswith(year)] if not trades.empty else trades
                        years.append({"policy": policy, "year": year, **metrics(part, selected, config["annual_trading_days"])})
    daily = pd.concat(all_daily, ignore_index=True)
    trades = pd.concat(all_trades, ignore_index=True)
    results = pd.DataFrame(results)
    periods = pd.DataFrame(periods)
    report = config["reporting"]
    interval_values = {"按波动切换": primary_returns["SWITCH"],
                       "切换减固定恢复": primary_returns["SWITCH"] - primary_returns["FIXED_RECOVERY"],
                       "切换减固定延续": primary_returns["SWITCH"] - primary_returns["FIXED_CONTINUATION"]}
    intervals = {name: block_mean_interval(values, report["block_bootstrap_days"], report["block_bootstrap_draws"], report["random_seed"]) for name, values in interval_values.items()}
    qualified = states.loc[states.regime != "WARMUP_NO_VIEW"]
    volume = bars.assign(date=pd.to_datetime(bars.timestamp).dt.strftime("%Y-%m-%d"), clock=pd.to_datetime(bars.timestamp).dt.strftime("%H:%M")).set_index(["date", "clock"]).volume
    zero_volume_entries = int(sum(volume.loc[(row.entry_date, row.entry_clock)] <= 0 for row in trades.itertuples()))
    summary = {"study_id": config["study_id"], "status": "COMPLETED_FIXED_PRICE_PROXY_ONLY", "source_rows": len(bars),
               "source_days": len(dates), "source_start": dates[0], "source_end": dates[-1],
               "evaluation_start": qualified.date.iloc[0], "evaluation_end": qualified.date.iloc[-1], "evaluation_days": len(qualified),
               "regime_day_counts": qualified.regime.value_counts().to_dict(), "episode_count": len(episodes),
               "episode_status_counts": episodes.status.value_counts().to_dict(), "scenario_count": len(results),
               "zero_volume_entry_proxy_rows_across_scenarios": zero_volume_entries,
               "original_m1_m2_expectancy": "NOT_COMPUTED", "original_m1_m2_sharpe": "NOT_COMPUTED",
               "independent_validation": "NOT_ESTABLISHED", "verified_fills": 0, "actual_orders": 0,
               "new_network_requests": 0, "goal_achieved": False, "position_impact": 0}
    checks = []
    def check(name: str, condition: bool) -> None:
        checks.append({"check": name, "passed": bool(condition)})
        if not condition:
            raise AssertionError(name)
    check("每日账户恒等式", np.allclose(daily.equity_cny, daily.cash_cny + daily.receivable_cny + daily.market_value_cny, atol=1e-7, rtol=1e-10))
    check("无借款且数量为整手", bool((daily.cash_cny >= -1e-7).all() and (trades.quantity % 100 == 0).all()))
    check("全部状态最后输入早于当日", bool((qualified.last_feature_date < qualified.date).all()))
    next_dates = dict(zip(dates[:-1], dates[1:]))
    check("每笔严格下一交易日退出", bool((trades.entry_date.map(next_dates) == trades.exit_date).all()))
    check("确认后下一分钟入场", bool((pd.to_datetime(trades.entry_date + " " + trades.entry_clock) > pd.to_datetime(trades.event_id.str[:10] + " " + trades.event_id.str[11:13] + ":" + trades.event_id.str[13:15]) + pd.Timedelta(minutes=5)).all()))
    check("每笔损益等于毛收益减费用摩擦", np.allclose(trades.net_pnl_cny, trades.gross_pnl_cny - trades.friction_cny - trades.total_fee_cny, atol=1e-7))
    for (policy, cost, initial), group in daily.groupby(["policy", "cost", "initial_cash_cny"]):
        selected = trades.loc[(trades.policy == policy) & (trades.cost == cost) & (trades.initial_cash_cny == initial)]
        check(f"{policy}/{cost}/{initial}现金日齐全及损益闭合", len(group) == len(qualified) and group.date.nunique() == len(qualified) and math_isclose(group.equity_cny.iloc[-1] - initial, selected.net_pnl_cny.sum()) and group.end_quantity.iloc[-1] == 0)
    check("输入与旧研究未改动", all(sha256(ROOT / item["path"]) == item["sha256"] for item in receipt["files"]))
    states.to_csv(OUTPUT / "01_每日波动状态.csv", index=False, encoding="utf-8-sig")
    episodes.to_parquet(OUTPUT / "02_全部固定冲击事件.parquet", index=False)
    daily.to_parquet(OUTPUT / "03_全账户每日账本.parquet", index=False)
    trades.to_csv(OUTPUT / "04_假设跨日交易.csv", index=False, encoding="utf-8-sig")
    results.to_csv(OUTPUT / "05_全部情景结果.csv", index=False, encoding="utf-8-sig")
    periods.to_csv(OUTPUT / "06_固定期间结果.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(years).to_csv(OUTPUT / "07_逐年结果.csv", index=False, encoding="utf-8-sig")
    episodes.groupby(["regime", "direction", "status"]).size().rename("events").reset_index().to_csv(OUTPUT / "08_事件按波动分组.csv", index=False, encoding="utf-8-sig")
    save_json(OUTPUT / "bootstrap_intervals.json", intervals)
    save_json(OUTPUT / "summary.json", summary)
    save_json(OUTPUT / "verification.json", {"status": "PASS_ACCOUNT_IDENTITIES_AND_CAUSAL_INDEXING", "checks": checks, "not_execution_validation": True})
    (OUTPUT / "00_波动切换首轮结论.md").write_text(human_report(summary, results, periods, intervals), encoding="utf-8")
    plot_result(daily, states, OUTPUT / "波动切换_假设账户.png")
    index = [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(OUTPUT.iterdir()) if p.is_file()]
    pd.DataFrame(index).to_csv(OUTPUT / "FILE_INDEX.csv", index=False, encoding="utf-8-sig")
    total = sum(p.stat().st_size for p in OUTPUT.iterdir() if p.is_file())
    if total > report["new_result_storage_limit_bytes"]:
        raise ValueError("新增研究结果超过事先固定的20MiB空间预算。")
    print(json.dumps({"完成": summary, "新增文件字节": total,
                      "主账户基准情景": results.loc[(results.initial_cash_cny == 200000) & (results.cost == "BASE"), ["policy", "trade_count", "cagr", "sharpe", "maximum_drawdown", "no_entry_days"]].replace({np.nan: None}).to_dict("records")}, ensure_ascii=False), flush=True)


def math_isclose(left: float, right: float) -> bool:
    return abs(left - right) < 1e-6 * max(1.0, abs(left), abs(right))


if __name__ == "__main__":
    main()
