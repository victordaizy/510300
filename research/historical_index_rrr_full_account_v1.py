"""把既有六次降准事件接入连续账户，固定规则，不新增信号或参数搜索。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import factor96_bottleneck_diagnostic_v1 as account_diagnostic
from research import factor96_margin_repair_v1 as core
from research import factor96_rapid_feasibility_v1 as rapid

OUT = ROOT / "reports/research/510300_historical_index_rrr_full_account_v1"
PRIOR = ROOT / "reports/research/510300_historical_index_liquidity_transmission_v1"
POLICY = "rrr_announcement_available_before_open"
REPORT_NAME = "历史发现_六次降准的连续账户与收益瓶颈.md"
NAMES = {
    "RRR_200000_STRESS": "降准公告，20万元主账户",
    "RRR_20000_STRESS": "降准公告，2万元参考账户",
    "BUY_HOLD50_200000_REFERENCE": "期初半仓买入持有参考",
    "CASH_200000_REFERENCE": "全现金参考",
}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def equity_metrics(equity, capital: float) -> dict:
    nav = np.r_[capital, np.asarray(equity, dtype=float)]
    returns = nav[1:] / nav[:-1] - 1
    volatility = returns.std(ddof=1)
    return {
        "days": len(returns),
        "end_equity": nav[-1],
        "net_profit": nav[-1] - capital,
        "net_sharpe": np.sqrt(242) * returns.mean() / volatility if volatility > 1e-15 else None,
        "cagr": (nav[-1] / capital) ** (242 / len(returns)) - 1,
        "max_drawdown": 1 - np.min(nav / np.maximum.accumulate(nav)),
    }


def load_inputs(protocol: dict):
    market = pd.read_parquet(rapid.PRICE_BASE / "market.parquet")
    market["date"] = pd.to_datetime(market.date)
    market = market.loc[market.date.le(protocol["account_calendar"][1])].reset_index(drop=True)
    assert market.date.is_monotonic_increasing and not market.date.duplicated().any()
    assert market.symbol.eq("510300.SH").all()
    raw_features = pd.read_parquet(rapid.PRICE_BASE / "price_features.parquet")
    raw_features["date"] = pd.to_datetime(raw_features.date)
    features = market[["date"]].merge(raw_features[["date", "es95"]], on="date", how="left", validate="one_to_one")
    dividends = core.normalize_dividends(pd.read_csv(rapid.PRICE_BASE / "dividends.csv"))
    expected = [row for row in read(PRIOR / "event_returns_and_breadth.json") if row["horizon"] == 20]
    assert {row["event_id"] for row in expected} == {row["id"] for row in protocol["events"]}
    assert len(expected) == len(protocol["events"]) == 6
    lookup = {row["event_id"]: row for row in expected}
    features[POLICY] = False
    clocks = []
    for event in protocol["events"]:
        source = lookup[event["id"]]
        announcement_day = pd.Timestamp(event["date"])
        entry_idx = int(np.flatnonzero(market.date.gt(announcement_day))[0])
        exit_idx = entry_idx + protocol["primary_holding_open_intervals"]
        entry_day, exit_day = market.date.iloc[entry_idx], market.date.iloc[exit_idx]
        assert entry_day == pd.Timestamp(source["entry_date"])
        assert exit_day == pd.Timestamp(source["exit_date"])
        entry_open, exit_open = float(market.open.iloc[entry_idx]), float(market.open.iloc[exit_idx])
        assert abs(entry_open - source["entry_open"]) < 1e-10
        assert abs(exit_open - source["exit_open"]) < 1e-10
        dividend_per_share = dividends.loc[
            dividends.record_date.ge(entry_day) & dividends.record_date.lt(exit_day),
            "cash_dividend_per_share",
        ].sum()
        assert abs(dividend_per_share - source["dividend_per_share"]) < 1e-10
        gross_return = (exit_open + dividend_per_share) / entry_open - 1
        assert abs(gross_return - source["gross_open_to_open_return"]) < 1e-12
        known_at = (announcement_day + pd.Timedelta(hours=23, minutes=59, seconds=59)).tz_localize("Asia/Shanghai")
        entry_at = (entry_day + pd.Timedelta(hours=9, minutes=30)).tz_localize("Asia/Shanghai")
        assert known_at < entry_at
        assert np.isfinite(features.es95.iloc[entry_idx - 1])
        # 引擎读取前一格信号；该格仅存储开盘委托标志，绝不代表公告已在前收盘公开。
        features.loc[entry_idx - 1, POLICY] = True
        clocks.append({
            "event_id": event["id"], "announcement_date": event["date"],
            "announcement_known_at_upper_bound": known_at.isoformat(),
            "planned_entry_at": entry_at.isoformat(), "entry_date": entry_day,
            "planned_exit_date": exit_day, "entry_idx": entry_idx, "exit_idx": exit_idx,
            "price_and_es_observation_date": market.date.iloc[entry_idx - 1],
            "entry_es95": float(features.es95.iloc[entry_idx - 1]),
            "entry_open": entry_open, "planned_exit_open": exit_open,
            "dividend_per_share_for_original_fixed_position": dividend_per_share,
            "old_event_net_return_on_invested_amount": source["net_return"],
            "old_event_gross_return_on_invested_amount": gross_return,
            "old_price_and_dividend_reproduction": True,
        })
    spec = importlib.util.spec_from_file_location("rrr_historical_frozen_account_engine", rapid.ENGINE_PATH)
    engine = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = engine
    spec.loader.exec_module(engine)
    return market, features, dividends, clocks, engine


def enrich_ledger(ledger: pd.DataFrame, market: pd.DataFrame, features: pd.DataFrame, clocks: list, candidate: bool):
    ledger = ledger.copy()
    by_day = {pd.Timestamp(row["entry_date"]): row for row in clocks}
    entry_days = set(by_day)
    ledger["scheduled_rrr_entry"] = ledger.date.isin(entry_days) if candidate else False
    ledger["event_id"] = [by_day[day]["event_id"] if candidate and day in by_day else "" for day in ledger.date]
    ledger["announcement_known_at_upper_bound"] = [
        by_day[day]["announcement_known_at_upper_bound"] if candidate and day in by_day else "" for day in ledger.date
    ]
    ledger["es_observation_date"] = [market.date.iloc[int(i) - 1] for i in ledger.idx]
    ledger["es95_used_before_open"] = [features.es95.iloc[int(i) - 1] for i in ledger.idx]
    ledger["close_exposure_times_prior_es"] = ledger.exposure * ledger.es95_used_before_open
    ledger["reason"] = ledger.reason.replace({"前收盘信号入场": "公告已公开后统一开盘入场" if candidate else "期初半仓参考入场"})
    ledger["cumulative_paid_friction"] = (ledger.commission + ledger.slippage_cost).cumsum()
    ledger["fixed_holdings_fee_refund_equity"] = ledger.equity + ledger.cumulative_paid_friction + ledger.terminal_exit_reserve
    return ledger


def check_account(ledger: pd.DataFrame, capital: float, clocks: list, candidate: bool) -> dict:
    assert ledger.cash.min() >= -1e-8
    assert ledger.shares.ge(0).all() and ledger.shares.mod(100).eq(0).all()
    reconstructed = ledger.cash + ledger.shares * ledger.mark + ledger.dividend_receivable - ledger.terminal_exit_reserve
    np.testing.assert_allclose(ledger.equity, reconstructed, rtol=0, atol=1e-7)
    if candidate:
        buys = ledger.loc[ledger.filled_quantity.gt(0)]
        assert buys.shares_before.eq(0).all()
        assert set(buys.date).issubset({pd.Timestamp(r["entry_date"]) for r in clocks})
    lots = []
    for row in ledger.itertuples(index=False):
        quantity = int(row.filled_quantity)
        if quantity > 0:
            lots.append([row.date, quantity])
        elif quantity < 0:
            to_sell = -quantity
            assert sum(q for day, q in lots if day < row.date) >= to_sell
            for lot in lots:
                if lot[0] < row.date:
                    sold = min(lot[1], to_sell)
                    lot[1] -= sold
                    to_sell -= sold
                if not to_sell:
                    break
            lots = [lot for lot in lots if lot[1] > 0]
        assert sum(q for _, q in lots) == row.shares
    statistics = equity_metrics(ledger.equity, capital)
    official = core.metrics(ledger, capital)
    for key in ["end_equity", "net_profit", "net_sharpe", "cagr", "max_drawdown"]:
        if statistics[key] is None:
            assert official[key] is None
        else:
            assert abs(statistics[key] - official[key]) < 1e-12
    return {"cash_and_positions_valid": True, "T_plus_1_valid": True,
            "equity_identity_max_abs_error": float((ledger.equity - reconstructed).abs().max()),
            "daily_pnl_identity_max_abs_error": float(ledger.accounting_error.abs().max()),
            "metrics_recomputed_from_equity": True,
            "buy_dates_subset_of_frozen_six_entries": True if candidate else None}


def event_attribution(ledger: pd.DataFrame, capital: float, clocks: list) -> list:
    cycles = core.cycle_records(ledger, capital)
    by_entry = {pd.Timestamp(row.entry): row for row in cycles.itertuples(index=False)}
    events = []
    for clock in clocks:
        day = pd.Timestamp(clock["entry_date"])
        cycle = by_entry.get(day)
        first = ledger.loc[ledger.date.eq(day)].iloc[0]
        item = dict(clock)
        item.update({"filled_entry_shares": int(first.filled_quantity), "entry_order_status": first.status})
        if cycle is None:
            item.update({"executed": False, "full_account_cycle_profit": 0., "skip_reason": first.reason})
        else:
            assert cycle.closed, "六次事件均应在本轮结束前完成；不能截去未结束持仓。"
            part = ledger.loc[ledger.date.between(day, cycle.exit)]
            item.update({
                "executed": True, "actual_exit_date": cycle.exit, "cycle_start_equity": cycle.start_equity,
                "full_account_cycle_profit": cycle.profit,
                "full_account_cycle_return": cycle.profit / cycle.start_equity,
                "fixed_holdings_gross_profit": cycle.profit + part.commission.sum() + part.slippage_cost.sum(),
                "commission": part.commission.sum(), "slippage": part.slippage_cost.sum(),
                "dividend_recognized": part.dividend_recognized.sum(), "dividend_paid": part.dividend_paid.sum(),
                "risk_reduction_fills": int((part.reason.eq("风险预算减仓") & part.filled_quantity.lt(0)).sum()),
                "mean_close_exposure": part.exposure.mean(), "max_close_exposure": part.exposure.max(),
                "actual_open_intervals": int(part.idx.iloc[-1] - part.idx.iloc[0]),
            })
        events.append(item)
    assert abs(sum(row["full_account_cycle_profit"] for row in events) - (ledger.equity.iloc[-1] - capital)) < 1e-6
    return events


def run_accounts(protocol, market, features, dividends, clocks, engine):
    all_metrics, all_checks, attributions, ledgers, refunds = {}, {}, {}, {}, {}
    calendar = market.date.between(*protocol["account_calendar"])
    first_idx = int(np.flatnonzero(calendar)[0])
    for case_id in protocol["account_scenarios"]:
        candidate = case_id.startswith("RRR_")
        benchmark = case_id.startswith("BUY_HOLD")
        capital = protocol["reference_capital"] if case_id == "RRR_20000_STRESS" else protocol["primary_capital"]
        feature_case = features.copy()
        if not candidate:
            feature_case[POLICY] = False
            if benchmark:
                feature_case.loc[first_idx - 1, POLICY] = True
        case = {"case_id": case_id, "capital": capital, "policy": POLICY,
                "hold": protocol["primary_holding_open_intervals"] if candidate else len(market) + 1,
                "start": protocol["account_calendar"][0], "end": protocol["account_calendar"][1]}
        ledger = account_diagnostic.simulate(market, feature_case, dividends, case,
                                            "CAP50_ONLY" if benchmark else "ORIGINAL", "STRESS", engine)
        assert ledger.date.reset_index(drop=True).equals(market.loc[calendar, "date"].reset_index(drop=True))
        ledger = enrich_ledger(ledger, market, features, clocks, candidate)
        metrics = core.metrics(ledger, capital)
        metrics.pop("sharpe_252_diagnostic")
        metrics.pop("cagr_252_diagnostic")
        metrics.update({
            "capital": capital, "role": "候选固定规则诊断" if candidate else "市场暴露或现金参考",
            "first_date": ledger.date.iloc[0], "last_date": ledger.date.iloc[-1],
            "holding_close_days": int(ledger.shares.gt(0).sum()),
            "cash_only_close_days": int(ledger.shares.eq(0).sum()),
            "no_position_all_session_days": int((ledger.shares.eq(0) & ledger.shares_before.eq(0)).sum()),
            "max_close_exposure": ledger.exposure.max(),
            "close_exposure_above_50pct_days": int(ledger.exposure.gt(.5 + 1e-12).sum()),
            "max_close_exposure_times_prior_es": ledger.close_exposure_times_prior_es.max(),
            "estimated_close_ES_above_budget_days": int(ledger.close_exposure_times_prior_es.gt(.025).sum()),
            "drawdown_stop_triggered": bool(ledger.risk_stopped.any()),
            "risk_reduction_fills": int((ledger.reason.eq("风险预算减仓") & ledger.filled_quantity.lt(0)).sum()),
            "entry_fills": int(ledger.filled_quantity.gt(0).sum()),
            "rejected_nonzero_orders": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum()),
            "dividends_recognized": ledger.dividend_recognized.sum(),
            "calendar_years": {},
            "passes_sample_sharpe_cagr_drawdown_targets": bool(candidate and metrics["net_sharpe"] is not None
                and metrics["net_sharpe"] >= 1.2 and metrics["cagr"] >= .1 and metrics["max_drawdown"] <= .1),
        })
        for year in [2018, 2019]:
            part = ledger.loc[ledger.date.dt.year.eq(year)]
            before = capital if part.index[0] == 0 else float(ledger.equity.iloc[part.index[0] - 1])
            metrics["calendar_years"][str(year)] = equity_metrics(part.equity, before)
        if candidate:
            attributions[case_id] = event_attribution(ledger, capital, clocks)
            refunds[case_id] = {
                "method": "同一持仓路径逐日返还累计已付佣金和滑点，并移除期末退出预留；不再投资，不改变份额。仅为费用解释。",
                "not_an_executable_zero_cost_strategy": True,
                "refunded_friction_cny": float(ledger.cumulative_paid_friction.iloc[-1] + ledger.terminal_exit_reserve.iloc[-1]),
                "fixed_holdings_refund_metrics": equity_metrics(ledger.fixed_holdings_fee_refund_equity, capital),
            }
        path = OUT / "ledgers" / f"{case_id}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        ledger.to_parquet(path, index=False)
        all_checks[case_id] = check_account(pd.read_parquet(path), capital, clocks, candidate)
        all_checks[case_id]["verified_from_saved_ledger"] = True
        all_metrics[case_id], ledgers[case_id] = metrics, ledger
        print(f"已完成：{NAMES[case_id]}，{len(ledger)}个交易日，净利润{metrics['net_profit']:.2f}元")
    return all_metrics, all_checks, attributions, ledgers, refunds


def plot_accounts(ledgers: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    palette = {"RRR_200000_STRESS": "#135c73", "RRR_20000_STRESS": "#bd7041",
               "BUY_HOLD50_200000_REFERENCE": "#969eaa", "CASH_200000_REFERENCE": "#c2c7cb"}
    for key, ledger in ledgers.items():
        capital = 20000 if key == "RRR_20000_STRESS" else 200000
        axes[0].plot(ledger.date, ledger.equity / capital - 1, label=NAMES[key], color=palette[key], linewidth=1.7)
    primary = ledgers["RRR_200000_STRESS"]
    axes[1].fill_between(primary.date, primary.exposure, color=palette["RRR_200000_STRESS"], alpha=.6)
    axes[0].set_title("六次降准公告接入连续账户｜2018—2019年全部交易日", loc="left", fontsize=15)
    axes[0].set_ylabel("累计净收益")
    axes[1].set_ylabel("主账户收盘仓位")
    axes[0].legend(loc="lower left", frameon=False, ncol=2)
    for axis in axes:
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(axis="y", alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(.09, .015, "统一20个开盘间隔；计入空仓期、佣金、滑点和分红。半仓持有仅为参考；历史发现不等于独立验证。", fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, 1))
    fig.savefig(OUT / "六次降准_完整账户与仓位.png", dpi=170)
    plt.close(fig)


def pct(value):
    return f"{value:.2%}"


def sharpe(value):
    return "不适用（零波动）" if value is None else f"{value:.3f}"


def write_report(result: dict, attribution: dict, refunds: dict):
    metrics = result["accounts"]
    main = metrics["RRR_200000_STRESS"]
    gross = refunds["RRR_200000_STRESS"]["fixed_holdings_refund_metrics"]
    events = attribution["RRR_200000_STRESS"]
    lines = [
        "# 历史发现：六次降准的连续账户与收益瓶颈", "",
        f"**固定规则未达到目标。** 2018—2019年，20万元主账户成本后夏普为{sharpe(main['net_sharpe'])}，年化收益{pct(main['cagr'])}，最大回撤{pct(main['max_drawdown'])}。",
        f"返还相同持仓的全部交易费用后，夏普为{sharpe(gross['net_sharpe'])}，年化收益{pct(gross['cagr'])}。费用结果是解释性核算，不是另一个可执行策略。", "",
        "这次新增的是六次事件共用本金的完整账户。六次降准的用途、资金替换和固定窗口收益此前已研究，本轮不把旧机制拆解重复计作发现，也不根据已知涨跌挑选事件。", "",
        "## 固定问题与口径", "",
        "研究单位为沪深300整体，以510300实现。全纳入2018年4月17日、6月24日、10月7日及2019年1月4日、5月6日、9月6日六次公告。实施日不重复交易。公告日结束后的下一ETF开盘尝试入场，统一持有20个开盘间隔；风控可提前减仓，禁止加仓或延长持有期。", "",
        f"完整日历为{pd.Timestamp(main['first_date']).date()}至{pd.Timestamp(main['last_date']).date()}，共{main['days']}个交易日；收盘持仓{main['holding_close_days']}天，收盘空仓{main['cash_only_close_days']}天。空仓日收益全部纳入夏普和年化。每年按242个交易日，现金与无风险收益按零。", "",
        "佣金每边万四、每笔至少5元，滑点每边千一，报价最小单位0.001元，100份整手。复用既有T+1、涨跌停、现金约束及分红登记/应收/支付处理。历史开盘成交模型不等于真实成交回执。", "",
        "50%为每次目标决策的仓位上限；条件五日ES95预算2.5%，单次−10%冲击预算5%，并限制其使用回撤剩余空间。收盘回撤达10%后，在下一个可卖开盘退出并停止本轮；不能将触发线理解为最大回撤保证。", "",
        "## 连续账户结果", "",
        "|账户|成本后夏普|年化收益|最大回撤|期末权益|平均收盘仓位|", "|---|---:|---:|---:|---:|---:|",
    ]
    for key, row in metrics.items():
        lines.append(f"|{NAMES[key]}|{sharpe(row['net_sharpe'])}|{pct(row['cagr'])}|{pct(row['max_drawdown'])}|{row['end_equity']:,.2f}元|{pct(row['mean_exposure'])}|")
    lines += ["", "半仓买入持有是市场暴露参考，期间仓位随价格漂移，不接受该候选策略的日常风险减仓；期末剩余持仓扣除退出费用预留。不能因其某项指标较好就当作达标策略。", "",
              "## 六次事件全部保留", "",
              "|公告日|入场日|实际退出日|原固定持仓20日净收益|本次主账户该轮盈亏|风险减仓笔数|",
              "|---|---|---|---:|---:|---:|"]
    for row in events:
        exit_date = str(pd.Timestamp(row["actual_exit_date"]).date()) if row["executed"] else "未入场"
        lines.append(f"|{row['announcement_date']}|{pd.Timestamp(row['entry_date']).date()}|{exit_date}|{pct(row['old_event_net_return_on_invested_amount'])}|{row['full_account_cycle_profit']:+,.2f}元|{row.get('risk_reduction_fills', 0)}|")
    lines += ["", "原固定持仓收益的分母是投入资金，本次盈亏属于同一笔20万元连续账户，两者分母和持仓路径不同，不能相加或互相替代。", "",
              "|固定自然年（仅描述）|主账户夏普|年化收益|期间盈亏|", "|---|---:|---:|---:|"]
    for year, row in main["calendar_years"].items():
        lines.append(f"|{year}|{sharpe(row['net_sharpe'])}|{pct(row['cagr'])}|{row['net_profit']:+,.2f}元|")
    lines += ["", "两年切片保留全部六次事件，只说明时间差异，不据此保留好年份、排除坏年份。", "",
              "## 费用、风险与信号的解释", "",
              f"主账户净盈亏{main['net_profit']:+,.2f}元，已付佣金{main['commission']:.2f}元、滑点{main['slippage']:.2f}元；相同份额返还费用后的盈亏{gross['net_profit']:+,.2f}元。该解释保留原仓位、原减仓和空仓天数，不允许费用返还资金再投资。", "",
              f"主账户共有{main['entry_fills']}次入场、{main['risk_reduction_fills']}次风险减仓；非零委托未成交{main['rejected_nonzero_orders']}次，回撤停机{'已触发' if main['drawdown_stop_triggered'] else '未触发'}。最高收盘仓位{pct(main['max_close_exposure'])}，有{main['close_exposure_above_50pct_days']}天收盘超过50%目标线。目标上限按决策时刻执行，收盘记录不能证明盘中持续满足。", "",
              f"收盘仓位乘以前收盘条件ES的最高值为{pct(main['max_close_exposure_times_prior_es'])}，超过2.5%的收盘日有{main['estimated_close_ES_above_budget_days']}天。这是既有风险估计的暴露诊断，不是实际未来损失或风险保证。", "",
              f"原六次固定持仓20日毛收益的等权平均仅{pct(np.mean([r['old_event_gross_return_on_invested_amount'] for r in events]))}，扣成本后的等权平均为{pct(np.mean([r['old_event_net_return_on_invested_amount'] for r in events]))}，只有两次为正。这是事件窗口的简单均值，不是连续账户收益或年化收益。", "",
              "本轮没有跑取消风控、扩大仓位、换期限或筛选年份的版本，因此不把风控造成的收益变化量冒充已识别的因果效应。原窗口毛收益本已很薄，相同账户持仓返还费用仍不足，不能仅靠降费解释未达标。", "",
              "## 指数机制：为什么降准方向不能直接当作指数方向", "",
              "以下复用既有政策机制卡，仅用于解释，不进入入场筛选。部分数量来自后续央行报告，不能冒充入场时已经知道的精确数字。", ""]
    for card in read(PRIOR / "policy_mechanism_cards.json"):
        lines.append(f"- {card['announcement_date']}：{card['mechanism']} {card['prior_policy_guidance']}")
    lines += ["", "历史结果能支持的范围是：银行负债成本和长期资金约束改善，并不足以单独决定沪深300的可交易方向。指数价格同时反映总需求与盈利、信用传导、要求回报率，以及持有期内的新消息。不能把政策托底、贷款投放和股票净买入视为同一件事，也不能从涨跌结果倒推某次必然超预期。", "",
              "2019年1月上涨，2018年10月及2019年5月下跌，只构成这条机械规则的正反例；事件同时包含海外、贸易和国内需求变化，六次观察不能识别降准本身的独立因果收益。", "",
              "## 必要核对与研究边界", "",
              "已核对六次入场/退出日期、ETF开盘价与分红均复现旧事件表；公告可得时间早于入场，前收盘仅作价格/ES基准。每日现金、份额、T+1、权益恒等式及直接从保存净值重算的收益指标一致。保留4份逐日账本，不增加通用审计流程。", "",
              "六次事件和2018—2019年数据已在此前研究中见过，本轮不是独立验证。没有当前市场判断、前瞻性预测、订单或实盘授权。", "",
              "**终止这一固定表达的参数搜索。** 不通过删除坏事件、选年份、反号、改持有期或扩大仓位挽救‘降准公告直接买入’。后续历史问题必须新增当时可得的信息：信贷供给、实体借款意愿与总需求，究竟哪一环发生变化，变化在公告前是否已经被指数价格反映。只有机械宣布宽松的日期，不足以回答这个问题。", "",
              "## 文件", "",
              "- `protocol.json`：计算前固定的全部六次事件与账户口径。",
              "- `result.json`：4个账户及完整年度切片指标。",
              "- `event_clocks.json`、`event_attribution.json`：信息时间与每次账户盈亏。",
              "- `fee_attribution.json`：同一持仓的费用解释。",
              "- `calculation_checks.json`：必要核算结果。",
              "- `ledgers/*.parquet`：完整日历逐日账本。",
              "- `六次降准_完整账户与仓位.png`：净值和仓位图。",
              "- 机制证据沿用上轮 `510300_historical_index_liquidity_transmission_v1/source_manifest.json`。", ""]
    (OUT / REPORT_NAME).write_text("\n".join(lines), encoding="utf-8")


def main():
    protocol = read(OUT / "protocol.json")
    assert protocol["primary_holding_open_intervals"] == 20
    assert protocol["new_signal_filters"] == 0 and protocol["no_parameter_search"]
    assert protocol["cost"] == {"commission": .0004, "minimum": 5., "slippage": .001, "tick": .001, "lot": 100}
    market, features, dividends, clocks, engine = load_inputs(protocol)
    metrics, checks, attribution, ledgers, refunds = run_accounts(protocol, market, features, dividends, clocks, engine)
    main_stats = metrics["RRR_200000_STRESS"]
    result = {
        "study_id": protocol["study_id"], "completed_at": core.now(),
        "classification": "PROGRESS_FIXED_RRR_ACCOUNT_TARGET_NOT_MET",
        "status": "COMPLETED_HISTORICAL_ACCOUNT_DIAGNOSTIC",
        "question": protocol["question"], "event_count": len(clocks), "full_accounts": len(metrics),
        "accounts": metrics, "parameter_searches": 0, "new_event_filters": 0,
        "original_event_equal_weight_summary": {
            "count": len(clocks),
            "mean_gross_20day_return": np.mean([row["old_event_gross_return_on_invested_amount"] for row in clocks]),
            "mean_net_20day_return": np.mean([row["old_event_net_return_on_invested_amount"] for row in clocks]),
            "positive_net_events": sum(row["old_event_net_return_on_invested_amount"] > 0 for row in clocks),
            "is_full_account_or_annualized_return": False,
        },
        "primary_target_met": main_stats["passes_sample_sharpe_cagr_drawdown_targets"],
        "independently_validated": False, "goal_achieved": False, "orders_authorized": False,
        "rule_disposition": "结束本轮降准公告直接买入的固定表达，不用本轮收益再筛事件、反号或调期限。",
        "next_historical_question": "在既有样本中区分信贷供给、借款需求和总需求的实际变化，以及公告前已计入价格的部分；不能把已知盈利事件拼成筛选条件。",
        "source_files": [
            str((OUT / "protocol.json").relative_to(ROOT)),
            str((PRIOR / "event_returns_and_breadth.json").relative_to(ROOT)),
            str((PRIOR / "policy_mechanism_cards.json").relative_to(ROOT)),
            str((PRIOR / "source_manifest.json").relative_to(ROOT)),
            str((rapid.PRICE_BASE / "market.parquet").relative_to(ROOT)),
            str((rapid.PRICE_BASE / "price_features.parquet").relative_to(ROOT)),
            str((rapid.PRICE_BASE / "dividends.csv").relative_to(ROOT)),
            str(rapid.ENGINE_PATH.relative_to(ROOT)),
            "research/factor96_bottleneck_diagnostic_v1.py",
            "research/factor96_margin_repair_v1.py",
        ],
    }
    if result["primary_target_met"]:
        result["classification"] = "PROGRESS_RRR_ACCOUNT_SAMPLE_TARGET_MET_NOT_INDEPENDENT"
        result["rule_disposition"] = "历史样本指标通过不构成独立验证或实盘授权。"
    core.save(OUT / "event_clocks.json", clocks)
    core.save(OUT / "event_attribution.json", attribution)
    core.save(OUT / "fee_attribution.json", refunds)
    core.save(OUT / "calculation_checks.json", {"six_original_events_reproduced": True, "accounts": checks})
    core.save(OUT / "result.json", result)
    plot_accounts(ledgers)
    write_report(result, attribution, refunds)
    print(f"主账户夏普{sharpe(main_stats['net_sharpe'])}，年化{pct(main_stats['cagr'])}，目标达成为{result['goal_achieved']}")
    print(f"研究报告：{OUT / REPORT_NAME}")


if __name__ == "__main__":
    main()
