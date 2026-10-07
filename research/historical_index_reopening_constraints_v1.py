"""全国防控约束变化的指数历史研究：固定四次修订、信息时序与连续账户。"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import factor96_bottleneck_diagnostic_v1 as account_model
from research import factor96_margin_repair_v1 as core
from research import factor96_rapid_feasibility_v1 as rapid
from research import historical_index_rrr_full_account_v1 as checks

OUT = ROOT / "reports/research/510300_historical_index_reopening_constraints_v1"
REPORT = OUT / "历史发现_活动约束改变与指数剩余收益.md"
SIGNAL = "national_rule_available_before_open"
COST = {"commission": .0004, "minimum": 5., "slippage": .001}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_facts():
    required = {
        "plan9_notice": ["2022-06-28", "2022年6月27日"],
        "plan9_qa_beijing": ["2022-06-28", "7天集中隔离医学观察", "动态清零"],
        "twenty_notice": ["2022-11-11", "5天集中隔离", "不再判定密接的密接"],
        "ten_notice_beijing": ["2022-12-07", "不再对跨地区流动人员", "5天居家隔离"],
        "prior_guidance_mct": ["2022-11-10", "19:42", "二十条措施"],
        "pmi_dec_review": ["2022年12月31日", "39.4", "61.3", "56.3"],
        "pmi_jan_release": ["2023/01/31", "09:30", "54.0", "50.1"],
        "rrr_nov25": ["2022-11-26", "5000亿元", "0.25"],
    }
    manifest = read(OUT / "source_manifest.json")
    lookup = {item["id"]: item for item in manifest}
    for key, tokens in required.items():
        body = (OUT / "sources" / f"{key}.txt").read_text(encoding="utf-8")
        compact = "".join(body.split())
        for token in tokens:
            assert token in compact, (key, token)
        lookup[key]["status"] = "CONTENT_REVIEWED"
    web = [
        {
            "id": "classb_web_fact", "page_date": "2022-12-26",
            "url": "https://www.ndcpa.gov.cn/jbkzzx/c100081/common/content/content_1715174894301736960.html",
            "status": "WEB_TOOL_FULL_PAGE_REVIEWED_NATIVE_FETCH_FAILED",
            "document_id": "联防联控机制综发〔2022〕144号",
            "facts_paraphrased": [
                "官方页面发布日期为2022年12月26日，执行时间为2023年1月8日。",
                "执行后取消感染者强制隔离、密接判定和高低风险区划分，并取消入境集中隔离。",
                "文件以病毒与疫情变化、疫苗普及、防治经验和药物准备解释调整基础，同时保留防范医疗资源冲击的安排。",
            ],
            "native_attempts_preserved": ["classb_notice", "classb_notice_ndcpa"],
        },
        {
            "id": "us_cpi_web_fact", "page_date": "2022-11-10",
            "url": "https://www.bls.gov/news.release/archives/cpi_11102022.htm",
            "status": "WEB_TOOL_FULL_PAGE_REVIEWED_NATIVE_FETCH_FAILED",
            "public_at": "2022-11-10T08:30:00-05:00",
            "china_public_at": "2022-11-10T21:30:00+08:00",
            "facts_paraphrased": ["当次首次发布的2022年10月CPI同比7.7%，核心同比6.3%，均低于前一个月同比。"],
            "market_consensus": None,
            "native_attempts_preserved": ["us_cpi_nov10"],
        },
    ]
    core.save(OUT / "web_resolved_facts.json", web)
    core.save(OUT / "source_manifest.json", manifest)
    cards = [
        {"event_id": "PLAN9_20220628", "source_ids": ["plan9_notice", "plan9_qa_beijing"],
         "constraint_change": "密接及入境隔离从14天集中加7天监测缩为7天集中加3天监测；高、中、低风险区和相应流动限制仍存在。",
         "stated_upstream_reason": "官方解释基于奥密克戎传播特征及既有防控实践，调整隔离和监测方式；仍沿用动态清零总方针。",
         "index_transmission_inference": "减少特定人群的隔离时间可能缓解流动成本，不能据此认为全国接触型消费、生产和就业约束已经全部解除。"},
        {"event_id": "TWENTY_20221111", "source_ids": ["twenty_notice", "prior_guidance_mct", "us_cpi_web_fact"],
         "constraint_change": "密接及入境改为5天集中加3天居家，不再判定次密接，风险区缩为高低两类；跨省落地检及清零方针仍保留。",
         "stated_upstream_reason": "文件强调病毒传播变化、人口及医疗资源约束，调整措施精度以降低经济社会干扰；并未宣告全面结束防控限制。",
         "index_transmission_inference": "经济活动受限预期可能改变，但11月10日晚已经公开政策方向，同晚美国通胀发布；不能用次日全部价格变化估计二十条细则的独立贡献。"},
        {"event_id": "TEN_20221207", "source_ids": ["ten_notice_beijing"],
         "constraint_change": "大部分场所取消核酸及健康码要求，跨地区流动取消查验和落地检；符合条件的轻症感染者、密接转为居家隔离。",
         "stated_upstream_reason": "文件依据当时疫情和病毒变化，以及落实二十条过程中出现的问题，继续减少流动与活动限制。",
         "index_transmission_inference": "行政限制减轻与短期感染导致缺勤、需求下降可能同时发生，长期恢复预期与当月经营结果有不同时间尺度。"},
        {"event_id": "CLASSB_20221226", "source_ids": ["classb_web_fact"],
         "constraint_change": "公告明确2023年1月8日调整管理类别，取消感染者隔离、密接和风险区制度，并取消入境集中隔离。",
         "stated_upstream_reason": "文件把病毒/疫情、疫苗接种、防治积累与药物准备列为调整基础，并要求准备医疗承压应对。",
         "index_transmission_inference": "政策确定生效安排与经济活动实际恢复分属不同信息；入场只能用12月26日已公开内容，不能提前使用1月经营数据。"},
    ]
    for card in cards:
        card["market_expectation_surprise"] = None
        card["causal_equity_return_identified"] = False
    core.save(OUT / "mechanism_cards.json", cards)
    activity = [
        {"economic_month": "2022-12", "public_at": "2022-12-31T23:59:59+08:00", "source_id": "pmi_dec_review",
         "service_activity_index": 39.4, "service_affected_by_epidemic_pct": 61.3,
         "manufacturing_affected_by_epidemic_pct": 56.3,
         "meaning": "统计局当时报告到岗、物流与需求受疫情冲击；问卷扩散指数不是指数成分利润，也不是疫情的随机实验。"},
        {"economic_month": "2023-01", "public_at": "2023-01-31T09:30:00+08:00", "source_id": "pmi_jan_release",
         "service_activity_index": 54.0, "manufacturing_pmi": 50.1,
         "meaning": "经营活动相对12月回升；该结果在四个政策公告日均不可知。"},
    ]
    core.save(OUT / "subsequent_activity_facts.json", activity)
    return lookup, cards, activity


def inputs(protocol):
    market = pd.read_parquet(rapid.PRICE_BASE / "market.parquet")
    market["date"] = pd.to_datetime(market.date)
    market = market.loc[market.date.le(protocol["account_calendar"][1])].reset_index(drop=True)
    features = pd.read_parquet(rapid.PRICE_BASE / "price_features.parquet")
    features["date"] = pd.to_datetime(features.date)
    features = market[["date"]].merge(features[["date", "es95"]], on="date", how="left", validate="one_to_one")
    assert market.date.is_monotonic_increasing and market.date.is_unique
    dividends = core.normalize_dividends(pd.read_csv(rapid.PRICE_BASE / "dividends.csv"))
    spec = importlib.util.spec_from_file_location("reopening_frozen_execution", rapid.ENGINE_PATH)
    engine = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = engine
    spec.loader.exec_module(engine)
    clocks = []
    features[SIGNAL] = False
    for event in protocol["events"]:
        day = pd.Timestamp(event["date"])
        event_idx = int(np.flatnonzero(market.date.eq(day))[0])
        entry_idx = int(np.flatnonzero(market.date.gt(day))[0])
        public_at = (day + pd.Timedelta(hours=23, minutes=59, seconds=59)).tz_localize("Asia/Shanghai")
        entry_at = (market.date.iloc[entry_idx] + pd.Timedelta(hours=9, minutes=30)).tz_localize("Asia/Shanghai")
        assert public_at < entry_at and np.isfinite(features.es95.iloc[entry_idx - 1])
        features.loc[entry_idx - 1, SIGNAL] = True
        clocks.append({**event, "event_id": event["id"], "event_idx": event_idx,
                       "public_at_upper_bound": public_at.isoformat(), "entry_at": entry_at.isoformat(),
                       "entry_date": market.date.iloc[entry_idx], "entry_idx": entry_idx,
                       "es_observation_date": market.date.iloc[entry_idx - 1],
                       "entry_es95": features.es95.iloc[entry_idx - 1],
                       "planned_exit_date": market.date.iloc[entry_idx + 20]})
    return market, features, dividends, engine, clocks


def fixed_event_window(market, dividends, engine, clock, horizon):
    first = int(clock["entry_idx"])
    last = first + horizon
    row0 = market.iloc[first]
    acct = engine.Account(100000.)
    cfg = {"lot": 100, "tick": .001, "limit_fraction": .1}
    planned = math.floor(100000 / row0.open / 100) * 100
    buy = engine.execute_order(acct, planned, float(row0.open), float(row0.previous_close), float(row0.dividend), first, COST, cfg)
    if buy["filled_quantity"] == 0:
        return {"event_id": clock["event_id"], "horizon": horizon, "status": "入场未成交", "net_return": None}
    shares = acct.shares
    events = dividends.to_dict("records")
    recognized = 0.
    for i in range(first, last + 1):
        row = market.iloc[i]
        for k, event in enumerate(events):
            if event["ex_date"] == row.date:
                amount = acct.entitlements.get(k, 0) * event["cash_dividend_per_share"]
                if amount:
                    acct.receivables[k] = amount
                    recognized += amount
            if event["payment_date"] <= row.date and k in acct.receivables:
                acct.cash += acct.receivables.pop(k)
        if i == last:
            sell = engine.execute_order(acct, -shares, float(row.open), float(row.previous_close), float(row.dividend), i, COST, cfg)
            if sell["filled_quantity"] != -shares:
                return {"event_id": clock["event_id"], "horizon": horizon, "status": "固定退出未成交", "net_return": None}
        for k, event in enumerate(events):
            if event["record_date"] == row.date:
                acct.entitlements[k] = acct.shares
    paid = buy["notional"] + buy["commission"]
    profit = acct.value(float(market.open.iloc[last])) - 100000
    commissions = buy["commission"] + sell["commission"]
    slip = buy["slippage_cost"] + sell["slippage_cost"]
    gross_pnl = shares * (market.open.iloc[last] - row0.open) + recognized
    assert abs(profit + commissions + slip - gross_pnl) < 1e-6
    return {"event_id": clock["event_id"], "title": clock["title"], "horizon": horizon, "status": "两端成交",
            "entry_date": row0.date, "exit_date": market.date.iloc[last],
            "entry_open": row0.open, "exit_open": market.open.iloc[last], "shares": shares,
            "paid_cny": paid, "net_pnl": profit, "net_return": profit / paid,
            "gross_return": gross_pnl / (shares * row0.open), "commissions": commissions, "slippage": slip,
            "dividend_recognized": recognized, "dividend_receivable_at_exit": acct.receivable(),
            "is_full_account": False}


def price_stages(market, clocks):
    stages = []
    for clock in clocks:
        a, e = clock["event_idx"], clock["entry_idx"]
        before, anchor = a - 1, a - 21
        # 这几个固定公告附近没有ETF除息；显式确认后才使用价格比拆分。
        assert market.dividend.iloc[anchor + 1:e + 1].eq(0).all()
        stages.append({"event_id": clock["event_id"], "title": clock["title"],
                       "pre20_anchor": market.date.iloc[anchor], "pre_announcement_date": market.date.iloc[before],
                       "announcement_date": market.date.iloc[a], "entry_date": market.date.iloc[e],
                       "pre20_return": market.close.iloc[before] / market.close.iloc[anchor] - 1,
                       "announcement_open_gap": market.open.iloc[a] / market.close.iloc[before] - 1,
                       "announcement_open_to_close": market.close.iloc[a] / market.open.iloc[a] - 1,
                       "announcement_close_to_entry_open": market.open.iloc[e] / market.close.iloc[a] - 1,
                       "pre_close_to_entry_open": market.open.iloc[e] / market.close.iloc[before] - 1,
                       "event_close": market.close.iloc[a], "entry_open": market.open.iloc[e]})
    for row in stages:
        recomposed = (1 + row["announcement_open_gap"]) * (1 + row["announcement_open_to_close"]) * (1 + row["announcement_close_to_entry_open"]) - 1
        assert abs(recomposed - row["pre_close_to_entry_open"]) < 1e-12
    return stages


def accounts(protocol, market, features, dividends, engine, clocks):
    statistics, ledger_paths, account_checks, event_decisions, cycle_map = {}, {}, {}, {}, {}
    active_dates = market.date.between(*protocol["account_calendar"])
    first_idx = int(np.flatnonzero(active_dates)[0])
    for case_id in protocol["accounts"]:
        candidate = case_id.startswith("POLICY")
        benchmark = case_id.startswith("BUY_HOLD")
        capital = 20000 if case_id == "POLICY_20000_STRESS" else 200000
        feature_case = features.copy()
        if not candidate:
            feature_case[SIGNAL] = False
            if benchmark:
                feature_case.loc[first_idx - 1, SIGNAL] = True
        case = {"case_id": case_id, "capital": capital, "policy": SIGNAL,
                "hold": 20 if candidate else len(market) + 1,
                "start": protocol["account_calendar"][0], "end": protocol["account_calendar"][1]}
        ledger = account_model.simulate(market, feature_case, dividends, case,
                                       "CAP50_ONLY" if benchmark else "ORIGINAL", "STRESS", engine)
        assert ledger.date.reset_index(drop=True).equals(market.loc[active_dates, "date"].reset_index(drop=True))
        ledger["reason"] = ledger.reason.replace({"前收盘信号入场": "全国规则已公开后开盘入场" if candidate else "期初半仓参考入场"})
        ledger["es_observation_date"] = [market.date.iloc[int(i) - 1] for i in ledger.idx]
        ledger["prior_es95"] = [features.es95.iloc[int(i) - 1] for i in ledger.idx]
        ledger["fee_refund_equity"] = ledger.equity + (ledger.commission + ledger.slippage_cost).cumsum() + ledger.terminal_exit_reserve
        path = OUT / "ledgers" / f"{case_id}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        ledger.to_parquet(path, index=False)
        saved = pd.read_parquet(path)
        checked = checks.check_account(saved, capital, clocks, candidate)
        checked["buy_dates_subset_of_frozen_four_entries"] = checked.pop("buy_dates_subset_of_frozen_six_entries")
        checked["verified_from_saved_ledger"] = True
        account_checks[case_id] = checked
        stats = core.metrics(saved, capital)
        stats.pop("sharpe_252_diagnostic")
        stats.pop("cagr_252_diagnostic")
        stats.update({"capital": capital, "candidate": candidate,
                      "entry_fills": int(saved.filled_quantity.gt(0).sum()),
                      "holding_close_days": int(saved.shares.gt(0).sum()),
                      "no_position_all_session_days": int((saved.shares.eq(0) & saved.shares_before.eq(0)).sum()),
                      "max_close_exposure": saved.exposure.max(),
                      "max_close_exposure_times_prior_ES": (saved.exposure * saved.prior_es95).max(),
                      "close_exposure_above_50pct_days": int(saved.exposure.gt(.5 + 1e-12).sum()),
                      "max_drawdown_stop_triggered": bool(saved.risk_stopped.any()),
                      "risk_reduction_fills": int((saved.reason.eq("风险预算减仓") & saved.filled_quantity.lt(0)).sum()),
                      "rejected_nonzero_orders": int((saved.requested_quantity.ne(0) & saved.filled_quantity.eq(0)).sum()),
                      "worst_day": saved.net_return.min(),
                      "turnover_traded_notional_over_initial_capital": saved.notional.sum() / capital,
                      "dividend_recognized": saved.dividend_recognized.sum(),
                      "same_holdings_fee_refund": checks.equity_metrics(saved.fee_refund_equity, capital) if candidate else None,
                      "sample_joint_target_met": bool(candidate and stats["net_sharpe"] is not None and stats["net_sharpe"] >= 1.2
                          and stats["cagr"] >= .1 and stats["max_drawdown"] <= .1)})
        if candidate:
            decisions = []
            for clock in clocks:
                row = saved.loc[saved.date.eq(clock["entry_date"])].iloc[0]
                action = "入场" if row.filled_quantity > 0 else ("已有持仓，不加仓不延长" if row.shares_before > 0 else "风险或成交约束未入场")
                decisions.append({"event_id": clock["event_id"], "entry_date": row.date,
                                  "shares_before": row.shares_before, "filled_quantity": row.filled_quantity,
                                  "action": action, "status": row.status})
            cycles = core.cycle_records(saved, capital)
            assert cycles.closed.all() and saved.shares.iloc[-1] == 0
            assert abs(cycles.profit.sum() - stats["net_profit"]) < 1e-6
            stats["worst_cycle_profit"] = cycles.profit.min()
            stats["best_cycle_profit"] = cycles.profit.max()
            stats["best_cycle_share_of_net_profit"] = cycles.profit.max() / stats["net_profit"] if stats["net_profit"] > 0 else None
            event_decisions[case_id] = decisions
            cycle_map[case_id] = cycles.to_dict("records")
        statistics[case_id] = stats
        ledger_paths[case_id] = str(path.relative_to(ROOT))
        print(f"完成账户：{case_id}，净利润{stats['net_profit']:.2f}元，夏普{stats['net_sharpe']}")
    return statistics, ledger_paths, account_checks, event_decisions, cycle_map


def overlapping_windows(clocks, market):
    windows = [{"event_id": row["event_id"], "first": int(row["entry_idx"]), "last": int(row["entry_idx"] + 20)} for row in clocks]
    shared = []
    for i, left in enumerate(windows):
        for right in windows[i + 1:]:
            first, last = max(left["first"], right["first"]), min(left["last"], right["last"])
            if first < last:
                shared.append({"left": left["event_id"], "right": right["event_id"], "shared_open_intervals": last - first,
                               "from_open": market.date.iloc[first], "until_open": market.date.iloc[last]})
    return shared


def activity_release_clock(market, dividends):
    first = market.loc[market.date.eq("2022-12-27")].iloc[0]
    rows = []
    for day, kind in [("2022-12-30", "close"), ("2023-01-03", "open"),
                      ("2023-01-30", "close"), ("2023-02-01", "open")]:
        row = market.loc[market.date.eq(day)].iloc[0]
        end_mask = dividends.record_date.le(row.date) if kind == "close" else dividends.record_date.lt(row.date)
        cash_per_share = dividends.loc[dividends.record_date.ge(first.date) & end_mask, "cash_dividend_per_share"].sum()
        rows.append({"date": day, "mark_kind": kind, "price": float(row[kind]),
                     "dividend_per_share": cash_per_share,
                     "gross_return_from_dec27_open": (row[kind] + cash_per_share) / first.open - 1})
    facts = {"entry_open": first.open, "rows": rows,
             "dec_activity_known_at": "2022-12-31T23:59:59+08:00",
             "jan_activity_known_at": "2023-01-31T09:30:00+08:00",
             "jan_activity_first_strict_next_open": "2023-02-01T09:30:00+08:00",
             "role": "结果后补充的历史解释；端点按经营数据公开时钟选定，不选择价格最高点，不构成新的退出规则。"}
    core.save(OUT / "activity_release_price_clock.json", facts)
    return facts


def make_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10})
    stages, events = read(OUT / "price_stages.json"), read(OUT / "event_returns.json")
    result = read(OUT / "result.json")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1.1, 1]})
    x = np.arange(4)
    width = .25
    events20 = {row["event_id"]: row for row in events if row["horizon"] == 20}
    axes[0].bar(x - width, [row["pre20_return"] for row in stages], width, label="公告前20日价格变化", color="#aab4bd")
    axes[0].bar(x, [row["pre_close_to_entry_open"] for row in stages], width,
                label="公告前收盘至下一入场开盘", color="#cc9861")
    axes[0].bar(x + width, [events20[row["event_id"]]["net_return"] for row in stages], width,
                label="入场后20日含分红净收益", color="#176b7a")
    axes[0].set_xticks(x, ["6月28日\n第九版", "11月11日\n二十条", "12月7日\n十条", "12月26日\n乙类乙管"])
    axes[0].set_title("活动约束改变：价格何时反映，账户实际得到多少", loc="left", fontsize=15)
    axes[0].legend(frameon=False, ncol=3, fontsize=9)
    axes[0].set_ylabel("分段收益（口径不同，不相加）")
    for key, label, color in [("POLICY_200000_STRESS", "20万元规则账户", "#176b7a"),
                              ("POLICY_20000_STRESS", "2万元规则账户", "#cc9861"),
                              ("BUY_HOLD50_200000_REFERENCE", "半仓持有参考", "#aab4bd")]:
        ledger = pd.read_parquet(ROOT / result["ledger_paths"][key])
        capital = result["accounts"][key]["capital"]
        axes[1].plot(ledger.date, ledger.equity / capital - 1, label=label, color=color)
    axes[1].set_ylabel("连续账户累计净收益")
    axes[1].legend(frameon=False, loc="lower left", ncol=3)
    for ax in axes:
        ax.axhline(0, color="#737c84", linewidth=.7)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.text(.09, .02, "全国四次总体修订｜2022年初至2023年一季度完整账户｜重叠公告不重复开仓｜历史发现不等于独立验证", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(OUT / "活动约束_信息时序与账户.png", dpi=170)
    plt.close(fig)


def publish():
    protocol = read(OUT / "protocol.json")
    result = read(OUT / "result.json")
    stages = read(OUT / "price_stages.json")
    windows = read(OUT / "event_returns.json")
    cards = read(OUT / "mechanism_cards.json")
    source_lookup = {row["id"]: row for row in read(OUT / "source_manifest.json") + read(OUT / "web_resolved_facts.json")}
    market, features, dividends, engine, clocks = inputs(protocol)
    activity = activity_release_clock(market, dividends)
    main = result["accounts"]["POLICY_200000_STRESS"]
    refund = main["same_holdings_fee_refund"]
    cycles = read(OUT / "account_cycles.json")["POLICY_200000_STRESS"]
    pre_jan_release = next(row for row in activity["rows"] if row["date"] == "2023-01-30")
    nov = next(row for row in stages if row["event_id"] == "TWENTY_20221111")
    five = [row["net_return"] for row in windows if row["horizon"] == 5]
    twenty = [row["net_return"] for row in windows if row["horizon"] == 20]
    finding = {
        "four_adjustments_have_different_constraints": True,
        "five_day_positive_count": sum(value > 0 for value in five),
        "twenty_day_positive_count": sum(value > 0 for value in twenty),
        "nov11_open_gap": nov["announcement_open_gap"],
        "nov11_intraday_return": nov["announcement_open_to_close"],
        "nov_preclose_to_nov14_entry": nov["pre_close_to_entry_open"],
        "dec27_entry_to_pre_jan_pmi_gross_return": pre_jan_release["gross_return_from_dec27_open"],
        "best_cycle_share_of_total_net_profit": main["best_cycle_share_of_net_profit"],
        "mean_twenty_day_net_event_return": np.mean(twenty),
        "event_means_are_not_full_account_metrics": True,
        "causal_returns_identified": False,
    }
    result["discovery"] = finding
    result["report"] = str(REPORT.relative_to(ROOT))
    result["rule_disposition"] = "公告后统一20日买入表达未达标，停止针对本样本选事件、改期限或提高仓位的搜索。"
    result["source_facts_and_price_timeline_completed"] = True
    core.save(OUT / "result.json", result)
    fmt = lambda value: f"{value:.2%}"
    source_link = lambda key, label: f"[{label}]({source_lookup[key]['url']})"
    lines = ["# 历史发现：活动约束改变与指数剩余收益", "",
             f"**20万元主账户有盈利，但目标未达成：净夏普{main['net_sharpe']:.3f}、年化收益{fmt(main['cagr'])}、最大回撤{fmt(main['max_drawdown'])}。** 本轮更有用的发现是：四次政策调整改变的经济约束不同；价格、规则公布、实际经营确认有不同时间，不能统一贴上一个‘放松利好’标签。", "",
             "本轮以沪深300整体为研究单位，用510300实现历史账户。没有分析或交易个股。先查已有贷款需求/审批、季末资金、信用与估值研究，再固定2022年四次全国总体修订；没有复活旧降准规则。", "",
             "## 上游原因和真正改变的约束", "",
             "范围为第九版、二十条、十条及乙类乙管四次总体规则修订，不等于2022年全部涉疫公告。地方措施、单一航空或行程卡调整、实施问答不重复作为触发；这些背景也没有被认定为不重要。", "",
             "|公开日期|变化的经济活动约束|当时仍存在的限制或竞争解释|", "|---|---|---|",
             "|2022-06-28|密接与入境隔离由14+7缩为7+3|清零方针及风险区流动限制继续存在，不能当作全面恢复活动|",
             "|2022-11-11|隔离缩为5+3，取消次密接，压缩风险区范围|仍有清零要求和跨省落地检；前一晚已有政策方向及海外通胀信息|",
             "|2022-12-07|多数场所和跨地区流动取消核酸/健康码查验，符合条件者居家隔离|行政限制减轻不保证感染、缺勤、物流和需求冲击立即消失|",
             "|2022-12-26|明确1月8日改变管理类别，取消感染者隔离、密接和风险区制度及入境集中隔离|未来生效安排、医疗承压和实际活动恢复仍需分别观察|", "",
             "第九版和二十条的原文仍把病毒传播、防控资源与经济社会影响共同列入决策背景；乙类乙管文件则以病毒/疫情变化、疫苗、防治经验和药物准备说明调整基础。以上是当时官方说明，不能当作已经识别了股价变化的独立原因。来源：" + source_link("plan9_qa_beijing", "第九版官方解读") + "、" + source_link("twenty_notice", "二十条原文") + "、" + source_link("ten_notice_beijing", "十条原文") + "、" + source_link("classb_web_fact", "乙类乙管总体方案") + "。", "",
             "对指数的机制推断是：出行、就业和生产限制减少，可以影响总需求与未来盈利；短期感染对供需的冲击、其他宏观政策，以及投资者已经支付的价格会同时改变剩余收益。本轮没有把四个政策的作用大小拟合成分数，没有用后续感染或经营数据筛选入场。", "",
             "## 价格已经反映了什么", "",
             "统一取文件公开日结束后的下一ETF开盘；第九版签发于6月27日，公开于6月28日，故从6月29日进入。以下前20日截至公告前一收盘；公告前收盘到入场的变化不计入策略收益。", "",
             "|政策|公告前20日价格变化|公告前收盘→入场开盘|入场日|5日净收益|20日净收益|", "|---|---:|---:|---|---:|---:|"]
    returns = {(row["event_id"], row["horizon"]): row for row in windows}
    for row in stages:
        lines.append(f"|{row['title']}|{fmt(row['pre20_return'])}|{fmt(row['pre_close_to_entry_open'])}|{pd.Timestamp(row['entry_date']).date()}|{fmt(returns[(row['event_id'], 5)]['net_return'])}|{fmt(returns[(row['event_id'], 20)]['net_return'])}|")
    lines += ["", "前两列为价格变化，后两列为含分红、扣交易费用的固定持仓回报，分母为该事件实际投入金额。不同列不能直接相加，事件均值也不是连续账户年化。四个5日窗口均未盈利，20日三个为正；这里只描述全部原窗口，不能据此再选最优持有期。", "",
              f"11月11日开盘相对前收盘已上涨{fmt(nov['announcement_open_gap'])}，当天开盘到收盘仅上涨{fmt(nov['announcement_open_to_close'])}；到本规则11月14日开盘，累计价格变化已达{fmt(nov['pre_close_to_entry_open'])}。11月10日19:42的官方转载已经公开部署二十条，同日21:30北京时间美国CPI数据发布。两者在同一个A股隔夜区间内，日线不能分配各自贡献，也没有已核验的市场共识去估算精确预期差。来源：" + source_link("prior_guidance_mct", "前一晚会议报道") + "、" + source_link("us_cpi_web_fact", "美国当次CPI原始发布") + "。", "",
              "第九版之前的20日已经上涨11.47%，但这不证明上涨全因政策预期；相应20日净回报随后为负。价格先涨与公告后继续涨是两个不同命题。二十条20日窗口还包含后来的十条措施及11月底降准，不能将整段盈利全归因于一份文件。降准背景沿用" + source_link("rrr_nov25", "央行决定的官方报道") + "。", "",
              "## 经营恢复的确认，晚于一部分价格恢复", "",
              "12月31日发布的统计局解读称，12月服务业商务活动指数为39.4，服务业样本中受疫情影响较大的比例为61.3%；制造业的相同比例为56.3%，并指出到岗、配送和需求受冲击。这支持‘行政限制缓和与短期经营承压可以同时发生’这一历史描述。1月31日09:30公布的1月服务业商务活动指数回升到54.0。问卷指数不是指数成分利润，这些后来公布的数据均未用于12月政策入场。来源：" + source_link("pmi_dec_review", "12月PMI官方解读") + "、" + source_link("pmi_jan_release", "1月PMI原始发布") + "。", "",
              f"从12月27日开盘到1月30日收盘，固定份额含分红毛涨幅已达{fmt(pre_jan_release['gross_return_from_dec27_open'])}。1月30日是经营数据发布前最近收盘，作为事后解释端点，并不是新找到的卖点。1月PMI与1月31日开盘同为09:30，不能假定读完数据还能成交在同一个开盘价；统一下一开盘已是2月1日，恰好也是原20日窗口退出日。", "",
              "这给出一个可复核的时间关系：本次价格上涨先于月度经营改善的正式确认。它没有证明当时已经能准确预测恢复程度，也不足以把‘等到PMI最差就买’变成规律。未来恢复被提前交易、其他政策与外部风险变化等解释仍需区分。", "",
              "## 统一本金的完整账户", "",
              "日历为2022年1月1日至2023年3月31日，共301个ETF交易日。年末政策保留下一完整季度用于固定持有期和空仓核算。主持有期20个开盘间隔；已有持仓不加仓、不延长，风险只减仓。四个事件在同一账户只形成三次入场，十条公布时二十条持仓尚未结束。", "",
              "|账户|成本后夏普|年化收益|最大回撤|期末权益|平均仓位|", "|---|---:|---:|---:|---:|---:|"]
    names = {"POLICY_200000_STRESS": "20万元主账户", "POLICY_20000_STRESS": "2万元参考账户",
             "BUY_HOLD50_200000_REFERENCE": "期初半仓持有参考", "CASH_200000_REFERENCE": "现金参考"}
    for key, row in result["accounts"].items():
        sr = "不适用" if row["net_sharpe"] is None else f"{row['net_sharpe']:.3f}"
        lines.append(f"|{names[key]}|{sr}|{fmt(row['cagr'])}|{fmt(row['max_drawdown'])}|{row['end_equity']:,.2f}元|{fmt(row['mean_exposure'])}|")
    lines += ["", "半仓持有允许价格漂移、没有候选的每日风险减仓，仅为市场暴露参考，不是符合全部约束的替代候选。现金及无风险收益取零，按242交易日年化。", "",
              "佣金每边万四、每笔至少5元，滑点每边千一，100份整手，0.001报价；T+1及涨跌停和现金约束沿用原引擎。分红按登记日持有、除息应收、支付到账处理。50%目标上限、五日条件ES95预算2.5%、−10%冲击预算5%及回撤余量规则继续执行；模拟不是真实成交回执。", "",
              "|主账户入场→退出|该轮净盈亏|", "|---|---:|"]
    for cycle in cycles:
        lines.append(f"|{pd.Timestamp(cycle['entry']).date()}→{pd.Timestamp(cycle['exit']).date()}|{cycle['profit']:+,.2f}元|")
    lines += ["", f"最好一轮贡献约为总净利润的{fmt(main['best_cycle_share_of_net_profit'])}，说明此前亏损抵消了部分盈利，并非每轮稳定贡献。费用返还但份额完全不变的解释性账户，夏普也只有{refund['net_sharpe']:.3f}、年化{fmt(refund['cagr'])}；降低费用不足以填平目标差距。没有运行取消风险预算、扩大仓位或筛选赢家的版本。", "",
              f"主账户有{main['risk_reduction_fills']}次风险减仓，未发生非零委托拒单或回撤停机；最高收盘仓位{fmt(main['max_close_exposure'])}，收盘仓位乘以前一收盘ES的最高值{fmt(main['max_close_exposure_times_prior_ES'])}。这些是本样本记录，不是风险上限在所有时刻或未来都能保证。", "",
              "## 本轮决定", "",
              "新增的是四类活动约束的原始文件、信息时序、8个固定事件窗口和4个连续账户。此前研究已见过相关市场价格，本轮属于历史发现，不是独立验证。",
              "",
              "公告后统一买入这一表达没有达到夏普1.2、年化10%的目标，结束对这四次事件的参数搜索。保留的机制认识是：政策为什么变、实际解除哪种约束、过渡期是否仍承压、价格已反映多少，必须分别回答。后续若研究恢复进度，需要当时已发布的独立信息，并先检查已有PMI、企业调查和价格研究，不能把本轮盈利区间改名为新策略。", "",
              "四份保存账本已重算权益与收益，核对T+1、现金、整手及事件时间；价格分段乘积一致。原始获取失败记录保留，乙类乙管与美国CPI由网页工具核阅官方全文并保存转述事实，未伪装成已下载原HTML。", "",
              "文件：`protocol.json`、`mechanism_cards.json`、`source_manifest.json`、`web_resolved_facts.json`、`price_stages.json`、`event_returns.json`、`activity_release_price_clock.json`、`account_event_decisions.json`、`account_cycles.json`、`result.json`及`ledgers/`。", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    make_plot()
    print(f"已生成报告：{REPORT}")


def calculate():
    protocol = read(OUT / "protocol.json")
    assert protocol["primary_account_hold_open_intervals"] == 20 and not protocol["parameter_search"]
    assert len(protocol["events"]) == 4
    source_facts()
    market, features, dividends, engine, clocks = inputs(protocol)
    event_rows = [fixed_event_window(market, dividends, engine, clock, horizon)
                  for clock in clocks for horizon in protocol["event_horizons"]]
    assert all(row["status"] == "两端成交" for row in event_rows)
    stages = price_stages(market, clocks)
    stats, paths, verification, decisions, cycles = accounts(protocol, market, features, dividends, engine, clocks)
    overlap = overlapping_windows(clocks, market)
    main = stats["POLICY_200000_STRESS"]
    result = {
        "study_id": protocol["study_id"], "completed_at": core.now(),
        "classification": "PROGRESS_REOPENING_CONSTRAINTS_PRICE_CLOCK_AND_FULL_ACCOUNT",
        "status": "COMPLETED_FIXED_HISTORICAL_STUDY", "events": 4, "event_return_rows": len(event_rows),
        "accounts": stats, "ledger_paths": paths, "overlap": overlap,
        "primary_sample_target_met": main["sample_joint_target_met"],
        "new_parameters_fitted": 0, "new_signal_filters": 0, "new_full_accounts": 4,
        "independent_validation": False, "net_sharpe": main["net_sharpe"],
        "goal_achieved": False, "orders_authorized": False,
        "research_action": "保留机制与信息时序发现；不得按四个已见结果选政策、改持有期或拼接账户。",
        "next_historical_question": "从这一活动约束变化中找可在当时观察的恢复进度与价格反映，不把随后公布的月度经营改善提前放入公告日；先与已有PMI、企业调查及价量研究去重。",
    }
    for name, obj in [("result.json", result), ("event_clocks.json", clocks), ("event_returns.json", event_rows),
                      ("price_stages.json", stages), ("account_event_decisions.json", decisions),
                      ("account_cycles.json", cycles), ("calculation_checks.json", verification)]:
        core.save(OUT / name, obj)
    print(f"本轮计算完成：4次政策、8个固定窗口、4份账户；目标达成：否。")


if __name__ == "__main__":
    if "--publish" not in sys.argv:
        calculate()
    publish()
