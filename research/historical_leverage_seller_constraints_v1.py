"""分解2024年初融资活动，保留官方事实与公告后历史收益。"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from historical_price_gap_causes_v1 import round_trip

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_leverage_seller_constraints_v1"
MARGIN_PATH = ROOT / "reports/research/510300_factor96_crowding_overlay_v1_0_1/inputs/margin.parquet"
MARKET_PATH = ROOT / "reports/research/510300_macro_dynamic_reframe_v1/inputs/market.parquet"
E8 = 100_000_000


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save(name, value):
    (OUT / name).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_facts():
    sources = json.loads((OUT / "source_index.json").read_text(encoding="utf-8"))
    for row in sources:
        assert row["status"] == "CONTENT_PRESENT", f"缺少原文：{row['key']}"
    reused = [
        ("gov_rrr_20240125", "2024-01-25", "https://app.www.gov.cn/govdata/gov/202401/25/511522/article.html",
         "reports/research/510300_policy_information_clock_v1/raw/rate_20240125_gov.html"),
        ("huijin_20240206", "2024-02-06", "https://www.huijin-inv.cn/huijin-inv/c100077/2024-02/1002262.shtml",
         "reports/research/510300_huijin_etf_disclosure_source_v1/raw/83b10cd4a724be5916c1.html"),
        ("sse_margin_definition", None, "https://www.sse.com.cn/market/othersdata/margin/sum/",
         "reports/research/510300_exchange_repo_source_probe_v1/raw/margin_definition.bin"),
    ]
    for key, date, url, name in reused:
        data = (ROOT / name).read_bytes()
        sources.append({"key": key, "publication_date": date, "url": url, "raw_path": name,
                        "raw_sha256": hashlib.sha256(data).hexdigest(), "mode": "REUSE_SAVED_SOURCE",
                        "historical_first_vintage_verified": False})
    # 保留全部来源，单次运行不会修改已有原件或重复写入原索引。
    sources = list({r["key"]: r for r in sources}.values())
    mapping = {r["key"]: r for r in sources}
    facts = [
        {"id": "F01", "source_key": "csrc_financing_20240205", "fact": "1月以来全市场两融累计平仓金额约9亿元；平均维持担保比例226%。",
         "closeout_cny": 900_000_000, "average_maintenance_percent": 226,
         "statistical_start": "2024-01-01", "cutoff_upper": "2024-02-05", "precise_statistical_cutoff": None,
         "meaning": "这份两融口径的实际平仓统计不覆盖全部场外杠杆和预防性减仓；平均数也不表示每户安全。"},
        {"id": "F02", "source_key": "csrc_financing_20240205", "fact": "引导券商延长追保时间、动态下调平仓线；主动卖股还款也可使融资余额下降。",
         "policy_status": "监管引导，未给出每家券商实际调整时间和幅度",
         "meaning": "可改变被迫卖出的时间约束，但未证明担保不足消失、每户均获延期或卖压当日归零。"},
        {"id": "F03", "source_key": "csrc_pledge_20240205", "fact": "年初以来股票质押违约强平金额2740.32万元；至2月2日共披露106单大股东补充质押。",
         "pledge_closeout_cny": 27_403_200, "additional_pledge_notices": 106,
         "additional_pledge_notices_cutoff": "2024-02-02", "closeout_exact_cutoff_repeated_in_source": False,
         "meaning": "补充质押发生在预警线约束，不能把公告数量当作实际强平次数；与两融平仓属于不同统计口径。"},
        {"id": "F04", "source_key": "csrc_lending_20240206", "fact": "暂停新增转融券规模，存量逐步了结；公布融券余额637亿元。",
         "securities_lending_balance_cny": 63_700_000_000,
         "meaning": "这是融券供给措施，不等于取消融资债务或直接向股票市场注入637亿元。"},
        {"id": "F05", "source_key": "csrc_dma_20240228", "fact": "DMA为多头一篮子股票并以股指期货套保的市场中性策略；春节后规模稳步下降，日均成交占全市场约3%。",
         "average_trading_share": 0.03, "scale_cny": None, "positions_cny": None,
         "meaning": "3%是成交占比，不是剩余持仓或卖压占比。降杠杆仍在进行；减多头并买回期货空头是可能路线，原文未给出每天各腿的交易。"},
        {"id": "F06", "source_key": "xinhua_rrr_20240124", "fact": "1月24日发布会宣布2月5日降准0.5个百分点，提供长期流动性约1万亿元。",
         "rrr_reduction_percentage_points": 0.5, "effective_date": "2024-02-05",
         "meaning": "新华社同日现场报道提供历史日末时钟，政府网所存央行通知核对政策内容；银行体系流动性不等于股票账户购买。"},
    ]
    for fact in facts:
        src = mapping[fact["source_key"]]
        fact.update(publication_date=src["publication_date"], source_url=src["url"], source_path=src["raw_path"])
    save("本轮来源及复用原件.json", sources)
    save("原始事实与解释边界.json", facts)
    return mapping, facts


def build_margin(protocol, market):
    f = pd.read_parquet(MARGIN_PATH).sort_values("date").copy()
    f["date"] = pd.to_datetime(f.date)
    f["balance_change"] = f.market_rzye.diff()
    f["implied_repayment"] = f.market_rzmre - f.balance_change
    begin, end = protocol["observation_period"]
    panel = f[f.date.between(begin, end)].copy()
    calendar = market[market.date.between(begin, end)].date
    assert set(panel.date) == set(calendar), "所选固定区间融资与交易日未覆盖一致"
    assert panel[["market_rzye", "market_rzmre", "implied_repayment"]].notna().all().all()
    assert panel.implied_repayment.ge(0).all()
    phases = []
    for label, start, stop in protocol["fixed_phases"]:
        p = f[f.date.between(start, stop)]
        prior = f[f.date < p.date.iloc[0]].iloc[-1]
        net = float(p.market_rzye.iloc[-1] - prior.market_rzye)
        assert abs(net - p.balance_change.sum()) < .01
        a = market[market.date < p.date.iloc[0]].iloc[-1]
        b = market[market.date <= p.date.iloc[-1]].iloc[-1]
        phases.append({"label": label, "first_session": p.date.iloc[0], "last_session": p.date.iloc[-1],
                       "trading_days": len(p), "start_balance": float(prior.market_rzye),
                       "end_balance": float(p.market_rzye.iloc[-1]), "balance_change": net,
                       "mean_daily_financing_buy": float(p.market_rzmre.mean()),
                       "mean_daily_implied_repayment": float(p.implied_repayment.mean()),
                       "mean_daily_net_change": float(p.balance_change.mean()),
                       "contemporaneous_total_price_return": float(b.wealth/a.wealth-1)})
    shifts = []
    for a, b in zip(phases, phases[1:]):
        buying = b["mean_daily_financing_buy"]-a["mean_daily_financing_buy"]
        repay = b["mean_daily_implied_repayment"]-a["mean_daily_implied_repayment"]
        net = b["mean_daily_net_change"]-a["mean_daily_net_change"]
        assert abs(net-buying+repay) < .01
        shifts.append({"from": a["label"], "to": b["label"], "mean_daily_net_shift": net,
                       "buying_change_contribution": buying, "repayment_change_contribution": -repay,
                       "interpretation": "恒等式分解，不识别交易动机或政策因果贡献。"})
    anchor = f[f.date.eq(pd.Timestamp("2023-12-29"))].iloc[0]
    comparisons = []
    for date in ["2024-02-02", "2024-02-05"]:
        last = f[f.date.eq(pd.Timestamp(date))].iloc[0]
        comparisons.append({"start": "2023-12-29", "end": date,
                            "financing_balance_drop_cny": float(anchor.market_rzye-last.market_rzye),
                            "reported_closeouts_cny": 900_000_000,
                            "comparison_is_causal_contribution_ratio": False,
                            "caveat": "余额净变化与累计平仓流量不同，且9亿元统计精确截止日未明；仅列金额量级，不计算贡献率。"})
    save("固定历史段融资活动.json", phases)
    save("融资净变化的相邻阶段分解.json", shifts)
    save("余额收缩与实际平仓的口径对照.json", comparisons)
    keep = ["date", "market_rzye", "market_rzmre", "balance_change", "implied_repayment", "source"]
    save("全区间融资日线.json", panel[keep].to_dict("records"))
    return panel, phases, shifts, comparisons


def event_returns(protocol, market, sources):
    specifications = [
        ("2024-01-24", "降准发布会", ["xinhua_rrr_20240124", "gov_rrr_20240125"], "一般流动性背景；不等于股票账户已经收到资金"),
        ("2024-02-05", "两融与质押缓释说明", ["csrc_financing_20240205", "csrc_pledge_20240205"], "延长追保及平仓线弹性；各家实施未知"),
        ("2024-02-06", "融券措施及汇金增持", ["csrc_lending_20240206", "huijin_20240206"], "供给约束和购买支持同日出现，不能拆出各自效果"),
        ("2024-02-28", "DMA降杠杆说明", ["csrc_dma_20240228"], "确认规模下降仍在进行，未宣布减仓结束"),
    ]
    assert [r[0] for r in specifications] == protocol["fixed_information_dates"]
    dividends = pd.read_csv(ROOT / "data/reference/510300_dividends.csv")
    dividends = dividends[dividends.symbol.eq("510300.SH")].copy()
    dividends["record_date"] = pd.to_datetime(dividends.record_date)
    results = []
    costs = protocol["event_return"]["costs"]
    for date, title, keys, meaning in specifications:
        entry_i = int(market.date.searchsorted(pd.Timestamp(date), side="right"))
        entry, previous = market.iloc[entry_i], market.iloc[entry_i-1]
        record = {"source_date": date, "title": title, "source_keys": keys, "source_urls": [sources[k]["url"] for k in keys],
                  "available_upper": date+"T23:59:59+08:00", "meaning_at_publication": meaning,
                  "entry_date": entry.date, "entry_open": float(entry.open),
                  "first_market_surprise_identified": False, "event_cluster": "2024年初压力及政策反应",
                  "gap_after_source_close": float((entry.open+entry.dividend)/previous.close-1)}
        for horizon in protocol["event_return"]["holding_sessions"]:
            end = market.iloc[entry_i+horizon]
            eligible = dividends.record_date.ge(entry.date) & dividends.record_date.lt(end.date)
            entitlement = float(dividends.loc[eligible, "cash_dividend_per_share"].sum())
            trade = round_trip(entry.open, end.open, entitlement, costs)
            assert abs(trade["shares"]*(trade["sell_price"]-trade["buy_price"]+entitlement)-trade["commissions_cny"]-trade["net_pnl_cny"]) < 1e-7
            record[f"holding_{horizon}"] = {"exit_date": end.date, "exit_open": float(end.open),
                                           "gross_return": float((end.open+entitlement)/entry.open-1),
                                           "dividend_per_share": entitlement, **trade}
        record["overlapping_previous_20_sessions"] = [r["source_date"] for r in results if entry.date < r["holding_20"]["exit_date"]]
        results.append(record)
    save("四个信息时点与费用后参考收益.json", results)
    return results


def make_figure(panel, phases, market):
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(2, 1, figsize=(13, 9.5), gridspec_kw={"height_ratios": [1.1, 1]})
    price = market[market.date.between("2023-12-01", "2024-03-29")]
    base_price = market[market.date.eq(pd.Timestamp("2023-12-29"))].wealth.iloc[0]
    base_debt = panel[panel.date.eq(pd.Timestamp("2023-12-29"))].market_rzye.iloc[0]
    axes[0].plot(price.date, price.wealth/base_price*100, label="510300含分红价格", color="#007F77", linewidth=2.2)
    axes[0].plot(panel.date, panel.market_rzye/base_debt*100, label="全市场融资余额", color="#C3754C", linewidth=2)
    for date in ["2024-01-24", "2024-02-05", "2024-02-28"]:
        axes[0].axvline(pd.Timestamp(date), color="#A4AEB7", linestyle="--", linewidth=.8)
    axes[0].axhline(100, color="#BBC1C6", linewidth=.8)
    axes[0].set_ylabel("2023年末 = 100")
    axes[0].set_title("价格修复与融资余额回升并不需要同时发生", loc="left", fontweight="bold", pad=12)
    axes[0].legend(loc="upper left", ncol=2, frameon=False)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    axes[0].grid(axis="y", alpha=.15)
    axes[0].text(.02,.04,"竖线：1/24降准发布会、2/5两融与质押说明、2/28 DMA说明\n同一轮市场压力的不同节点，不是独立政策试验", transform=axes[0].transAxes, fontsize=9, color="#596573")
    x = np.arange(len(phases))
    buys = np.array([p["mean_daily_financing_buy"]/E8 for p in phases])
    repayments = np.array([p["mean_daily_implied_repayment"]/E8 for p in phases])
    axes[1].bar(x-.18, buys, .34, color="#007F77", label="日均融资买入")
    axes[1].bar(x+.18, repayments, .34, color="#C3754C", label="日均反推偿还")
    for offset, values in [(-.18,buys),(.18,repayments)]:
        for i,v in enumerate(values): axes[1].text(i+offset,v+10,f"{v:.0f}",ha="center",fontsize=9)
    axes[1].set_xticks(x,["2023年12月","2024年1月","2/1至2/5","2/6至2/8","2/19至2/28","2/29至3/29"])
    axes[1].set_ylabel("亿元 / 交易日")
    axes[1].set_ylim(0,max(buys.max(),repayments.max())*1.22)
    axes[1].legend(ncol=2,frameon=False,loc="upper left")
    axes[1].set_title("余额净变化需要拆开新增买入与偿还",loc="left",fontweight="bold",pad=12)
    for ax in axes: ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("2024年初融资约束与价格修复的历史分解",x=.075,ha="left",fontsize=17,fontweight="bold")
    fig.text(.075,.02,"偿还由统计恒等式反推，包含直接还款、卖券还款、强平和权益调整；并非全部为股票卖出。价格未扣交易费。",fontsize=9,color="#596573")
    fig.subplots_adjust(left=.075,right=.98,top=.90,bottom=.09,hspace=.40)
    fig.savefig(OUT/"融资买入偿还与价格修复.png",dpi=170)
    plt.close(fig)


def report(phases, shifts, comparisons, events, sources, result):
    p = phases[3]
    rows = ["# 2024年初融资约束和卖方行为的历史发现", "",
        "本轮追查2024年初融资余额下降的构成，以及两融、股票质押和DMA的不同约束。新增发现是：1月净收缩加快主要来自融资买入额下降；2月初偿还额上升成为更大的变化来源；2月6日后净流出缩小来自买入恢复，偿还并没有下降。相同的余额因子，在不同阶段由不同分项主导。尚未形成达到夏普1.2的策略。", "",
        "样本保留2023年12月至2024年3月的全部日线，并按预先列明的日历月和公告日期分段。政策是在市场压力下发生的，分段对照不是政策效果的随机试验。", "",
        "![融资活动和价格对照](融资买入偿还与价格修复.png)", "",
        "**先拆融资余额。** 上交所统计定义为：余额变化等于融资买入减融资偿还；偿还又包含直接还款、卖券还款、强平及权益调整。因此，本轮以余额和买入反推的偿还量不能叫作实际卖股金额。沪市原始官方数据与深市已存批量数据均复用，深市历史批量源是金十并有既有官方抽样核对；没有将它写成全部逐日原始官方凭证。[上交所统计说明](https://www.sse.com.cn/market/othersdata/margin/sum/)。", "",
        "| 固定历史段 | 交易日 | 日均融资买入 亿元 | 日均反推偿还 亿元 | 日均净变化 亿元 | 期间余额变化 亿元 | 同期含分红价格回报 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in phases:
        rows.append(f"| {r['label']} | {r['trading_days']} | {r['mean_daily_financing_buy']/E8:.2f} | {r['mean_daily_implied_repayment']/E8:.2f} | {r['mean_daily_net_change']/E8:+.2f} | {r['balance_change']/E8:+.2f} | {r['contemporaneous_total_price_return']:+.2%} |")
    rows += ["", f"2月6日至8日，融资余额继续减少{abs(p['balance_change'])/E8:.2f}亿元，但510300相对2月5日收盘的含分红价格回报为{p['contemporaneous_total_price_return']:+.2%}。‘余额转正才算压力缓解’会把融资库存与股票价格强行绑定；这里并未识别谁接走了卖单，价格反弹也不能反过来证明融资卖压已经结束。", "",
        f"与2月1日至5日相比，这三天的日均净流出缩小{shifts[2]['mean_daily_net_shift']/E8:.2f}亿元，来自融资买入增加{shifts[2]['buying_change_contribution']/E8:.2f}亿元，减去偿还增加{abs(shifts[2]['repayment_change_contribution'])/E8:.2f}亿元。由此能排除‘净流出缩小必然是偿还活动减弱’的解释；也不能因此认定全部新增承接来自融资账户。", "",
        "进一步将相邻阶段的日均净变化拆成买入变化减偿还变化，保留每一段，不选择最有利解释：", "",
        "| 阶段转变 | 日均净变化的变化 亿元 | 买入变化贡献 亿元 | 偿还变化贡献 亿元 |",
        "| --- | ---: | ---: | ---: |"]
    for r in shifts:
        rows.append(f"| {r['from']} → {r['to']} | {r['mean_daily_net_shift']/E8:+.2f} | {r['buying_change_contribution']/E8:+.2f} | {r['repayment_change_contribution']/E8:+.2f} |")
    rows += ["", "第二、三列相加等于第一列。它们是统计来源，不等于投资者动机：买入降低可能涉及预期、风险预算或融资约束；偿还升高可能涉及现金还款、卖出或其他调整。这些是名义金额，也受价格、整体成交和标的结构影响，不能把买入额下降直接等同于意愿减弱。没有逐户记录时，不填入具体动机的比例。", "",
        "**再看实际被披露的约束。** 2月5日证监会披露，1月以来两融累计平仓约9亿元、平均维持担保比例226%，并明确主动卖股还款也会使余额下降；随后提出延长追保时间、动态下调平仓线。若落实，首先改变的是部分账户的时间与处置约束，原文没有给出各券商实施清单，也没有保证股票价值或价格。[两融融资业务说明](https://www.csrc.gov.cn/csrc/c100028/c7461992/content.shtml)。", "",
        f"既有余额数据中，2023年末至2月2日融资余额净减少{comparisons[0]['financing_balance_drop_cny']/E8:.2f}亿元，至2月5日净减少{comparisons[1]['financing_balance_drop_cny']/E8:.2f}亿元。它们与约9亿元实际平仓量级明显不同，不能把净收缩全部命名为强平。净余额和累计流量的口径不同，9亿元的精确统计截止日也未明，本轮没有计算‘强平贡献率’。统计未覆盖所有场外杠杆，也不能据此否定强平前已发生的防御性减仓。", "",
        "同日股票质押说明公布的违约强平为2740.32万元；106单补充质押公告对应的是预警后的担保补充，不能等同于106次强平。两融与质押数字分开保存，不相加成全市场强制卖出总额。[股票质押说明](https://www.csrc.gov.cn/csrc/c100028/c7461718/content.shtml)。", "",
        "2月28日DMA说明则确认另一条路径：私募持有股票组合并以股指期货套保，在净值回撤后降杠杆、降规模。春节后日均成交占比约3%，只是成交占比，无法推出剩余库存或未来待卖数量。缩减这种组合可能同时卖股票、买回期货空头，其指数净风险变化还取决于持仓和对冲；原文没有提供这些逐日数据。‘正在降杠杆’也不等于‘降杠杆已经结束’。[DMA业务说明](https://www.csrc.gov.cn/csrc/c100028/c7465086/content.shtml)。", "",
        "**最后检查信息公开后的剩余收益。** 四个时点均保守在来源日结束后，取下一交易日开盘；分别持有5、20个交易日后开盘退出。参考交易金额10万元，单边佣金万分之四、最低5元，单边滑点千分之一并按0.001元不利方向取整。收益分母为实际买入成本，含期间登记分红权益；这是事件交易参考，未形成完整账户或风险预算。", "",
        "| 信息时点 | 次日可用开盘 | 5日费用后收益 | 20日费用后收益 |",
        "| --- | --- | ---: | ---: |"]
    for r in events:
        rows.append(f"| {r['source_date']} {r['title']} | {r['entry_date']:%Y-%m-%d} | {r['holding_5']['net_return']:+.2%} | {r['holding_20']['net_return']:+.2%} |")
    rows += ["", "1月24日用新华社同日现场报道确认日末前已公开，次日政府网保存的央行通知只用于核对政策内容，没有提前把次日转载当时钟。2月5日两份说明合并为一个节点；2月6日融券措施和汇金购买同日发生，不能把后续收益单独归给任一项。[新华社现场报道](https://www.cnfin.com/yw-lb/detail/20240124/4004763_1.html)、[央行通知政府网存档](https://app.www.gov.cn/govdata/gov/202401/25/511522/article.html)、[2月6日融券说明](https://www.csrc.gov.cn/csrc/c100028/c7462169/content.shtml)、[汇金公告](https://www.huijin-inv.cn/huijin-inv/c100077/2024-02/1002262.shtml)。", "",
        "四个时点处于同一轮市场压力，持有区间有重叠，后面的信息还会进入前面交易的持有期；不计算独立胜率或把收益拼接成账户。5日和20日均保留，不事后挑较好的期限，也没有测得这几次消息相对市场共识的精确预期差。", "",
        "能够支持的机制是：价格下跌与净值回撤可能触发担保、止损或杠杆约束；投资者可以在正式强平前主动减仓；延长追保或补充购买力量可能改变卖压与承接的相对强弱。这里的‘主动’描述执行方式，不表示完全不受约束。未来是否继续卖出不能由一次余额下降单独确定。", "",
        "旧融资级联模型、IF强制资金流模型和逐日流动性修复账户保持原状态。IF那一项在事件数量门结束，收益未运行，本轮没有把未运行当成收益失败。新模型0、参数搜索0、完整账户0，夏普1.2仍未达到。", "",
        "下一项历史检验应围绕同类融资处置约束的改变，寻找其他已发生且有原始公告的案例，并保留政策出现后仍继续下跌的反例。只有从公开后可成交价格形成可重复优势，再检验完整账户。", "",
        "分项原值、来源日期、四个参考交易及所用旧结果保存在同目录JSON中；图表和本文由研究脚本生成。"]
    (OUT/"历史发现_融资收缩背后的不同卖方约束.md").write_text("\n".join(rows)+"\n",encoding="utf-8")


def main():
    protocol = json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    sources, facts = source_facts()
    market = pd.read_parquet(MARKET_PATH).sort_values("date").reset_index(drop=True)
    market["date"] = pd.to_datetime(market.date)
    panel, phases, shifts, comparisons = build_margin(protocol,market)
    events = event_returns(protocol,market,sources)
    old_files = ["reports/research/510300_participant_identity_clock_v1/evidence/融资压力旧失败.json",
                 "reports/research/510300_if_forced_flow_state_v1.json",
                 "reports/research/510300_daily_liquidity_insurance_tail_v1/result.json"]
    old_status = []
    for name in old_files:
        j = json.loads((ROOT/name).read_text(encoding="utf-8"))
        old_status.append({"source": name,"status":j["status"],"primary":j.get("primary"),
                           "portfolio":j.get("portfolio"),"interpretation":j.get("interpretation")})
    result = {"study_id":protocol["study_id"],"completed_at":datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "status":"HISTORICAL_BORROWING_REPAYMENT_AND_SELLER_CONSTRAINTS_DISTINGUISHED",
              "previous_goal_turn_classification":protocol["previous_goal_turn_classification"],
              "daily_observations":len(panel),"fixed_phases":len(phases),"information_nodes":len(events),
              "independent_policy_experiments":0,"reported_actual_financing_closeouts_cny":900_000_000,
              "preholiday_after_relief_phase":phases[3],"mean_flow_shifts":shifts,"balance_comparisons":comparisons,
              "new_mechanism_finding":"相对于前一固定阶段，1月买入金额下降主导净收缩加快，2月初偿还上升主导进一步收缩；2月6日至8日净流出缩小来自买入增加136.11亿元每天，偿还同时增加20.24亿元每天。仅为名义金额分解，非逐户动机识别。",
              "event_net_returns":[{"source_date":r["source_date"],"entry":r["entry_date"],
                 "net5":r["holding_5"]["net_return"],"net20":r["holding_20"]["net_return"]} for r in events],
              "reused_old_results_unchanged":old_status,"new_fitted_models":0,"new_parameter_searches":0,
              "new_full_accounts":0,"new_prospective_forecasts":0,"goal_achieved":False,"orders_authorized":False,
              "minimal_checks":{"fixed_phase_day_coverage":True,"balance_flow_identities":True,"post_source_open_only":True,"round_trip_pnl_identity":True},
              "next_question":"以融资处置约束改变为历史事件类型，查找2015和2018年延长追保、协商展期或调整担保处置的原始公告，比较公开后的收益与卖压是否继续。先查已有研究，案例不足就报告不足，不挑最有利起点或期限。"}
    save("result.json",result)
    make_figure(panel,phases,market)
    report(phases,shifts,comparisons,events,sources,result)
    print(json.dumps(clean({"状态":result["status"],"说明后春节前":phases[3],"四时点净收益":result["event_net_returns"]}),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
