"""分离指数、基金净值与成交价格，并解释既有完整周期的利润来源。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_index_etf_price_layer_v1"
MARKET = ROOT / "reports/research/510300_macro_dynamic_reframe_v1/inputs/market.parquet"
NAV = ROOT / "data/raw/fund/510300_nav_daily_raw.parquet"
INDEX = ROOT / "data/raw/r6/000300_daily.parquet"
TZ = ZoneInfo("Asia/Shanghai")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def encode(value):
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"不能编码的类型：{type(value)}")


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=encode, allow_nan=False) + "\n", encoding="utf-8")


def link(label, path):
    return f"[{label}](<{Path(path).absolute().as_posix()}>)"


def compute():
    protocol = read(OUT / "protocol.json")
    m = pd.read_parquet(MARKET).sort_values("date").reset_index(drop=True)
    n = pd.read_parquet(NAV)
    ix = pd.read_parquet(INDEX)
    d = m.merge(n[["date", "unit_nav", "nav_eastmoney", "nav_sina", "source_primary", "source_secondary"]],
                on="date", how="left", validate="one_to_one")
    d = d.merge(ix[["date", "close"]].rename(columns={"close": "index_close"}), on="date", how="left", validate="one_to_one")
    d = d.loc[d.date.between(protocol["window_start"], protocol["window_end"])].copy().reset_index(drop=True)
    assert len(d) == m.date.between(protocol["window_start"], protocol["window_end"]).sum()
    assert d[["unit_nav", "nav_eastmoney", "nav_sina", "index_close"]].notna().all().all()
    assert np.allclose(d.nav_eastmoney, d.nav_sina, rtol=0, atol=1e-10)
    d["spread_cny_per_share"] = d.close - d.unit_nav
    d["premium"] = d.close / d.unit_nav - 1
    d["price_change"] = d.close.diff()
    d["nav_change"] = d.unit_nav.diff()
    d["spread_change"] = d.spread_cny_per_share.diff()
    d["premium_declined"] = d.premium.diff().lt(0)
    d["price_up"] = d.price_change.gt(0)
    # 金额价差的缩小与比率的缩小分别保留，避免分母变化造成误读。
    d["spread_narrowed"] = d.spread_change.lt(0)
    d["eventual_nav_is_same_close_trading_input"] = False
    d.to_parquet(OUT / "daily_layers.parquet", index=False)
    save("daily_layers.json", json.loads(d.to_json(orient="records", date_format="iso", force_ascii=False)))
    table = d.set_index("date")
    segments = []
    for start, end, label in protocol["fixed_segments"]:
        a, b = table.loc[pd.Timestamp(start)], table.loc[pd.Timestamp(end)]
        dividends = d.loc[(d.date > pd.Timestamp(start)) & (d.date <= pd.Timestamp(end)), "dividend"].sum()
        gross = (b.close - a.close + dividends) / a.close
        nav_component = (b.unit_nav - a.unit_nav + dividends) / a.close
        spread_component = (b.spread_cny_per_share - a.spread_cny_per_share) / a.close
        assert abs(gross - nav_component - spread_component) < 1e-12
        segments.append({"start": start, "end": end, "label": label, "start_price": a.close, "end_price": b.close,
                         "start_nav": a.unit_nav, "end_nav": b.unit_nav, "start_premium": a.premium, "end_premium": b.premium,
                         "etf_gross_return": gross, "nav_own_total_return": (b.unit_nav - a.unit_nav + dividends) / a.unit_nav,
                         "nav_component_of_price_return": nav_component, "spread_component_of_price_return": spread_component,
                         "index_price_return": b.index_close / a.index_close - 1, "dividend_per_share": dividends,
                         "price_change": b.close - a.close, "nav_change": b.unit_nav - a.unit_nav,
                         "spread_change": b.spread_cny_per_share - a.spread_cny_per_share})
    save("fixed_segments.json", segments)

    t = pd.read_csv(ROOT / protocol["saved_trade_source"])
    trade = t.loc[t.entry_date.eq(protocol["saved_trade_entry"]) & t.exit_date.eq(protocol["saved_trade_exit"])].iloc[0]
    e, x = pd.Timestamp(trade.entry_date), pd.Timestamp(trade.exit_date)
    previous = m.loc[m.date < x, "date"].max()
    a, b = table.loc[e], table.loc[previous]
    quantity = int(trade.quantity)
    components = [
        ("入场成交至当日收盘", quantity * (a.close - trade.entry_price), "含入场执行滑点，无法仅用日线分离其中净值与溢价"),
        ("中段基金净值变化", quantity * (b.unit_nav - a.unit_nav), "入场日日终至退出前日日终的单位净值变动"),
        ("中段成交价与净值价差变化", quantity * (b.spread_cny_per_share - a.spread_cny_per_share), "相同两端日终，每份金额价差变动"),
        ("末日日终至退出成交", quantity * (trade.exit_price - b.close), "含退出执行滑点，无法仅用日线分离其中净值与溢价"),
        ("原周期分红权益", float(trade.dividend_cny), "复用原账户分红权益记录"),
        ("买卖佣金", -float(trade.entry_fee + trade.exit_fee), "原滑点已经在首尾成交价中，不重复扣除")]
    pieces = [{"component": name, "amount_cny": amount, "share_of_saved_net_profit": amount / trade.net_pnl, "meaning": meaning}
              for name, amount, meaning in components]
    reproduced = sum(r["amount_cny"] for r in pieces)
    assert abs(reproduced - trade.net_pnl) < 1e-7
    assert abs(quantity * (trade.exit_price - trade.entry_price) + trade.dividend_cny - trade.entry_fee - trade.exit_fee - trade.net_pnl) < 1e-7
    normalized = read(ROOT / "reports/research/510300_annual_five_or_edge_v1/result.json")
    total_profit = float(t.net_pnl.sum())
    bridge = {"entry_date": e, "exit_date": x, "last_close_date": previous, "entry_fill": trade.entry_price, "exit_fill": trade.exit_price,
              "entry_close": a.close, "last_close": b.close, "entry_close_nav": a.unit_nav, "last_close_nav": b.unit_nav,
              "quantity": quantity, "saved_profit_cny": trade.net_pnl, "reproduced_profit_cny": reproduced,
              "components": pieces, "all_saved_cycles": len(t), "all_saved_cycle_profit_cny": total_profit,
              "selected_cycle_share_of_all_cycle_profit": trade.net_pnl / total_profit,
              "selected_by_known_concentration": True, "new_trade_created": False,
              "source_account_exposure_near_full": True, "current_half_position_mandate_satisfied": False,
              "source_frequency_reassessment_status": normalized["status"]}
    save("saved_trade_bridge.json", bridge)
    decline = d.iloc[1:].copy()
    summary = {"trading_days": len(d), "adjacent_daily_intervals": len(d)-1,
               "premium_decline_intervals": int(decline.premium_declined.sum()),
               "premium_decline_and_price_up_intervals": int((decline.premium_declined & decline.price_up).sum()),
               "source_nav_values_agree": True,
               "formula_max_error": float((d.price_change-d.nav_change-d.spread_change).abs().max()),
               "saved_trade_profit_error_cny": abs(reproduced - trade.net_pnl),
               "fixed_segment_count": len(segments), "net_sharpe": None, "net_cagr": None,
               "new_accounts": 0, "new_candidates": 0, "goal_achieved": False}
    save("measurements.json", summary)
    save("input_sources.json", [{"path": p.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                                 for p in [MARKET, NAV, INDEX, ROOT / protocol["saved_trade_source"]]])
    for row in segments:
        print(row["label"],f"价格{row['etf_gross_return']:+.4%}",f"净值{row['nav_own_total_return']:+.4%}",
              f"指数价格{row['index_price_return']:+.4%}",f"溢价{row['start_premium']:+.4%}→{row['end_premium']:+.4%}",
              f"净值贡献{row['nav_component_of_price_return']:+.4%}",f"价差贡献{row['spread_component_of_price_return']:+.4%}")
    for row in pieces:
        print(row["component"],f"{row['amount_cny']:+.2f}元")
    print("既有净利润复现",round(reproduced,2),"元；新信号和新账户均为0。")


def finish():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    d = pd.read_parquet(OUT / "daily_layers.parquet")
    segments = read(OUT / "fixed_segments.json")
    bridge = read(OUT / "saved_trade_bridge.json")
    checks = read(OUT / "measurements.json")
    receipt = read(OUT / "source_receipt.json")
    assert d.dividend.eq(0).all(), "本图以无基金除息区间为前提"
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 10})
    fig = plt.figure(figsize=(15.7, 9.4), facecolor="#f8fafc")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.32, 1], height_ratios=[1.45, 1], hspace=.36, wspace=.38)
    upper, lower, right = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[:, 1])
    for ax in [upper, lower, right]:
        ax.set_facecolor("#f8fafc")
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        ax.grid(axis="x" if ax is right else "y", alpha=.16)
        ax.set_axisbelow(True)
    anchor = d.loc[d.date.eq(pd.Timestamp("2024-09-23"))].iloc[0]
    upper.plot(d.date, d.close / anchor.close * 100, color="#174b70", linewidth=2.1, label="510300成交收盘价")
    upper.plot(d.date, d.unit_nav / anchor.unit_nav * 100, color="#b47722", linewidth=1.8, label="基金单位净值")
    upper.plot(d.date, d.index_close / anchor.index_close * 100, color="#4f7370", linewidth=1.2, linestyle=":", label="沪深300价格指数")
    upper.set_title("同一市场环境中的三层价格", loc="left", fontsize=12, pad=12)
    upper.set_ylabel("2024/9/23收盘=100")
    upper.legend(loc="upper left", frameon=False, fontsize=9)
    lower.plot(d.date, d.premium * 100, color="#694480", linewidth=1.8, marker="o", markersize=2.4)
    lower.axhline(0, color="#8b95a1", linewidth=.7)
    lower.set_ylabel("收盘价相对日终净值溢价（%）")
    lower.set_title("溢价回落可以伴随价格上涨，也可以伴随下跌", loc="left", fontsize=11, pad=12)
    for ax in [upper, lower]:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        for day in ["2024-09-23", "2024-09-30", "2024-10-08"]:
            ax.axvline(pd.Timestamp(day), color="#667686", linewidth=.7, linestyle="--", alpha=.45)
    for day, offset in [("2024-09-30", (-47, 15)), ("2024-10-08", (10, 11))]:
        r = d.loc[d.date.eq(pd.Timestamp(day))].iloc[0]
        lower.annotate(f"{pd.Timestamp(day):%m/%d}  {r.premium:.2%}", (r.date, r.premium * 100), xytext=offset,
                       textcoords="offset points", fontsize=9, color="#694480")
    lower.set_ylim(min(-.65, d.premium.min()*100-.25), 4.1)
    components = bridge["components"]
    labels = ["入场段（含滑点）", "中段基金净值变化", "中段价格与净值价差", "退出段（含滑点）", "分红权益", "买卖佣金", "原周期净利润"]
    values = [r["amount_cny"] / 1000 for r in components] + [bridge["saved_profit_cny"] / 1000]
    yy = np.arange(len(values))
    colors = ["#ad4d48" if v < 0 else "#0b766e" for v in values[:-1]] + ["#174b70"]
    right.barh(yy, values, height=.58, color=colors)
    right.invert_yaxis()
    right.axvline(0, color="#718091", linewidth=.7)
    right.set_yticks(yy, labels, fontsize=9)
    right.set_xlim(-13, 57)
    for y, v in zip(yy, values):
        right.text(v + .55 if v >= 0 else .6, y, f"{v * 1000:+,.0f}元", va="center", ha="left", fontsize=9)
    right.set_title("原最大盈利周期的金额桥接\n2024/9/25—10/9，原份额53,300份", loc="left", fontsize=12, pad=18)
    right.set_xlabel("贡献金额（千元）；首尾段不能识别同步净值")
    fig.suptitle("指数行情、基金净值与ETF成交价，分别贡献了什么", x=.07, ha="left", fontsize=18, y=.97, color="#173348")
    fig.text(.07, .915, "固定2024年9—10月全部37个交易日｜已观察历史｜没有新增信号或账户", fontsize=11, color="#5b6673")
    fig.text(.07, .04, "日终净值是事后归因资料，不是当日收盘前可用报价。右图复用旧账户近满仓周期，不能作为当前50%仓位约束的结果。\n净值与价差分解是会计恒等式；没有识别单项政策、套利约束或某类投资者的因果份额。", fontsize=10, color="#5b6673")
    fig.subplots_adjust(left=.075, right=.97, top=.81, bottom=.15)
    figure = OUT / "指数净值与ETF成交价_历史分解.png"
    fig.savefig(figure, dpi=155, facecolor=fig.get_facecolor())
    plt.close(fig)

    holiday = segments[2]
    later = segments[3]
    report = OUT / "历史发现_指数净值与ETF交易价格.md"
    lines = [
        "# 指数、基金净值与ETF成交价：原大额盈利来自哪里", "",
        "本轮得到两项可直接使用的历史发现：2024年9月30日至10月8日，ETF溢价回落伴随成交价上涨，原因在算术上是净值上涨得更快；而10月后续的溢价回落伴随净值和成交价一起下降。与此同时，原日线模式中占绝大多数净利润的那笔交易，主要正贡献来自基金净值上升，价格偏离净值的扩大只解释较小部分。", "",
        "这把宏观共同重新定价与ETF本身的交易价格层分开。它没有找到一个新的溢价交易规则，也没有恢复旧日线模式。完整账户成本后夏普1.2仍未实现。", "",
        "## 一、固定两个自然月，保留四个阶段", "",
        "研究固定2024年9月2日至10月31日全部37个交易日。下表分界为政策前、9月政策后至月末、假期前后及10月后续，没有搜索收益最好的转折点。9月行情及原最大交易结果此前已经见过，本轮属于已知结果后的机制诊断。", "",
        "| 阶段（收盘至收盘） | ETF价格回报 | 基金净值回报 | 沪深300价格指数回报 | 期初溢价→期末溢价 |",
        "|---|---:|---:|---:|---:|"]
    for r in segments:
        lines.append(f"| {r['start']}—{r['end']}，{r['label']} | {r['etf_gross_return']:+.2%} | {r['nav_own_total_return']:+.2%} | {r['index_price_return']:+.2%} | {r['start_premium']:+.2%}→{r['end_premium']:+.2%} |")
    lines += ["", "各列各用自己的期初价格或净值，不能直接相减当作贡献。本区间无510300现金分红；000300为价格指数，与基金净值仅作同底层走势旁证，成分分红、费用和跟踪差异并未被假设消失。", "",
              "## 二、同样溢价下降，对价格方向的含义可以不同", "",
              "设每份ETF的成交价为P、单位净值为N，金额价差S=P−N。价差变化满足ΔS=ΔP−ΔN，溢价率另为P/N−1。两者必须分别记录。", "",
              f"9月30日至10月8日，成交价从{holiday['start_price']:.3f}升至{holiday['end_price']:.3f}元，增加{holiday['price_change']:.4f}元；单位净值从{holiday['start_nav']:.4f}升至{holiday['end_nav']:.4f}元，增加{holiday['nav_change']:.4f}元。净值增加更多，所以每份金额价差缩小{abs(holiday['spread_change']):.4f}元，溢价率也下降；持有者的价格回报仍为{holiday['etf_gross_return']:+.2%}。", "",
              f"10月8日至31日，基金净值回报{later['nav_own_total_return']:+.2%}，ETF价格回报{later['etf_gross_return']:+.2%}。此时是底层净值下跌，又叠加价差收窄。只写‘溢价回落’，会把两段经济和交易含义不同的过程混在一起。", "",
              f"在整个固定窗口的{checks['adjacent_daily_intervals']}个相邻交易日间隔中，有{checks['premium_decline_intervals']}次溢价率下降，其中{checks['premium_decline_and_price_up_intervals']}次成交价上涨。这只是路径计数，各日不独立，没有据此计算一个可以推广的胜率。", "",
              "把两项贡献统一除以期初ETF价格，可以精确相加：ETF价格回报=净值变化贡献+金额价差变化贡献。本区间无分红；含分红时把同一份分红权益计入净值项即可。", "",
              "| 固定阶段 | 净值变化对ETF价格回报的贡献 | 金额价差变化的贡献 | 合计ETF价格回报 |",
              "|---|---:|---:|---:|"]
    for r in segments:
        lines.append(f"| {r['label']} | {r['nav_component_of_price_return']*100:+.3f}个百分点 | {r['spread_component_of_price_return']*100:+.3f}个百分点 | {r['etf_gross_return']:+.3%} |")
    lines += ["", "9月23日至30日的价差扩大此前已有研究，本轮不将它重复算作新发现。新增的是之后两种收敛路径，以及与原完整交易的精确连接。", "",
              "## 三、为什么价格可以偏离净值", "",
              f"当年的基金文件提供了机制边界。2024年6月1日招募说明书第3页区分二级买入与一级申购的当日转卖资格；第45页载明最低申购赎回单位90万份；第100页说明成分股涨停、停牌、现金替代限制和退补价格的不确定性，可能影响申赎与ETF折溢价。已经查看这三页完整渲染图。[交易所披露的基金原文件]({receipt['url']})。", "",
              "这意味着套利压缩价差需要获取篮子、资金和申赎通道，并承担执行与替代结算的不确定性；它不是无成本、无容量限制的即时等式。90万份是该份说明书的规则，并非本研究重新认证了9—10月每日清单。本轮没有取得当时完整申赎清单、逐笔订单或通道失败记录，因而不能断言9月30日溢价就是申购失败造成的。", "",
              "保留两种竞争解释：ETF交易需求集中、套利执行来不及吸收；或者ETF价格先反映对底层随后重估的预期、底层报价再追赶。这两种解释都可能与观察一致。假期后净值追上来，是已发生的结果，不能倒推出此前溢价买入者必然拥有更准的信息。基金日终净值、盘中IOPV和真实即时组合价值也不能混称。", "",
              "对本项目的20万元、只持有510300和现金账户，一级申赎套利并不是当前策略。这里只把申赎机制作为ETF交易价格为什么可能偏离指数篮子的解释。", "",
              "## 四、原最大盈利周期的利润到底来自哪里", "",
              f"原日线顺序模式在2020—2026年保存了{bridge['all_saved_cycles']}个完整周期，净利润合计{bridge['all_saved_cycle_profit_cny']:,.2f}元。其中2024年9月25日至10月9日这笔净利润{bridge['saved_profit_cny']:,.2f}元，占{bridge['selected_cycle_share_of_all_cycle_profit']:.2%}。这项集中度是既有结论，本轮只解释其来源。原份额{bridge['quantity']:,}份、买入成交价{bridge['entry_fill']:.3f}、卖出成交价{bridge['exit_fill']:.3f}均保持原样。", "",
              "由于没有可靠的同步开盘净值，不能把开盘交易硬拆成完整的指数与溢价两项。本轮选择两个可对齐的日终——9月25日与10月8日——拆中间段，首尾价格变化单列：", "",
              "| 原交易组成 | 金额 | 含义 |", "|---|---:|---|"]
    for r in bridge["components"]:
        lines.append(f"| {r['component']} | {r['amount_cny']:+,.2f}元 | {r['meaning']} |")
    lines += [f"| 合计 | {bridge['reproduced_profit_cny']:+,.2f}元 | 与原交易净利润一致 |", "",
              "中段基金净值变化贡献46,227.09元，金额价差扩大贡献3,768.31元；入场段、退出段和费用抵消了一部分。这说明该笔利润主要依靠整体市场对应的基金资产重估；仅压缩ETF价差，并不能复制这次盈利。它没有证明交易规则提前识别了政策，也没有识别哪项政策造成多少净值涨幅。", "",
              "原账户当时接近满仓，当前研究的最高仓位为50%。这里只复用旧股数解释旧利润，没有把这笔利润冒充当前风险合同的结果，也没有把它从原账本删除后重算更有利的策略。", "",
              "## 五、对夏普目标的实际推进", "",
              "这次排清了一个具体问题：旧策略的大额利润不能主要归为ETF折溢价修复，其核心仍是捕捉到一段少见的指数整体重估；能否重复捕捉、同时控制其他阶段亏损，仍是待解决的收益优势问题。把‘价格偏离净值’加入指标列表本身不能解决它。", "",
              "后续对指数因子的研究应区分共同资产重估与ETF自身交易价格变化。研究某个因子调整的原因时，要分别回答它改变的是资产现金流或风险折价、即时交易需求，还是套利执行条件；再用当时已经可得的信息判断，不能用之后的净值补写当时的信号。", "",
              "本分支完成，不新增折价买入、溢价卖出或旧形态过滤器。没有新账户、没有参数搜索、没有前瞻任务，完整账户净夏普与年化收益均未计算，目标未达成。必要验证只保留37个交易日双源净值一致、固定阶段金额恒等及原交易利润桥接。", "",
              "## 资料与复现", "",
              f"- {link('固定研究范围', OUT / 'protocol.json')}；{link('研究脚本', ROOT / 'research/historical_index_etf_price_layer_v1.py')}，运行compute后运行finish。",
              f"- {link('37日完整价格层', OUT / 'daily_layers.json')}；{link('四个固定阶段', OUT / 'fixed_segments.json')}；{link('原交易利润组成', OUT / 'saved_trade_bridge.json')}。",
              f"- {link('原基金文件与获取记录', OUT / 'source_receipt.json')}；{link('原始数据路径', OUT / 'input_sources.json')}；{link('必要核对结果', OUT / 'measurements.json')}。",
              f"- {link('历史分解图', figure)}。", ""]
    report.write_text("\n".join(lines), encoding="utf-8")
    save("result.json", {"study_id": "510300_HISTORICAL_INDEX_ETF_PRICE_LAYER_V1", "completed_at": datetime.now(TZ).isoformat(),
                         "status": "COMPLETED_HISTORICAL_RETURN_SOURCE_DIAGNOSTIC_NO_NEW_SIGNAL",
                         "classification": "PROGRESS_INDEX_NAV_ETF_PRICE_AND_SAVED_PROFIT_DISTINGUISHED",
                         "report": report.relative_to(ROOT).as_posix(), "figure": figure.relative_to(ROOT).as_posix(),
                         "trading_days": len(d), "fixed_segments": len(segments), "saved_cycles_replayed": 0,
                         "one_saved_profit_reconciled": True, "new_primary_documents": 1, "new_candidates": 0,
                         "new_full_accounts": 0, "new_prospective_tasks": 0, "net_sharpe": None, "net_cagr": None,
                         "goal_achieved": False, "orders_authorized": False, "causal_share_identified": False,
                         "independent_validation": False,
                         "discovery": "2024年9月30日至10月8日溢价由3.20%降至1.61%，ETF仍涨4.23%，净值涨5.86%；10月后续净值与价格共同回落。原41,781.38元大额净利的中段净值贡献46,227.09元，价差扩大贡献3,768.31元，主要收益来自整体重估而非折溢价修复。",
                         "next_historical_question": None,
                         "continuation_boundary": "本分支完成；不从已观察的溢价或大额交易倒推新过滤器，不重启旧折价和日线模式。",
                         "report_status": "PENDING_REVIEW", "figure_visually_reviewed": False,
                         "limitations": ["选择最大盈利周期为事后归因", "同日净值不是收盘前交易输入", "首尾交易段缺同步净值，保持未拆", "缺当天申赎清单与订单，不能识别价差因果", "原账户近满仓不符合当前半仓风险合同"]})
    print("历史价格层、原交易利润桥接、报告和图已保存；没有生成新策略。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["compute", "finish"])
    args = parser.parse_args()
    {"compute": compute, "finish": finish}[args.action]()


if __name__ == "__main__":
    main()
