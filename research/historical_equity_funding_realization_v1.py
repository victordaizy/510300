"""将股票融资工具的历史兑现、公开时间与510300剩余收益连接。"""

from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from historical_price_gap_causes_v1 import round_trip

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_equity_funding_realization_v1"
OLD = ROOT / "reports/research/510300_policy_information_clock_v1"
REPORT = OUT / "历史发现_融资额度如何变成股票购买.md"
NEXT = "沿已知首批参与机构的历史原披露，查找互换操作后的实际融资成本、股票投资及对冲安排，判断资金获取能力如何转成净股票风险敞口；不把未用额度当成下一周必买资金。"


def save(path, value):
    def convert(item):
        if isinstance(item, dict):
            return {str(k): convert(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(v) for v in item]
        if isinstance(item, (pd.Timestamp, datetime)):
            return item.isoformat()
        if isinstance(item, np.generic):
            return convert(item.item())
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    path.write_text(json.dumps(convert(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def relative(path):
    return path.relative_to(ROOT).as_posix()


def link(path, text):
    return f"[{text}](<{path.as_posix()}>)"


def pct(value):
    return f"{value * 100:+.2f}%"


def source_map():
    sources = {}
    for key in ["joint_20240924_csrc", "sfisf_20241018_csrc", "sfisf_20241021_pboc", "sfisf_20241231_csrc", "sfisf_20250102_pboc", "joint_20250507_csrc"]:
        receipt = json.loads((OLD / "receipts" / (key + ".json")).read_text(encoding="utf-8"))
        path = OLD / receipt["raw_path"]
        sources[key] = {"url": receipt["url"], "path": relative(path), "kind": "REUSED_OFFICIAL_HTML"}
    for key in ["20241010_央行原公告新华社转载", "2025Q1_货币政策执行报告", "2025Q2_货币政策执行报告", "2025前三季度央行大事记"]:
        receipt = json.loads((OUT / "sources" / (key + ".receipt.json")).read_text(encoding="utf-8"))
        sources[key] = {"url": receipt["url"], "path": receipt["path"], "kind": "NEW_PRIMARY_DOCUMENT_CURRENT_COPY"}
    sources["20250123_工具实际使用"] = {
        "url": "https://www.csrc.gov.cn/csrc/c106311/c7535350/content.shtml",
        "path": relative(OUT / "sources/新增官方公告网页读取.json"), "kind": "OFFICIAL_WEB_TOOL_EXTRACT",
    }
    return sources


def nodes():
    rows = [
        ("2024-09-24", "11:42:50", "SFISF_AND_LOAN", "创设方向", "互换5000亿、再贷款3000亿", "joint_20240924_csrc", None,
         "新工具及计划额度；操作、融资和买入尚未发生确认。", "同场还有降准降息、房地产政策；随后9月26日政治局会议等共同影响市场。"),
        ("2024-10-10", "08:27:38", "SFISF", "开放申请", "互换工具正式接受申报", "20241010_央行原公告新华社转载", None,
         "新增可申请安排，5000亿元额度已于9月24日宣布。", "新华社为央行原公告当日转载；这里用转载时刻，不冒充最早首发分钟。"),
        ("2024-10-18", "23:59:59", "SFISF", "操作细则", "20家获准、申请超2000亿", "sfisf_20241018_csrc", None,
         "期限、质押品折扣与用途明确；获准和申请仍不是中标或买入。", "同时公布回购增持再贷款实施安排，属于混合政策节点。"),
        ("2024-10-21", "17:00:30", "SFISF", "互换操作", "首笔500亿，互换费率20bp", "sfisf_20241021_pboc", "2024-10-21",
         "当次互换操作确定，股票的实际逐日购买未知。", "操作规模不等于当日股票净买入；融资费率不止互换环节费用。"),
        ("2024-12-31", "23:59:59", "SFISF", "投放进度及扩围", "首批投放超90%，备选40家", "sfisf_20241231_csrc", "2024-12-31",
         "首批500亿操作的投放进度，以及备选机构扩围；两种信息同时出现。", "90%的基数为500亿首批操作，不是5000亿总额度；原文投放不细分每日净买入。"),
        ("2025-01-02", "17:00:30", "SFISF", "互换操作", "第二笔550亿，互换费率10bp", "sfisf_20250102_pboc", "2025-01-02",
         "新增第二次操作金额与中标费率，累计操作1050亿。", "2024年12月31日已预告第二次操作；未取得事前金额共识。"),
        ("2025-01-23", "23:59:59", "SFISF", "实际使用说明", "首笔500亿已用于融资增持", "20250123_工具实际使用", "2025-01-23",
         "确认首笔已用于融资增持股票，第二笔550亿已可供随时融资增持。", "同场中长期资金入市方案有其他措施；没有逐日交易、目标股票或实际全部成交日。"),
        ("2025-05-07", "10:50:54", "SFISF_AND_LOAN", "约束优化及原因", "8000亿合用、说明逆周期使用", "joint_20250507_csrc", None,
         "两工具已有额度合用；原文解释参与者自主择时及压力期使用增加。", "此前9:20的概要已公布额度合用，本节点待完整工具解释结束；还有降息等其他政策及经贸会谈消息。"),
        ("2025-05-09", "23:59:59", "SFISF", "实际投资累计量", "3月末支持增持约820亿", "2025Q1_货币政策执行报告", "2025-03-31",
         "28家机构、累计互换1050亿及支持增持约820亿的季度统计。", "报告比统计期末晚39天，不能把820亿当5月9日新买入或待买入。未证明该数值在全市场首次公开。"),
        ("2025-08-15", "23:59:59", "REPURCHASE_LOAN", "银行实际发放", "6月末签约3100亿、发放900亿", "2025Q2_货币政策执行报告", "2025-06-30",
         "拟申请上限超3200亿、贷款合同约3100亿、已发放约900亿，三者口径明确。", "贷款发放不是股票实际购买；报告比统计期末晚46天，未证明是统计数值首次公开。"),
    ]
    fields = ["publication_date", "time_upper", "channel", "stage", "short_title", "source_key", "statistics_cutoff", "information", "limitation"]
    result = []
    for number, row in enumerate(rows, 1):
        item = dict(zip(fields, row))
        item["node_id"] = f"N{number:02d}"
        item["available_upper"] = f"{item['publication_date']}T{item['time_upper']}+08:00"
        item["time_precision"] = "DATE_END_BOUND" if item["time_upper"] == "23:59:59" else "SOURCE_TIMESTAMP_UPPER_BOUND"
        item["publication_lag_calendar_days"] = (pd.Timestamp(item["publication_date"]) - pd.Timestamp(item["statistics_cutoff"])).days if item["statistics_cutoff"] else None
        item["independent_event"] = False
        item["market_consensus_surprise"] = None
        result.append(item)
    return result


def quantities():
    return [
        {"as_known_on": "2024-09-24", "channel": "SFISF", "metric": "初始工具额度", "amount_100m_cny": 5000, "qualifier": "计划规模"},
        {"as_known_on": "2024-09-24", "channel": "REPURCHASE_LOAN", "metric": "初始再贷款额度", "amount_100m_cny": 3000, "qualifier": "计划规模"},
        {"as_known_on": "2024-10-18", "channel": "SFISF", "metric": "首批申请", "amount_100m_cny": 2000, "qualifier": "大于"},
        {"as_known_on": "2024-10-21", "channel": "SFISF", "metric": "第一笔互换操作", "amount_100m_cny": 500, "qualifier": "当次操作"},
        {"as_known_on": "2024-12-31", "channel": "SFISF", "metric": "第一笔操作实际投放比例", "amount_100m_cny": None, "percent": 90, "qualifier": "大于；分母为500亿元，非5000亿元"},
        {"as_known_on": "2025-01-02", "channel": "SFISF", "metric": "第二笔互换操作", "amount_100m_cny": 550, "qualifier": "当次操作"},
        {"as_known_on": "2025-01-23", "channel": "SFISF", "metric": "第一笔已用于融资增持股票", "amount_100m_cny": 500, "qualifier": "累计；不能再次与第一笔操作相加"},
        {"as_known_on": "2025-05-07", "channel": "COMBINED_LIMIT", "metric": "两工具额度合用", "amount_100m_cny": 8000, "qualifier": "既有5000加3000合用，非新增8000"},
        {"as_known_on": "2025-05-09", "statistics_cutoff": "2025-03-31", "channel": "SFISF", "metric": "累计互换操作", "amount_100m_cny": 1050, "qualifier": "累计；不是当季度新操作"},
        {"as_known_on": "2025-05-09", "statistics_cutoff": "2025-03-31", "channel": "SFISF", "metric": "支持28家机构增持股票和ETF", "amount_100m_cny": 820, "qualifier": "约；累计描述，不当成报告当日买入"},
        {"as_known_on": "2025-08-15", "statistics_cutoff": "2025-06-30", "channel": "REPURCHASE_LOAN", "metric": "披露拟申请贷款上限", "amount_100m_cny": 3200, "qualifier": "大于"},
        {"as_known_on": "2025-08-15", "statistics_cutoff": "2025-06-30", "channel": "REPURCHASE_LOAN", "metric": "已签贷款合同", "amount_100m_cny": 3100, "qualifier": "约"},
        {"as_known_on": "2025-08-15", "statistics_cutoff": "2025-06-30", "channel": "REPURCHASE_LOAN", "metric": "贷款已发放", "amount_100m_cny": 900, "qualifier": "约；不是股票购买额"},
    ]


def verify_facts(sources):
    requirements = {
        "sfisf_20241018_csrc": ["互换期限1年", "折扣率", "股票、股票ETF的投资和做市", "首批申请额度已超2000亿元"],
        "sfisf_20241021_pboc": ["500亿元", "中标费率为20bp"],
        "sfisf_20241231_csrc": ["实际投放超过90%", "40家备选机构池"],
        "sfisf_20250102_pboc": ["550亿元", "中标费率为10bp"],
        "joint_20250507_csrc": ["自主决策购买股票的时机和规模", "互换便利使用量明显增加", "总额度8000亿元合并使用"],
        "2025Q1_货币政策执行报告": ["支持28家机构增持股票和ETF约820亿元", "累计操作1050亿元"],
        "2025Q2_货币政策执行报告": ["合同金额约3100亿元", "已发放约900亿元"],
        "20250123_工具实际使用": ["500亿元，已经全部用于融资增持股票", "550亿元，行业机构已经可以随时用于融资增持股票"],
        "20241010_央行原公告新华社转载": ["08:27:38", "即日起，接受符合条件"],
        "2025前三季度央行大事记": ["5月9日，中国人民银行发布《2025年第一季度中国货币政策执行报告》", "8月15日，中国人民银行发布《2025年第二季度中国货币政策执行报告》"],
    }
    for key, snippets in requirements.items():
        path = ROOT / sources[key]["path"]
        if path.suffix == ".pdf":
            text = path.with_suffix(".txt").read_text(encoding="utf-8")
        elif path.suffix == ".html":
            text = BeautifulSoup(path.read_bytes(), "html.parser").get_text(" ", strip=True)
        else:
            text = str(json.loads(path.read_text(encoding="utf-8")))
        compact = "".join(text.split())
        for snippet in snippets:
            if "".join(snippet.split()) not in compact:
                raise ValueError(f"来源未覆盖事实：{key}：{snippet}")
    return {"reviewed_source_keys": list(requirements), "required_snippets": requirements,
            "is_causal_effect_verification": False}


def compute(rows, costs):
    market = pd.read_parquet(ROOT / "reports/research/510300_macro_dynamic_reframe_v1/inputs/market.parquet").sort_values("date").reset_index(drop=True)
    open_times = pd.DatetimeIndex(market.date).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=30)
    dividends = pd.read_csv(ROOT / "data/reference/510300_dividends.csv")
    dividends = dividends[dividends.symbol.eq("510300.SH")].copy()
    dividends["record_date"] = pd.to_datetime(dividends.record_date)
    anchor = market[market.date.eq(pd.Timestamp("2024-09-23"))].iloc[0]
    records = []
    for item in rows:
        entry_i = open_times.searchsorted(pd.Timestamp(item["available_upper"]), side="right")
        entry, previous = market.iloc[entry_i], market.iloc[entry_i - 1]
        assert open_times[entry_i] > pd.Timestamp(item["available_upper"])
        record = {**item, "entry_date": entry.date, "entry_at": open_times[entry_i], "entry_open": entry.open,
                  "gap_return": (entry.open + entry.dividend) / previous.close - 1,
                  "prior_since_20240923_total_return": previous.wealth / anchor.wealth * (entry.open + entry.dividend) / previous.close - 1}
        for horizon in [5, 20]:
            end = market.iloc[entry_i + horizon]
            entitlement = dividends.loc[dividends.record_date.ge(entry.date) & dividends.record_date.lt(end.date), "cash_dividend_per_share"].sum()
            trade = round_trip(entry.open, end.open, entitlement, costs)
            record.update({f"exit{horizon}": end.date, f"exit{horizon}_open": end.open,
                           f"gross_return{horizon}": (end.open + entitlement) / entry.open - 1,
                           f"net_return{horizon}": trade["net_return"], f"paid{horizon}": trade["paid_cny"],
                           f"pnl{horizon}": trade["net_pnl_cny"], f"shares{horizon}": trade["shares"],
                           f"commissions{horizon}": trade["commissions_cny"], f"buy_fill{horizon}": trade["buy_price"],
                           f"sell_fill{horizon}": trade["sell_price"], f"dividend_per_share{horizon}": float(entitlement)})
            identity = trade["shares"] * (trade["sell_price"] - trade["buy_price"] + entitlement) - trade["commissions_cny"]
            assert abs(identity - trade["net_pnl_cny"]) < 1e-7
            assert abs(trade["net_pnl_cny"] / trade["paid_cny"] - trade["net_return"]) < 1e-12
        record["overlaps_previous_20d_example"] = any(entry.date < other["exit20"] for other in records)
        records.append(record)
    frame = pd.DataFrame(records)
    frame.to_parquet(OUT / "十个披露节点与可成交收益.parquet", index=False)
    frame.to_csv(OUT / "十个披露节点与可成交收益.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    return frame, market


def figure(events, market):
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(2, 1, figsize=(13.8, 8.5), gridspec_kw={"height_ratios": [1, 1.3]})
    fig.patch.set_facecolor("#f6f3ec")
    window = market[market.date.between("2024-09-23", "2025-09-15")]
    relative_wealth = (window.wealth / window.iloc[0].wealth - 1) * 100
    axes[0].plot(window.date, relative_wealth, color="#286f74", linewidth=1.6)
    axes[0].set(ylabel="累计含息变化（%）", title="510300历史路径：行情可以先于融资使用披露发生")
    labels = {"N01": "首次宣布", "N04": "首笔操作", "N07": "披露首笔已使用", "N09": "披露3月末增持量", "N10": "披露6月末放款量"}
    for index, row in events.iterrows():
        if row.node_id in labels:
            disclosure_date = pd.Timestamp(row.publication_date)
            value = market.loc[market.date.eq(disclosure_date), "wealth"].iloc[0]
            y = (value / window.iloc[0].wealth - 1) * 100
            axes[0].scatter([disclosure_date], [y], s=25, color="#be8843", zorder=3)
            axes[0].annotate(labels[row.node_id], (disclosure_date, y), xytext=(6, -24 if row.node_id == "N09" else 15), textcoords="offset points", fontsize=9)
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    positions = np.arange(len(events))
    axes[1].scatter(events.net_return5 * 100, positions - .13, s=42, color="#286f74", label="入场后5交易日")
    axes[1].scatter(events.net_return20 * 100, positions + .13, s=42, marker="s", color="#be8843", label="入场后20交易日")
    axes[1].set_yticks(positions, [f"{r.publication_date}  {r.short_title}" for r in events.itertuples()])
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="#9c9c94", linewidth=.8)
    axes[1].set(xlabel="费用后事件收益（%）", title="每个公开节点之后的剩余收益：保留全部十个节点，不按结果选日期")
    axes[1].legend(loc="lower right", frameon=False)
    for ax in axes:
        ax.set_facecolor("#fffdf8")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=.25)
        ax.set_axisbelow(True)
    fig.suptitle("股票融资工具：额度、实际使用与公开时点分开", fontsize=16)
    fig.text(.035, .023, "十个节点属于同一政策传导链，多个收益窗口重叠，不能视作十次独立试验或完整账户。\n上图按公开日期标注日末含息路径，不是工具的因果贡献；下图按消息后首个可用开盘、既定压力费用计算。", fontsize=9, color="#575c5b")
    fig.tight_layout(rect=[.01, .075, .99, .94])
    fig.savefig(OUT / "融资使用披露与行情先后.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def report(events, sources, result):
    q1 = sources["2025Q1_货币政策执行报告"]["url"]
    q2 = sources["2025Q2_货币政策执行报告"]["url"]
    lines = [
        "# 历史发现：融资额度如何变成股票购买", "",
        "2026年9月30日。只研究已发生的2024—2025年政策链及随后历史行情。复用原政策链和八家公司的融资条件诊断，新增央行实际投放证据，不重跑旧失败账户。", "",
        "**两项工具确实增加了特定主体的融资渠道，但额度、申请、互换操作、贷款发放和股票购买不是同一个数。实际使用又会对市场压力作出反应，且公开统计通常晚于交易发生。因而，先辨认钱处在哪个环节，再判断还有没有尚未完成的购买，比单看政策金额更有用。**", "",
        "本轮选定十个制度及实际使用披露节点。它们是同一政策链的不同阶段，不是十次独立政策试验；相关历史行情此前已被研究。节点和5日主窗口、20日背景窗口在本轮读取收益前已保存，没有按结果增删。", "",
        "**互换便利新增了什么。** 2024年10月10日原公告允许合格机构以债券、股票ETF、沪深300成分股等抵押，换入国债、央票等高流动性资产，开始接受申报。它改善融资可得性，后续仍需要融资和投资操作。[央行原公告当日转载](" + sources["20241010_央行原公告新华社转载"]["url"] + ")。", "",
        "10月18日细则进一步明确质押品折扣、1年互换期限及股票、股票ETF投资和做市用途。21日首笔操作500亿元，中标互换费率20个基点；2025年1月2日第二笔550亿元，费率10个基点。这里的费率属于互换环节，不能直接解释为全部资金成本；用途也不只是一笔永久持有的方向性买单。[操作细则](" + sources["sfisf_20241018_csrc"]["url"] + ")、[首笔操作](" + sources["sfisf_20241021_pboc"]["url"] + ")、[第二笔操作](" + sources["sfisf_20250102_pboc"]["url"] + ")。", "",
        "| 公开信息 | 实际口径 | 本轮能确认的量级 |",
        "|---|---|---:|",
        "| 2024-09-24宣布 | 互换便利初始额度 | 5000亿元 |",
        "| 2024-10-18 | 首批申请 | 超过2000亿元 |",
        "| 2024-10-21、2025-01-02 | 两笔互换操作 | 500＋550＝1050亿元 |",
        "| 2024-12-31 | 首笔500亿元的实际投放比例 | 超过90% |",
        "| 2025-01-23 | 首笔500亿元的使用说明 | 已用于融资增持股票 |",
        "| 2025-05-09报告，统计至3月末 | 支持28家机构增持股票和ETF | 约820亿元 |", "",
        "这些金额不能相加：820亿元是对工具使用效果的累计描述，与前面操作金额重叠。820/1050约为78.1%，只表示两个披露量的比值，既不是严格同口径的资金转化率，也不能把差额230亿元指定为随后必买股票的现金。具体每日买入、交易对象、持仓替代与其他资产卖出没有在该段报告中给出。[2025年一季度央行报告，正文第9页、PDF第15页](" + q1 + ")。", "",
        "1月23日原文已经区分：首笔500亿元已用于融资增持股票，第二笔550亿元已可随时用于融资增持。资金可用与实际已经使用是两种状态。这份公开说明也不能提供首笔资金逐日成交的时间表。[发布会原文](" + sources["20250123_工具实际使用"]["url"] + ")。", "",
        "**回购增持贷款又多了一层银行放款环节。** 截至2025年6月末，披露拟申请金额上限超过3200亿元，银行已签贷款合同约3100亿元，已发放约900亿元。900/3100约为29.0%，是同一统计时点两个金额的比例；已发放贷款仍不是已成交股票金额，更不能将合同与发放金额相加。[2025年二季度央行报告，正文第15页、PDF第21页](" + q2 + ")。", "",
        "已保存的八家公司原公告提供了可观察的执行约束：3家尚待股东大会审议，4家是既有已批准回购计划增加融资，1家同次披露董事会方案和融资。这个有限样本解释了为什么融资可得不等于即时新购买，但不能用它归因整个3100亿元合同与900亿元放款之间的差额。" + link(ROOT / "reports/research/510300_funded_repurchase_transmission_v1/八家公司融资与执行对照.md", "既有八家公司原文诊断") + "。", "",
        "**再往上追一层：资金使用为什么会增加。** 5月7日央行说明，购买时机和规模由参与主体自主决定，并列举2024年11月、2025年元旦前后和4月初外部冲击期间工具使用增加。原文将这解释为逆周期托底机制。本轮据此保留两个方向：资金买入可能缓冲卖压；市场承压也可能促使机构使用资金。缺少反事实时，下跌不能直接证明工具无效，反弹也不能直接算成工具全部功劳。[5月7日工具解释原文](" + sources["joint_20250507_csrc"]["url"] + ")。", "",
        "这也说明额度为什么不是直接的日频领先因子：额度解决可不可以融资，参与者还需要决定股票是否值得买、何时买、买多少。公开文件提供这一机制解释，没有提供可逐日重建的全部机构决策。", "",
        "**统计发生在前，报告披露在后。** 一季度报告5月9日披露3月31日统计，滞后39个自然日；二季度报告8月15日披露6月30日统计，滞后46日。日期由报告封面及央行大事记对应确认，但本轮未证明这些数值在全市场的最早披露时刻。严禁把820亿元放到5月9日当新增买入，也不能把900亿元放到8月15日当日买盘。[央行历史发布日期记录](" + sources["2025前三季度央行大事记"]["url"] + ")。", "",
        "**消息公布后可交易的空间。** 根据明确来源时刻选择其后的首个A股开盘；只有日期的来源用当日日末。10月10日央行原公告的新华社转载标注08:27:38，因此允许当日开盘；盘中消息采用下一开盘。没有把公告前跳涨或无法核对的盘中价格作为收益。", "",
        "| 来源日期 | 本次公开信息 | 入场开盘日 | 此前相对9月23日已发生含息变化 | 入场后5日净收益 | 入场后20日净收益 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in events.itertuples():
        lines.append(f"| {row.publication_date} | {row.short_title} | {row.entry_date:%Y-%m-%d} | {pct(row.prior_since_20240923_total_return)} | {pct(row.net_return5)} | {pct(row.net_return20)} |")
    lines += [
        "", "表内‘此前变化’是市场已发生的历史路径，不是政策已定价百分比或该工具的因果贡献。十个节点中有" + str(int(events.overlaps_previous_20d_example.sum())) + "个与更早20日窗口重叠；因此不计算独立胜率，不拼接成账户。", "",
        "9月24日有其他货币、地产政策，12月31日进度披露伴随机构扩围，1月23日还有中长期资金入市方案。5月9日报告对应的下一开盘为5月12日，此前已有中美经贸会谈进展消息，随后还有联合声明。即便某一节点后上涨，也不能从这些混合信息中分离融资工具的单独效应。", "",
        "费用沿用每笔10万元预算、佣金每边0.04%且最低5元、滑点每边0.10%、0.001元报价档位和100份整手，并计入持有期间的分红权益。例子用实际开盘价确定预算内股数，不宣称开盘前能按该精确数量完成成交。它们是事件投入金额收益，不是20万元完整账户。", "",
        "核对范围仅包括原文金额和日期、消息与入场先后、固定5/20交易日终点及分红费用恒等。新增账户、拟合参数、前瞻卡均为0。旧政策账户压力夏普0.690、年化4.61%的失败结果保持不变，本轮也没有形成夏普1.2达标证据。", "",
        "下一历史问题：" + NEXT, "",
        link(OUT / "十个披露节点与可成交收益.csv", "十个节点明细") + "；" + link(OUT / "分环节金额口径.csv", "分环节金额口径") + "；" + link(OUT / "融资使用披露与行情先后.png", "时间与收益图") + "；" + link(OUT / "source_index.json", "原始来源路径") + "。", "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def update_state(result):
    path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update({"current_round": result["study_id"], "latest_progress_receipt": relative(OUT / "result.json"),
                 "last_research_result": "股票融资兑现与公开时间已连接：3月末互换累计1050亿元、支持增持约820亿元；6月末回购增持贷款合同约3100亿元、发放约900亿元，报告分别滞后39和46天。实际使用也会响应市场压力，不能把累计量变成即时买盘。完成十个相关节点固定窗口诊断，未建立独立因果买入优势。",
                 "last_source_result": "复用6份官方政策原文和八家公司诊断，新增2份央行报告、原公告转载、央行大事记及1份官方发布会网页读取。",
                 "latest_historical_diagnostic_at": result["recorded_at"], "latest_historical_report": relative(REPORT),
                 "latest_historical_equity_funding_realization": relative(OUT / "result.json"),
                 "latest_continuation_report": relative(REPORT), "latest_continuation_classification": result["classification"],
                 "current_driver_continuation_classification": result["classification"],
                 "current_driver_consecutive_blocked_goal_turns": 0,
                 "goal_status": "active", "goal_achieved": False, "next_research_question": NEXT})
    save(path, data)
    path = ROOT / "config/510300_historical_cause_discovery_v1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update({"latest_completed_study": relative(OUT / "result.json"), "latest_report": relative(REPORT), "updated_at": result["recorded_at"]})
    save(path, data)
    path = ROOT / "RESEARCH_STATUS.md"
    old = path.read_text(encoding="utf-8")
    marker = "<!-- HISTORICAL_EQUITY_FUNDING_REALIZATION_V1_20260930 -->"
    text = marker + "\n\n## 2026-09-30 股票融资兑现与披露滞后已分开\n\n"
    text += "新增央行2025年一、二季度报告及工具实际使用说明。一季度末互换累计操作1050亿元，支持28家机构增持股票及ETF约820亿元；二季度末回购增持贷款拟申请上限超3200亿元、合同约3100亿元、发放约900亿元。对应报告较统计期末晚39和46天，不能把累计数放到报告日当新买盘。各环节不能相加，金额差不能预设为未来必买资金。\n\n"
    text += "5月7日官方原文明确参与者自主择时，并说明压力期使用增加。因而资金对价格的作用与价格压力对用钱行为的作用并存，尚未分离因果效果。固定十个政策链节点完成消息后首个开盘的5/20日费用后收益说明，多个窗口重叠，非十次独立事件、非完整账户。新增账户、拟合参数及前瞻卡均0，夏普1.2目标未达成，旧失败不改。\n\n"
    text += "详见" + link(REPORT, "融资额度到股票购买的历史发现") + "和" + link(OUT / "十个披露节点与可成交收益.csv", "十个节点明细") + "。\n\n"
    if marker not in old:
        path.write_text(text + old, encoding="utf-8")


def main():
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    sources = source_map()
    verification = verify_facts(sources)
    rows = nodes()
    for item in rows:
        item.update({"source_url": sources[item["source_key"]]["url"], "source_path": sources[item["source_key"]]["path"]})
    save(OUT / "source_index.json", sources)
    save(OUT / "事实与信息时点.json", rows)
    save(OUT / "原文定位.json", verification)
    pd.DataFrame(quantities()).to_csv(OUT / "分环节金额口径.csv", index=False, encoding="utf-8-sig")
    events, market = compute(rows, protocol["costs"])
    result = {
        "study_id": protocol["study_id"], "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "HISTORICAL_FUNDING_USE_AND_PUBLICATION_LAG_DISTINGUISHED",
        "classification": "PROGRESS_HISTORICAL_FUNDING_REALIZATION_AND_ENDOGENOUS_USE",
        "previous_goal_turn_classification": protocol["previous_goal_turn_classification"],
        "nodes": len(events), "overlapping_20d_nodes": int(events.overlaps_previous_20d_example.sum()),
        "quantity_ratios": {"sfisf_supported_purchase_to_operation_amount": 820 / 1050,
                            "bank_disbursement_to_loan_contract_amount": 900 / 3100,
                            "ratios_are_precise_conversion_rates": False,
                            "difference_is_future_stock_buying_schedule": False},
        "quarterly_statistics_publication_lags_days": [39, 46],
        "new_primary_documents": 4, "new_official_pressconference_web_extracts": 1, "reused_policy_documents": 6,
        "new_accounts": 0, "new_fitted_parameters": 0, "new_forecast_cards": 0,
        "old_accounts_unchanged": True, "independent_validation": False,
        "event_study_is_full_account": False, "causal_identification_established": False,
        "daily_net_stock_demand_reconstructed": False, "historical_first_version_authenticated": False,
        "wait_for_future_data": False, "goal_achieved": False, "orders_authorized": False,
        "report": relative(REPORT), "next_research_question": NEXT,
    }
    figure(events, market)
    report(events, sources, result)
    save(OUT / "result.json", result)
    update_state(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(events[["publication_date", "entry_date", "short_title", "prior_since_20240923_total_return", "net_return5", "net_return20"]].to_string(index=False))


if __name__ == "__main__":
    main()
