"""以指数为研究对象，区分信用约束、政策公开和实际落地。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from historical_index_credit_price_decomposition_v1 import macro_context
from historical_index_liquidity_transmission_v1 import clean
from historical_price_gap_causes_v1 import round_trip

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_index_credit_constraint_relief_v1"
MARKET = ROOT / "reports/research/510300_macro_dynamic_reframe_v1/inputs/market.parquet"
STUDY = "510300_HISTORICAL_INDEX_CREDIT_CONSTRAINT_RELIEF_V1"
SOURCES = [
    {"id": "private_announcement", "url": "https://www.yicai.com/news/100044182.html", "published": "2018-10-22", "kind": "央行原文同期转载", "required": ["债券融资支持工具", "2018-10-22", "中国人民银行"]},
    {"id": "tmlf_announcement", "url": "https://www.thepaper.cn/newsDetail_forward_2753992", "published": "2018-12-19", "kind": "央行公告及答记者问同期转载", "required": ["2018-12-19", "资本较为充足", "15个基点"]},
    {"id": "capital_guidance", "url": "https://www.cebnet.com.cn/20181226/102542254.html", "published": "2018-12-26", "kind": "央行会议消息同期转载", "required": ["12月25日", "永续债", "2018-12-26"]},
    {"id": "private_realization", "url": "https://www.safe.gov.cn/safe/2019/0110/12501.html", "published": "2019-01-10", "kind": "外汇局转载央行行长公开采访", "required": ["2019-01-10", "49家", "313亿元", "有效的融资需求有所下降"]},
    {"id": "tmlf_operation", "url": "https://jrj.wuhan.gov.cn/ynzx_57/xwzx/202011/t20201119_1510568.shtml", "published": "2019-01-24", "kind": "地方政府转载央行操作公告", "required": ["2019-01-24", "2575亿元", "2019年1月23日"]},
    {"id": "cbs_announcement", "url": "https://www.cfbond.com/2019/01/24/wap_99699259.html", "published": "2019-01-24", "kind": "央行公告同期转载", "required": ["2019-01-24", "央行网站", "央行票据互换工具"]},
    {"id": "cbs_operation", "url": "https://jrj.beijing.gov.cn/jrgzdt/201902/t20190221_708772.html", "published": "2019-02-21", "kind": "地方政府转载央行操作公告", "required": ["2019-02-21", "15亿元", "2019年2月20日"]},
    {"id": "private_briefing", "url": "https://wzdt.pbc.gov.cn/rmyh/2025-07/20/article_2025072016205489986.html", "published": "2018-10-26", "kind": "央行保存国新办发布会文字实录", "required": ["2018-10-26", "信用风险缓释", "市场化"]},
    {"id": "cbs_briefing", "url": "https://wzdt.pbc.gov.cn/rmyh/2025-07/20/article_2025072015211734634.html", "published": "2019-02-19", "kind": "央行保存国新办发布会文字实录", "required": ["2019-02-19", "基础货币", "信用风险"]},
    {"id": "january_briefing", "url": "https://m.mof.gov.cn/czxw/201901/t20190115_3123303.htm", "published": "2019-01-15", "kind": "财政部保存国新办发布会实录", "required": ["永续债", "资本约束", "70%"]},
]
EVENTS = [
    {"id": "PVT_LAUNCH", "date": "2018-10-22", "economic_date": "2018-10-22", "label": "民企债券支持工具设立", "family": "信用风险分担", "stage": "制度宣布", "source": "private_announcement", "known": "再贷款提供部分初始资金，通过风险缓释、担保增信等方式支持民企债券融资；同日新增再贷款再贴现额度1500亿元。", "unknown": "不等于全部民企风险获兜底，也未证明订单、利润或指数现金流改善。", "anticipation": "此前金融委10月20日会议已讨论支持计划，公开传播精确时点未完整建立；10月22日不是可确认的零预期起点。"},
    {"id": "TMLF_LAUNCH", "date": "2018-12-19", "economic_date": "2018-12-19", "label": "TMLF设立", "family": "银行资金成本", "stage": "制度宣布", "source": "tmlf_announcement", "known": "对合格且资本较充足银行提供优惠15个基点的资金，期限一年、可续做两次；金额与民营小微贷款表现及需求挂钩。", "unknown": "资金工具本身不补资本；不能证明借款主体新增有效融资需求。", "anticipation": "同期另有再贷款再贴现新增额度1000亿元，不能分离单一工具冲击。"},
    {"id": "CAPITAL_GUIDANCE", "date": "2018-12-26", "economic_date": "2018-12-25", "label": "推动银行永续债发行", "family": "银行资本能力", "stage": "政策方向", "source": "capital_guidance", "known": "金融委办公室专题会议推动多渠道补充银行资本、尽快启动永续债。", "unknown": "会议时尚不是募集资金到账或资本全部补足。", "anticipation": "会议25日、消息26日公开；不在25日入场，且更早资本工具创新政策已经存在。"},
    {"id": "PVT_REALIZATION", "date": "2019-01-10", "economic_date": "2019-01-08", "label": "披露民企融资改善", "family": "信用风险分担", "stage": "结果披露", "source": "private_realization", "known": "采访披露直接间接支持49家民企发行313亿元，2018年11—12月民企发债规模同比增70%；同时承认有效融资需求和银行风险偏好下降。", "unknown": "发债增加不等于新增资本开支或所有成分股盈利上修；无同比基数及同期反事实，不能把增幅全归因于工具。", "anticipation": "采访8日，新华社电讯9日，当前官方转载10日。本轮保守从10日结束后观察，不能称为最早市场可知。"},
    {"id": "TMLF_FIRST_OPERATION", "date": "2019-01-24", "economic_date": "2019-01-23", "label": "首次TMLF操作披露", "family": "银行资金成本", "stage": "实际操作", "source": "tmlf_operation", "known": "23日首次操作2575亿元、利率3.15%；按2018年第四季度民营小微贷款增量与机构需求确定。", "unknown": "操作量并非同日新发企业贷款，也不是股票净流入。", "anticipation": "1月上旬已公开预告下旬首次操作；本轮按24日官方转载时点，保守延后一日。"},
    {"id": "CBS_LAUNCH", "date": "2019-01-24", "economic_date": "2019-01-24", "label": "CBS设立及担保品扩围", "family": "银行资本能力", "stage": "制度宣布", "source": "cbs_announcement", "known": "允许合格永续债换入央票，并把合格银行永续债纳入央行融资工具担保品；目标为改善永续债流动性及认购意愿。", "unknown": "CBS不直接投放基础货币，也不转移一级交易商原有信用风险。", "anticipation": "银行永续债发行及配套支持此前已推进；首单25日定价落地与该窗口重叠。"},
    {"id": "CBS_FIRST_OPERATION", "date": "2019-02-21", "economic_date": "2019-02-20", "label": "首次CBS操作披露", "family": "银行资本能力", "stage": "实际操作", "source": "cbs_operation", "known": "20日首次CBS操作15亿元，期限一年，费率0.25%；只是部分已持有永续债的互换。", "unknown": "15亿元互换额不能当作基础货币净投放或新资本发行额。", "anticipation": "CBS1月已宣布，19日发布会进一步说明；本轮按21日官方转载，不把实施当意外创设。"},
]


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("已有设定，不覆盖。")
    prior = json.loads((ROOT / "reports/research/510300_historical_index_liquidity_transmission_v1/protocol.json").read_text(encoding="utf-8"))
    save("protocol.json", {
        "study_id": STUDY, "recorded_at": now(), "research_mode": "HISTORICAL_DISCOVERY_ONLY",
        "scope": "2018年第四季度至2019年第一季度，三条信用传导链的七个固定公开节点。先定日期再计算；不是该时期所有政策的穷尽事件库。",
        "research_unit": "整体指数、金融体系与实体融资约束；没有新增个股收益或公司研究。",
        "question": "资金便宜、风险分担和补资本分别改变什么；政策宣布与落地后的指数剩余收益是否相同？",
        "events": EVENTS, "sources": SOURCES, "costs": prior["costs"], "horizons_sessions": [5, 20],
        "entry_rule": "载明来源公开日结束后的下一实际交易日开盘；较晚官方转载统一按转载日，既不提前到采访/会议日，也不据收益切换来源日期。",
        "source_clock_limit": "部分公开日以同期转载或迁移后的官方旧页面为依据，没有2018年归档快照；保守日期只能避免提前入场，不能证明最早可交易时点。",
        "information_cutoff": "背景宏观数据只取公开节点日期00:00以前首次公布值；节点公告内容可在该日结束后使用。",
        "pre_entry_move": "节点日前最后收盘至下一交易日开盘的变化，只是时间窗描述，不是纯公告跳空或量化预期差。",
        "overlap": "5日、20日分别识别窗口交叠的连通组；完全相同入出场只算一个不同窗口，不把密集政策节点当独立成功。",
        "strategy_decision": "本轮没有预设交易过滤器，不从七个节点选赢家，不拟合阈值，不将事后涨幅组成完整账户。",
        "known_history": "这些年份及相邻政策窗口此前已经被多轮研究观察；不是独立样本外验证。",
        "interpretation": "同日其他政策、全球风险偏好、贸易消息及盈利变化均可能共同作用；事件收益不是单一政策因果贡献。",
        "excluded": ["央行例行全部操作及后续季度重复操作", "逐个受益企业和个股回报", "首家银行永续债发行作为新增独立交易节点", "未有当时披露证据的融资结果回填"],
        "new_parameters_fitted": 0, "new_full_accounts": 0, "net_sharpe": None, "goal_achieved": False, "orders_authorized": False,
    })
    print("已固定三条传导链、七个节点、5/20日收益及重叠处理口径。", flush=True)


def fetch_one(source):
    result = dict(source)
    path = OUT / "sources" / (source["id"] + ".html")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(source["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=(10, 25))
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        for item in soup(["script", "style"]):
            item.decompose()
        body = soup.get_text("\n", strip=True)
        missing = [word for word in source["required"] if word not in body]
        path.write_bytes(response.content)
        path.with_suffix(".txt").write_text(body, encoding="utf-8")
        result.update(status="RETRIEVED" if not missing else "BODY_CHECK_REQUIRED", missing_fragments=missing,
                      sha256=hashlib.sha256(response.content).hexdigest(), path=str(path.relative_to(ROOT)), retrieved_at=now())
    except Exception as exc:
        result.update(status="MISSING_LOCAL_SOURCE", error=str(exc), retrieved_at=now())
    return result


def fetch():
    if (OUT / "source_manifest.json").exists():
        raise RuntimeError("已有来源记录，不覆盖。")
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(fetch_one, SOURCES))
    save("source_manifest.json", result)
    print(json.dumps([{k: r.get(k) for k in ["id", "status", "missing_fragments", "error"]} for r in result], ensure_ascii=False, indent=2), flush=True)


def resolve_sources():
    if (OUT / "source_resolution.json").exists():
        raise RuntimeError("已有来源补充，不重复抓取。")
    manifest = json.loads((OUT / "source_manifest.json").read_text(encoding="utf-8"))
    notes = []
    for row in manifest:
        if row["id"] == "private_realization":
            body = (ROOT / row["path"]).with_suffix(".txt").read_text(encoding="utf-8")
            compact = "".join(body.split())
            assert all("".join(word.split()) in compact for word in row["required"])
            row.update(status="RETRIEVED", normalized_whitespace_check=True)
            notes.append("外汇局正文的49、313位于独立HTML标签；合并空白后与已读取网页一致，没有改数字或日期。")
        if row["id"] == "cbs_announcement":
            replacement = {**row, "url": "https://wallstreetcn.com/articles/3475732", "required": ["2019/01/24", "中国央行", "央行票据互换工具"],
                           "kind": "央行公告及答记者问同期转载，附失效原央行链接"}
            replacement.pop("error", None)
            replacement = fetch_one(replacement)
            replacement["original_url"] = row["url"]
            replacement["original_pbc_url"] = "https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/3752454/index.html"
            replacement["date_or_return_rule_changed"] = False
            row.clear()
            row.update(replacement)
            notes.append("财富网返回访问频率限制；改用同日央行原文转载。原央行旧链接404。保持1月24日及既定入场不变。")
    save("source_resolution.json", {"recorded_at": now(), "before_returns_computed": not (OUT / "result.json").exists(),
                                    "notes": notes, "sources": manifest})
    print(json.dumps({"说明": notes, "来源": [{"id": r["id"], "status": r["status"]} for r in manifest]}, ensure_ascii=False, indent=2), flush=True)


def assign_clusters(rows, horizon):
    selected = sorted([r for r in rows if r["horizon"] == horizon], key=lambda r: (r["entry_date"], r["exit_date"]))
    end, cluster = "", 0
    unique_windows = set()
    for row in selected:
        if row["entry_date"] >= end:
            cluster += 1
        end = max(end, row["exit_date"])
        row["overlap_cluster"] = cluster
        unique_windows.add((row["entry_date"], row["exit_date"]))
    return {"horizon": horizon, "event_nodes": len(selected), "distinct_windows": len(unique_windows),
            "connected_time_clusters": cluster, "independent_economic_experiments": None,
            "note": "不交叠不代表经济独立；连通组只揭示重复观察同一行情。"}


def analyze():
    if (OUT / "result.json").exists():
        raise RuntimeError("已有结果，不重算或改选日期。")
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    source_rows = json.loads((OUT / "source_resolution.json").read_text(encoding="utf-8"))["sources"]
    manifest = {r["id"]: r for r in source_rows}
    missing = [e["source"] for e in protocol["events"] if manifest[e["source"]]["status"] != "RETRIEVED"]
    if missing:
        raise RuntimeError("事件来源需先解决：" + "、".join(missing))
    market = pd.read_parquet(MARKET).sort_values("date").reset_index(drop=True)
    market["date"] = pd.to_datetime(market.date).dt.tz_localize(None).dt.normalize()
    dates = pd.DatetimeIndex(market.date)
    dividends = pd.read_csv(ROOT / "data/reference/510300_dividends.csv")
    dividends = dividends[dividends.symbol.eq("510300.SH")].copy()
    dividends["record_date"] = pd.to_datetime(dividends.record_date)
    context = macro_context(protocol["events"])
    rows, pnl_errors = [], []
    for event in protocol["events"]:
        date = pd.Timestamp(event["date"])
        entry_index = dates.searchsorted(date, side="right")
        before_index = dates.searchsorted(date, side="left") - 1
        entry, before = market.iloc[entry_index], market.iloc[before_index]
        assert entry.date > date and before.date < date
        for horizon in protocol["horizons_sessions"]:
            exit_row = market.iloc[entry_index + horizon]
            entitled = dividends[(dividends.record_date >= entry.date) & (dividends.record_date < exit_row.date)]
            cash_per_share = float(entitled.cash_dividend_per_share.sum())
            trade = round_trip(float(entry.open), float(exit_row.open), cash_per_share, protocol["costs"])
            gross = (float(exit_row.open) + cash_per_share) / float(entry.open) - 1
            identity = (trade["sell_price"] - trade["buy_price"]) * trade["shares"] - trade["commissions_cny"] + trade["dividend_entitlement_cny"]
            pnl_errors.append(abs(identity - trade["net_pnl_cny"]))
            assert trade["paid_cny"] <= protocol["costs"]["illustrative_event_budget_cny"]
            assert trade["shares"] % 100 == 0 and trade["net_return"] <= gross + 1e-12
            rows.append({"event_id": event["id"], "date": event["date"], "economic_date": event["economic_date"],
                         "label": event["label"], "family": event["family"], "stage": event["stage"], "horizon": horizon,
                         "entry_date": entry.date.strftime("%Y-%m-%d"), "exit_date": exit_row.date.strftime("%Y-%m-%d"),
                         "entry_open": float(entry.open), "exit_open": float(exit_row.open),
                         "before_date": before.date.strftime("%Y-%m-%d"), "before_close": float(before.close),
                         "pre_entry_price_change_pct": (float(entry.open) / float(before.close) - 1) * 100,
                         "gross_return": gross, "dividend_per_share": cash_per_share, **trade})
    clusters = [assign_clusters(rows, horizon) for horizon in protocol["horizons_sessions"]]
    checks = {"event_rows": len(rows), "event_count": len(protocol["events"]), "max_pnl_identity_error_cny": max(pnl_errors),
              "source_before_entry": all(r["date"] < r["entry_date"] for r in rows), "dividend_counting": "登记日持有所得权益；只作事件总收益，不模拟到账日现金再投资。"}
    assert max(pnl_errors) < 1e-7 and len(rows) == 14
    save("events_and_returns.json", rows)
    save("known_macro_context.json", context)
    save("calculation_checks.json", checks)
    save("result.json", {"study_id": STUDY, "computed_at": now(), "classification": "PROGRESS_INDEX_CONSTRAINTS_AND_POLICY_TIMELINE",
                         "event_count": len(protocol["events"]), "overlap": clusters, "new_parameters_fitted": 0,
                         "new_full_accounts": 0, "net_sharpe": None, "goal_achieved": False, "orders_authorized": False})
    print(pd.DataFrame(rows)[["date", "label", "horizon", "entry_date", "exit_date", "net_return", "overlap_cluster"]].to_string(index=False), flush=True)
    print(json.dumps(clusters, ensure_ascii=False, indent=2), flush=True)


def draw(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.4), gridspec_kw={"width_ratios": [1.05, 1.2]})
    fig.patch.set_facecolor("#f6f4ed")
    colors = ["#9aabb5", "#147d76"]
    dates = sorted({r["date"] for r in rows})
    labels = []
    for date in dates:
        items = [r for r in rows if r["date"] == date and r["horizon"] == 20]
        title = items[0]["label"] if len(items) == 1 else "TMLF操作披露／CBS设立"
        labels.append(date + "\n" + title)
    y = np.arange(len(dates))
    for offset, horizon, color in [(-.17, 5, colors[0]), (.17, 20, colors[1])]:
        values = [next(r["net_return"] * 100 for r in rows if r["date"] == d and r["horizon"] == horizon) for d in dates]
        bars = axes[0].barh(y + offset, values, height=.29, color=color, label=f"{horizon}日")
        for bar, value in zip(bars, values):
            axes[0].text(value + (.22 if value >= 0 else -.22), bar.get_y() + bar.get_height() / 2,
                         f"{value:+.2f}%", va="center", ha="left" if value >= 0 else "right", fontsize=9)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(-10.5, 21)
    axes[0].axvline(0, color="#9b9b92", lw=.8)
    axes[0].set_title("不同公开节点的510300成本后收益", loc="left", pad=18)
    axes[0].set_xlabel("占实际投入金额；相同日期仅展示一次")
    axes[0].legend(loc="upper right", frameon=False)

    twenty = [r for r in rows if r["horizon"] == 20]
    for i, row in enumerate(twenty):
        start = mdates.date2num(pd.Timestamp(row["entry_date"]).to_pydatetime())
        end = mdates.date2num(pd.Timestamp(row["exit_date"]).to_pydatetime())
        color = "#aa795b" if row["overlap_cluster"] == 1 else "#147d76"
        axes[1].barh(i, end - start, left=start, height=.5, color=color, alpha=.78)
    axes[1].set_yticks(range(len(twenty)), [r["label"] for r in twenty])
    axes[1].invert_yaxis()
    axes[1].xaxis_date()
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[1].tick_params(axis="x", labelsize=9, rotation=25)
    axes[1].set_title("七个节点只有六个不同20日窗口", loc="left", pad=18)
    axes[1].set_xlabel("绿色窗口相互连接，不能计作六次独立成功")
    for ax in axes:
        ax.set_facecolor("#f6f4ed")
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.grid(axis="x", alpha=.13)
        ax.set_axisbelow(True)
    fig.suptitle("宽信用需要拆解约束；密集政策不能重复验证同一段上涨", x=.055, ha="left", fontsize=17, fontweight="bold")
    fig.text(.055, .025, "历史发现：次一交易日开盘进入，固定5／20个交易日后开盘退出；含佣金、滑点和分红。没有生成完整账户或夏普。", fontsize=10, color="#535b5a")
    fig.subplots_adjust(left=.17, right=.985, top=.86, bottom=.17, wspace=.7)
    fig.savefig(OUT / "信用约束节点与指数窗口.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def report():
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    rows = json.loads((OUT / "events_and_returns.json").read_text(encoding="utf-8"))
    context = json.loads((OUT / "known_macro_context.json").read_text(encoding="utf-8"))
    sources = {s["id"]: s for s in json.loads((OUT / "source_resolution.json").read_text(encoding="utf-8"))["sources"]}
    link = lambda key, label: f"[{label}]({sources[key]['url']})"
    lines = ["# 510300历史发现：信用约束、政策阶段与指数收益", "",
        "研究主线已转为指数整体。2018年第四季度到2019年第一季度，政策依次涉及信用风险分担、优惠资金和银行资本补充；这不是一种可以用‘利率下降’概括的冲击。七个公开节点对应六个不同的20日窗口，按交叠关系仅分成两段，不能把同一轮行情重复当作成功证据。", "",
        "**本轮最有用的发现**", "",
        "第一，指数上涨与制造业订单已经全面恢复并不同步。2019年1月的两个观察节点，已公布的新订单指数仍为49.7；2月节点已公布值为49.6，但相应20日指数净收益为正。这排除了‘这些窗口都是等制造业订单回到扩张后才涨’的描述。它没有证明订单弱是利好，也没有直接识别出预期改善或风险溢价下降的幅度。", "",
        "这个区别还要放回整个经济：2019年1月制造业新订单49.6的同一份发布中，非制造业商务活动指数为54.7，服务业商务活动指数为53.6、比上月增加1.3点。因此不能把‘制造业订单偏弱’扩大成‘全部经济部门都没有改善’，更不能据此断言指数上涨完全由估值推动。这是收益计算后补充读取的竞争解释，没有新增过滤条件。[统计局2019年1月发布](https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900229.html)。", "",
        "第二，政策推出与实际融资不是同一个时间点。首次TMLF金额根据2018年第四季度贷款增量和机构需求确定；不能把操作量当作同日新增企业贷款，更不能把后来融资改善提前填到工具宣布当天。", "",
        "第三，2018年10月民企债券支持工具宣布后，5日和20日仍分别亏损7.14%、1.03%。明确了政策针对什么约束，并不自动获得指数买入优势。2019年初的正收益值得解释，但不能倒推出每一个重叠政策都是有效独立因子。", "",
        "**三种约束及其上游原因**", "",
        "| 传导环节 | 约束为什么存在 | 工具直接改变什么 | 不能据此确认什么 |", "|---|---|---|---|",
        "| 信用风险分担 | 部分民企经营和融资受压，债券投资者担心违约与再融资接续 | 通过信用风险缓释、担保增信等安排，改变特定风险的承担与定价 | 全部企业获兜底、实际订单回升、指数盈利全面改善 |",
        "| 银行资金成本和期限 | 负债成本、期限及放贷激励限制信用供给 | TMLF向合格银行提供较稳定、优惠15个基点的资金，与民营小微贷款表现及需求挂钩 | 最缺资本的银行都能获得工具；便宜资金自动转为有效贷款 |",
        "| 银行资本能力 | 银行承担贷款等风险资产需要资本；已有资金不代表仍有资本空间 | 永续债增加资本补充渠道，CBS及担保品扩围改善永续债流动性和购买意愿 | CBS本身注入资本、自动投放基础货币或转移原持有人信用风险 |", "",
        "民企支持工具的原始设计来自" + link("private_announcement", "2018年10月22日央行公告转载") + "及" + link("private_briefing", "10月26日央行发布会实录") + "。后一来源用于说明机制，未当作22日已知的新信息。", "",
        "TMLF准入条件包括资本较充足、资产质量健康及具备继续放贷潜力。因此它改善的是合格机构的边际资金供给，不能替代补资本。" + link("tmlf_announcement", "2018年12月19日央行公告及问答转载") + "。", "",
        "CBS通过等面值券券互换改善担保品条件。互换操作不自动投放基础货币，永续债所有权及信用风险仍由原持有人承担。央行2月19日发布会还说明，当时操作没有一个必须达到的数量目标。" + link("cbs_announcement", "1月24日央行公告及问答转载") + "、" + link("cbs_briefing", "2月19日央行发布会实录") + "。2月的解释是复盘机制证据，没有倒填到1月信号。", "",
        "从指数角度，传导可能经过整体再融资压力、投资与消费支出、总量盈利预期、风险补偿和股票配置需求；本轮数据没有识别这几条路径的各自贡献。制造业PMI也只代表部分需求，不能直接代替全部指数成分的景气。", "",
        "**公开节点后的可实现收益**", "",
        "统一以所采用来源载明公开日结束后的下一交易日开盘进入，固定5或20个交易日后的开盘退出。10万元事件预算，单边0.1%滑点、0.04%佣金且最低5元，按价格档位、100份整数及分红权益计算。下表收益分母为实际投入资金，未计算20万元完整账户。", "",
        "| 采用的公开日 | 节点 | 开盘进入 | 5日净收益 | 20日净收益 | 20日退出 |", "|---|---|---|---:|---:|---|"]
    for event in protocol["events"]:
        five = next(r for r in rows if r["event_id"] == event["id"] and r["horizon"] == 5)
        twenty = next(r for r in rows if r["event_id"] == event["id"] and r["horizon"] == 20)
        lines.append(f"| {event['date']} | {event['label']} | {five['entry_date']} | {five['net_return']*100:+.2f}% | {twenty['net_return']*100:+.2f}% | {twenty['exit_date']} |")
    lines += ["",
        "1月24日两行的入场、退出和收益完全相同，不是两笔不同证据。12月20日至次年3月22日的这些20日窗口通过重叠连接在一起；本轮不对它们计算‘六次成功’的胜率、显著性或夏普。连通时间段也不等于经济学上的独立试验。", "",
        "本轮仅选择三条既定机制链的阶段节点，没有穷尽该季度全部政策。全球金融条件、贸易消息和其他国内政策等竞争解释尚未分离；表中是事件之后的指数表现，不能写成某项政策的因果收益。", "",
        "**当时能知道什么**", "",
        "| 公开节点 | 最新制造业新订单 | 最新M1同比 | 最新M2同比 | 最新社融存量同比 | 对应统计月份（订单／货币／社融） |", "|---|---:|---:|---:|---:|---|"]
    seen_dates = set()
    for c in context:
        if c["date"] in seen_dates:
            continue
        seen_dates.add(c["date"])
        lines.append(f"| {c['date']} | {c['pmi_new_orders']:.1f} | {c['m1_yoy_pct']:.1f}% | {c['m2_yoy_pct']:.1f}% | {c['tsf_yoy_pct']:.1f}% | {c['pmi_month']}／{c['money_month']}／{c['tsf_month']} |")
    lines += ["",
        "这些背景数据均取节点日00:00之前已发布的最近值。同比、扩张阈值和环比变化都不是调查得到的市场共识，不能直接称为预期差。2018年社融统计范围曾有调整，表中不同月份的首次发布同比也不能简单相减解释为同口径新增信用。", "",
        "[统计局2018年12月PMI发布](https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900190.html)载明新订单49.7；[2019年1月发布](https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900229.html)载明49.6。2月节点已知[1月社融存量同比10.4%](https://www.pbc.gov.cn/diaochatongjisi/116219/116225/cb4e819a78fe46c2a97762cf364c60df/index.html)，仍需区分票据、贷款、政府债券等具体来源。", "",
        "1月上旬公开采访已同时披露民企债券发行改善和有效融资需求偏弱。支持49家民企发行313亿元，与11—12月整体民企发债同比增长70%是不同统计量，不能相乘或当作新增贷款；直接间接支持也不是单独工具的因果识别。" + link("private_realization", "外汇局1月10日转载的央行行长采访") + "。", "",
        "**时间与金额的易错点**", ""]
    for event in protocol["events"]:
        evidence_links = [link(event["source"], sources[event["source"]]["kind"])]
        anticipation = event["anticipation"]
        if event["id"] == "PVT_LAUNCH":
            evidence_links.append("[10月20日金融委会议官方转载](https://www.safe.gov.cn/hainan/2018/1022/798.html)")
        if event["id"] == "TMLF_FIRST_OPERATION":
            evidence_links.append(link("private_realization", "1月上旬央行采访预告"))
        if event["id"] in ["CBS_LAUNCH", "CBS_FIRST_OPERATION"]:
            evidence_links.append(link("cbs_briefing", "2月19日央行实录"))
        if event["id"] == "CAPITAL_GUIDANCE":
            anticipation = anticipation.replace("，且更早资本工具创新政策已经存在", "")
        lines += [f"- {event['label']}：{anticipation} 来源：{'、'.join(evidence_links)}。"]
    lines += ["",
        "1月首次TMLF2575亿元针对银行资金安排；首单银行永续债400亿元是资本工具发行；首次CBS15亿元是券券互换。这三个数量对应不同对象和会计关系，不能相加成为‘股市新增流动性’。前两项落地细节可由" + link("tmlf_operation", "央行TMLF公告官方转载") + "及" + link("cbs_briefing", "央行资本补充发布会") + "核对；CBS实际操作以" + link("cbs_operation", "2月21日官方转载") + "为依据。", "",
        "1月10日采访转载、1月24日TMLF转载和2月21日CBS转载均晚于采访或操作日，本轮已保守延后，不声称识别了市场最早反应。部分官网旧链接失效，以署名央行的同期转载保存正文；这些页面是当前取得的历史内容，没有同期网页归档快照。", "",
        "**对研究方向的影响**", "",
        "保留三类可解释约束：资金成本、信用风险承担和资本空间。停止把‘政策放松’当成一个单向因子，也不把2019年多个正收益节点合并包装成已验证策略。当前更具体的问题是：当时整体融资究竟增加在什么用途，是否仍主要是债务续接、短期周转和政府融资，还是企业长期投资及居民需求也发生变化。", "",
        "下一轮维持在指数层面，检查2018年第四季度至2019年第一季度每月首次公开的社融和贷款结构，保留所有发布月及反例，再与已知订单和指数阶段比较。不新增个股研究，不要求跨五年十年，也不等待未来数据。", "",
        "必要核对已完成：14条事件计算的信息日均早于入场，分红按登记日权益纳入，金额与价格现金流一致。没有参数搜索，没有新增完整账户；成本后夏普1.2及年化10%的共同目标仍未达成。", "",
        "文件：" + "、".join(f"[{label}](<{(OUT / name).as_posix()}>)" for name, label in [("protocol.json", "固定范围"), ("events_and_returns.json", "收益明细"), ("known_macro_context.json", "当时已知宏观数据"), ("source_resolution.json", "原始来源与时点"), ("信用约束节点与指数窗口.png", "收益及窗口图")]) + "。"]
    name = "历史发现_信用约束与指数政策阶段.md"
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    draw(rows)
    result.update(report=name, report_completed_at=now(),
                  discovery="信用约束分为资金、风险分担和资本；制造业订单仍弱时2019年初指数窗口已上涨，多个政策窗口重复观察同一行情，不能形成独立优势证明。",
                  next_historical_question="按2018年第四季度至2019年第一季度全部月度首次发布拆解社融与贷款用途，区分短期周转、票据、政府融资及长期贷款；解释指数阶段变化，不扩展个股或改救旧规则。")
    save("result.json", result)
    save("post_calculation_context.json", {
        "recorded_at": now(), "role": "补充竞争解释，未用于事件选择、分组或交易筛选", "returns_recalculated": False,
        "source_url": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900229.html", "source_published_at": "2019-01-31T09:00:00+08:00",
        "reference_period": "2019-01", "manufacturing_new_orders": 49.6, "nonmanufacturing_business_activity": 54.7,
        "services_business_activity": 53.6, "services_activity_month_change": 1.3,
        "implication": "制造业订单弱不等于所有经济部门均弱；不能把指数上升完全归因于风险溢价。不同分项不直接等权或按指数权重拼成新因子。"})
    print("已完成信用约束历史报告与窗口图；没有新增完整账户或宣称目标达成。", flush=True)


def record_progress():
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    assert (OUT / result["report"]).is_file()
    prefix = "reports/research/510300_historical_index_credit_constraint_relief_v1/"
    stamp, changes = now(), []
    for name in ["510300_historical_cause_discovery_v1.json", "510300_existing_data_training_mandate_v1.json"]:
        path = ROOT / "config" / name
        cfg = json.loads(path.read_text(encoding="utf-8"))
        before = dict(cfg)
        if name == "510300_historical_cause_discovery_v1.json":
            cfg.update(current_study=prefix + "protocol.json", latest_completed_study=prefix + "result.json",
                       latest_report=prefix + result["report"], updated_at=stamp)
        else:
            cfg.update(current_round=STUDY, latest_progress_receipt=prefix + "result.json",
                       latest_historical_index_credit_constraint_relief=prefix + "result.json",
                       latest_continuation_report=prefix + result["report"], latest_historical_report=prefix + result["report"],
                       latest_continuation_classification=result["classification"], current_driver_continuation_classification=result["classification"],
                       current_driver_consecutive_blocked_goal_turns=0, latest_historical_diagnostic_at=stamp,
                       latest_goal_service_status="active", latest_goal_service_status_observed_at=stamp, goal_status="active", goal_achieved=False,
                       local_goal_work_status="ACTIVE_HISTORICAL_ONLY", last_research_result=result["discovery"],
                       last_source_result="七个节点保存当时央行公告或采访原文，区分原事件、公开日与较晚转载；期货、个股及前瞻任务没有扩展。",
                       next_research_question=result["next_historical_question"])
        changes.append({"path": str(path.relative_to(ROOT)), "fields": {k: {"before": before.get(k), "after": v} for k, v in cfg.items() if before.get(k) != v}})
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save("authority_update.json", {"recorded_at": stamp, "study_id": STUDY, "previous_goal_turn_classification": "PROGRESS",
                                  "current_goal_turn_classification": "PROGRESS_COMPLETED_INDEX_CREDIT_CONSTRAINT_TIMELINE", "changes": changes,
                                  "goal_achieved": False, "orders_authorized": False})
    print("已登记指数信用约束研究进展；夏普目标保持未达成。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="指数信用约束解除历史研究")
    parser.add_argument("mode", choices=["prepare", "fetch", "resolve-sources", "analyze", "report", "record-progress"])
    args = parser.parse_args()
    {"prepare": prepare, "fetch": fetch, "resolve-sources": resolve_sources, "analyze": analyze,
     "report": report, "record-progress": record_progress}[args.mode]()


if __name__ == "__main__":
    main()
