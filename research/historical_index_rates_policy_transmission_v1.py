"""重建固定历史发布窗口中的资金价格、国债期限与政策操作，不搜索策略参数。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from historical_index_liquidity_transmission_v1 import clean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_index_rates_policy_transmission_v1"
PRIOR = ROOT / "reports/research/510300_historical_index_credit_expectation_gap_v1"
STUDY = "510300_HISTORICAL_INDEX_RATES_POLICY_TRANSMISSION_V1"
INPUTS = {
    "curve": "data/raw/macro/china_government_bond_yields_daily.parquet",
    "funding": "data/raw/macro/510300_macro_stress_2015_v2/fdr007_daily_2015_2026.parquet",
    "market": "reports/research/510300_macro_dynamic_reframe_v1/inputs/market.parquet",
    "policy_rates": "reports/research/510300_fiscal_execution_state_20d_v1/inputs/operation_rates.parquet",
}
OPERATIONS = ROOT / "data/raw/remediation/510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_1/pboc/articles"


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("已有范围，不覆盖。")
    prior = json.loads((PRIOR / "protocol.json").read_text(encoding="utf-8"))
    events = [e for e in prior["events"] if "2019-01-01" <= e["date"] <= "2019-04-30"]
    assert len(events) == 4
    save("protocol.json", {
        "study_id": STUDY, "recorded_at": now(), "previous_goal_turn_classification": "PROGRESS",
        "mode": "HISTORICAL_DISCOVERY_ONLY", "research_unit": "沪深300整体，510300为可交易价格观察",
        "question": "同一时期资金利率、短长国债定价和央行操作是否支持融资改善与近期宽松预期的不同传导？",
        "events": events, "series": ["fdr007", "cgb_1y", "cgb_10y", "slope_10y_1y_bp"],
        "local_inputs": INPUTS, "calendar_scope": ["2018-12-03", "2019-05-16"],
        "policy_document_scope": ["2019-01-01", "2019-04-30"],
        "event_baseline": "公布日期之前最后一个ETF交易日；当天原观测缺失时保留缺失，不以前值替代。",
        "response_endpoints": "公布日期之后第1、第5、第20个ETF交易日的利率观察，首次交易日计为第1日。响应终点不选公布当日，但前一交易日基准至终点的变化包含公布当日，不能称为纯公布后反应。",
        "response_horizons": [1, 5, 20],
        "point_vs_average": "FDR007既保存端点值，也列出终点截至当日最近20个有效原观测均值；不将上午定盘值称为全天DR007。",
        "calendar_comparison": "2018年12月至2019年4月每月最后一个ETF交易日及对应20观测资金均值，按月历固定，不按行情分段。",
        "rates_units": "原利率为年化百分数，变化乘100得到bp；10年减1年是期限差，不直接命名为期限溢价。",
        "availability": "FDR原账本给出11:30发布时间；中债估值曲线保守作为下一ETF交易日开盘才可使用。响应终点的当日值只作事后路径，不作为原入场信号。",
        "policy_interpretation": "操作金额不是股票流入；到期替换、财政税期、现金投放与政策利率改变须分别解释。",
        "returns": "复用此前已保存的四次5/20日事件净收益，不重算、截短或选择最好窗口。利率响应按收盘日期观察，ETF收益仍按原开盘进入与退出。",
        "prior_information_seen": "已见事件收益、部分2019年行情和同期政策解释，旧信用利差研究亦使用过该段历史；不是独立验证。本轮尚未计算这些固定窗口的期限及资金价格响应。",
        "old_strategy_disposition": "旧MACRO-02及信用成本分组失败保持原结论，不反号、不新增过滤救回。",
        "new_parameters_fitted": 0, "new_full_accounts": 0, "net_sharpe": None,
        "goal_achieved": False, "orders_authorized": False,
    })
    print("已固定四次发布、三个利率响应终点和全部月末；不生成新交易规则。", flush=True)


def build():
    if (OUT / "event_responses.json").exists():
        raise RuntimeError("已有利率比较，不覆盖。")
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    loaded = {k: pd.read_parquet(ROOT / v) for k, v in INPUTS.items()}
    receipts = []
    for key, path in INPUTS.items():
        p = ROOT / path
        receipts.append({"id": key, "path": path, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                         "rows": len(loaded[key]), "columns": list(loaded[key].columns)})
    save("input_receipts.json", receipts)
    market = loaded["market"].copy()
    market["date"] = pd.to_datetime(market["date"])
    calendar = pd.DatetimeIndex(market["date"].sort_values().unique())
    curve = loaded["curve"].copy().sort_values("date")
    curve["date"] = pd.to_datetime(curve["date"])
    funding = loaded["funding"].copy().sort_values("date")
    funding["date"] = pd.to_datetime(funding["date"])
    funding["fdr007"] = pd.to_numeric(funding["first_release_value"], errors="raise")
    assert not curve["date"].duplicated().any() and not funding["date"].duplicated().any()
    funding["fdr007_20obs_mean"] = funding["fdr007"].rolling(20, min_periods=20).mean()
    curve["slope_10y_1y_bp"] = (curve["cgb_10y"] - curve["cgb_1y"]) * 100
    daily = curve.merge(funding[["date", "fdr007", "fdr007_20obs_mean", "published_at", "raw_path"]], on="date", how="outer")
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["curve_available_at_rule"] = "NEXT_ETF_SESSION_OPEN_AFTER_OBSERVATION_DATE"
    scope = daily["date"].between(*protocol["calendar_scope"])
    scoped = daily.loc[scope].copy()
    scoped.to_parquet(OUT / "daily_rates.parquet", index=False)
    save("daily_rates.json", scoped.to_dict("records"))
    keys = ["fdr007", "fdr007_20obs_mean", "cgb_1y", "cgb_10y", "slope_10y_1y_bp"]

    def snapshot(date):
        row = daily.loc[daily["date"].eq(date)]
        if len(row) != 1:
            return {"date": date, **{k: None for k in keys}, "status": "MISSING_DATE"}
        row = row.iloc[0]
        next_day = calendar[calendar > date]
        return {"date": date, **{k: clean(row.get(k)) for k in keys},
                "curve_available_at": (next_day[0] + pd.Timedelta(hours=9, minutes=30)).tz_localize("Asia/Shanghai") if len(next_day) else None,
                "funding_published_at": row.get("published_at"),
                "status": "OBSERVED" if all(pd.notna(row[k]) for k in keys) else "PARTIAL_MISSING"}

    previous = json.loads((PRIOR / "comparison.json").read_text(encoding="utf-8"))["rows"]
    rows = []
    for event in protocol["events"]:
        date = pd.Timestamp(event["date"])
        before = snapshot(calendar[calendar < date][-1])
        after_dates = calendar[calendar > date]
        old = next(r for r in previous if r["stat_month"] == event["id"])
        for horizon in protocol["response_horizons"]:
            end = snapshot(after_dates[horizon - 1])
            changes = {}
            for key in keys:
                factor = 1 if key.endswith("_bp") else 100
                changes[key + "_change_bp"] = None if before[key] is None or end[key] is None else (end[key] - before[key]) * factor
            rows.append({"stat_month": event["id"], "release_date": event["date"], "horizon": horizon,
                         "before": before, "after": end, **changes,
                         "old_event_net_return_5d": old["net_return_5d"], "old_event_net_return_20d": old["net_return_20d"]})
    monthends = []
    for month in pd.period_range("2018-12", "2019-04", freq="M"):
        dates = calendar[calendar.to_period("M") == month]
        monthends.append(snapshot(dates[-1]))
    save("event_responses.json", {"computed_at": now(), "rows": rows})
    save("monthend_states.json", {"computed_at": now(), "rows": monthends})
    save("result.json", {"study_id": STUDY, "computed_at": now(), "classification": "PROGRESS_INDEX_RATES_POLICY_HISTORY",
                         "events": 4, "response_rows": len(rows), "monthends": len(monthends),
                         "new_parameters_fitted": 0, "new_full_accounts": 0, "net_sharpe": None,
                         "goal_achieved": False, "orders_authorized": False})
    print(pd.DataFrame([{k: r[k] for k in ["stat_month", "horizon", "fdr007_change_bp", "fdr007_20obs_mean_change_bp", "cgb_1y_change_bp", "cgb_10y_change_bp", "slope_10y_1y_bp_change_bp"]} for r in rows]).to_string(index=False), flush=True)
    print(pd.DataFrame(monthends)[["date"] + keys].to_string(index=False), flush=True)


def operations():
    if (OUT / "operation_documents.json").exists():
        raise RuntimeError("已提取操作公告，不覆盖。")
    rows = []
    for p in sorted(OPERATIONS.glob("2019-*.html")):
        date = p.name[:10]
        if not "2019-01-01" <= date <= "2019-04-30":
            continue
        soup = BeautifulSoup(p.read_bytes(), "html.parser")
        body = soup.select_one("#zoom") or soup.select_one(".TRS_Editor") or soup
        text = body.get_text(" ", strip=True)
        page = soup.get_text(" ", strip=True)
        match = re.search(r"2019-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}", page)
        title = soup.title.get_text(strip=True) if soup.title else "公开市场业务公告"
        identifier = p.stem.split("_")[-1]
        url = "https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125431/125475/" + identifier + "/index.html"
        rows.append({"date": date, "published_at": match[0] if match else None, "title": title, "text": text,
                     "raw_path": str(p.relative_to(ROOT)), "source_url": url,
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    assert len(rows) > 50
    save("operation_documents.json", {"recorded_at": now(), "scope": ["2019-01-01", "2019-04-30"], "rows": rows,
                                      "limits": "本地已有央行日公告；文件存在不保证全部营业日均覆盖，未将缺失日填为零操作。标题与正文均保留。"})
    print("已提取" + str(len(rows)) + "份同期操作公告。", flush=True)
    selected = {"2019-01-04", "2019-01-15", "2019-01-16", "2019-01-23", "2019-02-15", "2019-03-19", "2019-04-12", "2019-04-16", "2019-04-17", "2019-04-24", "2019-04-30"}
    for row in rows:
        if row["date"] in selected:
            print(json.dumps(row, ensure_ascii=False), flush=True)


def context():
    if (OUT / "policy_context.json").exists():
        raise RuntimeError("已有政策时序补充，不覆盖。")
    records = json.loads((OUT / "operation_documents.json").read_text(encoding="utf-8"))["rows"]
    rates = []
    for row in records:
        for match in re.finditer(r"7\s*天\s*([\d,.]+)\s*亿元\s*([\d.]+)\s*%", row["text"]):
            rates.append({"date": row["date"], "instrument": "七天逆回购", "amount_yi": float(match[1].replace(",", "")),
                          "rate_pct": float(match[2]), "source_url": row["source_url"], "raw_path": row["raw_path"]})
    assert rates
    monthly_coverage = []
    for month in pd.period_range("2019-01", "2019-04", freq="M"):
        selected = [r for r in records if r["date"].startswith(str(month))]
        monthly_coverage.append({"month": str(month), "documents": len(selected)})
    sources = [
        {"id": "nbs_march_pmi", "date": "2019-03-31", "url": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900267.html",
         "required": ["2019", "50.5", "51.6", "1.3"]},
        {"id": "nbs_q1_activity", "date": "2019-04-17", "url": "https://www.stats.gov.cn/sj/xwfbh/fbhwd/202302/t20230203_1900276.html",
         "required": ["2019", "6.4", "一季度"]},
    ]
    saved = []
    for source in sources:
        row = dict(source)
        try:
            response = requests.get(source["url"], timeout=(8, 20), headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            article = soup.select_one(".TRS_Editor") or soup
            body = article.get_text(" ", strip=True)
            page = soup.get_text(" ", strip=True)
            absent = [s for s in source["required"] if s not in body]
            if absent:
                raise ValueError("正文缺少：" + str(absent))
            path = OUT / "sources" / (source["id"] + ".html")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
            path.with_suffix(".txt").write_text(body, encoding="utf-8")
            path.with_suffix(".page.txt").write_text(page, encoding="utf-8")
            row.update(status="RETRIEVED", path=str(path.relative_to(ROOT)), retrieved_at=now(),
                       sha256=hashlib.sha256(response.content).hexdigest())
        except Exception as exc:
            row.update(status="MISSING_SOURCE", error=str(exc), retrieved_at=now())
        saved.append(row)
    save("context_sources.json", saved)
    if any(r["status"] != "RETRIEVED" for r in saved):
        raise RuntimeError("经济背景来源不完整，保留记录，不填写未核实的正文。")
    selected_dates = {"2019-01-15", "2019-01-16", "2019-01-23", "2019-02-15", "2019-04-12", "2019-04-17", "2019-04-18", "2019-04-24", "2019-04-26", "2019-04-30"}
    save("policy_context.json", {
        "recorded_at": now(), "selection_order": "看到固定窗口利率响应后补充同期原因，只用于解释，不定义策略分组。",
        "monthly_document_counts": monthly_coverage,
        "seven_day_reverse_repo_observations": rates,
        "seven_day_reverse_repo_unique_rates": sorted({r["rate_pct"] for r in rates}),
        "key_notices": [r for r in records if r["date"] in selected_dates],
        "growth_context": [
            {"source_id": "nbs_march_pmi", "published_date": "2019-03-31", "pmi": 50.5, "pmi_prior": 49.2,
             "new_orders": 51.6, "role": "4月金融数据公布前已有制造业恢复的信息，不能将4月全部债券变化归给4月12日公告。"},
            {"source_id": "nbs_q1_activity", "published_date": "2019-04-17", "gdp_yoy_initial_pct": 6.4,
             "role": "4月15日原入场后的新增长信息，没有提前放入入场条件，也没有据此重算交易。"},
        ],
        "limits": ["七天逆回购中标利率只是一个政策工具价格，不能证明所有政策姿态均未改变。",
                   "操作公告金额为各工具发放量，没有完整到期和其他负债表变动，不能称净投放或股市资金流入。",
                   "期限差是可观察的两个收益率之差，不等于已识别的期限溢价；利率上行亦未被识别为独立的加息、增长或通胀冲击。",
                   "央行公告标题和网页发布时间属于目前可核验的历史重建，未声称持有不可变更的当年首次快照。"],
    })
    print(json.dumps({"操作公告": len(records), "七天逆回购记录": len(rates), "七天利率": sorted({r["rate_pct"] for r in rates}),
                      "新增经济背景来源": [{"id": r["id"], "status": r["status"]} for r in saved]}, ensure_ascii=False), flush=True)


def draw():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False, "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    daily = pd.read_parquet(OUT / "daily_rates.parquet")
    daily = daily[daily["date"].between("2019-01-02", "2019-05-15")]
    market = pd.read_parquet(ROOT / INPUTS["market"])
    market["date"] = pd.to_datetime(market["date"])
    market = market[market["date"].between("2019-01-02", "2019-05-15")].sort_values("date")
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(3, 1, figsize=(13.6, 11.5), sharex=True)
    fig.patch.set_facecolor("#faf9f5")
    for ax in axes:
        ax.set_facecolor("#faf9f5")
        ax.grid(axis="y", alpha=.16)
        ax.set_axisbelow(True)
        for event in protocol["events"]:
            ax.axvline(pd.Timestamp(event["date"]), color="#687a83", lw=.9, linestyle=":", alpha=.7)
        ax.axvspan(pd.Timestamp("2019-05-06"), pd.Timestamp("2019-05-15"), color="#a55e54", alpha=.08)
    axes[0].plot(daily["date"], daily["fdr007"], color="#aeb7bc", lw=1.2, label="FDR007上午定盘")
    axes[0].plot(daily["date"], daily["fdr007_20obs_mean"], color="#b48345", lw=2.2, label="最近20次定盘均值")
    axes[0].plot(pd.to_datetime(["2019-01-02", "2019-04-30"]), [2.55, 2.55], color="#535f65", linestyle="--", lw=1.2,
                 label="1—4月七天逆回购中标利率2.55%")
    axes[0].set_ylabel("年化利率（%）")
    axes[0].set_ylim(1.7, 3.4)
    axes[0].legend(loc="upper left", frameon=False, ncol=3, fontsize=9.5)
    axes[0].set_title("A  资金价格的日波动，与央行操作利率分开观察", loc="left", fontsize=14, pad=10)
    for event in protocol["events"]:
        axes[0].text(pd.Timestamp(event["date"]), 3.08, event["date"][5:] + "\n金融数据", ha="center", va="top", fontsize=9, color="#58646c")
    axes[1].plot(daily["date"], daily["cgb_10y"], color="#286c6d", lw=2, label="10年国债收益率")
    axes[1].plot(daily["date"], daily["cgb_1y"], color="#b48345", lw=2, label="1年国债收益率")
    axes[1].set_ylabel("年化收益率（%）")
    axes[1].set_ylim(2.15, 3.62)
    axes[1].legend(loc="upper left", frameon=False, ncol=2)
    axes[1].set_title("B  国债定价包含不同期限的信息，不能直接叫作政策加息", loc="left", fontsize=14, pad=10)
    wealth = market["wealth"] / market["wealth"].iloc[0] * 100
    axes[2].plot(market["date"], wealth, color="#286c6d", lw=2.3)
    axes[2].set_ylabel("含分红参考序列\n1月2日＝100")
    axes[2].set_ylim(float(wealth.min()) - 4, float(wealth.max()) + 7)
    axes[2].text(pd.Timestamp("2019-05-06"), float(wealth.max()) + 3, "5月后续冲击", ha="right", color="#9b5b50", fontsize=10)
    axes[2].set_title("C  510300的上涨与国债收益率上升曾经并存", loc="left", fontsize=14, pad=10)
    axes[2].xaxis.set_major_locator(mdates.MonthLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[2].set_xlim(pd.Timestamp("2019-01-02"), pd.Timestamp("2019-05-15"))
    fig.suptitle("510300｜资金价格、国债期限与政策时序", x=.085, y=.975, ha="left", fontsize=20, fontweight="bold")
    fig.text(.085, .94, "固定观察2019年1—4月的四次金融数据发布；保留原20日观察窗口延伸至5月的部分", color="#536068", fontsize=11)
    fig.text(.085, .025, "曲线为历史观察，未将公布后变化提前用于入场。FDR007不是全天DR007，10年减1年亦不是纯期限溢价。\n下图未计交易费，不能当作账户净值或策略夏普。阴影提示后续消息进入窗口，不表示已识别其因果贡献。", color="#536068", fontsize=10)
    fig.subplots_adjust(left=.10, right=.975, top=.88, bottom=.105, hspace=.38)
    fig.savefig(OUT / "资金价格与指数历史时序.png", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def report():
    responses = json.loads((OUT / "event_responses.json").read_text(encoding="utf-8"))["rows"]
    monthends = json.loads((OUT / "monthend_states.json").read_text(encoding="utf-8"))["rows"]
    context = json.loads((OUT / "policy_context.json").read_text(encoding="utf-8"))
    sources = {r["id"]: r for r in json.loads((OUT / "context_sources.json").read_text(encoding="utf-8"))}
    notices = {r["date"]: r for r in context["key_notices"]}
    daily = pd.read_parquet(OUT / "daily_rates.parquet").set_index("date")
    march = daily.loc[pd.Timestamp("2019-03-29")]
    april = daily.loc[pd.Timestamp("2019-04-30")]
    april_before = daily.loc[pd.Timestamp("2019-04-11")]
    delta = {k: float((april[k] - march[k]) * 100) for k in ["fdr007_20obs_mean", "cgb_1y", "cgb_10y"]}
    prior_rise = float((april_before["cgb_10y"] - march["cgb_10y"]) * 100)
    february = next(r for r in responses if r["stat_month"] == "2019-01" and r["horizon"] == 20)
    five_april = next(r for r in responses if r["stat_month"] == "2019-03" and r["horizon"] == 5)
    local = lambda p, label: f"[{label}](<{p.as_posix()}>)"
    notice = lambda date, label: f"[{label}]({notices[date]['source_url']})"
    external = lambda key, label: f"[{label}]({sources[key]['url']})"
    lines = ["# 510300历史发现：资金价格、国债期限与政策时序", "",
        "研究单位为沪深300整体，以510300作为可交易价格观察。本轮按2019年1月至4月的四次金融数据发布日期固定比较，补出资金价格和国债收益率变化的来源；全部四次事件和1、5、20日终点均保留。", "",
        "**已经有证据的发现**", "",
        "一是，融资改善、银行资金价格、央行工具利率并不同步。二是，指数上涨和国债收益率上升曾经同时出现，不能将利率上行机械翻译成指数利空。三是，4月的国债上行并非都在强信用数据公布后才开始，政策操作还包含税期、期限替换及财政收支的影响。", "",
        "这些发现改变了因子的解释方式，尚未形成稳定收益规则。本轮没有证明哪一条通道主导股价，也没有据此改造旧失败策略。", "",
        "**先分清三种价格**", "",
        "FDR007是存款类机构以利率债作质押、上午交易形成的七天回购定盘参考利率，11:30起发布；它不是全天加权DR007。央行七天逆回购中标利率是另一种工具价格，国债1年及10年收益率则是不同期限的市场估值。三者有联系，却不可以互相替换。[中国货币网编制方案](https://www.chinamoney.com.cn/chinese/bkfrr/)。", "",
        "在本地83份1—4月央行操作公告中，提取到20笔七天逆回购操作，中标利率均为2.55%。这是已取得操作的事实，不等于所有工具或全部政策姿态不变。以下月末表按交易日历固定；20次均值是最近20个有效资金利率观测的平均，不按结果调整长度。", "",
        "| 月末交易日 | FDR007当日 | FDR007近20次均值 | 1年国债 | 10年国债 | 10年减1年 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in monthends:
        lines.append(f"| {row['date'][:10]} | {row['fdr007']:.4f}% | {row['fdr007_20obs_mean']:.4f}% | {row['cgb_1y']:.4f}% | {row['cgb_10y']:.4f}% | {row['slope_10y_1y_bp']:.2f} bp |")
    lines += ["",
        f"3月末至4月末，1年国债上行{delta['cgb_1y']:.2f} bp、10年上行{delta['cgb_10y']:.2f} bp，而FDR007近20次均值上行{delta['fdr007_20obs_mean']:.2f} bp。国债市场出现明显调整，但不能把幅度不同的价格都归成同一个‘流动性因子’。期限差包含多种预期和溢价，本轮没有将其分解为纯粹期限溢价。", "",
        "**操作数量为什么变：原始公告提供了具体约束**", "",
        "1月15日公告明确提到税收高峰造成银行体系流动性下降，同时用降准置换部分MLF，并开展1800亿元逆回购；1月16日又针对税期投放5700亿元。大额操作中有对冲和替换的成分，不能把总额等同于新增股票购买力。" + notice("2019-01-15", "1月15日公告") + "、" + notice("2019-01-16", "1月16日公告") + "。", "",
        "2月15日没有逆回购，公告说明现金回笼与逆回购到期、准备金缴存等因素对冲后，流动性仍较充裕。暂停某一工具，不足以单独判定紧缩。" + notice("2019-02-15", "2月15日公告") + "。", "",
        "4月17日，央行根据资金需求的期限结构开展1600亿元七天逆回购及2000亿元一年MLF，操作利率分别2.55%与3.30%；4月18日又明确对冲税期高峰。4月24日2674亿元TMLF的利率为3.15%，与1月23日首次操作相同；这次操作不能再命名为一次新的15 bp降息。" + notice("2019-04-17", "4月17日公告") + "、" + notice("2019-04-18", "税期公告") + "、" + notice("2019-04-24", "4月TMLF") + "、" + notice("2019-01-23", "1月TMLF") + "。", "",
        "4月末不开展逆回购的公告，转而解释为财政支出增加能够对冲到期资金，或流动性总量较高。这说明同样的‘不操作’，背后可以有不同的资产负债表原因。" + notice("2019-04-26", "4月26日公告") + "、" + notice("2019-04-30", "4月30日公告") + "。本轮没有完整到期和全部资金来源账本，上述操作量均不是净投放统计。", "",
        "**增长信息与政策节奏：先后关系不能倒置**", "",
        "3月31日，官方制造业PMI已从49.2升至50.5，新订单指数为51.6，早于4月12日金融数据。4月17日又公布一季度GDP初步同比增长6.4%，这是4月15日原入场之后出现的信息。制造业恢复和总量增长信息为增长预期变化提供了背景，但不是沪深300全部行业利润的直接测量，也没有证明收益率变化由它们单独造成。" + external("nbs_march_pmi", "3月PMI原文") + "、" + external("nbs_q1_activity", "4月17日经济发布") + "。", "",
        f"从固定3月末节点至4月11日，10年国债收益率已由{march['cgb_10y']:.4f}%升至{april_before['cgb_10y']:.4f}%，上涨{prior_rise:.2f} bp。这部分早于4月12日信用数据公布，不可以全部归为那条消息的反应。4月12日同期采访中已经出现‘经济企稳更有依据、近期额外宽松不再那么急’的看法，详见" + local(PRIOR / "历史发现_信用预期差与指数传导.md", "上一轮预期重建") + "。", "",
        "因此可以保留的机制假设是：增长信息会影响整体现金流和利率预期；财政税期及银行负债需求会影响短期资金价格；央行的工具选择又对这些条件作出反应。它们有内生关系，不能靠利率正负号就识别政策冲击。本轮未取得足以分离实际短率预期、通胀预期和期限溢价的数据。", "",
        "**固定窗口中的利率与指数表现**", "",
        "下表利率基准为发布日期之前最后一个ETF交易日；终点为发布日期之后第20个ETF交易日。两端之间的变化包含公布当日，不能称为只发生在公布后的反应；原范围文件中‘公布当日整日排除’的表述应理解为不选择当日作为响应终点，已另存澄清且没有改变任何端点或数值。FDR007当天只有上午定盘，国债为该日估值。原ETF收益仍为次日开盘进入、20个交易日后开盘退出，二者并非完全相同的盘中测量区间，也不应读成纯粹公告因果效应。", "",
        "| 金融数据公开日 | 利率比较起止 | FDR近20次均值变化 | 1年国债变化 | 10年国债变化 | 原20日事件净收益 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in responses:
        if row["horizon"] == 20:
            lines.append(f"| {row['release_date']} | {row['before']['date'][:10]}→{row['after']['date'][:10]} | {row['fdr007_20obs_mean_change_bp']:+.2f} bp | {row['cgb_1y_change_bp']:+.2f} bp | {row['cgb_10y_change_bp']:+.2f} bp | {row['old_event_net_return_20d']:+.2%} |")
    lines += ["",
        f"2月这一行还有一个易误读之处：FDR007单点从2.10%到2.75%，上行{february['fdr007_change_bp']:.0f} bp；相同起止时刻的20次均值却下降{-february['fdr007_20obs_mean_change_bp']:.2f} bp。春节前后低点与之后常态的端点比较，和一段时期的平均资金成本，并不是同一个问题。不能依据其中较符合股价的那个口径事后选因子，本轮两个结果都保留。", "",
        f"4月信用公布后第5个交易日，1年国债相对基准上行{five_april['cgb_1y_change_bp']:.2f} bp、10年上行{five_april['cgb_10y_change_bp']:.2f} bp，原5日ETF事件净收益仍为{five_april['old_event_net_return_5d']:+.2%}。延长至原20日窗口后，指数亏损且10年国债收益率又回落到基准附近。这说明路径和后续消息不能被一个终点符号代替。", "",
        "原4月入场至5月16日退出的窗口完整保留，含5月贸易冲击；此前研究也已确认4月15日至4月30日存在下跌。没有删除坏月份、回避外部冲击或事后选退出点。同期政策、增长新闻和利率变化可以作为竞争解释，尚不能分配各自的因果贡献。", "",
        "**本轮对后续交易研究的实际约束**", "",
        "不把‘央行投放越多’、‘FDR某天上涨’或‘国债收益率上行’直接转换为指数仓位。先区别税期与现金回笼的短期波动、融资需求与增长判断、政策工具利率和市场对未来政策节奏的判断。然后再问当时的指数价格，是否已经吸收了这些变化。", "",
        "下一项历史研究转向同一段指数的整体估值与盈利口径：在1月至4月已固定时间线上，区分价格上涨伴随的估值扩张和当时已公布的盈利变化，并核对这些指标的可得日期。只看指数汇总，不扩展个股；如果历史口径不足，保留缺失，不用后来财报倒填。", "",
        "本轮只核对单位、日期、来源定位和固定窗口，保存12行利率响应与5个月末状态；没有多轮参数搜索或新账户回测。原事件采用10万元预算及原有成本，分母为实际投入资金；重叠窗口不是独立试验，亦不是20万元完整账户。净夏普1.2与年化10%的目标仍未达成。", "",
        "历史曲线通过原始中债来源的既有文件复用，FDR通过中国货币网既有记录复用；字段叫作first_release_value不等于持有不可修改的当年首次快照。中债曲线统一按下一ETF交易日开盘才可使用，本次所有公布后终点仅用于解释历史路径，没有放回原入场。", "",
        "文件：" + "、".join([local(OUT / "event_responses.json", "全部响应与可得日期"), local(OUT / "monthend_states.json", "月末状态"),
                                       local(OUT / "policy_context.json", "政策操作与经济背景"), local(OUT / "operation_documents.json", "83份公告定位"),
                                       local(OUT / "daily_rates.parquet", "日度利率"), local(OUT / "资金价格与指数历史时序.png", "历史时序图"),
                                       local(ROOT / "research/historical_index_rates_policy_transmission_v1.py", "研究脚本")]) + "。", "",
    ]
    name = "历史发现_资金价格与指数政策传导.md"
    (OUT / name).write_text("\n".join(lines), encoding="utf-8")
    save("mechanism_findings.json", {"computed_at": now(), "april_calendar_changes_bp": delta,
                                     "ten_year_rise_before_april_credit_release_bp": prior_rise,
                                     "february_fdr_point_change_bp": february["fdr007_change_bp"],
                                     "february_fdr_20obs_mean_change_bp": february["fdr007_20obs_mean_change_bp"],
                                     "causal_contribution_identified": False, "new_rules_admitted": 0})
    draw()
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    result.update(report=name, report_completed_at=now(), policy_documents=83,
                  discovery="四个固定窗口显示资金单点、平均成本、国债期限与指数表现不同步；4月国债上行早于信用公布，同期七天逆回购中标利率未变且公告记录税期与财政原因，不能直接认定央行加息或单因子信号。",
                  next_historical_question="继续同一2019年1至4月指数时间线，检查沪深300整体估值、价格与当时已公布盈利口径，区分估值扩张与盈利变化；只做指数汇总，保留历史可得性与分母变化，不用后来财报倒填或救回旧估值策略。")
    save("result.json", result)
    print("已生成资金价格、国债期限与指数传导报告及图表；没有新增规则或达标声明。", flush=True)


def record_progress():
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    assert (OUT / result["report"]).is_file()
    prefix = "reports/research/510300_historical_index_rates_policy_transmission_v1/"
    stamp, changes = now(), []
    for name in ["510300_historical_cause_discovery_v1.json", "510300_existing_data_training_mandate_v1.json"]:
        path = ROOT / "config" / name
        cfg = json.loads(path.read_text(encoding="utf-8"))
        before = dict(cfg)
        if name == "510300_historical_cause_discovery_v1.json":
            cfg.update(current_study=prefix + "protocol.json", latest_completed_study=prefix + "result.json", latest_report=prefix + result["report"], updated_at=stamp)
        else:
            cfg.update(current_round=STUDY, latest_progress_receipt=prefix + "result.json",
                       latest_historical_index_rates_policy_transmission=prefix + "result.json",
                       latest_continuation_report=prefix + result["report"], latest_historical_report=prefix + result["report"],
                       latest_continuation_classification=result["classification"], current_driver_continuation_classification=result["classification"],
                       current_driver_consecutive_blocked_goal_turns=0, latest_historical_diagnostic_at=stamp,
                       latest_goal_service_status="active", latest_goal_service_status_observed_at=stamp, goal_status="active", goal_achieved=False,
                       local_goal_work_status="ACTIVE_HISTORICAL_ONLY", last_research_result=result["discovery"],
                       last_source_result="复用原始利率历史，核对83份同期央行操作公告，并补存统计局2019年3月PMI及一季度经济原文。",
                       next_research_question=result["next_historical_question"])
        changes.append({"path": str(path.relative_to(ROOT)), "fields": {k: {"before": before.get(k), "after": v} for k, v in cfg.items() if before.get(k) != v}})
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save("authority_update.json", {"recorded_at": stamp, "study_id": STUDY, "previous_goal_turn_classification": "PROGRESS",
                                  "current_goal_turn_classification": "PROGRESS_COMPLETED_INDEX_RATES_POLICY_HISTORY", "changes": changes,
                                  "goal_achieved": False, "orders_authorized": False})
    print("已登记指数利率与政策历史研究；完整账户目标保持未达成。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="指数利率与政策时序的历史研究")
    parser.add_argument("mode", choices=["prepare", "build", "operations", "context", "report", "record-progress"])
    args = parser.parse_args()
    {"prepare": prepare, "build": build, "operations": operations, "context": context,
     "report": report, "record-progress": record_progress}[args.mode]()


if __name__ == "__main__":
    main()
