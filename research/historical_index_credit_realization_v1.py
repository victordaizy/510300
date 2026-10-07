"""固定半年融资渠道、贷款期限与用途的指数历史比较。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
from research.historical_index_sector_repricing_v1 import clean, now, rel, digest, read, save
from research.credit_recency_structure_v23 import Ledger, FIELDS
from research import historical_index_reopening_constraints_v1 as execution

OUT = ROOT / "reports/research/510300_historical_index_credit_realization_v1"
MONTHS = [str(p) for p in pd.period_range("2022-10", "2023-03", freq="M")]
LOAN_LABELS = {"rmb_total": "人民币贷款合计", "corporate_total": "企事业贷款", "corporate_long": "企事业中长期", "corporate_short": "企事业短期", "bills": "已贴现票据融资", "household_total": "住户贷款", "household_long": "住户中长期", "household_short": "住户短期", "nonbank_total": "非银金融机构贷款"}
TSF_LABELS = {"rmb_real": "对实体经济发放的人民币贷款", "fx_real": "对实体经济发放的外币贷款折合人民币", "entrusted": "委托贷款", "trust": "信托贷款", "undiscounted_bills": "未贴现的银行承兑汇票", "corporate_bonds": "企业债券净融资", "government_bonds": "政府债券净融资", "equity": "非金融企业境内股票融资"}
FLOW_CODES = ["aa9f6708343245359814138dad70f74a", "905a82aff74f46b2a85829522d016afc", "76eb157c3e964915b9adcba2f4d0c643", "08913263a9fc4b398ca36a34d3356ed7", "10efa07960ae4dc687872f7a52489a7c", "c512e1903d2940028ca5285cc9a2287e"]
SOURCES = [{"id": "tsf_"+m.replace("-", ""), "month": m, "kind": "社融原报告", "url": "https://www.pbc.gov.cn/diaochatongjisi/116219/116225/"+code+"/index.html"} for m, code in zip(MONTHS, FLOW_CODES)] + [
    {"id": "purpose_2022q3", "kind": "贷款投向原报告官方转载", "period": "2022Q3", "page_date": "2022-10-31", "url": "https://jrj.sh.gov.cn/SCGK194/20221031/46df3ac35122452fb5c72fef1e3e1b10.html"},
    {"id": "purpose_2022q4", "kind": "贷款投向原报告官方转载", "period": "2022Q4", "page_date": "2023-02-04", "url": "https://app.www.gov.cn/govdata/gov/202302/04/496752/article.html"},
    {"id": "purpose_2023q1", "kind": "贷款投向原报告官方转载", "period": "2023Q1", "page_date": "2023-05-04", "url": "https://jrj.beijing.gov.cn/jrgzdt/202305/t20230504_3085545.html"},
    {"id": "briefing_2023q1", "kind": "央行金融统计发布会", "page_date": "2023-04-20", "url": "https://xining.pbc.gov.cn/goutongjiaoliu/113456/113469/2025092212553125171/index.html"},
]
INPUTS = ["reports/research/510300_credit_recency_structure_v23/inputs/originals.json", "reports/research/510300_credit_recency_structure_v23/inputs/monthly.csv", "reports/research/510300_historical_index_sector_repricing_v1/monthly_comparison.parquet", "reports/research/510300_historical_index_sector_repricing_v1/policy_facts.json", "reports/research/510300_factor96_t11_date_proxy_account_v1/inputs/market.parquet", "reports/research/510300_factor96_t11_date_proxy_account_v1/inputs/dividends.csv"]


def prepare():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("保留已有研究方案。")
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {"study_id": "510300_HISTORICAL_INDEX_CREDIT_REALIZATION_V1", "created_at": now(), "previous_goal_turn": "PROGRESS_FIXED_SECTOR_COMOVEMENT_AND_CREDIT_CONSTRAINT_CONTEXT", "question": "2022年末融资支持后，贷款期限、融资渠道与用途是否共同改善；总量变化在哪些环节不能等同于最终需求或指数剩余收益？", "mode": "HISTORICAL_INDEX_DISCOVERY_ONLY", "months": MONTHS, "quarterly_purpose_observations": ["2022Q3", "2022Q4", "2023Q1"], "why_calendar": "11月融资政策之前一个统计月、政策月及随后完整一季度，全部保留。此前月度行情和若干宏观事实已经见过，不是盲测。", "deduplication": {"V23_V26": "已有104个月贷款期限与公布后价格；仅复用本半年作为背景。", "historical_credit_flow_v1": "已有2018Q4至2019Q1渠道分解与失败总量表达；不恢复单独信用总量买入规则。", "new_increment": "六次社融原报告的渠道与当时同比，三个固定季度的贷款用途及央行上游解释，连接已见指数共同重估。"}, "loan_reuse": "沿用V23区间代数与原公告；单月和年内累计都保存，2023扩围不跨旧口径硬算分项同比。", "tsf": "原文直接月度分项优先；仅披露全年、一季度分项时保留累计范围，单月合计单列，不倒用下一期修订填月度分项。同比使用同页官方可比口径。", "purpose": "工业、服务业、基础设施等可能交叉，不相加；本外币与人民币不混合；期限与用途区分。", "event_windows": [5, 20], "event_clock": "六次贷款报告的原公开日上界后下一实际开盘，原成本、分红与整手；社融与用途来源另列时钟，晚到信息不回填。", "quarterly_context": "所取得官方转载日期只作为保守可见上界，不称最早公布日；4/5月的用途及解释不进入此前1/2月信息。", "market_end": "2023-05-31", "cost": execution.COST, "new_models": 0, "new_full_accounts": 0, "new_prospective_tasks": 0, "orders": 0, "independent_validation": False, "stop": "不把总量、多增、长贷占比或看到的赢家月份改为新的阈值或拼接策略；未识别净新发放和滚续金额就保留未知。", "goal_achieved": False}
    save(OUT / "protocol.json", protocol)
    save(OUT / "freeze.json", {"created_at": now(), "protocol_sha256": digest(OUT / "protocol.json")})
    save(OUT / "input_receipts.json", [{"path": p, "sha256": digest(ROOT / p)} for p in INPUTS])
    save(OUT / "source_plan.json", SOURCES)
    print("已固定六个月渠道比较、三个季度用途观察及原5/20间隔窗口。")


def collect():
    if (OUT / "source_manifest.json").exists():
        raise RuntimeError("已有采集记录，不重复请求。")
    (OUT / "sources").mkdir(exist_ok=True)
    def fetch(spec):
        row = {**spec, "retrieved_at": now()}
        try:
            response = requests.get(spec["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=(10, 30))
            p = OUT / "sources" / (spec["id"]+".html")
            p.write_bytes(response.content)
            row.update(http_status=response.status_code, path=rel(p), sha256=digest(p), bytes=len(response.content))
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser", from_encoding="utf-8")
            for tag in soup(["script", "style"]):
                tag.decompose()
            body = soup.get_text("\n", strip=True)
            p.with_suffix(".txt").write_text(body, encoding="utf-8")
            row.update(status="RETRIEVED", text_path=rel(p.with_suffix(".txt")), text_chars=len(body), date_candidates=re.findall(r"20\d{2}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?", body)[:8])
        except requests.RequestException as error:
            row.update(status="FAILED", error=str(error)[:300])
        return row
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(fetch, SOURCES))
    save(OUT / "source_manifest.json", rows)
    print(json.dumps([{k:v for k,v in r.items() if k in ["id", "status", "bytes", "text_chars", "date_candidates", "error"]} for r in rows], ensure_ascii=False, indent=2))


def loans():
    ledger = Ledger()
    rows = [ledger.row(month, field, window) for month in MONTHS for window in ["1", "YTD"] for field in FIELDS]
    for row in rows:
        row["label"] = LOAN_LABELS[row["field"]]
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "loan_intervals.parquet", index=False)
    df.to_csv(OUT / "loan_intervals.csv", index=False, encoding="utf-8-sig")
    selected = [ledger.by_month[m] for m in MONTHS]
    save(OUT / "loan_sources.json", selected)
    print(df[df.window.eq("1")].pivot(index="stat_month", columns="label", values="value_yi").to_string())


def money(number, unit):
    scale = 10000. if unit == "万亿元" else 1.
    decimals = len(number.split(".")[1]) if "." in number else 0
    return float(number)*scale, scale*.5*10**(-decimals)


def yoy(tail):
    m = re.search(r"(?:同比|比上年同期|比上年)(多增|少增|多减|少减|多|少)([\d.]+)(万亿元|亿元)", tail[:70])
    if m is None:
        return None
    value, error = money(m[2], m[3])
    return {"value_yi": value*(-1 if m[1] in ["少增", "多减", "少"] else 1), "rounding_half_yi": error, "literal": m[0]}


def tsf():
    manifest = {x["id"]: x for x in read(OUT / "source_manifest.json")}
    blocks, headlines = [], []
    for month in MONTHS:
        source = manifest["tsf_"+month.replace("-", "")]
        assert source["status"] == "RETRIEVED"
        body = "".join((ROOT / source["text_path"]).read_text(encoding="utf-8").split())
        body = body[body.index("初步统计"):]
        header = re.search(r"社会融资规模增量(?:累计)?为([\d.]+)(万亿元|亿元)", body)
        assert header is not None, month
        total, total_rounding = money(header[1], header[2])
        scope = "YTD" if "累计" in header[0] else "MONTH"
        fields = {}
        for key, label in TSF_LABELS.items():
            pattern = re.escape(label)+r"(增加|减少|为)?(-?[\d.]+)(万亿元|亿元)"
            m = re.search(pattern, body)
            assert m is not None, (month, key)
            value, rounding = money(m[2], m[3])
            if m[1] == "减少":
                value = -value
            fields[key] = {"label": label, "value_yi": value, "rounding_half_yi": rounding, "literal": m[0], "yoy_change": yoy(body[m.end():])}
        block = {"month": month, "scope": scope, "total_yi": total, "total_rounding_half_yi": total_rounding, "total_yoy_change": yoy(body[header.end():]), "fields": fields, "other_and_rounding_yi": total-sum(x["value_yi"] for x in fields.values()), "source_url": source["url"], "source_sha256": source["sha256"], "date_candidates": source["date_candidates"]}
        blocks.append(block)
        if scope == "MONTH" or month.endswith("-01"):
            headline = {"month": month, "value_yi": total, "yoy_change": block["total_yoy_change"], "component_scope": "DIRECT_MONTH"}
        else:
            pattern = str(int(month[-2:]))+r"月份?[，,]?社会融资规模增量为([\d.]+)(万亿元|亿元)"
            m = re.search(pattern, body)
            assert m is not None, (month, "月度合计")
            value, half = money(m[1], m[2])
            headline = {"month": month, "value_yi": value, "rounding_half_yi": half, "yoy_change": yoy(body[m.end():]), "component_scope": "CUMULATIVE_COMPONENTS_ONLY"}
        headlines.append(headline)
    save(OUT / "tsf_composition.json", {"blocks": blocks, "monthly_headlines": headlines})
    print(json.dumps({"区间": [{"月":b["month"], "分项范围":b["scope"], "合计亿元":b["total_yi"], "未列及舍入亿元":b["other_and_rounding_yi"]} for b in blocks], "月度合计": headlines}, ensure_ascii=False, indent=2))


def events():
    protocol = read(OUT / "protocol.json")
    assert digest(OUT / "protocol.json") == read(OUT / "freeze.json")["protocol_sha256"]
    old = read(execution.OUT / "protocol.json")
    old["account_calendar"][1] = protocol["market_end"]
    market, _, dividends, engine, _ = execution.inputs(old)
    outputs, clocks = [], []
    sources = read(OUT / "loan_sources.json")
    for row in sources:
        cutoff = pd.Timestamp(row["conservative_known_at"])
        entry = int(np.flatnonzero(market.date.gt(cutoff.tz_localize(None).normalize()))[0])
        clock = {"event_id": "LOAN_RELEASE_"+row["stat_month"], "title": row["stat_month"]+"贷款原报告", "entry_idx": entry, "source_url": row["source_url"], "public_at": row["published_at"], "known_at": row["conservative_known_at"], "entry_date": market.date.iloc[entry]}
        clocks.append(clock)
        for horizon in protocol["event_windows"]:
            result = execution.fixed_event_window(market, dividends, engine, clock, horizon)
            assert result["status"] == "两端成交"
            outputs.append(result)
    save(OUT / "event_clocks.json", clocks)
    save(OUT / "event_returns.json", outputs)
    print(pd.DataFrame(outputs)[["title", "horizon", "entry_date", "exit_date", "net_return"]].to_string(index=False))


def purpose():
    manifest = {x["id"]: x for x in read(OUT / "source_manifest.json")}
    patterns = {
        "企事业中长期": r"中长期贷款余额", "工业中长期": r"本外币工业中长期贷款余额",
        "基础设施中长期": r"本外币基础设施中长期贷款余额", "固定资产贷款": r"固定资产贷款余额",
        "房地产贷款合计": r"人民币房地产贷款\d?余额", "房地产开发贷款": r"房地产开发贷款余额",
        "个人住房贷款": r"个人住房贷款余额", "住户经营贷款": r"本外币住户经营性贷款余额",
        "住户非住房消费贷款": r"住户消费性贷款[（(]不含个人住房贷款[）)]余额", "住户贷款合计": r"本外币住户贷款余额",
    }
    rows = []
    for source_id in ["purpose_2022q3", "purpose_2022q4", "purpose_2023q1"]:
        source = manifest[source_id]
        text = "".join((ROOT / source["text_path"]).read_text(encoding="utf-8").split())
        text = text[text.index("人民银行统计"):]
        for label, pattern in patterns.items():
            m = re.search(pattern+r"([\d.]+)(万亿元|亿元)[，,]同比(增长|下降)([\d.]+)%", text)
            assert m is not None, (source_id, label)
            amount, rounding = money(m[1], m[2])
            rate = float(m[4])*(-1 if m[3] == "下降" else 1)
            currency = "人民币" if label in ["房地产贷款合计", "房地产开发贷款", "个人住房贷款"] else "本外币"
            rows.append({"source_id": source_id, "period": source["period"], "label": label, "currency": currency, "stock_yi": amount, "rounding_half_yi": rounding, "reported_stock_yoy_pct": rate, "literal": m[0], "available_upper_bound": source["page_date"]+"T23:59:59+08:00", "source_url": source["url"], "source_sha256": source["sha256"], "first_publication_verified": False, "used_as_entry_input": False})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "loan_purpose.csv", index=False, encoding="utf-8-sig")
    save(OUT / "loan_purpose.json", rows)
    source = manifest["briefing_2023q1"]
    text = "".join((ROOT / source["text_path"]).read_text(encoding="utf-8").split())
    specs = [
        ("经营", "短期", r"住户短期经营贷款余额是"),
        ("经营", "中长期", r"中长期经营贷款余额是"),
        ("消费", "短期", r"住户的短期消费贷款余额是"),
        ("消费", "中长期", r"住户的中长期消费贷款余额是"),
    ]
    parts = []
    for use, term, pattern in specs:
        m = re.search(pattern+r"([\d.]+)(万亿元|亿元)[，,]比年初增加了?([\d.]+)(万亿元|亿元)[，,]同比(多增|少增)([\d.]+)(万亿元|亿元)", text)
        assert m is not None, (use, term)
        stock, _ = money(m[1], m[2])
        amount, _ = money(m[3], m[4])
        delta, _ = money(m[6], m[7])
        parts.append({"use": use, "term": term, "stock_yi": stock, "net_increase_yi": amount, "official_yoy_increase_yi": delta*(-1 if m[5] == "少增" else 1), "literal": m[0]})
    p = pd.DataFrame(parts)
    long = p[p.term.eq("中长期")]
    share = float(long.loc[long.use.eq("经营"), "net_increase_yi"].sum()/long.net_increase_yi.sum())
    assert int(long.net_increase_yi.sum()) == 9442
    assert int(p.loc[p.use.eq("消费"), "net_increase_yi"].sum()) == 3148
    assert int(p.loc[p.use.eq("消费"), "official_yoy_increase_yi"].sum()) == -571
    anchors = ["购房需求尚未完全恢复", "支持个体工商户和小微企业主的生产经营活动", "结构性货币政策工具发挥的是牵引带动作用"]
    assert all(a in text for a in anchors)
    findings = {"period": "2023Q1", "source_url": source["url"], "source_path": source["path"], "source_sha256": source["sha256"], "public_at": "2023-04-20T18:55:00+08:00", "rows": parts, "long_business_share_of_same_source_long_increase": share, "source_scope": "同一发布会住户部门用途与期限分项；段落未单独写币种，不与人民币月报拼接计算。", "source_explanation": "央行将经营贷款多增联系到个体工商户和小微企业支持，将短期消费多增联系到居民需求回升，将中长期消费少增联系到购房需求尚未完全恢复。", "not_identified": ["新发放总额和还本总额", "逐笔借新还旧金额", "全部中长期消费贷款中住房与其他用途的流量", "贷款与股票实际资金流的对应", "这些解释的独立因果效应"], "used_as_entry_input": False}
    save(OUT / "household_purpose_bridge.json", findings)
    print(df.pivot(index="label", columns="period", values="reported_stock_yoy_pct").to_string())
    print(json.dumps({"住户中长期增加中经营用途占比": share, "四类分项": parts}, ensure_ascii=False, indent=2))


def compare():
    tsf_data = read(OUT / "tsf_composition.json")
    events_data = read(OUT / "event_returns.json")
    clocks = read(OUT / "event_clocks.json")
    rows, contributions = [], []
    for headline, block, clock in zip(tsf_data["monthly_headlines"], tsf_data["blocks"], clocks):
        assert headline["month"] == block["month"] == clock["event_id"].removeprefix("LOAN_RELEASE_")
        source_public = pd.Timestamp(block["date_candidates"][0], tz="Asia/Shanghai")
        assert source_public <= pd.Timestamp(clock["known_at"])
        matched = {x["horizon"]: x for x in events_data if x["event_id"] == clock["event_id"]}
        row = {"month": headline["month"], "tsf_yi": headline["value_yi"], "official_yoy_change_yi": headline["yoy_change"]["value_yi"], "component_scope": headline["component_scope"], "loan_public_at": clock["public_at"], "tsf_public_at": source_public, "entry_date": clock["entry_date"]}
        for horizon, event in matched.items():
            row[f"net{horizon}"] = event["net_return"]
            row[f"exit{horizon}"] = event["exit_date"]
        rows.append(row)
        if block["total_yoy_change"] is not None:
            delta = block["total_yoy_change"]["value_yi"]
            parts = [x["yoy_change"]["value_yi"] for x in block["fields"].values() if x["yoy_change"] is not None]
            assert len(parts) == 8
            contributions.append({"month": block["month"], "scope": block["scope"], "official_total_yoy_yi": delta, "sum_listed_yoy_yi": sum(parts), "other_and_rounding_yoy_yi": delta-sum(parts)})
    pd.DataFrame(rows).to_csv(OUT / "release_comparison.csv", index=False, encoding="utf-8-sig")
    save(OUT / "tsf_yoy_residuals.json", contributions)
    monthly = pd.DataFrame(rows)
    save(OUT / "descriptive_summary.json", {"events": len(rows), "net20_positive": int(monthly.net20.gt(0).sum()), "net20_mean": monthly.net20.mean(), "not_independent": True, "is_full_account": False, "headline_rule_reopened": False})
    print(monthly[["month", "tsf_yi", "official_yoy_change_yi", "net5", "net20"]].to_string(index=False))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"], "axes.unicode_minus": False, "font.size": 10})
    purpose_rows = pd.DataFrame(read(OUT / "loan_purpose.json"))
    table = purpose_rows.pivot(index="label", columns="period", values="reported_stock_yoy_pct")
    order = ["企事业中长期", "工业中长期", "基础设施中长期", "固定资产贷款", "房地产贷款合计", "房地产开发贷款", "个人住房贷款", "住户贷款合计", "住户经营贷款", "住户非住房消费贷款"]
    table = table.reindex(order)
    household = pd.DataFrame(read(OUT / "household_purpose_bridge.json")["rows"])
    tsf_data = read(OUT / "tsf_composition.json")["blocks"][-1]
    remainder = read(OUT / "tsf_yoy_residuals.json")[-1]["other_and_rounding_yoy_yi"]
    fig = plt.figure(figsize=(15, 11), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.45, 1])
    ax = fig.add_subplot(grid[0, :])
    y = np.arange(len(order))
    for i, (_, row) in enumerate(table.iterrows()):
        ax.plot(row.to_numpy(), [i]*len(row), color="#c9d0d6", lw=1.3, zorder=1)
    for offset, (period, color) in enumerate(zip(table.columns, ["#8b98a3", "#338b91", "#c57831"])):
        ax.scatter(table[period], y, s=65, color=color, label=period, zorder=3)
        for i, value in enumerate(table[period]):
            ax.annotate(f"{value:.1f}", (value, i), xytext=(0, -15 if offset == 0 else 8 if offset == 1 else -15), textcoords="offset points", ha="center", fontsize=8, color=color)
    ax.set_yticks(y, order)
    ax.set_ylim(len(order)-.2, -.9)
    ax.set_xlim(-1, 35)
    ax.set_xlabel("原报告贷款余额同比增速 / %；各类用途存在交叉，不能相加")
    ax.legend(loc="lower right", ncols=3, frameon=False)
    ax.grid(axis="x", color="#e5e7eb")
    ax.set_title("① 企业、开发商、居民经营与居民消费的恢复速度不同", loc="left", fontsize=14, weight="bold", pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax2 = fig.add_subplot(grid[1, 0])
    x = np.arange(len(household))
    ax2.bar(x-.19, household.net_increase_yi, width=.36, color="#32748e", label="一季度净增加")
    ax2.bar(x+.19, household.official_yoy_increase_yi, width=.36, color="#c57831", label="同比多增 / 少增")
    for i, row in enumerate(household.itertuples()):
        ax2.text(i-.19, row.net_increase_yi+230, f"{row.net_increase_yi:.0f}", ha="center", fontsize=9)
        value = row.official_yoy_increase_yi
        ax2.text(i+.19, value+(230 if value >= 0 else -600), f"{value:+.0f}", ha="center", fontsize=9)
    ax2.set_xticks(x, household["use"]+"\n"+household.term)
    ax2.axhline(0, color="#555555", lw=.8)
    ax2.set_ylim(-6500, 9000)
    ax2.set_ylabel("亿元")
    ax2.set_title("② 2023Q1住户用途 × 期限分解", loc="left", fontsize=13, weight="bold", pad=14)
    ax2.legend(loc="lower left", ncols=2, frameon=False, fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)
    ax3 = fig.add_subplot(grid[1, 1])
    labels = ["实体人民币贷款", "实体外币贷款", "委托贷款", "信托贷款", "未贴现承兑汇票", "企业债券", "政府债券", "境内股票融资", "未列其他及舍入"]
    values = [part["yoy_change"]["value_yi"] for part in tsf_data["fields"].values()]+[remainder]
    ax3.barh(np.arange(len(values)), values, color=["#32748e" if v>=0 else "#b36356" for v in values])
    for i, value in enumerate(values):
        ax3.text(value+(250 if value>=0 else -250), i, f"{value:+,.0f}", ha="left" if value>=0 else "right", va="center", fontsize=9)
    ax3.set_yticks(np.arange(len(labels)), labels)
    ax3.invert_yaxis()
    ax3.axvline(0, color="#555555", lw=.8)
    ax3.set_xlim(-9500, 30500)
    ax3.set_xlabel("同比多增 / 少增，亿元；社融合计 +24,700亿元")
    ax3.set_title("③ 2023Q1社融多增的渠道构成", loc="left", fontsize=13, weight="bold", pad=14)
    ax3.spines[["top", "right"]].set_visible(False)
    fig.suptitle("指数融资背景：总量、期限与用途需要分别观察", fontsize=18, weight="bold")
    fig.supxlabel("资料均为历史公告。季度用途及4月发布会属于后来确认，未回填到此前入场；贷款净增不能分离新发放与借新还旧。", fontsize=10, color="#4b5563")
    fig.savefig(OUT / "融资构成_期限与实际用途.png", dpi=160, facecolor="white")
    plt.close(fig)
    print("已生成季度用途、住户分解及社融渠道图。")


def publish():
    protocol = read(OUT / "protocol.json")
    assert digest(OUT / "protocol.json") == read(OUT / "freeze.json")["protocol_sha256"]
    manifest = {x["id"]: x for x in read(OUT / "source_manifest.json")}
    assert len(manifest) == 10 and all(x["status"] == "RETRIEVED" for x in manifest.values())
    loans_df = pd.read_parquet(OUT / "loan_intervals.parquet")
    purpose_rows = pd.DataFrame(read(OUT / "loan_purpose.json"))
    comparison = pd.read_csv(OUT / "release_comparison.csv")
    household = read(OUT / "household_purpose_bridge.json")
    tsf_data = read(OUT / "tsf_composition.json")
    tsf_blocks = {x["month"]: x for x in tsf_data["blocks"]}
    residuals = read(OUT / "tsf_yoy_residuals.json")
    event_data = read(OUT / "event_returns.json")
    for block, residual in zip(tsf_data["blocks"], residuals):
        assert abs(block["total_yi"]-sum(x["value_yi"] for x in block["fields"].values())-block["other_and_rounding_yi"]) < 1e-7
        assert abs(residual["official_total_yoy_yi"]-residual["sum_listed_yoy_yi"]-residual["other_and_rounding_yoy_yi"]) < 1e-7
    for event in event_data:
        assert abs(event["net_pnl"]/event["paid_cny"]-event["net_return"]) < 1e-12
    assert len(purpose_rows) == 30 and len(loans_df) == 108 and len(event_data) == 12
    share = household["long_business_share_of_same_source_long_increase"]
    feb = tsf_blocks["2023-02"]
    feb_three = sum(feb["fields"][k]["yoy_change"]["value_yi"] for k in ["rmb_real", "government_bonds", "undiscounted_bills"])
    table = ["| 报告期末 | 企业中长期 | 工业中长期 | 基础设施中长期 | 开发贷款 | 个人住房贷款 | 住户经营贷款 | 非住房消费贷款 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    pivot = purpose_rows.pivot(index="period", columns="label", values="reported_stock_yoy_pct")
    for period, row in pivot.iterrows():
        selected = [row[k] for k in ["企事业中长期", "工业中长期", "基础设施中长期", "房地产开发贷款", "个人住房贷款", "住户经营贷款", "住户非住房消费贷款"]]
        table.append("| "+period+" | "+" | ".join(f"{v:.1f}%" for v in selected)+" |")
    htable = ["| 住户贷款用途 | 一季度净增加/亿元 | 同比多增或少增/亿元 |", "|---|---:|---:|"]
    for row in household["rows"]:
        htable.append(f"| {row['use']}·{row['term']} | {row['net_increase_yi']:,.0f} | {row['official_yoy_increase_yi']:+,.0f} |")
    etable = ["| 统计月 | 原报告公开日 | 社融月度净增/亿元 | 官方同比多增或少增/亿元 | 发布后5间隔净收益 | 发布后20间隔净收益 |", "|---|---|---:|---:|---:|---:|"]
    for row in comparison.itertuples():
        etable.append(f"| {row.month} | {row.loan_public_at[:10]} | {row.tsf_yi:,.0f} | {row.official_yoy_change_yi:+,.0f} | {row.net5:+.2%} | {row.net20:+.2%} |")
    def link(path, label):
        return f"[{label}](<{Path(path).as_posix()}>)"
    source_link = lambda source_id, label: f"[{label}]({manifest[source_id]['url']})"
    classification = "PROGRESS_INDEX_CREDIT_CHANNEL_AND_END_DEMAND_SEPARATION"
    discovery = "六次社融报告、三个季度贷款用途及4月央行解释完成比较。2023Q1同源住户中长期净增中73.88%来自经营用途，消费中长期同比少增4261亿元；开发贷款增速回升同时个人住房贷款增速降至0.3%。2023年1月实体人民币贷款同比多增7308亿元而社融总量少增1959亿元。政策支持、渠道替代、最终需求与指数剩余收益不能合并成一个利好标签。新发放与滚续金额仍未识别，无新账户。"
    next_question = "核对2022Q4至2023Q1指数交易需求与风险承接的既有研究覆盖，优先已保存的ETF份额、融资交易及资金使用约束。若存在实质未覆盖的信息，再按完整固定区间研究；不把信用多增、长期贷款或本轮收益正负拼成新买入规则。"
    findings = {"classification": classification, "household_long_business_share": share, "february_three_channel_yoy_contribution_yi": feb_three, "february_official_total_yoy_yi": feb["total_yoy_change"]["value_yi"], "q1_tsf_rmb_real_yoy_yi": tsf_blocks["2023-03"]["fields"]["rmb_real"]["yoy_change"]["value_yi"], "q1_tsf_corporate_bonds_yoy_yi": tsf_blocks["2023-03"]["fields"]["corporate_bonds"]["yoy_change"]["value_yi"], "what_is_identified": ["社融同页同比差异的渠道构成", "贷款原文的期限和用途差异", "三个季度余额增速的分化", "六次固定公开日后可成交收益"], "unidentified": ["新发放与还本总额各自贡献", "滚续、置换和真正新增支出的逐笔规模", "这些融资数据公告前的精确一致预期", "各渠道变化对指数回报的独立因果贡献", "银行信用扩张到股票买入资金的数量映射"], "new_accounts": 0, "new_models": 0, "goal_achieved": False}
    save(OUT / "mechanism_findings.json", findings)
    report_path = OUT / "历史发现_融资期限用途与指数传导.md"
    figure = OUT / "融资构成_期限与实际用途.png"
    report = f"""**指数历史发现：融资期限、用途与最终需求为何不能混为一谈**

本轮新增的判断是：2022年末至2023年初，融资改善有明显的部门和用途差异。企业与开发商融资支持加强，居民经营贷款也在增长，但个人住房贷款增速继续回落；与此同时，非住房消费贷款出现恢复。把这些事实合并成一个“宽信用利好指数”的标签，会丢失最关键的传导差别。

范围固定为2022年10月至2023年3月六次统计月，另看2022Q3、2022Q4、2023Q1三个季度末用途数据。已有104个月贷款期限及公告后价格研究被复用；新增的是这段历史的六份社融渠道原文、三份用途原报告官方转载和一次央行说明。研究仍以指数整体为单位，未新增个股案例、模型或交易账户。

**先看用途，再解释贷款期限。**

下表均为各次原报告披露的贷款余额同比增速；不同用途可能交叉，不能相加。企业、工业、基建和住户经营/非住房消费为本外币口径，房地产开发和个人住房为人民币口径。表格比较各自变化，不跨币种拼接总额。

{chr(10).join(table)}

来源：{source_link('purpose_2022q3','2022Q3原报告官方转载')}、{source_link('purpose_2022q4','2022Q4原报告官方转载')}、{source_link('purpose_2023q1','2023Q1原报告官方转载')}。原季度值不等于当季新增额，前三季度累计和全年累计也没有被当作单季流量。

开发贷款的增速从2.2%升至3.7%、再至5.9%，个人住房贷款则从4.1%降至1.2%、再至0.3%。这支持“开发端融资改善和居民购房融资恢复速度不同”的描述。2022Q3开发贷款已为正增长，也反对把随后全部改善都归到11月新政策。非住房消费贷款增速从4.1%回到11.0%，又反对将整个居民需求概括为持续恶化。

**住户中长期贷款并非住房贷款的同义词。**

{chr(10).join(htable)}

同一发布会分项中，中长期经营与消费净增合计9442亿元，经营用途6976亿元，占{share:.2%}。消费中长期净增仍为正，但同比少增4261亿元；短期消费同比多增3690亿元。央行将这种分化分别联系到小微经营支持、消费回升和购房需求尚未完全恢复。实录该段未单独注明币种，分项仅在同源范围内计算，不与人民币月报混合；消费中长期也不全是房贷。{source_link('briefing_2023q1','央行2023年4月20日发布会')}。

这为指数分析提供的是用途差异：同一个“住户长期贷款增加”可能主要反映经营周转或经营投入，而不能直接推定住宅销售及整个消费链已经同步恢复。具体经营贷款有多少用于新增投资、多少置换旧负债，本轮没有逐笔用途与还款数据，保持未知。

**社融总量为何会与银行贷款方向不同。**

2023年1月，对实体经济的人民币贷款同比多增7308亿元，而社融合计同比少增1959亿元。企业债券少4352亿元、政府债券少1886亿元、未贴现承兑汇票少增1770亿元等渠道构成了抵消；其余分项与舍入残差保留，没有将银行贷款和社融人民币贷款相加。{source_link('tsf_202301','1月社融原报告')}。

2月社融同比多增1.95万亿元，其中实体人民币贷款、政府债券、未贴现承兑汇票分别贡献9241、5416、4158亿元，三项合计{feb_three:,.0f}亿元。未贴现票据当月实际减少70亿元，因此“同比少减”是改善来源之一，不等于该渠道已经净流入。{source_link('tsf_202302','2月社融原报告')}。

到一季度，社融同比多增2.47万亿元，实体人民币贷款多增2.36万亿元，企业债券反而少4718亿元。信用扩张以银行渠道为主与其他渠道偏弱可以同时出现；这些总量还不能证明企业逐笔以贷款替代了债券。{source_link('tsf_202303','一季度社融原报告')}。

对11月政策落地也要保留反证：11月企业债券净融资为596亿元、同比少3410亿元。个别增信工具开始实施，并不意味着整个企业债渠道已经同比改善。政策金额、获授信金额、债券发行总额、净融资和新增最终支出属于不同环节。{source_link('tsf_202211','11月社融原报告')}。

贷款净增和社融增量已经比授信意向更接近实际融资变化，但仍无法仅凭这几张总量表拆出新发放、还本、旧债滚续及会计调整的各自规模。不能把所有贷款净增都称为新投资，也不能反过来断言全部是借新还旧。

**信息出现之后，指数剩余收益并不与融资同比方向一一对应。**

{chr(10).join(etable)}

全部六次公开日都保留。六份社融原报告分别在同日贷款报告之前一分钟公布，本轮用该日结束后的下一实际交易日开盘观察，不采用收盘前已知的假设。窗口为原5/20个开盘间隔，使用510300、10万元独立名义预算、每边佣金0.04%且最低5元、每边滑点0.10%、100份整手、T+1和持有期间分红。收益以实际买入支付额为分母，不能相加当作20万元完整账户。

例如，2022年10月社融同比少增，11月11日开盘之后20间隔净收益却为+4.39%；但该开盘之前还有国内政策方向与美国CPI信息。2023年2月社融明显多增，3月13日起20间隔净收益+3.32%；3月同样多增，4月12日起却为-4.18%。这些反例说明共同信息和定价进程不能省略，没有证明反向信贷策略有效。窗口有重叠，也不把六行视作独立经济试验。

10月原文见{source_link('tsf_202210','10月社融报告')}；12月见{source_link('tsf_202212','全年与12月报告')}。12月及3月的原文仅对合计直接给出单月值，分项分别为全年和一季度，本轮没有用累计分项冒充单月，也未将后来修订的月表回填。

**对指数机制的实际约束。**

这一阶段更准确的传导链是：融资政策与银行供给条件改变，部分借款主体获得支持，融资用途与终端需求恢复存在差异，再经过行业盈利、利率和风险承担进入指数定价。银行、工业、地产链及消费行业的影响可能相互抵消，不能从信贷的总量或期限单独决定沪深300方向。宏观融资增长也不是已经观察到的股票买入资金。

新增数据具体排除了两个过强解释：“长期贷款增加主要就是购房恢复”，以及“银行贷款多增代表所有融资渠道同步改善”。它没有排除真实需求恢复：工业、基建及非住房消费的改善都保留。这里有机制支持的历史解释，未识别政策对价格的独立因果效果，也没有取得这六次融资数据公布前的精确一致预期。

**公开时间和统计范围。**

三份用途材料本次取得的是官方转载，页面日期分别为2022年10月31日、2023年2月4日和5月4日，只作为保守可见上界，不认定最早发布日期。4月20日的住户用途拆分不能放回1月或2月入场；5月4日取得的一季度用途表也不能作为4月12日的已知输入。它们只解释后来确认的融资结构，未用于改变历史交易规则。

2023年1月金融统计扩入三类银行业非存款机构，原文社融同比采用官方可比口径；本轮没有跨2022与2023旧原值硬算分项同比。贷款区间继续复用V23原公告代数，12月及3月部分单月期限分项由累计端点求差，显示舍入界完整保留。历史页面为现在取得的记录，不声称拥有当年不可改写的首次快照。

本轮完成108行贷款区间背景、30行用途观察、6组社融分项、12个成本后事件窗口和10份新增来源。保存金额与同源分项加总已核对，结果均为历史发现；没有新的完整账户，净夏普与年化收益为NOT_COMPUTED，夏普1.2及年化10%的原目标仍未实现。此前单独信用总量、PMI等拒绝封存表达均未修改。

下一项历史问题：{next_question}

{link(figure, '查看融资用途与渠道图')}。数据入口：{link(OUT / 'release_comparison.csv', '六次公开日及收益')}；{link(OUT / 'loan_purpose.csv', '全部季度用途')}；{link(OUT / 'household_purpose_bridge.json', '住户期限与用途分解')}；{link(OUT / 'tsf_composition.json', '社融原文构成')}；{link(Path(__file__), '完整研究脚本')}。

完成时间：{now()}。
"""
    report_path.write_text(report, encoding="utf-8")
    checks = {"created_at": now(), "loan_rows": len(loans_df), "purpose_rows": len(purpose_rows), "events": len(event_data), "raw_sources": len(manifest), "source_tsf_precedes_same_day_loan_cutoff": True, "net_event_returns_recomputed": True, "tsf_residuals_preserved": True, "household_parts_identity_checked": True, "new_accounts": 0, "independent_validation": False}
    save(OUT / "calculation_checks.json", checks)
    result = {"study_id": protocol["study_id"], "status": "COMPLETED_HISTORICAL_MECHANISM_STUDY_NO_TRADING_CANDIDATE", "classification": classification, "completed_at": now(), "discovery": discovery, "next_historical_question": next_question, "report": rel(report_path), "figure": rel(figure), "report_status": "WRITTEN_PENDING_REVIEW", "figure_visually_reviewed": False, "months": 6, "purpose_observations": 3, "new_accounts": 0, "new_models": 0, "net_sharpe": None, "net_cagr": None, "metric_status": "NOT_COMPUTED", "account_status": "NOT_RUN_NO_ADMITTED_TRADING_CANDIDATE", "goal_achieved": False}
    save(OUT / "result.json", result)
    files = [report_path, figure, OUT / "result.json", OUT / "loan_intervals.parquet", OUT / "loan_purpose.json", OUT / "household_purpose_bridge.json", OUT / "tsf_composition.json", OUT / "release_comparison.csv", OUT / "mechanism_findings.json", OUT / "calculation_checks.json"]
    save(OUT / "research_receipt.json", {"created_at": now(), "script": rel(Path(__file__)), "script_sha256": digest(Path(__file__)), "protocol_sha256": digest(OUT / "protocol.json"), "output_hashes": {rel(p): digest(p) for p in files}, "goal_achieved": False})
    print(json.dumps({"报告": rel(report_path), "新完整账户": 0, "目标完成": False}, ensure_ascii=False))


def finalize():
    result = read(OUT / "result.json")
    result.update(report_status="REVIEWED", figure_visually_reviewed=True, updated_at=now())
    save(OUT / "result.json", result)
    receipt = read(OUT / "research_receipt.json")
    receipt["output_hashes"][rel(OUT / "result.json")] = digest(OUT / "result.json")
    receipt.update(updated_at=now(), script_sha256=digest(Path(__file__)))
    save(OUT / "research_receipt.json", receipt)
    path = ROOT / "config/510300_historical_cause_discovery_v1.json"
    config = read(path)
    config.update(latest_completed_study=rel(OUT / "result.json"), latest_report=result["report"], current_study=rel(OUT / "protocol.json"), latest_result_summary=result["discovery"], next_historical_question=result["next_historical_question"], updated_at=now(), goal_achieved=False)
    save(path, config)
    path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    config = read(path)
    config.update(latest_historical_index_credit_realization=rel(OUT / "result.json"), latest_historical_report=result["report"], latest_historical_diagnostic_at=now(), current_driver_continuation_classification=result["classification"], current_driver_consecutive_blocked_goal_turns=0, next_research_question=result["next_historical_question"], local_goal_work_status="ACTIVE_HISTORICAL_ONLY", goal_achieved=False)
    save(path, config)
    print("已保存历史研究结果，目标保持未达成，后续主线继续为指数整体。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="指数融资改善的渠道与用途历史比较")
    parser.add_argument("action", choices=["prepare", "collect", "loans", "tsf", "events", "purpose", "compare", "plot", "publish", "finalize"])
    args = parser.parse_args()
    globals()[args.action]()
