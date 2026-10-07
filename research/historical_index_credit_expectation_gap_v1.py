"""用同期公开调查重建固定六个月的融资预期，不从指数涨跌反造预期。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup

from historical_index_liquidity_transmission_v1 import clean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_index_credit_expectation_gap_v1"
PRIOR = ROOT / "reports/research/510300_historical_index_credit_flow_composition_v1"
STUDY = "510300_HISTORICAL_INDEX_CREDIT_EXPECTATION_GAP_V1"
SOURCES = [
    {"id": "reuters_oct_pre", "month": "2018-10", "date": "2018-11-08", "tier": "PRE_RELEASE_PUBLICATION", "url": "https://www.investing.com/news/economic-indicators/chinas-october-new-loans-seen-lower-but-credit-easing-on-track-reuters-poll-1679378", "required": ["862 billion yuan", "32 analysts"]},
    {"id": "reuters_nov_pre", "month": "2018-11", "date": "2018-12-07", "tier": "PRE_RELEASE_PUBLICATION", "url": "https://www.investing.com/news/stock-market-news/china-november-new-loans-seen-rebounding-after-worrying-october-reuters-poll-1714365", "required": ["1.12 trillion yuan", "1.33 trillion yuan", "22 economists"]},
    {"id": "reuters_dec_recall", "month": "2018-12", "date": "2019-01-15", "tier": "RELEASE_REPORT_RECALL", "url": "https://www.livemint.com/Politics/EGNZx6xn80mXecKlX9Dh2H/China-signals-more-stimulus-as-economic-slowdown-deepens.html", "required": ["800 billion yuan", "15 Jan 2019", "Reuters"]},
    {"id": "reuters_jan_pre", "month": "2019-01", "date": "2019-02-08", "tier": "PRE_RELEASE_PUBLICATION", "url": "https://www.investing.com/news/economy-news/china-jan-new-bank-loans-seen-surging-as-pboc-keeps-liquidity-taps-open-reuters-poll-1774014", "required": ["2.8 trillion yuan", "3.25 trillion yuan", "19 economists"]},
    {"id": "reuters_feb_recall", "month": "2019-02", "date": "2019-03-11", "tier": "RELEASE_REPORT_RECALL", "url": "https://www.dailysabah.com/finance/2019/03/11/china-new-bank-loans-fall-sharply-but-policy-support-still-on-track", "required": ["975 billion yuan", "8.4 percent", "Reuters"]},
    {"id": "reuters_mar_recall", "month": "2019-03", "date": "2019-04-12", "tier": "RELEASE_REPORT_RECALL", "url": "https://finance.yahoo.com/news/china-march-loans-rebound-sharply-091148960.html", "required": ["1.744", "1.2 trillion yuan", "Reuters"]},
    {"id": "yicai_feb_pre", "month": "2019-02", "date": "2019-03-05", "tier": "PRE_RELEASE_PUBLICATION", "url": "https://www.yicai.com/news/100131728.html", "required": ["2019-03-05", "社会融资", "预测均值"]},
]


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("已有范围，不覆盖。")
    previous = json.loads((PRIOR / "protocol.json").read_text(encoding="utf-8"))
    save("protocol.json", {
        "study_id": STUDY, "recorded_at": now(), "previous_goal_turn_classification": "PROGRESS",
        "mode": "HISTORICAL_DISCOVERY_ONLY", "research_unit": "指数整体的融资预期与价格吸收",
        "events": previous["events"], "series": ["tsf_flow", "rmb_loan_flow", "m2_yoy"],
        "question": "历史同比改善是否也超出当时公开调查；调查的上游理由与分歧是什么，已知预期能否解释剩余指数收益？",
        "main_provider": "路透自身调查及署名原报道的同期转载；未找到发布前原文的月份单列证据等级，不能混称完整事前调查序列。",
        "cross_source": "第一财经自身调查作为独立预期口径；各项均值、中位数或未说明的聚合方式照原文分别保留，不平均或挑选有利方向。",
        "evidence_tiers": {"PRE_RELEASE_PUBLICATION": "原页面日期早于数据公开日", "RELEASE_REPORT_RECALL": "数据发布当日或稍后报道回述预测，未确证发布前公开页面，不能用于发布前决策"},
        "missing": "缺失预期、调查范围、机构样本或标准差保持缺失；不以同比、前值或后来涨跌代替。",
        "formula": "发布值减对应调查预期；单位为亿元或百分点，仅原始差值，不拟合Z分数、阈值或组合评分。",
        "historical_prices": "六个月的公布前20日与公布后5/20日结果已经见过，复用上一轮保存结果；没有新选择窗口、提前退出、删掉后续冲击。",
        "research_order": "预期来源检索已开始，部分预期数值已见；现在固定提取与比较规则，不声称盲法或独立验证。",
        "sources_planned": SOURCES, "returns_recalculated": False, "new_parameters_fitted": 0,
        "new_full_accounts": 0, "net_sharpe": None, "goal_achieved": False, "orders_authorized": False,
    })
    print("已登记固定六个月的预期重建，预先公开与事后回述分别列示，复用原收益。", flush=True)


def fetch_one(source):
    item = dict(source)
    try:
        response = requests.get(source["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=(10, 25))
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        meta_dates = {m.get("property", m.get("name", "")): m.get("content") for m in soup.select("meta[content]") if any(k in str(m).lower() for k in ["published", "modified", "pubdate", "timestamp"])}
        article = soup.select_one('[data-test="article-content"]') or soup.select_one("article") or soup
        for tag in article(["script", "style"]):
            tag.decompose()
        body = unicodedata.normalize("NFKC", article.get_text(" ", strip=True))
        page_body = unicodedata.normalize("NFKC", soup.get_text(" ", strip=True))
        compact = re.sub(r"\s+", "", page_body).lower()
        missing = [n for n in source["required"] if re.sub(r"\s+", "", n).lower() not in compact]
        path = OUT / "sources" / (source["id"] + ".html")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        path.with_suffix(".txt").write_text(body, encoding="utf-8")
        path.with_suffix(".page.txt").write_text(page_body, encoding="utf-8")
        item.update(status="RETRIEVED" if not missing else "BODY_CHECK_REQUIRED", missing=missing, path=str(path.relative_to(ROOT)),
                    sha256=hashlib.sha256(response.content).hexdigest(), meta_dates=meta_dates, retrieved_at=now())
    except Exception as exc:
        item.update(status="MISSING_SOURCE", error=str(exc), retrieved_at=now())
    return item


def fetch():
    if (OUT / "source_manifest.json").exists():
        raise RuntimeError("已有下载记录，不重复。")
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch_one, SOURCES))
    save("source_manifest.json", records)
    print(json.dumps([{k: r.get(k) for k in ["id", "status", "missing", "error", "meta_dates"]} for r in records], ensure_ascii=False, indent=2), flush=True)


def resolve_sources():
    if (OUT / "source_resolution.json").exists():
        raise RuntimeError("已有来源补充，不覆盖。")
    records = json.loads((OUT / "source_manifest.json").read_text(encoding="utf-8"))
    for row in records:
        web_path = OUT / "sources" / (row["id"] + ".web.txt")
        if row["status"] == "MISSING_SOURCE" and web_path.exists():
            text = web_path.read_text(encoding="utf-8")
            compact = re.sub(r"\s+", "", text).lower()
            assert all(re.sub(r"\s+", "", n).lower() in compact for n in row["required"])
            text = re.sub(r"L\d+:\s*", "", text)
            first = text.find("BEIJING (Reuters)")
            if first < 0:
                raise RuntimeError("网页提取缺少报道正文。")
            end = text.find("Latest comments", first)
            if end < 0:
                end = len(text)
            article_path = web_path.with_name(row["id"] + ".txt")
            article_path.write_text(text[first:end], encoding="utf-8")
            row.update(status="RETRIEVED_WEB_EXTRACT", original_request_error=row.get("error"),
                       path=str(web_path.relative_to(ROOT)), sha256=hashlib.sha256(web_path.read_bytes()).hexdigest(),
                       transport="网络工具的可读正文；不是原始HTML", resolved_at=now())
    save("source_resolution.json", {"recorded_at": now(), "sources": records,
                                    "notes": ["三个Investing页面本地请求返回403，改用网络工具已经取得的署名路透正文；原失败记录保留。", "数据发布后回述调查与发布前公开调查分开，不为补齐序列而互换。"]})
    print(json.dumps([{k: r.get(k) for k in ["id", "status", "tier"]} for r in records], ensure_ascii=False, indent=2), flush=True)


def build():
    if (OUT / "expectations.json").exists():
        raise RuntimeError("已有提取结果，不覆盖。")
    sources = {r["id"]: r for r in json.loads((OUT / "source_resolution.json").read_text(encoding="utf-8"))["sources"]}
    # 手工核对后的固定锚点只匹配正文中的预测，避免抽到上月值、实际值或当前网页行情。
    definitions = [
        ("reuters_oct_pre", "rmb_loan_flow", 8620, "862 billion yuan", 32, "文中调查代表值，未明确均值或中位数", None, None),
        ("reuters_oct_pre", "m2_yoy", 8.4, "Broad M2 money supply was seen rising 8.4 percent", 32, "文中调查代表值，未明确均值或中位数", None, None),
        ("reuters_nov_pre", "rmb_loan_flow", 11200, "1.12 trillion yuan", 22, "文中调查代表值，未明确均值或中位数", None, None),
        ("reuters_nov_pre", "tsf_flow", 13300, "1.33 trillion yuan", 22, "文中调查代表值，未明确均值或中位数", None, None),
        ("reuters_nov_pre", "m2_yoy", 8.0, "8.0 percent", 22, "文中调查代表值，未明确均值或中位数", None, None),
        ("reuters_dec_recall", "rmb_loan_flow", 8000, "market expectations of 800 billion yuan", None, "当日报道回述市场预期，聚合方法未明", None, None),
        ("reuters_jan_pre", "rmb_loan_flow", 28000, "2.8 trillion yuan", 19, "中位数", None, None),
        ("reuters_jan_pre", "tsf_flow", 32500, "3.25 trillion yuan", 19, "文中调查代表值，未明确该项聚合方法", None, None),
        ("reuters_jan_pre", "m2_yoy", 8.2, "annual M2 growth in January was seen at 8.2 percent", 19, "文中调查代表值，未明确该项聚合方法", None, None),
        ("reuters_feb_recall", "rmb_loan_flow", 9750, "975 billion yuan", None, "发布后报道回述路透调查，聚合方法未明", None, None),
        ("reuters_feb_recall", "m2_yoy", 8.4, "Analysts had expected a 8.4 percent rise in M2", None, "发布后报道回述路透调查，聚合方法未明", None, None),
        ("reuters_mar_recall", "rmb_loan_flow", 12000, "forecast 1.2 trln yuan", None, "当日报道回述路透调查，聚合方法未明", None, None),
        ("reuters_mar_recall", "tsf_flow", 17440, "forecast 1.744 trln yuan", None, "当日报道回述路透调查，聚合方法未明", None, None),
        ("reuters_mar_recall", "m2_yoy", 8.2, "forecast 8.2 pct", None, "当日报道回述路透调查，聚合方法未明", None, None),
        ("yicai_feb_pre", "rmb_loan_flow", 9524.76, "回落至9524.76亿元", None, "均值", None, None),
        ("yicai_feb_pre", "tsf_flow", 13300, "预测均值为1.33万亿元", None, "均值", 10000, 20000),
        ("yicai_feb_pre", "m2_yoy", 8.38, "下降至8.38%", None, "均值", 8.0, 8.5),
    ]
    rows = []
    for key, series, value, anchor, count, method, low, high in definitions:
        source = sources[key]
        text = (OUT / "sources" / (key + ".txt")).read_text(encoding="utf-8")
        assert re.sub(r"\s+", "", anchor).lower() in re.sub(r"\s+", "", text).lower(), (key, anchor)
        assert source["status"].startswith("RETRIEVED")
        rows.append({"source_id": key, "stat_month": source["month"], "series": series,
                     "provider": "第一财经自身调查" if key.startswith("yicai") else "路透自身调查或署名报道",
                     "expected": value, "unit": "%" if series == "m2_yoy" else "亿元", "aggregation": method,
                     "surprise_unit": "百分点" if series == "m2_yoy" else "亿元",
                     "article_survey_n": count, "series_specific_n": None, "forecast_min": low, "forecast_max": high,
                     "published_date": source["date"], "evidence_tier": source["tier"], "source_url": source["url"], "anchor": anchor})
    save("expectations.json", {"recorded_at": now(), "observations": rows,
                               "limits": ["篇章受访人数不等于每个指标的有效样本数；未推断标准差或置信区间。", "调查对象是经济学家，不是按资金规模加权的边际股票投资者。", "调查至公布之间可能有预期修正，本轮没有完整覆盖。"]})
    print("保存17项有正文锚点的预期；四个路透预期字段缺失，不填补。", flush=True)


def analyze():
    if (OUT / "comparison.json").exists():
        raise RuntimeError("已有比较结果，不覆盖。")
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    forecasts = json.loads((OUT / "expectations.json").read_text(encoding="utf-8"))["observations"]
    composition = json.loads((PRIOR / "composition.json").read_text(encoding="utf-8"))
    returns = json.loads((PRIOR / "events_and_returns.json").read_text(encoding="utf-8"))
    originals = json.loads((ROOT / "reports/research/510300_macro_transmission_context_v4/inputs/loan_originals.json").read_text(encoding="utf-8"))
    actuals = {}
    for event in protocol["events"]:
        month = event["id"]
        headline = next(r for r in composition["monthly_tsf_headlines"] if r["stat_month"] == month)
        loan = next(r for r in composition["loan_rows"] if r["stat_month"] == month and r["field"] == "rmb_total" and r["window"] == "1")
        original = next(r for r in originals if r["stat_month"] == month)
        text = unicodedata.normalize("NFKC", BeautifulSoup((ROOT / original["raw_path"]).read_bytes(), "html.parser").get_text(" ", strip=True))
        compact = re.sub(r"\s+", "", text)
        m2 = re.search(r"广义货币\(M2\)余额[\d.]+万亿元,同比增长([\d.]+)%", compact)
        if not m2:
            raise ValueError("M2正文定位失败：" + month)
        actuals[month] = {
            "tsf_flow": {"value": headline["total_yi"], "method": "社融同期月度报告合计", "rounding_half": headline.get("display_rounding_half_yi")},
            "rmb_loan_flow": {"value": loan["value_yi"], "method": loan["method"], "rounding_half": loan["rounding_half_yi"]},
            "m2_yoy": {"value": float(m2[1]), "method": "央行原文月末同比", "rounding_half": None},
        }
    rows = []
    for event in protocol["events"]:
        for series in protocol["series"]:
            matches = [r for r in forecasts if r["stat_month"] == event["id"] and r["series"] == series]
            if not any(r["source_id"].startswith("reuters") for r in matches):
                matches.insert(0, {"source_id": None, "provider": "路透自身调查或署名报道", "expected": None,
                                   "evidence_tier": "MISSING_EXPECTATION", "published_date": None})
            for forecast in matches:
                actual = actuals[event["id"]][series]
                five = next(r for r in returns if r["stat_month"] == event["id"] and r["horizon"] == 5)
                twenty = next(r for r in returns if r["stat_month"] == event["id"] and r["horizon"] == 20)
                expected = forecast["expected"]
                diff = None if expected is None else actual["value"] - expected
                if forecast["evidence_tier"] == "PRE_RELEASE_PUBLICATION":
                    assert pd.Timestamp(forecast["published_date"]) < pd.Timestamp(event["date"])
                rows.append({**forecast, "stat_month": event["id"], "series": series, "release_date": event["date"],
                             "unit": "%" if series == "m2_yoy" else "亿元",
                             "surprise_unit": "百分点" if series == "m2_yoy" else "亿元",
                             "actual": actual["value"], "actual_method": actual["method"], "actual_rounding_half": actual["rounding_half"],
                             "surprise_raw": diff, "surprise_sign": None if diff is None else (1 if diff > 1e-9 else -1 if diff < -1e-9 else 0),
                             "entry_date": five["entry_date"], "net_return_5d": five["net_return"], "net_return_20d": twenty["net_return"],
                             "pre_release_20d_return": twenty["pre_release_20d_total_return"],
                             "returns_source": str((PRIOR / "events_and_returns.json").relative_to(ROOT))})
    assert len(rows) == 21 and len({r["stat_month"] for r in rows}) == 6
    save("comparison.json", {"computed_at": now(), "rows": rows, "returns_recalculated": False})
    save("result.json", {"study_id": STUDY, "computed_at": now(), "classification": "PROGRESS_INDEX_CREDIT_EXPECTATION_RECONSTRUCTION",
                         "event_count": 6, "expectation_observations": len(forecasts),
                         "pre_release_published_observations": sum(r["evidence_tier"] == "PRE_RELEASE_PUBLICATION" for r in forecasts),
                         "main_provider_missing_fields": sum(r["evidence_tier"] == "MISSING_EXPECTATION" for r in rows),
                         "new_parameters_fitted": 0, "new_full_accounts": 0, "net_sharpe": None, "goal_achieved": False, "orders_authorized": False})
    print(pd.DataFrame(rows)[["stat_month", "series", "provider", "expected", "actual", "surprise_raw", "evidence_tier", "net_return_20d"]].to_string(index=False), flush=True)


def enrich_context():
    if (OUT / "context_supplement.json").exists():
        raise RuntimeError("已有同期解释补充，不覆盖。")
    source = {
        "id": "yicai_mar_policy_reaction", "date": "2019-04-12", "published_at": "2019-04-12T21:00:56+08:00",
        "url": "https://www.yicai.com/news/100161158.html",
        "required": ["2019-04-12", "近期降准概率下降", "1.69万亿元"],
    }
    record = fetch_one(source)
    save("supplement_sources.json", [record])
    if record["status"] != "RETRIEVED":
        raise RuntimeError("同期解释来源未通过正文定位，保留记录。")
    # 只纠正单位和证据描述，保留已保存的实际值、预期差及全部收益。
    corrections = []
    for name, field in [("expectations.json", "observations"), ("comparison.json", "rows")]:
        data = json.loads((OUT / name).read_text(encoding="utf-8"))
        for row in data[field]:
            updates = {"unit": "%" if row["series"] == "m2_yoy" else "亿元",
                       "surprise_unit": "百分点" if row["series"] == "m2_yoy" else "亿元"}
            if row["series"] == "m2_yoy" and "actual_rounding_half" in row:
                updates["actual_rounding_half"] = None
            if row["source_id"] == "reuters_feb_recall" and row["series"] == "rmb_loan_flow":
                updates["aggregation"] = "发布后报道回述路透调查，聚合方法未明"
            changes = {k: {"before": row.get(k), "after": v} for k, v in updates.items() if row.get(k) != v}
            if changes:
                corrections.append({"file": name, "month": row["stat_month"], "series": row["series"],
                                    "source_id": row["source_id"], "changes": changes})
            row.update(updates)
        save(name, data)
    save("metadata_corrections.json", {
        "recorded_at": now(), "changes": corrections, "forecast_actual_surprise_or_returns_changed": False,
        "notes": ["M2水平单位为%，预期差单位为百分点；未确定原文完整显示精度，不指定M2舍入界。",
                  "冻结范围文件将路透统称中位数的描述过宽，逐项提取为准：只有明确说明时才登记中位数，未说明时保持未知。"],
    })
    save("context_supplement.json", {
        "recorded_at": now(), "selected_after_return_and_surprise_comparison": True,
        "purpose": "补充整体增长预期与近期政策节奏的不同传导，不作新信号或月份分类。",
        "source": source,
        "historical_interpretations": [
            "4月12日同期采访中，融资改善被联系到经济企稳，同时部分受访者判断近期全面降准的必要性下降。",
            "近期总量放松必要性下降，与之后可能定向或结构性放松并不矛盾；必须保留政策工具和期限差别。",
            "同日路透报道既记录未来数季继续降准的调查，也记录分析师降低4月降准可能性的判断；不能拼成全市场一致反转。",
        ],
        "loan_actual_supplement": {"stat_month": "2019-03", "reported_monthly_actual_yi": 16900,
                                  "old_cumulative_difference_yi": 17000, "old_display_rounding_bound_yi": 100,
                                  "expected_yi": 12000, "reported_monthly_surprise_yi": 4900,
                                  "note": "同期单月报道1.69万亿与累计端点差约1.70万亿分列；没有重写旧账本，超预期方向不变。"},
        "causal_contribution_identified": False, "new_rules": 0, "returns_recalculated": False,
    })
    print("已补存4月12日政策预期采访，并纠正M2单位说明；实际值、预期差与收益没有重算。", flush=True)


def draw(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Patch

    font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False,
                         "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    months = sorted({r["stat_month"] for r in rows})
    base = [next(r for r in rows if r["stat_month"] == m and r["series"] == "tsf_flow" and
                 r["provider"].startswith("路透")) for m in months]
    fig, axes = plt.subplots(2, 1, figsize=(13.4, 9.5))
    fig.patch.set_facecolor("#faf9f5")
    for ax in axes:
        ax.set_facecolor("#faf9f5")
        ax.axhline(0, color="#58646b", lw=.8)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    for i, row in enumerate(base):
        if row["expected"] is None:
            axes[0].scatter(i - .12, 0, color="#9ca3a7", marker="x", s=42)
            axes[0].annotate("路透预期缺失", (i, 0), xytext=(0, 9), textcoords="offset points",
                             ha="center", fontsize=9, color="#697279")
        else:
            color = "#276b69" if row["evidence_tier"] == "PRE_RELEASE_PUBLICATION" else "#bd8642"
            bar = axes[0].bar(i, row["surprise_raw"], width=.5, color=color)
            axes[0].bar_label(bar, labels=[f"{row['surprise_raw']:+,.0f}"], padding=4, fontsize=11)
    extra = next(r for r in rows if r["source_id"] == "yicai_feb_pre" and r["series"] == "tsf_flow")
    bar = axes[0].bar(months.index("2019-02") + .12, extra["surprise_raw"], width=.34,
                      facecolor="#ebe4f0", edgecolor="#76628b", hatch="///")
    axes[0].bar_label(bar, labels=[f"{extra['surprise_raw']:+,.0f}"], padding=4, fontsize=11)
    axes[0].legend(handles=[Patch(color="#276b69", label="路透：公布前调查"),
                           Patch(color="#bd8642", label="路透：公布后报道回述"),
                           Patch(facecolor="#ebe4f0", edgecolor="#76628b", hatch="///", label="一财：公布前独立调查")],
                   loc="upper left", frameon=False, ncol=3, fontsize=10)
    axes[0].set_ylim(-9100, 20500)
    axes[0].set_xticks(range(6), months)
    axes[0].set_ylabel("实际社融减调查预期（亿元）")
    axes[0].set_title("A  先区别同比变化与调查预期差，缺失项保持缺失", loc="left", fontsize=14, pad=10)
    x = np.arange(6)
    for shift, key, color, label in [(-.19, "net_return_5d", "#c5a071", "公布后5日"),
                                    (.19, "net_return_20d", "#3f7272", "公布后20日")]:
        bars = axes[1].bar(x + shift, [r[key] * 100 for r in base], width=.35, color=color, label=label)
        axes[1].bar_label(bars, labels=[f"{r[key] * 100:+.2f}" for r in base], padding=3, fontsize=10)
    axes[1].set_ylim(-11, 15)
    axes[1].set_xticks(x, [f"{m}\n发布 {r['release_date'][5:]}" for m, r in zip(months, base)])
    axes[1].set_ylabel("510300事件净收益（%）")
    axes[1].legend(loc="upper left", frameon=False, ncol=2)
    axes[1].set_title("B  固定窗口中的收益不同步；后续新闻也进入窗口", loc="left", fontsize=14, pad=10)
    fig.suptitle("510300｜信用预期差与指数历史表现", x=.085, y=.976, ha="left", fontsize=20, fontweight="bold")
    fig.text(.085, .934, "统计月：2018年10月至2019年3月。两家调查分别列示；灰色叉号代表缺失，不代表预期差为零。", fontsize=11, color="#556166")
    fig.text(.085, .026, "收益复用既有六次结果：发布后下一交易日开盘进入；10万元事件预算，已计原设交易成本。\n20日窗口存在重叠，不是完整账户夏普；4月发布后的窗口包含5月贸易冲击。", fontsize=10, color="#556166")
    fig.subplots_adjust(left=.1, right=.975, top=.865, bottom=.12, hspace=.38)
    fig.savefig(OUT / "信用预期差与指数历史表现.png", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def report():
    rows = json.loads((OUT / "comparison.json").read_text(encoding="utf-8"))["rows"]
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    sources = {s["id"]: s for s in SOURCES}
    context = json.loads((OUT / "context_supplement.json").read_text(encoding="utf-8"))
    sources[context["source"]["id"]] = context["source"]
    link = lambda key, label: f"[{label}]({sources[key]['url']})"
    local = lambda path, label: f"[{label}](<{path.as_posix()}>)"
    by_month = {e["id"]: [r for r in rows if r["stat_month"] == e["id"] and r["provider"].startswith("路透")]
                for e in protocol["events"]}
    lines = ["# 510300历史发现：信用预期差与指数传导", "",
        "研究单位为沪深300整体，以510300 ETF作为可交易价格观察。固定观察2018年10月至2019年3月六次金融数据发布，补出当时公开调查，与已保存的事件收益对照。这里的发现用于解释历史，不提供当前或未来行情判断。", "",
        "**本轮结论**", "",
        "同比变化和预期差必须分开：2018年11月社融同比少增，却高于公布前路透调查；2019年2月社融低于一财调查中最低预测，随后指数仍上涨。因此，修正为‘看超预期’之后，单月融资方向依然不足以独立决定这段历史中的指数收益。", "",
        "进一步找到的同期机制是：信用改善可能同时增强增长判断、降低近期额外宽松的紧迫性。两条传导方向不同，且市场还面对外部利率、贸易消息与已发生的上涨。同期采访证明这些解释当时已经存在，尚未量出各自的价格贡献。", "",
        "**六个月的调查与发布值**", "",
        "贷款和社融单位为亿元；M2为同比百分数。每格均为‘调查预期 → 发布值’。主表只用路透自身调查或署名原报道，避免按结果更换预测来源。", "",
        "| 统计月／公开日 | 人民币贷款 | 社融 | M2同比 | 调查证据 |",
        "|---|---:|---:|---:|---|",
    ]
    for event in protocol["events"]:
        group = {r["series"]: r for r in by_month[event["id"]]}
        cells = []
        for key in ["rmb_loan_flow", "tsf_flow", "m2_yoy"]:
            r = group[key]
            fmt = (lambda x: f"{x:.1f}%") if key == "m2_yoy" else (lambda x: f"{x:,.0f}")
            expected = "缺失" if r["expected"] is None else fmt(r["expected"])
            star = "*" if event["id"] == "2019-03" and key == "rmb_loan_flow" else ""
            cells.append(f"{expected} → {fmt(r['actual'])}{star}")
        known = next(r for r in group.values() if r["source_id"])
        label = "公布前调查" if known["evidence_tier"] == "PRE_RELEASE_PUBLICATION" else "公布后报道回述"
        label += " " + known["published_date"]
        lines.append(f"| {event['id']}／{event['date']} | " + " | ".join(cells) + " | " + link(known["source_id"], label) + " |")
    lines += ["",
        "*3月贷款约17000亿元沿用旧账本的累计端点差，其显示舍入界为±100亿元；同期单月报道为16900亿元。对应超预期分别约5000和4900亿元，方向一致，原账本没有被悄悄改写。12月贷款10830亿元亦来自累计分段，详见comparison.json的计算方法字段。舍入界不是统计置信区间。", "",
        "主表18格中有4格预期缺失。2018年12月、2019年2月和3月的路透数字来自公布当日或稍后的回述，未找到此前公开页面，不能与已取得的公布前调查混称同一证据等级。调查对象是经济学家，不代表股票资金的加权预期；文中总受访人数也不是每项指标的有效样本数。路透只有明确说明的项目才记为中位数，其余聚合方式保持未知。", "",
        "**春节和年初集中放贷，能否解释2月全部落差**", "",
        "一财3月5日的独立调查已考虑春节、银行年初提前放贷及票据因素，但社融实际值7030亿元仍低于其最低预测10000亿元。季节性已有预期，不能单独解释全部落差。它与路透分别保留，不拼接成同一个一致预期序列。" + link("yicai_feb_pre", "第一财经原始调查") + "。", "",
        "| 2019年2月指标 | 一财调查均值 | 调查范围 | 发布值 | 实际减均值 |",
        "|---|---:|---:|---:|---:|",
        "| 人民币贷款（亿元） | 9,524.76 | 本轮未取得 | 8,858 | -666.76 |",
        "| 社融（亿元） | 13,300 | 10,000—20,000 | 7,030 | -6,270 |",
        "| M2同比 | 8.38% | 8.00%—8.50% | 8.00% | -0.38个百分点 |", "",
        "同份调查中的受访者仍讨论减税、信用传导及未来经济企稳；这些是当时的预期，不是后来实现的事实。该调查也没有显示普遍预期3月立即降准，因此不能用‘数据差就一定马上刺激’替代证据。M2落在调查区间下沿，社融则落在区间外，两者不能笼统合称同样程度的意外。", "",
        "**从总量好坏，推进到指数的传导机制**", "",
        "2018年末，路透公布前报道已提及影子融资收缩、银行风险偏好谨慎和地方融资约束。银行资金增加与愿意向实体承担风险之间仍有间隔；11月社融高于调查预期，也不自动等于整体信用约束已经解除。" + link("reuters_oct_pre", "10月调查背景") + "、" + link("reuters_nov_pre", "11月调查背景") + "。", "",
        "2019年1月，调查已把降准和银行年初放贷节奏纳入判断：预计社融3.25万亿元，实际4.64万亿元，超出的部分是1.39万亿元。不能把4.64万亿元全部叫作新增意外，也不能凭发布前已上涨10.47%就断言全部定价。" + link("reuters_jan_pre", "1月公布前路透调查") + "。", "",
        "2019年4月12日，一财同期采访记录了两种并存判断：融资改善支持经济企稳；额外总量宽松的近期紧迫性下降，政策可能更关注结构和传导。同日路透还记录了未来数季继续降准的调查及部分分析师调低4月降准可能性的看法。‘近期不急’与‘稍后仍可能宽松’涉及不同期限，不是可以简单投票的相反观点。" + link("yicai_mar_policy_reaction", "4月12日晚间采访") + "、" + link("reuters_mar_recall", "同日路透报道") + "。", "",
        "由此提出一个待检验的指数机制：增长改善可以上修整体现金流预期，也可能降低进一步宽松的预期；信用风险补偿和权益配置需求又有各自的变化。需要分别看这些渠道在当时是否有证据，不能看到后来下跌，就倒填为‘利好兑现’或‘政策转紧’。本轮未直接观测股票投资者的边际估值，也未识别纯粹外生信用冲击。", "",
        "**同一组事件的指数表现**", "",
        "公布前20日为含分红且未扣交易成本的涨跌。公布后从公开日结束后的下一交易日开盘进入，再过5或20个交易日开盘退出；下表收益沿用原有交易成本，未重新选择窗口。", "",
        "| 统计月 | 公布前20日 | 公布后5日净收益 | 公布后20日净收益 | 次日入场日 |",
        "|---|---:|---:|---:|---|",
    ]
    for month, group in by_month.items():
        row = group[0]
        lines.append(f"| {month} | {row['pre_release_20d_return']:+.2%} | {row['net_return_5d']:+.2%} | {row['net_return_20d']:+.2%} | {row['entry_date']} |")
    lines += ["",
        "这几次历史对照足以否定‘社融超预期必然对应正收益’的确定性说法，没有证明反向交易，也不足以估计稳定胜率。11月社融超预期1900亿元，随后20日净收益-3.54%；2月社融低于一财均值6270亿元，随后+10.06%；3月高于报道回述的路透预期11160亿元，随后-8.62%。", "",
        "3月数据后的窗口保留至5月16日，包括5月贸易冲击；但4月15日至4月30日指数已经下跌3.66%（毛收益），全部亏损也不能归给5月消息。同期政策预期报道与价格路径只能构成竞争解释，不能据此分配因果贡献或事后改退出日。上述并发消息和原始来源见上一轮" + local(PRIOR / "context_supplement.json", "窗口时序补充") + "。", "",
        "**可保留的发现与下一问题**", "",
        "已经落实的研究修正是：按整体融资约束、增长预期、政策反应和已有价格变化来研究指数，而非把个股叙事加总；区分数据同比、调查意外和发布后可得到的收益。单月社融预期差暂不升级为独立择时规则。", "",
        "下一项有具体价值的历史问题，是在已固定的2019年1月至4月时间线上，核对政策操作及其同期理由与不同期限人民币利率如何变化，区分资金供给、增长改善及宽松节奏预期。事件日期和比较区间应先由公布日固定，利率变化也不能自动归为政策预期。它只补指数传导证据，不复活已拒绝的旧信用利差策略。", "",
        "本轮新增17项有正文定位的调查数字，其中11项有早于公布日的公开页面；只复用原有收益。事件按10万元预算、单边0.1%滑点、0.04%佣金且最低5元，并纳入份额、价位及分红，收益分母为实际投入资金。它不是20万元完整账户回测。六个20日窗口相连为四段，亦不是六次独立验证。", "",
        "只完成了一次历史机制发现：没有拟合新参数，没有新账户或新夏普，成本后夏普1.2及年化10%的共同目标仍未达成。历史资料目前重新取得，不能声称拥有不可改写的首次发布快照；四格主来源预期缺失保持缺失。", "",
        "文件：" + "、".join([local(OUT / "expectations.json", "调查提取"), local(OUT / "comparison.json", "完整21行比较"),
                                       local(OUT / "context_supplement.json", "同期政策解释"), local(OUT / "source_resolution.json", "调查来源"),
                                       local(OUT / "metadata_corrections.json", "单位和证据说明更正"),
                                       local(OUT / "信用预期差与指数历史表现.png", "图表"),
                                       local(ROOT / "research/historical_index_credit_expectation_gap_v1.py", "研究脚本")]) + "。", "",
    ]
    name = "历史发现_信用预期差与指数传导.md"
    (OUT / name).write_text("\n".join(lines), encoding="utf-8")
    draw(rows)
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    result.update(report=name, report_completed_at=now(),
                  discovery="六个月公开调查区分了同比变化与预期差；超预期与发布后指数收益并非单向对应。同期材料支持增长预期与近期政策节奏的双重传导，尚未识别价格贡献。",
                  next_historical_question="固定在2019年1月至4月既有政策及金融数据公布日，检查人民币资金价格与不同期限国债利率、央行同期操作理由；区分资金供给与增长、政策节奏预期，先作指数传导历史证据，不恢复旧信用利差策略或按收益挑窗口。")
    save("result.json", result)
    print("已保存信用预期差与指数传导报告及图表，保留六个月和全部缺失项。", flush=True)


def record_progress():
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    if not (OUT / result["report"]).is_file():
        raise RuntimeError("报告尚未生成。")
    prefix = "reports/research/510300_historical_index_credit_expectation_gap_v1/"
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
                       latest_historical_index_credit_expectation_gap=prefix + "result.json",
                       latest_continuation_report=prefix + result["report"], latest_historical_report=prefix + result["report"],
                       latest_continuation_classification=result["classification"], current_driver_continuation_classification=result["classification"],
                       current_driver_consecutive_blocked_goal_turns=0, latest_historical_diagnostic_at=stamp,
                       latest_goal_service_status="active", latest_goal_service_status_observed_at=stamp, goal_status="active", goal_achieved=False,
                       local_goal_work_status="ACTIVE_HISTORICAL_ONLY", last_research_result=result["discovery"],
                       last_source_result="保存六个月路透调查或原报道回述，独立保留一财2月调查和4月12日政策预期采访；17项预测、4格路透预期缺失。",
                       next_research_question=result["next_historical_question"])
        changes.append({"path": str(path.relative_to(ROOT)), "fields": {k: {"before": before.get(k), "after": v} for k, v in cfg.items() if before.get(k) != v}})
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save("authority_update.json", {"recorded_at": stamp, "study_id": STUDY, "previous_goal_turn_classification": "PROGRESS",
                                  "current_goal_turn_classification": "PROGRESS_COMPLETED_INDEX_CREDIT_EXPECTATION_RECONSTRUCTION", "changes": changes,
                                  "goal_achieved": False, "orders_authorized": False})
    print("已登记指数融资预期研究进展；夏普目标保持未达成。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="指数融资调查预期的历史重建")
    parser.add_argument("mode", choices=["prepare", "fetch", "resolve-sources", "build", "analyze", "enrich-context", "report", "record-progress"])
    args = parser.parse_args()
    {"prepare": prepare, "fetch": fetch, "resolve-sources": resolve_sources, "build": build, "analyze": analyze,
     "enrich-context": enrich_context, "report": report, "record-progress": record_progress}[args.mode]()


if __name__ == "__main__":
    main()
