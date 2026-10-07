"""拆解固定半年信用融资渠道与贷款期限，不从已见收益筛选参数。"""
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

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from credit_recency_structure_v23 import Ledger, FIELDS, add
from historical_index_credit_price_decomposition_v1 import macro_context
from historical_index_liquidity_transmission_v1 import clean
from historical_index_credit_constraint_relief_v1 import assign_clusters
from historical_price_gap_causes_v1 import round_trip

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_index_credit_flow_composition_v1"
STUDY = "510300_HISTORICAL_INDEX_CREDIT_FLOW_COMPOSITION_V1"
MONTHS = [str(m) for m in pd.period_range("2018-10", "2019-03", freq="M")]
MARKET = ROOT / "reports/research/510300_macro_dynamic_reframe_v1/inputs/market.parquet"
ORIGINALS = ROOT / "reports/research/510300_macro_transmission_context_v4/inputs/loan_originals.json"
SOURCES = [
    {"id": "tsf_201810", "month": "2018-10", "date": "2018-11-13", "url": "https://news.cnstock.com/news,bwkx-201811-4297543.htm", "required": ["7141亿元", "7288亿元", "2018-11-13"], "kind": "央行金融与社融原报告同期转载"},
    {"id": "tsf_201811", "month": "2018-11", "date": "2018-12-11", "url": "https://news.cnstock.com/news,bwkx-201812-4309545.htm", "required": ["3163亿元", "1.52万亿元", "2018-12-11"], "kind": "同期报道逐项转述央行原报告"},
    {"id": "tsf_201812", "month": "2018-12", "date": "2019-01-15", "url": "https://m.cnr.cn/finance/20190115/t20190115_524483356.html", "required": ["19.26万亿元", "1.59万亿元", "2019-01-15"], "kind": "央行原报告同期转述，分项为全年"},
    {"id": "tsf_201901", "month": "2019-01", "date": "2019-02-15", "url": "https://www.chinanews.com/fortune/2019/02-15/8755422.shtml", "required": ["4.64万亿元", "3786亿元", "2019年02月15日"], "kind": "央行原报告同期转述"},
    {"id": "tsf_201902", "month": "2019-02", "date": "2019-03-10", "url": "https://www.yicai.com/news/100135175.html", "required": ["5.31万亿元", "7030亿元", "2019-03-10"], "kind": "央行署名原文同期转载"},
    {"id": "tsf_201903", "month": "2019-03", "date": "2019-04-12", "url": "https://finance.sina.com.cn/china/2019-04-12/doc-ihvhiewr5278482.shtml", "required": ["8.18万亿元", "2.86万亿元", "2019年04月12日"], "kind": "央行署名原文同期转载，分项为一季度"},
    {"id": "pboc_feb_briefing", "month": None, "date": "2019-02-15", "url": "https://finance.sina.cn/2019-02-15/detail-ihrfqzka6140654.d.html", "required": ["票据融资", "真实", "季节性"], "kind": "央行金融统计数据解读实录同期转载"},
]
TSF_LABELS = {
    "rmb_real": "对实体经济发放的人民币贷款", "fx_real": "对实体经济发放的外币贷款折合人民币",
    "entrusted": "委托贷款", "trust": "信托贷款", "undiscounted_bills": "未贴现的银行承兑汇票",
    "corporate_bonds": "企业债券净融资", "local_special_bonds": "地方政府专项债券净融资",
    "equity_financing": "非金融企业境内股票融资",
}
LOAN_LABELS = {"rmb_total": "金融机构人民币贷款合计", "household_total": "住户贷款", "household_short": "住户短期",
               "household_long": "住户中长期", "corporate_total": "企事业贷款", "corporate_short": "企事业短期",
               "corporate_long": "企事业中长期", "bills": "票据融资", "nonbank_total": "非银行业金融机构贷款"}


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("已有研究范围，不覆盖。")
    originals = json.loads(ORIGINALS.read_text(encoding="utf-8"))
    by_month = {r["stat_month"]: r for r in originals}
    prior = json.loads((ROOT / "reports/research/510300_historical_index_credit_constraint_relief_v1/protocol.json").read_text(encoding="utf-8"))
    events = [{"id": m, "date": by_month[m]["conservative_known_at"][:10], "loan_source": by_month[m]["source_url"]} for m in MONTHS]
    save("protocol.json", {
        "study_id": STUDY, "recorded_at": now(), "previous_goal_turn_classification": "PROGRESS",
        "mode": "HISTORICAL_DISCOVERY_ONLY", "research_unit": "整体指数及全社会信用传导",
        "question": "贷款增长是否被其他融资收缩抵消，短期周转与长期融资分别改变多少；改善数据公布后指数还有多少可实现收益？",
        "stat_months": MONTHS, "events": events, "sources": SOURCES, "costs": prior["costs"],
        "horizons_sessions": [5, 20], "pre_release_window_sessions": 20,
        "entry": "金融统计报告原可用日上界结束后的下一交易日开盘，固定5/20个交易日后开盘退出。",
        "price_before": "公布日前最后收盘相对20个交易日前收盘的含分红回报；不含公布日反应。另存该收盘至入场开盘价格变化。",
        "data_reuse": "复用既有贷款原报告及V23区间账本，原104个月研究与失败规则不改。新增同期社融流量原文及上游解释。",
        "loan_period": "原文直接单月优先；未直接披露时按既有累计端点差重建，保留显示舍入界与跨公告性质。",
        "tsf_period": "月度与年内累计分别保存。2018年末及2019年一季度若仅有累计分项，不将其当作12月或3月分项，也不倒用最新修订表。",
        "comparisons": "同一公告直接给出的同比多增优先。贷款2019年同比复用2018年对应区间；2018年缺上年分项则留缺失。春节比较保留2019年1月、1至2月及一季度，不选最佳区间。",
        "accounting_limits": ["银行贷款含非银金融机构贷款，社融中的人民币贷款面向实体，不能相加", "已贴现票据属于银行贷款；未贴现承兑汇票另列社融渠道", "贷款净增不是新发放总量，也不能直接识别投资、消费或股票资金流", "中长期期限不代表项目用途，未逐笔识别融资置换", "2018年已扩围ABS、贷款核销及地方专项债，其他分项和舍入残差不能强归因"],
        "model": "没有新增阈值、择时策略或收益分组优化；不把同比多增称为市场预期差。六次结果完整列示，交叠窗口明确标注。",
        "known_history": "市场历史及既有贷款结构研究均已见过；不是独立验证。原报告为当前保存的历史页面，不是首次发布不可改写快照。",
        "new_parameters_fitted": 0, "new_full_accounts": 0, "net_sharpe": None, "goal_achieved": False, "orders_authorized": False,
    })
    print("已固定六次月度发布，贷款区间沿用旧账本，新增社融渠道分解及公布前后收益。", flush=True)


def fetch_one(source):
    item = dict(source)
    path = OUT / "sources" / (source["id"] + ".html")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(source["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=(10, 25))
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        body = unicodedata.normalize("NFKC", soup.get_text(" ", strip=True))
        path.write_bytes(response.content)
        path.with_suffix(".txt").write_text(body, encoding="utf-8")
        compact = re.sub(r"\s+", "", body)
        missing = [word for word in source["required"] if word not in compact]
        item.update(status="RETRIEVED" if not missing else "BODY_CHECK_REQUIRED", missing=missing,
                    path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(response.content).hexdigest(), retrieved_at=now())
    except Exception as exc:
        item.update(status="MISSING_SOURCE", error=str(exc), retrieved_at=now())
    return item


def fetch():
    if (OUT / "source_manifest.json").exists():
        raise RuntimeError("已有来源记录，不覆盖。")
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch_one, SOURCES))
    save("source_manifest.json", records)
    print(json.dumps([{k: r.get(k) for k in ["id", "status", "missing", "error"]} for r in records], ensure_ascii=False, indent=2), flush=True)


def resolve_sources():
    if (OUT / "source_resolution.json").exists():
        raise RuntimeError("已有来源补充，不重复。")
    manifest = json.loads((OUT / "source_manifest.json").read_text(encoding="utf-8"))
    notes = []
    for row in manifest:
        if row["id"] == "tsf_201810":
            original_url = row["url"]
            replacement = {**row, "url": "https://finance.stockstar.com/SS2018111300000991.shtml", "kind": "同期央行报告全文转载"}
            path = ROOT / row["path"]
            body = re.sub(r"\s+", "", path.with_suffix(".txt").read_text(encoding="utf-8"))
            if "证券之星" in body and all(word in body for word in row["required"]):
                result = {**replacement, "status": "RETRIEVED", "missing": [],
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "retrieved_at": datetime.fromtimestamp(path.stat().st_mtime, ZoneInfo("Asia/Shanghai")).isoformat()}
            else:
                result = fetch_one(replacement)
            result["original_url"] = original_url
            result["original_incomplete_body_retained"] = False
            row.clear()
            row.update(result)
            notes.append("上证原页只有10月社融合计，改用同日含全部分项的央行报告转载；公布日与事件范围不变。")
        if row["id"] == "tsf_201901":
            path = ROOT / row["path"]
            soup = BeautifulSoup(path.read_bytes().decode("gb18030", errors="replace"), "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = unicodedata.normalize("NFKC", soup.get_text(" ", strip=True))
            compact = re.sub(r"\s+", "", text)
            assert all(word in compact for word in row["required"])
            path.with_suffix(".txt").write_text(text, encoding="utf-8")
            article_start = text.index("央行网站15日发布")
            article_end = text.index("非金融企业境内股票融资", article_start)
            assert "\ufffd" not in text[article_start:article_end + 70]
            row.update(status="RETRIEVED", decoding="gb18030，外围混合编码字符替换；数字正文无替换字符", missing=[])
            notes.append("中新网正文为GB编码，外围混合编码片段无法严格解码；修正自动误判，检查数字正文无替换字符，原始字节不变。")
        if row["id"] == "pboc_feb_briefing":
            body = (ROOT / row["path"]).with_suffix(".txt").read_text(encoding="utf-8")
            assert all(word in body for word in ["票据融资", "季节性", "7108亿元", "结构性去杠杆", "押品不足"])
            row.update(status="RETRIEVED", content_scope_note="原关键词真实并不存在；仅采用实际正文中的周转、期限、部门和信用约束解释。", missing=[])
            notes.append("央行实录正文完整；删除未在正文出现的预期关键词要求，不将票据一概解释为真实投资或套利。")
    save("source_resolution.json", {"recorded_at": now(), "before_returns": not (OUT / "result.json").exists(), "notes": notes, "sources": manifest})
    print(json.dumps({"说明": notes, "状态": [{"id": r["id"], "status": r["status"], "missing": r.get("missing")} for r in manifest]}, ensure_ascii=False, indent=2), flush=True)


def quantity(number, unit):
    scale = 10000 if unit == "万亿元" else 1
    digits = len(number.split(".")[1]) if "." in number else 0
    return float(number) * scale, .5 * scale * 10 ** (-digits)


def change_after(text):
    paired = re.search(r"分别比上月和上年同期多([\d.]+)(万亿元|亿元)和([\d.]+)(万亿元|亿元)", text[:65])
    if paired:
        value, bound = quantity(paired[3], paired[4])
        return {"value_yi": value, "display_rounding_half_yi": bound, "literal": paired[0]}
    found = re.search(r"(?:同比|比上年同期|比上年)(多增|少增|多减|少减|多|少)([\d.]+)(万亿元|亿元)", text[:65])
    if not found:
        return None
    value, bound = quantity(found[2], found[3])
    sign = -1 if found[1] in ["少增", "多减", "少"] else 1
    return {"value_yi": sign * value, "display_rounding_half_yi": bound, "literal": found[0]}


def extract_tsf_block(source, period, total_pattern, start_literal=None):
    text = re.sub(r"\s+", "", (OUT / "sources" / (source["id"] + ".txt")).read_text(encoding="utf-8"))
    if start_literal:
        pos = text.find(start_literal)
        if pos < 0:
            raise ValueError("未找到区间：" + start_literal)
        text = text[pos:]
    candidates = list(re.finditer(total_pattern + r"([\d.]+)(万亿元|亿元)", text))
    if not candidates:
        raise ValueError("未找到社融合计：" + source["id"] + "/" + period)
    total = next((m for m in candidates if change_after(text[m.end():]) is not None), candidates[0])
    value, bound = quantity(total[1], total[2])
    fields = {}
    for key, label in TSF_LABELS.items():
        pattern = label + (r"(增加|减少)([\d.]+)(万亿元|亿元)" if key not in ["corporate_bonds", "local_special_bonds", "equity_financing"] else r"(-?[\d.]+)(万亿元|亿元)")
        match = re.search(pattern, text)
        if not match:
            fields[key] = None
            continue
        if key not in ["corporate_bonds", "local_special_bonds", "equity_financing"]:
            amount, half = quantity(match[2], match[3])
            if match[1] == "减少":
                amount = -amount
        else:
            amount, half = quantity(match[1], match[2])
        fields[key] = {"value_yi": amount, "display_rounding_half_yi": half, "literal": match[0], "reported_yoy_change": change_after(text[match.end():])}
    residual = value - sum(f["value_yi"] for f in fields.values()) if all(fields.values()) else None
    return {"source_id": source["id"], "stat_month": source["month"], "period": period, "date": source["date"],
            "source_url": source["url"], "total_yi": value, "total_rounding_half_yi": bound,
            "total_literal": total[0], "reported_total_yoy_change": change_after(text[total.end():]), "fields": fields,
            "unlisted_and_rounding_residual_yi": residual,
            "interpretation": "分项净增及同比差异，不等于直接投资用途；残差保留为未单列其他及显示舍入，不能强称全是某一渠道。"}


def build():
    if (OUT / "composition.json").exists():
        raise RuntimeError("已完成分解，不覆盖。")
    manifest = json.loads((OUT / "source_resolution.json").read_text(encoding="utf-8"))["sources"]
    by_id = {r["id"]: r for r in manifest}
    failed = [r["id"] for r in manifest if r["month"] and r["status"] != "RETRIEVED"]
    if failed:
        raise RuntimeError("正文需先解决：" + "、".join(failed))
    specs = [
        ("tsf_201810", "MONTH", "10月份社会融资规模增量为", None),
        ("tsf_201811", "MONTH", "11月份社会融资规模增量为", None),
        ("tsf_201812", "YTD", "2018年社会融资规模增量累计为", None),
        ("tsf_201901", "MONTH", "1月份社会融资规模增量为", None),
        ("tsf_201902", "YTD", "2019年前两个月社会融资规模增量累计为", None),
        ("tsf_201902", "MONTH", "社会融资规模增量为", "2月当月,社会融资规模增量为"),
        ("tsf_201903", "YTD", "2019年一季度社会融资规模增量累计为", None),
    ]
    blocks = [extract_tsf_block(by_id[key], period, pattern, start) for key, period, pattern, start in specs]
    monthly_headlines = []
    for month in MONTHS:
        source = by_id["tsf_" + month.replace("-", "")]
        direct = next((b for b in blocks if b["stat_month"] == month and b["period"] == "MONTH"), None)
        if direct:
            monthly_headlines.append({"stat_month": month, "date": source["date"], "total_yi": direct["total_yi"], "reported_total_yoy_change": direct["reported_total_yoy_change"], "component_status": "DIRECT_MONTHLY_COMPONENTS"})
        else:
            body = re.sub(r"\s+", "", (OUT / "sources" / (source["id"] + ".txt")).read_text(encoding="utf-8"))
            pattern = ("12月份" if month == "2018-12" else "3月份") + r"社会融资规模增量为([\d.]+)(万亿元|亿元)"
            match = re.search(pattern, body)
            if not match:
                raise ValueError("缺单月合计：" + month)
            amount, half = quantity(match[1], match[2])
            monthly_headlines.append({"stat_month": month, "date": source["date"], "total_yi": amount, "display_rounding_half_yi": half,
                                      "reported_total_yoy_change": change_after(body[match.end():]), "component_status": "NO_DIRECT_MONTHLY_COMPONENTS_IN_SELECTED_REPORT"})
    ledger = Ledger()
    loan_rows, comparisons = [], []
    for month in MONTHS:
        windows = ["1", "YTD"] if month.startswith("2019") else ["1"]
        for window in windows:
            for field in FIELDS:
                row = ledger.row(month, field, window)
                row["label"] = LOAN_LABELS[field]
                loan_rows.append(row)
                if month.startswith("2019"):
                    prior_month = str(pd.Period(month, "M") - 12)
                    old = ledger.row(prior_month, field, window)
                    expr = add((1, json.loads(row["expression"])), (-1, json.loads(old["expression"])))
                    diff, bound, known, status = ledger.evaluate(expr)
                    comparisons.append({"stat_month": month, "window": window, "field": field, "label": LOAN_LABELS[field],
                                        "value_yi": row["value_yi"], "prior_year_value_yi": old["value_yi"], "yoy_cross_report_delta_yi": diff,
                                        "display_rounding_bound_yi": bound, "known_at": known, "status": status,
                                        "interpretation": "对应历史公告的同区间差，非后来统一修订同比；金额净增不等于新发放。"})
    save("composition.json", {"computed_at": now(), "tsf_blocks": blocks, "monthly_tsf_headlines": monthly_headlines,
                              "loan_rows": loan_rows, "loan_yoy_comparisons": comparisons})
    print("完成七个社融区间、六个月贷款结构；12月和3月社融分项保持累计口径。", flush=True)
    print(pd.DataFrame(comparisons).query("window == 'YTD'")[["stat_month", "label", "value_yi", "yoy_cross_report_delta_yi", "display_rounding_bound_yi"]].to_string(index=False), flush=True)


def analyze():
    if (OUT / "result.json").exists():
        raise RuntimeError("已有收益结果，不重算或改选。")
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    composition = json.loads((OUT / "composition.json").read_text(encoding="utf-8"))
    market = pd.read_parquet(MARKET).sort_values("date").reset_index(drop=True)
    market["date"] = pd.to_datetime(market.date).dt.tz_localize(None).dt.normalize()
    dates = pd.DatetimeIndex(market.date)
    dividends = pd.read_csv(ROOT / "data/reference/510300_dividends.csv")
    dividends = dividends[dividends.symbol.eq("510300.SH")].copy()
    dividends["record_date"] = pd.to_datetime(dividends.record_date)
    rows, checks = [], []
    for event in protocol["events"]:
        date = pd.Timestamp(event["date"])
        entry_idx = dates.searchsorted(date, side="right")
        before_idx = dates.searchsorted(date, side="left") - 1
        entry, before = market.iloc[entry_idx], market.iloc[before_idx]
        pre_return = float(before.wealth / market.iloc[before_idx - 20].wealth - 1)
        assert entry.date > date and before.date < date
        for horizon in protocol["horizons_sessions"]:
            end = market.iloc[entry_idx + horizon]
            cash_per_share = float(dividends[(dividends.record_date >= entry.date) & (dividends.record_date < end.date)].cash_dividend_per_share.sum())
            trade = round_trip(float(entry.open), float(end.open), cash_per_share, protocol["costs"])
            gross = (float(end.open) + cash_per_share) / float(entry.open) - 1
            identity = (trade["sell_price"] - trade["buy_price"]) * trade["shares"] + trade["dividend_entitlement_cny"] - trade["commissions_cny"]
            assert trade["paid_cny"] <= protocol["costs"]["illustrative_event_budget_cny"] and trade["net_return"] <= gross + 1e-12
            checks.append(abs(identity - trade["net_pnl_cny"]))
            rows.append({"event_id": event["id"], "stat_month": event["id"], "date": event["date"], "horizon": horizon,
                         "entry_date": entry.date.strftime("%Y-%m-%d"), "exit_date": end.date.strftime("%Y-%m-%d"),
                         "entry_open": float(entry.open), "exit_open": float(end.open), "before_date": before.date.strftime("%Y-%m-%d"),
                         "pre_release_20d_total_return": pre_return, "before_close_to_entry_price_return": float(entry.open / before.close - 1),
                         "dividend_per_share": cash_per_share, "gross_return": gross, **trade})
    clusters = [assign_clusters(rows, h) for h in protocol["horizons_sessions"]]
    context = macro_context(protocol["events"])
    assert len(rows) == 12 and max(checks) < 1e-7
    for loan in composition["loan_rows"]:
        event = next(e for e in protocol["events"] if e["id"] == loan["stat_month"])
        assert pd.Timestamp(loan["known_at"]).date() <= pd.Timestamp(event["date"]).date()
    save("events_and_returns.json", rows)
    save("known_macro_context.json", context)
    save("calculation_checks.json", {"rows": len(rows), "max_cash_identity_error_cny": max(checks), "loan_inputs_known_before_entry": True,
                                    "monthly_vs_ytd_separated": True, "expectations_measured": False})
    save("result.json", {"study_id": STUDY, "computed_at": now(), "classification": "PROGRESS_INDEX_CREDIT_FLOW_COMPOSITION",
                         "event_count": 6, "overlap": clusters, "new_parameters_fitted": 0, "new_full_accounts": 0,
                         "net_sharpe": None, "goal_achieved": False, "orders_authorized": False})
    print(pd.DataFrame(rows)[["stat_month", "date", "horizon", "entry_date", "exit_date", "pre_release_20d_total_return", "net_return"]].to_string(index=False), flush=True)


def enrich_context():
    """收益计算后的竞争解释，只描述既定窗口，不改变事件或出入场。"""
    if (OUT / "context_supplement.json").exists():
        raise RuntimeError("已有补充解释，不重复。")
    assert (OUT / "result.json").exists()
    extra_sources = [
        {"id": "fed_jan30", "month": None, "date": "2019-01-30", "url": "https://www.federalreserve.gov/newsevents/pressreleases/monetary20190130a.htm", "required": ["January30,2019", "patient", "mutedinflationpressures"], "kind": "美联储原始政策声明，美国当地日期"},
        {"id": "fed_mar20_balance", "month": None, "date": "2019-03-20", "url": "https://www.federalreserve.gov/newsevents/pressreleases/monetary20190320c.htm", "required": ["March20,2019", "$15billion", "September2019"], "kind": "美联储原始缩表计划，美国当地日期"},
        {"id": "mfa_may06", "month": None, "date": "2019-05-06", "url": "https://www.fmprc.gov.cn/web/wjdt_674879/fyrbt_674889/201905/t20190506_7814709.shtml", "required": ["2019-05-06", "加征关税", "赴美磋商"], "kind": "外交部原始例行记者会"},
        {"id": "ustr_may10", "month": None, "date": "2019-05-10", "url": "https://ustr.gov/about-us/policy-offices/press-office/press-releases/2019/may/statement-us-trade-representative", "required": ["May10,2019", "$200billion", "10percentto25percent"], "kind": "美国贸易代表办公室原始声明，美国当地日期"},
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch_one, extra_sources))
    save("supplement_sources.json", records)
    market = pd.read_parquet(MARKET).copy()
    market.index = pd.to_datetime(market.date).dt.normalize()
    prices = {d: {key: float(market.loc[d, key]) for key in ["open", "close", "dividend"]}
              for d in ["2019-04-15", "2019-04-30", "2019-05-06", "2019-05-16"]}
    assert market.loc["2019-04-15":"2019-05-16", "dividend"].abs().sum() == 0
    path_returns = {
        "entry_open_to_apr30_close": prices["2019-04-30"]["close"] / prices["2019-04-15"]["open"] - 1,
        "apr30_close_to_may06_open": prices["2019-05-06"]["open"] / prices["2019-04-30"]["close"] - 1,
        "apr30_close_to_may06_close": prices["2019-05-06"]["close"] / prices["2019-04-30"]["close"] - 1,
        "may06_close_to_fixed_exit_open": prices["2019-05-16"]["open"] / prices["2019-05-06"]["close"] - 1,
        "whole_fixed_window_gross_return": prices["2019-05-16"]["open"] / prices["2019-04-15"]["open"] - 1,
    }
    chain = (1 + path_returns["entry_open_to_apr30_close"]) * (1 + path_returns["apr30_close_to_may06_close"]) * (1 + path_returns["may06_close_to_fixed_exit_open"]) - 1
    assert abs(chain - path_returns["whole_fixed_window_gross_return"]) < 1e-12
    save("context_supplement.json", {
        "recorded_at": now(), "selected_after_returns": True, "role": "竞争解释及历史路径描述，不能作为已验证筛选器或止损规则", "returns_recalculated": False,
        "source_status": [{"id": r["id"], "status": r["status"]} for r in records],
        "pboc_january_domestic_and_fx_corporate": {
            "source_id": "pboc_feb_briefing", "published_date": "2019-02-15", "currency_scope": "企业及其他单位本外币贷款，与人民币贷款表分开",
            "total_yoy_extra_yi": 8273, "short_and_bills_yoy_extra_yi": 7108, "long_yoy_extra_yi": 907,
            "short_and_bills_share_of_yoy_extra": 7108 / 8273,
            "interpretation": "约85.9%是同比多增的构成，不是存量占比、新发放用途或股市资金流。",
            "mechanism_evidence": "发布会解释票据利率下降、短期限与周转便利、再贴现引导及年初季节性；并指出押品不足与信贷责任追究约束放贷。说明机制，不识别单一政策的因果贡献。",
            "counterevidence": "同一实录的制造业中长期贷款余额增速11.1%，比上年末高0.6个百分点，故不能概括为全部融资都是无效周转。",
        },
        "april_window_path": {"prices": prices, "gross_returns": path_returns, "selection_note": "完整20日亏损保留；按五一假期前后拆路径为收益后解释，不删除5月6日，也不改在4月30日退出。"},
        "competition": [
            "1月30日美联储维持利率区间并提出耐心，解释涉及全球金融变化及低通胀压力；它发生在12月金融数据发布后的20日窗口中。",
            "3月20日美联储宣布从5月放缓国债缩表、原计划9月底结束总持仓缩减；位于2月金融数据发布后的20日窗口，不能提前填入3月11日信息。",
            "5月6日外交部已公开回应加征关税威胁；5月10日USTR确认上调约2000亿美元商品税率。该后续贸易冲击与3月金融数据后的20日窗口重叠，未量化因果贡献。",
        ],
        "limits": "没有覆盖所有并发新闻，也没有市场共识预测、纯粹信用冲击或对照组；不能从事后涨跌直接量出预期差。",
    })
    print(json.dumps({"来源状态": [{"id": r["id"], "status": r["status"], "missing": r.get("missing")} for r in records], "四月窗口价格路径": path_returns}, ensure_ascii=False, indent=2), flush=True)


def draw(rows, comparisons):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False, "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 10.2), gridspec_kw={"height_ratios": [1.18, 1]})
    fig.patch.set_facecolor("#fbfaf7")
    for ax in axes:
        ax.set_facecolor("#fbfaf7")
        ax.axhline(0, color="#59646b", lw=.8)
        ax.grid(axis="y", alpha=.16)
        ax.set_axisbelow(True)
    x = np.arange(len(MONTHS))
    twenty = [next(r for r in rows if r["stat_month"] == m and r["horizon"] == 20) for m in MONTHS]
    five = [next(r for r in rows if r["stat_month"] == m and r["horizon"] == 5) for m in MONTHS]
    for offset, values, color, label in [
        (-.25, [r["pre_release_20d_total_return"] * 100 for r in twenty], "#a4abb3", "公布前20日含分红涨跌（未扣成本）"),
        (0, [r["net_return"] * 100 for r in five], "#c79046", "公布后5日事件净收益"),
        (.25, [r["net_return"] * 100 for r in twenty], "#276b69", "公布后20日事件净收益"),
    ]:
        bars = axes[0].bar(x + offset, values, width=.235, color=color, label=label)
        axes[0].bar_label(bars, labels=[f"{v:+.1f}" for v in values], fontsize=9, padding=3)
    axes[0].set_xticks(x, [f"{m}\n发布 {r['date'][5:]}" for m, r in zip(MONTHS, twenty)])
    axes[0].set_ylabel("收益（%）")
    axes[0].set_ylim(-11.5, 19.8)
    axes[0].set_title("A  六个月完整保留：融资数据好坏与随后指数涨跌并非一一对应", loc="left", pad=13, fontsize=14)
    axes[0].legend(loc="upper left", fontsize=10, frameon=False, ncol=1)
    yoy = {(r["stat_month"], r["field"]): r["yoy_cross_report_delta_yi"] for r in comparisons if r["window"] == "YTD"}
    three_months = MONTHS[-3:]
    bottom_series = [
        ("短期贷款＋票据融资", "#276b69", [yoy[m, "corporate_short"] + yoy[m, "bills"] for m in three_months]),
        ("企事业中长期贷款", "#c79046", [yoy[m, "corporate_long"] for m in three_months]),
        ("企事业贷款合计", "#a4abb3", [yoy[m, "corporate_total"] for m in three_months]),
    ]
    for offset, (label, color, values) in zip([-.25, 0, .25], bottom_series):
        bars = axes[1].bar(np.arange(3) + offset, values, width=.235, color=color, label=label)
        axes[1].bar_label(bars, labels=[f"{v:+,.0f}" for v in values], fontsize=10, padding=3)
    axes[1].set_xticks(np.arange(3), ["2019年1月", "2019年1—2月", "2019年一季度"])
    axes[1].set_ylabel("较上年同区间多增（亿元）")
    axes[1].set_ylim(-2200, 18500)
    axes[1].set_title("B  人民币贷款同比多增主要集中在短期与票据；累计区间互相嵌套", loc="left", pad=13, fontsize=14)
    axes[1].legend(loc="upper left", frameon=False, ncol=1, fontsize=10)
    fig.suptitle("510300｜融资结构与指数阶段的历史对照", x=.08, y=.976, ha="left", fontsize=19, fontweight="bold")
    fig.text(.08, .938, "固定观察2018年10月至2019年3月；发布后下一交易日开盘进入，5／20个交易日后开盘退出", color="#535e63", fontsize=11)
    fig.text(.08, .023, "事件收益按10万元预算及原有成本计算，不是完整账户。贷款对比沿用原公告金额，保留舍入误差。\n六个20日窗口按重叠关系分成四段，不作为六次独立验证。数据与代码见同目录报告。", color="#535e63", fontsize=10)
    fig.subplots_adjust(left=.09, right=.97, top=.88, bottom=.11, hspace=.38)
    fig.savefig(OUT / "融资结构与指数阶段.png", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def report():
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    composition = json.loads((OUT / "composition.json").read_text(encoding="utf-8"))
    rows = json.loads((OUT / "events_and_returns.json").read_text(encoding="utf-8"))
    context = json.loads((OUT / "known_macro_context.json").read_text(encoding="utf-8"))
    supplement = json.loads((OUT / "context_supplement.json").read_text(encoding="utf-8"))
    extra_sources = json.loads((OUT / "supplement_sources.json").read_text(encoding="utf-8"))
    if any(s["status"] != "RETRIEVED" for s in extra_sources):
        raise RuntimeError("补充来源正文尚需核对。")
    source_records = json.loads((OUT / "source_resolution.json").read_text(encoding="utf-8"))["sources"] + extra_sources
    sources = {r["id"]: r for r in source_records}
    link = lambda key, label: f"[{label}]({sources[key]['url']})"
    blocks = {(b["stat_month"], b["period"]): b for b in composition["tsf_blocks"]}
    comparisons = composition["loan_yoy_comparisons"]
    yoy = {(r["stat_month"], r["field"]): r for r in comparisons if r["window"] == "YTD"}
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    prices = supplement["april_window_path"]["gross_returns"]
    lines = ["# 510300历史发现：融资为什么变化，指数为何不同步", "",
        "本轮研究单位为整体指数。固定保留2018年10月至2019年3月六次金融数据发布，拆解社融渠道、贷款期限以及发布前后的指数表现。没有增加个股研究，也没有用收益选择月份、阈值或持有天数。", "",
        "**最有用的发现**", "",
        "融资数量、融资约束与股票定价是三个环节。2018年末银行贷款的同比多增，并未抵消委托、信托等渠道的收缩；2019年初信用多增主要落在短期贷款和票据，同时存在部分中长期改善。由此可以识别融资变化的来源，却不能直接推出指数的买卖方向。", "",
        "2019年2月单月社融同比少增4847亿元，公布后20日指数事件净收益为+10.06%；3月同比多增1.28万亿元，同样窗口却为-8.62%。这两个反例否定了本段历史中‘社融同比多增就买、少增就回避’的简单对应关系；没有证明反向交易有效。", "",
        "指数研究需要同时解释整体现金流预期、折现率、风险补偿及股票配置需求。统计上的融资扩张可先缓解债务接续压力，盈利和订单恢复则是另一环节；前者影响风险定价是一条合理机制，本轮尚未分离出它的实际价格贡献。", "",
        "**2018年末：银行贷款增加为何仍伴随整体信用偏弱**", "",
        "10月对实体人民币贷款同比多增506亿元，但委托、信托、未贴现承兑汇票三项合计同比少增3749亿元；11月对应数字为+874亿元和-3633亿元。11月企业债券融资同比多增2310亿元，又受到地方专项债同比少增2614亿元等项目抵消。只盯一项贷款，会漏掉渠道替换、表外收缩和政府融资节奏。" + link("tsf_201810", "10月央行报告同期转载") + "、" + link("tsf_201811", "11月报告同期报道") + "。", "",
        "上游机制方面，央行2月15日实录将此前表外融资回落与结构性去杠杆联系起来；同时指出押品不足和信贷人员责任约束会限制企业获得贷款。这个解释说明‘银行有资金’与‘愿意承担借款人信用风险’之间还有门槛。它是后来公开的机制复盘，未倒填为2018年10月的交易信息。" + link("pboc_feb_briefing", "央行金融统计解读实录") + "。", "",
        "**2019年初：哪类信用在恢复**", "",
        "央行同一场实录披露，1月企业及其他单位本外币贷款同比多增8273亿元，其中短期贷款与票据融资多增7108亿元，约占85.9%。这是‘同比额外增量’的构成，不能读成全部贷款的85.9%都是票据。实录将票据扩张联系到利率下降、短期限周转便利、再贴现支持和季节性；同时制造业中长期贷款余额增速也较上年末上升，不能把整轮改善归为无效融资。" + link("pboc_feb_briefing", "央行对期限及投向的解释") + "。", "",
        "下面使用人民币口径的历史公告作同区间对比，与上一段本外币口径分开。单位为亿元，数值后的±仅为原公告显示精度形成的舍入界，不能解释为统计置信区间。", "",
        "| 年内区间 | 企事业贷款同比多增 | 短期＋票据同比多增 | 中长期同比多增 | 住户中长期同比多增 |", "|---|---:|---:|---:|---:|"]
    for month, label in zip(MONTHS[-3:], ["2019年1月", "2019年1—2月", "2019年一季度"]):
        vals = []
        for field in ["corporate_total", "corporate_short", "corporate_long", "household_long"]:
            r = yoy[month, field]
            val, bound = r["yoy_cross_report_delta_yi"], r["display_rounding_bound_yi"]
            if field == "corporate_short":
                val += yoy[month, "bills"]["yoy_cross_report_delta_yi"]
                bound += yoy[month, "bills"]["display_rounding_bound_yi"]
            vals.append(f"{val:+,.0f} ±{bound:g}")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    lines += ["",
        "1—2月企事业中长期贷款反而同比少增约700亿元，短期及票据多增超过贷款合计的多增，是因为其他项目有抵消。到一季度中长期同比差转正约1200亿元，但短期及票据仍是主要增量。这支持‘信用接续和周转改善领先于广泛长期扩张’这一历史描述，不足以认定贷款实际用于新增产能。", "",
        "社融渠道也呈现不同过程：2019年1月委托和信托合计净减少354亿元，未贴现承兑汇票净增3786亿元；1—2月后者累计仅增683亿元。到一季度，委托和信托合计仍净减少1442亿元，但同比少减1140亿元。收缩减缓与已经转为正增长，是两种不同状态。" + link("tsf_201901", "1月数据") + "、" + link("tsf_201902", "1—2月数据") + "、" + link("tsf_201903", "一季度数据") + "。", "",
        "**六次发布前后，指数已经反映了多少**", "",
        "下表公布前收益截止于公布日前最后一个收盘，向前取20个交易日、含分红、不扣交易成本；公布后收益从公开日结束后的下一交易日开盘进入，5或20个交易日后开盘退出。社融同比多增是相对上年，不是相对市场共识。", "",
        "| 统计月／公开日 | 单月社融同比多增（亿元） | 公布前20日 | 公布后5日净收益 | 公布后20日净收益 | 20日入场→退出 |", "|---|---:|---:|---:|---:|---|"]
    for headline in composition["monthly_tsf_headlines"]:
        month = headline["stat_month"]
        five = next(r for r in rows if r["stat_month"] == month and r["horizon"] == 5)
        twenty = next(r for r in rows if r["stat_month"] == month and r["horizon"] == 20)
        change = headline["reported_total_yoy_change"]["value_yi"]
        lines.append(f"| {month}／{headline['date']} | {change:+,.0f} | {twenty['pre_release_20d_total_return']:+.2%} | {five['net_return']:+.2%} | {twenty['net_return']:+.2%} | {twenty['entry_date']}→{twenty['exit_date']} |")
    lines += ["",
        "1月、2月和3月数据公布前20日，指数分别已上涨10.47%、12.47%和7.62%。这只能证明价格已有变化，不能仅凭涨幅断言已充分定价，更不能把全部上涨归因于尚未公布的数据。1月16日至4月9日三个20日窗口互相连接，六个20日窗口按重叠关系只分成四段；不将其作为六次独立试验计算夏普或显著性。", "",
        "各发布日之前已公开的制造业新订单指数依次为" + "、".join(f"{c['pmi_new_orders']:.1f}" for c in context) + "。2月社融公布时，制造业新订单已从49.6回到50.6，因此‘单月融资转弱’与‘需求信息改善’可以同时存在。制造业订单只覆盖部分经济，不能代替整个沪深300的盈利情况。逐行来源保存在known_macro_context.json。", "",
        "**为什么不能把这轮涨跌全部归给信用数据**", "",
        "本轮正收益窗口同时包含外部金融条件变化。美联储1月30日维持利率区间、提出对后续调整保持耐心，解释涉及全球经济金融状况与较低通胀压力；3月20日又公布自5月放缓国债缩表、当时计划9月底结束总持仓缩减。这两份公告分别落在12月和2月数据公布后的20日窗口，属于竞争解释，没有提前用于入场。" + link("fed_jan30", "1月30日声明") + "、" + link("fed_mar20_balance", "3月20日缩表计划") + "。", "",
        "3月数据后的负收益窗口延伸到5月16日，包含贸易摩擦再升级。5月6日外交部已回应加征关税威胁，5月10日美国贸易代表办公室确认将约2000亿美元商品的附加税率从10%提高到25%。这些是后来发生的信息，不能凭4月12日金融数据预知。" + link("mfa_may06", "5月6日记者会") + "、" + link("ustr_may10", "5月10日声明") + "。", "",
        f"但也不能把全部亏损都归给关税：4月15日入场开盘至4月30日收盘已下跌{abs(prices['entry_open_to_apr30_close']):.2%}；4月30日收盘至5月6日收盘再跌{abs(prices['apr30_close_to_may06_close']):.2%}，其中下一交易日开盘缺口为{prices['apr30_close_to_may06_open']:.2%}；5月6日收盘至固定退出开盘变动{prices['may06_close_to_fixed_exit_open']:+.2%}。这是未扣成本的价格路径，复合后为完整窗口毛收益{prices['whole_fixed_window_gross_return']:+.2%}，并未更改退出日期。", "",
        "上述路径拆分与外部公告是在收益计算后补充的解释，只能帮助避免单因子归因；不能变成删去坏月份、提前止损或按新闻事后切段的策略。也没有覆盖全部并发消息。", "",
        "**保留下来的指数研究判断**", "",
        "可以保留的发现是：相同的融资总量变化，可能来自银行信贷替代、表外收缩放缓、票据周转、长期贷款或政府债券等不同来源，它们改变的经济约束并不相同。指数同时面对总量现金流、利率与风险补偿的变化，一个月的信贷方向不够确定买卖方向。", "",
        "尚未建立的是：这些机制中哪一条超出当时市场预期，以及它在可入场之后留下多少可重复的净收益。本轮没有拿上涨倒推‘政策可信度上升’，也没有把数据改善后的下跌都解释成‘利好兑现’。后续问题应维持在指数层面，补出这些历史发布之前的公开预期与预期分歧，并把后续外部冲击按时序记录；不能从已见涨跌反造预期差。", "",
        "**口径、范围和交付**", "",
        "贷款净增不等于新发放总量；期限不等于实际用途；银行人民币贷款包含非银金融机构，社融人民币贷款面向实体；已贴现和未贴现票据分属不同项目，不重复相加。2018年社融扩围背景下，未列其他项目和显示舍入残差保留在账本。2018年12月与2019年3月采用来源只给出累计分项，未将全年或一季度当作单月分项。", "",
        "不同公告也不是完全一致的历史版本：1月企业债券4990亿元加2月单月805亿元，与2月公告的前两月累计5546亿元相差249亿元，超过这些数值的显示舍入界。原因未在本轮确证，保留原文差异，不强凑相等；这也说明不能用较晚累计值无痕重写先前单月信息。", "",
        "事件收益按10万元预算、单边0.1%滑点和0.04%佣金（最低5元），纳入份额取整、价格档位与分红权益；分母为实际投入资金。12行收益的可用日期及现金流恒等式已核对。这不是20万元完整账户回测，没有计算新夏普，也没有证明成本后夏普1.2或年化10%达标。", "",
        "本轮历史价格与既有贷款结构曾被研究使用，不是独立验证。来源为现在保存的历史原报告或同期转载，未声称拥有不可改写的首次发布快照。", "",
        "文件：" + "、".join(f"[{label}](<{(OUT / name).as_posix()}>)" for name, label in [
            ("composition.json", "融资渠道及贷款结构"), ("events_and_returns.json", "六次事件收益明细"),
            ("context_supplement.json", "竞争解释与窗口路径"), ("source_resolution.json", "原报告来源"),
            ("protocol.json", "固定范围"), ("融资结构与指数阶段.png", "结构与收益图")]) + "。", "",
        "六个月贷款来源：" + "、".join(f"[{e['id']}]({e['loan_source']})" for e in protocol["events"]) + "。"]
    name = "历史发现_融资结构与指数阶段.md"
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    draw(rows, comparisons)
    result.update(report=name, report_completed_at=now(),
                  discovery="2018年末银行贷款未抵消表外收缩；2019年初信用多增偏短期及票据。六次发布反例说明单月融资方向不能直接决定指数交易，既有价格、订单和外部冲击必须按时序区分。",
                  next_historical_question="继续以指数整体为单位，检查2018年第四季度至2019年第一季度这些已固定发布之前的公开预期与分歧；分清同比改善、此前预期及后续外部冲击，预期缺失时保留缺失，不从收益反造预期差。")
    save("result.json", result)
    print("已保存融资结构与指数阶段报告、完整六次结果和图表；没有新策略或达标声明。", flush=True)


def record_progress():
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    assert (OUT / result["report"]).is_file()
    prefix = "reports/research/510300_historical_index_credit_flow_composition_v1/"
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
                       latest_historical_index_credit_flow_composition=prefix + "result.json",
                       latest_continuation_report=prefix + result["report"], latest_historical_report=prefix + result["report"],
                       latest_continuation_classification=result["classification"], current_driver_continuation_classification=result["classification"],
                       current_driver_consecutive_blocked_goal_turns=0, latest_historical_diagnostic_at=stamp,
                       latest_goal_service_status="active", latest_goal_service_status_observed_at=stamp, goal_status="active", goal_achieved=False,
                       local_goal_work_status="ACTIVE_HISTORICAL_ONLY", last_research_result=result["discovery"],
                       last_source_result="保存六个月同期社融原文及央行机制解释，复用原贷款账本；补充美联储和贸易政策原始公告，仅用于竞争解释。",
                       next_research_question=result["next_historical_question"])
        changes.append({"path": str(path.relative_to(ROOT)), "fields": {k: {"before": before.get(k), "after": v} for k, v in cfg.items() if before.get(k) != v}})
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save("authority_update.json", {"recorded_at": stamp, "study_id": STUDY, "previous_goal_turn_classification": "PROGRESS",
                                  "current_goal_turn_classification": "PROGRESS_COMPLETED_INDEX_CREDIT_FLOW_COMPOSITION", "changes": changes,
                                  "goal_achieved": False, "orders_authorized": False})
    print("已登记指数融资结构研究进展；夏普目标保持未达成。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="指数融资渠道与贷款结构历史分解")
    parser.add_argument("mode", choices=["prepare", "fetch", "resolve-sources", "build", "analyze", "enrich-context", "report", "record-progress"])
    args = parser.parse_args()
    {"prepare": prepare, "fetch": fetch, "resolve-sources": resolve_sources, "build": build, "analyze": analyze,
     "enrich-context": enrich_context, "report": report, "record-progress": record_progress}[args.mode]()


if __name__ == "__main__":
    main()
