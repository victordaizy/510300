"""在固定历史季度内拆解美债变化，连接美联储原文与指数事件收益。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
from io import StringIO
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests

from historical_index_credit_constraint_relief_v1 import assign_clusters
from historical_index_liquidity_transmission_v1 import clean
from historical_price_gap_causes_v1 import round_trip

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_index_us_rates_policy_v1"
OLD = ROOT / "reports/research/510300_historical_index_credit_flow_composition_v1"
V19 = ROOT / "reports/research/510300_real_yield_company_exposure_v19/sources"
REPORT = "历史发现_美债下降的两种构成与指数.md"
EVENTS = [
    {"id": "2018-11-08", "decision": "维持2.00%—2.25%，仍预计逐步加息", "path": "CONTINUED_GRADUAL_INCREASES"},
    {"id": "2018-12-19", "decision": "上调至2.25%—2.50%，仍预计部分进一步加息", "path": "SOME_FURTHER_GRADUAL_INCREASES"},
    {"id": "2019-01-30", "decision": "维持2.25%—2.50%，转为耐心判断调整", "path": "PATIENT"},
    {"id": "2019-03-20", "decision": "维持2.25%—2.50%，延续耐心并公布缩表调整计划", "path": "PATIENT"},
]
INPUTS = {
    "market": "reports/research/510300_macro_dynamic_reframe_v1/inputs/market.parquet",
    "price": "reports/research/510300_macro_earnings_pricing_bridge_v5/inputs/index_price.parquet",
    "pe": "reports/research/510300_macro_earnings_pricing_bridge_v5/inputs/pe.parquet",
    "dividends": "data/reference/510300_dividends.csv",
    "nominal2018": "reports/research/510300_rmb_residual_state_v1/sources/treasury_2018.xml",
    "nominal2019": "reports/research/510300_rmb_residual_state_v1/sources/treasury_2019.xml",
    "dkw": "reports/research/510300_real_yield_company_exposure_v19/sources/DKW_canonical.csv",
}
DKW_FIELDS = ["exp.real.short.rate.10", "exp.inflation.10", "real.term.prem.10", "inflation.risk.prem.10",
              "tips.liq.prem.10", "nominal.yield.raw.10", "nominal.yield.fitted.10", "ic.raw.10", "ic.fitted.10"]


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def save(name, value):
    (OUT / name).parent.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def source_plan():
    rows = []
    for event in EVENTS:
        tag = event["id"].replace("-", "")
        rows.append({"id": "fed_" + tag, "name": "fed_" + tag + ".html", "url": f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{tag}a.htm",
                     "reuse": str(OLD / "sources/fed_jan30.html") if tag == "20190130" else None})
    for tag in ["20190130", "20190320"]:
        rows.append({"id": "balance_" + tag, "name": "balance_" + tag + ".html", "url": f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{tag}c.htm",
                     "reuse": str(OLD / "sources/fed_mar20_balance.html") if tag == "20190320" else None})
    rows.append({"id": "press_20190130", "name": "press_20190130.pdf", "url": "https://www.federalreserve.gov/mediacenter/files/FOMCpresconf20190130.pdf", "reuse": None})
    for year in [2018, 2019]:
        rows.append({"id": f"tips_{year}", "name": f"tips_{year}.xml", "url": f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_real_yield_curve&field_tdr_date_value={year}", "reuse": None})
    return rows


def prepare():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("本轮方案已保存，不覆盖。")
    old = json.loads((ROOT / "reports/research/510300_historical_index_credit_regime_compare_v1/protocol.json").read_text(encoding="utf-8"))
    save("protocol.json", {
        "study_id": "510300_HISTORICAL_INDEX_US_RATES_POLICY_V1", "recorded_at": now(),
        "question": "2018Q4与2019Q1同为美债收益率下降时，下降构成、政策回应和指数已发生定价有何差异？",
        "research_unit": "沪深300整体与510300，不扩展公司案例", "mode": "HISTORICAL_DISCOVERY_ONLY",
        "period": ["2018-10-01", "2019-03-31"], "baseline_month": "2018-09", "events": EVENTS,
        "selection": "固定两个日历季度内全部四次FOMC会议；阶段行情已在旧研究见过，不称盲测或独立验证。",
        "horizons": [5, 20], "costs": old["costs"], "input_paths": INPUTS, "source_plan": source_plan(),
        "event_clock": "官方声明美国东部14:00转换北京时间，随后第一个中国交易日09:30开盘买入；入场后5/20交易日开盘卖出。",
        "curve_clock": "沿V19采用美国观测日23:59:59转北京时间的保守假设；另保留延迟一条美国观测记录的状态对照。不是已验证的原始发布时间。",
        "curve_admission": "名义与TIPS必须同一美国观测日；声明同日美债日值只作事后反应，不当成声明后首个中国开盘前的已知输入。",
        "snapshot": "每个中国月末21:00取最后一条按假设已知的美债记录，展示实际美国日期；中国收盘价是当日事实，PE遵循原available_at，季度PE用同日数据作事后描述。",
        "dkw": "仅用已有2025-11-12重估版解释历史构成，绝不进入历史交易信息；DKW零息收益率与Treasury固定期限票息曲线分开，显式保留拟合残差。",
        "expectation_gap": "政策文字相对上次声明的变化不等于相对市场预期的意外；缺当时期货/调查则市场意外保持空值。",
        "upstream": "重点查全球增长、金融条件、通胀压力及资产负债表回应；原话可观察，跨市场传导是待区分解释。",
        "causal_limit": "四个事件窗口含中国信用政策及其他消息，事件后收益不等于美联储因果贡献；不交叠不等于独立。",
        "anti_overfit": ["不按结果新增会议", "不搜索利率阈值", "不改方向或持有期限", "不拼接旧失败策略", "保留全部收益及亏损"],
        "full_account_runs": 0, "goal_achieved": False, "orders_authorized": False,
        "previous_result": "reports/research/510300_historical_index_credit_regime_compare_v1/result.json",
    })
    print("已保存两个固定季度、四次会议和原有成本窗口。", flush=True)


def fetch_one(source):
    destination = OUT / "sources" / source["name"]
    if destination.exists():
        raise RuntimeError("来源已存在，不静默重复请求：" + source["id"])
    receipt = {**source, "retrieved_or_reused_at": now(), "historical_first_vintage_verified": False}
    try:
        if source["reuse"]:
            raw = Path(source["reuse"]).read_bytes()
            receipt.update(status="REUSED_LOCAL_PRIMARY_DOCUMENT", final_url=source["url"])
        else:
            response = requests.get(source["url"], timeout=(15, 50), headers={"User-Agent": "Mozilla/5.0"})
            receipt.update(http_status=response.status_code, final_url=response.url)
            response.raise_for_status()
            raw = response.content
            receipt["status"] = "SAVED_PRIMARY_DOCUMENT"
        destination.write_bytes(raw)
        receipt.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        if destination.suffix == ".html":
            body = BeautifulSoup(raw, "html.parser").get_text("\n", strip=True)
            destination.with_suffix(".txt").write_text(body, encoding="utf-8")
        elif destination.suffix == ".pdf":
            import pdfplumber
            with pdfplumber.open(destination) as document:
                body = "\n".join(f"第{i + 1}页\n{page.extract_text() or ''}" for i, page in enumerate(document.pages))
            destination.with_suffix(".txt").write_text(body, encoding="utf-8")
    except Exception as error:
        receipt.update(status="ERROR", error=str(error))
    save("sources/" + source["name"] + ".receipt.json", receipt)
    return receipt


def fetch():
    sources = read("protocol.json")["source_plan"]
    (OUT / "sources").mkdir(exist_ok=True)
    receipts = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(fetch_one, source) for source in sources]):
            receipt = future.result()
            receipts.append(receipt)
            print(receipt["id"], receipt["status"], flush=True)
    save("source_manifest.json", sorted(receipts, key=lambda row: row["id"]))
    if any(row["status"] == "ERROR" for row in receipts):
        raise RuntimeError("部分来源未取得，查看保存的具体原因后再处理。")


def curve(path, real=False):
    rows = []
    for node in ET.fromstring(path.read_bytes()).iter():
        if node.tag.split("}")[-1] != "properties":
            continue
        values = {child.tag.split("}")[-1]: child.text for child in node}
        field = "TC_10YEAR" if real else "BC_10YEAR"
        if values.get("NEW_DATE") and values.get(field):
            row = {"date": pd.Timestamp(values["NEW_DATE"][:10]), "tips10" if real else "nominal10": float(values[field])}
            if not real:
                row["nominal2"] = float(values["BC_2YEAR"]) if values.get("BC_2YEAR") else np.nan
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values("date")
    if frame.empty or frame.date.duplicated().any():
        raise ValueError("曲线日期为空或重复：" + str(path))
    return frame


def resolve_pdf():
    import pdfplumber
    path = OUT / "sources/press_20190130.pdf"
    original = read("sources/press_20190130.pdf.receipt.json")
    assert original["http_status"] == 200
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original["sha256"]
    with pdfplumber.open(path) as document:
        body = "\n".join(f"第{i + 1}页\n{page.extract_text() or ''}" for i, page in enumerate(document.pages))
    path.with_suffix(".txt").write_text(body, encoding="utf-8")
    save("source_resolution.json", {"recorded_at": now(), "resolved_ids": ["press_20190130"],
                                    "reason": "PDF已成功下载；改用环境已有pdfplumber抽取文字，原失败回执保留。", "new_requests": 0})
    print("已从保存的PDF提取原文，没有重发请求。", flush=True)


def asof_rate(rates, cutoff, delay=False):
    field = "known_at_delay1_assumed" if delay else "known_at_assumed"
    result = rates.loc[rates[field].le(cutoff)].iloc[-1]
    return {"us_observation_date": result.date, "known_at_assumed": result[field],
            **{key: result[key] for key in ["nominal10", "tips10", "inflation_compensation", "nominal2",
                                             "nominal10_20obs_bp", "tips10_20obs_bp", "inflation_compensation_20obs_bp"]}}


def calculate():
    if (OUT / "events_and_returns.json").exists():
        raise RuntimeError("已有结果，不为改善收益重复计算。")
    protocol = read("protocol.json")
    resolved = read("source_resolution.json")["resolved_ids"] if (OUT / "source_resolution.json").exists() else []
    if any(row["status"] == "ERROR" and row["id"] not in resolved for row in read("source_manifest.json")):
        raise RuntimeError("来源仍有未处理错误。")
    nominal = pd.concat([curve(ROOT / INPUTS[f"nominal{year}"]) for year in [2018, 2019]])
    tips = pd.concat([curve(OUT / f"sources/tips_{year}.xml", real=True) for year in [2018, 2019]])
    rates = nominal.merge(tips, on="date", how="inner", validate="one_to_one").sort_values("date").reset_index(drop=True)
    rates["inflation_compensation"] = rates.nominal10 - rates.tips10
    rates["known_at_assumed"] = (rates.date + pd.Timedelta(hours=23, minutes=59, seconds=59)).dt.tz_localize("America/New_York").dt.tz_convert("Asia/Shanghai")
    rates["known_at_delay1_assumed"] = rates.known_at_assumed.shift(-1)
    for field in ["nominal10", "tips10", "inflation_compensation"]:
        rates[field + "_20obs_bp"] = rates[field].diff(20) * 100
    rates.loc[rates.date.between("2018-08-01", "2019-04-30")].to_parquet(OUT / "rates.parquet", index=False)
    market = pd.read_parquet(ROOT / INPUTS["market"]).sort_values("date").reset_index(drop=True)
    market["date"] = pd.to_datetime(market.date).dt.tz_localize(None).dt.normalize()
    calendar = pd.DatetimeIndex(market.date)
    opens = calendar.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=30)
    dividends = pd.read_csv(ROOT / INPUTS["dividends"])
    dividends = dividends.loc[dividends.symbol.eq("510300.SH")].copy()
    dividends["record_date"] = pd.to_datetime(dividends.record_date)
    price = pd.read_parquet(ROOT / INPUTS["price"]).sort_values("date").set_index("date")
    pe = pd.read_parquet(ROOT / INPUTS["pe"])
    pe["date"] = pd.to_datetime(pe.observation_date)
    pe["available_at"] = pd.to_datetime(pe.available_at, utc=True).dt.tz_convert("Asia/Shanghai")
    panel = price.reset_index()[["date", "close"]].merge(pe[["date", "available_at", "original_pe_official"]], on="date", validate="one_to_one").sort_values("date").set_index("date")
    states, returns, monthends = [], [], []
    for event in protocol["events"]:
        published = (pd.Timestamp(event["id"]) + pd.Timedelta(hours=14)).tz_localize("America/New_York").tz_convert("Asia/Shanghai")
        entry_pos = int(opens.searchsorted(published, side="right"))
        entry = market.iloc[entry_pos]
        cutoff = opens[entry_pos]
        latest_price = price.loc[price.index < entry.date].iloc[-1]
        price_pos = price.index.get_loc(latest_price.name)
        latest_pe = panel.loc[panel.available_at.le(cutoff)].iloc[-1]
        lookbacks = [{"horizon": h, "start_date": price.index[price_pos - h], "end_date": latest_price.name,
                      "price_return": float(latest_price.close / price.iloc[price_pos - h].close - 1)} for h in [20, 60]]
        us_day = rates.loc[rates.date.eq(pd.Timestamp(event["id"]))].iloc[0]
        prev_us = rates.loc[rates.date.lt(us_day.date)].iloc[-1]
        state = {**event, "published_cn": published, "entry_at": cutoff, "entry_date": entry.date.strftime("%Y-%m-%d"),
                 "index_last_date": latest_price.name, "index_last_close": latest_price.close,
                 "pe_date": latest_pe.name, "pe_available_at": latest_pe.available_at, "pe": latest_pe.original_pe_official,
                 "lookbacks": lookbacks, "known_curve": asof_rate(rates, cutoff), "delay1_known_curve": asof_rate(rates, cutoff, True),
                 "market_expectation_surprise": None,
                 "event_us_day_curve_not_entry_input": {"us_date": us_day.date, "known_at_assumed": us_day.known_at_assumed,
                                                        **{key + "_day_change_bp": float(100 * (us_day[key] - prev_us[key])) for key in ["nominal10", "tips10", "inflation_compensation"]}}}
        assert published < cutoff < us_day.known_at_assumed
        assert state["known_curve"]["us_observation_date"] < us_day.date
        assert latest_pe.available_at <= cutoff
        states.append(state)
        for horizon in protocol["horizons"]:
            end = market.iloc[entry_pos + horizon]
            div = float(dividends.loc[dividends.record_date.ge(entry.date) & dividends.record_date.lt(end.date), "cash_dividend_per_share"].sum())
            trade = round_trip(float(entry.open), float(end.open), div, protocol["costs"])
            assert trade["paid_cny"] <= protocol["costs"]["illustrative_event_budget_cny"]
            assert end.date > entry.date
            error = (trade["sell_price"] - trade["buy_price"]) * trade["shares"] + trade["dividend_entitlement_cny"] - trade["commissions_cny"] - trade["net_pnl_cny"]
            assert abs(error) < 1e-7
            returns.append({"event_id": event["id"], "horizon": horizon, "entry_date": entry.date.strftime("%Y-%m-%d"),
                            "exit_date": end.date.strftime("%Y-%m-%d"), "entry_open": float(entry.open), "exit_open": float(end.open),
                            "dividend_per_share": div, "gross_return": float((end.open + div) / entry.open - 1), **trade})
    for month in pd.period_range("2018-09", "2019-03", freq="M"):
        date = calendar[calendar.to_period("M") == month][-1]
        cutoff = date.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=21)
        row = panel.loc[date]
        usable_pe = panel.loc[panel.available_at.le(cutoff)].iloc[-1]
        monthends.append({"cn_date": date, "cutoff": cutoff, "index_close": row.close,
                          "same_day_pe_retrospective": row.original_pe_official, "same_day_pe_available_at": row.available_at,
                          "known_pe_date": usable_pe.name, "known_pe": usable_pe.original_pe_official,
                          "rates": asof_rate(rates, cutoff), "rates_delay1": asof_rate(rates, cutoff, True)})
    raw = (ROOT / INPUTS["dkw"]).read_text(encoding="utf-8")
    lines = raw.splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith('"date",'))
    dkw = pd.read_csv(StringIO(raw), skiprows=header, parse_dates=["date"]).set_index("date")
    quarters, model = [], []
    for label, start, end in [("2018Q4", monthends[0], monthends[3]), ("2019Q1", monthends[3], monthends[6])]:
        row = {"period": label, "cn_start": start["cn_date"], "cn_end": end["cn_date"],
               "us_start": start["rates"]["us_observation_date"], "us_end": end["rates"]["us_observation_date"],
               "index_price_return": end["index_close"] / start["index_close"] - 1,
               "same_day_pe_change_retrospective": end["same_day_pe_retrospective"] / start["same_day_pe_retrospective"] - 1}
        for field in ["nominal10", "tips10", "inflation_compensation", "nominal2"]:
            row[field + "_bp"] = 100 * (end["rates"][field] - start["rates"][field])
            row[field + "_delay1_bp"] = 100 * (end["rates_delay1"][field] - start["rates_delay1"][field])
        assert abs(row["nominal10_bp"] - row["tips10_bp"] - row["inflation_compensation_bp"]) < 1e-10
        quarters.append(row)
        dkw_start, dkw_end = dkw.loc[row["us_start"]], dkw.loc[row["us_end"]]
        changes = {key: float((dkw_end[key] - dkw_start[key]) * 100) for key in DKW_FIELDS}
        residual = changes["nominal.yield.raw.10"] - changes["nominal.yield.fitted.10"]
        ic_residual = changes["ic.raw.10"] - changes["ic.fitted.10"]
        component_sum = sum(changes[key] for key in DKW_FIELDS[:4])
        fitted_ic = changes["exp.inflation.10"] + changes["inflation.risk.prem.10"] - changes["tips.liq.prem.10"]
        assert abs(component_sum - changes["nominal.yield.fitted.10"]) < 1e-3
        assert abs(fitted_ic - changes["ic.fitted.10"]) < 1e-3
        model.append({"period": label, "us_start": row["us_start"], "us_end": row["us_end"], "changes_bp": changes,
                      "nominal_residual_change_bp": residual, "ic_residual_change_bp": ic_residual,
                      "nominal_cmt_minus_dkw_raw_change_bp": row["nominal10_bp"] - changes["nominal.yield.raw.10"],
                      "historical_entry_input_admitted": False, "model_refit_through": "2025-11-12"})
    for filename, rows in [("event_states.json", states), ("events_and_returns.json", returns), ("monthends.json", monthends),
                           ("quarter_comparison.json", quarters), ("dkw_retrospective_decomposition.json", model)]:
        save(filename, rows)
    save("calculation_checks.json", {"recorded_at": now(), "events": len(states), "windows": len(returns),
                                    "accounting_identity": "PASS", "clock_order": "PASS_ASSUMED_DAILY_CLOCK",
                                    "curve_identity": "PASS", "dkw_identity": "PASS_MODEL_ONLY",
                                    "overlap": [assign_clusters(returns, h) for h in protocol["horizons"]],
                                    "full_account_runs": 0, "goal_achieved": False})
    save("input_receipts.json", [{"id": key, "path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()} for key, path in INPUTS.items()])
    print(pd.DataFrame(quarters)[["period", "nominal10_bp", "tips10_bp", "inflation_compensation_bp", "index_price_return"]].to_string(index=False), flush=True)
    print(pd.DataFrame(returns)[["event_id", "horizon", "entry_date", "exit_date", "net_return"]].to_string(index=False), flush=True)
    print("仅完成构成与事件收益，没有拟合或完整账户绩效。", flush=True)


def evidence():
    specifications = [
        ("fed_20181108", ["further gradual increases", "2 to 2-1/4 percent"], "2018-11-08", "继续逐步加息的指引未变；当次维持区间。", "当时声明"),
        ("fed_20181219", ["some further gradual increases", "raise the target range"], "2018-12-19", "当次仍加息25基点，并保留部分进一步加息的指引，同时强调跟踪全球经济与金融发展。", "当时声明"),
        ("fed_20190130", ["will be patient", "muted inflation pressures", "survey-based measures"], "2019-01-30", "因全球经济、金融条件和温和通胀压力转向耐心；同时区分市场通胀补偿下降与调查通胀预期大致稳定。", "当时声明"),
        ("fed_20190320", ["growth of economic activity has slowed", "lower energy prices", "will be patient"], "2019-03-20", "增长、居民消费和固定投资放缓；能源价格压低总体通胀；仍维持利率并延续耐心。", "当时声明"),
        ("balance_20190130", ["ample supply of reserves", "adjust any of the details"], "2019-01-30", "保留充裕准备金操作框架，并允许根据经济金融变化调整缩表细节；不是当日开启QE。", "当时政策文件"),
        ("balance_20190320", ["$30 billion to $15 billion", "end of September 2019"], "2019-03-20", "当时公布的计划是5月起国债每月缩减上限从300亿降至150亿美元、9月底结束总持仓缩减；这是计划，不倒填后来实际执行。", "当时政策文件"),
        ("press_20190130", ["particularly in China and Europe", "Financial conditions tightened considerably", "recent drop in oil prices"], "2019-01-30", "鲍威尔解释了上游原因：中国与欧洲增长放缓，贸易、脱欧和政府停摆增加不确定性；此前金融条件收紧，油价下降减弱通胀压力。美国基准增长判断并未彻底转坏。", "公开新闻会见第1—3页；最终转录文件并非已核实的当日下载版本"),
    ]
    plan = {row["id"]: row for row in source_plan()}
    findings = []
    for source_id, needles, date, finding, location in specifications:
        body = (OUT / "sources" / (source_id + ".txt")).read_text(encoding="utf-8")
        compact = " ".join(body.split())
        for needle in needles:
            assert needle in compact, (source_id, needle)
        findings.append({"source_id": source_id, "date": date, "url": plan[source_id]["url"], "finding": finding, "location": location,
                         "role": "CONTEMPORANEOUS_PUBLIC_EXPLANATION_NOT_IDENTIFIED_MARKET_SURPRISE"})
    previous = json.loads((ROOT / "reports/research/510300_historical_index_credit_regime_compare_v1/monthends.json").read_text(encoding="utf-8"))
    current = json.loads((ROOT / "reports/research/510300_historical_index_rates_policy_transmission_v1/monthend_states.json").read_text(encoding="utf-8"))["rows"]
    old_map = {row["date"][:10]: row for row in previous}
    new_map = {row["date"][:10]: row for row in current}
    domestic = []
    for q in read("quarter_comparison.json"):
        first, last = q["cn_start"][:10], q["cn_end"][:10]
        a = old_map[first]
        b = old_map[last] if last in old_map else new_map[last]
        domestic.append({"period": q["period"], "start": first, "end": last,
                         "cgb10_change_bp": (b["cgb_10y"] - a["cgb_10y"]) * 100,
                         "fdr20_change_bp": (b.get("fdr20", b.get("fdr007_20obs_mean")) - a["fdr20"]) * 100,
                         "role": "固定端点的既有国内背景，事后描述，不作为美联储因果贡献或交易输入"})
    save("mechanism_evidence.json", {"recorded_at": now(), "findings": findings, "domestic_context": domestic,
        "measurement": {
            "identity": "名义10年收益率 = TIPS 10年收益率 + 两者之差（通胀补偿代理）",
            "dkw_nominal": "预期实际短端利率 + 预期通胀 + 实际期限溢价 + 通胀风险溢价",
            "dkw_tips": "预期实际短端利率 + 实际期限溢价 + TIPS流动性溢价",
            "dkw_ic": "预期通胀 + 通胀风险溢价 - TIPS流动性溢价",
            "interpretation": "TIPS流动性溢价在相加重建名义收益率时抵消，不是名义四项之外的第五项；本研究未识别独立违约补偿。",
            "method_url": "https://www.federalreserve.gov/econres/notes/feds-notes/tips-from-tips-update-and-discussions-20190521.html",
            "treasury_url": "https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics",
            "treasury_note": "名义及实际曲线来自约美东15:30指示报价；该报价时刻不等于历史网站实际可下载时刻。美国日终23:59是本研究沿用的保守假设。",
        },
        "expectation_missing": "没有核实四次会议前同口径市场路径预测，不能把声明变化认定为超预期。",
        "causal_scope": "汇率、外资流量、权益风险溢价和指数盈利的具体贡献未在本轮识别；将其列为传导渠道，而不声称已证明。"})
    print("已核对七份政策原文及已有国内背景；保留缺失的市场预期。", flush=True)


def draw():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"], "axes.unicode_minus": False, "font.size": 11})
    q = read("quarter_comparison.json")
    trades = read("events_and_returns.json")
    fig = plt.figure(figsize=(14, 9), facecolor="#fbfcfe")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], width_ratios=[1.65, 1], hspace=0.52, wspace=0.3)
    a, b, c = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])
    x = np.arange(2)
    for offset, key, label, color in [(-0.24, "nominal10_bp", "名义10年", "#235789"),
                                      (0, "tips10_bp", "TIPS 10年", "#278c81"),
                                      (0.24, "inflation_compensation_bp", "通胀补偿代理", "#bd8340")]:
        bars = a.bar(x + offset, [row[key] for row in q], 0.23, label=label, color=color)
        a.bar_label(bars, fmt="%+.0f", padding=4, fontsize=10)
    a.set_xticks(x, ["2018年第四季度", "2019年第一季度"])
    a.set_ylim(-60, 25)
    a.set_ylabel("基点变化")
    a.set_title("收益率都下降，构成却不同", loc="left", fontweight="bold")
    a.legend(loc="lower left", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0, -0.28))
    bars = b.bar(x, [row["index_price_return"] * 100 for row in q], width=0.5, color=["#235789", "#c75949"])
    b.bar_label(bars, fmt="%+.2f%%", padding=5)
    b.set_xticks(x, ["2018年第四季度", "2019年第一季度"])
    b.set_ylim(-20, 37)
    b.set_ylabel("沪深300价格涨跌（%）")
    b.set_title("指数整体结果", loc="left", fontweight="bold")
    event_x = np.arange(4)
    for offset, horizon, color in [(-0.18, 5, "#8397ab"), (0.18, 20, "#235789")]:
        values = [next(row["net_return"] * 100 for row in trades if row["event_id"] == event["id"] and row["horizon"] == horizon) for event in EVENTS]
        bars = c.bar(event_x + offset, values, width=0.34, color=color, label=f"{horizon}交易日")
        c.bar_label(bars, fmt="%+.2f%%", padding=4, fontsize=10)
    c.set_xticks(event_x, ["2018-11-08\n继续逐步加息指引", "2018-12-19\n当次加息25基点", "2019-01-30\n转向耐心判断", "2019-03-20\n延续耐心及缩表计划"])
    c.set_ylim(-6, 25)
    c.set_ylabel("510300事件资金净收益（%）")
    c.set_title("全部四次会议：下一中国交易日开盘买入后的固定窗口", loc="left", fontweight="bold")
    c.legend(loc="upper left", frameon=False)
    for ax in [a, b, c]:
        ax.axhline(0, color="#6b7785", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#dae1e9", alpha=0.6)
    fig.suptitle("指数历史研究：同为美债收益率下降，不能视为同一种交易条件", x=0.065, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.065, 0.055, "中国端点：2018-09-28、2018-12-28、2019-03-29；美国曲线：各端点按保守时钟已知的前一日数据。\n10万元事件预算，双边滑点各0.1%、佣金各0.04%及整手约束；收益不代表美联储因果贡献或完整账户夏普。", fontsize=10, color="#475569")
    fig.subplots_adjust(left=0.075, right=0.98, top=0.88, bottom=0.18)
    fig.savefig(OUT / "美债构成与指数阶段对照.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def report():
    q = read("quarter_comparison.json")
    model = read("dkw_retrospective_decomposition.json")
    events = read("event_states.json")
    trades = read("events_and_returns.json")
    proof = read("mechanism_evidence.json")
    protocol = read("protocol.json")
    first, second = q
    lines = ["# 历史发现：美债下降的构成、政策回应与沪深300", "",
             "本轮发现了可直接核对的阶段差异：2018年第四季度，名义10年美债下降29基点，TIPS收益率却上升12基点，通胀补偿代理压缩41基点；2019年第一季度，名义下降38基点，TIPS下降47基点，通胀补偿回升9基点。沪深300同期分别下跌12.45%、上涨28.62%。", "",
             "这支持先区分利率变化的构成，再查政策为什么调整；不足以把任一种构成直接晋升为买入信号。本轮继续以指数整体为研究单位，没有新增公司案例。", "",
             "| 固定日历阶段 | 名义10年变化 | TIPS 10年变化 | 通胀补偿代理变化 | 沪深300价格 | 同日PE事后变化 |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in q:
        lines.append(f"| {row['period']} | {row['nominal10_bp']:+.0f}bp | {row['tips10_bp']:+.0f}bp | {row['inflation_compensation_bp']:+.0f}bp | {row['index_price_return']:+.2%} | {row['same_day_pe_change_retrospective']:+.2%} |")
    lines += ["", "中国价格端点为2018-09-28、2018-12-28、2019-03-29。美债按各日北京时间21:00、沿既有美国日终可得假设，分别取2018-09-27、2018-12-27、2019-03-28；不是美国自然季度最后一天的端点。进一步延迟一条美国观测，两个阶段的名义/TIPS/通胀补偿变化分别为−25/+12/−37bp与−42/−49/+7bp，主要构成差异未消失。季度端点是描述，不能当作季度初已知的状态。", "",
              "PE栏使用对应中国收盘日原口径，作事后背景描述，其真实可得日另存在monthends.json；没有把当日尚未可得PE用于入场。PE变化不等于盈利变化，也不等于已识别的权益风险溢价。", "",
              "**TIPS也需要继续拆，不能停留在名义减实际。**", "",
              "通胀补偿只是名义与TIPS曲线之差，并非纯通胀预期。已有DKW模型将名义收益率分为预期实际短端利率、预期通胀、实际期限溢价和通胀风险溢价；TIPS还受到相对流动性溢价影响。该溢价在TIPS与通胀补偿相加时抵消，不能再作为名义四项之外的第五项。[美联储模型说明](https://www.federalreserve.gov/econres/notes/feds-notes/tips-from-tips-update-and-discussions-20190521.html)。", "",
              "下表是已保存的**2025-11-12重估版对历史的解释**，没有当时版本，全部禁止进入历史交易输入。模型零息曲线与上表Treasury曲线不同，必须保留拟合与口径残差。", "",
              "| 模型分项变化，单位bp | 2018Q4 | 2019Q1 |", "|---|---:|---:|"]
    labels = {"exp.real.short.rate.10": "预期实际短端利率", "exp.inflation.10": "预期通胀", "real.term.prem.10": "实际期限溢价", "inflation.risk.prem.10": "通胀风险溢价", "tips.liq.prem.10": "TIPS流动性溢价（另列，不再加进名义四项）", "nominal.yield.raw.10": "模型输入名义零息收益率", "nominal.yield.fitted.10": "四项合计的拟合名义收益率", "ic.raw.10": "模型输入通胀补偿", "ic.fitted.10": "拟合通胀补偿"}
    for key, label in labels.items():
        lines.append(f"| {label} | {model[0]['changes_bp'][key]:+.2f} | {model[1]['changes_bp'][key]:+.2f} |")
    for key, label in [("nominal_residual_change_bp", "名义收益率拟合残差变化"), ("ic_residual_change_bp", "通胀补偿拟合残差变化"), ("nominal_cmt_minus_dkw_raw_change_bp", "Treasury名义变化减模型零息输入变化")]:
        lines.append(f"| {label} | {model[0][key]:+.2f} | {model[1][key]:+.2f} |")
    lines += ["", "在这个模型版本里，2018Q4流动性溢价上升27.60bp，对通胀补偿形成负向影响；2019Q1回落23.95bp，对通胀补偿形成正向影响。两阶段模型的预期通胀分项都下降。因此，不能把前一阶段的−41bp全叫通胀预期恶化，也不能把后一阶段的+9bp全叫增长预期恢复。模型分配不是唯一因果解释，但足以指出简单命名的错误。", "",
              "**政策为什么改变，有当时的公开解释。**", ""]
    for row in proof["findings"]:
        lines.append(f"- {row['date']}：{row['finding']} [原文]({row['url']})。")
    lines += ["", "1月新闻会见是公开讲话，最终转录文件的首次上传时间未单独核实；本轮只引用其历史解释，不生成量化输入。没有拿数周后会议纪要或五年后内部材料当成当时已知的信息。", "",
              "从机制上看，增长担忧会同时影响盈利预期和利率；政策回应又可能减轻继续收紧的担忧。它们并存，名义收益率一个数字无法判断哪种力量对沪深300更重要。人民币汇率、跨境配置、国内融资和估值是可能的传导渠道，本轮没有识别其独立因果贡献。", "",
              "国内背景也不能省略。复用已有同一端点，中国10年国债在2018Q4下降38.39bp，在2019Q1下降16.32bp；资金利率FDR007近20次均值分别下降1.85bp、3.55bp。国内外债券收益率同降仍对应相反的指数季度结果，所以“利率降就是指数买点”不成立。国内信用构成与公布前预期差沿用前几轮账本，未因为新美债解释而重写旧亏损。", "",
              "**公开之后，还剩多少可成交收益。**", "",
              "两个季度全部四次会议均保留。北京时间首三次声明为次日03:00，3月为次日02:00；统一在随后第一个中国交易日09:30开盘买入，入场后第5或20交易日开盘卖出。10万元事件预算，双边滑点各0.1%，佣金各0.04%且单边最低5元，100份整手、0.001元价位；按实际买入支付金额计算净收益，分红按股权登记资格计入。不是20万元完整账户收益。", "",
              "| 美联储声明日（美国） | 中国入场日 | 5日净收益 | 20日净收益 | 20日退出 | 入场前沪深300近20日 | 入场前近60日 |", "|---|---|---:|---:|---|---:|---:|"]
    for event in events:
        trade5 = next(row for row in trades if row["event_id"] == event["id"] and row["horizon"] == 5)
        trade20 = next(row for row in trades if row["event_id"] == event["id"] and row["horizon"] == 20)
        lines.append(f"| {event['id']} | {event['entry_date']} | {trade5['net_return']:+.2%} | {trade20['net_return']:+.2%} | {trade20['exit_date']} | {event['lookbacks'][0]['price_return']:+.2%} | {event['lookbacks'][1]['price_return']:+.2%} |")
    lines += ["", "1月转向耐心后的20日窗口收益19.92%，是真实存在的历史剩余收益；但窗口也包含2月中国信用数据及其他信息，不能归功于美联储单一因素。3月延续耐心且公布缩表调整计划后，5日仍亏3.21%、20日转为盈利5.87%，不能省略这个路径差异。两个耐心阶段都赚钱的20日结果，尚不能估计可复制胜率。", "",
              "12月20日入场的20日窗口包含每份0.059元分红权益，不能只用未复权开盘价比值重算。1月31日和3月21日入场前20日，指数已经分别上涨6.70%和11.11%，这些涨幅均没有计入事件净收益。", "",
              "日线利率可得时间沿既有保守规则：美国观测日23:59:59转北京时间；实际报价约为美东15:30，但报价时刻不等于网页发布时间。声明当日美债日值在此规则下晚于中国次日开盘，已明确排除。延迟一条观测只比较状态，没有改变四个预定入场点。[财政部口径](https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics)。", "",
              "本轮没有取得四次会前同口径的市场预期路径。政策措辞比上次更温和，并不自动表示超出当时市场预期；market_expectation_surprise保持空值。四个窗口不重叠，但共享宏观环境，不视为四次独立经济实验。", "",
              "**研究取舍。**保留名义/TIPS/通胀补偿三项作为历史状态描述，并用当时政策原因解释；停止把美债名义下降或TIPS下降单独当成指数买入依据。2025重估的流动性溢价只用于机制解释，不作为补救交易阈值。没有优化方向、参数或持有期，也没有运行完整账户。", "",
              "下一步固定2019Q2作相邻阶段对照，保留两次FOMC会议和全部日历月，检查利率继续变化、政策回应与贸易冲击如何并存；先核对最接近的既有研究，不按盈利窗口扩样，也不使用季度标签开关仓位。仍以历史指数整体研究为主。", "",
              "图：美债构成与指数阶段对照.png。数据：quarter_comparison.json、dkw_retrospective_decomposition.json、event_states.json、events_and_returns.json、mechanism_evidence.json。代码：research/historical_index_us_rates_policy_v1.py。", "",
              "本轮必要核对已通过：同日曲线恒等式、模型分项与残差、声明/开盘先后、成交成本与分红会计。夏普未计算，完整账户夏普1.2和年化10%的共同目标尚未达到。", ""]
    (OUT / REPORT).write_text("\n".join(lines), encoding="utf-8")
    draw()
    save("result.json", {"study_id": protocol["study_id"], "completed_at": now(),
        "classification": "PROGRESS_INDEX_RATE_COMPOSITION_AND_POLICY_CAUSES", "events": 4, "return_rows": 8,
        "quarters": q, "event_expectation_surprises_missing": 4,
        "discovery": "2018Q4名义/TIPS/通胀补偿变化为-29/+12/-41bp，2019Q1为-38/-47/+9bp，指数分别-12.45%/+28.62%；后验模型提示TIPS流动性溢价不能忽略。四次会议窗口已计算成本，不能据此识别政策意外或完整账户优势。",
        "research_action": "RETAIN_RATE_COMPOSITION_FOR_INDEX_CONTEXT_NO_STANDALONE_ENTRY_RULE",
        "next_historical_question": "固定2019Q2相邻阶段全部两次FOMC及日历月，先查最近既有研究，再核对利率构成、政策回应与贸易冲击的指数层面差异；不用季度标签交易，不挑盈利事件扩样。",
        "source_documents": 9, "new_downloaded_documents": 7, "reused_documents": 2,
        "new_parameters_fitted": 0, "new_full_accounts": 0, "net_sharpe": None, "goal_achieved": False,
        "orders_authorized": False, "independent_validation": False, "report": REPORT})
    print("已生成阶段报告与图，完整账户目标仍未达到。", flush=True)


def record_progress():
    result = read("result.json")
    assert (OUT / result["report"]).is_file()
    prefix = OUT.relative_to(ROOT).as_posix() + "/"
    stamp, changes = now(), []
    for name in ["510300_historical_cause_discovery_v1.json", "510300_existing_data_training_mandate_v1.json"]:
        path = ROOT / "config" / name
        cfg = json.loads(path.read_text(encoding="utf-8"))
        before = dict(cfg)
        if name == "510300_historical_cause_discovery_v1.json":
            cfg.update(current_study=prefix + "protocol.json", latest_completed_study=prefix + "result.json", latest_report=prefix + REPORT, updated_at=stamp)
        else:
            cfg.update(current_round=result["study_id"], latest_progress_receipt=prefix + "result.json",
                       latest_historical_index_us_rates_policy=prefix + "result.json", latest_continuation_report=prefix + REPORT,
                       latest_historical_report=prefix + REPORT, latest_continuation_classification=result["classification"],
                       current_driver_continuation_classification=result["classification"], current_driver_consecutive_blocked_goal_turns=0,
                       latest_historical_diagnostic_at=stamp, latest_goal_service_status="active", latest_goal_service_status_observed_at=stamp,
                       goal_status="active", goal_achieved=False, local_goal_work_status="ACTIVE_HISTORICAL_ONLY",
                       last_research_result=result["discovery"], last_source_result="新增7份、复用2份官方历史来源；连接同日名义/TIPS曲线及政策原因。",
                       next_research_question=result["next_historical_question"])
        changes.append({"path": path.relative_to(ROOT).as_posix(), "fields": {key: {"before": before.get(key), "after": value} for key, value in cfg.items() if before.get(key) != value}})
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save("authority_update.json", {"recorded_at": stamp, "previous_goal_turn_classification": "PROGRESS",
                                  "current_goal_turn_classification": result["classification"], "changes": changes,
                                  "goal_achieved": False, "orders_authorized": False})
    print("已登记指数层面的利率构成发现；目标继续保持进行中。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="固定历史阶段的美债构成与指数研究")
    parser.add_argument("mode", choices=["prepare", "fetch", "resolve-pdf", "calculate", "evidence", "report", "record-progress"])
    args = parser.parse_args()
    {"prepare": prepare, "fetch": fetch, "resolve-pdf": resolve_pdf, "calculate": calculate,
     "evidence": evidence, "report": report, "record-progress": record_progress}[args.mode]()


if __name__ == "__main__":
    main()
