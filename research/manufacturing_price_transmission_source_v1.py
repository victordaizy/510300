"""从当期PMI原月报读取购进与出厂价格扩散指数，不用后报历史行回填。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
import re
import sys
from threading import Lock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.industrial_receivable_monthly_source_v1 as transport
from research.exchange_bank_funding_gap_daily_v1 import adapt
from research.selected_mix_reappraisal_v1 import read, save, now, digest

OUT = ROOT / "reports/research/510300_manufacturing_price_transmission_source_v1"
STUDY = "510300_MANUFACTURING_PRICE_TRANSMISSION_SOURCE_V1"
INHERITED = ROOT / "data/raw/macro/510300_macro_stress_2015_v2/pmi_new_orders_release_vintage_2015_2026.parquet"
PANEL = OUT / "released_manufacturing_price_transmission.parquet"
EXPECTED = [str(p) for p in pd.period_range("2017-01", "2026-08", freq="M")]
SOURCES = {
    "month_2026-08": "https://www.stats.gov.cn/sj/zxfb/202608/t20260831_1965154.html",
    "official_method": "https://www.stats.gov.cn/zs/tjws/zytjzbqs/cgzlzs/202501/t20250121_1958396.html",
}
FETCH = adapt(transport.fetch, {"OUT": OUT, "_lock": Lock(), "_request_count": len(list((OUT / "requests").glob("*.json")))})
REQUEST = adapt(transport.request_job, {"fetch": FETCH})


def compact(value):
    return re.sub(r"\s+", "", value)


def parse(item, receipt):
    soup = transport.soup_for(receipt)
    text, month = soup.get_text(" ", strip=True), item["stat_month"]
    flat = compact(text)
    year, number = map(int, month.split("-"))
    titles = sorted({compact(t.get_text()) for t in soup.find_all(["title", "h1", "h2"])
                     if f"{year}年{number}月" in compact(t.get_text()) and "采购经理指数" in compact(t.get_text())})
    assert titles, "报告标题与目标统计月不一致。"
    clocks = set(re.findall(r"20\d{2}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2}(?::\d{2})?", text))
    meta = soup.find("meta", attrs={"name": "PubDate"})
    if meta is not None:
        clocks.add(meta.get("content", ""))
    clocks = {pd.Timestamp(t).tz_localize("Asia/Shanghai") for t in clocks}
    dates = re.findall(r"成文日期(20\d{2})年(\d{2})月(\d{2})日", flat)
    if clocks:
        assert len(clocks) == 1, "原页公布时钟不唯一。"
        clock, precision = next(iter(clocks)), "MINUTE_OR_SECOND"
    else:
        assert len(set(dates)) == 1, "无可确认原公布日。"
        clock, precision = pd.Timestamp("-".join(dates[0])).tz_localize("Asia/Shanghai"), "DAY_ONLY"
    if dates:
        assert {"-".join(d) for d in dates} == {clock.strftime("%Y-%m-%d")}
    if item.get("inherited_published_at"):
        inherited = pd.Timestamp(item["inherited_published_at"])
        assert clock.normalize() == inherited.normalize(), "重新读取原页公布日与继承时钟不同。"
        if precision != "DAY_ONLY":
            assert clock == inherited, "原页日内时间与继承时钟不同。"
    assert str(clock.tz_localize(None).to_period("M")) in [month, str(pd.Period(month, freq="M") + 1)]
    target = f"{year}年{number}月"
    values, selected = [], []
    for table_number, table in enumerate(soup.find_all("table")):
        rows = [[compact(c.get_text()) for c in tr.find_all(["td", "th"], recursive=False)] for tr in table.find_all("tr")]
        headers = [r for r in rows if "出厂价格" in r and any("购进价格" in c for c in r)]
        if not headers:
            continue
        assert len(headers) == 1, "同一表存在多组不同字段头。"
        header = headers[0]
        purchase_positions = [i for i, cell in enumerate(header) if cell in ["主要原材料购进价格", "购进价格"]]
        output_positions = [i for i, cell in enumerate(header) if cell == "出厂价格"]
        assert len(purchase_positions) == len(output_positions) == 1
        current_rows = [r for r in rows if r and r[0] == target]
        assert len(current_rows) == 1, "本次公布月份行不唯一；不得用后报旧月份回填。"
        row = current_rows[0]
        assert len(row) == len(header), "本月行与表头长度不符。"
        purchase, output = row[purchase_positions[0]], row[output_positions[0]]
        assert re.fullmatch(r"\d{1,2}(?:\.\d)?", purchase) and re.fullmatch(r"\d{1,2}(?:\.\d)?", output)
        purchase, output = Decimal(purchase), Decimal(output)
        assert 0 < purchase < 100 and 0 < output < 100
        assert "经季节调整" in compact(table.get_text()) or any("经季节调整" in compact(p.get_text()) for p in table.find_all_previous("p", limit=8))
        values.append((float(purchase), float(output), float(output - purchase)))
        selected.append({"table_idx": table_number, "current_month_row": row, "header": header,
                         "input_column": purchase_positions[0], "output_column": output_positions[0]})
    assert values and len(set(values)) == 1, "当期购进与出厂价格值缺失，或重复表格冲突。"
    purchase, output, gap = values[0]
    samples = re.findall(r"(?<!非)制造业的(\d+)个行业大类[，,](\d+)家调查样本", flat)
    assert len(set(samples)) == 1, "制造业样本范围不唯一。"
    industries, firms = map(int, samples[0])
    standards = set(re.findall(r"GB/T4754[—－-](20\d{2})", flat))
    assert len(standards) == 1, "行业分类版本缺失或冲突。"
    assert "正向回答的企业个数百分比加上回答不变的百分比的一半" in flat, "扩散指数定义不匹配。"
    if month == "2017-01":
        assert "出厂价格指数于2017年1月起发布" in flat
    return {**item, "url": receipt["url"], "title": titles[0],
        "published_at": clock.isoformat() if precision != "DAY_ONLY" else clock.strftime("%Y-%m-%d"),
        "publication_precision": precision, "known_at": (clock.normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).isoformat(),
        "input_price_diffusion": purchase, "output_price_diffusion": output, "output_minus_input_diffusion_pp": gap,
        "manufacturing_industries": industries, "manufacturing_sample_firms": firms, "industry_classification_year": int(next(iter(standards))),
        "method_group": f"GB{next(iter(standards))}_{industries}IND_{firms}FIRMS",
        "current_month_row_only": True, "seasonally_adjusted_as_published": True,
        "matching_table_instances": len(values), "selected_tables": selected,
        "raw_path": receipt["raw_path"], "raw_sha256": receipt["sha256"], "historical_first_vintage_verified": False}


def run():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("购进出厂价格来源已冻结，不覆盖。")
    for folder in ["raw", "requests", "code"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    inherited = pd.read_parquet(INHERITED)
    assert len(inherited) == 139 and inherited.reference_period.is_unique
    used = inherited[inherited.reference_period.ge("2017-01")].copy()
    assert used.reference_period.tolist() == EXPECTED[:-1]
    files = {row.raw_path: row.source_hash for row in used.itertuples()}
    for path, sha in files.items():
        assert digest(ROOT / path) == sha, path
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY, "expected_months": EXPECTED,
        "inherited": INHERITED.relative_to(ROOT).as_posix(), "new_urls": SOURCES,
        "source_scope": "复用115份已保存当月官方PMI原文，只新增2026年8月报告和官方编制方法页；原2015—2016缺少出厂价格字段不填补。",
        "single_feature": "当次制造业出厂价格扩散指数减主要原材料购进价格扩散指数，单位为扩散指数百分点。",
        "mechanism": "两类涨价普遍程度的相对差异，可能反映企业承受采购成本与传导到销售端的环境；不是企业真实利润率、PPI涨跌幅、真实利率或市场预期差。",
        "first_release_limit": "只读该统计月当期报告对应行，不使用未来报告中可能调整的旧月份行。继承原始网页快照仍不能证明不可变首版。",
        "clock": "重新读取原页时钟并与继承时钟核对，统一公布日末可用；不将月末统计期与公布时点混同。",
        "methods": "逐份保留行业分类版本、制造业行业数、样本企业数与季调说明，不因同名字段宣称样本恒定。",
        "deduplication": "已核对新订单20日、PMI旧压力门、LPR联合响应与宏观意外旧家族；不修改其终态。research/config顶层针对购进/出厂价差的独立模型未发现命中，不等于穷尽全部归档。",
        "network": "2个官方URL，最多2并发；25秒超时、传输故障普通重试一次，403/429停止。",
        "new_accounts": 0, "new_strategy_returns": 0, "goal_achieved": False, "orders_authorized": False}, True)
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"), "code_sha256": digest(Path(__file__)),
        "inherited_panel_sha256": digest(INHERITED), "original_source_hashes": files, "transport_code_sha256": digest(Path(transport.__file__))}, True)
    (OUT / "code" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    records, failures = [], []
    for row in used.itertuples():
        item = {"stat_month": row.reference_period, "inherited_published_at": row.published_at,
                "original_snapshot_retrieved_at": row.retrieved_at}
        receipt = {"status": "HTTP_OK", "url": row.source_url, "raw_path": row.raw_path, "sha256": row.source_hash}
        try:
            records.append(parse(item, receipt))
        except (AssertionError, KeyError, ValueError) as exc:
            failures.append({"item": item, "receipt": receipt, "error": str(exc)})
        if (len(records) + len(failures)) % 25 == 0:
            print(f"当期购进/出厂价格已核对{len(records)}份，待核{len(failures)}份。", flush=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = [(url, key) for key, url in SOURCES.items()]
        receipts = dict(zip(SOURCES, executor.map(REQUEST, jobs)))
    latest = receipts["month_2026-08"]
    try:
        records.append(parse({"stat_month": "2026-08", "original_snapshot_retrieved_at": latest.get("received_at")}, latest))
    except (AssertionError, KeyError, ValueError) as exc:
        failures.append({"item": {"stat_month": "2026-08"}, "receipt": latest, "error": str(exc)})
    method = receipts["official_method"]
    method_text = compact(transport.soup_for(method).get_text()) if method["status"] == "HTTP_OK" else ""
    method_ok = all(t in method_text for t in ["扩散指数", "购进价格", "出厂价格"])
    save(OUT / "official_method_receipt.json", {**method, "definition_terms_present": method_ok}, True)
    records.sort(key=lambda r: r["stat_month"])
    frame = pd.DataFrame(records)
    if len(frame):
        frame["known_at"] = pd.to_datetime(frame.known_at).dt.tz_convert("Asia/Shanghai").dt.as_unit("ns")
        assert frame.known_at.is_monotonic_increasing and not frame.known_at.duplicated().any()
        frame.to_parquet(PANEL, index=False)
    save(OUT / "released_records.json", records, True)
    save(OUT / "unresolved_fields.json", failures, True)
    missing = sorted(set(EXPECTED) - {r["stat_month"] for r in records})
    transitions = []
    previous = None
    for r in records:
        if r["method_group"] != previous:
            transitions.append({"first_stat_month": r["stat_month"], "published_at": r["published_at"], "method_group": r["method_group"]})
            previous = r["method_group"]
    save(OUT / "result.json", {"at": now(), "study_id": STUDY,
        "status": "COMPLETE_MANUFACTURING_PRICE_TRANSMISSION_SOURCE_READY" if not missing and method_ok else "PARTIAL_MANUFACTURING_PRICE_TRANSMISSION_SOURCE",
        "expected_months": len(EXPECTED), "admitted_months": len(frame), "missing_months": missing,
        "unresolved_fields": len(failures), "method_transitions": transitions, "official_method_verified": method_ok,
        "latest_source": records[-1] if records else None, "new_accounts": 0, "new_strategy_returns": 0,
        "historical_first_vintage_verified": False, "current_market_view": "NO_VIEW", "goal_achieved": False, "orders_authorized": False}, True)
    print(f"购进与出厂价格来源结束：{len(records)}/{len(EXPECTED)}份。", flush=True)


if __name__ == "__main__":
    run()
