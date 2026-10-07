"""按原始月报重建70城二手住宅环比下跌范围，保留换基及无成交限制。"""
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

import research.housing_collateral_source_probe_v1 as probe
import research.industrial_receivable_monthly_source_v1 as transport
from research.exchange_bank_funding_gap_daily_v1 import adapt
from research.selected_mix_reappraisal_v1 import read, save, now, digest

OUT = ROOT / "reports/research/510300_housing_collateral_monthly_source_v1"
STUDY = "510300_HOUSING_COLLATERAL_MONTHLY_SOURCE_V1"
PANEL = OUT / "released_housing_breadth.parquet"
EXPECTED = [str(p) for p in pd.period_range("2021-01", "2026-08", freq="M")]
LEGACY = {
    "2021-01": probe.SOURCES["2021-01"],
    "2021-02": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1901022.html",
    "2021-03": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1901044.html",
    "2021-04": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1901096.html",
    "2021-05": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1901135.html",
    "2021-06": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1901152.html",
    "2021-07": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1901188.html",
    "2021-08": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1901215.html",
}
CITIES = set("北京 天津 石家庄 太原 呼和浩特 沈阳 大连 长春 哈尔滨 上海 南京 杭州 宁波 合肥 福州 厦门 南昌 济南 青岛 郑州 武汉 长沙 广州 深圳 南宁 海口 重庆 成都 贵阳 昆明 西安 兰州 西宁 银川 乌鲁木齐 唐山 秦皇岛 包头 丹东 锦州 吉林 牡丹江 无锡 徐州 扬州 温州 金华 蚌埠 安庆 泉州 九江 赣州 烟台 济宁 洛阳 平顶山 宜昌 襄阳 岳阳 常德 韶关 湛江 惠州 桂林 北海 三亚 泸州 南充 遵义 大理".split())
FETCH = adapt(transport.fetch, {"OUT": OUT, "_lock": Lock(), "_request_count": len(list((OUT / "requests").glob("*.json")))})
REQUEST = adapt(transport.request_job, {"fetch": FETCH})


def compact(value):
    return re.sub(r"\s+", "", value)


def title_month(title):
    found = re.match(r"^(20\d{2})年(\d{1,2})月份70个大中城市商品住宅销售价格变动情况", compact(title))
    return f"{found.group(1)}-{int(found.group(2)):02d}" if found else None


def parse(item, receipt):
    soup, month = transport.soup_for(receipt), item["stat_month"]
    text = soup.get_text(" ", strip=True)
    flat = compact(text)
    headings = [t.get_text(" ", strip=True) for t in soup.find_all(["title", "h1", "h2"])]
    titles = sorted({t for t in headings if title_month(t) == month})
    assert titles, "原文标题与统计月不符。"
    clocks = set(re.findall(r"20\d{2}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2}(?::\d{2})?", text))
    pub = soup.find("meta", attrs={"name": "PubDate"})
    if pub is not None:
        clocks.add(pub.get("content", ""))
    dates = re.findall(r"成文日期(20\d{2})年(\d{2})月(\d{2})日", flat)
    parsed = {pd.Timestamp(s).tz_localize("Asia/Shanghai") for s in clocks}
    if parsed:
        assert len(parsed) == 1, "原页公布时间不唯一。"
        clock, precision = next(iter(parsed)), "MINUTE_OR_SECOND"
    else:
        assert len(set(dates)) == 1, "原页无唯一公布日。"
        clock = pd.Timestamp("-".join(dates[0])).tz_localize("Asia/Shanghai")
        precision = "DAY_ONLY"
    if dates:
        assert {"-".join(d) for d in dates} == {clock.strftime("%Y-%m-%d")}
    if item.get("catalogue_date"):
        assert clock.strftime("%Y-%m-%d") == item["catalogue_date"], "原页与目录日期不同。"
        assert digest(ROOT / item["catalogue_raw_path"]) == item["catalogue_sha256"]
    assert clock.tz_localize(None).to_period("M").strftime("%Y-%m") == str(pd.Period(month, freq="M") + 1)
    candidates = []
    for idx, table in enumerate(soup.find_all("table")):
        rows = [[compact(c.get_text()) for c in row.find_all(["td", "th"], recursive=False)] for row in table.find_all("tr")]
        rows = [r for r in rows if r]
        if not rows or len(rows[0]) not in [6, 8] or rows[0][:3] != ["城市", "环比", "同比"]:
            continue
        captions = [compact(p.get_text()) for p in table.find_all_previous("p", limit=12)]
        captions = [c for c in captions if re.match(r"^表\d+[：:]", c)]
        assert captions, "未能定位表格标题。"
        caption = captions[0]
        if "二手住宅销售价格指数" not in caption or "分类" in caption:
            continue
        year, number = map(int, month.split("-"))
        assert f"{year}年{number}月70个大中城市二手住宅销售价格指数" in caption
        width = len(rows[0]) // 2
        assert rows[0][width:width + 3] == ["城市", "环比", "同比"]
        assert rows[1][0] == "上月=100" and rows[1][1] == "上年同月=100"
        values = {}
        for row in rows[2:]:
            assert len(row) == 2 * width, "城市行长度不符。"
            for offset in [0, width]:
                city, value = row[offset], row[offset + 1]
                assert city in CITIES and city not in values, "城市集合重复或发生变化。"
                assert re.fullmatch(r"\d{2,3}\.\d", value), "环比缺失或精度变化。"
                assert 80 < Decimal(value) < 120, "环比超出格式核对范围。"
                values[city] = value
        assert set(values) == CITIES and len(values) == 70
        candidates.append({"table_idx": idx, "caption": caption, "column_count": 2 * width, "city_values": values})
    assert candidates, "找不到二手住宅总指数表。"
    assert all(c["city_values"] == candidates[0]["city_values"] for c in candidates), "桌面与移动重复表格不一致。"
    values = candidates[0]["city_values"]
    falling = sum(Decimal(v) < 100 for v in values.values())
    flat_count = sum(Decimal(v) == 100 for v in values.values())
    rising = sum(Decimal(v) > 100 for v in values.values())
    assert falling + flat_count + rising == 70
    notes = flat[flat.index("附注"):].split("表1")[0]
    assert "重点调查" in notes and "典型调查" in notes and "不包括县" in notes
    assert "无成交" in notes and "无变动" in notes
    base_year = 2020 if month < "2026-01" else 2025
    if month == "2021-01":
        assert "2020年作为新一轮对比基期" in notes and "权数" in notes
    if month >= "2026-01":
        assert "2025年作为新一轮对比基期" in notes and "权数" in notes
    result = {**item, "title": titles[0], "url": receipt["url"],
        "published_at": clock.isoformat() if precision != "DAY_ONLY" else clock.strftime("%Y-%m-%d"),
        "publication_precision": precision, "clock_support": "PAGE_AND_CATALOGUE" if item.get("catalogue_date") else "ORIGINAL_PAGE_ONLY_LEGACY",
        "known_at": (clock.normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).isoformat(),
        "second_hand_down_count": falling, "second_hand_flat_count": flat_count, "second_hand_up_count": rising,
        "second_hand_down_share": falling / 70, "index_base_year": base_year,
        "raw_path": receipt["raw_path"], "raw_sha256": receipt["sha256"], "notes": notes,
        "table_instances_agree": True, "matching_table_instances": len(candidates),
        "table_caption": candidates[0]["caption"], "table_column_count": candidates[0]["column_count"],
        "historical_first_vintage_verified": False}
    city_rows = [{"stat_month": month, "city": city, "published_mom_index": float(value),
                  "index_base_year": base_year, "raw_sha256": receipt["sha256"]} for city, value in sorted(values.items())]
    return result, city_rows


def run():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("住宅月报研究已经固定，不覆盖旧结果。")
    for folder in ["raw", "requests", "code"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    lead = read(probe.OUT / "catalogue_lead.json")
    found = {r["stat_month"]: r for r in lead["catalogue_rows"]}
    found.update({month: {"stat_month": month, "url": url} for month, url in LEGACY.items()})
    assert sorted(found) == EXPECTED and len(CITIES) == 70
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY, "expected_months": EXPECTED,
        "sources": list(found.values()), "candidate": "每月已公布二手住宅总指数环比小于100的城市数/70，阈值100来自上月=100定义。",
        "measurement": "使用公布的一位小数环比；持平包括四舍五入及无成交的代入规则，不把持平当流动性充足。城市等权广度不是全国市值损失率。",
        "tables": "依据二手住宅总指数标题和环比表头解析，排除新房/分类表/同比；要求同一70城，桌面移动表完全一致。每城原值保留。",
        "method": "从2021新基期起收集；2026基期变为2025并调整类别权数。2021与2026官方平均同比影响不保证环比下跌城市数不变，必须标记，不反算旧权重。",
        "clock": "目录60个月与原页核对；2021年前8个月仅原页时间。统一原公布日日末可用，未认证不可变首版。",
        "network": "复用已存4份目标月报；至多64个新增URL、至多2并发、传输错误普通重试一次。",
        "new_accounts": 0, "new_strategy_returns": 0, "orders_authorized": False}, True)
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
        "code_sha256": digest(Path(__file__)), "probe_measurements_sha256": digest(probe.OUT / "source_measurements.json"),
        "catalogue_lead_sha256": digest(probe.OUT / "catalogue_lead.json"), "transport_code_sha256": digest(Path(transport.__file__))}, True)
    (OUT / "code" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    cache = {r["receipt"]["url"]: r["receipt"] for r in read(probe.OUT / "source_measurements.json")}
    records, city_rows, unresolved = [], [], []

    def consume(item, receipt):
        try:
            row, cities = parse(item, receipt)
            records.append(row)
            city_rows.extend(cities)
        except (AssertionError, ValueError, KeyError) as exc:
            unresolved.append({"item": item, "receipt": receipt, "error": str(exc)})

    todo = []
    for month in EXPECTED:
        item = found[month]
        if item["url"] in cache:
            consume(item, cache[item["url"]])
        else:
            todo.append(item)
    with ThreadPoolExecutor(max_workers=2) as executor:
        for start in range(0, len(todo), 2):
            batch = todo[start:start + 2]
            jobs = [(r["url"], "month_" + r["stat_month"]) for r in batch]
            for item, receipt in zip(batch, executor.map(REQUEST, jobs)):
                consume(item, receipt)
            if (start // 2) % 4 == 0 or start + 2 >= len(todo):
                print(f"住宅月报准入{len(records)}/68份，待核{len(unresolved)}份。", flush=True)
            if (OUT / "REQUESTS_STOPPED.json").exists():
                break
    records.sort(key=lambda r: r["stat_month"])
    frame = pd.DataFrame(records)
    if len(frame):
        frame["known_at"] = pd.to_datetime(frame.known_at).dt.tz_convert("Asia/Shanghai").dt.as_unit("ns")
        assert frame.known_at.is_monotonic_increasing and not frame.known_at.duplicated().any()
        frame.to_parquet(PANEL, index=False)
        pd.DataFrame(city_rows).sort_values(["stat_month", "city"]).to_parquet(OUT / "city_original_values.parquet", index=False)
    save(OUT / "released_records.json", records, True)
    save(OUT / "unresolved_fields.json", unresolved, True)
    missing = sorted(set(EXPECTED) - {r["stat_month"] for r in records})
    save(OUT / "result.json", {"at": now(), "study_id": STUDY,
        "status": "COMPLETE_HOUSING_BREADTH_SOURCES_READY_WITH_REBASE_BREAK" if not missing else "PARTIAL_HOUSING_BREADTH_SOURCES",
        "admitted_months": len(records), "expected_months": 68, "missing_months": missing, "city_observations": len(city_rows),
        "base_groups": frame.index_base_year.value_counts().to_dict() if len(frame) else {},
        "latest_source": records[-1] if records else None, "unresolved_fields": len(unresolved),
        "historical_first_vintage_verified": False, "new_accounts": 0, "new_strategy_returns": 0,
        "current_market_view": "NO_VIEW", "goal_achieved": False, "orders_authorized": False}, True)
    print(f"住宅下跌范围来源处理完成：{len(records)}/68份。", flush=True)


if __name__ == "__main__":
    run()
