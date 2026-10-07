"""从保存的日表计算事件前后变化；只做机制复盘，不回测择时。"""

import json
from datetime import datetime
from decimal import Decimal

from bs4 import BeautifulSoup

from company_revision_driver_sources_v1 import now, save
from september_rate_shock_sources_v1 import ROOT, OUT


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def parse_rates():
    soup = BeautifulSoup((OUT / "sources/fed_h15_current.html").read_text(encoding="utf-8"), "html.parser")
    candidates = [table for table in soup.select("table") if "Inflation indexed" in table.get_text(" ", strip=True)]
    if len(candidates) != 1:
        raise ValueError("未识别唯一的H.15利率表。")
    rows = [[c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])] for tr in candidates[0].select("tr")]
    dates = [datetime.strptime(x, "%Y %b %d").date().isoformat() for x in rows[0][1:]]
    values = {day: {} for day in dates}
    section = None
    for cells in rows[1:]:
        if not cells:
            continue
        label = cells[0]
        if label.startswith("Nominal"):
            section = "nominal"
        elif label.startswith("Inflation indexed"):
            section = "real"
        key = "effective_federal_funds" if label.startswith("Federal funds (effective)") else None
        if section == "nominal" and label in {"2-year", "10-year"}:
            key = "nominal_" + label.replace("-year", "y")
        if section == "real" and label == "10-year":
            key = "real_10y"
        if key:
            for day, value in zip(dates, cells[1:]):
                values[day][key] = float(Decimal(value))
    for day, row in values.items():
        if set(row) != {"effective_federal_funds", "nominal_2y", "nominal_10y", "real_10y"}:
            raise ValueError("关键利率字段缺失：" + day)
        row["nominal_minus_real_10y"] = round(row["nominal_10y"] - row["real_10y"], 6)
    return values


def rate_change(rates, start, end):
    return {key: round((rates[end][key] - rates[start][key]) * 100, 6) for key in rates[start]}


def price_moves():
    index_source = ROOT / "reports/research/510300_index_repricing_odds_v1/sources/csi300_perf.json"
    etf_source = ROOT / "reports/research/510300_weight_company_driver_bridge_20260929/prices/sh510300.raw"
    index = {datetime.strptime(r["tradeDate"], "%Y%m%d").date().isoformat(): r for r in read(index_source)["data"]}
    etf = {r[0]: {"open": float(r[1]), "close": float(r[2])} for r in read(etf_source)["data"]["sh510300"]["day"]}
    output = []
    for symbol, observations in [("000300", index), ("510300", etf)]:
        for previous, day in [("2026-09-22", "2026-09-23"), ("2026-09-23", "2026-09-24"), ("2026-09-24", "2026-09-28")]:
            prev_close = observations[previous]["close"]
            open_price, close = observations[day]["open"], observations[day]["close"]
            output.append({
                "symbol": symbol, "previous_session": previous, "session": day,
                "previous_close": prev_close, "open": open_price, "close": close,
                "gap_pct": 100 * (open_price / prev_close - 1),
                "open_to_close_pct": 100 * (close / open_price - 1),
                "close_to_close_pct": 100 * (close / prev_close - 1),
                "event_relation": "PMI公布前" if day == "2026-09-23" else "PMI后首个A股交易日" if day == "2026-09-24" else "休市后多信息混合窗口",
            })
    return output


def main():
    if (OUT / "reviewed_facts.json").exists():
        raise SystemExit("已保存事件事实，不覆盖。")
    rates = parse_rates()
    changes = {start + "_to_" + end: rate_change(rates, start, end) for start, end in [
        ("2026-09-22", "2026-09-23"), ("2026-09-23", "2026-09-25"), ("2026-09-22", "2026-09-25")
    ]}
    facts = {
        "recorded_at": now(),
        "rates": {"unit": "年化百分数", "source": "sources/fed_h15_current.html", "release_date": "2026-09-28",
                  "last_observation": "2026-09-25", "rows": rates, "change_unit": "基点", "changes": changes,
                  "scope": "名义与TIPS实际收益率的同期限日收盘代理；差额不是纯通胀预期。实际收益率仍含期限及流动性等补偿。",
                  "unidentified": ["纯预期实际短率", "期限溢价", "纯通胀预期", "流动性溢价", "信用风险份额"],
                  "original_event_time_vintage_obtained": False},
        "pmi": {"source": "sources/spglobal_commentary_web_receipt.json", "publication_date": "2026-09-23",
                "composite_output_actual": 58.4, "previous": 56.0,
                "forward_business_output_expectations": "与8月相比不变，不能等同股票市场盈利预期。",
                "mechanisms": ["当期需求增强伴随积压和交付延迟", "岗位增加仍有合适员工短缺", "燃料运输成本增加", "竞争约束部分售价传导"],
                "local_pdf_status": "HTTP403未取得，已保存网页工具对官方PDF的文本读取回执"},
        "expectation_proxy": {
            "source": "sources/squawk_pmi_preview.html", "status": "RETROSPECTIVELY_RECONSTRUCTED_DATED_PUBLIC_PREVIEW_NOT_ORIGINAL_SURVEY",
            "displayed_publication_utc": "2026-09-23T13:11:00Z", "displayed_update_utc": "2026-09-23T13:45:00Z",
            "claimed_underlying_survey": "Bloomberg；没有取得原始调查参与者、范围与截止时点。",
            "composite": 55.3, "manufacturing": 53.7, "services": 55.8,
            "composite_difference_index_points": round(58.4 - 55.3, 6),
            "limits": "网页同时保留事后更新，当前才读取；不是本账户事前保存的预期，也不是股价隐含预期。",
            "excluded_econoday": "页面正文日期9月25日、标题9月28日，却含9月23日前瞻；日期冲突未解，不把53.8/56.0当已证实的事前基准。",
        },
        "timing_shanghai": [
            {"time": "2026-09-23 15:00", "event": "A股当日收盘，尚未到本次美国PMI发布时点"},
            {"time": "2026-09-23 21:11", "event": "预告页面所标时间；事后重建"},
            {"time": "2026-09-23 21:45", "event": "美国PMI初值，预告及其更新所标09:45 EDT"},
            {"time": "2026-09-23 22:05", "event": "美联储日历安排Barr讲话；预定时点不等于证实网页首次公开时点"},
            {"time": "2026-09-24 09:30", "event": "PMI后首个A股交易日开盘"},
        ],
        "barr": {"source": "sources/fed_barr_20260923.html", "calendar": "sources/fed_september_calendar.html",
                 "policy_signal": "个人基准判断仍可能需要进一步政策调整，并非FOMC新的加息决定。",
                 "why": "通胀风险提高、劳动市场风险缓和；讲话列及关税、冲突与AI投资需求冲击。",
                 "independent_of_pmi": False},
        "prices": price_moves(),
        "price_scope": "未复权日线价格变化，不含分红、费用、账户、T+1成交，不是可执行策略收益。",
        "causal_attribution": "日频包含PMI、官员表态及其他同日信息，不识别单个事件的精确因果份额。",
        "remaining_edge": "未识别；过去利率上升及A股下跌，不证明9月29日继续下跌或反弹有净优势。",
        "current_conditional_view": "若需求仍强而有效供给、人员和能源约束未缓解，外部实际回报要求的压力不易仅靠单次国内数据改善消退；这是条件推断。",
        "most_discriminating_next_evidence": "需求能否维持，同时交期、投入与销售价格压力缓解；利率回落若伴随需求下滑，须同时降低盈利判断。",
        "new_forecast_cards": 0, "new_accounts": 0, "new_return_tests": 0, "new_fitted_parameters": 0,
    }
    save(OUT / "reviewed_facts.json", facts)
    print("利率变化（基点）", json.dumps(changes, ensure_ascii=False))
    for row in facts["prices"]:
        print(row["symbol"], row["session"], "跳空", round(row["gap_pct"], 4), "日内", round(row["open_to_close_pct"], 4), "全天", round(row["close_to_close_pct"], 4))
    print("事实及条件推断已保存；没有新增收益预测卡或账户测试。")


if __name__ == "__main__":
    main()
