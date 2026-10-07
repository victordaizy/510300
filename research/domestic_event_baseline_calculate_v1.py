"""拆解发布前预期、PMI构成和逆回购续作，不创建收益信号。"""

import json
import re
from datetime import datetime, timedelta, timezone

from bs4 import BeautifulSoup

from company_revision_driver_sources_v1 import now, save
from domestic_event_baseline_sources_v1 import ROOT, OUT


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def calendar_row():
    source = OUT / "sources/te_nbs_manufacturing_calendar.html"
    soup = BeautifulSoup(source.read_text(encoding="utf-8"), "html.parser")
    rows = []
    for tr in soup.select("table tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        if cells and cells[0] == "2026-09-30":
            rows.append(cells)
    if len(rows) != 1 or len(rows[0]) != 8 or rows[0][2] != "NBS Manufacturing PMI":
        raise ValueError("未识别唯一的待公布制造业日历行。")
    cells = rows[0]
    if cells[4] != "":
        raise ValueError("实际值已非空，不能作为本轮发布前预期。")
    receipt = read(OUT / "receipts/te_nbs_manufacturing_calendar.json")
    release = datetime(2026, 9, 30, 9, 30, tzinfo=timezone(timedelta(hours=8)))
    if datetime.fromisoformat(receipt["received_at"]) >= release:
        raise ValueError("接收时间晚于本轮预定发布时点。")
    return {"previous": float(cells[5]), "consensus": float(cells[6]), "vendor_model_forecast": float(cells[7]),
            "actual": None, "received_at": receipt["received_at"], "raw_cells": cells}


def pmi_components():
    source = ROOT / "reports/research/510300_current_driver_outlook_20260929/sources/nbs_august_pmi.html"
    soup = BeautifulSoup(source.read_text(encoding="utf-8"), "html.parser")
    months = {}
    for table in soup.select("table"):
        text = table.get_text(" ", strip=True)
        if not all(term in text for term in ["新订单", "生产", "供应商"]):
            continue
        for tr in table.select("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            if len(cells) != 7:
                continue
            period = re.sub(r"\s+", "", cells[0])
            if period in {"2026年7月", "2026年8月"}:
                row = dict(zip(["headline", "production", "new_orders", "inventory", "employment", "delivery"], map(float, cells[1:])))
                if period in months and months[period] != row:
                    raise ValueError("桌面与移动端的PMI同月数据不一致。")
                months[period] = row
    if len(months) != 2:
        raise ValueError("7月或8月PMI构成缺失。")
    weights = {"new_orders": .30, "production": .25, "employment": .20, "delivery": .15, "inventory": .10}
    reconstructed = {}
    for period, row in months.items():
        reconstructed[period] = round(sum(weight * (100 - row[key] if key == "delivery" else row[key]) for key, weight in weights.items()), 6)
    july, august = months["2026年7月"], months["2026年8月"]
    contribution = {key: round((august[key] - july[key]) * weight * (-1 if key == "delivery" else 1), 6) for key, weight in weights.items()}
    return {"source": source.relative_to(ROOT).as_posix(), "months": months, "official_weights": weights,
            "reconstructed_from_rounded_components": reconstructed,
            "july_to_august_contribution_index_points": contribution,
            "reconstructed_change": round(sum(contribution.values()), 6),
            "published_headline_change": round(august["headline"] - july["headline"], 6),
            "rounding_note": "分项只公布一位小数，重建与公布总指数允许舍入差；贡献是构成恒等式而非经济因果份额。"}


def operations():
    result = {}
    for date in ["20260922", "20260923", "20260928", "20260929"]:
        text = (OUT / ("sources/pbc_" + date + "_operations.txt")).read_text(encoding="utf-8")
        flat = re.sub(r"\s+", "", text)
        seven = re.search(r"开展了(\d+)亿元7天期逆回购", flat)
        overnight = re.search(r"开展了(\d+)亿元隔夜逆回购", flat)
        rate = re.search(r"7天(\d+\.\d+)%", flat)
        if not seven or not rate:
            raise ValueError("公开市场操作关键字段未找到：" + date)
        result[date] = {"seven_day_100m": int(seven[1]), "seven_day_rate_percent": float(rate[1]),
                        "overnight_100m": int(overnight[1]) if overnight else None}
    old, today = result["20260928"], result["20260929"]
    return {"announcements": result,
            "september29": {"seven_day_new": today["seven_day_100m"], "overnight_new": today["overnight_100m"],
                            "known_gross": today["seven_day_100m"] + today["overnight_100m"],
                            "seven_day_maturing": result["20260922"]["seven_day_100m"], "overnight_maturing": old["overnight_100m"],
                            "known_maturing": result["20260922"]["seven_day_100m"] + old["overnight_100m"],
                            "known_principal_net": today["seven_day_100m"] + today["overnight_100m"] - result["20260922"]["seven_day_100m"] - old["overnight_100m"]},
            "september30_known_maturities": {"overnight": today["overnight_100m"], "seven_day": result["20260923"]["seven_day_100m"],
                                           "total": today["overnight_100m"] + result["20260923"]["seven_day_100m"]},
            "unit": "亿元人民币本金",
            "maturity_derivation": "按公告操作日和隔夜/7天期限推算29、30日到期；这两个到期日均为营业日。未把未知其他渠道填零。",
            "scope": "所核对逆回购本金的净变化；不含利息、其他工具、财政收支、现金需求，不是银行体系完整流动性净变化。",
            "overnight_plan": {"prior_cap": 6000, "prior_window": "9月14至17日", "current_cap": 10000,
                               "current_window": "9月28日至10月8日", "announced_at": "2026-09-23 17:00:00 Asia/Shanghai",
                               "remaining_cap_is_commitment": False,
                               "not_a_permanent_policy_rate_cut": True}}


def main():
    if (OUT / "reviewed_facts.json").exists():
        raise SystemExit("发布前基准已经保存，不覆盖。")
    calendar = calendar_row()
    web = read(OUT / "sources/investing_calendars_web_receipt.json")
    if "50.1  | 49.8" not in web["result"] or "49.3  | 49.0" not in web["result"]:
        raise ValueError("网页工具回执未包含所需的待公布日历行。")
    previous_cards = read(ROOT / "reports/research/510300_current_driver_outlook_20260929/forecast_cards.json")
    bulletin = read(OUT / "sources/chinamoney_daily_bulletin.json")
    facts = {
        "recorded_at": now(), "expected_release": "2026-09-30 09:30 Asia/Shanghai",
        "public_expectation_snapshot": {"manufacturing_te": calendar,
            "manufacturing_investing": {"forecast": 50.1, "previous": 49.8, "actual": None},
            "nonmanufacturing_business_activity_investing": {"forecast": 49.3, "previous": 49.0, "actual": None},
            "status": "PRE_EVENT_PUBLIC_CALENDAR_BASELINE_CAPTURED",
            "scope": "供应商公开预期代理，调查样本和离散度未取得；两个相同数字不当作两次独立调查。",
            "not_replaced": ["F1制造业与非制造业新订单", "F2购进与出厂价格差", "F3跨季后资金利差"]},
        "pmi_composition": pmi_components(), "funding": operations(),
        "forecast_state": [{"id": c["id"], "status": c["status"]} for c in previous_cards["cards"]],
        "dr007": {"status": "NO_ADMITTED_OFFICIAL_FINAL_OBSERVATION",
                  "endpoint_business_code": bulletin.get("head", {}).get("rep_code"),
                  "endpoint_message": bulletin.get("head", {}).get("rep_message"),
                  "F3_evaluated": False, "substitute_used": False,
                  "note": "HTTP200内的业务错误不能当成有效行情；本轮未取得官方最终DR007，9月30日及节后观测尚未发生。"},
        "unidentified": ["新订单分项的完整市场共识", "PMI变化对应沪深300未来盈利增量", "实际价格已反映的份额", "当前成本后正收益优势"],
        "interpretation_before_release": [
            "标题PMI只回到50.1对应已保存的公开预期，不自动成为新利好。",
            "配送变慢能机械抬高标题PMI；先区分需求拥挤与供给堵塞。",
            "购进价格回落既可能来自供给修复也可能来自需求下降；同时看订单和交付。",
            "隔夜工具规模增加先看续作与到期，政策供给不是股票新增购买。",
        ],
        "current_index_return_forecast_made": False, "new_accounts": 0, "new_return_tests": 0,
        "new_fitted_parameters": 0, "new_forecast_cards": 0,
    }
    save(OUT / "reviewed_facts.json", facts)
    save(OUT / "release_readiness.json", {
        "recorded_at": now(), "state": "BASELINES_CAPTURED_RELEASE_AND_FUTURE_RATES_NOT_YET_OBSERVED",
        "specific_events": ["9月30日官方PMI第一版", "9月29、30日最终DR007", "10月8、9、12日最终DR007与适用政策利率"],
        "original_cards": "reports/research/510300_current_driver_outlook_20260929/forecast_cards.json",
        "read_sequence": ["按原F1/F2逐项判对错", "按已保存标题预期计算偏离，分项缺共识保留未知", "把变化拆到订单、交付、成本和行业范围", "对照当时实际价格与成本，再判断剩余空间"],
        "live_process_or_tool_handle": None, "verified_wait": False, "automation_created": False,
        "no_substitution": "未来结果未发生时不提前评分；证据不足不是实际现金持仓声明。",
    })
    print("制造业公开预期", calendar["consensus"], "供应商模型", calendar["vendor_model_forecast"])
    print("PMI构成变化", facts["pmi_composition"]["july_to_august_contribution_index_points"])
    print("9月29日已核对逆回购本金净增", facts["funding"]["september29"]["known_principal_net"], "亿元")
    print("9月30日已知到期本金", facts["funding"]["september30_known_maturities"]["total"], "亿元")
    print("原预测仍待未来信息，未生成股票收益信号。")


if __name__ == "__main__":
    main()
