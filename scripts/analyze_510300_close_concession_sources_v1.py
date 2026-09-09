"""离线核对510300来源样例；不生成价差信号、收益、账户或订单。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports/research/510300_close_concession_source_feasibility_v1"
CN = timezone(timedelta(hours=8))
SELECT = "name,last,chg_rate,change,open,prev_close,high,low,volume,amount,tradephase,cpxxextendname,iopv,fp_volume,fp_amount,fp_phase,cpxxsubtype".split(",")
COMPONENT_FIELDS = {
    "InstrumentID": ("INSTRUMENT_ID", "text"),
    "InstrumentName": ("INSTRUMENT_NAME", "text"),
    "Quantity": ("QUANTITY", "number"),
    "SubstitutionFlag": ("SUBSTITUTION_FLAG", "text"),
    "CreationPremiumRate": ("CREATION_PREMIUM_RATE", "optional_number"),
    "RedemptionDiscountRate": ("REDEMPTION_DISCOUNT_RATE", "optional_number"),
    "SubstitutionCashAmount": ("SUBSTITUTION_CASH_AMOUNT", "optional_number"),
    "UnderlyingSecurityID": ("UNDERLYION_SECURITY_ID", "text"),
}
HEADER_FIELDS = {
    "FundInstrumentID": ("TRADE_CODE", "text"),
    "TradingDay": ("TRADING_DAY", "text"),
    "PreTradingDay": ("PRE_TRADING_DAY", "text"),
    "CreationRedemptionUnit": ("CREATION_REDEMPTION_UNIT", "number"),
    "NAVperCU": ("NAVPERCU", "number"),
    "NAV": ("NAV", "number"),
    "PreCashComponent": ("PRE_CASH_COMPONENT", "number"),
    "EstimatedCashComponent": ("ESTIMATED_CASH_COMPONENT", "number"),
    "MaxCashRatio": ("MAX_CASH_RATIO", "number"),
    "RedemptionLimit": ("REDEMPTION_LIMIT", "number"),
    "RecordNumber": ("RECORD_NUM", "number"),
    "CreationRedemptionMechanism": ("CREATION_REDEMPTION_MECHANISM", "text"),
    "CreationRedemptionSwitch": ("CREATION_REDEMPTION", "switch"),
    "PublishIOPVFlag": ("PUBLISH_IOPV", "publish"),
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                      default=lambda x: str(x) if isinstance(x, Decimal) else _unsupported(x)) + "\n"


def _unsupported(value: object) -> None:
    raise TypeError(f"未定义的输出类型：{type(value).__name__}")


def read_jsonp(raw: bytes) -> dict:
    text = raw.decode("utf-8-sig").strip()
    if not text.startswith("{"):
        match = re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$.]*\s*\((.*)\)\s*;?", text, re.S)
        if not match:
            raise ValueError("来源不是JSON对象或明确的JSONP对象")
        text = match.group(1)
    result = json.loads(text, parse_float=Decimal)
    if not isinstance(result, dict):
        raise ValueError("来源顶层必须为对象")
    return result


def number(value: object, optional: bool = False) -> Decimal | None:
    if value is None or str(value).strip() in {"", "-", "--"}:
        if optional:
            return None
        raise ValueError("必需的数值缺失，不能补零")
    text = str(value).strip().replace("￥", "").replace("¥", "").replace(",", "")
    percent = text.endswith("%")
    if percent:
        text = text[:-1]
    try:
        result = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError("来源数值无法解析") from exc
    if not result.is_finite():
        raise ValueError("来源数值必须为有限数")
    return result / 100 if percent else result


def normalize(value: object, kind: str, xml: bool = False) -> object:
    if kind in {"number", "optional_number"}:
        return number(value, optional=kind == "optional_number")
    if kind == "switch" and xml:
        return {"1": "申购和赎回皆允许"}.get(str(value), f"未解释:{value}")
    if kind == "publish" and xml:
        return {"B": "是", "N": "否"}.get(str(value), f"未解释:{value}")
    if value is None or str(value).strip() == "":
        raise ValueError("必需的字符字段缺失")
    return str(value).strip()


def pcf_from_xml(raw: bytes, expected_date: str = "20260907") -> tuple[dict, dict]:
    root = ET.fromstring(raw)
    if root.tag != "ETFDefinition":
        raise ValueError("当前解析器只接受ETFDefinition，不自动混用公告文件")
    header = {child.tag: child.text for child in root if child.tag != "ComponentList"}
    if header.get("FundInstrumentID") != "510300" or header.get("TradingDay") != expected_date:
        raise ValueError("PCF证券身份或交易日不符")
    rows = {}
    for node in root.findall("./ComponentList/Component"):
        row = {child.tag: child.text for child in node}
        identity = (row.get("UnderlyingSecurityID"), row.get("InstrumentID"))
        if None in identity or identity in rows:
            raise ValueError("PCF成分身份缺失或重复")
        rows[identity] = row
    if len(rows) != int(header["RecordNumber"]):
        raise ValueError("PCF声明记录数与实收记录数不一致")
    return header, rows


def compare_pcf(xml_raw: bytes, basic_raw: bytes, component_raw: bytes) -> tuple[dict, list, list]:
    header, manager_rows = pcf_from_xml(xml_raw)
    basic_records = read_jsonp(basic_raw)["result"]
    if len(basic_records) != 1 or basic_records[0].get("TRADE_CODE") != "510300":
        raise ValueError("官网PCF基本信息身份不唯一或不符")
    basic = basic_records[0]
    sse_rows = {}
    for row in read_jsonp(component_raw)["result"]:
        identity = (row.get("UNDERLYION_SECURITY_ID"), row.get("INSTRUMENT_ID"))
        if None in identity or identity in sse_rows or row.get("ETF_VERSION") != "XML":
            raise ValueError("官网成分身份重复、缺失或版本不符")
        sse_rows[identity] = row
    header_checks = []
    for xml_key, (sse_key, kind) in HEADER_FIELDS.items():
        left, right = normalize(header.get(xml_key), kind, True), normalize(basic.get(sse_key), kind)
        header_checks.append({"xml_field": xml_key, "sse_field": sse_key,
                              "xml_raw": header.get(xml_key), "sse_raw": basic.get(sse_key),
                              "xml_normalized": left, "sse_normalized": right, "match": left == right})
    checks = []
    for identity in sorted(set(manager_rows) | set(sse_rows)):
        left_row, right_row = manager_rows.get(identity), sse_rows.get(identity)
        if left_row is None or right_row is None:
            checks.append({"market_id": identity[0], "instrument_id": identity[1], "field": "ROW_IDENTITY",
                           "xml_raw": None, "sse_raw": None, "xml_normalized": None, "sse_normalized": None,
                           "xml_presence": left_row is not None, "sse_presence": right_row is not None, "match": False})
            continue
        for xml_key, (sse_key, kind) in COMPONENT_FIELDS.items():
            left, right = normalize(left_row.get(xml_key), kind), normalize(right_row.get(sse_key), kind)
            checks.append({"market_id": identity[0], "instrument_id": identity[1], "field": xml_key,
                           "xml_raw": left_row.get(xml_key), "sse_raw": right_row.get(sse_key),
                           "xml_normalized": left, "sse_normalized": right,
                           "xml_presence": xml_key in left_row, "sse_presence": sse_key in right_row, "match": left == right})
    delta = number(header["NAV"]) * number(header["CreationRedemptionUnit"]) - number(header["NAVperCU"])
    rounding = Decimal("0.00005") * number(header["CreationRedemptionUnit"])
    if any(r["SubstitutionFlag"] not in {"0", "1", "2"} for r in manager_rows.values()):
        raise ValueError("本轮估值输入说明不解释其他现金替代标志")
    live_price_rows = [r for r in manager_rows.values() if r["SubstitutionFlag"] in {"0", "1"} and number(r["Quantity"]) > 0]
    fixed_cash = sum((number(r.get("SubstitutionCashAmount")) for r in manager_rows.values() if r["SubstitutionFlag"] == "2"), Decimal(0))
    result = {"status": "MATCHED_SAVED_PCF_FIELDS" if all(x["match"] for x in header_checks + checks) else "PCF_CROSSCHECK_MISMATCH",
              "fund_id": "510300", "trading_day": header["TradingDay"], "nav_economic_day": header["PreTradingDay"],
              "manager_rows": len(manager_rows), "sse_rows": len(sse_rows),
              "common_header_checks": len(header_checks), "component_field_checks": len(checks),
              "mismatch_count": sum(not x["match"] for x in header_checks + checks),
              "market_counts": dict(sorted(Counter(k[0] for k in manager_rows).items())),
              "substitution_flag_counts": dict(sorted(Counter(r["SubstitutionFlag"] for r in manager_rows.values()).items())),
              "quantity_zero_count": sum(number(r["Quantity"]) == 0 for r in manager_rows.values()),
              "xml_cash_amount_absent_count": sum("SubstitutionCashAmount" not in r for r in manager_rows.values()),
              "xml_creation_premium_absent_count": sum("CreationPremiumRate" not in r for r in manager_rows.values()),
              "xml_redemption_discount_absent_count": sum("RedemptionDiscountRate" not in r for r in manager_rows.values()),
              "sse_cash_amount_dash_count": sum(r.get("SUBSTITUTION_CASH_AMOUNT", "").strip() == "-" for r in sse_rows.values()),
              "creation_redemption_unit": number(header["CreationRedemptionUnit"]),
              "prior_nav_per_share": number(header["NAV"]), "prior_nav_per_creation_unit": number(header["NAVperCU"]),
              "estimated_cash_component_cny_per_creation_unit": number(header["EstimatedCashComponent"]),
              "prior_cash_component_cny_per_creation_unit": number(header["PreCashComponent"]),
              "cash_dividend_xml_only": number(header["CashDividend"]),
              "publish_iopv_flag_xml": header["PublishIOPVFlag"],
              "nav_rounding_difference_cny": delta, "nav_four_decimal_half_unit_bound_cny": rounding,
              "nav_difference_within_display_rounding_bound": abs(delta) <= rounding,
              "manager_document_root": "ETFDefinition", "exchange_confirmed_file_captured": False,
              "original_publication_time_proven": False, "revision_history_available": False,
              "independent_fair_value_validated": False,
              "valuation_input_recipe": {
                  "state": "DEFINED_FROM_PROSPECTUS_NOT_VALUED",
                  "required_positive_quantity_price_count": len(live_price_rows),
                  "required_price_market_counts": dict(sorted(Counter(r["UnderlyingSecurityID"] for r in live_price_rows).items())),
                  "required_underlying_close_prices_acquired_this_round": 0,
                  "must_cash_substitution_total_cny_per_creation_unit": fixed_cash,
                  "estimated_cash_component_cny_per_creation_unit": number(header["EstimatedCashComponent"]),
                  "denominator_fund_shares": number(header["CreationRedemptionUnit"]),
                  "formula": "(sum(quantity_i * synchronous_price_i for flags 0/1) + sum(fixed_cash_i for flag 2) + estimated_cash) / creation_unit",
                  "source": "2026年第1号招募说明书PDF物理页33-34、36-39",
                  "non_shanghai_allowed_cash_rows_still_need_prices_for_iopv": True,
                  "cash_substitution_premium_is_not_a_price_concession_or_valuation_error_bound": True,
                  "actual_fund_nav": "NOT_COMPUTED",
                  "closing_iopv_replica": "NOT_COMPUTED",
                  "valuation_error_bound": "NOT_IDENTIFIED"},
              "scope": "当日保存的定义文件与官网展示字段一致；不证明实际持仓、独立估值或开盘前可用性"}
    return result, header_checks, checks


def snapshot(raw: bytes, fields: list[str] = SELECT) -> dict:
    data = read_jsonp(raw)
    if data.get("code") != "510300" or str(data.get("date")) != "20260907":
        raise ValueError("快照证券身份或交易日不符")
    if len(data.get("snap", [])) != len(fields):
        raise ValueError("快照长度与已保存官网字段合同不一致")
    result = dict(zip(fields, data["snap"]))
    result.update({"source_envelope_date": data["date"], "source_envelope_time": data["time"]})
    return result


def iopv_state(value: object) -> str:
    parsed = number(value, optional=True)
    return "NOT_VALID_AS_REFERENCE" if parsed is None or parsed <= 0 else "VALUE_PRESENT_TIMESTAMP_NOT_PROVEN"


def observation_state(source_hhmmss: int, receive_iso: str) -> str:
    datetime.strptime(str(source_hhmmss).zfill(6), "%H%M%S")
    received = datetime.fromisoformat(receive_iso)
    if received.utcoffset() is None:
        raise ValueError("接收时间必须包含时区")
    received = received.astimezone(CN)
    receive_clock = received.hour * 10000 + received.minute * 100 + received.second
    if not (150500 <= source_hhmmss <= 153000 and 150500 <= receive_clock <= 153000):
        return "NOT_OBSERVED_IN_POST_CLOSE_LEGAL_WINDOW"
    return "WINDOW_ONLY_NOT_A_QUALIFIED_SIGNAL"


def verify_source_receipts(report: Path) -> tuple[dict, dict]:
    receipts_by_id = {}
    coverage = []
    for receipt_path in sorted((report / "source_batches").glob("*/receipts.json")):
        receipts = json.loads(receipt_path.read_text(encoding="utf-8"))
        manifest = json.loads((receipt_path.parent / "request_manifest.json").read_text(encoding="utf-8"))
        if {j["id"] for j in manifest["jobs"]} != {r["id"] for r in receipts}:
            raise ValueError("请求清单与回执不对应")
        jobs = {j["id"]: j for j in manifest["jobs"]}
        for receipt in receipts:
            if receipt["id"] in receipts_by_id or receipt["requested_url"] != jobs[receipt["id"]]["url"]:
                raise ValueError("回执编号重复或URL与预存请求不同")
            if datetime.fromisoformat(manifest["recorded_before_requests_at"]) > datetime.fromisoformat(receipt["request_started_at"]):
                raise ValueError("请求清单没有先于请求保存")
            if receipt["raw_path"]:
                target = (report / receipt["raw_path"]).resolve()
                if not target.is_relative_to(report.resolve()):
                    raise ValueError("原始响应路径超出报告目录")
                raw = target.read_bytes()
                if len(raw) != receipt["bytes"] or digest(raw) != receipt["sha256"]:
                    raise ValueError("原始响应大小或哈希与回执不一致")
            receipts_by_id[receipt["id"]] = receipt
        coverage.append({"batch": receipt_path.parent.name, "requests": len(receipts),
                         "script_sha256_before_request": manifest["probe_script_sha256"]})
    receipt_values = list(receipts_by_id.values())
    return {"status": "PASS_SAVED_REQUEST_AND_RAW_IDENTITY", "batches": coverage,
            "request_count": len(receipt_values), "raw_file_count": sum(bool(r["raw_path"]) for r in receipt_values),
            "statuses": dict(sorted(Counter(r["status"] for r in receipt_values).items())),
            "first_request_started_at": min(r["request_started_at"] for r in receipt_values),
            "last_response_received_at": max(r["received_at"] for r in receipt_values),
            "failed_requests": [{"id": r["id"], "status": r["status"], "error": r.get("error")} for r in receipt_values if r["status"] != "RAW_RESPONSE_CAPTURED_CONTENT_NOT_YET_ADMITTED"],
            "clock_offset_proven": False,
            "tls_receipt_note": "采集脚本tls_verified表示启用默认TLS验证；失败请求并未因此证明握手成功。"}, receipts_by_id


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("不能把缺失的核对结果写成空表通过")
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(report: Path) -> tuple[dict, list, list]:
    receipts_result, receipts = verify_source_receipts(report)

    def raw(identity: str) -> bytes:
        receipt = receipts[identity]
        if receipt["status"] != "RAW_RESPONSE_CAPTURED_CONTENT_NOT_YET_ADMITTED" or not receipt["raw_path"]:
            raise ValueError(f"必要原始输入未成功保存：{identity}")
        return (report / receipt["raw_path"]).read_bytes()

    pcf, header_checks, component_checks = compare_pcf(raw("manager_pcf_xml"), raw("sse_pcf_basic"), raw("sse_pcf_components"))
    detail = raw("etf_detail_new_js").decode("utf-8-sig")
    selection = re.search(r'function getSnapData\(callBack\)\s*\{\s*var select\s*=\s*"([^"]+)"', detail)
    if selection is None or selection.group(1).split(",") != SELECT:
        raise ValueError("保存的官网详情脚本字段与解析器不一致")
    first, second = snapshot(raw("sse_snapshot_existing_fields")), snapshot(raw("sse_snapshot_second_lunch"))
    default = read_jsonp(raw("sse_snapshot_default_schema"))
    if default["code"] != "510300" or default["date"] != 20260907 or len(default["snap"]) != 15:
        raise ValueError("默认盘口样例结构变化，不能继续沿用下标")
    quote_text = raw("tencent_510300_quote").decode("gb18030").strip()
    quote_match = re.fullmatch(r'v_sh510300="([^"]+)";', quote_text)
    if quote_match is None:
        raise ValueError("腾讯交叉核对样例身份不符")
    quote = quote_match.group(1).split("~")
    if quote[2] != "510300":
        raise ValueError("腾讯证券身份不符")
    minute = read_jsonp(raw("tencent_510300_intraday"))["data"]["sh510300"]["data"]
    if minute["date"] != "20260907":
        raise ValueError("分钟字段交叉核对日期不符")
    last_minute = minute["data"][-1].split()
    book_checks = []
    for side, sse_index, qq_start in [("bid", 13, 9), ("ask", 14, 19)]:
        book = default["snap"][sse_index]
        if len(book) != 10:
            raise ValueError("普通五档盘口长度不符")
        for level in range(5):
            book_checks.append({"side": side, "level": level + 1,
                                "price_match": number(book[2 * level]) == number(quote[qq_start + 2 * level]),
                                "quantity_after_hands_to_shares_match": number(book[2 * level + 1]) == number(quote[qq_start + 2 * level + 1]) * 100})
    time_rows = []
    for identity, item in [("sse_snapshot_existing_fields", first), ("sse_snapshot_second_lunch", second)]:
        received = datetime.fromisoformat(receipts[identity]["received_at"])
        envelope = datetime.strptime(str(item["source_envelope_date"]) + str(item["source_envelope_time"]).zfill(6), "%Y%m%d%H%M%S").replace(tzinfo=CN)
        time_rows.append({"source_id": identity, "envelope_timestamp_assuming_china_timezone": envelope.isoformat(),
                          "received_at_local_clock": received.isoformat(),
                          "raw_receive_minus_envelope_seconds_not_latency": (received - envelope).total_seconds(),
                          "qualified_latency_seconds": None})
    selected_same = all(first[k] == second[k] for k in SELECT)
    quote_result = {"selected_fields_from_official_client": SELECT, "selected_field_count": len(SELECT),
                    "official_client_last_modified_header": receipts["etf_detail_new_js"].get("last_modified_header"),
                    "first_snapshot": first, "second_snapshot": second, "all_selected_values_unchanged": selected_same,
                    "source_envelope_time_advanced": second["source_envelope_time"] > first["source_envelope_time"],
                    "time_evidence": time_rows, "price_or_iopv_update_time_proven": False,
                    "iopv_state": iopv_state(first["iopv"]),
                    "post_close_observation_state": observation_state(first["source_envelope_time"], receipts["sse_snapshot_existing_fields"]["received_at"]),
                    "post_close_pending_buy_qty": None, "post_close_pending_sell_qty": None,
                    "post_close_queue_state": "NOT_EXPOSED_IN_INSPECTED_OFFICIAL_DETAIL_CONTRACT",
                    "ordinary_five_level_crosschecks": book_checks,
                    "ordinary_five_level_all_match": all(r["price_match"] and r["quantity_after_hands_to_shares_match"] for r in book_checks),
                    "ordinary_book_is_post_close_queue": False,
                    "tencent_timestamp_raw": quote[30],
                    "minute_rows_received_for_schema_inspection": len(minute["data"]),
                    "minute_first_time": minute["data"][0].split()[0], "minute_last_time": last_minute[0],
                    "last_price_matches_sse_tencent_and_minute": first["last"] == number(quote[3]) == number(last_minute[1]),
                    "cum_amount_cny_matches_sse_and_minute": number(first["amount"]) == number(last_minute[3]),
                    "sse_cum_volume_shares": number(first["volume"]), "tencent_cum_volume_hands": number(quote[6]),
                    "hands_conversion_minus_sse_shares": number(quote[6]) * 100 - number(first["volume"]),
                    "volume_equality_claim": "数量差异与整手显示舍入相容；不是精确相等或完整舍入协议的证明",
                    "data_entitlement_or_latency_guarantee_proven": False}
    result = {"study_id": "510300_CLOSE_CONCESSION_SOURCE_FEASIBILITY_V1",
              "status": "COMPLETED_SOURCE_FEASIBILITY_PARTIAL_NOT_SIGNAL_READY",
              "receipts": receipts_result, "pcf": pcf, "quote": quote_result,
              "model_action": "ABSTAIN", "model_position_target": "UNSET", "position_impact": 0,
              "qualified_close_observations": 0, "qualified_post_close_observations": 0,
              "new_forward_quality_days": 0, "old_forward_ledger_writes": 0,
              "reference_value": "NOT_COMPUTED", "price_concession": "NOT_COMPUTED",
              "future_event_returns": "NOT_COMPUTED", "account_paths": "NOT_COMPUTED", "net_sharpe": "NOT_COMPUTED",
              "actual_fill_fraction": "UNKNOWN_NO_ORDER_EVIDENCE", "broker_connections": 0,
              "source_probe_only": True, "strategy_registered": False, "external_review_received": False}
    if pcf["mismatch_count"] or not quote_result["ordinary_five_level_all_match"]:
        result["status"] = "COMPLETED_SOURCE_FEASIBILITY_WITH_CROSSCHECK_FAILURE"
    return result, header_checks, component_checks


def main() -> None:
    parser = argparse.ArgumentParser(description="离线复核510300已保存来源，不联网、不读取后续收益")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--verify-saved", action="store_true", help="只读复算并比较已保存输出")
    args = parser.parse_args()
    result, header_checks, component_checks = analyze(args.report)
    output = encode(result)
    target = args.report / "analysis_result.json"
    if args.verify_saved:
        if target.read_text(encoding="utf-8") != output:
            raise ValueError("复算结果与保存输出不同")
        for name, rows in [("pcf_header_crosscheck.csv", header_checks), ("pcf_component_crosscheck.csv", component_checks)]:
            import io
            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            saved = (args.report / name).read_bytes().decode("utf-8-sig")
            if saved != buffer.getvalue():
                raise ValueError(f"保存核对表与复算不同：{name}")
        print("已保存来源离线复算通过；未联网、未生成账户、未读取后续收益。")
    else:
        target.write_text(output, encoding="utf-8", newline="\n")
        write_csv(args.report / "pcf_header_crosscheck.csv", header_checks)
        write_csv(args.report / "pcf_component_crosscheck.csv", component_checks)
        print(encode({"status": result["status"], "requests": result["receipts"]["request_count"],
                      "pcf_rows": result["pcf"]["manager_rows"], "pcf_mismatches": result["pcf"]["mismatch_count"],
                      "qualified_post_close_observations": result["qualified_post_close_observations"]}))


if __name__ == "__main__":
    main()
