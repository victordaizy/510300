"""510300官方日频份额版本观察，保存真实取得时刻；不生成交易信号。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).absolute().parent.parent
OUT = ROOT / "reports/research/510300_current_official_share_source_v1"
CALENDAR = ROOT / "data/reference/sse_trade_calendar_2026.csv"
ZONE = ZoneInfo("Asia/Shanghai")
BASE_URL = "https://query.sse.com.cn/commonQuery.do"
SQL_ID = "COMMON_SSE_ZQPZ_ETFZL_XXPL_ETFGM_SEARCH_L"
TARGET = "510300"
EARLIEST = date(2026, 10, 8)
PARAMS = {"isPagination": "true", "pageHelp.pageSize": "10000", "pageHelp.pageNo": "1",
          "pageHelp.beginPage": "1", "pageHelp.cacheSize": "1", "pageHelp.endPage": "1", "sqlId": SQL_ID}


def stamp():
    return datetime.now(ZONE)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def calendar(path=CALENDAR):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not all(r.get("source", "").startswith("https://www.sse.com.cn/") for r in rows):
        raise ValueError("交易日历缺少官方来源，停止。")
    dates = [date.fromisoformat(r["trade_date"]) for r in rows]
    if dates != sorted(set(dates)):
        raise ValueError("交易日历重复或无序。")
    return dates


def decode_response(raw):
    text = raw.decode("utf-8-sig").strip()
    if not text.startswith(("{", "[")):
        match = re.fullmatch(r"[A-Za-z_$][\w.$]*\s*\(([\s\S]*)\)\s*;?", text)
        if not match:
            raise ValueError("官方响应不是JSON或合法JSONP，不执行返回内容。")
        text = match.group(1)
    value = json.loads(text)
    if not isinstance(value, dict) or not isinstance(value.get("result"), list):
        raise ValueError("官方响应缺少result列表，不能猜测字段。")
    return value


def positive_decimal(value):
    if isinstance(value, bool) or value is None:
        raise ValueError("份额数值缺失或类型错误。")
    try:
        number = Decimal(str(value).strip().replace(",", ""))
    except InvalidOperation as error:
        raise ValueError("份额数值不能解析。") from error
    if not number.is_finite() or number <= 0:
        raise ValueError("份额必须是有限正数，未知不能填0。")
    return number


def normalize(payload, captured_at, expected_date, sessions, mode):
    if mode not in ("SEED", "OBSERVE"):
        raise ValueError("未知观察模式。")
    if captured_at.tzinfo is None:
        raise ValueError("实际取得时刻必须带时区。")
    rows = [r for r in payload["result"] if isinstance(r, dict) and str(r.get("SEC_CODE", "")) == TARGET]
    if len(rows) != 1:
        return {"status": "NO_VIEW_TARGET_ROW_MISSING_OR_DUPLICATED", "target_rows": len(rows), "mode": mode}
    row = rows[0]
    if "TOT_VOL" not in row or "STAT_DATE" not in row:
        return {"status": "NO_VIEW_REQUIRED_SHARE_FIELDS_MISSING", "mode": mode}
    stats = date.fromisoformat(str(row["STAT_DATE"]))
    if stats not in sessions:
        return {"status": "NO_VIEW_STATISTICS_DATE_NOT_IN_FROZEN_EXCHANGE_CALENDAR", "statistics_date": stats.isoformat(), "mode": mode}
    if stats > captured_at.astimezone(ZONE).date():
        raise ValueError("统计日来自未来，不能登记。")
    units_10k = positive_decimal(row["TOT_VOL"])
    if mode == "OBSERVE" and (expected_date is None or stats != expected_date):
        return {"status": "NO_VIEW_STALE_OR_DIFFERENT_REQUESTED_STATISTICS_DATE", "statistics_date": stats.isoformat(), "mode": mode}
    if mode == "OBSERVE" and stats < EARLIEST:
        raise ValueError("冻结日前历史份额不能伪登记为新观察。")
    if mode == "OBSERVE" and captured_at < datetime.combine(stats, time(16), ZONE):
        return {"status": "NO_VIEW_BEFORE_STATS_SESSION_CLOSE", "statistics_date": stats.isoformat(), "mode": mode}
    return {"status": "BASELINE_SEED_NOT_PROSPECTIVE" if mode == "SEED" else "NEW_OFFICIAL_OBSERVATION",
            "mode": mode, "security": "510300.SH", "source_code": TARGET, "statistics_date": stats.isoformat(),
            "reported_total_units_10k": str(units_10k), "total_units": str(units_10k * 10000),
            "original_numeric_field": "TOT_VOL", "original_unit": "万份", "normalized_unit": "份",
            "available_at": captured_at.isoformat(), "availability_basis": "ACTUALLY_CAPTURED_OFFICIAL_RESPONSE_VERSION",
            "historical_first_publication": "NOT_ESTABLISHED", "first_publication_is_not_capture_time": True,
            "split_adjustment_status": "NOT_ESTABLISHED", "trade_signal": "NOT_GENERATED",
            "cash_flow_cny": "NOT_IDENTIFIED", "financial_admission": "NOT_ADMITTED"}


def daily_change(current, previous, decision_time, sessions):
    if decision_time.tzinfo is None:
        raise ValueError("使用时点必须带时区。")
    if any(x.get("status") != "NEW_OFFICIAL_OBSERVATION" for x in (current, previous)):
        return {"status": "NOT_COMPUTED_NEEDS_TWO_NEW_OBSERVED_VERSIONS", "fractional_unit_change": None}
    if any(x.get("security") != "510300.SH" or x.get("normalized_unit") != "份" for x in (current, previous)):
        raise ValueError("标的或单位不一致。")
    d, prior = date.fromisoformat(current["statistics_date"]), date.fromisoformat(previous["statistics_date"])
    if d not in sessions or prior not in sessions or sessions.index(d) == 0 or sessions[sessions.index(d) - 1] != prior:
        return {"status": "NOT_COMPUTED_NONCONSECUTIVE_EXCHANGE_DATES", "fractional_unit_change": None}
    if any(datetime.fromisoformat(x["available_at"]) > decision_time for x in (current, previous)):
        return {"status": "NOT_COMPUTED_VERSION_NOT_YET_KNOWN", "fractional_unit_change": None}
    a, b = positive_decimal(current["total_units"]), positive_decimal(previous["total_units"])
    return {"status": "COMPUTED_NEW_VERSION_PAIR_OBSERVATION_ONLY", "statistics_date": d.isoformat(),
            "previous_statistics_date": prior.isoformat(), "unit_change": str(a - b),
            "fractional_unit_change": str(a / b - 1), "cash_flow_cny": "NOT_IDENTIFIED",
            "split_adjustment_status": "NOT_ESTABLISHED", "interpretation": "RAW_UNIT_CHANGE_NOT_CONFIRMED_NET_CREATION",
            "trade_signal": "NOT_GENERATED", "financial_admission": "NOT_ADMITTED"}


def freeze():
    source = read(OUT / "source_protocol.json")
    tests = read(OUT / "observer_tests_receipt.json")
    if tests["exit_code"] != 0 or tests["passed"] != 10 or tests["observer_sha256"] != digest(Path(__file__)):
        raise ValueError("十项必要观察时序/单位/标的测试未通过或代码版本不同。")
    sessions = calendar()
    if EARLIEST not in sessions:
        raise ValueError("首个新交易日不在官方日历。")
    write(OUT / "daily_observer_protocol.json", {
        "at": stamp().isoformat(), "role": "DAILY_OFFICIAL_SHARE_VERSIONS_ONLY_NOT_TRADING_POLICY",
        "source_registration": source["registration"], "endpoint": BASE_URL, "sql_id": SQL_ID,
        "request_params": PARAMS, "target": "510300.SH", "first_future_statistics_date": EARLIEST.isoformat(),
        "source_unit": "万份", "normalized_unit": "份", "availability": "实际完成下载并保存的时刻，不能回填首次公布。",
        "clock_scope": "首页SCALE总规模23:00不移接到单产品TOT_VOL；清算后份额不用于当天16:00决策。",
        "baseline": "一次空日期当前查询；所有基线统计日均不计新前瞻，不与新版本混算份额变化。",
        "observe": "只查询已实际结束且在官方日历的当前新交易日；一个日一次，旧日期/失败/空值不补跑、不历史批量查询。",
        "daily_change": "两份相邻官方交易日新版本均在使用时点前实际取得，才计算份额增量；缺日不插值，数值不当人民币净流入。",
        "no_automatic_collection": True, "broker_orders": False, "trade_signal": "NOT_GENERATED", "financial_admission": "NOT_ADMITTED",
        "calendar_path": str(CALENDAR.absolute().relative_to(ROOT)), "calendar_sha256": digest(CALENDAR),
        "calendar_start": sessions[0].isoformat(), "calendar_end": sessions[-1].isoformat(),
        "observer_sha256": digest(Path(__file__)), "tests_sha256": tests["tests_sha256"],
        "original_forward_fields_unchanged": True, "new_fits": 0, "new_labels": 0, "new_accounts": 0, "new_financial_runs": 0})
    print("独立日频份额观察合同已冻结：真实采集钟、基线排除、相邻新版本、0交易信号。", flush=True)


def capture(mode, expected_date=None):
    protocol = read(OUT / "daily_observer_protocol.json")
    if digest(Path(__file__)) != protocol["observer_sha256"] or digest(CALENDAR) != protocol["calendar_sha256"]:
        raise ValueError("冻结后的观察器或日历发生变化。")
    current = stamp()
    sessions = calendar()
    if mode == "OBSERVE":
        if expected_date is None or expected_date not in sessions or expected_date < EARLIEST:
            raise ValueError("请求日不是允许的官方新交易日。")
        if current < datetime.combine(expected_date, time(23, 5), ZONE) or current.date() != expected_date:
            raise ValueError("仅在该实际统计日23:05之后观察；不会等待、提前、补采或按工作日猜测。")
    folder = OUT / ("seed_snapshot" if mode == "SEED" else f"daily_observations/{expected_date.isoformat()}")
    folder.mkdir(parents=True, exist_ok=False)
    params = {**PARAMS, "STAT_DATE": "" if mode == "SEED" else expected_date.isoformat()}
    receipt = {"started_at": current.isoformat(), "mode": mode, "requested_statistics_date": params["STAT_DATE"],
               "url": BASE_URL, "params": params, "requests": 1, "retries": 0}
    write(folder / "request_started.json", receipt)
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"Referer": "https://www.sse.com.cn/", "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        response = session.get(BASE_URL, params=params, timeout=(10, 25))
        receipt.update(http_status=response.status_code, final_url=response.url,
                       response_headers={k: response.headers.get(k) for k in ["Content-Type", "Date", "Last-Modified"]})
        if len(response.content) > 10_000_000:
            raise ValueError("官方响应超过本次10MB上限。")
        with (folder / "response.body").open("xb") as stream:
            stream.write(response.content)
        receipt["raw_response_sha256"] = digest(folder / "response.body")
        response.raise_for_status()
        payload = decode_response(response.content)
        captured = stamp()
        normalized = normalize(payload, captured, expected_date, sessions, mode)
        receipt.update(status="SAVED_OFFICIAL_RESPONSE", captured_at=captured.isoformat(), source_rows=len(payload["result"]))
        write(folder / "normalized.json", normalized)
    except Exception as error:
        receipt.update(status="SOURCE_OR_NORMALIZATION_FAILED", error_type=type(error).__name__, error=str(error))
        normalized = {"status": "NO_VIEW_SOURCE_OR_NORMALIZATION_FAILED", "mode": mode, "financial_admission": "NOT_ADMITTED"}
        write(folder / "normalized.json", normalized)
    finally:
        session.close()
        receipt["ended_at"] = stamp().isoformat()
        write(folder / "receipt.json", receipt)
    print(f"{mode}：{receipt['status']}，数据状态{normalized['status']}；没有生成交易信号。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="510300官方日频份额版本观察")
    parser.add_argument("action", choices=("freeze", "seed", "observe"))
    parser.add_argument("--date", type=date.fromisoformat, help="实际当前统计交易日，YYYY-MM-DD")
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    else:
        capture("SEED" if args.action == "seed" else "OBSERVE", args.date)


if __name__ == "__main__":
    main()
