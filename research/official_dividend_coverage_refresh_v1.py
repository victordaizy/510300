"""用实际访问的两处官方记录更新分红覆盖凭证，不改变分红事件或模型。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import uuid
from datetime import date, datetime, time
from decimal import Decimal
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import requests
import yaml
from bs4 import BeautifulSoup

from scripts.priority_forward_data_paths_v1 import WorkspacePaths, APPROVED_DATA_ROOT

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "config/510300_official_dividend_coverage_refresh_v1.json"
MANIFEST = "config/510300_official_dividend_coverage_refresh_v1_manifest.json"
TZ = ZoneInfo("Asia/Shanghai")


class CoverageError(RuntimeError):
    """资料不足时不更新正式覆盖日期。"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(root: Path, path: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def verify_runtime(root: Path, expected_hash: str) -> dict:
    paths = WorkspacePaths(root, APPROVED_DATA_ROOT)
    paths.validate_data_junction()
    manifest_path = paths.checked(MANIFEST)
    if not re.fullmatch("[a-f0-9]{64}", expected_hash) or sha(manifest_path) != expected_hash:
        raise CoverageError("分红维护清单哈希不一致")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest["files"]:
        actual = paths.checked(row["path"])
        if sha(actual) != row["sha256"] or actual.stat().st_size != row["bytes"]:
            raise CoverageError("分红维护依赖文件与冻结记录不一致：" + row["path"])
    config = json.loads((root / CONFIG).read_text(encoding="utf-8"))
    if config["data_purchase_budget_cny"] != 0 or config["new_dividend_auto_append"] is not False:
        raise CoverageError("只允许免费核对已登记分红")
    observer = json.loads((root / config["observer_manifest"]).read_text(encoding="utf-8"))
    for row in observer["identities"]:
        if sha(paths.checked(row["path"])) != row["sha256"]:
            raise CoverageError("B1冻结身份发生变化：" + row["path"])
    return config


def calendar_dates(path: Path) -> list[date]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        values = [date.fromisoformat(row["trade_date"][:10]) for row in csv.DictReader(handle)]
    if values != sorted(set(values)) or not values:
        raise CoverageError("官方交易日历无效")
    return values


def completed_trade_day(values: list[date], moment: datetime) -> date:
    local = moment.astimezone(TZ)
    if local.date() > values[-1]:
        raise CoverageError("官方交易日历已到期")
    eligible = [day for day in values if day < local.date() or day == local.date() and local.time() >= time(15)]
    if not eligible:
        raise CoverageError("没有已收盘的可覆盖交易日")
    return eligible[-1]


def is_origin_window(root: Path, config: dict, moment: datetime) -> bool:
    from research import b1_dsv5_forecast_observatory_v1 as observer

    frozen = yaml.safe_load((root / config["observer_config"]).read_text(encoding="utf-8"))
    grid = observer.origins(observer.calendar(root, frozen), frozen)
    local = moment.astimezone(TZ)
    return local.date() in {item.date() for item in grid} and time(19, 30) <= local.time() <= time(23, 59, 59)


def ledger_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        result = list(csv.DictReader(handle))
    if not result or any(row["symbol"] != "510300.SH" for row in result):
        raise CoverageError("冻结分红账本的基金身份无效")
    return result


def parse_manager(content: bytes, fund_code: str) -> list[dict]:
    text = content.decode("utf-8", errors="strict")
    if not re.search(r"project\s*=\s*seekProject\(['\"]" + re.escape(fund_code) + r"['\"]\)", text):
        raise CoverageError("基金产品页身份不匹配")
    soup = BeautifulSoup(text, "html.parser")
    bodies = soup.select("#fhTbody")
    if len(bodies) != 1 or not soup.select_one("#fhxx"):
        raise CoverageError("缺少唯一的官网完整分红表")
    section = soup.select_one("#fhxx")
    heading = section.get_text(" ", strip=True)
    for required in ("基金分红记录", "权益登记日", "红利发放日/再投资日", "每10份基金份额分红（元）"):
        if required not in heading:
            raise CoverageError("官网分红表口径发生变化：" + required)
    if '$("#fhTbody>tr").length' not in text or "showFhTable(1,allPage,10" not in text:
        raise CoverageError("无法确认官网分页使用已经完整返回的分红行")
    rows = []
    for tr in bodies[0].find_all("tr"):
        values = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if len(values) != 4:
            raise CoverageError("官网分红记录字段数不符")
        record = date.fromisoformat(values[1])
        payment = date.fromisoformat(values[2])
        amount = Decimal(values[3]) / 10
        if not amount.is_finite() or amount <= 0 or int(values[0]) != record.year or record > payment:
            raise CoverageError("官网分红记录的日期或金额无效")
        rows.append({"record_date": record.isoformat(), "payment_date": payment.isoformat(), "cash_dividend_per_share": str(amount)})
    if not rows or len({row["record_date"] for row in rows}) != len(rows):
        raise CoverageError("官网分红表为空或有重复登记日")
    return rows


def compare_manager(rows: list[dict], baseline: list[dict]) -> None:
    def normalize(items):
        return sorted((row["record_date"], row["payment_date"], Decimal(row["cash_dividend_per_share"])) for row in items)
    if normalize(rows) != normalize(baseline):
        raise CoverageError("NO_VIEW_OFFICIAL_DIVIDEND_EVENTS_REQUIRE_RECONCILIATION")


def validate_announcements(pages: list[dict], config: dict, end_date: date, baseline: dict) -> dict:
    if not pages:
        raise CoverageError("缺少上交所公告查询结果")
    first = pages[0]["pageHelp"]
    total, count = int(first["total"]), int(first["pageCount"])
    if total <= 0 or count != math.ceil(total / config["page_size"]) or len(pages) != count:
        raise CoverageError("公告分页没有完整取得，不能据此证明没有新增分红")
    rows = []
    for number, page in enumerate(pages, 1):
        info = page["pageHelp"]
        if page.get("actionErrors") or page.get("fieldErrors") or page.get("sqlId") != config["sse_sql_id"]:
            raise CoverageError("公告查询有业务错误或身份不符")
        if int(info["pageNo"]) != number or int(info["total"]) != total or int(info["pageCount"]) != count or int(info["pageSize"]) != config["page_size"]:
            raise CoverageError("公告分页在取数中发生变化")
        part = page.get("result")
        if not isinstance(part, list):
            raise CoverageError("公告结果不是记录数组")
        expected_size = min(config["page_size"], total - (number - 1) * config["page_size"])
        if len(part) != expected_size:
            raise CoverageError("公告页实际行数与分页总量不一致")
        rows.extend(part)
    seen = set()
    dividends = []
    for row in rows:
        day = date.fromisoformat(row["SSEDATE"][:10])
        url = urljoin("https://www.sse.com.cn/", row["URL"])
        if row["SECURITY_CODE"] != config["fund_code"] or not date.fromisoformat(config["sse_query_start"]) <= day <= end_date:
            raise CoverageError("公告查询未遵守基金代码或日期范围")
        if (day, url) in seen:
            raise CoverageError("公告分页重复返回记录")
        seen.add((day, url))
        title = row["TITLE"]
        if any(word in title for word in ("分红", "收益分配", "折算", "拆分", "份额合并")):
            if url not in baseline["official_sources"]:
                raise CoverageError("NO_VIEW_NEW_OFFICIAL_CORPORATE_ACTION_REQUIRES_RECONCILIATION")
            dividends.append({"date": day.isoformat(), "title": title, "url": url})
    if config["required_known_dividend_url"] not in {row["url"] for row in dividends}:
        raise CoverageError("公告查询没有返回已知分红锚点，不能确认覆盖完整")
    return {"total_announcements": total, "pages": count, "known_dividend_announcements": dividends, "new_unreconciled_dividend_announcements": 0}


def check_http_clock(headers: dict, retrieved_at: datetime, covered_day: date, config: dict) -> None:
    value = headers.get("Date") or headers.get("date")
    if not value:
        raise CoverageError("官方响应没有可核对的服务器日期")
    server = parsedate_to_datetime(value).astimezone(TZ)
    close = datetime.combine(covered_day, time(15), tzinfo=TZ)
    if server < close or (server - retrieved_at).total_seconds() > config["maximum_server_clock_ahead_seconds"]:
        raise CoverageError("官方响应时钟不能支持本次覆盖日期")


def fetch(root: Path, directory: Path, name: str, url: str, params: dict | None, covered_day: date, config: dict) -> tuple[bytes, dict]:
    with requests.Session() as session:
        session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": config["sse_page_url"] if "sse.com.cn" in url else "https://www.huatai-pb.com/"})
        response = session.get(url, params=params, timeout=(8, config["request_timeout_seconds"]), verify=True)
    retrieved = datetime.now(TZ)
    if len(response.content) > config["maximum_response_bytes"]:
        raise CoverageError("官方响应超出维护体积上限")
    digest = hashlib.sha256(response.content).hexdigest()
    path = directory / f"{name}_{digest}.raw"
    with path.open("xb") as handle:
        handle.write(response.content)
    receipt = {
        "url": url, "request_params": params, "final_url": response.url,
        "retrieved_at": retrieved.isoformat(), "http_status": response.status_code,
        "server_date": response.headers.get("Date"), "tls_verified": True,
        "source_file": identity(root, path), "access_cost_cny": 0,
    }
    save_json(directory / f"{name}.receipt.json", receipt)
    response.raise_for_status()
    if response.url.split("?", 1)[0] != url:
        raise CoverageError("官方请求发生了未登记的重定向")
    check_http_clock(dict(response.headers), retrieved, covered_day, config)
    return response.content, receipt


def refresh(root: Path, config: dict, apply: bool) -> dict:
    moment = datetime.now(TZ)
    covered = completed_trade_day(calendar_dates(root / config["calendar"]), moment)
    paths = WorkspacePaths(root, APPROVED_DATA_ROOT)
    live_path = paths.checked(config["live_coverage"])
    previous_bytes = live_path.read_bytes()
    previous_hash = hashlib.sha256(previous_bytes).hexdigest()
    previous = json.loads(previous_bytes)
    if date.fromisoformat(previous["coverage_end"]) > covered:
        raise CoverageError("现有覆盖日期已经更晚，不允许回退")
    baseline = json.loads((root / config["baseline_coverage"]).read_text(encoding="utf-8"))
    if sha(root / config["live_dividends"]) != baseline["distribution_file_sha256"]:
        raise CoverageError("已登记分红事件发生变化，本维护版本不能自动接纳")
    for record in baseline["official_source_snapshots"]:
        if sha(paths.checked(record["saved_file"])) != record["sha256"]:
            raise CoverageError("原官方分红证据字节不一致")
    stamp = moment.strftime("%Y%m%dT%H%M%S%f") + "_" + uuid.uuid4().hex[:8]
    directory = paths.checked(Path(config["output_root"]) / stamp)
    directory.mkdir(parents=True, exist_ok=False)
    save_json(directory / "request.json", {"started_at": moment.isoformat(), "requested_coverage_end": covered.isoformat(), "previous_coverage_sha256": previous_hash, "apply_requested": apply, "forecast_generation": False})
    applied = False
    try:
        body, manager_receipt = fetch(root, directory, "manager", config["manager_url"], None, covered, config)
        manager_rows = parse_manager(body, config["fund_code"])
        compare_manager(manager_rows, ledger_rows(root / config["baseline_dividends"]))
        pages, source_receipts = [], [manager_receipt]
        expected_pages = 1
        end = moment.date()
        for number in range(1, config["maximum_pages"] + 1):
            params = {
                "isPagination": "true", "pageHelp.pageSize": config["page_size"], "pageHelp.pageNo": number,
                "pageHelp.beginPage": number, "pageHelp.cacheSize": 1, "pageHelp.endPage": number,
                "type": "inParams", "sqlId": config["sse_sql_id"], "TITLE": "", "SECURITY_CODE": config["fund_code"],
                "BULLETIN_TYPE": "", "START_DATE": config["sse_query_start"], "END_DATE": end.isoformat(),
                "DATE_DESC": 1, "DATE_ASC": "", "CODE_DESC": "", "CODE_ASC": "",
            }
            raw, receipt = fetch(root, directory, f"sse_page_{number}", config["sse_query_url"], params, covered, config)
            value = json.loads(raw)
            pages.append(value)
            source_receipts.append(receipt)
            if number == 1:
                expected_pages = int(value["pageHelp"]["pageCount"])
                if expected_pages < 1 or expected_pages > config["maximum_pages"]:
                    raise CoverageError("公告查询分页总数不在已登记范围")
            if number == expected_pages:
                break
        checked = validate_announcements(pages, config, end, baseline)
        candidate = dict(baseline)
        candidate.update({
            "coverage_end": covered.isoformat(), "retrieved_at": datetime.now(TZ).isoformat(),
            "coverage_extension_method": config["coverage_rule"],
            "coverage_is_available_information_as_of_retrieval": True,
            "coverage_refresh_evidence": {"manager_event_count": len(manager_rows), **checked},
            "previous_coverage_receipt": {"saved_file": (directory / "previous_coverage.json").relative_to(root).as_posix(), "sha256": previous_hash},
            "event_ledger_changed": False, "source_cost_cny": 0,
        })
        candidate["official_sources"] = list(baseline["official_sources"]) + [config["manager_url"], config["sse_query_url"]]
        candidate["secondary_cross_check"] = {
            **baseline["secondary_cross_check"],
            "carried_forward_from_baseline": True,
            "original_retrieved_at": baseline["retrieved_at"],
            "refetched_in_this_refresh": False,
        }
        candidate["official_source_snapshots"] = list(baseline["official_source_snapshots"]) + [
            {"role": "COVERAGE_REGISTER_SNAPSHOT", "url": receipt["url"], "request_params": receipt["request_params"], "retrieved_at": receipt["retrieved_at"], "server_date": receipt["server_date"], "saved_file": receipt["source_file"]["path"], "sha256": receipt["source_file"]["sha256"], "bytes": receipt["source_file"]["bytes"]}
            for receipt in source_receipts
        ]
        candidate_bytes = (json.dumps(candidate, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        (directory / "previous_coverage.json").write_bytes(previous_bytes)
        (directory / "candidate_coverage.json").write_bytes(candidate_bytes)
        save_json(directory / "reconciliation.json", {"manager_rows": manager_rows, "sse_query": checked, "old_event_ledger_preserved": True})
        if apply:
            if sha(live_path) != previous_hash:
                raise CoverageError("覆盖凭证被其他进程更新，本次不覆盖")
            temporary = live_path.with_name(live_path.name + "." + uuid.uuid4().hex + ".tmp")
            with temporary.open("xb") as handle:
                handle.write(candidate_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, live_path)
            applied = True
        result = {
            "status": "PASS_OFFICIAL_COVERAGE_REFRESH_APPLIED" if apply else "PASS_OFFICIAL_COVERAGE_CANDIDATE_ONLY",
            "completed_at": datetime.now(TZ).isoformat(), "coverage_end": covered.isoformat(),
            "previous_coverage_end": previous["coverage_end"], "event_count": len(manager_rows),
            "announcement_count": checked["total_announcements"], "announcement_pages": checked["pages"],
            "live_coverage_updated": apply, "event_ledger_changed": False,
            "candidate_coverage_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
            "evidence_directory": directory.relative_to(root).as_posix(),
            "new_forecasts_generated": 0, "data_purchase_budget_cny": 0, "position_impact": 0,
        }
        save_json(directory / "result.json", result)
        return result
    except Exception as error:
        save_json(directory / "result.json", {"status": "COVERAGE_APPLIED_POST_WRITE_RECORD_FAILURE" if applied else "NO_VIEW_OFFICIAL_COVERAGE_NOT_UPDATED", "error_type": type(error).__name__, "error": str(error) if isinstance(error, CoverageError) else "官方来源访问、格式或文件处理失败", "live_coverage_updated": applied, "completed_at": datetime.now(TZ).isoformat()})
        raise
