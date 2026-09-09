from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
TIMEZONE = ZoneInfo("Asia/Shanghai")
AS_OF_DATE = date(2026, 8, 24)

FINANCIAL_ARCHIVE = (
    ROOT / "data/raw/fundamentals/csi300_financials_point_in_time_extended_v2.parquet"
)
OUTPUT_SAMPLE = (
    ROOT / "data/audit/a_share_hs_cninfo_periodic_report_feasibility_v1_sample.csv"
)
OUTPUT_JSON = (
    ROOT / "reports/audit/A_SHARE_HS_CNINFO_PERIODIC_REPORT_FEASIBILITY_V1.json"
)
OUTPUT_MARKDOWN = (
    ROOT / "reports/audit/A_SHARE_HS_CNINFO_PERIODIC_REPORT_FEASIBILITY_V1.md"
)

STOCK_MASTER_URL = "https://www.cninfo.com.cn/new/data/szse_stock.json"
QUERY_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
SEARCH_PAGE_URL = (
    "https://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search"
)
PDF_BASE_URL = "https://static.cninfo.com.cn/"
REPORT_CATEGORIES = (
    "category_ndbg_szsh;"
    "category_bndbg_szsh;"
    "category_yjdbg_szsh;"
    "category_sjdbg_szsh;"
)

TITLE_EXCLUSIONS = (
    "摘要",
    "取消",
    "更正",
    "修订",
    "更新后",
    "英文",
    "英文版",
    "提示性公告",
)


@dataclass(frozen=True)
class QueryReceipt:
    symbol: str
    org_id: str
    page_count: int
    raw_record_count: int
    total_announcement_declared: int | None
    completed: bool
    error: str | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/140 Safari/537.36"
            ),
            "Referer": SEARCH_PAGE_URL,
            "Origin": "https://www.cninfo.com.cn",
            "Accept": "application/json, text/plain, */*",
            "X-Requested-With": "XMLHttpRequest",
        }
    )
    return session


def _request_json(
    session: requests.Session,
    method: str,
    url: str,
    *,
    timeout_seconds: int,
    maximum_attempts: int,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    final_error: Exception | None = None
    for attempt in range(1, maximum_attempts + 1):
        try:
            response = session.request(
                method,
                url,
                data=data,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            return json.loads(response.content.decode("utf-8"))
        except Exception as error:  # noqa: BLE001 - 需要把网络失败写入审计证据
            final_error = error
            if attempt < maximum_attempts:
                time.sleep(float(attempt))
    raise RuntimeError(f"请求失败：{url}；{final_error}")


def load_stock_master(session: requests.Session) -> dict[str, dict[str, Any]]:
    payload = _request_json(
        session,
        "GET",
        STOCK_MASTER_URL,
        timeout_seconds=30,
        maximum_attempts=4,
    )
    records = payload.get("stockList") or []
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        code = str(record.get("code") or "").zfill(6)
        org_id = str(record.get("orgId") or "")
        if re.fullmatch(r"\d{6}", code) and org_id:
            result[code] = record
    if len(result) < 5_000:
        raise RuntimeError(f"巨潮股票主表记录异常偏少：{len(result)}")
    return result


def deterministic_symbol_sample(
    financials: pd.DataFrame,
    *,
    per_exchange: int,
) -> list[str]:
    codes = sorted(financials["con_code"].dropna().astype(str).unique().tolist())
    selected: list[str] = []
    for suffix in (".SH", ".SZ"):
        candidates = [code for code in codes if code.endswith(suffix)]
        ranked = sorted(
            candidates,
            key=lambda value: hashlib.sha256(
                f"CNINFO_PERIODIC_REPORT_FEASIBILITY_V1|{value}".encode("utf-8")
            ).hexdigest(),
        )
        selected.extend(ranked[:per_exchange])
    if len(selected) != per_exchange * 2:
        raise RuntimeError("无法构造固定数量的沪深分层样本")
    return selected


def query_symbol_announcements(
    session: requests.Session,
    *,
    symbol: str,
    org_id: str,
    start_date: date,
    end_date: date,
    maximum_pages: int = 30,
) -> tuple[list[dict[str, Any]], QueryReceipt]:
    code = symbol.split(".", maxsplit=1)[0]
    records: list[dict[str, Any]] = []
    declared_total: int | None = None
    try:
        for page_number in range(1, maximum_pages + 1):
            body = {
                "pageNum": page_number,
                "pageSize": 30,
                "column": "szse",
                "tabName": "fulltext",
                "plate": "",
                "stock": f"{code},{org_id}",
                "searchkey": "",
                "secid": "",
                "category": REPORT_CATEGORIES,
                "trade": "",
                "seDate": f"{start_date.isoformat()}~{end_date.isoformat()}",
                "sortName": "",
                "sortType": "",
                "isHLtitle": "true",
            }
            payload = _request_json(
                session,
                "POST",
                QUERY_URL,
                timeout_seconds=90,
                maximum_attempts=4,
                data=body,
            )
            if declared_total is None and payload.get("totalAnnouncement") is not None:
                declared_total = int(payload["totalAnnouncement"])
            page_records = payload.get("announcements") or []
            for record in page_records:
                normalized = dict(record)
                normalized["query_symbol"] = symbol
                normalized["query_org_id"] = org_id
                records.append(normalized)
            if not bool(payload.get("hasMore")):
                return records, QueryReceipt(
                    symbol=symbol,
                    org_id=org_id,
                    page_count=page_number,
                    raw_record_count=len(records),
                    total_announcement_declared=declared_total,
                    completed=True,
                    error=None,
                )
        raise RuntimeError(f"翻页超过上限{maximum_pages}页")
    except Exception as error:  # noqa: BLE001 - 审计必须保留逐证券失败原因
        return records, QueryReceipt(
            symbol=symbol,
            org_id=org_id,
            page_count=max(0, (len(records) + 29) // 30),
            raw_record_count=len(records),
            total_announcement_declared=declared_total,
            completed=False,
            error=str(error),
        )


def normalize_title(value: Any) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", "", str(value or "")))
    return re.sub(r"[\s\u3000]+", "", text).replace("－", "-")


def classify_periodic_report(title: str) -> tuple[str | None, pd.Timestamp | None]:
    normalized = normalize_title(title)
    if any(token in normalized for token in TITLE_EXCLUSIONS):
        return None, None
    year_match = re.search(r"(?<!\d)(20\d{2})年", normalized)
    if year_match is None:
        return None, None
    report_year = int(year_match.group(1))
    if "第一季度报告" in normalized:
        return "Q1", pd.Timestamp(report_year, 3, 31)
    if "半年度报告" in normalized or "中期报告" in normalized:
        return "H1", pd.Timestamp(report_year, 6, 30)
    if "第三季度报告" in normalized:
        return "Q3", pd.Timestamp(report_year, 9, 30)
    if "年度报告" in normalized and "半年度报告" not in normalized:
        return "FY", pd.Timestamp(report_year, 12, 31)
    return None, None


def _url_publication_date(relative_url: Any) -> pd.Timestamp | None:
    match = re.search(r"finalpage/(\d{4}-\d{2}-\d{2})/", str(relative_url or ""))
    if match is None:
        return None
    return pd.Timestamp(match.group(1))


def _timestamp_publication_date(value: Any) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.to_datetime(int(value), unit="ms", utc=True).tz_convert(TIMEZONE)
    return pd.Timestamp(timestamp.date())


def normalize_official_records(
    records: Iterable[dict[str, Any]],
    *,
    classifier: Callable[[str], tuple[str | None, pd.Timestamp | None]] = classify_periodic_report,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        title = normalize_title(record.get("announcementTitle"))
        period_type, report_period = classifier(title)
        if period_type is None or report_period is None:
            continue
        relative_url = str(record.get("adjunctUrl") or "")
        rows.append(
            {
                "con_code": str(record["query_symbol"]),
                "org_id": str(record["query_org_id"]),
                "announcement_id": str(record.get("announcementId") or ""),
                "announcement_title": title,
                "period_type": period_type,
                "report_period": report_period,
                "announcement_time_ms": record.get("announcementTime"),
                "official_timestamp_date": _timestamp_publication_date(
                    record.get("announcementTime")
                ),
                "official_url_date": _url_publication_date(relative_url),
                "relative_pdf_url": relative_url,
                "official_pdf_url": urljoin(PDF_BASE_URL, relative_url),
                "adjunct_type": str(record.get("adjunctType") or ""),
                "adjunct_size_kb": record.get("adjunctSize"),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "con_code",
                "org_id",
                "announcement_id",
                "announcement_title",
                "period_type",
                "report_period",
                "announcement_time_ms",
                "official_timestamp_date",
                "official_url_date",
                "relative_pdf_url",
                "official_pdf_url",
                "adjunct_type",
                "adjunct_size_kb",
            ]
        )
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        ["con_code", "report_period", "official_url_date", "announcement_time_ms", "announcement_id"],
        kind="stable",
    )
    # 同一期旧格式可能同时存在“正文”和“全文”；事件日取官方最早完整披露，绝不挑选收益更优版本。
    return frame.drop_duplicates(["con_code", "report_period"], keep="first").reset_index(
        drop=True
    )


def verify_pdf_prefix(
    session: requests.Session,
    url: str,
    *,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    host = (urlparse(url).hostname or "").lower()
    result: dict[str, Any] = {
        "url": url,
        "official_host": host == "static.cninfo.com.cn",
        "status_code": None,
        "content_type": None,
        "pdf_header": False,
        "error": None,
    }
    try:
        with session.get(url, timeout=timeout_seconds, stream=True) as response:
            result["status_code"] = int(response.status_code)
            result["content_type"] = response.headers.get("Content-Type")
            response.raise_for_status()
            result["pdf_header"] = response.raw.read(5) == b"%PDF-"
    except Exception as error:  # noqa: BLE001 - 逐文件记录失败而非隐式跳过
        result["error"] = str(error)
    return result


def _serializable_record(record: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, (pd.Timestamp, datetime, date)):
            result[key] = value.isoformat()
        elif pd.isna(value) if not isinstance(value, (list, dict)) else False:
            result[key] = None
        else:
            result[key] = value
    return result


def build_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    gates = report["gates"]
    lines = [
        "# A股定期报告官方事件日期可行性审计 V1",
        "",
        f"- 状态：`{report['status']}`",
        f"- 审计时间：`{report['audited_at']}`",
        f"- 证据截止日：`{report['as_of_date']}`",
        f"- 样本：沪深各 {report['sampling']['per_exchange']} 只，共 {metrics['sample_symbol_count']} 只",
        "- 边界：本审计只核验官方事件日期与公告文件；没有读取、计算或配对任何未来收益。",
        "",
        "## 核心结果",
        "",
        f"- 巨潮查询成功证券：{metrics['completed_symbol_count']}/{metrics['sample_symbol_count']}",
        f"- 官方原始公告记录：{metrics['raw_announcement_count']}",
        f"- 去重后的原始完整定期报告事件：{metrics['official_event_count']}",
        f"- 本地供应商事件在官方记录中的覆盖率：{metrics['official_coverage_ratio']:.6%}",
        f"- 已匹配事件的公告日期完全一致率：{metrics['matched_date_agreement_ratio']:.6%}",
        f"- 官方时间戳日期与 PDF 路径日期一致率：{metrics['official_internal_date_agreement_ratio']:.6%}",
        f"- PDF 抽样核验通过：{metrics['pdf_verified_count']}/{metrics['pdf_verification_count']}",
        "",
        "## 验收门",
        "",
    ]
    for gate in gates:
        lines.append(
            f"- {'PASS' if gate['passed'] else 'FAIL'} `{gate['gate_id']}`：{gate['description']}"
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "- 若通过，只能说明可用巨潮官方发布日期构造事件日，并在其后的交易日形成信号。",
            "- 2026年统一回取的财务数值仍存在追溯修订风险，继续禁止作为严格PIT盈利意外值。",
            "- 本审计不构成 alpha、强 beta、可交易性或实盘有效性的证据。",
            "- 任何正式事件研究必须另行冻结唯一候选、成本、对照、样本期和停止规则。",
            "",
            "## 产物",
            "",
            f"- 样本明细：`{report['artifacts']['sample_csv']}`",
            f"- JSON报告：`{report['artifacts']['report_json']}`",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(
    *,
    per_exchange: int,
    classifier: Callable[
        [str], tuple[str | None, pd.Timestamp | None]
    ] = classify_periodic_report,
    output_sample: Path = OUTPUT_SAMPLE,
    output_json: Path = OUTPUT_JSON,
    output_markdown: Path = OUTPUT_MARKDOWN,
    audit_id: str = "A_SHARE_HS_CNINFO_PERIODIC_REPORT_FEASIBILITY_V1",
) -> dict[str, Any]:
    if not FINANCIAL_ARCHIVE.exists():
        raise FileNotFoundError(f"缺少财务事件索引：{FINANCIAL_ARCHIVE}")
    financial_hash_before = sha256_file(FINANCIAL_ARCHIVE)
    financials = pd.read_parquet(
        FINANCIAL_ARCHIVE,
        columns=["con_code", "report_period", "announcement_date"],
    )
    financials["report_period"] = pd.to_datetime(financials["report_period"]).dt.normalize()
    financials["announcement_date"] = pd.to_datetime(
        financials["announcement_date"]
    ).dt.normalize()
    symbols = deterministic_symbol_sample(financials, per_exchange=per_exchange)

    session = _session()
    # 先访问查询页，取得与浏览器同源的会话 Cookie。
    session.get(SEARCH_PAGE_URL, timeout=30).raise_for_status()
    master = load_stock_master(session)

    raw_records: list[dict[str, Any]] = []
    receipts: list[QueryReceipt] = []
    for symbol in symbols:
        code = symbol.split(".", maxsplit=1)[0]
        master_record = master.get(code)
        if master_record is None:
            receipts.append(
                QueryReceipt(
                    symbol=symbol,
                    org_id="",
                    page_count=0,
                    raw_record_count=0,
                    total_announcement_declared=None,
                    completed=False,
                    error="巨潮股票主表缺少该证券",
                )
            )
            continue
        records, receipt = query_symbol_announcements(
            session,
            symbol=symbol,
            org_id=str(master_record["orgId"]),
            start_date=date(2012, 1, 1),
            end_date=AS_OF_DATE,
        )
        raw_records.extend(records)
        receipts.append(receipt)

    official = normalize_official_records(raw_records, classifier=classifier)
    sample_vendor = (
        financials.loc[financials["con_code"].isin(symbols)]
        .dropna(subset=["report_period", "announcement_date"])
        .drop_duplicates(["con_code", "report_period"], keep="first")
        .rename(columns={"announcement_date": "vendor_announcement_date"})
    )
    comparison = sample_vendor.merge(
        official,
        on=["con_code", "report_period"],
        how="left",
        validate="one_to_one",
    )
    comparison["official_event_found"] = comparison["announcement_id"].notna()
    comparison["vendor_official_date_equal"] = (
        comparison["vendor_announcement_date"] == comparison["official_url_date"]
    ) & comparison["official_event_found"]
    comparison["official_internal_date_equal"] = (
        comparison["official_timestamp_date"] == comparison["official_url_date"]
    ) & comparison["official_event_found"]

    pdf_candidates = (
        official.sort_values(["con_code", "report_period"])
        .groupby("con_code", observed=True, group_keys=False)
        .apply(lambda group: group.iloc[[0, -1]] if len(group) > 1 else group, include_groups=False)
        .reset_index(drop=True)
    )
    pdf_checks = [
        verify_pdf_prefix(session, str(url))
        for url in pdf_candidates["official_pdf_url"].dropna().drop_duplicates().tolist()
    ]

    matched = comparison.loc[comparison["official_event_found"]]
    completed_count = sum(receipt.completed for receipt in receipts)
    official_coverage = (
        float(comparison["official_event_found"].mean()) if len(comparison) else 0.0
    )
    matched_date_agreement = (
        float(matched["vendor_official_date_equal"].mean()) if len(matched) else 0.0
    )
    internal_date_agreement = (
        float(matched["official_internal_date_equal"].mean()) if len(matched) else 0.0
    )
    pdf_verified_count = sum(
        bool(item["official_host"])
        and item["status_code"] == 200
        and bool(item["pdf_header"])
        for item in pdf_checks
    )

    gates = [
        {
            "gate_id": "G1_ALL_SYMBOL_QUERIES_COMPLETE",
            "description": "确定性样本全部完成官方查询",
            "passed": completed_count == len(symbols),
        },
        {
            "gate_id": "G2_OFFICIAL_EVENT_COVERAGE",
            "description": "供应商事件索引的官方完整报告覆盖率不低于95%",
            "passed": official_coverage >= 0.95,
        },
        {
            "gate_id": "G3_MATCHED_DATE_AGREEMENT",
            "description": "已匹配事件的供应商日期与官方PDF路径日期一致率不低于99%",
            "passed": matched_date_agreement >= 0.99,
        },
        {
            "gate_id": "G4_OFFICIAL_INTERNAL_DATE_AGREEMENT",
            "description": "官方时间戳日期与官方PDF路径日期一致率不低于99.9%",
            "passed": internal_date_agreement >= 0.999,
        },
        {
            "gate_id": "G5_PDF_SAMPLE_VERIFIED",
            "description": "每只样本证券最早和最晚官方PDF均可达且文件头有效",
            "passed": len(pdf_checks) > 0 and pdf_verified_count == len(pdf_checks),
        },
        {
            "gate_id": "G6_FROZEN_INPUT_UNCHANGED",
            "description": "审计前后本地财务事件索引哈希不变",
            "passed": financial_hash_before == sha256_file(FINANCIAL_ARCHIVE),
        },
    ]
    passed_count = sum(bool(gate["passed"]) for gate in gates)
    status = (
        "PASS_OFFICIAL_PERIODIC_REPORT_EVENT_DATE_FEASIBLE"
        if passed_count == len(gates)
        else "NO_VIEW_OFFICIAL_PERIODIC_REPORT_EVENT_DATE_FEASIBILITY_FAILED"
    )

    output_sample.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_sample, index=False, encoding="utf-8-sig")
    report: dict[str, Any] = {
        "audit_id": audit_id,
        "status": status,
        "audited_at": datetime.now(TIMEZONE).isoformat(),
        "as_of_date": AS_OF_DATE.isoformat(),
        "scope": {
            "purpose": "只核验巨潮官方定期报告事件日期是否可用于后续事件研究",
            "future_return_read": False,
            "signal_calculated": False,
            "alpha_claim_allowed": False,
            "financial_value_vintage_unlocked": False,
        },
        "sources": {
            "stock_master_url": STOCK_MASTER_URL,
            "announcement_query_url": QUERY_URL,
            "pdf_base_url": PDF_BASE_URL,
            "local_event_index": FINANCIAL_ARCHIVE.relative_to(ROOT).as_posix(),
            "local_event_index_sha256": financial_hash_before,
        },
        "sampling": {
            "method": "按沪深后缀分层，对固定盐加证券代码做SHA256排序后各取最小值",
            "per_exchange": per_exchange,
            "symbols": symbols,
        },
        "metrics": {
            "sample_symbol_count": len(symbols),
            "completed_symbol_count": completed_count,
            "raw_announcement_count": len(raw_records),
            "official_event_count": len(official),
            "vendor_event_count": len(comparison),
            "official_matched_event_count": int(comparison["official_event_found"].sum()),
            "official_coverage_ratio": official_coverage,
            "matched_date_agreement_count": int(
                matched["vendor_official_date_equal"].sum()
            ),
            "matched_date_agreement_ratio": matched_date_agreement,
            "official_internal_date_agreement_count": int(
                matched["official_internal_date_equal"].sum()
            ),
            "official_internal_date_agreement_ratio": internal_date_agreement,
            "pdf_verification_count": len(pdf_checks),
            "pdf_verified_count": pdf_verified_count,
        },
        "query_receipts": [asdict(receipt) for receipt in receipts],
        "pdf_checks": pdf_checks,
        "gates": gates,
        "passed_gate_count": passed_count,
        "total_gate_count": len(gates),
        "blocking_boundary": (
            "即使事件日期可用，2026年统一回取的财务数值仍不可声明为严格PIT；"
            "后续候选只能使用官方事件日与在该日之后真实观察到的信息。"
        ),
        "artifacts": {
            "sample_csv": output_sample.relative_to(ROOT).as_posix(),
            "sample_csv_sha256": sha256_file(output_sample),
            "report_json": output_json.relative_to(ROOT).as_posix(),
            "report_markdown": output_markdown.relative_to(ROOT).as_posix(),
        },
    }
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    output_markdown.write_text(build_markdown(report), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计巨潮定期报告官方事件日期可行性")
    parser.add_argument(
        "--per-exchange",
        type=int,
        default=6,
        help="沪深市场各抽取的证券数量，默认6",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.per_exchange < 1:
        raise ValueError("--per-exchange 必须大于等于1")
    report = run_audit(per_exchange=args.per_exchange)
    print(
        json.dumps(
            {
                "状态": report["status"],
                "通过门数": report["passed_gate_count"],
                "总门数": report["total_gate_count"],
                "样本证券数": report["metrics"]["sample_symbol_count"],
                "官方事件覆盖率": report["metrics"]["official_coverage_ratio"],
                "公告日期一致率": report["metrics"]["matched_date_agreement_ratio"],
                "报告": report["artifacts"]["report_markdown"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
