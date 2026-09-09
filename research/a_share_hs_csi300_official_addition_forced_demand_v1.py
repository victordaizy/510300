"""沪深300官方定期调入事件的无收益数据契约与交易机制纯函数。

本模块刻意不读取价格、不计算收益，也不生成任何曲线。历史收益评估只能在
D1-D8 全部通过、协议清单冻结后，由后续独立入口执行。
"""

from __future__ import annotations

import hashlib
import io
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import pandas as pd
from bs4 import BeautifulSoup


STRATEGY_ID = "A_SHARE_HS_CSI300_OFFICIAL_ADDITION_FORCED_DEMAND_V1"
INDEX_CODE = "000300"
ALLOWED_OFFICIAL_HOSTS = frozenset({"www.csindex.com.cn", "oss-ch.csindex.com.cn"})

EVENT_LEDGER_COLUMNS = (
    "strategy_id",
    "event_key",
    "cycle_id",
    "announcement_id",
    "announcement_title",
    "announcement_date",
    "entry_date",
    "effective_date",
    "security_code",
    "ts_code",
    "security_name",
    "exchange",
    "action",
    "official_sequence",
    "cycle_addition_count",
    "source_detail_url",
    "source_detail_sha256",
    "source_attachment_url",
    "source_attachment_sha256",
    "source_attachment_path",
    "source_format",
    "extraction_method",
    "candidate_state",
    "return_evaluation",
)

FORBIDDEN_RESULT_COLUMN_TOKENS = (
    "return",
    "收益",
    "pnl",
    "profit",
    "alpha",
    "excess",
    "drawdown",
    "sharpe",
    "open_price",
    "close_price",
    "entry_price",
    "exit_price",
    "price_change",
)


def sha256_bytes(payload: bytes) -> str:
    """返回字节串的 SHA-256。"""

    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """以流式方式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_security_code(value: Any) -> str | None:
    """将 Excel、HTML 或 PDF 中的 A 股代码规范为六位字符串。"""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    decimal_match = re.fullmatch(r"(\d{1,6})\.0+", text)
    if decimal_match:
        text = decimal_match.group(1)
    exact_match = re.fullmatch(r"\d{1,6}", text)
    if exact_match:
        return text.zfill(6)
    embedded_match = re.search(r"(?<!\d)(\d{6})(?!\d)", text)
    return embedded_match.group(1) if embedded_match else None


def infer_exchange(security_code: str) -> str:
    """按 A 股代码前缀映射交易所；不认识的代码立即拒绝。"""

    code = normalize_security_code(security_code)
    if code is None:
        raise ValueError(f"证券代码不可识别：{security_code!r}")
    if code.startswith(("5", "6", "9")):
        return "SH"
    if code.startswith(("0", "1", "2", "3")):
        return "SZ"
    if code.startswith(("4", "8")):
        return "BJ"
    raise ValueError(f"证券代码交易所不可识别：{code}")


def to_ts_code(security_code: str) -> str:
    code = normalize_security_code(security_code)
    if code is None:
        raise ValueError(f"证券代码不可识别：{security_code!r}")
    return f"{code}.{infer_exchange(code)}"


def _parse_iso_date(value: str | date | datetime | pd.Timestamp) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def extract_effective_date(content_html: str, publish_date: str | date) -> str:
    """从官方正文提取公告后 60 天内的明确生效日期。

    先移除空白以兼容被逐字分隔的日期，再仅接受公告日之后的日期。若正文
    同时出现多个候选，优先选择“生效/实施/调整”邻域中的日期；仍不唯一则
    拒绝自动推断。
    """

    publish = _parse_iso_date(publish_date)
    text = BeautifulSoup(content_html or "", "html.parser").get_text(" ", strip=True)
    compact = re.sub(r"\s+", "", text).replace("／", "/").replace("－", "-")
    matches: list[tuple[date, int, int]] = []

    for match in re.finditer(r"(20\d{2})[年\-/.](\d{1,2})[月\-/.](\d{1,2})日?", compact):
        try:
            matches.append((date(int(match.group(1)), int(match.group(2)), int(match.group(3))), match.start(), match.end()))
        except ValueError:
            continue

    for match in re.finditer(r"(?<!\d)(\d{1,2})月(\d{1,2})日", compact):
        try:
            candidate = date(publish.year, int(match.group(1)), int(match.group(2)))
            matches.append((candidate, match.start(), match.end()))
        except ValueError:
            continue

    future = []
    seen: set[date] = set()
    for candidate, start, end in matches:
        if publish < candidate <= publish + timedelta(days=60) and candidate not in seen:
            context = compact[max(0, start - 24) : min(len(compact), end + 24)]
            score = int(any(token in context for token in ("生效", "实施", "调整", "更换")))
            future.append((candidate, score))
            seen.add(candidate)

    if not future:
        raise ValueError(f"公告正文未找到公告日后 60 天内的明确生效日：{publish.isoformat()}")
    best_score = max(score for _, score in future)
    best = sorted(candidate for candidate, score in future if score == best_score)
    if len(best) != 1:
        raise ValueError(f"公告正文存在多个无法消歧的生效日：{[item.isoformat() for item in best]}")
    return best[0].isoformat()


def _normalise_official_url(url: str) -> str:
    cleaned = str(url).strip().replace("notice%2F", "notice/").replace("notice%2f", "notice/")
    parsed = urlparse(cleaned)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_OFFICIAL_HOSTS:
        raise ValueError(f"附件不是允许的中证官方 HTTPS 地址：{url}")
    return cleaned


def extract_official_attachment_urls(detail_payload: Mapping[str, Any]) -> list[dict[str, str]]:
    """合并结构化附件与正文链接，只保留中证官方 HTTPS 对象。"""

    candidates: list[tuple[str, str]] = []
    for item in detail_payload.get("enclosureList") or []:
        if isinstance(item, Mapping) and item.get("fileUrl"):
            candidates.append((str(item["fileUrl"]), str(item.get("fileName") or "")))

    soup = BeautifulSoup(str(detail_payload.get("content") or ""), "html.parser")
    for anchor in soup.find_all("a", href=True):
        candidates.append((str(anchor["href"]), anchor.get_text(" ", strip=True)))

    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw_url, raw_name in candidates:
        try:
            url = _normalise_official_url(raw_url)
        except ValueError:
            continue
        if url in seen:
            continue
        seen.add(url)
        decoded_name = Path(unquote(urlparse(url).path)).name
        result.append({"url": url, "file_name": raw_name.strip() or decoded_name})
    return result


def _deduplicate_changes(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        code = normalize_security_code(row.get("security_code"))
        if code is None or code in seen:
            continue
        seen.add(code)
        name = str(row.get("security_name") or "").strip()
        result.append({"security_code": code, "security_name": name})
    return result


def extract_html_csi300_changes(content_html: str) -> dict[str, Any] | None:
    """从 2018-2022 官方正文的首个沪深300调样表提取名单。"""

    soup = BeautifulSoup(content_html or "", "html.parser")
    for table_index, table in enumerate(soup.find_all("table")):
        additions: list[dict[str, str]] = []
        deletions: list[dict[str, str]] = []
        for row in table.find_all("tr"):
            cells = [re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).strip() for cell in row.find_all(["td", "th"])]
            if len(cells) < 4:
                continue
            deletion_code = normalize_security_code(cells[0])
            addition_code = normalize_security_code(cells[2])
            if deletion_code and addition_code:
                deletions.append({"security_code": deletion_code, "security_name": cells[1]})
                additions.append({"security_code": addition_code, "security_name": cells[3]})
        additions = _deduplicate_changes(additions)
        deletions = _deduplicate_changes(deletions)
        if len(additions) >= 2 and len(additions) == len(deletions):
            return {
                "additions": additions,
                "deletions": deletions,
                "method": f"OFFICIAL_DETAIL_HTML_TABLE_{table_index + 1}",
            }
    return None


def _rows_for_index_sheet(frame: pd.DataFrame) -> list[dict[str, str]]:
    """提取一个官方旧版工作表中的 000300 成分记录。"""

    rows: list[dict[str, str]] = []
    current_index: str | None = None
    for values in frame.itertuples(index=False, name=None):
        cells = list(values)
        codes = [normalize_security_code(value) for value in cells]
        if INDEX_CODE in codes:
            current_index = INDEX_CODE
        elif any(code is not None and code != INDEX_CODE for code in codes[:2]):
            current_index = None
        if current_index != INDEX_CODE:
            continue

        index_position = codes.index(INDEX_CODE) if INDEX_CODE in codes else -1
        security_position: int | None = None
        for position, code in enumerate(codes):
            if code and code != INDEX_CODE and (index_position < 0 or position > index_position):
                security_position = position
                break
        if security_position is None:
            continue
        security_code = codes[security_position]
        name = ""
        if security_position + 1 < len(cells) and not pd.isna(cells[security_position + 1]):
            name = str(cells[security_position + 1]).strip()
        rows.append({"security_code": str(security_code), "security_name": name})
    return _deduplicate_changes(rows)


def extract_excel_csi300_changes(payload: bytes, suffix: str) -> dict[str, Any]:
    """按冻结角色契约解析 2016-2017 官方工作簿。"""

    excel = pd.ExcelFile(io.BytesIO(payload))
    qualifying: list[tuple[str, list[dict[str, str]]]] = []
    for sheet_name in excel.sheet_names:
        frame = pd.read_excel(io.BytesIO(payload), sheet_name=sheet_name, header=None, dtype=object)
        rows = _rows_for_index_sheet(frame)
        if rows:
            qualifying.append((str(sheet_name), rows))
    if len(qualifying) < 2:
        raise ValueError(f"官方旧版工作簿中包含 000300 记录的工作表少于两个：{excel.sheet_names}")
    additions = qualifying[0][1]
    deletions = qualifying[1][1]
    if len(additions) != len(deletions):
        raise ValueError(f"官方旧版工作簿调入调出数量不一致：{len(additions)} != {len(deletions)}")
    return {
        "additions": additions,
        "deletions": deletions,
        "method": f"OFFICIAL_LEGACY_WORKBOOK_FIRST_TWO_000300_SHEETS_{suffix.upper().lstrip('.')}",
        "source_sheet_additions": qualifying[0][0],
        "source_sheet_deletions": qualifying[1][0],
    }


def extract_pdf_csi300_changes(payload: bytes) -> dict[str, Any]:
    """从官方 PDF 的沪深300首段双栏名单提取调入和调出代码。"""

    import pdfplumber

    with pdfplumber.open(io.BytesIO(payload)) as document:
        text = "\n".join(page.extract_text() or "" for page in document.pages)
    return extract_csi300_changes_from_pdf_text(text)


def extract_csi300_changes_from_pdf_text(text: str) -> dict[str, Any]:
    """解析 pdfplumber 提取的文本，同时保留证券代码之间的列间空白。"""

    start_match = re.search(r"沪\s*深\s*300\s*指数样本调整名单\s*[:：]?", text)
    if not start_match:
        raise ValueError("官方 PDF 中未找到沪深300指数样本调整名单标题")
    section = text[start_match.end() :]
    end_match = re.search(
        r"(?:中\s*证\s*(?:500|1000|A500)|上\s*证\s*\d+|深\s*证\s*\d+)\s*指数样本调整名单",
        section,
    )
    if end_match:
        section = section[: end_match.start()]

    additions: list[dict[str, str]] = []
    deletions: list[dict[str, str]] = []
    row_pattern = re.compile(r"^\s*(\d{6})\s+(.+?)\s+(\d{6})\s+(.+?)\s*$")
    for raw_line in section.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        match = row_pattern.match(line)
        if not match:
            continue
        deletions.append({"security_code": match.group(1), "security_name": match.group(2).strip()})
        additions.append({"security_code": match.group(3), "security_name": match.group(4).strip()})
    additions = _deduplicate_changes(additions)
    deletions = _deduplicate_changes(deletions)
    if len(additions) < 2 or len(additions) != len(deletions):
        raise ValueError(f"官方 PDF 沪深300名单解析失败或数量不一致：调入 {len(additions)}，调出 {len(deletions)}")
    return {"additions": additions, "deletions": deletions, "method": "OFFICIAL_PDF_CSI300_FIRST_SECTION"}


def parse_official_csi300_changes(
    detail_payload: Mapping[str, Any],
    attachments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """按 HTML、旧版工作簿、PDF 顺序解析，并返回明确来源。"""

    html_result = extract_html_csi300_changes(str(detail_payload.get("content") or ""))
    if html_result is not None:
        return {**html_result, "attachment": None, "source_format": "HTML"}

    errors: list[str] = []
    for attachment in attachments:
        suffix = str(attachment.get("suffix") or Path(urlparse(str(attachment.get("url") or "")).path).suffix).lower()
        payload = attachment.get("payload")
        if not isinstance(payload, bytes):
            errors.append(f"{attachment.get('url')}: 缺少字节内容")
            continue
        try:
            if suffix in {".xls", ".xlsx"}:
                result = extract_excel_csi300_changes(payload, suffix)
            elif suffix == ".pdf" or payload.startswith(b"%PDF"):
                result = extract_pdf_csi300_changes(payload)
            else:
                continue
            return {**result, "attachment": dict(attachment), "source_format": suffix.lstrip(".").upper()}
        except Exception as exc:  # noqa: BLE001 - 错误会汇总后拒绝该周期
            errors.append(f"{attachment.get('url')}: {type(exc).__name__}: {exc}")
    raise ValueError("无法从官方正文或附件解析沪深300名单；" + " | ".join(errors))


def load_open_trading_days(calendar_path: Path, expected_sha256: str | None = None) -> list[date]:
    """读取只用于事件时钟的交易日，并验证冻结哈希。"""

    if expected_sha256 and sha256_file(calendar_path) != expected_sha256:
        raise ValueError("事件时钟交易日文件 SHA-256 与冻结配置不一致")
    frame = pd.read_parquet(calendar_path, columns=["date", "is_open"])
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.date
    open_mask = frame["is_open"].astype(bool)
    days = sorted(set(frame.loc[open_mask, "date"].tolist()))
    if not days:
        raise ValueError("事件时钟交易日文件没有开放交易日")
    return days


def first_trading_day_after(announcement_date: str | date, open_days: Sequence[date]) -> str:
    announcement = _parse_iso_date(announcement_date)
    for trading_day in open_days:
        if trading_day > announcement:
            return trading_day.isoformat()
    raise ValueError(f"交易日历未覆盖公告日后的第一个交易日：{announcement.isoformat()}")


def build_event_ledger(
    cycle_records: Sequence[Mapping[str, Any]],
    open_days: Sequence[date],
) -> pd.DataFrame:
    """生成严格无价格、无收益的官方调入事件账本。"""

    rows: list[dict[str, Any]] = []
    for cycle in cycle_records:
        announcement_date = _parse_iso_date(cycle["announcement_date"]).isoformat()
        effective_date = _parse_iso_date(cycle["effective_date"]).isoformat()
        entry_date = first_trading_day_after(announcement_date, open_days)
        additions = list(cycle["additions"])
        attachment = cycle.get("attachment") or {}
        for sequence, addition in enumerate(additions, start=1):
            code = normalize_security_code(addition.get("security_code"))
            if code is None:
                raise ValueError(f"{cycle['cycle_id']} 包含不可识别证券代码：{addition}")
            event_key = f"{cycle['cycle_id']}|{to_ts_code(code)}"
            rows.append(
                {
                    "strategy_id": STRATEGY_ID,
                    "event_key": event_key,
                    "cycle_id": str(cycle["cycle_id"]),
                    "announcement_id": int(cycle["announcement_id"]),
                    "announcement_title": str(cycle["announcement_title"]),
                    "announcement_date": announcement_date,
                    "entry_date": entry_date,
                    "effective_date": effective_date,
                    "security_code": code,
                    "ts_code": to_ts_code(code),
                    "security_name": str(addition.get("security_name") or "").strip(),
                    "exchange": infer_exchange(code),
                    "action": "ADD",
                    "official_sequence": sequence,
                    "cycle_addition_count": len(additions),
                    "source_detail_url": str(cycle["source_detail_url"]),
                    "source_detail_sha256": str(cycle["source_detail_sha256"]),
                    "source_attachment_url": str(attachment.get("url") or ""),
                    "source_attachment_sha256": str(attachment.get("sha256") or ""),
                    "source_attachment_path": str(attachment.get("archive_path") or ""),
                    "source_format": str(cycle["source_format"]),
                    "extraction_method": str(cycle["extraction_method"]),
                    "candidate_state": "PRE_RETURN_CANDIDATE_ALL_OFFICIAL_ADDITIONS",
                    "return_evaluation": "NOT_ALLOWED",
                }
            )
    ledger = pd.DataFrame(rows, columns=EVENT_LEDGER_COLUMNS)
    validate_event_ledger(ledger)
    return ledger


def validate_no_result_columns(columns: Iterable[str]) -> None:
    offending = [
        str(column)
        for column in columns
        if any(token in str(column).lower() for token in FORBIDDEN_RESULT_COLUMN_TOKENS)
        and str(column) != "return_evaluation"
    ]
    if offending:
        raise ValueError(f"D8 违规：无收益账本出现结果或价格字段：{offending}")


def validate_event_ledger(ledger: pd.DataFrame) -> None:
    missing = [column for column in EVENT_LEDGER_COLUMNS if column not in ledger.columns]
    if missing:
        raise ValueError(f"事件账本缺少字段：{missing}")
    validate_no_result_columns(ledger.columns)
    if ledger.empty:
        raise ValueError("事件账本为空")
    if ledger["event_key"].duplicated().any():
        duplicates = ledger.loc[ledger["event_key"].duplicated(keep=False), "event_key"].tolist()
        raise ValueError(f"事件键重复：{duplicates}")
    if set(ledger["strategy_id"]) != {STRATEGY_ID}:
        raise ValueError("事件账本策略标识不一致")
    if set(ledger["action"]) != {"ADD"}:
        raise ValueError("事件账本包含非调入事件")
    if set(ledger["return_evaluation"]) != {"NOT_ALLOWED"}:
        raise ValueError("正式冻结前必须保持 RETURN_EVALUATION=NOT_ALLOWED")
    dates = ledger[["announcement_date", "entry_date", "effective_date"]].apply(pd.to_datetime, errors="raise")
    invalid_clock = ~((dates["announcement_date"] < dates["entry_date"]) & (dates["entry_date"] <= dates["effective_date"]))
    if invalid_clock.any():
        raise ValueError(f"事件时钟顺序无效：{ledger.loc[invalid_clock, 'event_key'].tolist()}")


def round_board_lot_shares(target_notional: float, execution_price: float, board_lot: int = 100) -> int:
    """按不超预算原则向下取整至整手。"""

    if target_notional < 0 or execution_price <= 0 or board_lot <= 0:
        raise ValueError("目标名义金额、成交价或整手参数无效")
    return int(math.floor(target_notional / execution_price / board_lot) * board_lot)


def commission_cny(notional: float, rate: float = 0.0003, minimum: float = 5.0) -> float:
    if notional < 0 or rate < 0 or minimum < 0:
        raise ValueError("佣金输入不得为负")
    if notional == 0:
        return 0.0
    calculated = Decimal(str(notional)) * Decimal(str(rate))
    amount = max(calculated, Decimal(str(minimum))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return float(amount)


def sell_stamp_duty_cny(notional: float, trade_date: str | date) -> float:
    if notional < 0:
        raise ValueError("卖出名义金额不得为负")
    rate = Decimal("0.0005") if _parse_iso_date(trade_date) >= date(2023, 8, 28) else Decimal("0.001")
    amount = (Decimal(str(notional)) * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return float(amount)


def dividend_cash_cny(shares: int, cash_per_share_after_tax: float) -> float:
    if shares < 0 or cash_per_share_after_tax < 0:
        raise ValueError("分红现金输入不得为负")
    amount = (Decimal(shares) * Decimal(str(cash_per_share_after_tax))).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    return float(amount)


def entry_fill_status(*, suspended: bool, one_price_limit_up: bool) -> str:
    if suspended:
        return "NO_FILL_SUSPENDED"
    if one_price_limit_up:
        return "NO_FILL_ONE_PRICE_LIMIT_UP"
    return "FILL_AT_RAW_OPEN"


def resolve_exit_clock(
    effective_date: str | date,
    sellability_by_date: Mapping[str | date, bool],
) -> tuple[str, str | None]:
    """生效日可卖则收盘退出，否则顺延到首个可卖开盘。"""

    effective = _parse_iso_date(effective_date)
    normalised = sorted((_parse_iso_date(day), bool(sellable)) for day, sellable in sellability_by_date.items())
    if any(day == effective and sellable for day, sellable in normalised):
        return "EXIT_AT_EFFECTIVE_CLOSE", effective.isoformat()
    for day, sellable in normalised:
        if day > effective and sellable:
            return "EXIT_AT_FIRST_SELLABLE_OPEN", day.isoformat()
    return "CENSORED_EXIT_PENDING_FIRST_SELLABLE_OPEN", None
