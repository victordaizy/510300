"""沪深300点时成分与权重数据源修复的纯函数。

本模块只处理指数成分、权重、交易日历和来源凭证。任何价格、收益、标签、
模型、仓位或订单字段均不在允许输入范围内。
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
from bs4 import BeautifulSoup

from research.a_share_hs_csi300_official_addition_forced_demand_v1 import (
    parse_official_csi300_changes,
)


REMEDIATION_ID = "510300_CSI300_PIT_MEMBERSHIP_WEIGHTS_SOURCE_REMEDIATION_V1"
FORBIDDEN_COLUMN_TOKENS = (
    "open",
    "high",
    "low",
    "close",
    "price",
    "return",
    "future",
    "label",
    "target",
    "position",
    "order",
    "pnl",
    "sharpe",
)
ALLOWED_CALENDAR_COLUMN_NAMES = {
    "is_open",
    "open_date",
    "first_open_date",
    "next_open_date",
}


@dataclass(frozen=True)
class OfficialCycle:
    """一轮经原始官方正文或附件重新解析的调样。"""

    cycle_id: str
    announcement_id: int
    announcement_date: date
    effective_date: date
    additions: frozenset[str]
    deletions: frozenset[str]
    source_format: str
    extraction_method: str
    detail_path: str
    detail_sha256: str
    attachment_paths: tuple[str, ...]
    attachment_sha256s: tuple[str, ...]


@dataclass(frozen=True)
class MembershipTransition:
    """将官方表述日期映射到实际交易会话后的成分状态迁移。"""

    cycle: OfficialCycle
    transition_session: date
    clock_rule: str


def sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_symbol(value: object) -> str:
    """将六位 A 股证券代码规范化为 Tushare 风格代码。"""

    raw = str(value).strip().upper()
    if raw.endswith((".SH", ".SZ")) and len(raw) == 9:
        return raw
    digits = "".join(character for character in raw if character.isdigit())
    if len(digits) != 6:
        raise ValueError(f"无法规范化证券代码：{value!r}")
    suffix = "SH" if digits.startswith(("5", "6", "9")) else "SZ"
    return f"{digits}.{suffix}"


def assert_outcome_blind_columns(columns: Iterable[object]) -> None:
    """拒绝任何价格、收益、标签或交易结果字段。"""

    violations: list[str] = []
    for column in columns:
        normalized = str(column).strip().lower()
        if normalized in ALLOWED_CALENDAR_COLUMN_NAMES:
            continue
        words = {word for word in re.split(r"[^a-z0-9]+", normalized) if word}
        if any(token in words for token in FORBIDDEN_COLUMN_TOKENS):
            violations.append(str(column))
    if violations:
        raise ValueError(f"数据源修复输入包含禁止字段：{sorted(set(violations))}")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _verify_archived_object(root: Path, record: Mapping[str, Any]) -> tuple[Path, str]:
    relative_path = str(record["archive_path"])
    expected_sha256 = str(record["sha256"]).lower()
    path = _resolve(root, relative_path)
    if not path.is_file():
        raise FileNotFoundError(f"归档来源不存在：{relative_path}")
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"归档来源哈希漂移：{relative_path}，期望 {expected_sha256}，实际 {observed_sha256}"
        )
    return path, observed_sha256


def load_official_cycles(root: Path, manifest_path: Path) -> list[OfficialCycle]:
    """从本地官方原件重新解析调入和调出，不信任派生名单。"""

    manifest = _load_json(manifest_path)
    records = manifest.get("cycles")
    if not isinstance(records, list) or not records:
        raise ValueError("官方公告清单没有周期记录")

    cycles: list[OfficialCycle] = []
    for record in records:
        detail_record = record.get("detail")
        if not isinstance(detail_record, Mapping):
            raise ValueError(f"{record.get('cycle_id')} 缺少官方详情归档")
        detail_path, detail_digest = _verify_archived_object(root, detail_record)
        wrapped_detail = _load_json(detail_path)
        detail = wrapped_detail.get("data") if isinstance(wrapped_detail, Mapping) else None
        if not isinstance(detail, Mapping):
            raise ValueError(f"{record.get('cycle_id')} 官方详情响应缺少 data 对象")

        attachment_payloads: list[dict[str, Any]] = []
        attachment_paths: list[str] = []
        attachment_sha256s: list[str] = []
        for attachment in record.get("attachments") or []:
            attachment_path, attachment_digest = _verify_archived_object(root, attachment)
            attachment_payloads.append(
                {
                    **dict(attachment),
                    "payload": attachment_path.read_bytes(),
                }
            )
            attachment_paths.append(str(attachment["archive_path"]))
            attachment_sha256s.append(attachment_digest)

        parsed = parse_official_csi300_changes(detail, attachment_payloads)
        additions = frozenset(normalize_symbol(item["security_code"]) for item in parsed["additions"])
        deletions = frozenset(normalize_symbol(item["security_code"]) for item in parsed["deletions"])
        manifest_additions = frozenset(
            normalize_symbol(item["security_code"]) for item in record.get("additions") or []
        )
        expected_addition_count = int(record["addition_count"])
        expected_deletion_count = int(record["deletion_count"])

        if additions != manifest_additions:
            raise ValueError(f"{record['cycle_id']} 重新解析的调入集合与冻结清单不一致")
        if len(additions) != expected_addition_count:
            raise ValueError(f"{record['cycle_id']} 调入数量与冻结清单不一致")
        if len(deletions) != expected_deletion_count:
            raise ValueError(f"{record['cycle_id']} 调出数量与冻结清单不一致")
        if len(additions) != len(deletions):
            raise ValueError(f"{record['cycle_id']} 调入和调出数量不相等")
        if additions & deletions:
            raise ValueError(f"{record['cycle_id']} 调入和调出集合重叠")

        cycle = OfficialCycle(
            cycle_id=str(record["cycle_id"]),
            announcement_id=int(record["announcement_id"]),
            announcement_date=date.fromisoformat(str(record["announcement_date"])),
            effective_date=date.fromisoformat(str(record["effective_date"])),
            additions=additions,
            deletions=deletions,
            source_format=str(parsed["source_format"]),
            extraction_method=str(parsed["method"]),
            detail_path=str(detail_record["archive_path"]),
            detail_sha256=detail_digest,
            attachment_paths=tuple(attachment_paths),
            attachment_sha256s=tuple(attachment_sha256s),
        )
        if cycle.announcement_date >= cycle.effective_date:
            raise ValueError(f"{cycle.cycle_id} 公告日不早于生效日")
        cycles.append(cycle)

    ordered = sorted(cycles, key=lambda item: item.effective_date)
    if ordered != cycles:
        raise ValueError("官方周期清单未按生效日严格升序排列")
    if len({cycle.cycle_id for cycle in cycles}) != len(cycles):
        raise ValueError("官方周期 ID 重复")
    if len({cycle.announcement_id for cycle in cycles}) != len(cycles):
        raise ValueError("官方公告 ID 重复")
    return cycles


def load_official_special_cycles(root: Path, manifest_path: Path) -> list[OfficialCycle]:
    """加载已归档并由中证调样附件与上交所退市日共同确认的临时调样。"""

    manifest = _load_json(manifest_path)
    if manifest.get("status") != "PASS_OFFICIAL_SPECIAL_REBALANCE_SOURCES_ACQUIRED":
        raise ValueError("官方临时调样来源清单未处于通过状态")
    for source in manifest.get("source_objects") or []:
        archive_path = _resolve(root, str(source["archive_path"]))
        if not archive_path.is_file():
            raise FileNotFoundError(f"临时调样归档来源不存在：{source['archive_path']}")
        expected = str(source["sha256"]).lower()
        observed = sha256_file(archive_path)
        if observed != expected:
            raise ValueError(f"临时调样归档来源哈希漂移：{source['archive_path']}")

    cycles: list[OfficialCycle] = []
    for record in manifest.get("cycles") or []:
        additions = frozenset(normalize_symbol(item["symbol"]) for item in record["additions"])
        deletions = frozenset(normalize_symbol(item["symbol"]) for item in record["deletions"])
        if len(additions) != 1 or len(deletions) != 1 or additions & deletions:
            raise ValueError(f"临时调样 {record.get('cycle_id')} 不是一进一出")
        detail_source = record["detail_source"]
        attachment_sources = record["attachment_sources"]
        effective_source = record["effective_date_source"]
        cycle = OfficialCycle(
            cycle_id=str(record["cycle_id"]),
            announcement_id=int(record["announcement_id"]),
            announcement_date=date.fromisoformat(str(record["announcement_date"])),
            effective_date=date.fromisoformat(str(record["effective_date"])),
            additions=additions,
            deletions=deletions,
            source_format="XLSX_PLUS_SSE_HTML",
            extraction_method="CSI_SPECIAL_ATTACHMENT_000300_ROW_PLUS_SSE_DELISTING_DATE",
            detail_path=str(detail_source["archive_path"]),
            detail_sha256=str(detail_source["sha256"]),
            attachment_paths=tuple(
                [str(item["archive_path"]) for item in attachment_sources]
                + [str(effective_source["archive_path"])]
            ),
            attachment_sha256s=tuple(
                [str(item["sha256"]) for item in attachment_sources]
                + [str(effective_source["sha256"])]
            ),
        )
        if cycle.announcement_date >= cycle.effective_date:
            raise ValueError(f"临时调样 {cycle.cycle_id} 公告日不早于生效日")
        cycles.append(cycle)
    if len(cycles) != int(manifest.get("special_cycle_count", -1)):
        raise ValueError("临时调样周期数量与清单不一致")
    return sorted(cycles, key=lambda item: item.effective_date)


def _six_digit_security_code(value: object) -> str | None:
    """从官方旧表单单元格中提取六位证券或指数代码。"""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    raw = str(value).strip().upper()
    digits = "".join(character for character in raw if character.isdigit())
    if len(digits) == 6:
        return digits
    return None


def parse_2015_extension_cycle_changes(
    detail: Mapping[str, Any],
    attachments: Sequence[Mapping[str, Any]],
    parse_mode: str,
    expected_count: int,
) -> dict[str, Any]:
    """重新解析 2015 年官方正文或附件中的沪深300调样记录。"""

    additions: list[dict[str, str]] = []
    deletions: list[dict[str, str]] = []
    extraction_method = ""
    source_format = ""

    if parse_mode == "INLINE_INDEXED_HTML_TABLES":
        soup = BeautifulSoup(str(detail.get("content") or ""), "html.parser")
        for table in soup.find_all("table"):
            for row in table.find_all("tr"):
                cells = [
                    re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).strip()
                    for cell in row.find_all(["td", "th"])
                ]
                if len(cells) < 6 or _six_digit_security_code(cells[0]) != "000300":
                    continue
                deletion_code = _six_digit_security_code(cells[2])
                addition_code = _six_digit_security_code(cells[4])
                if deletion_code is None or addition_code is None:
                    continue
                deletions.append(
                    {"security_code": deletion_code, "security_name": cells[3]}
                )
                additions.append(
                    {"security_code": addition_code, "security_name": cells[5]}
                )
        extraction_method = "OFFICIAL_DETAIL_ALL_INDEXED_HTML_TABLES_000300_ROWS"
        source_format = "HTML"
    elif parse_mode == "STANDARD_OFFICIAL_PARSER":
        parsed = parse_official_csi300_changes(detail, attachments)
        additions = [dict(item) for item in parsed["additions"]]
        deletions = [dict(item) for item in parsed["deletions"]]
        extraction_method = str(parsed["method"])
        source_format = str(parsed["source_format"])
    elif parse_mode == "COMBINED_EXCEL_000300_ROW":
        matches: list[tuple[dict[str, str], dict[str, str], str]] = []
        for attachment in attachments:
            payload = attachment.get("payload")
            if not isinstance(payload, bytes):
                continue
            suffix = str(attachment.get("suffix") or "").lower()
            if suffix not in {".xls", ".xlsx"}:
                continue
            excel = pd.ExcelFile(io.BytesIO(payload))
            for sheet_name in excel.sheet_names:
                frame = pd.read_excel(
                    io.BytesIO(payload),
                    sheet_name=sheet_name,
                    header=None,
                    dtype=object,
                )
                for values in frame.itertuples(index=False, name=None):
                    cells = ["" if pd.isna(value) else str(value).strip() for value in values]
                    if len(cells) < 5 or _six_digit_security_code(cells[0]) != "000300":
                        continue
                    deletion_code = _six_digit_security_code(cells[2])
                    addition_code = _six_digit_security_code(cells[4])
                    if deletion_code is None or addition_code is None:
                        continue
                    deletion_name = cells[3] if len(cells) > 3 else ""
                    addition_name = cells[5] if len(cells) > 5 else ""
                    matches.append(
                        (
                            {
                                "security_code": deletion_code,
                                "security_name": deletion_name,
                            },
                            {
                                "security_code": addition_code,
                                "security_name": addition_name,
                            },
                            str(sheet_name),
                        )
                    )
        deletions = [item[0] for item in matches]
        additions = [item[1] for item in matches]
        extraction_method = "OFFICIAL_COMBINED_EXCEL_ALL_SHEETS_000300_ROW"
        source_format = "XLS_OR_XLSX"
    else:
        raise ValueError(f"未知的 2015 扩展解析模式：{parse_mode}")

    def deduplicate(records: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        seen: set[str] = set()
        for record in records:
            code = _six_digit_security_code(record.get("security_code"))
            if code is None or code in seen:
                continue
            seen.add(code)
            result.append(
                {
                    "security_code": code,
                    "security_name": str(record.get("security_name") or "").strip(),
                }
            )
        return result

    additions = deduplicate(additions)
    deletions = deduplicate(deletions)
    if len(additions) != expected_count or len(deletions) != expected_count:
        raise ValueError(
            "2015 官方调样解析数量不符："
            f"模式={parse_mode}，期望={expected_count}，"
            f"调入={len(additions)}，调出={len(deletions)}"
        )
    addition_codes = {item["security_code"] for item in additions}
    deletion_codes = {item["security_code"] for item in deletions}
    if addition_codes & deletion_codes:
        raise ValueError("2015 官方调样的调入与调出集合重叠")
    return {
        "additions": additions,
        "deletions": deletions,
        "method": extraction_method,
        "source_format": source_format,
    }


def load_official_2015_extension_cycles(root: Path, manifest_path: Path) -> list[OfficialCycle]:
    """从归档原件重解析 2015 年条件扩展的五轮标准沪深300调样。"""

    manifest = _load_json(manifest_path)
    expected_status = "PASS_OFFICIAL_2015_MEMBERSHIP_EXTENSION_SOURCES_ACQUIRED"
    if manifest.get("status") != expected_status:
        raise ValueError("2015 官方扩展来源清单未处于通过状态")
    for source in manifest.get("source_objects") or []:
        _verify_archived_object(root, source)

    cycles: list[OfficialCycle] = []
    for record in manifest.get("cycles") or []:
        detail_source = record.get("detail_source")
        if not isinstance(detail_source, Mapping):
            raise ValueError(f"{record.get('cycle_id')} 缺少官方详情来源")
        detail_path, detail_digest = _verify_archived_object(root, detail_source)
        wrapped_detail = _load_json(detail_path)
        detail = wrapped_detail.get("data") if isinstance(wrapped_detail, Mapping) else None
        if not isinstance(detail, Mapping):
            raise ValueError(f"{record.get('cycle_id')} 官方详情缺少 data")

        attachment_payloads: list[dict[str, Any]] = []
        attachment_paths: list[str] = []
        attachment_sha256s: list[str] = []
        for attachment in record.get("attachment_sources") or []:
            attachment_path, attachment_digest = _verify_archived_object(root, attachment)
            suffix = Path(attachment_path).suffix.lower()
            attachment_payloads.append(
                {
                    **dict(attachment),
                    "payload": attachment_path.read_bytes(),
                    "suffix": suffix,
                }
            )
            attachment_paths.append(str(attachment["archive_path"]))
            attachment_sha256s.append(attachment_digest)
        for effective_source in record.get("effective_date_sources") or []:
            _, effective_digest = _verify_archived_object(root, effective_source)
            attachment_paths.append(str(effective_source["archive_path"]))
            attachment_sha256s.append(effective_digest)

        parsed = parse_2015_extension_cycle_changes(
            detail,
            attachment_payloads,
            str(record["parse_mode"]),
            int(record["change_count"]),
        )
        additions = frozenset(
            normalize_symbol(item["security_code"]) for item in parsed["additions"]
        )
        deletions = frozenset(
            normalize_symbol(item["security_code"]) for item in parsed["deletions"]
        )
        manifest_additions = frozenset(
            normalize_symbol(item["symbol"]) for item in record.get("additions") or []
        )
        manifest_deletions = frozenset(
            normalize_symbol(item["symbol"]) for item in record.get("deletions") or []
        )
        if additions != manifest_additions or deletions != manifest_deletions:
            raise ValueError(f"{record['cycle_id']} 原件重解析集合与来源清单不一致")

        cycle = OfficialCycle(
            cycle_id=str(record["cycle_id"]),
            announcement_id=int(record["announcement_id"]),
            announcement_date=date.fromisoformat(str(record["announcement_date"])),
            effective_date=date.fromisoformat(str(record["effective_date"])),
            additions=additions,
            deletions=deletions,
            source_format=str(parsed["source_format"]),
            extraction_method=str(parsed["method"]),
            detail_path=str(detail_source["archive_path"]),
            detail_sha256=detail_digest,
            attachment_paths=tuple(attachment_paths),
            attachment_sha256s=tuple(attachment_sha256s),
        )
        if cycle.announcement_date >= cycle.effective_date:
            raise ValueError(f"{cycle.cycle_id} 公告日不早于生效日")
        cycles.append(cycle)

    if len(cycles) != int(manifest.get("cycle_count", -1)):
        raise ValueError("2015 官方扩展周期数量与清单不一致")
    ordered = sorted(cycles, key=lambda item: item.effective_date)
    if ordered != cycles:
        raise ValueError("2015 官方扩展周期未按生效日升序排列")
    if len({cycle.cycle_id for cycle in cycles}) != len(cycles):
        raise ValueError("2015 官方扩展周期 ID 重复")
    if len({cycle.announcement_id for cycle in cycles}) != len(cycles):
        raise ValueError("2015 官方扩展公告 ID 重复")
    return cycles


def load_weight_snapshots(path: Path) -> tuple[pd.DataFrame, dict[date, frozenset[str]]]:
    """读取月度权重快照并执行结构校验。"""

    frame = pd.read_parquet(path)
    required = {"con_code", "trade_date", "weight", "source", "retrieved_at"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"权重快照缺少字段：{sorted(missing)}")
    assert_outcome_blind_columns(frame.columns)
    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise").dt.date
    frame["symbol"] = frame["con_code"].map(normalize_symbol)
    frame["weight"] = pd.to_numeric(frame["weight"], errors="raise")
    if frame.duplicated(["trade_date", "symbol"]).any():
        raise ValueError("权重快照存在日期和证券重复行")

    grouped = frame.groupby("trade_date", sort=True)
    counts = grouped["symbol"].nunique()
    sums = grouped["weight"].sum()
    if not counts.eq(300).all():
        raise ValueError(f"权重快照并非每期 300 只：{counts[counts.ne(300)].to_dict()}")
    if not sums.between(98.0, 102.0, inclusive="both").all():
        raise ValueError(f"权重和超出 98%-102%：{sums[~sums.between(98.0, 102.0)].to_dict()}")
    snapshots = {
        snapshot_date: frozenset(group["symbol"].tolist())
        for snapshot_date, group in grouped
    }
    return frame, snapshots


def load_current_official_anchor(path: Path) -> tuple[date, frozenset[str], pd.DataFrame]:
    """读取保存自中证指数官网 closeweight 文件的当前 300 只锚点。"""

    frame = pd.read_parquet(path)
    required = {"effective_date", "stock_code", "weight_pct", "source", "retrieved_at"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"当前官方锚点缺少字段：{sorted(missing)}")
    assert_outcome_blind_columns(frame.columns)
    frame = frame.copy()
    frame["effective_date"] = pd.to_datetime(frame["effective_date"], errors="raise").dt.date
    dates = sorted(frame["effective_date"].unique())
    if len(dates) != 1:
        raise ValueError(f"当前官方锚点应只有一个日期，实际为 {dates}")
    frame["symbol"] = frame["stock_code"].map(normalize_symbol)
    if frame["symbol"].nunique() != 300 or len(frame) != 300:
        raise ValueError("当前官方锚点不是 300 个唯一证券")
    if not 98.0 <= float(pd.to_numeric(frame["weight_pct"], errors="raise").sum()) <= 102.0:
        raise ValueError("当前官方锚点权重和超出 98%-102%")
    return dates[0], frozenset(frame["symbol"]), frame


def replay_forward(
    anchor_date: date,
    anchor_set: frozenset[str],
    cycles: Sequence[OfficialCycle],
) -> tuple[dict[date, frozenset[str]], list[dict[str, Any]]]:
    """从锚点向前重放所有后续定期调样。"""

    state = set(anchor_set)
    states = {anchor_date: frozenset(state)}
    audits: list[dict[str, Any]] = []
    for cycle in cycles:
        if cycle.effective_date <= anchor_date:
            continue
        missing_deletions = sorted(cycle.deletions - state)
        existing_additions = sorted(cycle.additions & state)
        before_count = len(state)
        state.difference_update(cycle.deletions)
        state.update(cycle.additions)
        after_count = len(state)
        audits.append(
            {
                "direction": "FORWARD",
                "cycle_id": cycle.cycle_id,
                "effective_date": cycle.effective_date.isoformat(),
                "before_count": before_count,
                "after_count": after_count,
                "missing_deletions": missing_deletions,
                "existing_additions": existing_additions,
            }
        )
        states[cycle.effective_date] = frozenset(state)
    return states, audits


def replay_backward(
    anchor_date: date,
    anchor_set: frozenset[str],
    cycles: Sequence[OfficialCycle],
) -> tuple[dict[date, frozenset[str]], list[dict[str, Any]]]:
    """从当前官方锚点逆向撤销定期调样。"""

    state = set(anchor_set)
    states = {anchor_date: frozenset(state)}
    audits: list[dict[str, Any]] = []
    for cycle in sorted(cycles, key=lambda item: item.effective_date, reverse=True):
        if cycle.effective_date > anchor_date:
            continue
        missing_additions = sorted(cycle.additions - state)
        existing_deletions = sorted(cycle.deletions & state)
        before_count = len(state)
        state.difference_update(cycle.additions)
        state.update(cycle.deletions)
        after_count = len(state)
        audits.append(
            {
                "direction": "BACKWARD",
                "cycle_id": cycle.cycle_id,
                "effective_date": cycle.effective_date.isoformat(),
                "before_count": before_count,
                "after_count": after_count,
                "missing_additions": missing_additions,
                "existing_deletions": existing_deletions,
            }
        )
        states[cycle.effective_date] = frozenset(state)
    return states, audits


def derive_past_anchor(
    current_anchor_date: date,
    current_anchor_set: frozenset[str],
    target_date: date,
    cycles: Sequence[OfficialCycle],
) -> tuple[frozenset[str], list[dict[str, Any]]]:
    """由当前官方集合逆向撤销目标日之后的定期调样，得到历史锚点。"""

    if target_date >= current_anchor_date:
        raise ValueError("历史锚点目标日必须早于当前官方锚点日")
    state = set(current_anchor_set)
    audits: list[dict[str, Any]] = []
    applicable = [
        cycle
        for cycle in cycles
        if target_date < cycle.effective_date <= current_anchor_date
    ]
    for cycle in sorted(applicable, key=lambda item: item.effective_date, reverse=True):
        missing_additions = sorted(cycle.additions - state)
        existing_deletions = sorted(cycle.deletions & state)
        before_count = len(state)
        state.difference_update(cycle.additions)
        state.update(cycle.deletions)
        audits.append(
            {
                "direction": "BACKWARD_TO_TARGET",
                "cycle_id": cycle.cycle_id,
                "effective_date": cycle.effective_date.isoformat(),
                "before_count": before_count,
                "after_count": len(state),
                "missing_additions": missing_additions,
                "existing_deletions": existing_deletions,
            }
        )
    return frozenset(state), audits


def state_on_or_before(
    state_by_effective_date: Mapping[date, frozenset[str]],
    observation_date: date,
) -> frozenset[str]:
    """选择观察日当日已生效的最近状态。"""

    eligible = [state_date for state_date in state_by_effective_date if state_date <= observation_date]
    if not eligible:
        raise ValueError(f"观察日 {observation_date} 之前没有可用状态")
    return state_by_effective_date[max(eligible)]


def compare_replay_to_snapshots(
    state_by_effective_date: Mapping[date, frozenset[str]],
    snapshots: Mapping[date, frozenset[str]],
    start_date: date,
    end_date: date,
) -> list[dict[str, Any]]:
    """逐月比较重放集合和独立权重快照集合。"""

    rows: list[dict[str, Any]] = []
    for snapshot_date in sorted(snapshots):
        if snapshot_date < start_date or snapshot_date > end_date:
            continue
        expected = state_on_or_before(state_by_effective_date, snapshot_date)
        observed = snapshots[snapshot_date]
        missing_from_replay = sorted(observed - expected)
        extra_in_replay = sorted(expected - observed)
        rows.append(
            {
                "snapshot_date": snapshot_date.isoformat(),
                "replay_count": len(expected),
                "snapshot_count": len(observed),
                "set_difference_count": len(missing_from_replay) + len(extra_in_replay),
                "missing_from_replay": missing_from_replay,
                "extra_in_replay": extra_in_replay,
            }
        )
    return rows


def detect_monthly_snapshot_transitions(
    snapshots: Mapping[date, frozenset[str]],
) -> list[dict[str, Any]]:
    """识别相邻月末快照之间的成分变化，用于定位临时调整。"""

    rows: list[dict[str, Any]] = []
    previous_date: date | None = None
    previous_set: frozenset[str] | None = None
    for snapshot_date in sorted(snapshots):
        current_set = snapshots[snapshot_date]
        if previous_date is not None and previous_set is not None:
            additions = sorted(current_set - previous_set)
            deletions = sorted(previous_set - current_set)
            if additions or deletions:
                rows.append(
                    {
                        "previous_snapshot_date": previous_date.isoformat(),
                        "snapshot_date": snapshot_date.isoformat(),
                        "additions": additions,
                        "deletions": deletions,
                        "addition_count": len(additions),
                        "deletion_count": len(deletions),
                    }
                )
        previous_date = snapshot_date
        previous_set = current_set
    return rows


def load_unique_open_dates(path: Path) -> list[date]:
    """从两交易所观察日历中生成严格唯一、升序的开市日期。"""

    frame = pd.read_parquet(path)
    required = {"exchange", "date", "is_open", "available_at", "source"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"交易日历缺少字段：{sorted(missing)}")
    assert_outcome_blind_columns(frame.columns)
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.date
    if frame["is_open"].isna().any():
        raise ValueError("交易日历存在空的 is_open")
    open_rows = frame.loc[frame["is_open"].astype(bool)].copy()
    if open_rows.empty:
        raise ValueError("交易日历没有开市日期")
    exchange_sets = {
        str(exchange): frozenset(group["date"])
        for exchange, group in open_rows.groupby("exchange", sort=True)
    }
    if not {"SSE", "SZSE"}.issubset(exchange_sets):
        raise ValueError(f"交易日历没有同时覆盖 SSE 与 SZSE：{sorted(exchange_sets)}")
    if exchange_sets["SSE"] != exchange_sets["SZSE"]:
        only_sse = sorted(exchange_sets["SSE"] - exchange_sets["SZSE"])
        only_szse = sorted(exchange_sets["SZSE"] - exchange_sets["SSE"])
        raise ValueError(
            f"SSE/SZSE 开市日集合不一致：仅SSE={only_sse[:10]}，仅SZSE={only_szse[:10]}"
        )
    dates = sorted(exchange_sets["SSE"])
    if len(dates) != len(set(dates)):
        raise ValueError("去重后的交易日历仍有重复日期")
    return dates


def _official_detail_text(root: Path, cycle: OfficialCycle) -> str:
    payload = _load_json(_resolve(root, cycle.detail_path))
    detail = payload.get("data") if isinstance(payload, Mapping) else None
    if not isinstance(detail, Mapping):
        raise ValueError(f"{cycle.cycle_id} 官方详情缺少 data")
    content = str(detail.get("content") or "")
    plain = re.sub(r"<[^>]+>", " ", content)
    return re.sub(r"\s+", " ", plain.replace("&nbsp;", " ")).strip()


def resolve_membership_transitions(
    root: Path,
    cycles: Sequence[OfficialCycle],
    open_dates: Sequence[date],
    correction_manifest_path: Path,
) -> list[MembershipTransition]:
    """按冻结时钟更正把官方日期映射到实际状态迁移会话。"""

    correction = _load_json(correction_manifest_path)
    if correction.get("status") != "FROZEN_BEFORE_DAILY_MEMBERSHIP_OUTPUT":
        raise ValueError("成分时钟更正未处于冻结状态")
    for record in correction.get("frozen_inputs") or []:
        path = _resolve(root, str(record["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"时钟更正冻结输入不存在：{record['path']}")
        observed = sha256_file(path)
        if observed != str(record["sha256"]).lower():
            raise ValueError(f"时钟更正冻结输入哈希漂移：{record['path']}")

    rule_by_cycle: dict[str, Mapping[str, Any]] = {}
    for rule in correction.get("corrected_rules") or []:
        for cycle_id in rule.get("cycle_ids") or []:
            if cycle_id in rule_by_cycle:
                raise ValueError(f"时钟更正重复覆盖周期：{cycle_id}")
            rule_by_cycle[str(cycle_id)] = rule
    cycle_ids = {cycle.cycle_id for cycle in cycles}
    missing_rules = cycle_ids - set(rule_by_cycle)
    extra_rules = set(rule_by_cycle) - cycle_ids
    if missing_rules or extra_rules:
        raise ValueError(
            f"时钟更正与官方周期不一一对应：缺规则={sorted(missing_rules)}，"
            f"多余规则={sorted(extra_rules)}"
        )

    ordered_open_dates = sorted(set(open_dates))
    open_date_set = set(ordered_open_dates)
    transitions: list[MembershipTransition] = []
    for cycle in cycles:
        rule = rule_by_cycle[cycle.cycle_id]
        clock_rule = str(rule["daily_state_transition_session"])
        if clock_rule == "FIRST_UNIQUE_OPEN_DATE_STRICTLY_AFTER_OFFICIAL_STATED_CLOSE_DATE":
            detail_text = _official_detail_text(root, cycle)
            required_phrases = [str(item) for item in rule.get("source_text_must_contain_one_of") or []]
            if not any(phrase in detail_text for phrase in required_phrases):
                raise ValueError(f"{cycle.cycle_id} 官方正文没有冻结的收市后生效表述")
            eligible = [value for value in ordered_open_dates if value > cycle.effective_date]
            if not eligible:
                raise ValueError(f"{cycle.cycle_id} 生效表述日之后没有交易日")
            transition_session = eligible[0]
        elif clock_rule in {
            "OFFICIAL_EFFECTIVE_DATE_AT_START_OF_DAY",
            "OFFICIAL_EXACT_DELISTING_EFFECTIVE_DATE_AT_START_OF_DAY",
        }:
            transition_session = cycle.effective_date
            if transition_session not in open_date_set:
                raise ValueError(f"{cycle.cycle_id} 冻结迁移日不是开市日：{transition_session}")
        else:
            raise ValueError(f"未知的成分时钟规则：{clock_rule}")
        if cycle.announcement_date >= transition_session:
            raise ValueError(f"{cycle.cycle_id} 公告日不早于实际迁移会话")
        transitions.append(
            MembershipTransition(
                cycle=cycle,
                transition_session=transition_session,
                clock_rule=clock_rule,
            )
        )

    transitions = sorted(transitions, key=lambda item: item.transition_session)
    sessions = [item.transition_session for item in transitions]
    if len(sessions) != len(set(sessions)):
        raise ValueError("多个官方周期映射到同一个迁移会话，需显式合并后再运行")
    return transitions


def derive_past_anchor_from_transitions(
    current_anchor_date: date,
    current_anchor_set: frozenset[str],
    target_date: date,
    transitions: Sequence[MembershipTransition],
) -> tuple[frozenset[str], list[dict[str, Any]]]:
    """按实际迁移会话从当前官方集合逆向得到目标日状态。"""

    if target_date >= current_anchor_date:
        raise ValueError("历史锚点目标日必须早于当前官方锚点日")
    state = set(current_anchor_set)
    audits: list[dict[str, Any]] = []
    applicable = [
        item
        for item in transitions
        if target_date < item.transition_session <= current_anchor_date
    ]
    for transition in sorted(applicable, key=lambda item: item.transition_session, reverse=True):
        cycle = transition.cycle
        missing_additions = sorted(cycle.additions - state)
        existing_deletions = sorted(cycle.deletions & state)
        before_count = len(state)
        state.difference_update(cycle.additions)
        state.update(cycle.deletions)
        audits.append(
            {
                "direction": "BACKWARD_TO_TARGET_BY_SESSION",
                "cycle_id": cycle.cycle_id,
                "stated_effective_date": cycle.effective_date.isoformat(),
                "transition_session": transition.transition_session.isoformat(),
                "clock_rule": transition.clock_rule,
                "before_count": before_count,
                "after_count": len(state),
                "missing_additions": missing_additions,
                "existing_deletions": existing_deletions,
            }
        )
    return frozenset(state), audits


def replay_membership_transitions(
    anchor_date: date,
    anchor_set: frozenset[str],
    transitions: Sequence[MembershipTransition],
) -> tuple[dict[date, frozenset[str]], list[dict[str, Any]]]:
    """从历史锚点按实际迁移会话向前重放成分状态。"""

    state = set(anchor_set)
    states = {anchor_date: frozenset(state)}
    audits: list[dict[str, Any]] = []
    for transition in transitions:
        if transition.transition_session <= anchor_date:
            continue
        cycle = transition.cycle
        missing_deletions = sorted(cycle.deletions - state)
        existing_additions = sorted(cycle.additions & state)
        before_count = len(state)
        state.difference_update(cycle.deletions)
        state.update(cycle.additions)
        audits.append(
            {
                "direction": "FORWARD_BY_SESSION",
                "cycle_id": cycle.cycle_id,
                "announcement_date": cycle.announcement_date.isoformat(),
                "stated_effective_date": cycle.effective_date.isoformat(),
                "transition_session": transition.transition_session.isoformat(),
                "clock_rule": transition.clock_rule,
                "addition_count": len(cycle.additions),
                "deletion_count": len(cycle.deletions),
                "before_count": before_count,
                "after_count": len(state),
                "missing_deletions": missing_deletions,
                "existing_additions": existing_additions,
            }
        )
        states[transition.transition_session] = frozenset(state)
    return states, audits


def build_daily_membership_panel(
    open_dates: Sequence[date],
    state_by_transition_session: Mapping[date, frozenset[str]],
    start_date: date,
    end_date: date,
    transition_cycle_id_by_session: Mapping[date, str],
    *,
    source_contract: str = "CSI_OFFICIAL_REBALANCE_REPLAY_V1_0_1",
    anchor_cycle_id: str = "REVERSED_CURRENT_OFFICIAL_ANCHOR",
) -> pd.DataFrame:
    """把稀疏官方状态展开为每个开市日恰好 300 行的成分面板。"""

    if start_date > end_date:
        raise ValueError("成分面板开始日期不得晚于结束日期")
    sessions = sorted(
        value for value in set(open_dates) if start_date <= value <= end_date
    )
    if not sessions:
        raise ValueError("目标区间没有开市日")
    rows: list[dict[str, Any]] = []
    state_sessions = sorted(state_by_transition_session)
    for membership_date in sessions:
        eligible = [value for value in state_sessions if value <= membership_date]
        if not eligible:
            raise ValueError(f"{membership_date} 之前没有成分状态")
        state_session = eligible[-1]
        state = state_by_transition_session[state_session]
        if len(state) != 300:
            raise ValueError(f"{membership_date} 的成分数不是 300：{len(state)}")
        cycle_id = transition_cycle_id_by_session.get(
            state_session,
            anchor_cycle_id,
        )
        for symbol in sorted(state):
            rows.append(
                {
                    "membership_date": membership_date,
                    "index_code": "000300",
                    "symbol": symbol,
                    "state_transition_session": state_session,
                    "state_cycle_id": cycle_id,
                    "source_contract": source_contract,
                }
            )
    output = pd.DataFrame(rows)
    assert_outcome_blind_columns(output.columns)
    if output.duplicated(["membership_date", "symbol"]).any():
        raise ValueError("每日成分面板存在日期和证券重复行")
    counts = output.groupby("membership_date")["symbol"].nunique()
    if not counts.eq(300).all():
        raise ValueError(f"每日成分面板存在非 300 只日期：{counts[counts.ne(300)].to_dict()}")
    return output
