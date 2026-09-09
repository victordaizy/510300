"""510300 非对称压力风险 V1 的三项来源修复纯函数。

本模块只处理 DR007、央行 7 天期逆回购操作利率和申万点时行业归属。
它不读取 BAD10 标签、价格未来值、组合结果，也不构造 M/F/T 特征。
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


REMEDIATION_ID = (
    "510300_ASYMMETRIC_STRESS_HAZARD_V1_SOURCE_REMEDIATION_V1_0_1"
)
OBSERVATION_START = date(2015, 1, 5)
OBSERVATION_CUTOFF = date(2026, 8, 14)
SHANGHAI_TIMEZONE = "Asia/Shanghai"
SW_HISTORY_URL = (
    "https://www.swsresearch.com/swindex/pdf/SwClass2021/"
    "StockClassifyUse_stock.xls"
)
SW_CODEBOOK_URL = (
    "https://www.swsresearch.com/swindex/pdf/SwClass2021/SwClassCode_2021.xls"
)
PBOC_LIST_ROOT = (
    "https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125431/125475"
)
CHINAMONEY_DAILY_URL = (
    "https://www.chinamoney.com.cn/ags/ms/cm-u-dlrp/PrDlyBltn"
)
FORBIDDEN_DR007_SUBSTITUTES = frozenset(
    {"FDR007", "R007", "FR007", "EXCHANGE_REPO_R_007"}
)


@dataclass(frozen=True)
class PbocNoticeResult:
    """一篇央行公开市场业务交易公告的 7 天期解析结果。"""

    notice_date: date
    published_at: datetime
    title: str
    announcement_number: str
    source_url: str
    raw_sha256: str
    has_seven_day_row: bool
    seven_day_operation_amount_100m: float | None
    seven_day_rate_percent: float | None
    parse_status: str


def sha256_bytes(payload: bytes) -> str:
    """计算字节对象的 SHA-256。"""

    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _six_digit_code(value: object, *, field: str) -> str:
    raw = _normalized_text(value)
    match = re.fullmatch(r"(?:\D*)(\d{6})(?:\D*)", raw)
    if match is None:
        raise ValueError(f"{field} 不是六位证券代码：{value!r}")
    return match.group(1)


def normalize_a_share_symbol(value: object) -> str:
    """把申万六位代码规范化为项目统一的 A 股代码。"""

    code = _six_digit_code(value, field="股票代码")
    if code.startswith(("4", "8", "9")):
        return f"{code}.BJ"
    if code.startswith(("5", "6")):
        return f"{code}.SH"
    return f"{code}.SZ"


def parse_sw_official_workbooks(
    history_payload: bytes,
    codebook_payload: bytes,
    *,
    retrieved_at: str,
    observation_cutoff: date = OBSERVATION_CUTOFF,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """解析申万官方历史变更表和 2021 版代码表。

    ``更新日期`` 被保留为保守的最早可证明可得时钟。它不被解释为行业
    生效日，也不会把后来维护的历史记录倒灌到更早交易日。
    """

    history = pd.read_excel(io.BytesIO(history_payload), dtype=str)
    codebook = pd.read_excel(io.BytesIO(codebook_payload), dtype=str)
    expected_history_columns = ["股票代码", "计入日期", "行业代码", "更新日期"]
    expected_codebook_columns = [
        "行业代码",
        "一级行业名称",
        "二级行业名称",
        "三级行业名称",
    ]
    if list(history.columns) != expected_history_columns:
        raise ValueError(
            "申万历史行业工作簿列发生漂移："
            f"期望={expected_history_columns}，实际={list(history.columns)}"
        )
    if list(codebook.columns) != expected_codebook_columns:
        raise ValueError(
            "申万行业代码工作簿列发生漂移："
            f"期望={expected_codebook_columns}，实际={list(codebook.columns)}"
        )
    if history.empty or codebook.empty:
        raise ValueError("申万官方工作簿为空")

    normalized = history.rename(
        columns={
            "股票代码": "stock_code",
            "计入日期": "effective_date",
            "行业代码": "industry_code",
            "更新日期": "record_updated_at",
        }
    ).copy()
    normalized["stock_code"] = normalized["stock_code"].map(
        lambda value: _six_digit_code(value, field="股票代码")
    )
    normalized["symbol"] = normalized["stock_code"].map(normalize_a_share_symbol)
    normalized["industry_code"] = normalized["industry_code"].map(
        lambda value: _six_digit_code(value, field="行业代码")
    )
    normalized["industry_l1_code"] = normalized["industry_code"].str[:2]
    normalized["effective_date"] = pd.to_datetime(
        normalized["effective_date"], errors="coerce"
    ).astype("datetime64[ns]").dt.normalize()
    normalized["record_updated_at"] = pd.to_datetime(
        normalized["record_updated_at"], errors="coerce"
    ).astype("datetime64[ns]")
    if normalized[["effective_date", "record_updated_at"]].isna().any().any():
        raise ValueError("申万历史行业工作簿含不可解析日期")
    if normalized.duplicated(["stock_code", "effective_date"]).any():
        duplicates = normalized.loc[
            normalized.duplicated(["stock_code", "effective_date"], keep=False),
            ["stock_code", "effective_date"],
        ].head(10)
        raise ValueError(f"申万历史行业工作簿含重复证券生效日：{duplicates.to_dict('records')}")

    normalized = normalized.sort_values(
        ["stock_code", "effective_date", "record_updated_at"]
    ).reset_index(drop=True)
    normalized["out_date"] = normalized.groupby("stock_code", sort=False)[
        "effective_date"
    ].shift(-1)
    normalized["classification_standard"] = np.where(
        normalized["effective_date"] >= pd.Timestamp("2021-07-30"),
        "SW_2021",
        "SW_2014",
    )
    normalized["classification_usage"] = (
        "POINT_IN_TIME_BITEMPORAL_RECORD_UPDATE_CLOCK"
    )
    normalized["availability_clock"] = (
        "RECORD_UPDATED_AT_AS_CONSERVATIVE_EARLIEST_PROVABLE_AVAILABILITY"
    )
    normalized["availability_timezone"] = SHANGHAI_TIMEZONE
    normalized["source"] = "申万宏源研究官方行业分类历史变更表"
    normalized["source_url"] = SW_HISTORY_URL
    normalized["source_sha256"] = sha256_bytes(history_payload)
    normalized["retrieved_at"] = retrieved_at

    full_row_count = int(len(normalized))
    normalized = normalized.loc[
        normalized["effective_date"] <= pd.Timestamp(observation_cutoff)
    ].reset_index(drop=True)

    codebook_normalized = codebook.rename(
        columns={
            "行业代码": "industry_code",
            "一级行业名称": "industry_l1_name",
            "二级行业名称": "industry_l2_name",
            "三级行业名称": "industry_l3_name",
        }
    ).copy()
    codebook_normalized["industry_code"] = codebook_normalized[
        "industry_code"
    ].map(lambda value: _six_digit_code(value, field="行业代码"))
    codebook_normalized["industry_l1_code"] = codebook_normalized[
        "industry_code"
    ].str[:2]
    codebook_normalized["source"] = "申万宏源研究官方2021版行业代码表"
    codebook_normalized["source_url"] = SW_CODEBOOK_URL
    codebook_normalized["source_sha256"] = sha256_bytes(codebook_payload)
    codebook_normalized["retrieved_at"] = retrieved_at
    if codebook_normalized["industry_code"].duplicated().any():
        raise ValueError("申万 2021 版行业代码表含重复行业代码")

    metadata = {
        "history_raw_sha256": sha256_bytes(history_payload),
        "history_raw_bytes": len(history_payload),
        "codebook_raw_sha256": sha256_bytes(codebook_payload),
        "codebook_raw_bytes": len(codebook_payload),
        "history_full_row_count": full_row_count,
        "history_rows_effective_by_cutoff": int(len(normalized)),
        "history_stock_count": int(normalized["stock_code"].nunique()),
        "history_first_effective_date": normalized["effective_date"]
        .min()
        .date()
        .isoformat(),
        "history_last_effective_date_by_cutoff": normalized["effective_date"]
        .max()
        .date()
        .isoformat(),
        "history_last_record_updated_at": normalized["record_updated_at"]
        .max()
        .isoformat(),
        "codebook_row_count": int(len(codebook_normalized)),
        "retrieved_at": retrieved_at,
    }
    return normalized, codebook_normalized, metadata


def _select_industry_events_for_member_dates(
    member_dates: np.ndarray,
    events: pd.DataFrame,
    *,
    enforce_record_availability: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if events.empty:
        return (
            np.zeros(len(member_dates), dtype=bool),
            np.zeros(len(member_dates), dtype=np.int64),
        )
    effective = events["effective_date"].to_numpy(dtype="datetime64[ns]")
    valid = effective[None, :] <= member_dates[:, None]
    if enforce_record_availability:
        updated = events["record_updated_at"].to_numpy(dtype="datetime64[ns]")
        market_close = member_dates + np.timedelta64(15, "h")
        valid &= updated[None, :] <= market_close[:, None]
    minimum = np.iinfo(np.int64).min
    scores = np.where(valid, effective[None, :].astype(np.int64), minimum)
    selected = scores.argmax(axis=1)
    return valid.any(axis=1), selected


def build_csi300_member_day_sw_industry(
    membership: pd.DataFrame,
    history: pd.DataFrame,
    *,
    observation_start: date = OBSERVATION_START,
    observation_cutoff: date = OBSERVATION_CUTOFF,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """以申万记录更新时间作可得时钟，生成 CSI300 成分日行业映射。"""

    required_membership = {"membership_date", "symbol"}
    required_history = {
        "symbol",
        "industry_code",
        "industry_l1_code",
        "effective_date",
        "record_updated_at",
        "classification_standard",
        "source_sha256",
    }
    missing_membership = sorted(required_membership.difference(membership.columns))
    missing_history = sorted(required_history.difference(history.columns))
    if missing_membership:
        raise ValueError(f"CSI300 点时成分缺少列：{missing_membership}")
    if missing_history:
        raise ValueError(f"申万历史事件缺少列：{missing_history}")

    members = membership.loc[:, ["membership_date", "symbol"]].copy()
    members["membership_date"] = pd.to_datetime(
        members["membership_date"], errors="coerce"
    ).astype("datetime64[ns]").dt.normalize()
    if members["membership_date"].isna().any():
        raise ValueError("CSI300 点时成分含不可解析日期")
    members = members.loc[
        members["membership_date"].between(
            pd.Timestamp(observation_start), pd.Timestamp(observation_cutoff)
        )
    ].copy()
    if members.duplicated(["membership_date", "symbol"]).any():
        raise ValueError("CSI300 点时成分含重复 member-day")
    counts = members.groupby("membership_date")["symbol"].nunique()
    if counts.empty or not counts.eq(300).all():
        raise ValueError("CSI300 点时成分不是每个交易日严格 300 只")

    source_digest_values = set(history["source_sha256"].astype(str))
    if len(source_digest_values) != 1:
        raise ValueError("申万历史事件包含多个或空来源哈希")
    history_groups = {
        symbol: group.sort_values(["effective_date", "record_updated_at"])
        .reset_index(drop=True)
        for symbol, group in history.groupby("symbol", sort=False)
    }

    output_parts: list[pd.DataFrame] = []
    structural_covered = 0
    for symbol, group in members.groupby("symbol", sort=False):
        group = group.sort_values("membership_date").reset_index(drop=True)
        member_dates = group["membership_date"].to_numpy(dtype="datetime64[ns]")
        events = history_groups.get(symbol, history.iloc[0:0])
        structural_valid, _ = _select_industry_events_for_member_dates(
            member_dates,
            events,
            enforce_record_availability=False,
        )
        available, selected = _select_industry_events_for_member_dates(
            member_dates,
            events,
            enforce_record_availability=True,
        )
        structural_covered += int(structural_valid.sum())
        part = group.copy()
        part["industry_code"] = pd.Series(pd.NA, index=part.index, dtype="string")
        part["industry_l1_code"] = pd.Series(
            pd.NA, index=part.index, dtype="string"
        )
        part["event_effective_date"] = pd.NaT
        part["record_available_at"] = pd.NaT
        part["classification_standard"] = pd.Series(
            pd.NA, index=part.index, dtype="string"
        )
        if available.any():
            chosen = events.iloc[selected[available]].reset_index(drop=True)
            available_indices = np.flatnonzero(available)
            part.loc[available_indices, "industry_code"] = chosen[
                "industry_code"
            ].astype("string").to_numpy()
            part.loc[available_indices, "industry_l1_code"] = chosen[
                "industry_l1_code"
            ].astype("string").to_numpy()
            part.loc[available_indices, "event_effective_date"] = chosen[
                "effective_date"
            ].to_numpy()
            part.loc[available_indices, "record_available_at"] = chosen[
                "record_updated_at"
            ].to_numpy()
            part.loc[available_indices, "classification_standard"] = chosen[
                "classification_standard"
            ].astype("string").to_numpy()
        part["mapping_status"] = np.where(
            available,
            "PIT_AVAILABLE_BY_MARKET_CLOSE",
            "NO_VIEW_NO_PROVABLE_RECORD_BY_MARKET_CLOSE",
        )
        part["availability_clock"] = (
            "RECORD_UPDATED_AT_NOT_LATER_THAN_MEMBERSHIP_DATE_15_00_ASIA_SHANGHAI"
        )
        part["source"] = "申万宏源研究官方行业分类历史变更表"
        part["source_sha256"] = next(iter(source_digest_values))
        output_parts.append(part)

    output = pd.concat(output_parts, ignore_index=True).sort_values(
        ["membership_date", "symbol"]
    ).reset_index(drop=True)
    output["event_effective_date"] = pd.to_datetime(
        output["event_effective_date"], errors="coerce"
    ).astype("datetime64[ns]")
    output["record_available_at"] = pd.to_datetime(
        output["record_available_at"], errors="coerce"
    ).astype("datetime64[ns]")
    available_mask = output["mapping_status"].eq("PIT_AVAILABLE_BY_MARKET_CLOSE")
    daily_counts = output.assign(_available=available_mask).groupby(
        "membership_date"
    )["_available"].sum()
    no_view = ~available_mask
    structural_missing = int(len(output) - structural_covered)
    metrics = {
        "member_day_count": int(len(output)),
        "session_count": int(output["membership_date"].nunique()),
        "structural_effective_date_covered_member_days": int(structural_covered),
        "structural_effective_date_missing_member_days": structural_missing,
        "pit_available_member_days": int(available_mask.sum()),
        "pit_no_view_member_days": int(no_view.sum()),
        "pit_coverage_ratio": float(available_mask.mean()),
        "full_300_member_session_count": int(daily_counts.eq(300).sum()),
        "no_view_session_count": int(daily_counts.lt(300).sum()),
        "minimum_available_member_count": int(daily_counts.min()),
        "first_full_300_member_session": (
            daily_counts.loc[daily_counts.eq(300)].index.min().date().isoformat()
            if daily_counts.eq(300).any()
            else None
        ),
        "first_session": output["membership_date"].min().date().isoformat(),
        "last_session": output["membership_date"].max().date().isoformat(),
        "industry_source_sha256": next(iter(source_digest_values)),
        "missing_value_rule": "NO_VIEW_NO_INTERPOLATION",
    }
    return output, metrics


def _meta_content(soup: BeautifulSoup, name: str) -> str | None:
    node = soup.find("meta", attrs={"name": name})
    if node is None:
        return None
    value = node.get("content")
    return _normalized_text(value) or None


def _direct_row_cells(row: Any) -> list[str]:
    cells = row.find_all(["td", "th"], recursive=False)
    return [_normalized_text(cell.get_text(" ", strip=True)) for cell in cells]


def parse_pboc_open_market_notice(
    payload: bytes,
    *,
    source_url: str,
) -> PbocNoticeResult:
    """从央行官方公告 HTML 中提取 7 天期逆回购实际披露值。"""

    try:
        html = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("央行公告不是 UTF-8 HTML") from exc
    soup = BeautifulSoup(html, "html.parser")
    title = _meta_content(soup, "ArticleTitle")
    if title is None and soup.title is not None:
        title = _normalized_text(soup.title.get_text(" ", strip=True))
    if not title or "公开市场业务交易公告" not in title:
        raise ValueError(f"央行公告标题身份不一致：{title!r}")
    notice_date_text = _meta_content(soup, "PubDate")
    if notice_date_text is None:
        raise ValueError("央行公告缺少 PubDate")
    notice_date = date.fromisoformat(notice_date_text)
    visible_text = _normalized_text(soup.get_text(" ", strip=True))
    timestamp_match = re.search(
        r"文章来源[：:]?\s*(20\d{2}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})",
        visible_text,
    )
    if timestamp_match is None:
        raise ValueError("央行公告缺少原始发布时间")
    published_at = datetime.strptime(
        timestamp_match.group(1), "%Y-%m-%d %H:%M:%S"
    )
    if published_at.date() != notice_date:
        raise ValueError("央行公告 PubDate 与正文发布时间日期不一致")
    announcement_match = re.search(r"\[(20\d{2})\]\s*第\s*(\d+)\s*号", title)
    announcement_number = (
        f"{announcement_match.group(1)}-{announcement_match.group(2)}"
        if announcement_match is not None
        else "UNNUMBERED"
    )

    candidate_rows: list[tuple[str, ...]] = []
    for row in soup.find_all("tr"):
        cells = _direct_row_cells(row)
        if len(cells) < 2:
            continue
        tenor = re.sub(r"\s+", "", cells[0])
        if re.fullmatch(r"7天(?:期)?", tenor):
            candidate_rows.append(tuple(cells))
    unique_rows = list(dict.fromkeys(candidate_rows))
    if not unique_rows:
        return PbocNoticeResult(
            notice_date=notice_date,
            published_at=published_at,
            title=title,
            announcement_number=announcement_number,
            source_url=source_url,
            raw_sha256=sha256_bytes(payload),
            has_seven_day_row=False,
            seven_day_operation_amount_100m=None,
            seven_day_rate_percent=None,
            parse_status="NO_7D_OPERATION_ROW",
        )

    parsed_rows: list[tuple[float | None, float | None]] = []
    for cells in unique_rows:
        rate_values: list[float] = []
        amount_values: list[float] = []
        for cell in cells[1:]:
            numeric_cell = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", cell)
            numeric_cell = re.sub(r"(?<=\d)\s+(?=\d)", "", numeric_cell)
            numeric_cell = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", numeric_cell)
            rate_values.extend(
                float(value)
                for value in re.findall(
                    r"([0-9]+(?:\.[0-9]+)?)\s*%", numeric_cell
                )
            )
            amount_values.extend(
                float(value)
                for value in re.findall(
                    r"([0-9]+(?:\.[0-9]+)?)\s*(?:亿\s*元|亿元)",
                    numeric_cell,
                )
            )
        rates = sorted(set(rate_values))
        if len(rates) > 1:
            raise ValueError(f"央行公告同一 7 天期行出现多个利率：{cells}")
        rate = rates[0] if rates else None
        amount = amount_values[-1] if amount_values else None
        parsed_rows.append((amount, rate))
    unique_parsed = list(dict.fromkeys(parsed_rows))
    if len(unique_parsed) != 1:
        raise ValueError(f"央行公告出现冲突的 7 天期行：{unique_rows}")
    amount, rate = unique_parsed[0]
    if rate is not None and not 0.0 < rate < 20.0:
        raise ValueError(f"央行 7 天期逆回购利率越界：{rate}")
    if amount is not None and amount < 0.0:
        raise ValueError(f"央行 7 天期逆回购操作量为负：{amount}")
    if rate is not None:
        status = "PUBLISHED_7D_OPERATION_RATE"
    elif amount == 0.0:
        status = "ZERO_7D_OPERATION_RATE_NOT_PUBLISHED"
    else:
        status = "UNPARSED_NONZERO_7D_OPERATION"
    return PbocNoticeResult(
        notice_date=notice_date,
        published_at=published_at,
        title=title,
        announcement_number=announcement_number,
        source_url=source_url,
        raw_sha256=sha256_bytes(payload),
        has_seven_day_row=True,
        seven_day_operation_amount_100m=amount,
        seven_day_rate_percent=rate,
        parse_status=status,
    )


def summarize_pboc_policy_rate_ledger(ledger: pd.DataFrame) -> dict[str, Any]:
    """核验央行公告解析结果，并输出实际披露的利率状态变更。"""

    required = {
        "notice_date",
        "published_at",
        "source_url",
        "raw_path",
        "raw_sha256",
        "parse_status",
        "seven_day_operation_amount_100m",
        "seven_day_rate_percent",
    }
    missing = sorted(required.difference(ledger.columns))
    if missing:
        raise ValueError(f"央行公告账本缺少列：{missing}")
    work = ledger.copy()
    work["notice_date"] = pd.to_datetime(
        work["notice_date"], errors="coerce"
    ).astype("datetime64[ns]").dt.normalize()
    work["published_at"] = pd.to_datetime(work["published_at"], errors="coerce")
    if work[["notice_date", "published_at"]].isna().any().any():
        raise ValueError("央行公告账本含不可解析日期")
    unparsed = work["parse_status"].eq("UNPARSED_NONZERO_7D_OPERATION")
    if unparsed.any():
        examples = work.loc[unparsed, ["notice_date", "source_url"]].head(10)
        raise ValueError(f"央行非零 7 天期操作未解析利率：{examples.to_dict('records')}")
    rates = pd.to_numeric(work["seven_day_rate_percent"], errors="coerce")
    invalid_rate = rates.notna() & ((rates <= 0) | (rates >= 20))
    if invalid_rate.any():
        raise ValueError("央行公告账本含越界 7 天期利率")
    published = work.loc[rates.notna()].copy()
    published["seven_day_rate_percent"] = rates.loc[rates.notna()].astype(float)
    daily_rate_counts = published.groupby("notice_date")[
        "seven_day_rate_percent"
    ].nunique()
    if daily_rate_counts.gt(1).any():
        raise ValueError("同一公告日出现冲突的央行 7 天期逆回购利率")
    daily = (
        published.sort_values(["notice_date", "published_at"])
        .drop_duplicates("notice_date", keep="last")
        .reset_index(drop=True)
    )
    daily["previous_rate_percent"] = daily["seven_day_rate_percent"].shift(1)
    changed = daily["previous_rate_percent"].isna() | ~np.isclose(
        daily["seven_day_rate_percent"],
        daily["previous_rate_percent"],
        rtol=0.0,
        atol=1e-12,
    )
    changes = daily.loc[
        changed,
        [
            "notice_date",
            "published_at",
            "seven_day_rate_percent",
            "previous_rate_percent",
            "source_url",
            "raw_path",
            "raw_sha256",
        ],
    ].reset_index(drop=True)
    return {
        "notice_count": int(len(work)),
        "notice_date_count": int(work["notice_date"].nunique()),
        "notice_first_date": work["notice_date"].min().date().isoformat(),
        "notice_last_date": work["notice_date"].max().date().isoformat(),
        "published_7d_rate_row_count": int(len(published)),
        "published_7d_rate_date_count": int(len(daily)),
        "zero_operation_without_published_rate_count": int(
            work["parse_status"].eq(
                "ZERO_7D_OPERATION_RATE_NOT_PUBLISHED"
            ).sum()
        ),
        "no_7d_operation_row_count": int(
            work["parse_status"].eq("NO_7D_OPERATION_ROW").sum()
        ),
        "unique_rate_values": sorted(
            float(value) for value in daily["seven_day_rate_percent"].unique()
        ),
        "rate_change_count": int(len(changes)),
        "rate_changes": changes,
        "missing_value_rule": "NO_VIEW_WHEN_RATE_NOT_ACTUALLY_PUBLISHED",
        "interpolation_performed": False,
        "inferred_change_dates_used": False,
    }


def validate_dr007_daily(
    frame: pd.DataFrame,
    *,
    market_dates: Sequence[object],
    source_identity: str,
    observation_start: date = OBSERVATION_START,
    observation_cutoff: date = OBSERVATION_CUTOFF,
) -> dict[str, Any]:
    """核验许可来源导入的 DR007 日度加权平均利率。"""

    normalized_identity = source_identity.upper().strip()
    if normalized_identity in FORBIDDEN_DR007_SUBSTITUTES:
        raise ValueError(f"禁止把 {source_identity} 作为 DR007")
    required = {"date", "dr007"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"DR007 日度序列缺少列：{missing}")
    work = frame.loc[:, ["date", "dr007"]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").astype(
        "datetime64[ns]"
    ).dt.normalize()
    work["dr007"] = pd.to_numeric(work["dr007"], errors="coerce")
    if work.isna().any().any():
        raise ValueError("DR007 日度序列含空值或不可解析值")
    if work["date"].duplicated().any():
        raise ValueError("DR007 日度序列含重复日期")
    if ((work["dr007"] <= 0) | (work["dr007"] >= 20)).any():
        raise ValueError("DR007 日度序列含越界利率")
    work = work.sort_values("date").reset_index(drop=True)
    if work.empty:
        raise ValueError("DR007 日度序列为空")
    if work["date"].min() > pd.Timestamp(observation_start):
        raise ValueError("DR007 日度序列没有覆盖冻结观察起点")
    if work["date"].max() < pd.Timestamp(observation_cutoff):
        raise ValueError("DR007 日度序列没有覆盖冻结观察截止日")
    source_dates = pd.DatetimeIndex(
        work.loc[
            work["date"].between(
                pd.Timestamp(observation_start), pd.Timestamp(observation_cutoff)
            ),
            "date",
        ]
    )
    expected_market_dates = pd.DatetimeIndex(
        pd.to_datetime(list(market_dates), errors="coerce")
    ).normalize()
    expected_market_dates = expected_market_dates[
        (expected_market_dates >= pd.Timestamp(observation_start))
        & (expected_market_dates <= pd.Timestamp(observation_cutoff))
    ]
    missing_market_dates = expected_market_dates.difference(source_dates)
    return {
        "row_count": int(len(work)),
        "first_date": work["date"].min().date().isoformat(),
        "last_date": work["date"].max().date().isoformat(),
        "market_session_count": int(len(expected_market_dates)),
        "covered_market_session_count": int(
            len(expected_market_dates) - len(missing_market_dates)
        ),
        "missing_market_session_count": int(len(missing_market_dates)),
        "missing_market_session_examples": [
            value.date().isoformat() for value in missing_market_dates[:20]
        ],
        "source_identity": source_identity,
        "selected_series": "DR007",
        "weighted_average_field_validated": True,
        "forbidden_substitute_used": False,
        "missing_value_rule": "NO_VIEW_NO_INTERPOLATION",
    }
