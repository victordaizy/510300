"""结算双影子V1不可变同日前瞻claim的真实Shadow绩效。

正式前瞻验收只使用连续、同日形成且未回填的claim。每个signal date的状态
在下一交易日开盘执行；使用冻结的50万元、100股整数手、现金收益、佣金和
基础/压力滑点，与H00300全收益指数比较。五个起点各需要至少242个执行日，
因此完整前瞻资格要求最新连续claim块至少246个已执行交易日。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "research") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "research"))

import validate_etf_microstructure_dual_shadow_v1_frozen as frozen
from binary_state_feasibility_v1 import (
    build_benchmark_ledger,
    simulate_binary_path,
    summarize_path,
)


PROJECT_ID = "510300_ETF_MICROSTRUCTURE_DUAL_SHADOW_V1"
POLICY = "MICRO_CONSENSUS3_OR_SHORT_EXTREMES_DUAL_SHADOW"
TIMEZONE = ZoneInfo("Asia/Shanghai")
CALENDAR_ROOT = PROJECT_ROOT / "data" / "reference"
IMPLEMENTATION_RECEIPT = (
    PROJECT_ROOT
    / "reports"
    / "frozen"
    / "510300_etf_microstructure_dual_shadow_v1_prospective_settlement_receipt.json"
)
CLAIM_ROOT = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "510300_etf_microstructure_dual_shadow_v1"
    / "prospective_claims"
)
RUN_ROOT = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "510300_etf_microstructure_dual_shadow_v1"
    / "prospective_validation_runs"
)
STATUS_FILE = (
    PROJECT_ROOT
    / "reports"
    / "forward"
    / "510300_etf_microstructure_dual_shadow_v1_prospective_validation_status.json"
)
DIVIDEND_SNAPSHOT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "510300_etf_microstructure_dual_shadow_v1"
    / "dividend_snapshots"
)
REQUIRED_EXECUTION_DAYS_PER_START = 242
REQUIRED_LATEST_CONTIGUOUS_EXECUTION_DAYS = 246


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def _verify_implementation() -> dict[str, str]:
    receipt = json.loads(IMPLEMENTATION_RECEIPT.read_text(encoding="utf-8"))
    if receipt.get("project_id") != PROJECT_ID:
        raise ValueError("前瞻结算实现收据项目编号不匹配")
    key = "research\\settle_510300_etf_microstructure_dual_shadow_v1_prospective.py"
    expected = receipt["implementations"][key]["sha256"]
    actual = _sha256(Path(__file__).resolve())
    if actual != expected:
        raise ValueError(
            f"前瞻结算器实现哈希漂移：expected={expected},actual={actual}"
        )
    return {
        "file": _relative(Path(__file__).resolve()),
        "sha256": actual,
        "receipt": _relative(IMPLEMENTATION_RECEIPT),
        "receipt_sha256": _sha256(IMPLEMENTATION_RECEIPT),
    }


def _calendar() -> tuple[pd.DatetimeIndex, dict[str, Any]]:
    files = sorted(CALENDAR_ROOT.glob("sse_trade_calendar_[0-9][0-9][0-9][0-9].csv"))
    if not files:
        raise FileNotFoundError("没有可用的上交所年度交易日历")
    frames: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for path in files:
        year = int(path.stem.rsplit("_", 1)[1])
        metadata_path = path.with_suffix(".metadata.json")
        calendar_receipt_path = (
            PROJECT_ROOT
            / "reports"
            / "frozen"
            / f"510300_etf_microstructure_dual_shadow_v1_calendar_{year}_receipt.json"
        )
        if not metadata_path.exists():
            raise FileNotFoundError(f"年度交易日历缺少元数据：{metadata_path}")
        if not calendar_receipt_path.exists():
            raise FileNotFoundError(f"年度交易日历缺少冻结收据：{calendar_receipt_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        calendar_receipt = json.loads(
            calendar_receipt_path.read_text(encoding="utf-8")
        )
        if (
            calendar_receipt.get("project_id") != PROJECT_ID
            or int(calendar_receipt.get("year")) != year
            or calendar_receipt.get("status") != "OFFICIAL_ANNUAL_CALENDAR_FROZEN"
            or calendar_receipt.get("calendar_file") != _relative(path)
            or calendar_receipt.get("calendar_sha256") != _sha256(path)
            or calendar_receipt.get("metadata_file") != _relative(metadata_path)
            or calendar_receipt.get("metadata_sha256") != _sha256(metadata_path)
        ):
            raise ValueError(f"年度交易日历冻结收据不匹配：{calendar_receipt_path}")
        if metadata.get("status") != "PASS" or int(metadata.get("year")) != year:
            raise ValueError(f"年度交易日历元数据状态或年份无效：{metadata_path}")
        if metadata.get("sha256") != _sha256(path):
            raise ValueError(f"年度交易日历哈希漂移：{path}")
        if metadata.get("file") != _relative(path):
            raise ValueError(f"年度交易日历元数据文件路径不匹配：{metadata_path}")
        official_source = str(metadata.get("official_source", ""))
        host = (urlparse(official_source).hostname or "").lower()
        if host != "sse.com.cn" and not host.endswith(".sse.com.cn"):
            raise ValueError(f"年度交易日历官方来源不是上交所：{metadata_path}")
        frame = pd.read_csv(path, dtype=str)
        if "trade_date" not in frame.columns:
            raise ValueError(f"上交所年度交易日历缺少trade_date：{path}")
        frame = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    frame["trade_date"], errors="raise"
                ).dt.normalize()
            }
        )
        if not frame["trade_date"].dt.year.eq(year).all():
            raise ValueError(f"年度交易日历含跨年日期：{path}")
        if frame["trade_date"].duplicated().any():
            raise ValueError(f"年度交易日历存在重复日期：{path}")
        frames.append(frame)
        audits.append(
            {
                "year": year,
                "file": _relative(path),
                "sha256": _sha256(path),
                "metadata": _relative(metadata_path),
                "metadata_sha256": _sha256(metadata_path),
                "freeze_receipt": _relative(calendar_receipt_path),
                "freeze_receipt_sha256": _sha256(calendar_receipt_path),
                "official_source": official_source,
                "trading_day_count": int(len(frame)),
                "coverage_start": frame["trade_date"].min().date().isoformat(),
                "coverage_end": frame["trade_date"].max().date().isoformat(),
            }
        )
    combined = pd.concat(frames, ignore_index=True)
    if combined["trade_date"].duplicated().any():
        raise ValueError("多个年度交易日历之间存在重复日期")
    result = pd.DatetimeIndex(combined["trade_date"].sort_values().unique())
    audit = {
        "status": "PASS",
        "annual_files": audits,
        "trading_day_count": int(len(result)),
        "coverage_start": result.min().date().isoformat(),
        "coverage_end": result.max().date().isoformat(),
        "future_year_extension_contract": "新增年度必须有上交所来源元数据和匹配SHA-256",
    }
    return result, audit


def _next_trade_date(date: pd.Timestamp, calendar: pd.DatetimeIndex) -> pd.Timestamp:
    later = calendar[calendar > date]
    if len(later) == 0:
        raise ValueError(f"交易日历没有{date.date()}之后的交易日")
    return pd.Timestamp(later[0])


def _verify_claim_evidence(claim: dict[str, Any]) -> None:
    evidence = claim["evidence"]
    for file_key, hash_key in [
        ("base_snapshot_manifest", "base_snapshot_manifest_sha256"),
        ("sse_snapshot_manifest", "sse_snapshot_manifest_sha256"),
        ("replay_report", "replay_report_sha256"),
    ]:
        path = PROJECT_ROOT / evidence[file_key]
        actual = _sha256(path) if path.exists() else "MISSING"
        if actual != evidence[hash_key]:
            raise ValueError(
                f"前瞻claim证据哈希漂移：{file_key},expected={evidence[hash_key]},actual={actual}"
            )


def _load_claims(
    calendar: pd.DatetimeIndex, frozen_at: pd.Timestamp
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    if not CLAIM_ROOT.exists():
        return pd.DataFrame(), audits
    for path in sorted(CLAIM_ROOT.glob("*.json")):
        claim = json.loads(path.read_text(encoding="utf-8"))
        if claim.get("project_id") != PROJECT_ID:
            raise ValueError(f"前瞻claim项目编号不匹配：{path}")
        if claim.get("status") != "PROSPECTIVE_SIGNAL_CLAIMED_SHADOW_ONLY":
            raise ValueError(f"前瞻claim状态无效：{path}")
        _verify_claim_evidence(claim)
        signal_date = pd.Timestamp(claim["signal_date"]).normalize()
        claimed_at = pd.Timestamp(claim["claimed_at_asia_shanghai"])
        if signal_date not in calendar:
            raise ValueError(f"前瞻claim不是交易日：{signal_date.date()}")
        if claimed_at <= frozen_at:
            raise ValueError(f"前瞻claim早于或等于候选冻结时刻：{path}")
        if claimed_at.tz_localize(None).normalize() != signal_date:
            raise ValueError(f"前瞻claim不是同日形成：{path}")
        expected_execution = _next_trade_date(signal_date, calendar)
        if claim["expected_t_plus_1_execution_date"] != expected_execution.date().isoformat():
            raise ValueError(f"前瞻claim的T+1执行日不匹配：{path}")
        state = int(claim["fixed_rule_state"]["state_final"])
        if state not in {0, 1}:
            raise ValueError(f"前瞻claim状态不是0或1：{path}")
        rows.append(
            {
                "signal_date": signal_date,
                "execution_date": expected_execution,
                "state_final": state,
                "cash_raw": bool(claim["fixed_rule_state"]["cash_raw"]),
                "shadow_gate_active": bool(
                    claim["fixed_rule_state"]["shadow_gate_active"]
                ),
                "claim_file": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "claim_sha256": _sha256(path),
                "claimed_at": claimed_at,
            }
        )
        audits.append(
            {
                "signal_date": signal_date.date().isoformat(),
                "claim_file": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "claim_sha256": _sha256(path),
                "evidence_hashes_pass": True,
                "same_day_claim": True,
            }
        )
    claims = pd.DataFrame(rows)
    if claims.empty:
        return claims, audits
    claims.sort_values("signal_date", inplace=True)
    claims.reset_index(drop=True, inplace=True)
    if claims["signal_date"].duplicated().any():
        raise ValueError("前瞻claim存在重复signal_date")
    return claims, audits


def _segments(
    claims: pd.DataFrame, calendar: pd.DatetimeIndex
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if claims.empty:
        return claims.assign(segment_id=pd.Series(dtype="int64")), []
    calendar_position = {date: index for index, date in enumerate(calendar)}
    segment_ids: list[int] = []
    segment_id = 0
    previous: pd.Timestamp | None = None
    for date in claims["signal_date"]:
        current = pd.Timestamp(date)
        if previous is not None and calendar_position[current] != calendar_position[previous] + 1:
            segment_id += 1
        segment_ids.append(segment_id)
        previous = current
    output = claims.copy()
    output["segment_id"] = segment_ids
    summary: list[dict[str, Any]] = []
    for identifier, group in output.groupby("segment_id", sort=True):
        first = pd.Timestamp(group["signal_date"].min())
        last = pd.Timestamp(group["signal_date"].max())
        expected = calendar[(calendar >= first) & (calendar <= last)]
        summary.append(
            {
                "segment_id": int(identifier),
                "first_signal_date": first.date().isoformat(),
                "last_signal_date": last.date().isoformat(),
                "claim_count": int(len(group)),
                "expected_trade_date_count": int(len(expected)),
                "complete": bool(len(group) == len(expected)),
            }
        )
    return output, summary


def _resolve_latest_snapshot_manifest(
    claims: pd.DataFrame, explicit: str | None
) -> Path | None:
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else PROJECT_ROOT / path
    if claims.empty:
        return None
    latest_claim = json.loads(
        (PROJECT_ROOT / claims.iloc[-1]["claim_file"]).read_text(encoding="utf-8")
    )
    return PROJECT_ROOT / latest_claim["evidence"]["sse_snapshot_manifest"]


def _load_snapshot_market(manifest_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("project_id") != PROJECT_ID:
        raise ValueError("结算快照项目编号不匹配")
    frames: dict[str, pd.DataFrame] = {}
    audit: dict[str, Any] = {
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "manifest_sha256": _sha256(manifest_path),
        "datasets": {},
    }
    for name in ["etf", "benchmark"]:
        record = manifest["datasets"][name]
        path = PROJECT_ROOT / record["file"]
        if not path.resolve().is_relative_to(manifest_path.parent.resolve()):
            raise ValueError(f"结算快照文件越出快照目录：{path}")
        actual = _sha256(path)
        if actual != record["sha256"]:
            raise ValueError(f"结算快照{name}哈希漂移")
        frame = pd.read_parquet(path)
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        frame.sort_values("date", inplace=True)
        frame.reset_index(drop=True, inplace=True)
        if frame["date"].duplicated().any():
            raise ValueError(f"结算快照{name}存在重复日期")
        frames[name] = frame
        audit["datasets"][name] = {
            "file": record["file"],
            "sha256": actual,
            "rows": int(len(frame)),
            "first_date": frame["date"].min().date().isoformat(),
            "last_date": frame["date"].max().date().isoformat(),
        }
    return frames["etf"], frames["benchmark"], audit


def _combined_market(
    config: dict[str, Any], snapshot_etf: pd.DataFrame, snapshot_benchmark: pd.DataFrame
) -> pd.DataFrame:
    paths = {
        name: PROJECT_ROOT / item["path"] for name, item in config["inputs"].items()
    }
    cutoff = pd.Timestamp(config["dates"]["frozen_execution_market_cutoff"])
    historical_etf = pd.read_parquet(paths["etf_execution"])
    historical_benchmark = pd.read_parquet(paths["benchmark_total_return"])
    for frame in [historical_etf, historical_benchmark]:
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    etf = pd.concat(
        [
            historical_etf.loc[historical_etf["date"].le(cutoff)],
            snapshot_etf.loc[snapshot_etf["date"].gt(cutoff)],
        ],
        ignore_index=True,
        sort=False,
    )
    benchmark = pd.concat(
        [
            historical_benchmark.loc[historical_benchmark["date"].le(cutoff)],
            snapshot_benchmark.loc[snapshot_benchmark["date"].gt(cutoff)],
        ],
        ignore_index=True,
        sort=False,
    )
    etf.sort_values("date", inplace=True)
    benchmark.sort_values("date", inplace=True)
    etf.drop_duplicates("date", keep="last", inplace=True)
    benchmark.drop_duplicates("date", keep="last", inplace=True)
    market = etf[["date", "open", "close"]].rename(
        columns={"open": "etf_open", "close": "etf_close"}
    )
    market = market.merge(
        benchmark[["date", "close"]].rename(columns={"close": "benchmark_close"}),
        on="date",
        how="inner",
        validate="one_to_one",
    )
    for column in ["etf_open", "etf_close", "benchmark_close"]:
        market[column] = pd.to_numeric(market[column], errors="raise")
    if market[["etf_open", "etf_close", "benchmark_close"]].isna().any(axis=None):
        raise ValueError("前瞻结算市场数据存在空值")
    market.sort_values("date", inplace=True)
    market.reset_index(drop=True, inplace=True)
    return market


def _base_dividends(config: dict[str, Any]) -> pd.DataFrame:
    path = PROJECT_ROOT / config["inputs"]["cash_distributions"]["path"]
    dividends = pd.read_csv(path)
    for column in ["record_date", "ex_date", "payment_date"]:
        dividends[column] = pd.to_datetime(dividends[column], errors="raise").dt.normalize()
    dividends["cash_dividend_per_share"] = pd.to_numeric(
        dividends["cash_dividend_per_share"], errors="raise"
    )
    dividends.sort_values("ex_date", inplace=True)
    dividends.reset_index(drop=True, inplace=True)
    return dividends


def _resolve_latest_dividend_manifest(explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else PROJECT_ROOT / path
    if not DIVIDEND_SNAPSHOT_ROOT.exists():
        return None
    candidates = sorted(DIVIDEND_SNAPSHOT_ROOT.glob("*/manifest.json"))
    return candidates[-1] if candidates else None


def _verified_event_frame(events: list[dict[str, Any]]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(
            columns=[
                "symbol",
                "record_date",
                "ex_date",
                "payment_date",
                "cash_dividend_per_share",
                "source",
            ]
        )
    rows: list[dict[str, Any]] = []
    for event in events:
        record_date = pd.Timestamp(event["record_date"]).normalize()
        ex_date = pd.Timestamp(event["ex_date"]).normalize()
        payment_date = pd.Timestamp(event["payment_date"]).normalize()
        if not record_date < ex_date <= payment_date:
            raise ValueError(f"官方核验分红事件日期顺序无效：{ex_date.date()}")
        amount = float(event["cash_dividend_per_share"])
        if not np.isfinite(amount) or amount <= 0:
            raise ValueError(f"官方核验分红事件金额无效：{ex_date.date()}")
        host = (urlparse(event["source_url"]).hostname or "").lower()
        if host != "sse.com.cn" and not host.endswith(".sse.com.cn"):
            raise ValueError(f"官方核验分红来源不是上交所：{ex_date.date()}")
        rows.append(
            {
                "symbol": "510300.SH",
                "record_date": record_date,
                "ex_date": ex_date,
                "payment_date": payment_date,
                "cash_dividend_per_share": amount,
                "source": event["source_url"],
            }
        )
    frame = pd.DataFrame(rows)
    if frame["ex_date"].duplicated().any():
        raise ValueError("官方核验的新分红事件存在重复除息日")
    return frame


def _load_dividends(
    config: dict[str, Any],
    candidate_manifest: dict[str, Any],
    manifest_path: Path | None,
) -> tuple[pd.DataFrame, dict[str, Any], bool]:
    base = _base_dividends(config)
    base_path = PROJECT_ROOT / config["inputs"]["cash_distributions"]["path"]
    frozen_key = str(base_path.relative_to(PROJECT_ROOT)).replace("/", "\\")
    expected_base_hash = candidate_manifest["historical_input_files_at_freeze"][
        frozen_key
    ]
    actual_base_hash = _sha256(base_path)
    if actual_base_hash != expected_base_hash:
        raise ValueError("前瞻结算发现冻结分红表哈希漂移")
    if manifest_path is None:
        return (
            base,
            {
                "status": "MISSING_DIVIDEND_SNAPSHOT",
                "settlement_qualification_allowed": False,
                "frozen_distribution_file": _relative(base_path),
                "frozen_distribution_sha256": actual_base_hash,
            },
            False,
        )

    resolved = manifest_path.resolve()
    if not resolved.is_relative_to(DIVIDEND_SNAPSHOT_ROOT.resolve()):
        raise ValueError("分红快照清单必须位于前瞻分红快照目录内")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("project_id") != PROJECT_ID or payload.get("schema_version") != 1:
        raise ValueError("分红快照清单项目编号或版本无效")
    implementation = payload.get("implementation", {})
    implementation_receipt = (PROJECT_ROOT / implementation.get("receipt", "")).resolve()
    collector_file = (PROJECT_ROOT / implementation.get("file", "")).resolve()
    expected_collector_file = (
        PROJECT_ROOT
        / "scripts"
        / "collect_510300_etf_microstructure_dual_shadow_v1_dividend_snapshot.py"
    ).resolve()
    if collector_file != expected_collector_file:
        raise ValueError("分红快照采集器路径不匹配")
    if (
        implementation_receipt != IMPLEMENTATION_RECEIPT.resolve()
        or not implementation_receipt.exists()
        or _sha256(implementation_receipt) != implementation.get("receipt_sha256")
        or not collector_file.exists()
        or _sha256(collector_file) != implementation.get("sha256")
    ):
        raise ValueError("分红快照采集器或实现收据哈希漂移")
    settlement_receipt = json.loads(
        implementation_receipt.read_text(encoding="utf-8")
    )
    collector_key = (
        "scripts\\collect_510300_etf_microstructure_dual_shadow_v1_dividend_snapshot.py"
    )
    if settlement_receipt["implementations"][collector_key]["sha256"] != implementation[
        "sha256"
    ]:
        raise ValueError("分红快照采集器哈希不在冻结实现收据中")
    frozen_audit = payload["frozen_history"]
    if (
        frozen_audit.get("file") != _relative(base_path)
        or frozen_audit.get("expected_sha256") != expected_base_hash
        or frozen_audit.get("actual_sha256") != actual_base_hash
        or frozen_audit.get("candidate_manifest_sha256")
        != _sha256(frozen.MANIFEST_FILE)
    ):
        raise ValueError("分红快照的冻结历史锚点不匹配")

    source = payload["source"]
    source_files: dict[str, dict[str, Any]] = {}
    for label, file_key, hash_key in [
        ("raw", "raw_file", "raw_sha256"),
        ("normalized", "normalized_file", "normalized_sha256"),
    ]:
        path = (PROJECT_ROOT / source[file_key]).resolve()
        if not path.is_relative_to(resolved.parent):
            raise ValueError(f"分红快照{label}文件越出快照目录")
        if not path.exists() or _sha256(path) != source[hash_key]:
            raise ValueError(f"分红快照{label}文件缺失或哈希漂移")
        source_files[label] = {
            "file": _relative(path),
            "sha256": _sha256(path),
        }
    normalized = pd.read_parquet(PROJECT_ROOT / source["normalized_file"])
    normalized["ex_date"] = pd.to_datetime(
        normalized["ex_date"], errors="raise"
    ).dt.normalize()
    detected_from_file = {
        pd.Timestamp(row.ex_date).date().isoformat(): float(row.cash_dividend_implied)
        for row in normalized.loc[normalized["is_new_event"].astype(bool)].itertuples(
            index=False
        )
    }
    detected_from_manifest = {
        str(event["ex_date"]): float(event["cash_dividend_per_share"])
        for event in payload.get("detected_new_events", [])
    }
    if set(detected_from_file) != set(detected_from_manifest):
        raise ValueError("分红快照清单与标准化文件的新事件日期集合不一致")
    for date, amount in detected_from_file.items():
        if abs(amount - detected_from_manifest[date]) > 1e-9:
            raise ValueError(f"分红快照新事件金额不一致：{date}")

    status = str(payload.get("status"))
    allowed = bool(payload.get("settlement_qualification_allowed"))
    accepted = {"PASS_NO_NEW_EVENT", "PASS_VERIFIED_NEW_EVENTS"}
    if allowed != (status in accepted):
        raise ValueError("分红快照状态与结算资格标记不一致")
    verified_events = payload.get("verified_new_events", [])
    if status == "PASS_NO_NEW_EVENT" and (
        detected_from_manifest or verified_events
    ):
        raise ValueError("无新事件状态却包含新分红事件")
    if status == "PASS_VERIFIED_NEW_EVENTS":
        official = payload.get("official_verification")
        if not isinstance(official, dict):
            raise ValueError("已核验新事件状态缺少官方核验清单")
        official_manifest = (PROJECT_ROOT / official["manifest"]).resolve()
        if not official_manifest.exists() or _sha256(official_manifest) != official[
            "manifest_sha256"
        ]:
            raise ValueError("官方核验清单缺失或哈希漂移")
        if set(detected_from_manifest) != {
            str(event["ex_date"]) for event in verified_events
        }:
            raise ValueError("发现的新事件与官方核验事件集合不一致")
        for event in verified_events:
            source_path = (PROJECT_ROOT / event["source_snapshot_file"]).resolve()
            if not source_path.is_relative_to(official_manifest.parent):
                raise ValueError("官方分红原文文件越出核验清单目录")
            if not source_path.exists() or _sha256(source_path) != event[
                "source_snapshot_sha256"
            ]:
                raise ValueError("官方分红原文文件缺失或哈希漂移")

    verified_frame = _verified_event_frame(verified_events if allowed else [])
    dividends = pd.concat([base, verified_frame], ignore_index=True, sort=False)
    dividends.sort_values("ex_date", inplace=True)
    dividends.reset_index(drop=True, inplace=True)
    if dividends["ex_date"].duplicated().any():
        raise ValueError("冻结分红与前瞻核验分红存在重复除息日")
    audit = {
        "status": status,
        "settlement_qualification_allowed": allowed,
        "manifest": _relative(resolved),
        "manifest_sha256": _sha256(resolved),
        "frozen_distribution_file": _relative(base_path),
        "frozen_distribution_sha256": actual_base_hash,
        "detected_new_event_count": int(len(detected_from_manifest)),
        "verified_new_event_count": int(len(verified_events)),
        "effective_distribution_event_count": int(len(dividends)),
        "source_files": source_files,
        "errors": payload.get("errors", []),
    }
    return dividends, audit, allowed


def _evaluate_starts(
    block: pd.DataFrame,
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    if block.empty:
        return pd.DataFrame()
    execution = block.merge(
        market,
        left_on="execution_date",
        right_on="date",
        how="inner",
        validate="one_to_one",
    )
    execution.sort_values("execution_date", inplace=True)
    execution.reset_index(drop=True, inplace=True)
    if execution.empty:
        return pd.DataFrame()
    market_positions = {date: index for index, date in enumerate(market["date"])}
    first_execution = pd.Timestamp(execution.iloc[0]["execution_date"])
    previous_position = market_positions[first_execution] - 1
    if previous_position < 0:
        raise ValueError("前瞻结算缺少首个执行日前一交易日行情")
    previous_date = pd.Timestamp(market.iloc[previous_position]["date"])
    states_by_date = {
        pd.Timestamp(row["execution_date"]): int(row["state_final"])
        for _, row in execution.iterrows()
    }
    available_execution_dates = list(execution["execution_date"])
    base_costs, stress_costs, objective = frozen._cost_models(config)
    initial_capital = float(config["account"]["initial_capital_cny"])
    perturbations = [
        int(value) for value in config["objective"]["start_perturbations_trading_days"]
    ]
    rows: list[dict[str, Any]] = []
    for perturbation in perturbations:
        selected_dates = available_execution_dates[perturbation:]
        if not selected_dates:
            continue
        first_date = pd.Timestamp(selected_dates[0])
        first_market_position = market_positions[first_date]
        prior_date = pd.Timestamp(market.iloc[first_market_position - 1]["date"])
        last_date = pd.Timestamp(selected_dates[-1])
        sample = market.loc[
            market["date"].between(prior_date, last_date),
            ["date", "etf_open", "etf_close", "benchmark_close"],
        ].copy()
        expected_dates = [prior_date, *[pd.Timestamp(value) for value in selected_dates]]
        if sample["date"].tolist() != expected_dates:
            raise ValueError("前瞻claim执行日期与市场交易日不连续")
        states = np.zeros(len(sample), dtype=np.int8)
        states[1:] = [states_by_date[pd.Timestamp(date)] for date in selected_dates]
        benchmark_ledger = build_benchmark_ledger(sample, initial_capital)
        base_ledger, base_trades = simulate_binary_path(
            sample,
            dividends,
            states,
            costs=base_costs,
            initial_capital=initial_capital,
            reinvest_paid_dividends=True,
        )
        stress_ledger, stress_trades = simulate_binary_path(
            sample,
            dividends,
            states,
            costs=stress_costs,
            initial_capital=initial_capital,
            reinvest_paid_dividends=True,
        )
        base_summary = summarize_path(
            base_ledger, base_trades, benchmark_ledger, objective=objective
        )
        stress_summary = summarize_path(
            stress_ledger, stress_trades, benchmark_ledger, objective=objective
        )
        row: dict[str, Any] = {
            "policy": POLICY,
            "start_perturbation": perturbation,
            "first_signal_date": pd.Timestamp(block.iloc[perturbation]["signal_date"]),
            "first_execution_date": first_date,
            "last_execution_date": last_date,
            "execution_day_count": int(len(selected_dates)),
            "eligible_242d": bool(len(selected_dates) >= REQUIRED_EXECUTION_DAYS_PER_START),
            "cash_day_count": int(sum(states[1:] == 0)),
            "cash_day_share": float(np.mean(states[1:] == 0)),
        }
        for prefix, summary in [("base", base_summary), ("stress", stress_summary)]:
            for key, value in summary.items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _daily_outcomes(
    block: pd.DataFrame,
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    if block.empty:
        return pd.DataFrame()
    market_positions = {date: index for index, date in enumerate(market["date"])}
    dividend_map = dividends.groupby("ex_date")["cash_dividend_per_share"].sum().to_dict()
    cash_daily = float(config["account"]["cash_annual_rate"]) / int(
        config["account"]["trading_days_per_year"]
    )
    rows: list[dict[str, Any]] = []
    for _, claim in block.iterrows():
        execution_date = pd.Timestamp(claim["execution_date"])
        position = market_positions.get(execution_date)
        if position is None:
            maturity = "PENDING_T_PLUS_1_MARKET"
            following_date = None
            etf_open_return = None
        elif position + 1 >= len(market):
            maturity = "PENDING_T_PLUS_2_OPEN"
            following_date = None
            etf_open_return = None
        else:
            following = market.iloc[position + 1]
            following_date = pd.Timestamp(following["date"])
            distribution = float(dividend_map.get(following_date, 0.0))
            etf_open_return = (
                float(following["etf_open"]) + distribution
            ) / float(market.iloc[position]["etf_open"]) - 1.0
            maturity = "MATURED"
        state = int(claim["state_final"])
        gross_state_return = (
            etf_open_return if state == 1 and etf_open_return is not None else cash_daily
            if state == 0 and etf_open_return is not None
            else None
        )
        rows.append(
            {
                **claim.to_dict(),
                "following_open_date": following_date,
                "maturity_status": maturity,
                "etf_open_to_open_total_return": etf_open_return,
                "cash_daily_return": cash_daily,
                "gross_state_open_to_open_return": gross_state_return,
            }
        )
    return pd.DataFrame(rows)


def _aggregate_metrics(metrics: pd.DataFrame) -> dict[str, Any]:
    if metrics.empty:
        return {
            "start_count": 0,
            "all_five_starts_present": False,
            "all_five_starts_eligible_242d": False,
            "stress_annualized_excess_minimum": None,
            "stress_rolling_242d_excess_minimum": None,
            "every_start_both_20pct_gates": False,
        }
    eligible = metrics["eligible_242d"].astype(bool)
    annual = pd.to_numeric(metrics["stress_annualized_excess"], errors="coerce")
    rolling = pd.to_numeric(
        metrics["stress_rolling_242d_excess_median"], errors="coerce"
    )
    both = metrics["stress_both_20pct_gates"].fillna(False).astype(bool)
    return {
        "start_count": int(len(metrics)),
        "all_five_starts_present": bool(len(metrics) == 5),
        "all_five_starts_eligible_242d": bool(len(metrics) == 5 and eligible.all()),
        "stress_annualized_excess_minimum": _safe_float(annual.min()),
        "stress_annualized_excess_median": _safe_float(annual.median()),
        "stress_rolling_242d_excess_minimum": _safe_float(rolling.dropna().min()),
        "every_start_both_20pct_gates": bool(
            len(metrics) == 5 and eligible.all() and both.all()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="结算双影子V1同日前瞻Shadow绩效")
    parser.add_argument("--snapshot-manifest", default=None)
    parser.add_argument("--dividend-snapshot-manifest", default=None)
    parser.add_argument("--run-id", default=None)
    arguments = parser.parse_args()
    now = datetime.now(TIMEZONE)
    run_id = arguments.run_id or now.strftime("%Y%m%dT%H%M%S%z")
    output_dir = RUN_ROOT / run_id
    if output_dir.exists():
        raise FileExistsError(f"前瞻结算目录已存在，拒绝覆盖：{output_dir}")

    implementation_audit = _verify_implementation()
    config, candidate_manifest, freeze_hash_mismatches = frozen._load_freeze()
    calendar, calendar_audit = _calendar()
    dividend_manifest_path = _resolve_latest_dividend_manifest(
        arguments.dividend_snapshot_manifest
    )
    dividends, dividend_audit, dividend_qualification_allowed = _load_dividends(
        config, candidate_manifest, dividend_manifest_path
    )
    frozen_at = pd.Timestamp(candidate_manifest["frozen_at_asia_shanghai"])
    claims, claim_audit = _load_claims(calendar, frozen_at)
    segmented, segment_summary = _segments(claims, calendar)
    snapshot_manifest_path = _resolve_latest_snapshot_manifest(
        segmented, arguments.snapshot_manifest
    )

    if snapshot_manifest_path is None:
        status = (
            "COLLECTING_NO_PROSPECTIVE_CLAIMS"
            if dividend_qualification_allowed
            else "DIVIDEND_COVERAGE_NOT_READY_NO_PROSPECTIVE_CLAIMS"
        )
        payload = {
            "project_id": PROJECT_ID,
            "status": status,
            "evaluated_at_asia_shanghai": now.isoformat(),
            "implementation_audit": implementation_audit,
            "candidate_manifest_sha256": _sha256(frozen.MANIFEST_FILE),
            "freeze_hash_mismatches": freeze_hash_mismatches,
            "calendar": calendar_audit,
            "dividend_coverage": dividend_audit,
            "prospective_claim_count": 0,
            "latest_contiguous_claim_count": 0,
            "latest_contiguous_executed_count": 0,
            "required_latest_contiguous_execution_days": REQUIRED_LATEST_CONTIGUOUS_EXECUTION_DAYS,
            "position_mapping": "DISABLED",
            "order_generation": "DISABLED",
            "live_trading_authorized": False,
        }
        output_dir.mkdir(parents=True, exist_ok=False)
        _atomic_json(payload, output_dir / "validation_report.json")
        _atomic_json(payload, STATUS_FILE)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    snapshot_etf, snapshot_benchmark, snapshot_audit = _load_snapshot_market(
        snapshot_manifest_path.resolve()
    )
    market = _combined_market(config, snapshot_etf, snapshot_benchmark)
    latest_segment_id = int(segmented["segment_id"].max())
    latest_block = segmented.loc[segmented["segment_id"].eq(latest_segment_id)].copy()
    executed_block = latest_block.loc[
        latest_block["execution_date"].isin(set(market["date"]))
    ].copy()
    executed_block.sort_values("signal_date", inplace=True)
    executed_block.reset_index(drop=True, inplace=True)
    executed_count = int(len(executed_block))
    if not dividend_qualification_allowed:
        output_dir.mkdir(parents=True, exist_ok=False)
        claims_path = output_dir / "verified_claims.parquet"
        _atomic_parquet(segmented, claims_path)
        payload = {
            "project_id": PROJECT_ID,
            "candidate_policy": POLICY,
            "status": "DIVIDEND_COVERAGE_BLOCKED_QUALIFICATION_PAUSED",
            "evaluated_at_asia_shanghai": now.isoformat(),
            "run_id": run_id,
            "implementation_audit": implementation_audit,
            "freeze": {
                "candidate_manifest_sha256": _sha256(frozen.MANIFEST_FILE),
                "candidate_config_sha256": _sha256(frozen.CONFIG_FILE),
                "freeze_hash_mismatches": freeze_hash_mismatches,
            },
            "claim_integrity": {
                "prospective_claim_count": int(len(segmented)),
                "claims_verified": int(len(claim_audit)),
                "segments": segment_summary,
                "latest_segment_id": latest_segment_id,
                "latest_contiguous_claim_count": int(len(latest_block)),
                "latest_contiguous_executed_count": executed_count,
            },
            "qualification": {
                "eligible": False,
                "paused_only_for_dividend_coverage": True,
                "performance_metrics_computed": False,
                "remaining_execution_days_before_other_gates": int(
                    max(REQUIRED_LATEST_CONTIGUOUS_EXECUTION_DAYS - executed_count, 0)
                ),
            },
            "calendar": calendar_audit,
            "dividend_coverage": dividend_audit,
            "snapshot": snapshot_audit,
            "evidence": {
                "verified_claims": _relative(claims_path),
            },
            "boundaries": {
                "unverified_dividend_never_assumed_zero": True,
                "research_validation_ledger_only": True,
                "position_mapping": "DISABLED",
                "order_generation": "DISABLED",
                "broker_connection": "DISABLED",
                "live_trading_authorized": False,
            },
        }
        _atomic_json(payload, output_dir / "validation_report.json")
        _atomic_json(payload, STATUS_FILE)
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 2
    outcomes = _daily_outcomes(latest_block, market, dividends, config)
    metrics = _evaluate_starts(executed_block, market, dividends, config)
    aggregate = _aggregate_metrics(metrics)
    eligible = bool(
        executed_count >= REQUIRED_LATEST_CONTIGUOUS_EXECUTION_DAYS
        and aggregate["all_five_starts_eligible_242d"]
    )
    if not eligible:
        status = "COLLECTING_NOT_ELIGIBLE_FORWARD_242D"
    elif aggregate["every_start_both_20pct_gates"]:
        status = "PROSPECTIVE_FORWARD_OBJECTIVE_PASS_ALL_FIVE_STARTS"
    else:
        status = "PROSPECTIVE_FORWARD_OBJECTIVE_FAILED_NO_PARAMETER_RESCUE"

    output_dir.mkdir(parents=True, exist_ok=False)
    claims_path = output_dir / "verified_claims.parquet"
    outcomes_path = output_dir / "daily_outcomes.parquet"
    metrics_path = output_dir / "start_metrics.parquet"
    _atomic_parquet(segmented, claims_path)
    _atomic_parquet(outcomes, outcomes_path)
    _atomic_parquet(metrics, metrics_path)
    payload = {
        "project_id": PROJECT_ID,
        "candidate_policy": POLICY,
        "status": status,
        "evaluated_at_asia_shanghai": now.isoformat(),
        "run_id": run_id,
        "implementation_audit": implementation_audit,
        "freeze": {
            "candidate_manifest_sha256": _sha256(frozen.MANIFEST_FILE),
            "candidate_config_sha256": _sha256(frozen.CONFIG_FILE),
            "freeze_hash_mismatches": freeze_hash_mismatches,
        },
        "claim_integrity": {
            "prospective_claim_count": int(len(segmented)),
            "claims_verified": int(len(claim_audit)),
            "all_claim_evidence_hashes_pass": True,
            "segments": segment_summary,
            "latest_segment_id": latest_segment_id,
            "latest_contiguous_claim_count": int(len(latest_block)),
            "latest_contiguous_executed_count": executed_count,
            "latest_signal_date": latest_block["signal_date"].max().date().isoformat(),
            "latest_executed_signal_date": (
                executed_block["signal_date"].max().date().isoformat()
                if not executed_block.empty
                else None
            ),
        },
        "maturity": {
            "matured_open_to_open_outcomes": int(
                outcomes["maturity_status"].eq("MATURED").sum()
            ),
            "pending_t_plus_1_market": int(
                outcomes["maturity_status"].eq("PENDING_T_PLUS_1_MARKET").sum()
            ),
            "pending_t_plus_2_open": int(
                outcomes["maturity_status"].eq("PENDING_T_PLUS_2_OPEN").sum()
            ),
        },
        "qualification": {
            "required_execution_days_per_start": REQUIRED_EXECUTION_DAYS_PER_START,
            "required_latest_contiguous_execution_days": REQUIRED_LATEST_CONTIGUOUS_EXECUTION_DAYS,
            "remaining_execution_days": int(
                max(REQUIRED_LATEST_CONTIGUOUS_EXECUTION_DAYS - executed_count, 0)
            ),
            "eligible": eligible,
            "aggregate": aggregate,
        },
        "calendar": calendar_audit,
        "dividend_coverage": dividend_audit,
        "snapshot": snapshot_audit,
        "evidence": {
            "verified_claims": str(claims_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "daily_outcomes": str(outcomes_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "start_metrics": str(metrics_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "boundaries": {
            "strict_same_day_claims_only": True,
            "missed_day_starts_new_contiguous_segment": True,
            "research_validation_ledger_only": True,
            "position_mapping": "DISABLED",
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "live_trading_authorized": False,
            "failed_forward_parameter_rescue": "FORBIDDEN",
        },
    }
    report_path = output_dir / "validation_report.json"
    _atomic_json(payload, report_path)
    _atomic_json(payload, STATUS_FILE)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 2 if status == "PROSPECTIVE_FORWARD_OBJECTIVE_FAILED_NO_PARAMETER_RESCUE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
