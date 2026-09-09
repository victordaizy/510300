"""采集并审计 V1.4 入选证券的历史分红、送股和转增事件；不读取组合收益。"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import akshare as ak
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/a_share_hs_concentrated_low_risk_trend_v1_5_corporate_action_audit.yaml"
TIMEZONE = ZoneInfo("Asia/Shanghai")

RAW_COLUMNS = [
    "implementation_date",
    "dividend_type",
    "bonus_ratio_per_10_shares",
    "transfer_ratio_per_10_shares",
    "cash_dividend_per_10_shares",
    "record_date",
    "ex_right_date",
    "ex_dividend_date",
    "shares_arrival_date",
    "implementation_note",
    "report_period",
]
DATE_COLUMNS = [
    "implementation_date",
    "record_date",
    "ex_right_date",
    "ex_dividend_date",
    "shares_arrival_date",
]
NUMERIC_COLUMNS = [
    "bonus_ratio_per_10_shares",
    "transfer_ratio_per_10_shares",
    "cash_dividend_per_10_shares",
]


def project_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    path.relative_to(ROOT.resolve())
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _normalize_provider_frame(frame: pd.DataFrame, ts_code: str, retrieved_at: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        output = pd.DataFrame(columns=RAW_COLUMNS)
    else:
        if frame.shape[1] < len(RAW_COLUMNS):
            raise ValueError(f"{ts_code} 公司行动字段数不足：{frame.shape[1]}")
        output = frame.iloc[:, : len(RAW_COLUMNS)].copy()
        output.columns = RAW_COLUMNS
    for column in DATE_COLUMNS:
        output[column] = pd.to_datetime(output[column], errors="coerce").dt.normalize()
    for column in NUMERIC_COLUMNS:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    for column in ("dividend_type", "implementation_note", "report_period"):
        output[column] = output[column].astype("string")
    output.insert(0, "ts_code", ts_code)
    output.insert(1, "source_row_index", np.arange(len(output), dtype=np.int64))
    output["source_provider"] = "CNINFO_HISTORY_DIVIDEND"
    output["source_url"] = f"https://webapi.cninfo.com.cn/#/company?companyid={ts_code}"
    output["retrieved_at"] = retrieved_at
    return output


def _fetch_one(ts_code: str, retrieved_at: str, attempts: int = 3) -> tuple[pd.DataFrame, dict[str, Any]]:
    code = ts_code.split(".", 1)[0]
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            provider_frame = ak.stock_dividend_cninfo(symbol=code)
            normalized = _normalize_provider_frame(provider_frame, ts_code, retrieved_at)
            return normalized, {
                "ts_code": ts_code,
                "status": "SUCCESS",
                "attempts": attempt,
                "rows": int(len(normalized)),
                "source_provider": "CNINFO_HISTORY_DIVIDEND",
                "source_url": f"https://webapi.cninfo.com.cn/#/company?companyid={ts_code}",
            }
        except Exception as error:  # 网络服务偶发失败时保留失败原因，最终由门禁决定状态。
            last_error = f"{type(error).__name__}: {error}"
            if attempt < attempts:
                time.sleep(2.0 * attempt)
    return pd.DataFrame(columns=["ts_code", *RAW_COLUMNS, "source_provider", "source_url", "retrieved_at"]), {
        "ts_code": ts_code,
        "status": "FAILED",
        "attempts": attempts,
        "rows": 0,
        "error": last_error,
        "source_provider": "CNINFO_HISTORY_DIVIDEND",
        "source_url": f"https://webapi.cninfo.com.cn/#/company?companyid={ts_code}",
    }


def _normalize_events(
    raw: pd.DataFrame,
    coverage_start: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if raw.empty:
        columns = [
            "ts_code", "implementation_date", "dividend_type", "cash_dividend_per_share",
            "bonus_ratio_per_share", "transfer_ratio_per_share", "record_date", "ex_right_date",
            "ex_dividend_date", "shares_arrival_date", "effective_date", "implementation_note",
            "report_period", "source_provider", "source_url", "retrieved_at", "source_row_index",
        ]
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)
    events = raw.copy()
    for column in DATE_COLUMNS:
        events[column] = pd.to_datetime(events[column], errors="coerce").dt.normalize()
    events["cash_dividend_per_share"] = events["cash_dividend_per_10_shares"].fillna(0.0) / 10.0
    events["bonus_ratio_per_share"] = events["bonus_ratio_per_10_shares"].fillna(0.0) / 10.0
    events["transfer_ratio_per_share"] = events["transfer_ratio_per_10_shares"].fillna(0.0) / 10.0
    events["action_total"] = (
        events["cash_dividend_per_share"]
        + events["bonus_ratio_per_share"]
        + events["transfer_ratio_per_share"]
    )
    events["effective_date"] = events[["ex_right_date", "ex_dividend_date"]].min(axis=1)
    action_rows = events.loc[events["action_total"].gt(0)].copy()
    normalized_columns = [
        "ts_code", "implementation_date", "dividend_type", "cash_dividend_per_share",
        "bonus_ratio_per_share", "transfer_ratio_per_share", "record_date", "ex_right_date",
        "ex_dividend_date", "shares_arrival_date", "effective_date", "implementation_note",
        "report_period", "source_provider", "source_url", "retrieved_at", "source_row_index",
    ]
    action_rows = action_rows[normalized_columns].sort_values(
        ["ts_code", "effective_date", "source_row_index"], kind="mergesort"
    ).reset_index(drop=True)
    evaluation_window = action_rows.loc[
        action_rows["effective_date"].between(coverage_start, cutoff, inclusive="both")
    ].copy()
    return evaluation_window, action_rows


def build_audit() -> dict[str, Any]:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    selected_path = project_path(contract["sources"]["selected_symbols"])
    receipt_path = project_path(contract["sources"]["signal_freeze_receipt"])
    selected = pd.read_parquet(selected_path, columns=["ts_code", "signal_date"])
    selected["ts_code"] = selected["ts_code"].astype(str)
    selected_codes = sorted(selected["ts_code"].unique())
    signal_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if signal_receipt.get("portfolio_return_values_read") is not False:
        raise RuntimeError("上游信号回执已读取组合收益，禁止进入本审计")
    raw_columns = ["ts_code", "source_row_index", *RAW_COLUMNS, "source_provider", "source_url", "retrieved_at"]
    existing_raw_path = project_path(contract["outputs"]["raw_events"])
    existing_receipt_path = project_path(contract["outputs"]["receipt"])
    reused_snapshot = False
    retrieved_at = datetime.now(TIMEZONE).isoformat()
    fetch_receipts: list[dict[str, Any]] = []
    raw = pd.DataFrame(columns=raw_columns)
    if existing_raw_path.is_file() and existing_receipt_path.is_file():
        existing_receipt = json.loads(existing_receipt_path.read_text(encoding="utf-8"))
        recorded_hash = existing_receipt.get("artifacts", {}).get("raw_events", {}).get("sha256")
        recorded_fetch = existing_receipt.get("fetch", [])
        fetched_codes = {str(item.get("ts_code")) for item in recorded_fetch if item.get("status") == "SUCCESS"}
        if (
            recorded_hash == sha256_file(existing_raw_path)
            and fetched_codes == set(selected_codes)
            and all(item.get("status") == "SUCCESS" for item in recorded_fetch)
        ):
            raw = pd.read_parquet(existing_raw_path).reindex(columns=raw_columns)
            fetch_receipts = recorded_fetch
            retrieved_at = str(existing_receipt.get("audited_at", retrieved_at))
            reused_snapshot = True
            print(json.dumps({"状态": "复用完整公司行动原始快照", "证券": len(selected_codes), "行数": len(raw)}, ensure_ascii=False), flush=True)
    if not reused_snapshot:
        raw_frames: list[pd.DataFrame] = []
        prior_raw = pd.DataFrame(columns=raw_columns)
        prior_receipts: list[dict[str, Any]] = []
        prior_source = contract["sources"].get("prior_raw_snapshot")
        prior_receipt_source = contract["sources"].get("prior_fetch_receipt")
        if prior_source and prior_receipt_source:
            prior_raw_path = project_path(prior_source)
            prior_receipt_path = project_path(prior_receipt_source)
            if prior_raw_path.is_file() and prior_receipt_path.is_file():
                prior_receipt = json.loads(prior_receipt_path.read_text(encoding="utf-8"))
                prior_hash = prior_receipt.get("artifacts", {}).get("raw_events", {}).get("sha256")
                prior_fetch = prior_receipt.get("fetch", [])
                if prior_hash == sha256_file(prior_raw_path) and all(item.get("status") == "SUCCESS" for item in prior_fetch):
                    prior_raw = pd.read_parquet(prior_raw_path).reindex(columns=raw_columns)
                    prior_raw = prior_raw.loc[prior_raw["ts_code"].isin(selected_codes)].copy()
                    prior_receipts = [item for item in prior_fetch if item.get("ts_code") in set(selected_codes)]
        raw_frames.append(prior_raw)
        reused_prior_codes = {str(item.get("ts_code")) for item in prior_receipts}
        missing_codes = [code for code in selected_codes if code not in reused_prior_codes]
        for position, ts_code in enumerate(missing_codes, start=1):
            frame, item = _fetch_one(ts_code, retrieved_at)
            if not frame.empty:
                raw_frames.append(frame)
            fetch_receipts.append(item)
            print(json.dumps({"进度": f"{position}/{len(missing_codes)}", "证券": ts_code, "状态": item["status"], "行数": item["rows"]}, ensure_ascii=False), flush=True)
        fetch_receipts = prior_receipts + fetch_receipts
        if prior_receipts:
            print(json.dumps({"状态": "复用上游公司行动记录", "证券": len(reused_prior_codes), "新增采集": len(missing_codes)}, ensure_ascii=False), flush=True)
        raw = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame(columns=raw_columns)
        raw = raw.reindex(columns=raw_columns)
    cutoff = pd.Timestamp(contract["market_data_end"]).normalize()
    coverage_start = pd.to_datetime(selected["signal_date"], errors="raise").min().normalize()
    events, all_action_events = _normalize_events(raw, coverage_start, cutoff)
    unresolved_lookback_start = coverage_start - pd.Timedelta(
        days=int(contract["definition"]["unresolved_action_lookback_days"])
    )
    report_announced_past_unresolved = int(
        (
            all_action_events["implementation_date"].between(
                unresolved_lookback_start, cutoff, inclusive="both"
            ).fillna(False)
            & all_action_events["effective_date"].isna()
        ).sum()
    )
    pre_window_unresolved = int(
        (
            all_action_events["implementation_date"].lt(unresolved_lookback_start).fillna(False)
            & all_action_events["effective_date"].isna()
        ).sum()
    )
    failed = [item for item in fetch_receipts if item["status"] != "SUCCESS"]
    duplicate_exact = int(events.duplicated(
        ["ts_code", "effective_date", "cash_dividend_per_share", "bonus_ratio_per_share", "transfer_ratio_per_share"],
        keep=False,
    ).sum())
    numeric_valid = bool(
        events[["cash_dividend_per_share", "bonus_ratio_per_share", "transfer_ratio_per_share"]]
        .apply(lambda column: np.isfinite(column.to_numpy(dtype=float)).all() and (column >= 0).all())
        .all()
    ) if not events.empty else True
    date_valid = bool(
        events["record_date"].isna().sum() == 0
        or (events["record_date"].isna() | events["effective_date"].isna() | (events["record_date"] <= events["effective_date"])).all()
    ) if not events.empty else True
    blocking_reasons: list[str] = []
    if failed:
        blocking_reasons.append(f"CNINFO 公司行动接口采集失败 {len(failed)} 只证券")
    if report_announced_past_unresolved:
        blocking_reasons.append(f"截至市场数据终点仍有 {report_announced_past_unresolved} 条已公告但无生效日期的公司行动")
    if duplicate_exact:
        blocking_reasons.append(f"公司行动事件存在 {duplicate_exact} 条重复键记录")
    if not numeric_valid:
        blocking_reasons.append("公司行动金额或比例存在非法数值")
    if not date_valid:
        blocking_reasons.append("公司行动登记日晚于生效日")
    status = "READY_FOR_RETURN_READ" if not blocking_reasons else "BLOCKED_BEFORE_RETURN_READ"
    return {
        "contract": contract,
        "selected_codes": selected_codes,
        "retrieved_at": retrieved_at,
        "raw": raw,
        "events": events,
        "all_action_events": all_action_events,
        "fetch_receipts": fetch_receipts,
        "cutoff": cutoff,
        "coverage_start": coverage_start,
        "unresolved_lookback_start": unresolved_lookback_start,
        "status": status,
        "blocking_reasons": blocking_reasons,
        "report_announced_past_unresolved": report_announced_past_unresolved,
        "pre_window_unresolved": pre_window_unresolved,
        "duplicate_exact": duplicate_exact,
        "numeric_valid": numeric_valid,
        "date_valid": date_valid,
        "signal_receipt": signal_receipt,
        "reused_snapshot": reused_snapshot,
    }


def main() -> int:
    result = build_audit()
    contract = result["contract"]
    raw_path = project_path(contract["outputs"]["raw_events"])
    events_path = project_path(contract["outputs"]["normalized_events"])
    receipt_path = project_path(contract["outputs"]["receipt"])
    report_path = project_path(contract["outputs"]["report_json"])
    report_md_path = project_path(contract["outputs"]["report_markdown"])
    _atomic_parquet(result["raw"], raw_path)
    _atomic_parquet(result["events"], events_path)
    receipt = {
        "contract_id": contract["contract_id"],
        "parent_signal_contract_id": contract["parent_signal_contract_id"],
        "audited_at": result["retrieved_at"],
        "status": "CORPORATE_ACTIONS_AUDITED_NO_RETURN_READ" if result["status"] == "READY_FOR_RETURN_READ" else result["status"],
        "view_status": "NO_VIEW",
        "portfolio_return_values_read": False,
        "positions_generated": False,
        "orders_generated": False,
        "selected_symbol_count": len(result["selected_codes"]),
        "raw_row_count": int(len(result["raw"])),
        "evaluation_window_action_event_count": int(len(result["events"])),
        "action_event_count_all_dates": int(len(result["all_action_events"])),
        "coverage_start": result["coverage_start"].date().isoformat(),
        "market_data_end": result["cutoff"].date().isoformat(),
        "reused_complete_raw_snapshot": result["reused_snapshot"],
        "fetch": result["fetch_receipts"],
        "artifacts": {
            "raw_events": {"path": raw_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(raw_path)},
            "normalized_events": {"path": events_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(events_path)},
        },
    }
    _atomic_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", receipt_path)
    report = {
        "schema_version": "A_SHARE_HS_CORPORATE_ACTION_AUDIT_V1",
        "contract_id": contract["contract_id"],
        "parent_signal_contract_id": contract["parent_signal_contract_id"],
        "audited_at": result["retrieved_at"],
        "status": result["status"],
        "view_status": "NO_VIEW",
        "portfolio_return_values_read": False,
        "selected_symbol_count": len(result["selected_codes"]),
        "raw_row_count": int(len(result["raw"])),
        "evaluation_window_action_event_count": int(len(result["events"])),
        "action_event_count_all_dates": int(len(result["all_action_events"])),
        "symbols_with_actions_in_evaluation_window": int(result["events"]["ts_code"].nunique()) if not result["events"].empty else 0,
        "symbols_without_actions_in_evaluation_window": sorted(set(result["selected_codes"]) - set(result["events"]["ts_code"].unique())),
        "coverage_start": result["coverage_start"].date().isoformat(),
        "market_data_end": result["cutoff"].date().isoformat(),
        "unresolved_lookback_start": result["unresolved_lookback_start"].date().isoformat(),
        "failed_fetch_count": sum(item["status"] != "SUCCESS" for item in result["fetch_receipts"]),
        "unresolved_past_action_count": result["report_announced_past_unresolved"],
        "pre_window_unresolved_nonblocking_count": result["pre_window_unresolved"],
        "reused_complete_raw_snapshot": result["reused_snapshot"],
        "duplicate_exact_count": result["duplicate_exact"],
        "numeric_valid": result["numeric_valid"],
        "date_valid": result["date_valid"],
        "blocking_reasons": result["blocking_reasons"],
        "return_files_opened": [],
        "signals_generated": False,
        "positions_generated": False,
        "orders_generated": False,
        "artifacts": receipt["artifacts"],
        "conclusion": "公司行动覆盖审计完成；只有状态为 READY_FOR_RETURN_READ 时才允许进入后续组合收益读取。",
    }
    _atomic_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", report_path)
    markdown = (
        "# 沪深 A 股 V1.5 公司行动覆盖审计\n\n"
        f"- 状态：`{report['status']}`\n"
        f"- 入选证券：{report['selected_symbol_count']} 只\n"
        f"- 评估窗口有效公司行动：{report['evaluation_window_action_event_count']} 条\n"
        f"- 覆盖起点：{report['coverage_start']}\n"
        f"- 市场数据终点：{report['market_data_end']}\n"
        "- 组合收益读取：否\n\n"
        "金额与比例按 CNINFO 每 10 股口径换算为每股；生效日取除权日和除息日中最早的非空日期。\n"
    )
    if result["blocking_reasons"]:
        markdown += "\n## 阻断原因\n\n" + "\n".join(f"- {reason}" for reason in result["blocking_reasons"]) + "\n"
    else:
        markdown += "\n公司行动覆盖满足后续收益读取前置门禁；仍未读取组合收益。\n"
    _atomic_text(markdown, report_md_path)
    print(json.dumps({"状态": result["status"], "入选证券": len(result["selected_codes"]), "评估窗口公司行动": len(result["events"]), "组合收益读取": False}, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "READY_FOR_RETURN_READ" else 2


if __name__ == "__main__":
    raise SystemExit(main())
