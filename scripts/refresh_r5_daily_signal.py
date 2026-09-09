"""按R5冻结参数刷新当日基础纸面信号，不改写R5历史报告。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.defensive_valuation_timing_engine import (
    build_model_positions,
    build_timing_features,
    schedule_asymmetric_execution,
)
from backtest.valuation_fvg_engine import build_valuation_signals
from research.v3_forward_validation import ETF_FILE, INDEX_FILE, VALUATION_FILE
from research.valuation_v2_expected_return import TOTAL_RETURN_FILE
from scripts.download_000300_valuation import fetch_valuation
from scripts.refresh_v3_forward_inputs import _atomic_parquet, refresh_market_series


TIMEZONE = ZoneInfo("Asia/Shanghai")
CONFIG_FILE = ROOT / "config" / "round5_defensive_valuation_timing.yaml"
FRESHNESS_CONTRACT_FILE = ROOT / "config" / "data_freshness_contract.yaml"
OUTPUT_FILE = ROOT / "paper" / "round5_latest_signal.json"
STATUS_FILE = ROOT / "paper" / "round5_daily_refresh_status.json"


class Round5InputDateMismatchError(ValueError):
    """Round5必需输入不满足联合as-of日期合约。"""

    status = "FAILED_INPUT_DATE_MISMATCH"

    def __init__(self, reasons: list[str], input_asof_dates: dict[str, str | None]):
        self.reasons = reasons
        self.input_asof_dates = input_asof_dates
        super().__init__("；".join(reasons))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """原子写JSON，避免消费者读取半写入文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def load_round5_freshness_contract(
    path: Path = FRESHNESS_CONTRACT_FILE,
) -> dict[str, Any]:
    """读取Round5联合新鲜度合约。"""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        return payload["contracts"]["round5_daily_signal"]
    except (KeyError, TypeError) as exc:
        raise ValueError("缺少contracts.round5_daily_signal新鲜度合约") from exc


def validate_round5_input_asof_alignment(
    *,
    signal_date: pd.Timestamp,
    frames: dict[str, pd.DataFrame],
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """联合校验全部必需输入；任一错期即失败，不允许消费旧信号。"""

    target = pd.Timestamp(signal_date).normalize()
    rules = contract or load_round5_freshness_contract()
    required = rules["required_inputs"]
    reasons: list[str] = []
    latest_dates: dict[str, str | None] = {}
    normalized_latest: dict[str, pd.Timestamp | None] = {}

    for name, input_rule in required.items():
        frame = frames.get(name)
        date_column = str(input_rule["date_column"])
        if frame is None or frame.empty or date_column not in frame.columns:
            latest_dates[name] = None
            normalized_latest[name] = None
            reasons.append(f"{name}输入为空或缺少{date_column}字段")
            continue
        dates = pd.to_datetime(frame[date_column], errors="coerce").dropna()
        if dates.empty:
            latest_dates[name] = None
            normalized_latest[name] = None
            reasons.append(f"{name}没有有效日期")
            continue
        latest = pd.Timestamp(dates.max()).normalize()
        normalized_latest[name] = latest
        latest_dates[name] = str(latest.date())
        alignment = str(input_rule["alignment"])
        maximum_age = int(input_rule["maximum_calendar_age_days"])
        age_days = int((target - latest).days)
        if latest > target:
            reasons.append(f"{name}最新日期{latest.date()}晚于目标日{target.date()}")
        elif alignment == "EXACT_TARGET_DATE" and latest != target:
            reasons.append(f"{name}最新日期{latest.date()}未到目标日{target.date()}")
        elif age_days > maximum_age:
            reasons.append(
                f"{name}最新日期{latest.date()}距目标日{age_days}天，超过允许的{maximum_age}天"
            )

    same_date_group = list(rules.get("core_market_same_date_group", []))
    core_dates = {
        normalized_latest.get(name)
        for name in same_date_group
        if normalized_latest.get(name) is not None
    }
    if len(core_dates) > 1:
        detail = {name: latest_dates.get(name) for name in same_date_group}
        reasons.append(f"核心行情as_of_date不一致：{detail}")

    if reasons:
        raise Round5InputDateMismatchError(reasons, latest_dates)
    return {
        "status": "PASS",
        "target_date": str(target.date()),
        "input_asof_dates": latest_dates,
        "contract_file": FRESHNESS_CONTRACT_FILE.relative_to(ROOT).as_posix(),
        "contract_sha256": _sha256(FRESHNESS_CONTRACT_FILE),
    }


def _write_non_consumable_signal(
    *,
    target_date: pd.Timestamp,
    failure_status: str,
    error: str,
    input_asof_dates: dict[str, str | None] | None,
    output_file: Path | None = None,
) -> dict[str, Any]:
    """用失败墓碑替换消费者路径，防止旧信号被误认作今日信号。"""

    destination = output_file or OUTPUT_FILE
    prior_reference: dict[str, Any] | None = None
    if destination.exists():
        prior_hash = _sha256(destination)
        try:
            prior = json.loads(destination.read_text(encoding="utf-8"))
            prior_reference = {
                "signal_date": prior.get("signal_date"),
                "status": prior.get("status"),
                "sha256": prior_hash,
            }
        except (json.JSONDecodeError, OSError):
            prior_reference = {"signal_date": None, "status": None, "sha256": prior_hash}
    payload: dict[str, Any] = {
        "schema_version": "ROUND5_PAPER_SIGNAL_TOMBSTONE_V1",
        "status": failure_status,
        "target_date": str(pd.Timestamp(target_date).date()),
        "consumable": False,
        "signal": None,
        "automatic_ordering_authorized": False,
        "input_asof_dates": input_asof_dates or {},
        "error": error,
        "prior_signal_reference": prior_reference,
    }
    _atomic_json(destination, payload)
    return payload


def _continuous_valuation_position(valuation: pd.DataFrame) -> pd.DataFrame:
    """复现R5市值带中会相消的连续估值仓位，避免依赖滞后的成分市值面板。"""

    result = valuation.copy()
    width = (result["fair_high_12m"] - result["fair_low_12m"]).replace(0.0, np.nan)
    result["raw_continuous_position"] = (
        (result["fair_high_12m"] - result["index_close"]) / width
    ).clip(0.0, 1.0)
    return result


def build_latest_signal(
    *,
    signal_date: pd.Timestamp,
    config: dict[str, Any],
    etf: pd.DataFrame,
    index: pd.DataFrame,
    total_return: pd.DataFrame,
    valuation_raw: pd.DataFrame,
) -> dict[str, Any]:
    """用截至目标日的点时输入生成R5基础信号和完整执行许可。"""

    target = pd.Timestamp(signal_date).normalize()
    alignment = validate_round5_input_asof_alignment(
        signal_date=target,
        frames={
            "510300": etf,
            "000300": index,
            "H00300": total_return,
            "valuation": valuation_raw,
        },
    )
    latest_dates = alignment["input_asof_dates"]

    timing = build_timing_features(index.loc[pd.to_datetime(index["date"]).le(target)], config)
    valuation = build_valuation_signals(
        valuation_raw.loc[pd.to_datetime(valuation_raw["date"]).le(target)],
        config["valuation"],
    )
    valuation = _continuous_valuation_position(valuation)
    model = build_model_positions(timing, config, valuation)
    scheduled = schedule_asymmetric_execution(model, config)
    available_dates = pd.DataFrame(
        {
            "date": pd.to_datetime(etf.loc[pd.to_datetime(etf["date"]).le(target), "date"])
        }
    )
    aligned = available_dates.merge(scheduled, on="date", how="left", validate="one_to_one")
    required = [
        "target_position",
        "continuous_model_position",
        "trade_allowed",
        "is_review_day",
        "risk_off_override",
        "allow_position_increase",
        "allow_position_decrease",
    ]
    latest = aligned.iloc[-1]
    if latest[required].isna().any():
        missing = latest[required].index[latest[required].isna()].tolist()
        raise ValueError(f"目标日R5信号字段不完整：{missing}")
    return {
        "signal_date": str(pd.Timestamp(latest["date"]).date()),
        "strategic_position": float(latest["strategic_position"]),
        "trend_score": float(latest["trend_score"]),
        "risk_penalty": float(latest["risk_penalty"]),
        "continuous_model_position": float(latest["continuous_model_position"]),
        "target_position": float(latest["target_position"]),
        "trade_allowed": bool(latest["trade_allowed"]),
        "is_review_day": bool(latest["is_review_day"]),
        "risk_off_override": bool(latest["risk_off_override"]),
        "allow_position_increase": bool(latest["allow_position_increase"]),
        "allow_position_decrease": bool(latest["allow_position_decrease"]),
        "signal_reason": str(latest["signal_reason"]),
        "automatic_ordering_authorized": False,
        "frozen_parameter_file": CONFIG_FILE.relative_to(ROOT).as_posix(),
        "frozen_parameter_sha256": _sha256(CONFIG_FILE),
        "input_latest_dates": latest_dates,
        "input_alignment_status": alignment["status"],
        "freshness_contract_sha256": alignment["contract_sha256"],
        "input_hashes": {
            ETF_FILE.name: _sha256(ETF_FILE),
            INDEX_FILE.name: _sha256(INDEX_FILE),
            TOTAL_RETURN_FILE.name: _sha256(TOTAL_RETURN_FILE),
            VALUATION_FILE.name: _sha256(VALUATION_FILE),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="刷新R5冻结模型的当日基础信号")
    parser.add_argument("--date", help="目标交易日YYYY-MM-DD，默认上海本地当天")
    parser.add_argument("--skip-refresh", action="store_true", help="只使用本地输入，供测试与审计")
    parser.add_argument("--audit-only", action="store_true", help="允许历史日期审计，但绝不覆盖最新纸面信号")
    args = parser.parse_args()
    local_today = pd.Timestamp(datetime.now(TIMEZONE).date())
    target = pd.Timestamp(args.date or local_today).normalize()
    if target != local_today and not args.audit_only:
        print("禁止历史回填：历史日期只能配合--audit-only且不得覆盖纸面信号", file=sys.stderr)
        return 2
    generated_at = datetime.now(TIMEZONE)
    started = perf_counter()
    status: dict[str, Any] = {
        "target_date": str(target.date()),
        "generated_at": generated_at.isoformat(),
        "status": "RUNNING",
        "steps": {},
        "automatic_ordering_authorized": False,
    }
    try:
        if not args.skip_refresh:
            step_started = perf_counter()
            status["steps"]["market"] = {
                "status": "SUCCESS",
                **refresh_market_series(target, generated_at),
                "duration_seconds": perf_counter() - step_started,
            }
            step_started = perf_counter()
            valuation = fetch_valuation()
            valuation = valuation.loc[pd.to_datetime(valuation["date"]).le(target)].copy()
            _atomic_parquet(valuation.sort_values("date").reset_index(drop=True), VALUATION_FILE)
            status["steps"]["valuation"] = {
                "status": "SUCCESS",
                "latest": str(pd.to_datetime(valuation["date"]).max().date()),
                "duration_seconds": perf_counter() - step_started,
            }
        step_started = perf_counter()
        signal = build_latest_signal(
            signal_date=target,
            config=yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8")),
            etf=pd.read_parquet(ETF_FILE),
            index=pd.read_parquet(INDEX_FILE),
            total_return=pd.read_parquet(TOTAL_RETURN_FILE),
            valuation_raw=pd.read_parquet(VALUATION_FILE),
        )
        status["steps"]["signal"] = {
            "status": "SUCCESS",
            "signal_date": signal["signal_date"],
            "duration_seconds": perf_counter() - step_started,
        }
        if args.audit_only:
            status["status"] = "AUDIT_ONLY_NO_PAPER_WRITE"
        else:
            signal["status"] = "SUCCESS"
            signal["consumable"] = True
            _atomic_json(OUTPUT_FILE, signal)
            status["status"] = "SUCCESS"
        print(json.dumps(signal, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        status["status"] = (
            exc.status if isinstance(exc, Round5InputDateMismatchError) else "FAILED"
        )
        status["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, Round5InputDateMismatchError):
            status["input_asof_dates"] = exc.input_asof_dates
            status["mismatch_reasons"] = exc.reasons
        if not args.audit_only:
            tombstone = _write_non_consumable_signal(
                target_date=target,
                failure_status=status["status"],
                error=status["error"],
                input_asof_dates=status.get("input_asof_dates"),
            )
            status["paper_signal_consumable"] = tombstone["consumable"]
        print(f"R5当日基础信号刷新失败：{status['error']}", file=sys.stderr)
        return 1
    finally:
        status["duration_seconds"] = perf_counter() - started
        _atomic_json(STATUS_FILE, status)


if __name__ == "__main__":
    raise SystemExit(main())
