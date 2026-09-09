"""只记录冻结R6失败族的新日期研究信号；不生成仓位、订单或候选排名。"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_r6_walkforward import (
    MATRIX_FILE,
    R5_CONFIG_FILE,
    _build_r5_positions,
    _candidate_targets,
    _grid_targets,
    _load_inputs,
)


PROTOCOL_FILE = ROOT / "config" / "r6_postmortem_protocol.yaml"
ARCHIVE_DIR = ROOT / "data" / "forward" / "r6_failed_family_observation"
CANDIDATE_HISTORY_FILE = ARCHIVE_DIR / "candidate_signal_observations.parquet"
FAMILY_HISTORY_FILE = ARCHIVE_DIR / "family_median_observations.parquet"
STATUS_FILE = ROOT / "paper" / "r6_failed_family_observation" / "status.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _append_immutable_parquet(
    path: Path,
    additions: pd.DataFrame,
    *,
    unique_columns: list[str],
) -> pd.DataFrame:
    """只允许按日期向前追加；已有唯一键永不覆盖。"""

    additions = additions.copy()
    additions["observation_date"] = pd.to_datetime(additions["observation_date"])
    if path.exists():
        existing = pd.read_parquet(path)
        existing["observation_date"] = pd.to_datetime(existing["observation_date"])
        if not existing.empty and additions["observation_date"].min() <= existing[
            "observation_date"
        ].max():
            raise ValueError("观察档案只允许追加严格晚于已有记录的新日期")
        combined = pd.concat([existing, additions], ignore_index=True)
    else:
        combined = additions
    if combined.duplicated(unique_columns).any():
        raise ValueError(f"观察档案唯一键重复：{unique_columns}")
    combined = combined.sort_values(unique_columns).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    combined.to_parquet(temporary, index=False)
    temporary.replace(path)
    return combined


def _base_status(protocol: dict[str, Any]) -> dict[str, Any]:
    archive = protocol["observation_archive"]
    return {
        "archive": archive["name"],
        "status": "WAITING_FOR_START_DATE",
        "configured_start_date": archive["start_date"],
        "historical_data_cutoff": protocol["periods"]["historical_pseudo_oos"][1],
        "last_check_at": None,
        "latest_available_market_date": protocol["periods"]["historical_pseudo_oos"][1],
        "last_observation_date": None,
        "new_trading_days_observed": 0,
        "minimum_new_trading_days_before_evaluation": int(
            archive["minimum_new_trading_days_before_evaluation"]
        ),
        "evaluation_allowed": False,
        "primary_statistic": archive["primary_statistic"],
        "candidate_count": int(protocol["protocol"]["historical_candidates"]),
        "candidate_approved": False,
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
        "historical_reselection_allowed": False,
        "live_ranking_display_enabled": False,
        "failure_category": "NOT_STARTED",
        "message": "等待首个未见交易日；不得回填历史日期，也不产生仓位或订单。",
    }


def _write_waiting_status(
    protocol: dict[str, Any],
    *,
    checked_at: datetime,
    latest_market_date: pd.Timestamp | None,
    failure_category: str,
    message: str,
    status_name: str = "WAITING_FOR_CURRENT_DATA",
) -> dict[str, Any]:
    status = _base_status(protocol)
    if STATUS_FILE.exists():
        previous = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        for key in (
            "last_observation_date",
            "new_trading_days_observed",
            "evaluation_allowed",
        ):
            status[key] = previous.get(key, status[key])
    status.update(
        {
            "status": status_name,
            "last_check_at": checked_at.isoformat(),
            "latest_available_market_date": (
                str(latest_market_date.date()) if latest_market_date is not None else None
            ),
            "failure_category": failure_category,
            "message": message,
        }
    )
    _atomic_json_write(STATUS_FILE, status)
    return status


def build_observation_snapshot(
    matrix: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    observation_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算全部冻结候选和族中位数，仅输出研究信号字段。"""

    r5_config = yaml.safe_load(R5_CONFIG_FILE.read_text(encoding="utf-8"))
    r5_positions = _build_r5_positions(frames["index"], frames["valuation"], r5_config)
    execution = matrix["execution"]
    rows: list[dict[str, Any]] = []
    for candidate in matrix["candidates"]:
        continuous = _candidate_targets(
            candidate, frames["etf"], r5_positions, execution
        )
        current = continuous.loc[
            pd.to_datetime(continuous["date"]).eq(observation_date)
        ]
        if len(current) != 1 or pd.isna(current.iloc[0]["target_position"]):
            raise ValueError(f"{candidate['id']}在观察日没有唯一、有效的研究信号")
        grid10 = _grid_targets(continuous, float(execution["research_grid_step"]))
        grid25 = _grid_targets(continuous, float(execution["small_account_grid_step"]))
        grid10_current = grid10.loc[pd.to_datetime(grid10["date"]).eq(observation_date)]
        grid25_current = grid25.loc[pd.to_datetime(grid25["date"]).eq(observation_date)]
        row = current.iloc[0]
        rows.append(
            {
                "observation_date": observation_date,
                "candidate_id": candidate["id"],
                "family": candidate["family"],
                "continuous_research_signal": float(row["target_position"]),
                "grid_10pct_research_signal": float(
                    grid10_current.iloc[0]["target_position"]
                ),
                "grid_25pct_research_signal": float(
                    grid25_current.iloc[0]["target_position"]
                ),
                "signal_reason": str(row["signal_reason"]),
                "review_day": bool(row["trade_allowed"]),
                "risk_off_override": bool(row["risk_off_override"]),
                "historically_rejected": True,
                "approved": False,
            }
        )
    candidates = pd.DataFrame(rows).sort_values("candidate_id").reset_index(drop=True)
    signal_columns = [
        "continuous_research_signal",
        "grid_10pct_research_signal",
        "grid_25pct_research_signal",
    ]
    families = (
        candidates.groupby(["observation_date", "family"], as_index=False)[
            signal_columns
        ]
        .median()
        .rename(columns={column: f"family_median_{column}" for column in signal_columns})
    )
    families["historically_rejected"] = True
    families["approved"] = False
    return candidates, families


def main() -> int:
    checked_at = datetime.now(ZoneInfo("Asia/Shanghai"))
    today = pd.Timestamp(checked_at.date())
    protocol = yaml.safe_load(PROTOCOL_FILE.read_text(encoding="utf-8"))
    matrix = yaml.safe_load(MATRIX_FILE.read_text(encoding="utf-8"))
    frozen_hash = protocol["protocol"]["source_experiment_matrix_sha256"]
    if _sha256(MATRIX_FILE) != frozen_hash:
        raise ValueError("R6实验矩阵已偏离冻结哈希，拒绝记录")
    if len(matrix["candidates"]) != int(protocol["protocol"]["historical_candidates"]):
        raise ValueError("R6冻结候选数量不一致，拒绝记录")

    start_date = pd.Timestamp(protocol["observation_archive"]["start_date"])
    if today < start_date:
        status = _write_waiting_status(
            protocol,
            checked_at=checked_at,
            latest_market_date=pd.Timestamp(
                protocol["periods"]["historical_pseudo_oos"][1]
            ),
            failure_category="NOT_STARTED",
            message=f"观察从{start_date.date()}开始；今天尚未到开始日期。",
            status_name="WAITING_FOR_START_DATE",
        )
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0

    frames, paths = _load_inputs(matrix)
    latest_market_date = pd.Timestamp(frames["etf"]["date"].max()).normalize()
    if latest_market_date != today:
        status = _write_waiting_status(
            protocol,
            checked_at=checked_at,
            latest_market_date=latest_market_date,
            failure_category="CURRENT_DATE_DATA_UNAVAILABLE",
            message=(
                f"今天是{today.date()}，ETF数据最新为{latest_market_date.date()}；"
                "未用旧日期冒充今日观察，也未回填。"
            ),
        )
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0

    if CANDIDATE_HISTORY_FILE.exists():
        existing = pd.read_parquet(CANDIDATE_HISTORY_FILE)
        existing_dates = pd.to_datetime(existing["observation_date"])
        if today in set(existing_dates):
            status = _write_waiting_status(
                protocol,
                checked_at=checked_at,
                latest_market_date=latest_market_date,
                failure_category="ALREADY_OBSERVED",
                message=f"{today.date()}已记录；不可覆盖或重复追加。",
            )
            status["status"] = "OBSERVING"
            _atomic_json_write(STATUS_FILE, status)
            print(json.dumps(status, ensure_ascii=False, indent=2))
            return 0

    candidate_snapshot, family_snapshot = build_observation_snapshot(
        matrix, frames, today
    )
    candidate_history = _append_immutable_parquet(
        CANDIDATE_HISTORY_FILE,
        candidate_snapshot,
        unique_columns=["observation_date", "candidate_id"],
    )
    _append_immutable_parquet(
        FAMILY_HISTORY_FILE,
        family_snapshot,
        unique_columns=["observation_date", "family"],
    )
    observed_days = int(candidate_history["observation_date"].nunique())
    minimum_days = int(
        protocol["observation_archive"]["minimum_new_trading_days_before_evaluation"]
    )
    status = _base_status(protocol)
    status.update(
        {
            "status": "OBSERVING",
            "last_check_at": checked_at.isoformat(),
            "latest_available_market_date": str(latest_market_date.date()),
            "last_observation_date": str(today.date()),
            "new_trading_days_observed": observed_days,
            "evaluation_allowed": observed_days >= minimum_days,
            "failure_category": "NONE",
            "message": "已追加冻结失败族研究信号；未产生仓位、订单或候选排名。",
            "source_hashes": {
                MATRIX_FILE.relative_to(ROOT).as_posix(): _sha256(MATRIX_FILE),
                R5_CONFIG_FILE.relative_to(ROOT).as_posix(): _sha256(R5_CONFIG_FILE),
                **{
                    path.relative_to(ROOT).as_posix(): _sha256(path)
                    for path in paths.values()
                },
            },
        }
    )
    _atomic_json_write(STATUS_FILE, status)
    print(
        json.dumps(
            {
                "status": status["status"],
                "observation_date": status["last_observation_date"],
                "new_trading_days_observed": observed_days,
                "evaluation_allowed": status["evaluation_allowed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
