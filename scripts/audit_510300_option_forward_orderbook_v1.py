"""审计510300期权前向收盘盘口V1，不计算因子、仓位或策略收益。"""

from __future__ import annotations

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

from scripts.collect_510300_option_forward_orderbook_v1 import (  # noqa: E402
    sha256,
    validate_snapshot,
)
from scripts.freeze_510300_option_forward_orderbook_v1 import verify_protocol  # noqa: E402


TIMEZONE = ZoneInfo("Asia/Shanghai")
MARKET_FILE = ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet"
DIVIDEND_FILE = ROOT / "data" / "reference" / "510300_dividends.csv"


def _json_value(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _forward_outcome(
    market: pd.DataFrame,
    dividend_by_date: pd.Series,
    signal_position: int,
    horizon: int,
    cash_rate: float,
    trading_days: int,
) -> dict[str, Any] | None:
    entry_position = signal_position + 1
    maturity_position = entry_position + horizon - 1
    if maturity_position >= len(market):
        return None
    entry = market.iloc[entry_position]
    entry_open = float(entry["open"])
    if entry_open <= 0:
        raise ValueError("510300出现非正开盘价")
    wealth = float(entry["close"]) / entry_open
    previous_close = float(entry["close"])
    for position in range(entry_position + 1, maturity_position + 1):
        day = market.iloc[position]
        dividend = float(dividend_by_date.get(pd.Timestamp(day["date"]), 0.0))
        wealth *= (float(day["close"]) + dividend) / previous_close
        previous_close = float(day["close"])
    cash_wealth = (1.0 + cash_rate) ** (horizon / trading_days)
    return {
        "entry_date": pd.Timestamp(entry["date"]),
        "maturity_date": pd.Timestamp(market.iloc[maturity_position]["date"]),
        "relative_return": float(wealth - cash_wealth),
    }


def build_forward_outcomes(eligible_dates: list[pd.Timestamp], registry: dict) -> pd.DataFrame:
    columns = [
        "signal_date",
        "signal_market_position",
        "maturity_date20",
        "relative_return20",
        "bad20",
        "maturity_date60",
        "relative_return60",
    ]
    if not eligible_dates or not MARKET_FILE.exists() or not DIVIDEND_FILE.exists():
        return pd.DataFrame(columns=columns)
    market = pd.read_parquet(MARKET_FILE)
    required_market = {"date", "open", "close"}
    missing_market = sorted(required_market.difference(market.columns))
    if missing_market:
        raise ValueError(f"510300日线缺少字段：{missing_market}")
    market = market.copy()
    market["date"] = pd.to_datetime(market["date"], errors="coerce").dt.normalize()
    market = market.dropna(subset=["date", "open", "close"]).sort_values("date")
    if market["date"].duplicated().any():
        raise ValueError("510300日线日期重复")
    market = market.reset_index(drop=True)
    position_by_date = pd.Series(market.index, index=market["date"]).to_dict()
    dividends = pd.read_csv(DIVIDEND_FILE)
    required_dividends = {"symbol", "ex_date", "cash_dividend_per_share"}
    missing_dividends = sorted(required_dividends.difference(dividends.columns))
    if missing_dividends:
        raise ValueError(f"510300分红表缺少字段：{missing_dividends}")
    dividends = dividends.loc[dividends["symbol"].astype(str).eq("510300.SH")].copy()
    dividends["ex_date"] = pd.to_datetime(dividends["ex_date"], errors="coerce").dt.normalize()
    dividends["cash_dividend_per_share"] = pd.to_numeric(
        dividends["cash_dividend_per_share"], errors="coerce"
    )
    if dividends[["ex_date", "cash_dividend_per_share"]].isna().any(axis=None):
        raise ValueError("510300分红表存在无效日期或金额")
    dividend_by_date = dividends.groupby("ex_date")["cash_dividend_per_share"].sum()
    cash_rate = float(registry["targets"]["cash_annual_rate"])
    trading_days = int(registry["targets"]["trading_days_per_year"])
    bad20_threshold = float(registry["targets"]["primary"]["threshold"])
    rows: list[dict[str, Any]] = []
    for date in sorted(pd.Timestamp(value).normalize() for value in eligible_dates):
        position = position_by_date.get(date)
        if position is None:
            continue
        outcome20 = _forward_outcome(
            market, dividend_by_date, int(position), 20, cash_rate, trading_days
        )
        outcome60 = _forward_outcome(
            market, dividend_by_date, int(position), 60, cash_rate, trading_days
        )
        rows.append(
            {
                "signal_date": date,
                "signal_market_position": int(position),
                "maturity_date20": outcome20["maturity_date"] if outcome20 else pd.NaT,
                "relative_return20": outcome20["relative_return"] if outcome20 else float("nan"),
                "bad20": (
                    bool(outcome20["relative_return"] <= bad20_threshold)
                    if outcome20
                    else pd.NA
                ),
                "maturity_date60": outcome60["maturity_date"] if outcome60 else pd.NaT,
                "relative_return60": outcome60["relative_return"] if outcome60 else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def count_independent_bad_episodes(outcomes: pd.DataFrame, merge_gap: int) -> int:
    bad = outcomes.loc[outcomes["bad20"].eq(True)].sort_values("signal_market_position")
    if bad.empty:
        return 0
    positions = bad["signal_market_position"].astype(int).tolist()
    return 1 + sum(current - previous > merge_gap for previous, current in zip(positions, positions[1:]))


def audit_snapshot_pair(
    snapshot_path: Path,
    master_path: Path,
    expected_date: pd.Timestamp,
    contract: dict,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "date": str(expected_date.date()),
        "snapshot": snapshot_path.relative_to(ROOT).as_posix(),
        "master_snapshot": master_path.relative_to(ROOT).as_posix(),
    }
    if not master_path.exists():
        return {**result, "status": "NO_VIEW", "errors": ["缺少同日动态主表"]}
    try:
        snapshot = pd.read_parquet(snapshot_path)
        master = pd.read_parquet(master_path)
        missing = sorted(set(contract["quality"]["required_columns"]).difference(snapshot.columns))
        errors: list[str] = []
        if missing:
            errors.append(f"盘口缺少字段：{missing}")
        if snapshot.empty:
            errors.append("盘口为空")
        if master.empty:
            errors.append("动态主表为空")
        if errors:
            return {**result, "status": "NO_VIEW", "errors": errors}
        file_dates = pd.to_datetime(snapshot["expected_trade_date"], errors="coerce").dt.normalize()
        if file_dates.isna().any() or set(file_dates) != {expected_date.normalize()}:
            errors.append("文件内预期交易日与文件名不一致")
        started_values = pd.to_datetime(snapshot["capture_started_at"], errors="coerce")
        finished_values = pd.to_datetime(snapshot["capture_finished_at"], errors="coerce")
        if started_values.isna().any() or finished_values.isna().any():
            errors.append("采集墙钟时间无效")
        if started_values.nunique() != 1 or finished_values.nunique() != 1:
            errors.append("同一快照存在多个采集墙钟时间")
        if errors:
            return {**result, "status": "NO_VIEW", "errors": errors}
        started_at = started_values.iloc[0].to_pydatetime()
        finished_at = finished_values.iloc[0].to_pydatetime()
        audit = validate_snapshot(snapshot, master, expected_date, started_at, finished_at, contract)
        status = "PASS" if audit["status"] == "PASS" else "NO_VIEW"
        return {
            **result,
            "status": status,
            "errors": [] if status == "PASS" else ["至少一个冻结质量门槛失败"],
            "snapshot_sha256": sha256(snapshot_path),
            "master_snapshot_sha256": sha256(master_path),
            "audit": audit,
        }
    except Exception as exc:
        return {
            **result,
            "status": "NO_VIEW",
            "errors": [f"{type(exc).__name__}: {str(exc)[:500]}"],
        }


def render_markdown(report: dict) -> str:
    gates = report["forward_gates"]
    lines = [
        "# 510300期权前向收盘盘口V1审计",
        "",
        f"- 状态：`{report['status']}`",
        f"- 生成时间：{report['generated_at']}",
        f"- 合格盘口日：{report['eligible_orderbook_days']} / {gates['eligible_orderbook_days']['required']}",
        f"- 已成熟20日结果：{report['matured_20d_outcomes']} / {gates['matured_20d_outcomes']['required']}",
        f"- 已成熟60日结果：{report['matured_60d_outcomes']} / {gates['matured_60d_outcomes']['required']}",
        f"- 独立BAD20状态：{report['independent_bad_state_episodes']} / {gates['independent_bad_state_episodes']['required']}",
        f"- 首个合格前向日：{report['first_eligible_date'] or '尚无'}",
        f"- 最近合格前向日：{report['last_eligible_date'] or '尚无'}",
        "",
        "## 治理结论",
        "",
        report["conclusion"],
    ]
    failures = [item for item in report["daily_audits"] if item["status"] != "PASS"]
    if failures:
        lines.extend(["", "## 失败日期", ""])
        for item in failures:
            lines.append(f"- {item['date']}：{'；'.join(item['errors'])}")
    return "\n".join(lines) + "\n"


def main() -> int:
    contract, manifest = verify_protocol()
    registry = yaml.safe_load(
        (ROOT / contract["protocol"]["upstream_registry"]).read_text(encoding="utf-8")
    )
    snapshot_dir = ROOT / contract["paths"]["snapshot_directory"]
    master_dir = ROOT / contract["paths"]["master_snapshot_directory"]
    daily_audits: list[dict[str, Any]] = []
    snapshot_files = (
        {path.name: path for path in snapshot_dir.glob("*.parquet")}
        if snapshot_dir.exists()
        else {}
    )
    master_files = (
        {path.name: path for path in master_dir.glob("*.parquet")}
        if master_dir.exists()
        else {}
    )
    for filename in sorted(set(snapshot_files) | set(master_files)):
        snapshot_path = snapshot_files.get(filename, snapshot_dir / filename)
        master_path = master_files.get(filename, master_dir / filename)
        try:
            expected_date = pd.Timestamp(
                datetime.strptime(snapshot_path.stem, "%Y%m%d").date()
            )
        except ValueError:
            daily_audits.append(
                {
                    "date": snapshot_path.stem,
                    "snapshot": snapshot_path.relative_to(ROOT).as_posix(),
                    "status": "NO_VIEW",
                    "errors": ["文件名不是YYYYMMDD"],
                }
            )
            continue
        daily_audits.append(
            audit_snapshot_pair(
                snapshot_path,
                master_path,
                expected_date,
                contract,
            )
        )
    eligible_dates = [pd.Timestamp(item["date"]) for item in daily_audits if item["status"] == "PASS"]
    outcomes = build_forward_outcomes(eligible_dates, registry)
    matured20 = int(outcomes["maturity_date20"].notna().sum()) if len(outcomes) else 0
    matured60 = int(outcomes["maturity_date60"].notna().sum()) if len(outcomes) else 0
    merge_gap = int(registry["evaluation"]["independent_episode_merge_gap_trading_days"])
    bad_episodes = count_independent_bad_episodes(outcomes, merge_gap) if len(outcomes) else 0
    required = contract["forward_gates"]
    gate_values = {
        "eligible_orderbook_days": (len(eligible_dates), int(required["eligible_orderbook_days_minimum"])),
        "matured_20d_outcomes": (matured20, int(required["matured_20d_outcomes_minimum"])),
        "matured_60d_outcomes": (matured60, int(required["matured_60d_outcomes_minimum"])),
        "independent_bad_state_episodes": (bad_episodes, int(required["independent_bad_state_episodes_minimum"])),
    }
    gates = {
        key: {"actual": actual, "required": threshold, "pass": actual >= threshold}
        for key, (actual, threshold) in gate_values.items()
    }
    invalid_count = sum(item["status"] != "PASS" for item in daily_audits)
    all_gates = all(item["pass"] for item in gates.values())
    if invalid_count:
        status = "NO_VIEW_INVALID_FORWARD_SNAPSHOT"
        conclusion = "至少一个日期文件未通过冻结质量门槛；不得计算O1至O4，更不得进行收益检验。"
    elif all_gates:
        status = "DATA_GATE_PASS"
        conclusion = "四项前向数据门槛均已通过；仅授权另建冻结的信号检验，不自动授权仓位、订单或实盘。"
    elif eligible_dates:
        status = "COLLECTING"
        conclusion = "前向证据仍未成熟；不得计算O1至O4、策略收益或每日买卖指南。"
    else:
        status = "COLLECTING_NO_ELIGIBLE_DAYS"
        conclusion = "尚无合格的冻结后前向盘口；当前唯一有效结论是NO_VIEW。"
    start_path = ROOT / contract["paths"]["forward_start_status"]
    forward_start = json.loads(start_path.read_text(encoding="utf-8")) if start_path.exists() else None
    report = {
        "project_id": contract["protocol"]["project_id"],
        "status": status,
        "generated_at": datetime.now(TIMEZONE).isoformat(),
        "protocol_manifest_sha256": sha256(ROOT / contract["paths"]["protocol_manifest"]),
        "protocol_frozen_at": manifest["frozen_at"],
        "forward_start": forward_start,
        "eligible_orderbook_days": len(eligible_dates),
        "matured_20d_outcomes": matured20,
        "matured_60d_outcomes": matured60,
        "independent_bad_state_episodes": bad_episodes,
        "first_eligible_date": str(min(eligible_dates).date()) if eligible_dates else None,
        "last_eligible_date": str(max(eligible_dates).date()) if eligible_dates else None,
        "market_data_last_date": (
            str(pd.to_datetime(pd.read_parquet(MARKET_FILE, columns=["date"])["date"]).max().date())
            if MARKET_FILE.exists()
            else None
        ),
        "forward_gates": gates,
        "return_test_authorized": bool(all_gates and invalid_count == 0),
        "position_mapping": "DISABLED",
        "order_generation": "DISABLED",
        "live_trading": "NOT_AUTHORIZED",
        "daily_audits": daily_audits,
        "conclusion": conclusion,
    }
    json_path = ROOT / contract["paths"]["audit_json"]
    markdown_path = ROOT / contract["paths"]["audit_markdown"]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_value), encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_value))
    return 0 if invalid_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
