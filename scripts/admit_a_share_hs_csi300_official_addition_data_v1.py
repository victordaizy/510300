"""执行沪深300官方调入候选 D4/D5 无收益数据准入。"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
import uuid
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.a_share_hs_csi300_official_addition_data_admission_v1 import (  # noqa: E402
    applicable_price_limit_rule,
    classify_entry_execution,
    classify_sellability,
    component_gate,
    market_row_component_coverage,
)
from research.a_share_hs_csi300_official_addition_forced_demand_v1 import (  # noqa: E402
    STRATEGY_ID,
    sha256_file,
    validate_no_result_columns,
)


DEFAULT_CONFIG = ROOT / "config" / "a_share_hs_csi300_official_addition_forced_demand_v1.yaml"
EVENT_STATUS_COLUMNS = (
    "event_key",
    "cycle_id",
    "ts_code",
    "entry_date",
    "effective_date",
    "holding_open_day_count",
    "market_row_coverage",
    "raw_open_coverage",
    "raw_close_coverage",
    "amount_coverage",
    "suspension_coverage",
    "market_available_at_coverage",
    "price_limit_phase_coverage",
    "linked_adjustment_factor_coverage",
    "corporate_action_query_covered",
    "entry_execution_status",
    "exit_resolution_status",
    "resolved_exit_date",
    "data_admission_status",
    "return_evaluation",
)


def now_shanghai() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_bytes(path, (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.parquet")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, path)


def assert_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} 缺失：{path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 不一致：expected={expected}, actual={actual}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": actual,
        "bytes": path.stat().st_size,
    }


def load_open_days(path: Path) -> list[date]:
    frame = pd.read_parquet(path, columns=["date", "is_open"])
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.date
    return sorted(set(frame.loc[frame["is_open"].astype(bool), "date"]))


def build_expected_grid(ledger: pd.DataFrame, open_days: list[date]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in ledger.itertuples(index=False):
        entry = pd.Timestamp(event.entry_date).date()
        effective = pd.Timestamp(event.effective_date).date()
        left = bisect.bisect_left(open_days, entry)
        right = bisect.bisect_right(open_days, effective)
        event_days = open_days[left:right]
        if not event_days or event_days[0] != entry or event_days[-1] != effective:
            raise ValueError(f"事件交易时钟不在冻结交易日历中：{event.event_key}")
        for trading_day in event_days:
            rows.append(
                {
                    "event_key": str(event.event_key),
                    "cycle_id": str(event.cycle_id),
                    "ts_code": str(event.ts_code),
                    "date": trading_day,
                    "entry_date": entry,
                    "effective_date": effective,
                    "is_entry": trading_day == entry,
                    "is_effective": trading_day == effective,
                }
            )
    grid = pd.DataFrame(rows)
    if grid.empty:
        raise ValueError("D4 预期事件日网格为空")
    return grid


def load_market_rows(panel_path: Path, grid: pd.DataFrame, allowed_columns: list[str]) -> pd.DataFrame:
    required = {
        "con_code",
        "date",
        "pre_close",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "amount",
        "volume",
        "is_suspended",
        "observed_traded_row",
        "observation_status",
        "available_at",
        "linked_adjustment_factor",
        "exchange",
        "market_day_index",
    }
    if set(allowed_columns) != required:
        raise ValueError("D4 行情允许字段集合与实现不一致")
    registered = grid.copy()
    registered["date"] = pd.to_datetime(registered["date"])
    connection = duckdb.connect(database=":memory:")
    try:
        connection.register("event_grid", registered)
        query = f"""
            SELECT
                g.event_key,
                g.cycle_id,
                g.ts_code,
                CAST(g.date AS DATE) AS date,
                CAST(g.entry_date AS DATE) AS entry_date,
                CAST(g.effective_date AS DATE) AS effective_date,
                g.is_entry,
                g.is_effective,
                m.con_code IS NOT NULL AS market_row_present,
                m.pre_close,
                m.raw_open,
                m.raw_high,
                m.raw_low,
                m.raw_close,
                m.amount,
                m.volume,
                m.is_suspended,
                m.observed_traded_row,
                m.observation_status,
                m.available_at,
                m.linked_adjustment_factor,
                m.exchange,
                m.market_day_index
            FROM event_grid AS g
            LEFT JOIN read_parquet('{sql_path(panel_path)}') AS m
              ON m.con_code = g.ts_code
             AND m.date = CAST(g.date AS DATE)
            ORDER BY g.event_key, g.date
        """
        result = connection.execute(query).fetch_df()
    finally:
        connection.close()
    if len(result) != len(grid):
        raise ValueError(f"行情连接后行数变化，可能存在重复键：{len(result)} != {len(grid)}")
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    result["entry_date"] = pd.to_datetime(result["entry_date"]).dt.normalize()
    result["effective_date"] = pd.to_datetime(result["effective_date"]).dt.normalize()
    result["available_at"] = pd.to_datetime(result["available_at"], errors="coerce")
    return result


def prepare_reference_maps(
    master_path: Path,
    status_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, pd.DataFrame]]:
    master = pd.read_parquet(
        master_path,
        columns=["ts_code", "list_date", "delist_date", "security_type", "source"],
    )
    master["ts_code"] = master["ts_code"].astype(str)
    master["list_date"] = pd.to_datetime(master["list_date"], errors="coerce").dt.normalize()
    master["delist_date"] = pd.to_datetime(master["delist_date"], errors="coerce").dt.normalize()
    duplicate_codes = master.loc[master["ts_code"].duplicated(keep=False), "ts_code"].unique().tolist()
    if duplicate_codes:
        raise ValueError(f"证券主表代码不唯一：{duplicate_codes[:20]}")
    master_map = {str(row.ts_code): row._asdict() for row in master.itertuples(index=False)}

    status = pd.read_parquet(
        status_path,
        columns=["ts_code", "status", "valid_from", "valid_to", "available_at", "source"],
    )
    status["ts_code"] = status["ts_code"].astype(str)
    for column in ("valid_from", "valid_to", "available_at"):
        status[column] = pd.to_datetime(status[column], errors="coerce")
    status_map = {
        str(code): group.sort_values(["valid_from", "valid_to"], kind="mergesort").reset_index(drop=True)
        for code, group in status.groupby("ts_code", sort=False)
    }
    return master_map, status_map


def listing_trade_day_number(list_date: pd.Timestamp | None, trading_day: pd.Timestamp, open_days: list[date]) -> int | None:
    if list_date is None or pd.isna(list_date):
        return None
    listed = pd.Timestamp(list_date).date()
    observed = pd.Timestamp(trading_day).date()
    if listed > observed:
        return None
    return bisect.bisect_right(open_days, observed) - bisect.bisect_left(open_days, listed)


def enrich_market_rows(
    rows: pd.DataFrame,
    *,
    master_map: dict[str, dict[str, Any]],
    status_map: dict[str, pd.DataFrame],
    open_days: list[date],
    status_contract: dict[str, Any],
    price_limit_rules: dict[str, Any],
) -> pd.DataFrame:
    inadmissible_sources = set(map(str, status_contract.get("inadmissible_sources") or []))
    enriched: list[dict[str, Any]] = []
    for raw in rows.to_dict("records"):
        ts_code = str(raw["ts_code"])
        trading_day = pd.Timestamp(raw["date"]).normalize()
        intervals = status_map.get(ts_code, pd.DataFrame())
        if intervals.empty:
            total_status_matches = 0
            admissible = intervals
        else:
            covering = intervals.loc[
                intervals["valid_from"].le(trading_day)
                & intervals["valid_to"].ge(trading_day)
            ].copy()
            total_status_matches = len(covering)
            day_close = pd.Timestamp.combine(trading_day.date(), time(23, 59, 59))
            admissible = covering.loc[
                covering["available_at"].le(day_close)
                & ~covering["source"].astype(str).isin(inadmissible_sources)
            ].copy()
        if len(admissible) == 1:
            status_value = str(admissible.iloc[0]["status"])
            status_source = str(admissible.iloc[0]["source"])
        else:
            status_value = None
            status_source = None
        master = master_map.get(ts_code) or {}
        list_date = master.get("list_date")
        listing_day = listing_trade_day_number(list_date, trading_day, open_days)
        rule = applicable_price_limit_rule(
            ts_code=ts_code,
            trade_date=trading_day,
            security_status=status_value,
            listing_trade_day_number=listing_day,
            list_date=list_date,
            rules=price_limit_rules,
        )
        available_at = raw.get("available_at")
        market_available = bool(
            raw.get("market_row_present")
            and available_at is not None
            and not pd.isna(available_at)
            and pd.Timestamp(available_at) <= pd.Timestamp.combine(trading_day.date(), time(23, 59, 59))
        )
        output = {
            **raw,
            "status_total_match_count": total_status_matches,
            "status_admissible_match_count": len(admissible),
            "security_status": status_value,
            "security_status_source": status_source,
            "list_date": list_date,
            "delist_date": master.get("delist_date"),
            "listing_trade_day_number": listing_day,
            "price_limit_board": rule.board,
            "price_limit_fraction": rule.fraction,
            "price_limit_rule_status": rule.status,
            "market_available_at_covered": market_available,
        }
        output.update(market_row_component_coverage(output, rule))
        enriched.append(output)
    return pd.DataFrame(enriched)


def rule_from_enriched(row: dict[str, Any]) -> Any:
    from research.a_share_hs_csi300_official_addition_data_admission_v1 import PriceLimitRule

    fraction = row.get("price_limit_fraction")
    return PriceLimitRule(
        str(row.get("price_limit_board")),
        None if fraction is None or pd.isna(fraction) else float(fraction),
        str(row.get("price_limit_rule_status")),
    )


def corporate_action_success_codes(contract: dict[str, Any]) -> tuple[set[str], list[dict[str, Any]]]:
    success_codes: set[str] = set()
    evidence: list[dict[str, Any]] = []
    for item in contract["corporate_action_success_receipts"]["paths"]:
        path = resolve_path(item["path"])
        evidence.append(assert_hash(path, str(item["sha256"]), "公司行为查询回执"))
        payload = json.loads(path.read_text(encoding="utf-8"))
        fetch_items = payload.get("fetch") or payload.get("fetch_receipts") or []
        for fetch in fetch_items:
            if str(fetch.get("status")) == "SUCCESS" and fetch.get("ts_code"):
                success_codes.add(str(fetch["ts_code"]))
    return success_codes, evidence


def make_search_grid(blocked: pd.DataFrame, open_days: list[date], coverage_end: date) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in blocked.itertuples(index=False):
        effective = pd.Timestamp(event.effective_date).date()
        start = bisect.bisect_right(open_days, effective)
        end = bisect.bisect_right(open_days, coverage_end)
        for trading_day in open_days[start:end]:
            rows.append(
                {
                    "event_key": str(event.event_key),
                    "cycle_id": str(event.cycle_id),
                    "ts_code": str(event.ts_code),
                    "date": trading_day,
                    "entry_date": pd.Timestamp(event.entry_date).date(),
                    "effective_date": effective,
                    "is_entry": False,
                    "is_effective": False,
                }
            )
    return pd.DataFrame(rows)


def build_event_status(
    ledger: pd.DataFrame,
    enriched_grid: pd.DataFrame,
    *,
    corporate_success_codes: set[str],
    panel_path: Path,
    allowed_columns: list[str],
    master_map: dict[str, dict[str, Any]],
    status_map: dict[str, pd.DataFrame],
    open_days: list[date],
    status_contract: dict[str, Any],
    price_limit_rules: dict[str, Any],
    coverage_end: date,
) -> pd.DataFrame:
    phase_rows = enriched_grid.loc[enriched_grid["is_entry"] | enriched_grid["is_effective"]].copy()
    entry_rows = phase_rows.loc[phase_rows["is_entry"]].set_index("event_key")
    effective_rows = phase_rows.loc[phase_rows["is_effective"]].set_index("event_key")
    event_records: list[dict[str, Any]] = []
    blocked_records: list[dict[str, Any]] = []

    for event in ledger.itertuples(index=False):
        event_key = str(event.event_key)
        group = enriched_grid.loc[enriched_grid["event_key"] == event_key]
        entry = entry_rows.loc[event_key].to_dict()
        effective = effective_rows.loc[event_key].to_dict()
        entry_status = classify_entry_execution(entry, rule_from_enriched(entry))
        effective_status = classify_sellability(
            effective,
            rule_from_enriched(effective),
            phase="EFFECTIVE_CLOSE",
        )
        if effective_status == "SELLABLE_EFFECTIVE_CLOSE":
            exit_status = "EXIT_AT_EFFECTIVE_CLOSE"
            resolved_exit_date = pd.Timestamp(event.effective_date).date().isoformat()
        elif effective_status.startswith("BLOCKED_EFFECTIVE_"):
            exit_status = "PENDING_DELAYED_EXIT_SEARCH"
            resolved_exit_date = None
            blocked_records.append(
                {
                    "event_key": event_key,
                    "cycle_id": str(event.cycle_id),
                    "ts_code": str(event.ts_code),
                    "entry_date": str(event.entry_date),
                    "effective_date": str(event.effective_date),
                }
            )
        else:
            exit_status = effective_status.replace("EFFECTIVE", "EXIT")
            resolved_exit_date = None
        event_records.append(
            {
                "event_key": event_key,
                "cycle_id": str(event.cycle_id),
                "ts_code": str(event.ts_code),
                "entry_date": str(event.entry_date),
                "effective_date": str(event.effective_date),
                "holding_open_day_count": int(len(group)),
                "market_row_coverage": float(group["market_row_covered"].mean()),
                "raw_open_coverage": float(group["raw_open_covered"].mean()),
                "raw_close_coverage": float(group["raw_close_covered"].mean()),
                "amount_coverage": float(group["amount_covered"].mean()),
                "suspension_coverage": float(group["suspension_covered"].mean()),
                "market_available_at_coverage": float(group["market_available_at_covered"].mean()),
                "price_limit_phase_coverage": float(
                    np.mean([bool(entry["price_limit_covered"]), bool(effective["price_limit_covered"])])
                ),
                "linked_adjustment_factor_coverage": float(group["linked_adjustment_factor_covered"].mean()),
                "corporate_action_query_covered": str(event.ts_code) in corporate_success_codes,
                "entry_execution_status": entry_status,
                "exit_resolution_status": exit_status,
                "resolved_exit_date": resolved_exit_date,
                "data_admission_status": "PENDING_GATE_AGGREGATION",
                "return_evaluation": "NOT_ALLOWED",
            }
        )

    event_status = pd.DataFrame(event_records)
    if blocked_records:
        blocked = pd.DataFrame(blocked_records)
        search_grid = make_search_grid(blocked, open_days, coverage_end)
        if search_grid.empty:
            event_status.loc[
                event_status["exit_resolution_status"].eq("PENDING_DELAYED_EXIT_SEARCH"),
                "exit_resolution_status",
            ] = "CENSORED_EXIT_PENDING_FIRST_SELLABLE_OPEN"
        else:
            search_market = load_market_rows(panel_path, search_grid, allowed_columns)
            search_enriched = enrich_market_rows(
                search_market,
                master_map=master_map,
                status_map=status_map,
                open_days=open_days,
                status_contract=status_contract,
                price_limit_rules=price_limit_rules,
            )
            for event_key, group in search_enriched.groupby("event_key", sort=False):
                resolution_status = "CENSORED_EXIT_PENDING_FIRST_SELLABLE_OPEN"
                resolved_date: str | None = None
                for row in group.sort_values("date", kind="mergesort").to_dict("records"):
                    state = classify_sellability(row, rule_from_enriched(row), phase="DELAYED_OPEN")
                    if state == "SELLABLE_DELAYED_OPEN":
                        resolution_status = "EXIT_AT_FIRST_SELLABLE_OPEN"
                        resolved_date = pd.Timestamp(row["date"]).date().isoformat()
                        break
                    if state.startswith("BLOCKED_DELAYED_"):
                        continue
                    resolution_status = state.replace("DELAYED", "EXIT")
                    break
                mask = event_status["event_key"].eq(event_key)
                event_status.loc[mask, "exit_resolution_status"] = resolution_status
                event_status.loc[mask, "resolved_exit_date"] = resolved_date

    return event_status[list(EVENT_STATUS_COLUMNS)]


def render_markdown(report: dict[str, Any]) -> str:
    d4 = report["gates"]["D4"]
    d5 = report["gates"]["D5"]
    component_lines = [
        f"- {name}：{value['covered']}/{value['required']}，覆盖率 {value['coverage']:.4%}，状态 `{value['status']}`。"
        for name, value in d4["components"].items()
    ]
    return "\n".join(
        [
            f"# {STRATEGY_ID} D4/D5 数据准入",
            "",
            f"- 总状态：`{report['status']}`",
            f"- 收益评估：`{report['return_evaluation']}`",
            f"- D4：`{d4['status']}`；要求每一组件均不低于 {d4['minimum_coverage']:.2%}。",
            *component_lines,
            f"- D5：`{d5['status']}`；已解决退出 {d5['resolved_events']}/{d5['required_events']}。",
            "",
            "本准入只输出覆盖布尔值、状态代码与日期；未输出价格或收益。",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="执行沪深300官方调入候选 D4/D5 无收益数据准入")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    contract = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if contract.get("strategy_id") != STRATEGY_ID:
        raise ValueError("策略标识不一致")
    admission = contract["data_admission_contract"]
    if admission["decision"] != "EXISTING_FROZEN_INPUTS_ONLY_NO_SOURCE_RESCUE":
        raise ValueError("D4/D5 必须禁止看到缺口后切换来源")
    if not admission["prohibit_return_columns_read"] or not admission["prohibit_price_values_in_outputs"]:
        raise ValueError("D4/D5 无收益和无价格输出边界未开启")

    artifacts = contract["artifacts"]
    ledger_path = resolve_path(artifacts["event_ledger_without_returns"])
    source_report_path = resolve_path(artifacts["acquisition_report_json"])
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    for gate in ("D1", "D2", "D3", "D8"):
        if source_report["gates"][gate]["status"] != "PASS":
            raise RuntimeError(f"{gate} 未通过，禁止执行 D4/D5")
    if source_report.get("return_evaluation") != "NOT_ALLOWED":
        raise RuntimeError("D4/D5 前收益评估必须保持 NOT_ALLOWED")

    evidence: dict[str, Any] = {}
    market_contract = admission["market_panel"]
    panel_path = resolve_path(market_contract["path"])
    evidence["market_panel"] = assert_hash(panel_path, market_contract["sha256"], "冻结行情面板")
    manifest_path = resolve_path(market_contract["input_manifest_path"])
    evidence["market_input_manifest"] = assert_hash(
        manifest_path,
        market_contract["input_manifest_sha256"],
        "行情输入清单",
    )
    panel_receipt_path = resolve_path(market_contract["receipt_path"])
    evidence["market_panel_receipt"] = assert_hash(
        panel_receipt_path,
        market_contract["receipt_sha256"],
        "行情面板回执",
    )
    master_contract = admission["security_master"]
    master_path = resolve_path(master_contract["path"])
    evidence["security_master"] = assert_hash(master_path, master_contract["sha256"], "证券主表")
    status_contract = admission["security_status_intervals"]
    status_path = resolve_path(status_contract["path"])
    evidence["security_status_intervals"] = assert_hash(
        status_path,
        status_contract["sha256"],
        "证券状态区间",
    )
    calendar_path = resolve_path(contract["calendar_contract"]["path"])
    evidence["trading_calendar"] = assert_hash(
        calendar_path,
        contract["calendar_contract"]["sha256"],
        "事件时钟交易日历",
    )

    ledger = pd.read_parquet(
        ledger_path,
        columns=["event_key", "cycle_id", "ts_code", "entry_date", "effective_date", "return_evaluation"],
    )
    if set(ledger["return_evaluation"]) != {"NOT_ALLOWED"}:
        raise RuntimeError("事件账本收益边界已变化")
    if len(ledger) != 425 or ledger["event_key"].nunique() != 425:
        raise ValueError(f"D4/D5 冻结事件数必须为 425，当前为 {len(ledger)}")
    open_days = load_open_days(calendar_path)
    expected_grid = build_expected_grid(ledger, open_days)
    market_rows = load_market_rows(panel_path, expected_grid, list(market_contract["allowed_columns"]))
    master_map, status_map = prepare_reference_maps(master_path, status_path)
    enriched = enrich_market_rows(
        market_rows,
        master_map=master_map,
        status_map=status_map,
        open_days=open_days,
        status_contract=status_contract,
        price_limit_rules=admission["price_limit_rules"],
    )
    corporate_success, corporate_evidence = corporate_action_success_codes(admission)
    evidence["corporate_action_success_receipts"] = corporate_evidence
    candidate_codes = set(ledger["ts_code"].astype(str))
    coverage_end = pd.Timestamp(market_contract["coverage_end"]).date()
    event_status = build_event_status(
        ledger,
        enriched,
        corporate_success_codes=corporate_success,
        panel_path=panel_path,
        allowed_columns=list(market_contract["allowed_columns"]),
        master_map=master_map,
        status_map=status_map,
        open_days=open_days,
        status_contract=status_contract,
        price_limit_rules=admission["price_limit_rules"],
        coverage_end=coverage_end,
    )

    minimum = float(contract["data_gates"]["D4"]["minimum_coverage"])
    phase_rows = enriched.loc[enriched["is_entry"] | enriched["is_effective"]]
    component_values = {
        "market_rows": (int(enriched["market_row_covered"].sum()), len(enriched)),
        "raw_open": (int(enriched["raw_open_covered"].sum()), len(enriched)),
        "raw_close": (int(enriched["raw_close_covered"].sum()), len(enriched)),
        "amount": (int(enriched["amount_covered"].sum()), len(enriched)),
        "suspension": (int(enriched["suspension_covered"].sum()), len(enriched)),
        "market_available_at": (int(enriched["market_available_at_covered"].sum()), len(enriched)),
        "price_limit_rule_and_fields": (int(phase_rows["price_limit_covered"].sum()), len(phase_rows)),
        "linked_adjustment_factor": (
            int(enriched["linked_adjustment_factor_covered"].sum()),
            len(enriched),
        ),
        "corporate_action_query": (len(candidate_codes & corporate_success), len(candidate_codes)),
    }
    components: dict[str, Any] = {}
    component_coverages: dict[str, float] = {}
    for name, (covered, required) in component_values.items():
        coverage = covered / required if required else 0.0
        component_coverages[name] = coverage
        components[name] = {
            "covered": covered,
            "required": required,
            "coverage": coverage,
            "status": "PASS" if coverage >= minimum else "FAIL",
        }
    d4_pass, d4_failures = component_gate(component_coverages, minimum)
    resolved_mask = event_status["exit_resolution_status"].isin(
        {"EXIT_AT_EFFECTIVE_CLOSE", "EXIT_AT_FIRST_SELLABLE_OPEN"}
    )
    resolved_count = int(resolved_mask.sum())
    d5_required = len(event_status)
    d5_coverage = resolved_count / d5_required if d5_required else 0.0
    d5_pass = d5_coverage >= float(contract["data_gates"]["D5"]["minimum_resolution"])

    overall_pass = d4_pass and d5_pass
    event_status["data_admission_status"] = np.where(
        overall_pass,
        "DATA_CONTRACT_PASS_PRE_RETURN",
        "NO_VIEW_DATA_CONTRACT_FAILED",
    )
    validate_no_result_columns(event_status.columns)
    forbidden_output_columns = {
        "pre_close",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "amount",
        "volume",
        "linked_adjustment_factor",
        "price_limit_fraction",
    }
    leaked = sorted(forbidden_output_columns & set(event_status.columns))
    if leaked:
        raise RuntimeError(f"D8 违规：事件准入产物泄露价格数值字段：{leaked}")

    event_status_path = resolve_path(artifacts["data_admission_event_status_without_prices"])
    atomic_write_parquet(event_status_path, event_status)
    report_path = resolve_path(artifacts["data_admission_report_json"])
    markdown_path = resolve_path(artifacts["data_admission_report_md"])
    report = {
        "schema_version": "1.0.0",
        "strategy_id": STRATEGY_ID,
        "generated_at": now_shanghai(),
        "status": "DATA_CONTRACT_PASS_READY_FOR_SINGLE_HISTORICAL_EVALUATION"
        if overall_pass
        else "NO_VIEW_DATA_CONTRACT_FAILED",
        "return_evaluation": "ALLOWED_ONCE_AFTER_FINAL_FREEZE" if overall_pass else "NOT_ALLOWED",
        "gates": {
            "D4": {
                "status": "PASS" if d4_pass else "FAIL",
                "minimum_coverage": minimum,
                "component_rule": admission["d4_component_rule"],
                "components": components,
                "failed_components": d4_failures,
            },
            "D5": {
                "status": "PASS" if d5_pass else "FAIL",
                "minimum_resolution": float(contract["data_gates"]["D5"]["minimum_resolution"]),
                "resolved_events": resolved_count,
                "required_events": d5_required,
                "resolution_fraction": d5_coverage,
                "unresolved_status_counts": event_status.loc[
                    ~resolved_mask,
                    "exit_resolution_status",
                ].value_counts().sort_index().to_dict(),
            },
            "D8": {
                "status": "PASS",
                "return_columns_read": [],
                "forbidden_market_columns_read": [],
                "price_values_output": False,
                "event_status_output_columns": list(event_status.columns),
            },
        },
        "event_count": int(len(event_status)),
        "unique_candidate_security_count": len(candidate_codes),
        "expected_event_day_rows": int(len(expected_grid)),
        "inputs": evidence,
        "market_columns_read": list(market_contract["allowed_columns"]),
        "market_columns_explicitly_not_read": list(market_contract["forbidden_columns"]),
        "event_status_artifact": {
            "path": event_status_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(event_status_path),
            "rows": int(len(event_status)),
            "columns": list(event_status.columns),
        },
        "decision": "READY_FOR_FINAL_PRE_RETURN_FREEZE"
        if overall_pass
        else "STOP_NO_HISTORICAL_RETURN_EVALUATION",
        "source_rescue_allowed": False,
        "historical_return_attempts_used": 0,
    }
    atomic_write_json(report_path, report)
    atomic_write_text(markdown_path, render_markdown(report))
    print(
        json.dumps(
            {
                "状态": report["status"],
                "D4": report["gates"]["D4"],
                "D5": report["gates"]["D5"],
                "收益评估": report["return_evaluation"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

