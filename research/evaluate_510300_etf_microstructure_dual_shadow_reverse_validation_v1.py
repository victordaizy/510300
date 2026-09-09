"""按冻结规则评估510300双影子候选的2012至2021反向历史快照。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = PROJECT_ROOT / "research"
if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

import validate_etf_microstructure_dual_shadow_v1_frozen as frozen_candidate


PROJECT_ID = "510300_ETF_MICROSTRUCTURE_DUAL_SHADOW_REVERSE_VALIDATION_V1"
POLICY = "MICRO_CONSENSUS3_OR_SHORT_EXTREMES_DUAL_SHADOW"
CONFIG_FILE = (
    PROJECT_ROOT
    / "config"
    / "510300_etf_microstructure_dual_shadow_reverse_validation_v1.yaml"
)
FREEZE_MANIFEST = CONFIG_FILE.with_name(
    "510300_etf_microstructure_dual_shadow_reverse_validation_v1_manifest.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")


def _write_new_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"拒绝覆盖评估证据：{path}")
    content = (
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
    ).encode("utf-8")
    with path.open("xb") as handle:
        handle.write(content)


def _write_new_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"拒绝覆盖评估证据：{path}")
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.rename(path)


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _normalize(frame: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    result = frame.copy()
    result[column] = pd.to_datetime(result[column], errors="raise").dt.normalize()
    result.sort_values(column, inplace=True)
    result.reset_index(drop=True, inplace=True)
    if result[column].duplicated().any():
        raise ValueError(f"{column}存在重复日期")
    return result


def _load_freeze() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not CONFIG_FILE.exists() or not FREEZE_MANIFEST.exists():
        raise FileNotFoundError("反向验证配置或冻结清单不存在")
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    if config["protocol"]["project_id"] != PROJECT_ID:
        raise ValueError("反向验证项目编号不匹配")
    if freeze.get("project_id") != PROJECT_ID or not freeze.get(
        "implementation_frozen", False
    ):
        raise ValueError("反向验证冻结清单无效")
    mismatches: dict[str, str] = {}
    for group in ["tracked_files", "historical_input_files_at_freeze"]:
        for relative, expected in freeze[group].items():
            path = PROJECT_ROOT / relative
            actual = _sha256(path) if path.exists() else "MISSING"
            if actual != expected:
                mismatches[relative] = f"expected={expected},actual={actual}"
    if mismatches:
        raise ValueError(f"反向验证冻结文件哈希漂移：{mismatches}")
    candidate_config_path = PROJECT_ROOT / config["candidate_contract"]["frozen_config"]
    candidate_config = yaml.safe_load(candidate_config_path.read_text(encoding="utf-8"))
    return config, freeze, candidate_config


def _verify_objective_contract(
    config: dict[str, Any], candidate_config: dict[str, Any]
) -> dict[str, Any]:
    expected = config["objective"]
    account = candidate_config["account"]
    costs = candidate_config["costs"]
    objective = candidate_config["objective"]
    checks = {
        "initial_capital_cny": float(account["initial_capital_cny"]),
        "cash_annual_rate": float(account["cash_annual_rate"]),
        "annualization_trading_days": int(account["trading_days_per_year"]),
        "lot_size_shares": int(account["lot_size_shares"]),
        "commission_rate_per_leg": float(costs["commission_rate_per_leg"]),
        "base_slippage_bps_per_leg": float(costs["base_slippage_bps_per_leg"]),
        "stress_slippage_bps_per_leg": float(costs["stress_slippage_bps_per_leg"]),
        "minimum_annualized_net_excess": float(
            objective["minimum_annualized_net_excess"]
        ),
        "rolling_window_trading_days": int(
            objective["rolling_window_trading_days"]
        ),
        "minimum_rolling_242d_excess_median": float(
            objective["minimum_rolling_242d_excess_median"]
        ),
        "start_perturbations_trading_days": [
            int(value) for value in objective["start_perturbations_trading_days"]
        ],
    }
    for key, actual in checks.items():
        frozen_value = expected[key]
        if isinstance(actual, list):
            frozen_value = [int(value) for value in frozen_value]
            equal = actual == frozen_value
        elif isinstance(actual, int):
            equal = actual == int(frozen_value)
        else:
            equal = float(actual) == float(frozen_value)
        if not equal:
            raise ValueError(
                f"外部验证目标契约与候选冻结契约不一致：{key}，"
                f"external={frozen_value},candidate={actual}"
            )
    return checks


def _load_snapshot(
    config: dict[str, Any], run_id: str
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], pd.DataFrame]:
    snapshot_root = (PROJECT_ROOT / config["paths"]["snapshot_root"]).resolve()
    snapshot_dir = (snapshot_root / run_id).resolve()
    if not snapshot_dir.is_relative_to(snapshot_root):
        raise ValueError("run-id导致快照目录越界")
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"快照清单不存在：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("project_id") != PROJECT_ID
        or manifest.get("status") != "PASS_COMPLETE_EXTERNAL_SIGNAL_SNAPSHOT"
        or manifest.get("run_id") != run_id
    ):
        raise ValueError("快照清单状态或身份不匹配")
    if manifest.get("freeze_manifest_sha256") != _sha256(FREEZE_MANIFEST):
        raise ValueError("快照绑定的反向验证冻结清单不匹配")
    frames: dict[str, pd.DataFrame] = {}
    for name in ["etf", "benchmark", "nav", "fund_share", "margin"]:
        record = manifest["datasets"][name]
        path = PROJECT_ROOT / record["file"]
        if not path.resolve().is_relative_to(snapshot_dir):
            raise ValueError(f"快照数据越出run目录：{name}")
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"快照数据哈希漂移：{name}")
        frame = _normalize(pd.read_parquet(path))
        if int(len(frame)) != int(record["rows"]):
            raise ValueError(f"快照数据行数漂移：{name}")
        frames[name] = frame
    dividend_record = manifest["datasets"]["dividends"]
    dividend_path = PROJECT_ROOT / dividend_record["file"]
    if not dividend_path.resolve().is_relative_to(snapshot_dir):
        raise ValueError("分红快照越出run目录")
    if _sha256(dividend_path) != dividend_record["sha256"]:
        raise ValueError("分红快照哈希漂移")
    dividends = pd.read_csv(dividend_path)
    for column in ["record_date", "ex_date", "payment_date"]:
        dividends[column] = pd.to_datetime(dividends[column], errors="raise").dt.normalize()
    dividends["cash_dividend_per_share"] = pd.to_numeric(
        dividends["cash_dividend_per_share"], errors="raise"
    )
    dividends.sort_values("ex_date", inplace=True)
    dividends.reset_index(drop=True, inplace=True)
    manifest["manifest_file"] = _relative(manifest_path)
    manifest["manifest_sha256"] = _sha256(manifest_path)
    return manifest, frames, dividends


def _build_signal_and_market(
    config: dict[str, Any],
    frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dates = config["dates"]
    signal_start = pd.Timestamp(dates["signal_collection_start"])
    signal_end = pd.Timestamp(dates["signal_collection_end"])
    market_end = pd.Timestamp(dates["execution_market_end"])
    calendar = frames["etf"].loc[
        frames["etf"]["date"].between(signal_start, signal_end), ["date"]
    ].copy()
    signal = (
        calendar.merge(
            frames["nav"][["date", "close_premium_to_nav"]],
            on="date",
            how="left",
            validate="one_to_one",
        )
        .merge(
            frames["fund_share"][["date", "fund_shares"]],
            on="date",
            how="left",
            validate="one_to_one",
        )
        .merge(
            frames["margin"][["date", "rzye", "rqye", "rqyl", "rzrqye"]],
            on="date",
            how="left",
            validate="one_to_one",
        )
    )
    required = [
        "close_premium_to_nav",
        "fund_shares",
        "rzye",
        "rqye",
        "rqyl",
        "rzrqye",
    ]
    if signal[required].isna().any(axis=None):
        missing = signal.loc[signal[required].isna().any(axis=1), "date"]
        raise ValueError(
            f"外部信号输入存在缺失：{missing.dt.strftime('%Y-%m-%d').tolist()[:20]}"
        )
    market = frames["etf"].loc[
        frames["etf"]["date"].between(signal_start, market_end),
        ["date", "open", "close"],
    ].rename(columns={"open": "etf_open", "close": "etf_close"})
    market = market.merge(
        frames["benchmark"][["date", "close"]].rename(
            columns={"close": "benchmark_close"}
        ),
        on="date",
        how="left",
        validate="one_to_one",
    )
    if market[["etf_open", "etf_close", "benchmark_close"]].isna().any(axis=None):
        raise ValueError("外部执行行情或H00300存在缺失")
    audit = {
        "signal_rows": int(len(signal)),
        "signal_first_date": signal["date"].min().date().isoformat(),
        "signal_last_date": signal["date"].max().date().isoformat(),
        "market_rows": int(len(market)),
        "market_first_date": market["date"].min().date().isoformat(),
        "market_last_date": market["date"].max().date().isoformat(),
        "complete_required_fields": True,
    }
    return signal, market, audit


def _evaluate(
    *,
    config: dict[str, Any],
    candidate_config: dict[str, Any],
    features: pd.DataFrame,
    market: pd.DataFrame,
    dividends: pd.DataFrame,
) -> pd.DataFrame:
    execution = market.merge(
        features[["date", "cash_final"]],
        on="date",
        how="left",
        validate="one_to_one",
    )
    execution["cash_execution"] = execution["cash_final"].shift(1).fillna(False)
    base_costs, stress_costs, objective = frozen_candidate._cost_models(candidate_config)
    initial_capital = float(candidate_config["account"]["initial_capital_cny"])
    perturbations = [
        int(value)
        for value in config["objective"]["start_perturbations_trading_days"]
    ]
    rows: list[dict[str, Any]] = []
    for period, period_dates in config["dates"]["periods"].items():
        start = pd.Timestamp(period_dates["start"])
        end = pd.Timestamp(period_dates["end"])
        selected = np.flatnonzero(execution["date"].between(start, end).to_numpy())
        if len(selected) <= max(perturbations):
            raise ValueError(f"期间{period}不足以执行五起点评价")
        last = int(selected[-1])
        for perturbation in perturbations:
            first = int(selected[perturbation])
            if first <= 0:
                raise ValueError(f"期间{period}没有前一交易日用于账户初始化")
            sample = execution.iloc[first - 1 : last + 1][
                ["date", "etf_open", "etf_close", "benchmark_close"]
            ].reset_index(drop=True)
            states = np.ones(len(sample), dtype=np.int8)
            states[0] = 0
            states[1:] = np.where(
                execution.loc[first:last, "cash_execution"].to_numpy(bool),
                0,
                1,
            )
            benchmark = frozen_candidate.build_benchmark_ledger(sample, initial_capital)
            base_ledger, base_trades = frozen_candidate.simulate_binary_path(
                sample,
                dividends,
                states,
                costs=base_costs,
                initial_capital=initial_capital,
                reinvest_paid_dividends=True,
            )
            stress_ledger, stress_trades = frozen_candidate.simulate_binary_path(
                sample,
                dividends,
                states,
                costs=stress_costs,
                initial_capital=initial_capital,
                reinvest_paid_dividends=True,
            )
            base = frozen_candidate.summarize_path(
                base_ledger, base_trades, benchmark, objective=objective
            )
            stress = frozen_candidate.summarize_path(
                stress_ledger, stress_trades, benchmark, objective=objective
            )
            row: dict[str, Any] = {
                "policy": POLICY,
                "period": period,
                "start_perturbation": perturbation,
                "first_execution_date": execution.loc[first, "date"],
                "last_execution_date": execution.loc[last, "date"],
                "execution_day_count": int(last - first + 1),
                "cash_day_count": int((states[1:] == 0).sum()),
                "cash_day_share": float((states[1:] == 0).mean()),
            }
            for prefix, summary in [("base", base), ("stress", stress)]:
                for key, value in summary.items():
                    row[f"{prefix}_{key}"] = value
            rows.append(row)
    return pd.DataFrame(rows)


def _aggregate(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for period, group in metrics.groupby("period", sort=False):
        annual = pd.to_numeric(group["stress_annualized_excess"], errors="coerce")
        rolling = pd.to_numeric(
            group["stress_rolling_242d_excess_median"], errors="coerce"
        )
        pass_flags = group["stress_both_20pct_gates"].fillna(False).astype(bool)
        rows.append(
            {
                "period": period,
                "start_count": int(len(group)),
                "stress_annualized_excess_minimum": _safe_float(annual.min()),
                "stress_annualized_excess_median": _safe_float(annual.median()),
                "stress_rolling_242d_excess_minimum": _safe_float(
                    rolling.dropna().min()
                ),
                "stress_rolling_242d_excess_median_across_starts": _safe_float(
                    rolling.dropna().median()
                ),
                "stress_start_pass_count": int(pass_flags.sum()),
                "stress_start_pass_ratio": float(pass_flags.mean()),
                "every_start_both_20pct_gates": bool(pass_flags.all()),
                "cash_day_share_minimum": _safe_float(group["cash_day_share"].min()),
                "cash_day_share_maximum": _safe_float(group["cash_day_share"].max()),
            }
        )
    return rows


def _feature_diagnostics(
    features: pd.DataFrame, config: dict[str, Any]
) -> list[dict[str, Any]]:
    start = pd.Timestamp(config["dates"]["external_evaluation_start"])
    end = pd.Timestamp(config["dates"]["external_evaluation_end"])
    selected = features.loc[features["date"].between(start, end)].copy()
    selected["year"] = selected["date"].dt.year
    rows: list[dict[str, Any]] = []
    for year, group in selected.groupby("year", sort=True):
        rows.append(
            {
                "calendar_year": int(year),
                "signal_rows": int(len(group)),
                "raw_cash_days": int(group["cash_raw"].sum()),
                "raw_cash_share": float(group["cash_raw"].mean()),
                "final_cash_days": int(group["cash_final"].sum()),
                "final_cash_share": float(group["cash_final"].mean()),
                "shadow_gate_active_share": float(group["shadow_gate_active"].mean()),
                "vote_count_mean": float(group["micro_risk_vote_count"].mean()),
            }
        )
    return rows


def _run(run_id: str) -> tuple[dict[str, Any], Path, Path]:
    config, freeze, candidate_config = _load_freeze()
    objective_contract = _verify_objective_contract(config, candidate_config)
    snapshot, frames, dividends = _load_snapshot(config, run_id)
    signal, market, input_audit = _build_signal_and_market(config, frames)
    features = frozen_candidate._rebuild_features(
        signal,
        market,
        dividends,
        candidate_config,
    )
    metrics = _evaluate(
        config=config,
        candidate_config=candidate_config,
        features=features,
        market=market,
        dividends=dividends,
    )
    aggregates = _aggregate(metrics)
    required_periods = list(config["objective"]["required_pass_periods"])
    aggregate_by_period = {row["period"]: row for row in aggregates}
    if set(aggregate_by_period) != set(required_periods):
        raise ValueError("实际评价期间与预声明通过期间不一致")
    passed = bool(
        all(
            aggregate_by_period[period]["start_count"] == 5
            and aggregate_by_period[period]["every_start_both_20pct_gates"]
            for period in required_periods
        )
    )
    status = (
        "TIME_REVERSED_EXTERNAL_VALIDATION_PASS_PROSPECTIVE_STILL_REQUIRED"
        if passed
        else "TIME_REVERSED_EXTERNAL_VALIDATION_REJECTED_NO_RESCUE"
    )

    evaluation_root = (PROJECT_ROOT / config["paths"]["evaluation_root"]).resolve()
    output_dir = (evaluation_root / run_id).resolve()
    if not output_dir.is_relative_to(evaluation_root):
        raise ValueError("run-id导致评估目录越界")
    report_root = (PROJECT_ROOT / config["paths"]["validation_report_root"]).resolve()
    report_path = report_root / f"{run_id}.json"
    evaluation_manifest_path = output_dir / "evaluation_manifest.json"
    if evaluation_manifest_path.exists() or report_path.exists():
        raise FileExistsError("该run-id已有评估证据，拒绝覆盖")
    feature_path = output_dir / "daily_features.parquet"
    metric_path = output_dir / "start_perturbation_metrics.parquet"
    _write_new_parquet(features, feature_path)
    _write_new_parquet(metrics, metric_path)
    payload = {
        "project_id": PROJECT_ID,
        "status": status,
        "passed": passed,
        "candidate_policy": POLICY,
        "evidence_class": config["protocol"]["evidence_class"],
        "interpretation": config["protocol"]["interpretation"],
        "config_sha256": _sha256(CONFIG_FILE),
        "freeze_manifest_sha256": _sha256(FREEZE_MANIFEST),
        "candidate_config_sha256": _sha256(
            PROJECT_ROOT / config["candidate_contract"]["frozen_config"]
        ),
        "candidate_validator_sha256": _sha256(
            PROJECT_ROOT / config["candidate_contract"]["frozen_validator"]
        ),
        "snapshot": {
            "run_id": run_id,
            "manifest_file": snapshot["manifest_file"],
            "manifest_sha256": snapshot["manifest_sha256"],
            "collection_status": snapshot["status"],
            "seam_crosscheck": snapshot["collection_audit"]["seam_crosscheck"],
        },
        "scope": config["scope"],
        "objective_contract_verified_equal_to_candidate": objective_contract,
        "periods": config["dates"]["periods"],
        "required_pass_periods": required_periods,
        "aggregates": aggregates,
        "calendar_year_feature_diagnostics": _feature_diagnostics(features, config),
        "input_audit": input_audit,
        "adjudication": {
            "rule": config["objective"]["final_pass_rule"],
            "result": "PASS" if passed else "REJECT",
            "parameter_rescue_allowed": False,
            "failed_period_averaging_allowed": False,
            "candidate_action": (
                "保留冻结候选并继续严格前瞻246日验证，不授权交易"
                if passed
                else "反向外部历史验证拒绝；保留失败证据，不修改当前前瞻规则，不以调参救回"
            ),
        },
        "evidence": {
            "daily_features": _relative(feature_path),
            "daily_features_sha256": _sha256(feature_path),
            "start_perturbation_metrics": _relative(metric_path),
            "start_perturbation_metrics_sha256": _sha256(metric_path),
            "evaluator": _relative(Path(__file__).resolve()),
            "evaluator_sha256": _sha256(Path(__file__).resolve()),
        },
        "boundaries": {
            "candidate_formula_changed": False,
            "thresholds_changed": False,
            "costs_changed": False,
            "benchmark_changed": False,
            "time_reversed_external_historical_validation": True,
            "pristine_prospective_holdout": False,
            "strict_prospective_validation_still_required": True,
            "research_only": True,
            "paper_or_shadow_position_mapping": "DISABLED",
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "live_trading_authorized": False,
        },
    }
    _write_new_json(payload, report_path)
    evaluation_manifest = {
        "project_id": PROJECT_ID,
        "status": status,
        "passed": passed,
        "run_id": run_id,
        "report": {
            "file": _relative(report_path),
            "sha256": _sha256(report_path),
        },
        "features": {
            "file": _relative(feature_path),
            "sha256": _sha256(feature_path),
            "rows": int(len(features)),
        },
        "metrics": {
            "file": _relative(metric_path),
            "sha256": _sha256(metric_path),
            "rows": int(len(metrics)),
        },
        "snapshot_manifest_sha256": snapshot["manifest_sha256"],
        "freeze_manifest_sha256": _sha256(FREEZE_MANIFEST),
    }
    _write_new_json(evaluation_manifest, evaluation_manifest_path)
    return payload, report_path, evaluation_manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="评估510300双影子规则反向历史快照")
    parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", arguments.run_id) is None:
        raise ValueError("run-id只能包含字母、数字、点、下划线和连字符，且长度不超过80")
    if ".." in arguments.run_id:
        raise ValueError("run-id不得包含连续点号")
    payload, report_path, evaluation_manifest_path = _run(arguments.run_id)
    summary = {
        "project_id": PROJECT_ID,
        "status": payload["status"],
        "passed": payload["passed"],
        "aggregates": payload["aggregates"],
        "report": _relative(report_path),
        "report_sha256": _sha256(report_path),
        "evaluation_manifest": _relative(evaluation_manifest_path),
        "evaluation_manifest_sha256": _sha256(evaluation_manifest_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
