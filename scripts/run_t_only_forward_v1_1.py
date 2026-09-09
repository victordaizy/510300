"""运行交易日历约束的 T_ONLY_FORWARD_V1.1 并生成每日操作卡。"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.t_only_forward_v1_1 import (
    apply_causal_weekly_states,
    build_daily_guide,
    guide_markdown,
    latest_diagnostic,
    load_calendar,
    load_config as load_v1_1_config,
)
from research.t_only_robustness_forward_v1 import (
    load_stress_config,
    profitable_cycle_concentration,
    simulate_t_only,
)
from research.weekly_daily_technical_v1 import (
    build_features,
    load_config as load_base_config,
    performance_metrics,
    simulate_static_exposure,
)
from scripts.run_t_only_forward_v1 import execution_compliance, load_dividends


TIMEZONE = ZoneInfo("Asia/Shanghai")
MANIFEST = ROOT / "config" / "t_only_forward_v1_1_freeze_manifest.json"
DATA_GATE = ROOT / "reports" / "data_quality" / "t_only_forward_v1_data_gate.json"
REPORT_JSON = ROOT / "reports" / "forward" / "t_only_forward_v1_1_status.json"
REPORT_MARKDOWN = ROOT / "reports" / "forward" / "t_only_forward_v1_1_status.md"
GUIDE_JSON = ROOT / "reports" / "forward" / "t_only_forward_v1_1_daily_guide.json"
GUIDE_MARKDOWN = ROOT / "reports" / "forward" / "t_only_forward_v1_1_daily_guide.md"
OUTPUT_DIRECTORY = ROOT / "paper" / "t_only_forward_v1_1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def verify_manifest() -> dict[str, Any]:
    if not MANIFEST.exists():
        raise RuntimeError("NO_VIEW：T_ONLY_FORWARD_V1.1冻结清单缺失")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for section in (
        "protocol_files",
        "implementation_files",
        "calendar_input_files",
        "parent_freeze_files",
    ):
        for relative_path, expected in manifest[section].items():
            path = ROOT / relative_path
            if not path.exists() or sha256(path) != expected:
                raise RuntimeError(f"NO_VIEW：V1.1冻结文件指纹变化：{relative_path}")
    return manifest


def status_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# T_ONLY_FORWARD_V1.1 前瞻影子状态",
        "",
        f"- 状态：`{report['status']}`",
        f"- 行情截止：{report['as_of_market_date']}",
        f"- 前瞻信号起点：{report['forward_signal_start']}",
        f"- 已收集新交易日：{report['new_trading_days']}",
        f"- 已闭合周期：{report['closed_cycles']}",
        f"- 当前视图：`{report['current_view']}`",
        f"- 周线因果闸门：`{report['weekly_causality_audit']['status']}`",
        "- V1.1只修正每日增量运行的未完成周识别，不修改策略参数。",
        "- 安全边界：影子账本，不读取真实持仓，不生成订单，不连接券商。",
        "",
        "## 当前影子信号",
        "",
    ]
    signal = report.get("current_shadow_signal")
    if signal is None:
        lines.append("无。")
    else:
        lines.extend(
            [
                f"- 信号日：{signal['signal_date']}",
                f"- 原因：`{signal['reason']}`",
                f"- 模型目标仓位：{float(signal['target_exposure']):.1%}",
            ]
        )
    lines.extend(
        [
            "",
            "## 判定纪律",
            "",
            "至少收集252个新交易日且闭合3个周期前，不评价策略通过或失败，也不据此调整参数。",
            "",
        ]
    )
    return "\n".join(lines)


def base_report(
    *,
    v1_1_config: dict[str, Any],
    data_gate: dict[str, Any],
    causality_audit: dict[str, Any],
    diagnostic: dict[str, Any],
) -> dict[str, Any]:
    return {
        "project_id": v1_1_config["protocol"]["project_id"],
        "supersedes_forward_runtime": v1_1_config["protocol"][
            "supersedes_forward_runtime"
        ],
        "status": "COLLECTING_FORWARD_NOT_STARTED",
        "current_view": "NO_VIEW_NO_FORWARD_TRADING_DAY",
        "as_of_market_date": data_gate["actual_last_date"],
        "forward_signal_start": v1_1_config["protocol"]["forward_signal_start"],
        "new_trading_days": 0,
        "closed_cycles": 0,
        "maturity": {
            "trading_days_required": 252,
            "closed_cycles_required": 3,
            "mature": False,
        },
        "current_shadow_signal": None,
        "metrics": None,
        "gates": "NOT_EVALUATED_BEFORE_MATURITY",
        "weekly_causality_audit": causality_audit,
        "latest_indicator_diagnostic": diagnostic,
        "data_gate": data_gate,
        "safety": v1_1_config["governance"],
    }


def main() -> int:
    verify_manifest()
    v1_1_config = load_v1_1_config()
    parent_config = load_stress_config()
    base_config = load_base_config()
    forward = parent_config["forward_data"]
    market_path = ROOT / forward["refreshed_market_file"]
    metadata_path = ROOT / forward["metadata_file"]
    if not DATA_GATE.exists() or not market_path.exists() or not metadata_path.exists():
        raise RuntimeError("NO_VIEW_DATA_NOT_REFRESHED")
    data_gate = json.loads(DATA_GATE.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if data_gate.get("status") != "PASS" or metadata.get("status") != "PASS":
        raise RuntimeError(f"NO_VIEW_DATA_GATE_{data_gate.get('status', 'UNKNOWN')}")
    if sha256(market_path) != metadata.get("sha256"):
        raise RuntimeError("NO_VIEW_REFRESHED_MARKET_HASH_MISMATCH")
    latest = pd.Timestamp(metadata["actual_last_date"]).normalize()
    coverage = pd.Timestamp(metadata["dividend_coverage_effective_through"]).normalize()
    if bool(forward["require_dividend_coverage_through_market_date"]) and coverage < latest:
        raise RuntimeError("NO_VIEW_DIVIDEND_COVERAGE_STALE")
    age = (pd.Timestamp(datetime.now(TIMEZONE).date()) - latest).days
    if age > int(forward["maximum_calendar_age_days"]):
        raise RuntimeError("NO_VIEW_MARKET_DATA_STALE")

    market = pd.read_parquet(market_path)
    market["date"] = pd.to_datetime(market["date"], errors="raise").dt.normalize()
    dividends = load_dividends(ROOT / forward["dividend_file"])
    features, weekly = build_features(market, dividends, base_config)
    calendar = load_calendar(v1_1_config)
    features, causality_audit = apply_causal_weekly_states(
        features,
        weekly,
        calendar,
        signal_start=calendar["trade_date"].min(),
    )
    diagnostic = latest_diagnostic(features)
    signal_start = pd.Timestamp(v1_1_config["protocol"]["forward_signal_start"])
    forward_features = features.loc[features["date"] >= signal_start].reset_index(
        drop=True
    )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)

    if forward_features.empty:
        report = safe(
            base_report(
                v1_1_config=v1_1_config,
                data_gate=data_gate,
                causality_audit=causality_audit,
                diagnostic=diagnostic,
            )
        )
        shadow_state = {
            "mode": "CASH",
            "tier": 0,
            "shares": 0,
            "equity_cny": float(
                base_config["price_and_execution"]["initial_capital_cny"]
            ),
        }
    else:
        result = simulate_t_only(
            forward_features,
            dividends,
            base_config,
            scenario_id="T_ONLY_FORWARD_V1_1",
            allow_terminal_pending_signal=True,
        )
        double_cost = simulate_t_only(
            forward_features,
            dividends,
            base_config,
            scenario_id="T_ONLY_FORWARD_V1_1_DOUBLE_COST",
            commission_multiplier=2.0,
            slippage_bps=10.0,
            allow_terminal_pending_signal=True,
        )
        cash_rate = float(base_config["price_and_execution"]["cash_annual_rate"])
        strategy_metrics = performance_metrics(result["ledger"], cash_rate)
        cash_ledger = simulate_static_exposure(
            forward_features, dividends, base_config, 0.0
        )
        cash_metrics = performance_metrics(cash_ledger, cash_rate)
        matched_ledger = simulate_static_exposure(
            forward_features,
            dividends,
            base_config,
            float(strategy_metrics["average_exposure"]),
        )
        matched_metrics = performance_metrics(matched_ledger, cash_rate)
        double_cost_metrics = performance_metrics(double_cost["ledger"], cash_rate)
        compliance = execution_compliance(result["executions"], forward_features["date"])
        concentration = profitable_cycle_concentration(result["cycles"])
        days = int(len(forward_features))
        closed_cycles = int(len(result["cycles"]))
        maturity_config = parent_config["forward_maturity"]
        mature = bool(
            days >= int(maturity_config["minimum_new_trading_days"])
            and closed_cycles >= int(maturity_config["minimum_closed_cycles"])
        )
        gate_config = parent_config["acceptance_gates"]
        gates = {
            "net_return_above_cash": strategy_metrics["total_return"]
            > cash_metrics["total_return"],
            "net_return_above_matched_exposure": strategy_metrics["total_return"]
            > matched_metrics["total_return"],
            "double_cost_total_return_positive": double_cost_metrics["total_return"]
            > 0.0,
            "maximum_drawdown_within_limit": strategy_metrics["maximum_drawdown"]
            >= float(gate_config["maximum_drawdown_floor"]),
            "cycle_concentration_within_limit": concentration is not None
            and concentration
            <= float(gate_config["maximum_single_profitable_cycle_share"]),
            "execution_compliance": compliance is not None
            and compliance >= float(gate_config["execution_compliance_required"]),
        }
        status = (
            "PASS_FORWARD_GATES"
            if mature and all(gates.values())
            else "FAIL_FORWARD_GATES"
            if mature
            else "COLLECTING"
        )
        report = safe(
            {
                "project_id": v1_1_config["protocol"]["project_id"],
                "supersedes_forward_runtime": v1_1_config["protocol"][
                    "supersedes_forward_runtime"
                ],
                "status": status,
                "current_view": (
                    "MATURE_SHADOW_EVIDENCE_ONLY"
                    if mature
                    else "NO_VIEW_UNTIL_FORWARD_MATURITY"
                ),
                "as_of_market_date": str(latest.date()),
                "forward_signal_start": str(signal_start.date()),
                "new_trading_days": days,
                "closed_cycles": closed_cycles,
                "maturity": {
                    "trading_days_required": maturity_config[
                        "minimum_new_trading_days"
                    ],
                    "closed_cycles_required": maturity_config[
                        "minimum_closed_cycles"
                    ],
                    "mature": mature,
                },
                "current_shadow_signal": result["state"]["pending_signal"],
                "metrics": {
                    "strategy": strategy_metrics,
                    "cash": cash_metrics,
                    "matched_exposure": matched_metrics,
                    "double_total_cost": double_cost_metrics,
                    "profitable_cycle_concentration": concentration,
                    "execution_compliance": compliance,
                },
                "gates": gates if mature else "NOT_EVALUATED_BEFORE_MATURITY",
                "weekly_causality_audit": causality_audit,
                "latest_indicator_diagnostic": diagnostic,
                "data_gate": data_gate,
                "safety": v1_1_config["governance"],
            }
        )
        last = result["ledger"].iloc[-1]
        shadow_state = {
            "mode": str(last["mode"]),
            "tier": int(last["tier"]),
            "shares": int(last["shares"]),
            "equity_cny": float(last["equity"]),
        }
        result["ledger"].to_parquet(
            OUTPUT_DIRECTORY / "daily_ledger.parquet", index=False
        )
        result["executions"].to_csv(
            OUTPUT_DIRECTORY / "execution_events.csv", index=False
        )
        result["cycles"].to_csv(
            OUTPUT_DIRECTORY / "closed_cycles.csv", index=False
        )
        result["skipped_orders"].to_csv(
            OUTPUT_DIRECTORY / "skipped_orders.csv", index=False
        )
        forward_features.to_parquet(
            OUTPUT_DIRECTORY / "causal_feature_audit.parquet", index=False
        )

    guide = safe(
        build_daily_guide(
            status_report=report,
            diagnostic=diagnostic,
            config=v1_1_config,
            shadow_state=shadow_state,
        )
    )
    atomic_text(REPORT_JSON, json.dumps(report, ensure_ascii=False, indent=2))
    atomic_text(REPORT_MARKDOWN, status_markdown(report))
    atomic_text(GUIDE_JSON, json.dumps(guide, ensure_ascii=False, indent=2))
    atomic_text(GUIDE_MARKDOWN, guide_markdown(guide))
    print(
        json.dumps(
            {
                "status": report["status"],
                "new_trading_days": report["new_trading_days"],
                "daily_decision": guide["decision"],
                "weekly_state": diagnostic["weekly_state_at_close"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
