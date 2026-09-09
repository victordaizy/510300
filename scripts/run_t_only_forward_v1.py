"""生成 T_ONLY_FORWARD_V1 前瞻影子账本和成熟度状态。"""

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

from research.t_only_robustness_forward_v1 import (
    load_stress_config,
    profitable_cycle_concentration,
    simulate_t_only,
)
from research.weekly_daily_technical_v1 import (
    build_features,
    load_config,
    performance_metrics,
    simulate_static_exposure,
)


TIMEZONE = ZoneInfo("Asia/Shanghai")
MANIFEST = ROOT / "config" / "t_only_forward_v1_freeze_manifest.json"
DATA_GATE = ROOT / "reports" / "data_quality" / "t_only_forward_v1_data_gate.json"
REPORT_JSON = ROOT / "reports" / "forward" / "t_only_forward_v1_status.json"
REPORT_MARKDOWN = ROOT / "reports" / "forward" / "t_only_forward_v1_status.md"
OUTPUT_DIRECTORY = ROOT / "paper" / "t_only_forward_v1"


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


def verify_manifest() -> dict[str, Any]:
    if not MANIFEST.exists():
        raise RuntimeError("NO_VIEW：T_ONLY 前瞻冻结清单缺失")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for section in (
        "protocol_files",
        "implementation_files",
        "parent_files",
        "parent_input_files",
    ):
        for relative_path, expected in manifest[section].items():
            path = ROOT / relative_path
            if not path.exists() or sha256(path) != expected:
                raise RuntimeError(f"NO_VIEW：冻结文件指纹变化：{relative_path}")
    return manifest


def load_dividends(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ("record_date", "ex_date", "payment_date"):
        frame[column] = pd.to_datetime(frame[column], errors="raise").dt.normalize()
    return frame.sort_values("ex_date").reset_index(drop=True)


def execution_compliance(
    executions: pd.DataFrame, forward_dates: pd.Series
) -> float | None:
    if executions.empty:
        return None
    date_index = {
        pd.Timestamp(date).normalize(): index
        for index, date in enumerate(pd.to_datetime(forward_dates))
    }
    checks = []
    for row in executions.itertuples(index=False):
        signal = pd.Timestamp(row.signal_date).normalize()
        execution = pd.Timestamp(row.execution_date).normalize()
        checks.append(
            signal in date_index
            and execution in date_index
            and date_index[execution] == date_index[signal] + 1
        )
    return float(sum(checks) / len(checks))


def empty_report(config: dict[str, Any], data_gate: dict[str, Any]) -> dict[str, Any]:
    protocol = config["protocol"]
    return {
        "project_id": protocol["project_id"],
        "status": "COLLECTING_FORWARD_NOT_STARTED",
        "current_view": "NO_VIEW_NO_FORWARD_TRADING_DAY",
        "as_of_market_date": data_gate["actual_last_date"],
        "forward_signal_start": protocol["forward_signal_start"],
        "new_trading_days": 0,
        "closed_cycles": 0,
        "maturity": {
            "trading_days_required": config["forward_maturity"][
                "minimum_new_trading_days"
            ],
            "closed_cycles_required": config["forward_maturity"][
                "minimum_closed_cycles"
            ],
            "mature": False,
        },
        "current_shadow_signal": None,
        "metrics": None,
        "gates": "NOT_EVALUATED_BEFORE_MATURITY",
        "data_gate": data_gate,
        "safety": config["governance"],
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# T_ONLY_FORWARD_V1 前瞻影子状态",
        "",
        f"- 状态：`{report['status']}`",
        f"- 行情截止：{report['as_of_market_date']}",
        f"- 前瞻信号起点：{report['forward_signal_start']}",
        f"- 已收集新交易日：{report['new_trading_days']}",
        f"- 已闭合周期：{report['closed_cycles']}",
        f"- 当前视图：`{report['current_view']}`",
        "- 安全边界：影子账本，不读取真实持仓，不生成订单，不连接券商。",
        "",
    ]
    signal = report.get("current_shadow_signal")
    if signal is None:
        lines.extend(["## 当前影子信号", "", "无。", ""])
    else:
        lines.extend(
            [
                "## 当前影子信号",
                "",
                f"- 信号日：{signal['signal_date']}",
                f"- 原因：{signal['reason']}",
                f"- 模型目标仓位：{float(signal['target_exposure']):.1%}",
                "- 该记录仅用于下一交易日影子核验，不是订单，也不映射真实仓位。",
                "",
            ]
        )
    if report.get("metrics") is not None:
        metrics = report["metrics"]
        lines.extend(
            [
                "## 尚未成熟的描述性指标",
                "",
                f"- 影子总收益：{float(metrics['strategy']['total_return']):.2%}",
                f"- 最大回撤：{float(metrics['strategy']['maximum_drawdown']):.2%}",
                f"- 现金总收益：{float(metrics['cash']['total_return']):.2%}",
                f"- 同平均仓位基准总收益：{float(metrics['matched_exposure']['total_return']):.2%}",
                "",
            ]
        )
    lines.extend(
        [
            "## 判定纪律",
            "",
            "至少收集 252 个新交易日且闭合 3 个周期前，不评价策略通过或失败，也不据此调整参数。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    verify_manifest()
    config = load_stress_config()
    base_config = load_config()
    forward = config["forward_data"]
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
    features, _ = build_features(market, dividends, base_config)
    signal_start = pd.Timestamp(config["protocol"]["forward_signal_start"])
    forward_features = features.loc[features["date"] >= signal_start].reset_index(
        drop=True
    )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)

    if forward_features.empty:
        report = safe(empty_report(config, data_gate))
        REPORT_JSON.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        REPORT_MARKDOWN.write_text(markdown(report), encoding="utf-8")
        print(json.dumps({"status": report["status"], "new_trading_days": 0}, ensure_ascii=False))
        return 0

    result = simulate_t_only(
        forward_features,
        dividends,
        base_config,
        scenario_id="T_ONLY_FORWARD_V1",
        allow_terminal_pending_signal=True,
    )
    double_cost = simulate_t_only(
        forward_features,
        dividends,
        base_config,
        scenario_id="T_ONLY_FORWARD_V1_DOUBLE_COST",
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
    maturity_config = config["forward_maturity"]
    mature = bool(
        days >= int(maturity_config["minimum_new_trading_days"])
        and closed_cycles >= int(maturity_config["minimum_closed_cycles"])
    )
    gate_config = config["acceptance_gates"]
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
    current_view = (
        "MATURE_SHADOW_EVIDENCE_ONLY"
        if mature
        else "NO_VIEW_UNTIL_FORWARD_MATURITY"
    )
    report = safe(
        {
            "project_id": config["protocol"]["project_id"],
            "status": status,
            "current_view": current_view,
            "as_of_market_date": str(latest.date()),
            "forward_signal_start": str(signal_start.date()),
            "new_trading_days": days,
            "closed_cycles": closed_cycles,
            "maturity": {
                "trading_days_required": maturity_config["minimum_new_trading_days"],
                "closed_cycles_required": maturity_config["minimum_closed_cycles"],
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
            "data_gate": data_gate,
            "safety": config["governance"],
        }
    )
    result["ledger"].to_parquet(OUTPUT_DIRECTORY / "daily_ledger.parquet", index=False)
    result["executions"].to_csv(OUTPUT_DIRECTORY / "execution_events.csv", index=False)
    result["cycles"].to_csv(OUTPUT_DIRECTORY / "closed_cycles.csv", index=False)
    result["skipped_orders"].to_csv(
        OUTPUT_DIRECTORY / "skipped_orders.csv", index=False
    )
    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REPORT_MARKDOWN.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"status": status, "new_trading_days": days}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
