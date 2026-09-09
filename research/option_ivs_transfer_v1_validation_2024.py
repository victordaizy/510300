"""冻结IVS公式的2024时间留出验证；不读取2025后数据。"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.option_ivs_transfer_v1 import (  # noqa: E402
    build_daily_ivs,
    build_monthly_rows,
    evaluate_strategy,
    load_config as load_formula_config,
)
from research.option_trade_envelope_v1 import assert_input_hashes  # noqa: E402
from research.option_technical_rule_discovery_v1 import load_sources  # noqa: E402
from scripts.freeze_510300_option_ivs_transfer_v1 import (  # noqa: E402
    verify_protocol as verify_formula_protocol,
)
from scripts.freeze_510300_option_technical_rule_discovery_v1 import (  # noqa: E402
    verify_protocol as verify_data_loader_protocol,
)
from scripts.freeze_510300_option_trade_envelope_v1 import (  # noqa: E402
    verify_protocol as verify_vehicle_protocol,
)


CONFIG = ROOT / "config" / "510300_option_ivs_transfer_v1_validation_2024.yaml"
TIMEZONE = ZoneInfo("Asia/Shanghai")


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def validation_predictions(
    monthly: pd.DataFrame,
    formula_config: dict,
    validation_config: dict,
) -> pd.DataFrame:
    entry_start = pd.Timestamp(validation_config["validation"]["entry_start"])
    exit_end = pd.Timestamp(validation_config["validation"]["exit_end"])
    minimum = int(formula_config["model"]["minimum_mature_training_months"])
    evaluation = monthly.loc[
        monthly["entry_date"].ge(entry_start) & monthly["exit_date"].le(exit_end)
    ].copy()
    rows: list[dict[str, Any]] = []
    for current in evaluation.to_dict("records"):
        signal_date = pd.Timestamp(current["signal_date"])
        training = monthly.loc[
            monthly["exit_date"].le(signal_date)
            & monthly["ivs"].notna()
            & monthly["forward_return"].notna()
        ].copy()
        base = {**current, "training_months": len(training)}
        if not math.isfinite(float(current["ivs"])):
            rows.append({**base, "status": "NO_VIEW_STALE_IVS", "forecast": np.nan})
            continue
        if len(training) < minimum:
            rows.append({**base, "status": "NO_MODEL_MINIMUM_TRAINING", "forecast": np.nan})
            continue
        design = np.column_stack([np.ones(len(training)), training["ivs"].to_numpy(float)])
        alpha, beta_raw = np.linalg.lstsq(
            design, training["forward_return"].to_numpy(float), rcond=None
        )[0]
        beta = min(float(beta_raw), 0.0)
        if beta == 0.0:
            alpha = float(training["forward_return"].mean())
        forecast = float(alpha + beta * float(current["ivs"]))
        rows.append(
            {
                **base,
                "status": "MODEL_READY",
                "alpha": float(alpha),
                "beta_raw": float(beta_raw),
                "beta": beta,
                "forecast": forecast,
                "side": "C" if forecast >= 0 else "P",
                "latest_training_exit": training["exit_date"].max(),
            }
        )
    result = pd.DataFrame(rows)
    ready = result.loc[result["status"].eq("MODEL_READY")]
    if len(ready) and ready["latest_training_exit"].gt(ready["signal_date"]).any():
        raise RuntimeError("2024验证训练使用了信号日后才成熟的标签")
    if len(result) and (
        result["entry_date"].lt(entry_start).any() or result["exit_date"].gt(exit_end).any()
    ):
        raise RuntimeError("2024验证结果越界")
    return result


def write_outputs(
    metrics: dict,
    ledger: list[dict[str, Any]],
    predictions: pd.DataFrame,
    config: dict,
    manifest: dict,
) -> None:
    threshold = float(config["validation"]["annualized_excess_minimum"])
    minimum_gate = bool(metrics["all_opening_trades_at_least_5000"])
    validation_pass = metrics["annualized_excess"] >= threshold and minimum_gate
    metrics = {key: value for key, value in metrics.items() if key != "gates"}
    payload = {
        "project_id": config["protocol"]["project_id"],
        "status": "VALIDATION_2024_PASS_HOLDOUT_NOT_AUTHORIZED" if validation_pass else "VALIDATION_2024_REJECTED_FROZEN",
        "generated_at": datetime.now(TIMEZONE).isoformat(),
        "manifest_frozen_at": manifest["frozen_at"],
        "evidence_label": "STRICT_2024_TIME_VALIDATION_FINAL_HOLDOUT_UNREAD",
        "factor_count": 1,
        **metrics,
        "validation_gates": {
            "annualized_excess_at_least_20pct": metrics["annualized_excess"] >= threshold,
            "minimum_opening_trade_gate": minimum_gate,
            "validation_pass": validation_pass,
        },
        "ledger": ledger,
        "safety": config["governance"],
    }
    json_path = ROOT / config["paths"]["result_json"]
    markdown_path = ROOT / config["paths"]["result_markdown"]
    prediction_path = ROOT / config["paths"]["predictions"]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(json_path)
    predictions.to_parquet(prediction_path, index=False)
    markdown_path.write_text(
        "\n".join(
            [
                "# 510300期权IVS迁移V1 2024严格验证",
                "",
                f"- 状态：`{payload['status']}`",
                f"- 证据：`{payload['evidence_label']}`",
                f"- 区间：{payload['period_start']} 至 {payload['period_end']}",
                f"- 模型可用月：{payload['model_ready_month_count']} / {payload['prediction_month_count']}",
                f"- 方向准确率：{payload['direction_accuracy']:.2%}",
                f"- 策略年化：{payload['strategy_cagr']:.2%}",
                f"- H00300年化：{payload['benchmark_cagr']:.2%}",
                f"- 年化超额：{payload['annualized_excess']:.2%}",
                f"- 期末权益：{payload['final_equity_cny']:.2f}元",
                "",
                f"- 年化超额至少20%：{payload['validation_gates']['annualized_excess_at_least_20pct']}",
                f"- 开仓至少5000元：{payload['validation_gates']['minimum_opening_trade_gate']}",
                f"- 2024验证通过：{payload['validation_gates']['validation_pass']}",
                "",
                "公式、斜率约束、阈值、合约筛选和成交包络均继承开发期冻结版本。",
                "2025-01-01后的最终留出未授权读取；验证失败不得修改V1后再看最终留出。",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    from scripts.freeze_510300_option_ivs_transfer_v1_validation_2024 import verify_protocol

    config, manifest = verify_protocol()
    formula_config, _ = verify_formula_protocol()
    verify_data_loader_protocol()
    vehicle_config, _ = verify_vehicle_protocol()
    assert_input_hashes(vehicle_config)
    cutoff = pd.Timestamp(config["validation"]["exit_end"])
    panel, benchmark = load_sources(cutoff, vehicle_config)
    daily_ivs = build_daily_ivs(panel, formula_config)
    monthly = build_monthly_rows(benchmark, daily_ivs, formula_config)
    predictions = validation_predictions(monthly, formula_config, config)
    metrics, ledger = evaluate_strategy(
        predictions, panel, benchmark, formula_config
    )
    write_outputs(metrics, ledger, predictions, config, manifest)
    concise = {key: value for key, value in metrics.items() if key != "gates"}
    print(json.dumps(concise, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
