"""验证冻结清单后首次运行X1历史BAD20预测评价。"""

from __future__ import annotations

import json
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

from research.x1_common_tail_prediction import (  # noqa: E402
    X1Rules,
    build_r6_information_features,
    build_return_tail_outcomes,
    evaluate_x1_prediction,
    walk_forward_x1_probabilities,
)
from scripts.freeze_x1_common_tail_prediction_v1 import (  # noqa: E402
    CONTRACT_FILE,
    FROZEN_FILES,
    MANIFEST_FILE,
    sha256,
    tree_sha256,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is pd.NaT:
        return None
    return value


def _verify_manifest(contract: dict) -> dict:
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError("X1实现尚未冻结；禁止读取BAD20历史结果")
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    if manifest.get("state") != "FROZEN_BEFORE_HISTORICAL_OUTCOME_READ":
        raise RuntimeError("X1清单状态不是历史结果读取前冻结")
    changed: list[str] = []
    for relative, expected in manifest["frozen_files"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            changed.append(relative)
    directory_relative = contract["inputs"]["component_history_cache"]
    for relative, expected in manifest["input_files"].items():
        actual = (
            tree_sha256(ROOT / relative)
            if relative == directory_relative
            else sha256(ROOT / relative) if (ROOT / relative).exists() else None
        )
        if actual != expected:
            changed.append(relative)
    if changed:
        raise RuntimeError(f"X1冻结指纹变化：{sorted(set(changed))}")
    if set(manifest["frozen_files"]) != set(FROZEN_FILES):
        raise RuntimeError("X1冻结文件集合与运行器声明不一致")
    return manifest


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


def _render_markdown(report: dict[str, Any]) -> str:
    evaluation = report["evaluation"]
    bad = evaluation["bad20"]
    probability = evaluation["probability"]
    incremental = evaluation["incremental_vs_r6"]
    gate_lines = [
        f"- {'PASS' if passed else 'FAIL'} `{name}`"
        for name, passed in evaluation["gates"].items()
    ]
    return "\n".join(
        [
            "# 510300 X1共同尾部未来风险预测 V1 历史结果",
            "",
            f"- 总状态：`{report['status']}`",
            f"- 数据截止：`{report['data_cutoff']}`",
            "- 历史标签：`HISTORICALLY_CONTAMINATED`",
            "- 预测资格：`NOT_AUTHORIZED`",
            "- 组合、仓位、订单、券商：`DISABLED`",
            "",
            "## 样本与原始X1",
            "",
            f"- 有效信号日：{report['row_counts']['ready_signal_days']}",
            f"- 成熟主评价信号：{evaluation['primary_oos_signal_days']}",
            f"- 走步概率预测：{evaluation['probability_oos_predictions']}",
            f"- BAD20无条件事件：{bad['event_count']} / {bad['observations']}，发生率 {bad['baseline_rate']}",
            f"- 最高风险组事件：{bad['selected_event_count']} / {bad['selected_observations']}，发生率 {bad['selected_rate']}",
            f"- BAD20 Lift：{bad['lift']}",
            f"- Bootstrap 95%区间：{evaluation['bootstrap_bad20_lift']['interval_95pct']}",
            f"- 一日延迟额外Lift保留率：{evaluation['delay_excess_lift_retention']}",
            f"- 前半段Lift：{evaluation['first_half_bad20']['lift']}",
            f"- 后半段Lift：{evaluation['second_half_bad20']['lift']}",
            "",
            "## 概率与R6增量",
            "",
            f"- X1 Brier：{probability['x1_brier']}；无条件：{probability['unconditional_brier']}；Skill：{probability['brier_skill']}",
            f"- X1 Log Loss：{probability['x1_log_loss']}；无条件：{probability['unconditional_log_loss']}",
            f"- R6 Brier：{incremental['r6_brier']}；R6+X1：{incremental['r6_plus_x1_brier']}",
            f"- R6 Log Loss：{incremental['r6_log_loss']}；R6+X1：{incremental['r6_plus_x1_log_loss']}",
            f"- R6最高十分位BAD20率：{incremental['r6_top_decile_bad20']['selected_rate']}；R6+X1：{incremental['r6_plus_x1_top_decile_bad20']['selected_rate']}",
            f"- R6单位避险贡献：{incremental['r6_unit_avoidance_contribution']}；R6+X1：{incremental['r6_plus_x1_unit_avoidance_contribution']}",
            "",
            "## 冻结门槛",
            "",
            *gate_lines,
            "",
            "## 边界",
            "",
            "结果无论通过或失败均不映射仓位。失败不得修改5日、504日、5%尾部、90%风险组或20日目标补救。",
        ]
    ) + "\n"


def main() -> int:
    contract = yaml.safe_load(CONTRACT_FILE.read_text(encoding="utf-8"))
    manifest = _verify_manifest(contract)
    inputs = contract["inputs"]
    signal_status = json.loads(
        (ROOT / inputs["x1_signal_status"]).read_text(encoding="utf-8")
    )
    if signal_status.get("status") != "READY_FOR_OUTCOME_FREEZE":
        raise RuntimeError("X1信号数据门不再有效")
    rules = X1Rules.from_contract(contract)
    signal = pd.read_parquet(ROOT / inputs["x1_signal_feature"])
    etf_daily = pd.read_parquet(ROOT / inputs["etf_daily"])
    dividends = pd.read_csv(ROOT / inputs["etf_dividends"])
    valuation = pd.read_parquet(ROOT / inputs["index_valuation"])
    outcomes = build_return_tail_outcomes(
        etf_daily,
        dividends,
        signal.loc[signal["signal_output"].eq("SIGNAL_READY"), "date"],
        rules,
    )
    r6_features = build_r6_information_features(
        etf_daily, dividends, valuation, rules
    )
    analysis, forecasts = walk_forward_x1_probabilities(
        signal, outcomes, r6_features, rules
    )
    if forecasts.empty:
        raise RuntimeError("没有满足冻结训练长度的X1走步概率预测")
    if forecasts["research_output"].ne("PREDICTION_AUDIT_ONLY").any():
        raise AssertionError("X1历史运行非法输出预测资格")
    evaluation = evaluate_x1_prediction(analysis, forecasts, rules)
    if evaluation["safety"]["forecast_eligible_emitted"]:
        raise AssertionError("X1历史运行非法输出FORECAST_ELIGIBLE")

    outputs = contract["outputs"]
    _atomic_parquet(outcomes, ROOT / outputs["outcomes"])
    _atomic_parquet(forecasts, ROOT / outputs["forecasts"])
    report = _json_safe(
        {
            "project_id": contract["protocol"]["project_id"],
            "version": contract["protocol"]["version"],
            "status": evaluation["status"],
            "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "data_cutoff": str(rules.data_cutoff.date()),
            "manifest_sha256": sha256(MANIFEST_FILE),
            "frozen_at": manifest["frozen_at"],
            "parent_integrity": manifest["parent_integrity"],
            "row_counts": {
                "signal_days": int(len(signal)),
                "ready_signal_days": int(signal["signal_output"].eq("SIGNAL_READY").sum()),
                "no_view_signal_days": int(signal["signal_output"].eq("NO_VIEW").sum()),
                "warmup_signal_days": int(signal["signal_output"].eq("WARMUP").sum()),
                "mature_outcome_rows": int(len(outcomes)),
                "probability_forecasts": int(len(forecasts)),
            },
            "evaluation": evaluation,
            "governance": contract["governance"],
            "safety": {
                "forecast_eligible": "NOT_AUTHORIZED",
                "combination_model": "DISABLED",
                "position_policy": "DISABLED",
                "position_mapping": "DISABLED",
                "order_generation": "DISABLED",
                "broker_connection": "DISABLED",
            },
        }
    )
    json_path = ROOT / outputs["result_json"]
    markdown_path = ROOT / outputs["result_markdown"]
    _atomic_text(json.dumps(report, ensure_ascii=False, indent=2), json_path)
    _atomic_text(_render_markdown(report), markdown_path)
    print(
        json.dumps(
            {
                "状态": report["status"],
                "成熟主信号": evaluation["primary_oos_signal_days"],
                "概率预测": evaluation["probability_oos_predictions"],
                "BAD20事件": evaluation["bad20"]["event_count"],
                "BAD20_Lift": evaluation["bad20"]["lift"],
                "Brier_Skill": evaluation["probability"]["brier_skill"],
                "预测资格": "NOT_AUTHORIZED",
                "结果": str(json_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
