"""验证冻结清单后运行板块未来贡献预测 V1 历史伪样本外评价。"""

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

from research.sector_future_return_prediction import (  # noqa: E402
    PredictionRules,
    build_etf_forward_returns,
    build_sector_forward_targets,
    evaluate_sector_forecasts,
    walk_forward_sector_forecasts,
)
from scripts.freeze_sector_forward_return_prediction_v1 import (  # noqa: E402
    CONTRACT_FILE,
    FROZEN_FILES,
    MANIFEST_FILE,
    sha256,
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
        raise FileNotFoundError("板块预测 V1 尚未冻结；禁止读取历史未来收益")
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    if manifest.get("state") != "FROZEN_BEFORE_HISTORICAL_EVALUATION":
        raise RuntimeError("板块预测清单状态不是历史评价前冻结")
    changed: list[str] = []
    for relative, expected in manifest["frozen_files"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            changed.append(relative)
    for relative, expected in manifest["input_files"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            changed.append(relative)
    if changed:
        raise RuntimeError(f"板块预测冻结指纹变化：{sorted(set(changed))}")
    if set(manifest["frozen_files"]) != set(FROZEN_FILES):
        raise RuntimeError("冻结文件集合与运行器声明不一致")
    if contract["governance"]["historical_run_may_emit_forecast_eligible"]:
        raise RuntimeError("历史运行不得输出 FORECAST_ELIGIBLE")
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
    bootstrap_interval = evaluation["bootstrap_spearman"]["interval_95pct"]
    gate_lines = [
        f"- {'PASS' if passed else 'FAIL'} `{name}`"
        for name, passed in evaluation["gates"].items()
    ]
    lines = [
        "# 510300 板块未来贡献预测 V1 历史结果",
        "",
        f"- 总状态：`{report['status']}`",
        f"- 数据截止：`{report['data_cutoff']}`",
        "- 历史标签：`HISTORICALLY_CONTAMINATED`",
        "- 预测资格：`NOT_AUTHORIZED`",
        "- 仓位、订单、券商：`DISABLED`",
        "",
        "## 固定预测问题",
        "",
        "信号日官方权重与中信一级行业固定，目标从下一交易日开盘至第60个交易日收盘。缺失端点不重加权。S1 只使用板块权重、盈利收益率、账面收益率、ROE、TTM盈利和收入同比。",
        "",
        "## 样本",
        "",
        f"- 成熟目标快照：{report['row_counts']['mature_target_snapshots']}",
        f"- NO_VIEW 目标快照：{report['row_counts']['no_view_target_snapshots']}",
        f"- 板块目标行：{report['row_counts']['sector_target_rows']}",
        f"- 伪样本外预测：{evaluation['mature_oos_predictions']}",
        f"- 已校准分布预测：{evaluation['calibrated_predictions']}",
        f"- 预测日期：{evaluation['first_prediction_date']} 至 {evaluation['last_mature_prediction_date']}",
        "",
        "## 唯一候选 S1",
        "",
        f"- Spearman：{evaluation['spearman']}",
        f"- 移动块 Bootstrap 95%区间：{bootstrap_interval}",
        f"- S1 MAE：{evaluation['model_mae']}",
        f"- B0 聚合估值 MAE：{evaluation['aggregate_baseline_mae']}",
        f"- 扩展历史均值 MAE：{evaluation['expanding_mean_mae']}",
        f"- 相对 B0 改善：{evaluation['mae_improvement_vs_aggregate']}",
        f"- 相对历史均值改善：{evaluation['mae_improvement_vs_expanding_mean']}",
        f"- 方向准确率：{evaluation['direction_accuracy']}",
        f"- 前半段 Spearman：{evaluation['first_half_spearman']}",
        f"- 后半段 Spearman：{evaluation['second_half_spearman']}",
        f"- 与 ETF_X60 Spearman：{evaluation['etf_x60_spearman']}",
        f"- Brier Skill：{evaluation['brier_skill']}",
        f"- 80%区间覆盖率：{evaluation['interval_80_coverage']}",
        "",
        "## 冻结门槛",
        "",
        *gate_lines,
        "",
        "## 解释边界",
        "",
        "历史通过也只允许进入真正前向准备；历史失败则冻结失败。任何结果都不生成仓位、订单或券商动作，也不允许改期限、alpha、特征或门槛补救。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    contract = yaml.safe_load(CONTRACT_FILE.read_text(encoding="utf-8"))
    manifest = _verify_manifest(contract)
    inputs = contract["inputs"]
    data_status = json.loads(
        (ROOT / inputs["sector_data_status"]).read_text(encoding="utf-8")
    )
    if data_status.get("status") != "READY_FOR_TARGET_FREEZE":
        raise RuntimeError("板块点时数据未通过目标冻结门")

    sector_panel = pd.read_parquet(ROOT / inputs["sector_panel"])
    weights = pd.read_parquet(ROOT / inputs["weights"])
    constituent_daily = pd.read_parquet(ROOT / inputs["constituent_daily"])
    industry_intervals = pd.read_parquet(ROOT / inputs["industry_intervals"])
    etf_daily = pd.read_parquet(ROOT / inputs["etf_daily"])
    dividends = pd.read_csv(ROOT / inputs["etf_dividends"])

    rules = PredictionRules.from_contract(contract)
    targets = build_sector_forward_targets(
        sector_panel,
        weights,
        constituent_daily,
        industry_intervals,
        rules,
    )
    if targets.empty:
        raise RuntimeError("没有可评价的成熟60日目标")
    etf_targets = build_etf_forward_returns(
        etf_daily,
        dividends,
        targets["date"].drop_duplicates(),
        rules,
    )
    forecasts = walk_forward_sector_forecasts(
        sector_panel, targets, etf_targets, rules
    )
    if forecasts.empty:
        raise RuntimeError("没有满足冻结训练长度的伪样本外预测")
    if forecasts["research_output"].ne("PREDICTION_AUDIT_ONLY").any():
        raise AssertionError("历史运行非法输出预测资格")
    evaluation = evaluate_sector_forecasts(forecasts, rules)
    if evaluation["safety"]["forecast_eligible_emitted"]:
        raise AssertionError("历史运行非法输出 FORECAST_ELIGIBLE")

    output_contract = contract["outputs"]
    _atomic_parquet(targets, ROOT / output_contract["targets"])
    _atomic_parquet(forecasts, ROOT / output_contract["forecasts"])
    target_snapshot = targets.drop_duplicates("date")
    report = _json_safe(
        {
            "project_id": contract["protocol"]["project_id"],
            "version": contract["protocol"]["version"],
            "status": evaluation["status"],
            "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "data_cutoff": str(rules.data_cutoff.date()),
            "manifest_sha256": sha256(MANIFEST_FILE),
            "frozen_at": manifest["frozen_at"],
            "row_counts": {
                "sector_target_rows": int(len(targets)),
                "mature_target_snapshots": int(
                    target_snapshot["target_output"].eq("TARGET_READY").sum()
                ),
                "no_view_target_snapshots": int(
                    target_snapshot["target_output"].eq("NO_VIEW").sum()
                ),
                "etf_target_snapshots": int(len(etf_targets)),
                "oos_forecasts": int(len(forecasts)),
            },
            "target_date_range": {
                "first_signal_date": str(targets["date"].min().date()),
                "last_mature_signal_date": str(targets["date"].max().date()),
                "last_maturity_date": str(targets["maturity_date"].max().date()),
                "minimum_endpoint_weight_coverage": float(
                    target_snapshot["endpoint_weight_coverage"].min()
                ),
            },
            "evaluation": evaluation,
            "governance": contract["governance"],
            "safety": {
                "forecast_eligible": "NOT_AUTHORIZED",
                "position_mapping": "DISABLED",
                "order_generation": "DISABLED",
                "broker_connection": "DISABLED",
            },
        }
    )
    json_path = ROOT / output_contract["result_json"]
    markdown_path = ROOT / output_contract["result_markdown"]
    _atomic_text(json.dumps(report, ensure_ascii=False, indent=2), json_path)
    _atomic_text(_render_markdown(report), markdown_path)
    print(
        json.dumps(
            {
                "状态": report["status"],
                "成熟预测数": evaluation["mature_oos_predictions"],
                "Spearman": evaluation["spearman"],
                "相对B0_MAE改善": evaluation["mae_improvement_vs_aggregate"],
                "相对历史均值_MAE改善": evaluation[
                    "mae_improvement_vs_expanding_mean"
                ],
                "方向准确率": evaluation["direction_accuracy"],
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
