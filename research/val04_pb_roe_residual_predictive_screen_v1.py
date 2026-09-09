"""VAL-04 预冻结预测屏幕；只做预测诊断，不做策略回测。"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from research.normalized_valuation_5y_predictive_screen_v1 import (
    audit_frozen_labels,
    build_family_reports,
    compute_model_diagnostics,
)
from research.val04_pb_roe_residual_data_feasibility_v1 import (
    canonical_hash,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = (
    ROOT / "config" / "val04_pb_roe_residual_predictive_screen_v1.yaml"
)
MODEL_ID = "VAL04_PB_ROE_RESIDUAL_V1"


def load_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("VAL-04 预测配置不是映射")
    return config


def verify_input_hashes(config: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, contract in config["data_contracts"].items():
        path = ROOT / str(contract["file"])
        actual = sha256_file(path) if path.is_file() else None
        rows.append(
            {
                "dataset": name,
                "file": str(contract["file"]),
                "expected_sha256": str(contract["sha256"]),
                "actual_sha256": actual,
                "matches": actual == contract["sha256"],
            }
        )
    return {
        "status": "PASS" if all(row["matches"] for row in rows) else "FAILED",
        "files": rows,
    }


def run_screen(
    signals: pd.DataFrame,
    labels: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = config or load_config()
    hash_audit = verify_input_hashes(config)
    if hash_audit["status"] != "PASS":
        raise RuntimeError("VAL-04 预测输入哈希漂移")
    label_audit = audit_frozen_labels(labels, config)
    if label_audit["status"] != "PASS":
        raise RuntimeError("VAL-04 冻结标签结构或成熟数不一致")
    diagnostics = {
        MODEL_ID: compute_model_diagnostics(
            signals, labels, MODEL_ID, config
        )
    }
    model_reports, family = build_family_reports(
        diagnostics, label_audit, hash_audit, config
    )
    model = model_reports[MODEL_ID]
    family["global_research_ledger_still_required"] = True
    family["alpha_pass"] = False
    family["historical_strategy_backtest_run"] = False
    return model, family


def render_model_report(report: Mapping[str, Any]) -> str:
    gate = report["primary_predictive_gate"]
    primary = report["diagnostics"][str(gate["horizon_trading_days"])][
        "etf_total_return"
    ]
    confirmation = report["diagnostics"][str(gate["horizon_trading_days"])][
        "h00300_total_return"
    ]
    bucket_rows = [
        "|分档|样本|ETF均值|ETF中位数|",
        "|---|---:|---:|---:|",
    ]
    for row in primary["buckets"]:
        bucket_rows.append(
            f"|{row['bucket']}|{row['observations']}|{row['mean_return']:.4%}|{row['median_return']:.4%}|"
        )
    return "\n".join(
        [
            "# VAL-04 PB—ROE 残余收益预测诊断 V1",
            "",
            f"> 状态：`{report['status']}`。历史信号为事后重建，不是严格样本外。",
            "",
            "## 242日主要结果",
            "",
            f"- 成熟样本：{primary['matured_observations']}。",
            f"- ETF Spearman IC：{primary['spearman_ic']}。",
            f"- 前半/后半 IC：{primary['chronological_halves']['first']['spearman_ic']} / {primary['chronological_halves']['second']['spearman_ic']}。",
            f"- HAC 单侧原始 p：{gate['raw_hac_one_sided_p_value']}；Holm 调整 p：{gate['holm_adjusted_p_value']}。",
            f"- ETF 最便宜减最贵平均收益：{primary['cheapest_minus_expensive_mean_return']}。",
            f"- H00300 Spearman IC：{confirmation['spearman_ic']}。",
            f"- 失败闸门：{gate['failed_checks']}。",
            "",
            *bucket_rows,
            "",
            "## 边界",
            "",
            "- 未计算策略净值、交易成本、仓位、当前建议、目标股数或订单。",
        ]
    ) + "\n"


def render_family_report(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# VAL-04 PB—ROE 残余收益预测屏幕总表 V1",
            "",
            f"> 状态：`{report['status']}`。",
            "",
            f"- 通过模型：{report['passed_models']}。",
            f"- 拒绝模型：{report['rejected_models']}。",
            f"- Holm 检验：{report['holm_bonferroni']['ordered_tests']}。",
            "- 本屏幕不构成 ALPHA_PASS，且未运行策略回测。",
        ]
    ) + "\n"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_results(
    model_report: Mapping[str, Any],
    family_report: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = config["artifacts"]
    paths = {
        "model_report_json": ROOT / artifacts["model_report_json"],
        "model_report_markdown": ROOT / artifacts["model_report_markdown"],
        "family_report_json": ROOT / artifacts["family_report_json"],
        "family_report_markdown": ROOT / artifacts["family_report_markdown"],
    }
    _atomic_text(
        paths["model_report_json"],
        json.dumps(model_report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    _atomic_text(paths["model_report_markdown"], render_model_report(model_report))
    _atomic_text(
        paths["family_report_json"],
        json.dumps(family_report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    _atomic_text(paths["family_report_markdown"], render_family_report(family_report))
    protocol_path = ROOT / artifacts["protocol_manifest"]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_PREDICTIVE_SCREEN_PROTOCOL":
        raise RuntimeError("VAL-04 预测协议未冻结")
    payload = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "status": "FROZEN_PREDICTIVE_SCREEN_RESULTS_NO_STRATEGY_BACKTEST",
        "frozen_at": datetime.now(
            ZoneInfo(config["protocol"]["timezone"])
        ).isoformat(),
        "protocol_manifest": {
            "file": str(protocol_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(protocol_path),
            "manifest_content_sha256": protocol["manifest_content_sha256"],
        },
        "input_files": {
            name: {"file": contract["file"], "sha256": contract["sha256"]}
            for name, contract in config["data_contracts"].items()
        },
        "output_files": {
            name: {
                "file": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
        "family_status": family_report["status"],
        "model_status": family_report["model_status"],
        "strategy_evaluation_authorized_models": family_report[
            "strategy_evaluation_authorized_models"
        ],
        "governance": {
            **family_report["governance"],
            "alpha_pass": False,
            "current_signal_mapped": False,
            "target_shares_generated": False,
        },
        "meaning": "冻结预测筛选结果；未运行策略、成本回测或当前信号映射。",
    }
    payload["manifest_content_sha256"] = canonical_hash(payload)
    manifest_path = ROOT / artifacts["result_manifest"]
    _atomic_text(
        manifest_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    return {
        "file": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256_file(manifest_path),
        "manifest_content_sha256": payload["manifest_content_sha256"],
    }


__all__ = [
    "MODEL_ID",
    "load_config",
    "render_family_report",
    "render_model_report",
    "run_screen",
    "verify_input_hashes",
    "write_results",
]
