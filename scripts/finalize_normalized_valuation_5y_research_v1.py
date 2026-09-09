"""冻结正常化估值五年研究的终局拒绝证据。"""

from __future__ import annotations

from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.normalized_valuation_5y_models_v1 import canonical_hash, sha256_file


FINAL_REPORT_JSON = (
    ROOT / "reports" / "research" / "normalized_valuation_5y_final_audit_v1.json"
)
FINAL_REPORT_MARKDOWN = (
    ROOT / "reports" / "research" / "normalized_valuation_5y_final_audit_v1.md"
)
FINAL_MANIFEST = (
    ROOT / "config" / "normalized_valuation_5y_final_audit_v1_manifest.json"
)

FROZEN_FILES = (
    "config/val01_norm_ey_5y_financial_extension_v1_manifest.json",
    "config/normalized_valuation_5y_models_v1_protocol_manifest.json",
    "config/val01_norm_ey_5y_v1_manifest.json",
    "config/val02_norm_ey_spread_5y_v1_manifest.json",
    "config/normalized_valuation_5y_predictive_screen_v1_protocol_manifest.json",
    "config/normalized_valuation_5y_predictive_screen_v1_result_manifest.json",
    "reports/data_quality/000300_normalized_earnings_financial_extension_v1.json",
    "reports/data_quality/000300_point_in_time_valuation_v2_reconstructibility_v1_2.json",
    "reports/data_quality/000300_val01_norm_ey_5y_v1_signal_input_gate.json",
    "reports/data_quality/000300_val02_norm_ey_spread_5y_v1_signal_input_gate.json",
    "reports/research/val01_norm_ey_5y_v1_predictive_diagnostics.json",
    "reports/research/val02_norm_ey_spread_5y_v1_predictive_diagnostics.json",
    "reports/research/normalized_valuation_5y_family_predictive_screen_v1.json",
    "data/features/000300_val01_norm_ey_5y_v1_signal_inputs.parquet",
    "data/features/000300_val02_norm_ey_spread_5y_v1_signal_inputs.parquet",
    "data/labels/000300_val01_raw_ey_5y_v1_forward_return_labels.parquet",
    "config/normalized_valuation_5y_models_v1.yaml",
    "config/normalized_valuation_5y_predictive_screen_v1.yaml",
    "docs/VAL01_NORM_EY_5Y_V1_MODEL_CARD.md",
    "docs/VAL02_NORM_EY_SPREAD_5Y_V1_MODEL_CARD.md",
    "docs/NORMALIZED_VALUATION_5Y_PREDICTIVE_SCREEN_V1_PROTOCOL.md",
    "research/normalized_valuation_5y_models_v1.py",
    "research/normalized_valuation_5y_predictive_screen_v1.py",
    "scripts/build_normalized_valuation_5y_signals_v1.py",
    "scripts/run_normalized_valuation_5y_predictive_screen_v1.py",
    "scripts/finalize_normalized_valuation_5y_research_v1.py",
    "tests/test_normalized_valuation_5y_models_v1.py",
    "tests/test_normalized_valuation_5y_predictive_screen_v1.py",
)


def load_json(relative_path: str) -> dict[str, Any]:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def verify_manifest_self_hash(relative_path: str) -> dict[str, Any]:
    payload = load_json(relative_path)
    expected = payload.pop("manifest_content_sha256")
    actual = canonical_hash(payload)
    if actual != expected:
        raise RuntimeError(f"清单自哈希不一致：{relative_path}")
    return {
        "file": relative_path,
        "expected": expected,
        "actual": actual,
        "matches": True,
    }


def independent_recompute(
    signal_path: str, predictor: str, ready: str
) -> dict[str, Any]:
    labels = pd.read_parquet(
        ROOT / "data/labels/000300_val01_raw_ey_5y_v1_forward_return_labels.parquet"
    )
    signals = pd.read_parquet(ROOT / signal_path)
    labels["signal_observation_date"] = pd.to_datetime(
        labels["signal_observation_date"]
    )
    signals["date"] = pd.to_datetime(signals["date"])
    values = signals.loc[signals[ready], ["date", predictor]].rename(
        columns={"date": "signal_observation_date"}
    )
    frame = labels.loc[
        labels["horizon_trading_days"].eq(242)
        & labels["label_status"].eq("MATURED")
    ].merge(values, on="signal_observation_date", validate="many_to_one")
    frame = frame.sort_values("signal_observation_date").reset_index(drop=True)
    frame["bucket"] = pd.cut(
        frame[predictor],
        [-1e-12, 0.20, 0.40, 0.60, 1.0000001],
        right=False,
        labels=["B1_EXPENSIVE", "B2", "B3", "B4_CHEAP"],
    )
    midpoint = len(frame) // 2
    bucket = (
        frame.groupby("bucket", observed=False)["etf_total_return"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    return {
        "matured_observations": int(len(frame)),
        "spearman_ic": float(
            frame[[predictor, "etf_total_return"]].corr(method="spearman").iloc[0, 1]
        ),
        "first_half_spearman_ic": float(
            frame.iloc[:midpoint][[predictor, "etf_total_return"]]
            .corr(method="spearman")
            .iloc[0, 1]
        ),
        "second_half_spearman_ic": float(
            frame.iloc[midpoint:][[predictor, "etf_total_return"]]
            .corr(method="spearman")
            .iloc[0, 1]
        ),
        "buckets": [
            {
                "bucket": str(row.bucket),
                "observations": int(row.count),
                "mean_return": None if pd.isna(row.mean) else float(row.mean),
                "median_return": None if pd.isna(row.median) else float(row.median),
            }
            for row in bucket.itertuples(index=False)
        ],
    }


def assert_matches_report(
    independent: dict[str, Any], report: dict[str, Any]
) -> None:
    official = report["diagnostics"]["242"]["etf_total_return"]
    comparisons = (
        (independent["spearman_ic"], official["spearman_ic"], "整体IC"),
        (
            independent["first_half_spearman_ic"],
            official["chronological_halves"]["first"]["spearman_ic"],
            "前半IC",
        ),
        (
            independent["second_half_spearman_ic"],
            official["chronological_halves"]["second"]["spearman_ic"],
            "后半IC",
        ),
    )
    for actual, expected, label in comparisons:
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"独立复算{label}不一致：{actual} != {expected}")
    actual_counts = [row["observations"] for row in independent["buckets"]]
    expected_counts = [row["observations"] for row in official["buckets"]]
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"独立复算分档数不一致：{actual_counts} != {expected_counts}"
        )


def run_tests(arguments: list[str]) -> dict[str, Any]:
    command = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        "-m",
        "pytest",
        *arguments,
        "-q",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"终局冻结测试失败：\n{result.stdout}\n{result.stderr}")
    return {
        "command": command,
        "return_code": result.returncode,
        "stdout": result.stdout.strip(),
    }


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def render_report(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# 正常化估值五年研究终局审计 V1",
            "",
            f"> 状态：`{report['status']}`。",
            "",
            "## 分支结论",
            "",
            f"- 原始EY：`{report['branch_status']['VAL01_RAW_EY_5Y']}`。",
            f"- 正常化EY：`{report['branch_status']['VAL01_NORM_EY_5Y']}`。",
            f"- 正常化EY利差：`{report['branch_status']['VAL02_NORM_EY_SPREAD_5Y']}`。",
            f"- 7年分支：`{report['branch_status']['NORMALIZED_7Y']}`。",
            "",
            "## 审计结论",
            "",
            "- 财务扩展、60月信号输入和数据覆盖均通过。",
            "- 两个5年模型均在预冻结242日预测硬门槛中失败。",
            "- 拒绝后未运行策略净值、仓位、成本回测、当前建议、份额或订单。",
            "- 禁止在同一试验内改分档、期限、正常化公式或利率期限进行参数救援。",
            "",
        ]
    )


def main() -> int:
    missing = [path for path in FROZEN_FILES if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"终局冻结文件缺失：{missing}")
    val01_report = load_json(
        "reports/research/val01_norm_ey_5y_v1_predictive_diagnostics.json"
    )
    val02_report = load_json(
        "reports/research/val02_norm_ey_spread_5y_v1_predictive_diagnostics.json"
    )
    family = load_json(
        "reports/research/normalized_valuation_5y_family_predictive_screen_v1.json"
    )
    expected_reject = "REJECT_PREDICTIVE_SCREEN_STOP_NO_STRATEGY_BACKTEST"
    if val01_report["status"] != expected_reject or val02_report["status"] != expected_reject:
        raise RuntimeError("两个模型并非都处于预期拒绝状态")
    if family["status"] != "ALL_REGISTERED_MODELS_REJECTED":
        raise RuntimeError("家族报告并非全部拒绝")
    for report in (val01_report, val02_report):
        governance = report["governance"]
        prohibited = [
            key
            for key in (
                "historical_strategy_return_calculated",
                "historical_position_mapping_performed",
                "current_position_mapping_performed",
                "target_shares_generated",
                "orders_generated",
                "broker_connection_performed",
                "alpha_pass",
            )
            if governance.get(key) is not False
        ]
        if prohibited:
            raise RuntimeError(f"拒绝后发生越权动作：{prohibited}")
    val01_independent = independent_recompute(
        "data/features/000300_val01_norm_ey_5y_v1_signal_inputs.parquet",
        "norm_ey_percentile_60m",
        "norm_ey_percentile_60m_ready",
    )
    val02_independent = independent_recompute(
        "data/features/000300_val02_norm_ey_spread_5y_v1_signal_inputs.parquet",
        "norm_ey_spread_percentile_60m",
        "norm_ey_spread_percentile_60m_ready",
    )
    assert_matches_report(val01_independent, val01_report)
    assert_matches_report(val02_independent, val02_report)
    self_hash_audits = [
        verify_manifest_self_hash(path)
        for path in (
            "config/val01_norm_ey_5y_financial_extension_v1_manifest.json",
            "config/normalized_valuation_5y_models_v1_protocol_manifest.json",
            "config/val01_norm_ey_5y_v1_manifest.json",
            "config/val02_norm_ey_spread_5y_v1_manifest.json",
            "config/normalized_valuation_5y_predictive_screen_v1_protocol_manifest.json",
            "config/normalized_valuation_5y_predictive_screen_v1_result_manifest.json",
        )
    ]
    targeted = run_tests(
        [
            "tests/test_val01_norm_ey_5y_financial_extension.py",
            "tests/test_val01_norm_ey_5y_post_acquisition_audit.py",
            "tests/test_normalized_valuation_5y_models_v1.py",
            "tests/test_normalized_valuation_5y_predictive_screen_v1.py",
        ]
    )
    full_suite = run_tests([])
    report: dict[str, Any] = {
        "project_id": "NORMALIZED_VALUATION_5Y_FINAL_AUDIT_V1",
        "version": "1.0.0",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "HISTORICAL_REJECTED_FROZEN_NO_STRATEGY_BACKTEST",
        "branch_status": {
            "VAL01_RAW_EY_5Y": "REJECTED_PREDICTIVE_SCREEN_NOT_REOPENED",
            "VAL01_NORM_EY_5Y": expected_reject,
            "VAL02_NORM_EY_SPREAD_5Y": expected_reject,
            "NORMALIZED_7Y": "BLOCKED_HISTORY_SHORT_AND_ACQUISITION_REQUIRED",
        },
        "family_status": family["status"],
        "holm_bonferroni": family["holm_bonferroni"],
        "independent_recomputation": {
            "VAL01_NORM_EY_5Y": val01_independent,
            "VAL02_NORM_EY_SPREAD_5Y": val02_independent,
            "matches_formal_reports": True,
        },
        "manifest_self_hash_audit": self_hash_audits,
        "tests": {"targeted": targeted, "full_suite": full_suite},
        "governance": {
            "strategy_returns_calculated": False,
            "positions_generated": False,
            "current_signal_mapped": False,
            "target_shares_generated": False,
            "orders_generated": False,
            "broker_connection_performed": False,
            "rejected_model_parameter_rescue_performed": False,
        },
        "next_allowed_step": (
            "停止VAL01/VAL02五年分支；如继续模型层，只能登记正交候选，"
            "不得在本试验内改分档、窗口、正常化公式或利率期限。"
        ),
    }
    atomic_text(
        FINAL_REPORT_JSON,
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
    )
    atomic_text(FINAL_REPORT_MARKDOWN, render_report(report))
    manifest: dict[str, Any] = {
        "project_id": report["project_id"],
        "version": report["version"],
        "status": report["status"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "frozen_files": {
            path: {"sha256": sha256_file(ROOT / path), "size_bytes": (ROOT / path).stat().st_size}
            for path in FROZEN_FILES
        },
        "final_report": {
            "json": {
                "file": FINAL_REPORT_JSON.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(FINAL_REPORT_JSON),
            },
            "markdown": {
                "file": FINAL_REPORT_MARKDOWN.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(FINAL_REPORT_MARKDOWN),
            },
        },
        "branch_status": report["branch_status"],
        "test_results": report["tests"],
        "governance": report["governance"],
        "meaning": "冻结历史预测拒绝结果并停止；不是当前买卖指令。",
    }
    manifest["manifest_content_sha256"] = canonical_hash(manifest)
    atomic_text(FINAL_MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "status": report["status"],
                "branch_status": report["branch_status"],
                "independent_recomputation_matches": True,
                "targeted_tests": targeted["stdout"],
                "full_suite": full_suite["stdout"],
                "manifest": FINAL_MANIFEST.relative_to(ROOT).as_posix(),
                "manifest_content_sha256": manifest["manifest_content_sha256"],
                "governance": report["governance"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
