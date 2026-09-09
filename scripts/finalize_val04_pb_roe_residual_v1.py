"""独立复算并冻结 VAL-04 的终局拒绝证据。"""

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

from research.val04_pb_roe_residual_data_feasibility_v1 import (
    canonical_hash,
    sha256_file,
)


FINAL_REPORT_JSON = ROOT / "reports" / "research" / "val04_pb_roe_residual_final_audit_v1.json"
FINAL_REPORT_MARKDOWN = ROOT / "reports" / "research" / "val04_pb_roe_residual_final_audit_v1.md"
FINAL_MANIFEST = ROOT / "config" / "val04_pb_roe_residual_final_audit_v1_manifest.json"

SELF_HASH_MANIFESTS = (
    "config/val04_pb_roe_residual_data_feasibility_v1_protocol_manifest.json",
    "config/val04_pb_roe_residual_data_feasibility_v1_result_manifest.json",
    "config/val04_pb_roe_residual_data_feasibility_v2_protocol_manifest.json",
    "config/val04_pb_roe_residual_data_feasibility_v2_result_manifest.json",
    "config/val04_pb_roe_residual_v1_protocol_manifest.json",
    "config/val04_pb_roe_residual_v1_manifest.json",
    "config/val04_pb_roe_residual_predictive_screen_v1_protocol_manifest.json",
    "config/val04_pb_roe_residual_predictive_screen_v1_result_manifest.json",
)

FROZEN_EVIDENCE = (
    *SELF_HASH_MANIFESTS,
    "config/val04_pb_roe_residual_data_feasibility_v1.yaml",
    "config/val04_pb_roe_residual_data_feasibility_v2.yaml",
    "config/val04_pb_roe_residual_v1.yaml",
    "config/val04_pb_roe_residual_predictive_screen_v1.yaml",
    "docs/VAL04_PB_ROE_RESIDUAL_DATA_FEASIBILITY_V1_PROTOCOL.md",
    "docs/VAL04_PB_ROE_RESIDUAL_DATA_FEASIBILITY_V2_ADDENDUM.md",
    "docs/VAL04_PB_ROE_RESIDUAL_V1_MODEL_CARD.md",
    "docs/VAL04_PB_ROE_RESIDUAL_PREDICTIVE_SCREEN_V1_PROTOCOL.md",
    "research/val04_pb_roe_residual_data_feasibility_v1.py",
    "research/val04_pb_roe_residual_v1.py",
    "research/val04_pb_roe_residual_predictive_screen_v1.py",
    "scripts/freeze_val04_pb_roe_residual_data_feasibility_v1_protocol.py",
    "scripts/audit_val04_pb_roe_residual_data_feasibility_v1.py",
    "scripts/freeze_val04_pb_roe_residual_data_feasibility_v2_protocol.py",
    "scripts/audit_val04_pb_roe_residual_data_feasibility_v2.py",
    "scripts/freeze_val04_pb_roe_residual_v1_protocol.py",
    "scripts/build_val04_pb_roe_residual_v1_signals.py",
    "scripts/freeze_val04_pb_roe_residual_predictive_screen_v1_protocol.py",
    "scripts/run_val04_pb_roe_residual_predictive_screen_v1.py",
    "scripts/finalize_val04_pb_roe_residual_v1.py",
    "tests/test_val04_pb_roe_residual_data_feasibility_v1.py",
    "tests/test_val04_pb_roe_residual_v1.py",
    "tests/test_val04_pb_roe_residual_predictive_screen_v1.py",
    "data/features/000300_val04_pb_roe_residual_feasibility_constituents_v1.parquet",
    "data/audit/000300_val04_pb_roe_residual_feasibility_snapshots_v1.parquet",
    "data/features/000300_val04_pb_roe_residual_feasibility_constituents_v2.parquet",
    "data/audit/000300_val04_pb_roe_residual_feasibility_snapshots_v2.parquet",
    "data/features/000300_val04_pb_roe_residual_v1_company_residuals.parquet",
    "data/features/000300_val04_pb_roe_residual_v1_signal_inputs.parquet",
    "reports/data_quality/000300_val04_pb_roe_residual_data_feasibility_v1.json",
    "reports/data_quality/000300_val04_pb_roe_residual_data_feasibility_v1.md",
    "reports/data_quality/000300_val04_pb_roe_residual_data_feasibility_v2.json",
    "reports/data_quality/000300_val04_pb_roe_residual_data_feasibility_v2.md",
    "reports/data_quality/000300_val04_pb_roe_residual_v1_signal_input_gate.json",
    "reports/data_quality/000300_val04_pb_roe_residual_v1_signal_input_gate.md",
    "reports/research/val04_pb_roe_residual_v1_predictive_diagnostics.json",
    "reports/research/val04_pb_roe_residual_v1_predictive_diagnostics.md",
    "reports/research/val04_pb_roe_residual_family_predictive_screen_v1.json",
    "reports/research/val04_pb_roe_residual_family_predictive_screen_v1.md",
)


def load_json(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def audit_self_hash(relative: str) -> dict[str, Any]:
    payload = load_json(relative)
    expected = payload.pop("manifest_content_sha256")
    actual = canonical_hash(payload)
    if actual != expected:
        raise RuntimeError(f"清单自哈希不一致：{relative}")
    return {
        "file": relative,
        "expected": expected,
        "actual": actual,
        "matches": True,
    }


def audit_protocol_frozen_files(relative: str) -> dict[str, Any]:
    payload = load_json(relative)
    drift: list[str] = []
    for file, contract in payload.get("frozen_files", {}).items():
        expected = contract if isinstance(contract, str) else contract["sha256"]
        path = ROOT / file
        actual = sha256_file(path) if path.is_file() else None
        if actual != expected:
            drift.append(file)
    if drift:
        raise RuntimeError(f"协议冻结文件漂移：{relative} -> {drift}")
    return {"file": relative, "status": "PASS", "checked_file_count": len(payload.get("frozen_files", {}))}


def frozen_bucket(value: float) -> str:
    if value < 0.20:
        return "B1_EXPENSIVE"
    if value < 0.40:
        return "B2"
    if value < 0.60:
        return "B3"
    return "B4_CHEAP"


def independent_recompute() -> dict[str, Any]:
    signals = pd.read_parquet(
        ROOT / "data/features/000300_val04_pb_roe_residual_v1_signal_inputs.parquet"
    )
    labels = pd.read_parquet(
        ROOT / "data/labels/000300_val01_raw_ey_5y_v1_forward_return_labels.parquet"
    )
    predictor = "pb_roe_residual_percentile_60m"
    ready = signals.loc[
        signals["pb_roe_residual_percentile_60m_ready"], ["date", predictor]
    ].rename(columns={"date": "signal_observation_date"})
    ready["signal_observation_date"] = pd.to_datetime(ready["signal_observation_date"])
    labels["signal_observation_date"] = pd.to_datetime(labels["signal_observation_date"])
    frame = labels.loc[
        labels["horizon_trading_days"].eq(242)
        & labels["label_status"].eq("MATURED"),
        ["signal_observation_date", "etf_total_return", "h00300_total_return"],
    ].merge(ready, on="signal_observation_date", validate="one_to_one")
    frame = frame.sort_values("signal_observation_date").reset_index(drop=True)
    midpoint = len(frame) // 2
    frame["bucket"] = frame[predictor].map(frozen_bucket)
    buckets = []
    for bucket in ("B1_EXPENSIVE", "B2", "B3", "B4_CHEAP"):
        values = frame.loc[frame["bucket"].eq(bucket), "etf_total_return"]
        buckets.append(
            {
                "bucket": bucket,
                "observations": int(len(values)),
                "mean_return": float(values.mean()),
                "median_return": float(values.median()),
            }
        )
    return {
        "matured_observations": int(len(frame)),
        "etf_spearman_ic": float(
            frame[[predictor, "etf_total_return"]].corr(method="spearman").iloc[0, 1]
        ),
        "h00300_spearman_ic": float(
            frame[[predictor, "h00300_total_return"]].corr(method="spearman").iloc[0, 1]
        ),
        "first_half_etf_spearman_ic": float(
            frame.iloc[:midpoint][[predictor, "etf_total_return"]]
            .corr(method="spearman")
            .iloc[0, 1]
        ),
        "second_half_etf_spearman_ic": float(
            frame.iloc[midpoint:][[predictor, "etf_total_return"]]
            .corr(method="spearman")
            .iloc[0, 1]
        ),
        "buckets": buckets,
    }


def assert_independent_matches(
    independent: dict[str, Any], report: dict[str, Any]
) -> None:
    official = report["diagnostics"]["242"]
    etf = official["etf_total_return"]
    h00300 = official["h00300_total_return"]
    comparisons = (
        (independent["etf_spearman_ic"], etf["spearman_ic"], "ETF整体IC"),
        (independent["h00300_spearman_ic"], h00300["spearman_ic"], "H00300整体IC"),
        (
            independent["first_half_etf_spearman_ic"],
            etf["chronological_halves"]["first"]["spearman_ic"],
            "ETF前半IC",
        ),
        (
            independent["second_half_etf_spearman_ic"],
            etf["chronological_halves"]["second"]["spearman_ic"],
            "ETF后半IC",
        ),
    )
    for actual, expected, label in comparisons:
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"独立复算{label}不一致：{actual} != {expected}")
    actual_buckets = independent["buckets"]
    official_buckets = etf["buckets"]
    for actual, expected in zip(actual_buckets, official_buckets, strict=True):
        if actual["observations"] != expected["observations"]:
            raise RuntimeError("独立复算分档样本数不一致")
        for field in ("mean_return", "median_return"):
            if not math.isclose(
                actual[field], expected[field], rel_tol=0.0, abs_tol=1e-12
            ):
                raise RuntimeError(f"独立复算分档{field}不一致")


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
        raise RuntimeError(f"VAL-04 终局测试失败：\n{result.stdout}\n{result.stderr}")
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
    primary = report["predictive_result"]["diagnostics"]["242"]["etf_total_return"]
    gate = report["predictive_result"]["primary_predictive_gate"]
    return "\n".join(
        [
            "# VAL-04 PB—ROE 残余收益研究终局审计 V1",
            "",
            f"> 状态：`{report['status']}`。",
            "",
            "## 分支状态",
            "",
            f"- 中信行业数据 V1：`{report['branch_status']['DATA_V1_CITIC']}`。",
            f"- 申万点时行业数据 V2：`{report['branch_status']['DATA_V2_SW']}`。",
            f"- 无收益信号构建：`{report['branch_status']['SIGNAL_BUILD']}`。",
            f"- 预测屏幕：`{report['branch_status']['PREDICTIVE_SCREEN']}`。",
            "",
            "## 242日主要证据",
            "",
            f"- 成熟样本：{primary['matured_observations']}。",
            f"- 整体 ETF Spearman IC：{primary['spearman_ic']:.6f}。",
            f"- 前半/后半 IC：{primary['chronological_halves']['first']['spearman_ic']:.6f} / {primary['chronological_halves']['second']['spearman_ic']:.6f}。",
            f"- HAC 单侧 p / Holm p：{gate['raw_hac_one_sided_p_value']:.6f} / {gate['holm_adjusted_p_value']:.6f}。",
            f"- 最便宜减最贵平均收益：{primary['cheapest_minus_expensive_mean_return']:.4%}。",
            f"- 失败闸门：`{gate['failed_checks']}`。",
            "",
            "## 结论",
            "",
            "整体相关和端点价差为正，但后半样本方向反转，且最便宜档没有继续优于第三档。按预冻结规则必须拒绝；不得运行策略回测或参数救援。",
            "",
            "## 治理",
            "",
            "- 未计算策略净值或交易成本。",
            "- 未映射当前信号、仓位、目标股数或订单。",
            "- 未连接券商；本报告不是买卖指令。",
        ]
    ) + "\n"


def main() -> int:
    missing = [relative for relative in FROZEN_EVIDENCE if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"VAL-04 终局证据缺失：{missing}")
    self_hash_audit = [audit_self_hash(relative) for relative in SELF_HASH_MANIFESTS]
    protocol_audit = [
        audit_protocol_frozen_files(relative)
        for relative in SELF_HASH_MANIFESTS
        if "protocol_manifest" in relative
    ]
    v1_data = load_json(
        "reports/data_quality/000300_val04_pb_roe_residual_data_feasibility_v1.json"
    )
    v2_data = load_json(
        "reports/data_quality/000300_val04_pb_roe_residual_data_feasibility_v2.json"
    )
    signal = load_json(
        "reports/data_quality/000300_val04_pb_roe_residual_v1_signal_input_gate.json"
    )
    predictive = load_json(
        "reports/research/val04_pb_roe_residual_v1_predictive_diagnostics.json"
    )
    if predictive["status"] != "REJECT_PREDICTIVE_SCREEN_STOP_NO_STRATEGY_BACKTEST":
        raise RuntimeError("VAL-04 预测状态不是预期拒绝")
    independent = independent_recompute()
    assert_independent_matches(independent, predictive)
    targeted = run_tests(
        [
            "tests/test_val04_pb_roe_residual_data_feasibility_v1.py",
            "tests/test_val04_pb_roe_residual_v1.py",
            "tests/test_val04_pb_roe_residual_predictive_screen_v1.py",
        ]
    )
    full = run_tests([])
    report = {
        "project_id": "VAL04_PB_ROE_RESIDUAL_FINAL_AUDIT_V1",
        "version": "1.0.0",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "HISTORICAL_REJECTED_FROZEN_NO_STRATEGY_BACKTEST",
        "branch_status": {
            "DATA_V1_CITIC": v1_data["status"],
            "DATA_V2_SW": v2_data["status"],
            "SIGNAL_BUILD": signal["status"],
            "PREDICTIVE_SCREEN": predictive["status"],
        },
        "data_evidence": {
            "citic_v1_window": v1_data["window"],
            "sw_v2_window": v2_data["window"],
            "sw_v2_coverage_summary": v2_data["coverage_summary"],
        },
        "signal_evidence": signal["signal_summary"],
        "predictive_result": {
            "status": predictive["status"],
            "primary_predictive_gate": predictive["primary_predictive_gate"],
            "diagnostics": {"242": predictive["diagnostics"]["242"]},
        },
        "independent_recompute": independent,
        "manifest_audit": {
            "self_hashes": self_hash_audit,
            "protocol_frozen_files": protocol_audit,
        },
        "test_results": {"targeted": targeted, "full_suite": full},
        "governance": {
            "strategy_returns_calculated": False,
            "positions_generated": False,
            "current_signal_mapped": False,
            "target_shares_generated": False,
            "orders_generated": False,
            "broker_connection_performed": False,
            "rejected_model_parameter_rescue_performed": False,
            "parent_frozen_files_mutated": False,
        },
        "meaning": "冻结历史预测拒绝结果并停止；不是当前买卖指令。",
    }
    atomic_text(
        FINAL_REPORT_JSON, json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    atomic_text(FINAL_REPORT_MARKDOWN, render_report(report))
    frozen_inventory = {
        relative: {
            "sha256": sha256_file(ROOT / relative),
            "size_bytes": (ROOT / relative).stat().st_size,
        }
        for relative in FROZEN_EVIDENCE
    }
    manifest = {
        "project_id": report["project_id"],
        "version": report["version"],
        "status": report["status"],
        "frozen_at": report["generated_at"],
        "frozen_evidence": frozen_inventory,
        "final_reports": {
            "json": {
                "file": str(FINAL_REPORT_JSON.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(FINAL_REPORT_JSON),
            },
            "markdown": {
                "file": str(FINAL_REPORT_MARKDOWN.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(FINAL_REPORT_MARKDOWN),
            },
        },
        "branch_status": report["branch_status"],
        "test_results": report["test_results"],
        "governance": report["governance"],
        "meaning": report["meaning"],
    }
    manifest["manifest_content_sha256"] = canonical_hash(manifest)
    atomic_text(
        FINAL_MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "状态": report["status"],
                "分支状态": report["branch_status"],
                "独立复算": independent,
                "定向测试": targeted["stdout"],
                "全量测试": full["stdout"],
                "最终清单": str(FINAL_MANIFEST),
                "内容哈希": manifest["manifest_content_sha256"],
                "治理": report["governance"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
