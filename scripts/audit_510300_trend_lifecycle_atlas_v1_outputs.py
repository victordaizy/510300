"""聚焦复核趋势生命周期图谱的隔离、事件、领先门和裁决。"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from trend_lifecycle_atlas_v1 import (  # noqa: E402
    build_dual_label_candidates,
    evaluate_leading_results,
    load_config,
    project_path,
    sha256_file,
    validate_manifest,
)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str)
        + "\n",
    )


def _same_boolean_frame(
    stored: pd.DataFrame,
    recomputed: pd.DataFrame,
    keys: list[str],
    boolean_columns: list[str],
) -> bool:
    left = stored[keys + boolean_columns].copy().sort_values(keys).reset_index(drop=True)
    right = (
        recomputed[keys + boolean_columns]
        .copy()
        .sort_values(keys)
        .reset_index(drop=True)
    )
    if len(left) != len(right) or not left[keys].equals(right[keys]):
        return False
    return all(
        left[column].astype(bool).equals(right[column].astype(bool))
        for column in boolean_columns
    )


def audit() -> dict[str, Any]:
    config = load_config()
    manifest = validate_manifest(config)
    paths = {key: project_path(value) for key, value in config["paths"].items()}
    required_outputs = [
        "feature_panel",
        "outcome_label_panel",
        "label_a_events",
        "label_b_events",
        "event_catalog",
        "event_trajectories",
        "trajectory_summary",
        "leading_results",
        "dual_label_candidates",
        "lifecycle_reference_panel",
        "result_json",
        "result_markdown",
    ]
    missing = [key for key in required_outputs if not paths[key].exists()]
    if missing:
        raise FileNotFoundError(f"缺少正式输出：{missing}")

    feature = pd.read_parquet(paths["feature_panel"])
    outcome = pd.read_parquet(paths["outcome_label_panel"])
    label_a = pd.read_csv(paths["label_a_events"], parse_dates=["event_date", "confirmation_date"])
    label_b = pd.read_csv(
        paths["label_b_events"],
        parse_dates=["event_date", "episode_start_date", "episode_end_date"],
    )
    catalog = pd.read_csv(paths["event_catalog"], parse_dates=["event_date"])
    trajectories = pd.read_parquet(paths["event_trajectories"])
    leading = pd.read_csv(paths["leading_results"])
    candidates = pd.read_csv(paths["dual_label_candidates"])
    reference = pd.read_parquet(paths["lifecycle_reference_panel"])
    report = json.loads(paths["result_json"].read_text(encoding="utf-8"))

    prohibited_feature_columns = [
        column
        for column in feature.columns
        if column.lower().startswith(("future_", "forward_"))
        or "target" in column.lower()
    ]
    recomputed_leading = evaluate_leading_results(config, trajectories)
    recomputed_candidates, recomputed_structure = build_dual_label_candidates(
        config, recomputed_leading
    )
    leading_matches = _same_boolean_frame(
        leading,
        recomputed_leading,
        ["label_system", "event_type", "feature"],
        ["passed"],
    )
    candidate_matches = _same_boolean_frame(
        candidates,
        recomputed_candidates,
        ["event_type", "feature"],
        [
            "label_a_passed",
            "label_b_passed",
            "equal_weight_sensitivity_passed",
            "dual_label_candidate_passed",
        ],
    )
    start = int(config["event_alignment"]["full_window_start"])
    end = int(config["event_alignment"]["full_window_end"])
    expected_rows = end - start + 1
    trajectory_shape_ok = True
    if not trajectories.empty:
        trajectory_shape_ok = bool(
            trajectories.groupby(["event_id", "feature"])
            .size()
            .eq(expected_rows)
            .all()
            and set(trajectories["relative_trading_day"].unique())
            == set(range(start, end + 1))
        )
    allowed_states = set(config["scope"]["internal_states"])
    checks = {
        "manifest_hashes_current": manifest["state"]
        == "FROZEN_BEFORE_FIRST_NEW_LIFECYCLE_OUTCOME_READ",
        "feature_panel_unique_increasing_dates": bool(
            not feature["date"].duplicated().any()
            and feature["date"].is_monotonic_increasing
        ),
        "feature_panel_has_no_future_or_target_columns": not prohibited_feature_columns,
        "outcome_panel_is_physically_separate": bool(
            "future_path_label" in outcome.columns
            and "future_path_label" not in feature.columns
        ),
        "outcome_right_censoring_exact": bool(
            outcome.tail(60)["future_path_label"].eq("CENSORED_FUTURE_60D").all()
        ),
        "label_a_pivot_confirmation_order": bool(
            (label_a["confirmation_date"] >= label_a["event_date"]).all()
        ),
        "event_ids_unique": bool(catalog["event_id"].is_unique),
        "catalog_count_matches_report": int(len(catalog))
        == int(report["events"]["catalog_count"]),
        "label_counts_match_report": bool(
            len(label_a) == report["events"]["label_a_count"]
            and len(label_b) == report["events"]["label_b_count"]
        ),
        "trajectory_shape": trajectory_shape_ok,
        "leading_gate_recomputed": leading_matches,
        "dual_label_candidates_recomputed": candidate_matches,
        "structure_gate_recomputed": recomputed_structure
        == report["structure_gate"],
        "reference_states_allowed": set(reference["reference_state"].unique()).issubset(
            allowed_states
        ),
        "reference_state_is_explicitly_retrospective": reference[
            "reference_semantics"
        ].eq("RETROSPECTIVE_REFERENCE_ONLY_NOT_A_POINT_IN_TIME_SIGNAL").all(),
        "source_provenance_not_promoted": report[
            "source_provenance_gate_for_phase_2"
        ]["passed"]
        is False,
        "no_strategy_metrics_computed": all(
            value == "NOT_COMPUTED"
            for value in report["portfolio_metrics"].values()
        ),
        "no_execution_authorization": bool(
            report["boundaries"]["portfolio_mapping"] == "DISABLED"
            and report["boundaries"]["paper_signal"] == "DISABLED"
            and report["boundaries"]["shadow_signal"] == "DISABLED"
            and report["boundaries"]["order_generation"] == "DISABLED"
            and report["boundaries"]["broker_connection"] == "DISABLED"
            and report["boundaries"]["live_trading_authorized"] is False
        ),
    }
    passed = all(bool(value) for value in checks.values())
    output_hashes = {
        config["paths"][key]: sha256_file(paths[key]) for key in required_outputs
    }
    payload = {
        "project_id": "510300_TREND_LIFECYCLE_ATLAS_V1",
        "status": (
            "PASS_FOCUSED_TREND_LIFECYCLE_OUTPUT_RECOMPUTATION"
            if passed
            else "FAIL_TREND_LIFECYCLE_OUTPUT_RECOMPUTATION"
        ),
        "generated_at_asia_shanghai": datetime.now(
            ZoneInfo("Asia/Shanghai")
        ).isoformat(),
        "passed": passed,
        "checks": checks,
        "prohibited_feature_columns": prohibited_feature_columns,
        "output_hashes": output_hashes,
        "boundaries": {
            "generic_security_scan_run": False,
            "strategy_backtest": "NOT_RUN",
            "position_mapping": "DISABLED",
            "live_trading_authorized": False,
        },
    }
    markdown = "\n".join(
        [
            "# 510300 趋势生命周期图谱 V1 聚焦输出审计",
            "",
            f"- 状态：`{payload['status']}`",
            f"- 聚焦检查：{sum(bool(value) for value in checks.values())}/{len(checks)}通过。",
            "- 已复算：未来字段隔离、右删失、事件计数、轨迹形状、领先门、双标签候选、七状态语义和执行边界。",
            "- 未运行与本研究无关的通用安全扫描。",
            "",
        ]
    )
    _atomic_json(paths["output_audit_json"], payload)
    _atomic_text(paths["output_audit_markdown"], markdown)
    return payload


def main() -> int:
    payload = audit()
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
