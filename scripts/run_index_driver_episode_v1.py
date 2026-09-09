"""验证冻结清单后运行 510300 驱动周期历史发现 V1。"""

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

from research.index_driver_episode import (  # noqa: E402
    FORECAST_ELIGIBLE,
    EpisodeRules,
    EvaluationRules,
    OutcomeRules,
    build_driver_episode_structure,
    build_forward_outcomes,
    evaluate_episode_candidates,
)
from scripts.freeze_index_driver_episode_v1 import (  # noqa: E402
    CONTRACT_FILE,
    FROZEN_FILES,
    MANIFEST_FILE,
    PARENT_MANIFEST_FILE,
    sha256,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is pd.NaT or (not isinstance(value, (str, bytes)) and pd.isna(value)):
        return None
    return value


def _verify_manifest(contract: dict) -> dict:
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError("驱动周期 V1 尚未冻结；禁止读取历史结果")
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    changed: list[str] = []
    for relative, expected in manifest["frozen_files"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            changed.append(relative)
    for relative, expected in manifest["input_files"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            changed.append(relative)
    parent = manifest["parent_attribution_manifest"]
    if (
        not PARENT_MANIFEST_FILE.exists()
        or sha256(PARENT_MANIFEST_FILE) != parent["sha256"]
    ):
        changed.append(parent["path"])
    if changed:
        raise RuntimeError(f"驱动周期冻结指纹变化：{sorted(set(changed))}")
    if set(manifest["frozen_files"]) != set(FROZEN_FILES):
        raise RuntimeError("冻结文件集合与实现声明不一致")
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
    fish = evaluation["fish_middle"]
    tail = evaluation["tail_warning"]

    def gate_lines(candidate: dict[str, Any]) -> list[str]:
        return [
            f"- {'PASS' if passed else 'FAIL'} `{name}`"
            for name, passed in candidate["gates"].items()
        ]

    lines = [
        "# 510300 驱动周期历史发现 V1 结果",
        "",
        f"- 总状态：`{report['status']}`",
        f"- 数据截止：`{report['data_cutoff']}`",
        "- 历史标签：`HISTORICALLY_CONTAMINATED`",
        "- 预测资格：`NOT_AUTHORIZED`",
        "- 仓位、订单、券商：`DISABLED`",
        "",
        "## 周期结构",
        "",
        f"- 归因有效日：{report['row_counts']['valid_attribution_days']}",
        f"- NO_VIEW 日：{report['row_counts']['no_view_days']}",
        f"- 驱动周期：{report['row_counts']['episodes']}",
        f"- M1 成熟事件：{fish['mature_event_count']}",
        f"- T1 成熟事件：{tail['mature_event_count']}",
        "",
        "## M1 鱼中候选",
        "",
        f"- 状态：`{fish['status']}`",
        f"- MIDDLE20 Lift：{fish['lift']}",
        f"- Bootstrap 95% Lift 区间：{fish['bootstrap']['lift_interval_95pct']}",
        f"- X20 中位数：{fish['median_x20']}",
        f"- TAIL20 发生率：{fish['tail20_rate']}；基准：{fish['baseline_tail20_rate']}",
        *gate_lines(fish),
        "",
        "## T1 鱼尾候选",
        "",
        f"- 状态：`{tail['status']}`",
        f"- TAIL20 Lift：{tail['lift']}",
        f"- Bootstrap 95% Lift 区间：{tail['bootstrap']['lift_interval_95pct']}",
        *gate_lines(tail),
        "",
        "## 解释边界",
        "",
        "历史通过只允许进入真正前向准备；历史失败则冻结失败。任何结果都不生成仓位、订单或券商动作。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    contract = yaml.safe_load(CONTRACT_FILE.read_text(encoding="utf-8"))
    manifest = _verify_manifest(contract)
    inputs = contract["inputs"]
    attribution_status = json.loads(
        (ROOT / inputs["attribution_status"]).read_text(encoding="utf-8")
    )
    if not str(attribution_status.get("status", "")).startswith("PASS_ATTRIBUTION"):
        raise RuntimeError("父归因数据没有通过 V1.3 数据门槛")

    cross = pd.read_parquet(ROOT / inputs["cross_section"])
    industry = pd.read_parquet(ROOT / inputs["industry_attribution"])
    daily = pd.read_parquet(ROOT / inputs["daily_attribution"])
    etf_daily = pd.read_parquet(ROOT / inputs["etf_daily"])
    dividends = pd.read_csv(ROOT / inputs["etf_dividends"])

    episode_rules = EpisodeRules.from_contract(contract)
    outcome_rules = OutcomeRules.from_contract(contract)
    evaluation_rules = EvaluationRules.from_contract(contract)
    structure = build_driver_episode_structure(
        cross, industry, daily, episode_rules
    )
    outcomes = build_forward_outcomes(etf_daily, dividends, outcome_rules)
    evaluation = evaluate_episode_candidates(
        structure, outcomes, evaluation_rules
    )
    if structure.daily_states["research_output"].eq(FORECAST_ELIGIBLE).any():
        raise AssertionError("历史运行非法输出 FORECAST_ELIGIBLE")

    events_with_outcomes = structure.events.merge(
        outcomes, left_on="event_date", right_on="signal_date", how="left"
    )
    output_contract = contract["outputs"]
    output_frames = {
        output_contract["daily_states"]: structure.daily_states,
        output_contract["episodes"]: structure.episodes,
        output_contract["events"]: events_with_outcomes,
        output_contract["outcomes"]: outcomes,
    }
    for relative, frame in output_frames.items():
        _atomic_parquet(frame, ROOT / relative)

    state_counts = structure.daily_states["research_output"].value_counts().to_dict()
    phase_counts = (
        structure.daily_states["phase"].dropna().value_counts().to_dict()
        if "phase" in structure.daily_states
        else {}
    )
    termination_counts = (
        structure.episodes["termination_reason"].value_counts().to_dict()
        if not structure.episodes.empty
        else {}
    )
    latest = structure.daily_states.iloc[-1].to_dict()
    report = _json_safe(
        {
            "project_id": contract["protocol"]["project_id"],
            "version": contract["protocol"]["version"],
            "status": evaluation["overall_status"],
            "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "data_cutoff": str(episode_rules.data_cutoff.date()),
            "manifest_sha256": sha256(MANIFEST_FILE),
            "parent_attribution_manifest_sha256": sha256(PARENT_MANIFEST_FILE),
            "row_counts": {
                "daily_states": len(structure.daily_states),
                "valid_attribution_days": int(
                    structure.daily_states["research_output"].ne("NO_VIEW").sum()
                ),
                "no_view_days": int(
                    structure.daily_states["research_output"].eq("NO_VIEW").sum()
                ),
                "episodes": len(structure.episodes),
                "events": len(structure.events),
                "mature_outcome_days": len(outcomes),
            },
            "state_counts": state_counts,
            "phase_counts": phase_counts,
            "termination_counts": termination_counts,
            "latest_state": latest,
            "evaluation": evaluation,
            "governance": contract["governance"],
            "frozen_at": manifest["frozen_at"],
            "output_hashes": {
                relative: sha256(ROOT / relative) for relative in output_frames
            },
        }
    )
    json_path = ROOT / output_contract["result_json"]
    markdown_path = ROOT / output_contract["result_markdown"]
    _atomic_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        json_path,
    )
    _atomic_text(_render_markdown(report), markdown_path)
    print(
        json.dumps(
            {
                "状态": report["status"],
                "周期数": report["row_counts"]["episodes"],
                "M1": report["evaluation"]["fish_middle"]["status"],
                "T1": report["evaluation"]["tail_warning"]["status"],
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
