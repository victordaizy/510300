"""冻结510300“可预测才进入”最终决策层。"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.predictable_fish_middle_decision_v1 import (  # noqa: E402
    sha256,
    validate_parent_hashes,
)


CONFIG_PATH = ROOT / "config" / "510300_predictable_fish_middle_decision_v1.yaml"


def hash_map(paths: list[Path]) -> dict[str, str]:
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths}


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    actual_parent_hashes = validate_parent_hashes(ROOT, config)
    protocol_files = [
        CONFIG_PATH,
        ROOT / "docs" / "510300_PREDICTABLE_FISH_MIDDLE_DECISION_V1_SPEC.md",
        ROOT / "research" / "predictable_fish_middle_decision_v1.py",
        ROOT / "scripts" / "build_510300_predictable_fish_middle_decision_v1.py",
        ROOT / "scripts" / "freeze_510300_predictable_fish_middle_decision_v1.py",
        ROOT / "tests" / "test_510300_predictable_fish_middle_decision_v1.py",
    ]
    parent_files = [ROOT / value for value in config["parents"].values()]
    output_files = [
        ROOT / value
        for key, value in config["outputs"].items()
        if key != "manifest"
    ]
    all_files = protocol_files + parent_files + output_files
    missing = [path.relative_to(ROOT).as_posix() for path in all_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"最终决策层冻结缺少文件：{missing}")

    report_path = ROOT / config["outputs"]["json"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["completion_state"] != "MODEL_LOGIC_COMPLETE_CURRENT_NO_ENTRY":
        raise RuntimeError("最终决策层没有完成当前闭环")
    if report["final_decision"]["current_entry_state"] != "WAIT_NO_PREDICTIVE_ENTRY":
        raise RuntimeError("当前进入状态与冻结证据不一致")
    if report["final_decision"]["current_action"] != "WAIT_NO_NEW_ENTRY":
        raise RuntimeError("当前动作与冻结证据不一致")
    if report["final_decision"]["all_research_entry_gates_passed"] is not False:
        raise RuntimeError("当前不应通过全部研究进入门")
    if report["safety"]["synthetic_numeric_score"] is not None:
        raise RuntimeError("最终决策层违规生成合成数值分数")
    if report["statistics_semantics"]["score_conditioned_win_rate"] != (
        "UNAVAILABLE_AWAITING_TRUE_FORWARD"
    ):
        raise RuntimeError("最终决策层违规生成条件胜率")
    for switch in (
        "may_call_predictive_edge",
        "current_holdings_read_enabled",
        "position_mapping_enabled",
        "order_generation_enabled",
        "broker_connection_enabled",
        "live_trading_enabled",
    ):
        if report["safety"][switch] is not False:
            raise RuntimeError(f"最终决策层安全开关违规：{switch}")

    manifest = {
        "version": config["version"],
        "status": "FROZEN_RESEARCH_ONLY_CURRENT_WAIT",
        "as_of_date": config["as_of_date"],
        "information_cutoff": config["information_cutoff"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_files": hash_map(protocol_files),
        "parent_files": hash_map(parent_files),
        "output_files": hash_map(output_files),
        "parent_integrity": {
            key: {
                "expected": config["parent_integrity"][key],
                "actual": actual_parent_hashes[key],
            }
            for key in config["parents"]
        },
        "current_frozen_conclusion": {
            "research_view": report["final_decision"]["current_research_view"],
            "entry_state": report["final_decision"]["current_entry_state"],
            "action": report["final_decision"]["current_action"],
            "positive_structure_weight": report["index_sector_structure"][
                "positive_structure_weight"
            ],
            "failed_gates": [
                blocker["gate_id"]
                for blocker in report["final_decision"]["blockers"]
            ],
        },
        "invariants": {
            "synthetic_numeric_score_allowed": False,
            "qualitative_direction_is_probability": False,
            "score_conditioned_win_rate": "UNAVAILABLE_AWAITING_TRUE_FORWARD",
            "rejected_models_may_be_rescued": False,
            "may_call_predictive_edge": False,
            "current_holdings_read_enabled": False,
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
            "live_trading_enabled": False,
        },
    }
    manifest_path = ROOT / config["outputs"]["manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(manifest_path)
    print(
        json.dumps(
            {
                "manifest": manifest_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(manifest_path),
                "status": manifest["status"],
                "entry_state": manifest["current_frozen_conclusion"]["entry_state"],
                "action": manifest["current_frozen_conclusion"]["action"],
                "parent_files_unchanged": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

