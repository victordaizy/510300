"""在首次读取策略收益前冻结跨ETF被迫资金流二元筛选V1。"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.cross_etf_forced_flow_binary_screen_v1 import (
    CANDIDATE_CONTRACT_PATH,
    CANDIDATE_CONTRACT_SHA256,
    CONFIG_PATH,
    ROOT,
    align_weekly_states_to_daily_targets,
    atomic_json,
    build_candidate_weekly_states,
    build_weekly_features,
    load_config,
    load_inputs,
    sha256_file,
)


def main() -> int:
    """核对全部实现和输入，通过测试后写入首跑冻结清单。"""

    config, contract = load_config(CONFIG_PATH)
    manifest_path = ROOT / config["artifacts"]["freeze_manifest"]
    output_paths = [
        ROOT / config["artifacts"][name]
        for name in ["weekly_features", "daily_states", "metrics", "report_json", "report_markdown"]
    ]
    existing_outputs = [path.relative_to(ROOT).as_posix() for path in output_paths if path.exists()]
    if existing_outputs:
        raise FileExistsError(f"首次筛选输出已经存在，拒绝重冻：{existing_outputs}")
    if manifest_path.exists():
        raise FileExistsError("冻结清单已经存在，拒绝覆盖")

    source_artifacts: dict[str, str] = {}
    for item in config["source_contracts"].values():
        path = ROOT / item["file"]
        actual = sha256_file(path)
        if actual != item["required_sha256"]:
            raise ValueError(f"源码合同哈希不匹配：{item['file']}")
        source_artifacts[item["file"]] = actual
    input_artifacts: dict[str, str] = {}
    for item in config["data_contracts"].values():
        path = ROOT / item["file"]
        actual = sha256_file(path)
        if actual != item["required_sha256"]:
            raise ValueError(f"输入合同哈希不匹配：{item['file']}")
        input_artifacts[item["file"]] = actual

    test_command = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        "-m",
        "pytest",
        str(ROOT / "tests" / "test_cross_etf_forced_flow_binary_screen_v1.py"),
        "-q",
    ]
    completed = subprocess.run(
        test_command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "冻结前测试失败：\n" + completed.stdout + "\n" + completed.stderr
        )

    inputs, _ = load_inputs(config, contract)
    features = build_weekly_features(
        inputs["shares"],
        inputs["prices"],
        broad_symbols=contract["universe_freeze"]["broad_pool_expected_symbols"],
        hs300_symbols=contract["universe_freeze"]["hs300_pool_symbols"],
        rolling_window=int(contract["features"]["rolling_percentile_window_weeks"]),
        minimum_observations=int(
            contract["features"]["rolling_percentile_minimum_weeks"]
        ),
    )
    states = build_candidate_weekly_states(features)
    daily = align_weekly_states_to_daily_targets(inputs["market"], states)
    if len(features) != int(config["data_contracts"]["official_weekly_shares"]["required_snapshot_count"]):
        raise ValueError("冻结前特征周数与官方快照数不一致")
    if not all(set(daily[item["candidate_id"]].unique()).issubset({0, 1}) for item in contract["candidates"]):
        raise ValueError("冻结前日度目标不是严格二元状态")

    manifest = {
        "status": "FROZEN_BEFORE_FIRST_SCREEN_RUN",
        "study_id": contract["protocol"]["study_id"],
        "version": contract["protocol"]["version"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_sha256": sha256_file(CONFIG_PATH),
        "candidate_contract_sha256": CANDIDATE_CONTRACT_SHA256,
        "candidate_contract_path": CANDIDATE_CONTRACT_PATH.relative_to(ROOT).as_posix(),
        "candidate_prefreeze_receipt_sha256": config["protocol"][
            "candidate_prefreeze_receipt_sha256"
        ],
        "source_artifacts": dict(sorted(source_artifacts.items())),
        "input_artifacts": dict(sorted(input_artifacts.items())),
        "preperformance_checks": {
            "pytest_return_code": int(completed.returncode),
            "pytest_stdout": completed.stdout.strip(),
            "weekly_feature_rows": int(len(features)),
            "daily_target_rows": int(len(daily)),
            "candidate_count": len(contract["candidates"]),
            "performance_evaluation_called": False,
        },
        "first_screen_command": (
            ".\\.venv\\Scripts\\python.exe "
            ".\\research\\cross_etf_forced_flow_binary_screen_v1.py"
        ),
        "boundaries": contract["boundaries"],
    }
    atomic_json(manifest, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
