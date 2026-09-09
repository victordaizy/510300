"""冻结510300 IF真实期限结构二元筛选V1.0.1的协议、程序、测试与输入。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "510300_if_true_term_structure_binary_screen_v1_0_1.yaml"
CANDIDATE_CONTRACT_PATH = (
    ROOT
    / "config"
    / "510300_if_true_term_structure_binary_screen_v1_0_1_candidates.yaml"
)
EXPECTED_CANDIDATE_SHA256 = (
    "0d406748722608296935ec7b398e9e4a1fab153b6d411dd83fb70bbe654fa4c1"
)
SOURCE_PATHS = [
    ROOT / "research" / "if_true_term_structure_binary_screen_v1_0_1.py",
    ROOT / "research" / "etf_share_premium_level_binary_screen_v1.py",
    ROOT / "research" / "intraday_binary_livermore_screen_v1.py",
    ROOT / "scripts" / "download_cffex_if_true_term_structure_inputs_v1_0_1.py",
    ROOT / "scripts" / "freeze_510300_if_true_term_structure_binary_screen_v1_0_1.py",
    ROOT / "tests" / "test_if_true_term_structure_binary_screen_v1_0_1.py",
]


def sha256_file(path: Path) -> str:
    """流式计算文件SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    """仅在绩效结果不存在时生成一次性冻结清单。"""

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    contract = yaml.safe_load(CANDIDATE_CONTRACT_PATH.read_text(encoding="utf-8"))
    protocol = config["protocol"]
    if protocol["study_id"] != "510300_IF_TRUE_TERM_STRUCTURE_BINARY_SCREEN_V1_0_1":
        raise ValueError("主协议研究编号不匹配")
    if protocol["state"] != "PREFREEZE_IMPLEMENTATION_COMPLETE":
        raise ValueError("主协议尚未达到冻结前实现完成状态")
    if not bool(protocol["no_parameter_rescue"]):
        raise ValueError("主协议必须禁止结果后参数营救")
    candidate_hash = sha256_file(CANDIDATE_CONTRACT_PATH)
    if candidate_hash != EXPECTED_CANDIDATE_SHA256:
        raise ValueError("正式全历史获取前固定的候选合同发生漂移")
    if protocol["candidate_contract_sha256"] != candidate_hash:
        raise ValueError("主协议记录的候选合同哈希不匹配")
    candidate_ids = [item["candidate_id"] for item in contract["candidates"]]
    if len(candidate_ids) != 8 or len(set(candidate_ids)) != 8:
        raise ValueError("冻结候选必须是八个唯一规则")
    if list(contract["scope"]["allowed_target_states"]) != [0, 1]:
        raise ValueError("目标状态必须严格为0和1")
    if float(contract["objective"]["minimum_annualized_excess"]) != 0.20:
        raise ValueError("年化净超额硬门必须为20%")
    if float(contract["objective"]["minimum_rolling_excess_median"]) != 0.20:
        raise ValueError("滚动净超额硬门必须为20%")

    manifest_path = ROOT / config["artifacts"]["freeze_manifest"]
    result_paths = [
        ROOT / config["artifacts"]["daily_features"],
        ROOT / config["artifacts"]["daily_states"],
        ROOT / config["artifacts"]["metrics"],
        ROOT / config["artifacts"]["report_json"],
    ]
    if manifest_path.exists():
        raise FileExistsError("冻结清单已存在，禁止覆盖")
    for path in result_paths:
        if path.exists():
            raise FileExistsError(f"绩效结果已存在，不能倒序冻结：{path}")
    for path in SOURCE_PATHS:
        if not path.exists():
            raise FileNotFoundError(f"冻结源文件不存在：{path}")

    source_artifacts = {
        path.relative_to(ROOT).as_posix(): sha256_file(path) for path in SOURCE_PATHS
    }
    for name, source_contract in config["source_contracts"].items():
        path = ROOT / source_contract["file"]
        actual = sha256_file(path)
        if actual != source_contract["required_sha256"]:
            raise ValueError(
                f"源依赖固定哈希错误：{name}，期望 {source_contract['required_sha256']}，实际 {actual}"
            )
        if source_artifacts[path.relative_to(ROOT).as_posix()] != actual:
            raise AssertionError("源依赖哈希核对不一致")

    input_artifacts: dict[str, str] = {}
    for name, data_contract in config["data_contracts"].items():
        path = ROOT / data_contract["file"]
        if not path.exists():
            raise FileNotFoundError(f"冻结输入不存在：{path}")
        actual = sha256_file(path)
        if actual != data_contract["required_sha256"]:
            raise ValueError(
                f"输入固定哈希错误：{name}，期望 {data_contract['required_sha256']}，实际 {actual}"
            )
        input_artifacts[path.relative_to(ROOT).as_posix()] = actual

    manifest = {
        "study_id": protocol["study_id"],
        "version": protocol["version"],
        "status": "FROZEN_BEFORE_FIRST_SCREEN_RUN",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_path": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha256_file(CONFIG_PATH),
        "candidate_contract_path": CANDIDATE_CONTRACT_PATH.relative_to(ROOT).as_posix(),
        "candidate_contract_sha256": candidate_hash,
        "source_artifacts": source_artifacts,
        "input_artifacts": input_artifacts,
        "candidate_ids": candidate_ids,
        "candidate_count": len(candidate_ids),
        "allowed_target_states": list(contract["scope"]["allowed_target_states"]),
        "signal_cutoff": contract["information_clock"]["signal_cutoff"],
        "execution_time": contract["information_clock"]["execution_time"],
        "same_day_execution_forbidden": bool(
            contract["information_clock"]["same_day_execution_forbidden"]
        ),
        "cost_scenarios": list(contract["evaluation"]["cost_scenarios"]),
        "start_offsets": list(
            contract["evaluation"]["start_date_perturbations_trading_days"]
        ),
        "period_ids": [
            period["id"] for period in contract["evaluation"]["periods"]
        ],
        "hard_gates": {
            "minimum_annualized_excess": float(
                contract["objective"]["minimum_annualized_excess"]
            ),
            "minimum_rolling_242d_excess_median": float(
                contract["objective"]["minimum_rolling_excess_median"]
            ),
            "every_stress_period_and_start_required": True,
        },
        "post_run_tail_percentile_dte_or_contract_selection_rescue": "FORBIDDEN",
        "position_mapping": "DISABLED",
        "live_trading_authorized": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest": manifest_path.relative_to(ROOT).as_posix(),
                "manifest_sha256": sha256_file(manifest_path),
                "protocol_sha256": manifest["protocol_sha256"],
                "candidate_contract_sha256": candidate_hash,
                "candidate_count": manifest["candidate_count"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
