"""冻结510300北向旧口径流量二元筛选的协议、程序、测试、依赖与输入。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "510300_northbound_legacy_flow_binary_screen_v1.yaml"
SOURCE_PATHS = [
    ROOT / "research" / "northbound_legacy_flow_binary_screen_v1.py",
    ROOT / "research" / "intraday_binary_livermore_screen_v1.py",
    ROOT / "scripts" / "download_northbound_legacy_net_flow_v1.py",
    ROOT / "scripts" / "freeze_510300_northbound_legacy_flow_binary_screen_v1.py",
    ROOT / "tests" / "test_northbound_legacy_flow_binary_screen_v1.py",
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
    protocol = config["protocol"]
    if protocol["study_id"] != "510300_NORTHBOUND_LEGACY_FLOW_BINARY_SCREEN_V1":
        raise ValueError("研究编号不匹配")
    if protocol["state"] != "PREFREEZE_IMPLEMENTATION_COMPLETE":
        raise ValueError("协议尚未达到冻结前实现完成状态")
    if not bool(protocol["no_parameter_rescue"]):
        raise ValueError("协议必须禁止结果后参数营救")
    candidate_ids = [candidate["candidate_id"] for candidate in config["candidates"]]
    if len(candidate_ids) != 6 or len(set(candidate_ids)) != 6:
        raise ValueError("冻结候选必须是六个唯一规则")
    if list(config["scope"]["allowed_target_states"]) != [0, 1]:
        raise ValueError("目标状态必须严格为0和1")
    if float(config["objective"]["minimum_annualized_excess"]) != 0.20:
        raise ValueError("年化净超额硬门必须为20%")
    if float(config["objective"]["minimum_rolling_excess_median"]) != 0.20:
        raise ValueError("滚动净超额硬门必须为20%")

    manifest_path = ROOT / config["artifacts"]["freeze_manifest"]
    report_path = ROOT / config["artifacts"]["report_json"]
    metrics_path = ROOT / config["artifacts"]["metrics"]
    state_path = ROOT / config["artifacts"]["daily_states"]
    if manifest_path.exists():
        raise FileExistsError("冻结清单已存在，禁止覆盖")
    for path in [report_path, metrics_path, state_path]:
        if path.exists():
            raise FileExistsError(f"绩效结果已存在，不能倒序冻结：{path}")
    for path in SOURCE_PATHS:
        if not path.exists():
            raise FileNotFoundError(f"冻结源文件不存在：{path}")

    source_artifacts = {
        path.relative_to(ROOT).as_posix(): sha256_file(path) for path in SOURCE_PATHS
    }
    source_contract_names = {
        "acquisition_implementation",
        "execution_ledger_dependency",
    }
    input_artifacts: dict[str, str] = {}
    for name, contract in config["data_contracts"].items():
        path = ROOT / contract["file"]
        if not path.exists():
            raise FileNotFoundError(f"冻结工件不存在：{path}")
        actual = sha256_file(path)
        if actual != contract["required_sha256"]:
            raise ValueError(
                f"配置固定哈希错误：{name}，期望 {contract['required_sha256']}，实际 {actual}"
            )
        relative = path.relative_to(ROOT).as_posix()
        if name in source_contract_names:
            if source_artifacts[relative] != actual:
                raise AssertionError("源依赖哈希核对不一致")
        else:
            input_artifacts[relative] = actual

    manifest = {
        "study_id": protocol["study_id"],
        "version": protocol["version"],
        "status": "FROZEN_BEFORE_FIRST_SCREEN_RUN",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_path": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha256_file(CONFIG_PATH),
        "source_artifacts": source_artifacts,
        "input_artifacts": input_artifacts,
        "candidate_ids": candidate_ids,
        "candidate_count": len(candidate_ids),
        "allowed_target_states": list(config["scope"]["allowed_target_states"]),
        "signal_information_cutoff": config["timing"]["signal_information_cutoff"],
        "execution_time": config["timing"]["execution_time"],
        "cost_scenarios": list(config["evaluation"]["cost_scenarios"]),
        "start_offsets": list(
            config["evaluation"]["start_date_perturbations_trading_days"]
        ),
        "period_ids": [period["id"] for period in config["evaluation"]["periods"]],
        "hard_gates": {
            "minimum_annualized_excess": float(
                config["objective"]["minimum_annualized_excess"]
            ),
            "minimum_rolling_242d_excess_median": float(
                config["objective"]["minimum_rolling_excess_median"]
            ),
            "every_stress_period_and_start_required": True,
        },
        "post_run_parameter_sign_or_window_rescue": "FORBIDDEN",
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
