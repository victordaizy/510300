"""在首次候选收益计算前冻结510300日内二元状态筛选 V1。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "510300_intraday_binary_livermore_screen_v1.yaml"
EVALUATOR_PATH = ROOT / "research" / "intraday_binary_livermore_screen_v1.py"
FREEZER_PATH = Path(__file__).resolve()
TEST_PATH = ROOT / "tests" / "test_intraday_binary_livermore_screen_v1.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("配置文件不是映射结构")
    return config


def freeze(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = _load_config(config_path)
    manifest_path = ROOT / config["artifacts"]["freeze_manifest"]
    if manifest_path.exists():
        raise FileExistsError(f"冻结清单已经存在，禁止覆盖：{manifest_path}")

    required_sources = [config_path, EVALUATOR_PATH, FREEZER_PATH, TEST_PATH]
    missing_sources = [str(path) for path in required_sources if not path.exists()]
    if missing_sources:
        raise FileNotFoundError(f"冻结所需源文件缺失：{missing_sources}")

    artifact_keys = [
        "report_json",
        "report_markdown",
        "candidate_summary_parquet",
        "period_results_parquet",
        "top_candidate_daily_ledger_parquet",
        "top_candidate_trades_parquet",
    ]
    artifact_absence = {
        config["artifacts"][key]: not (ROOT / config["artifacts"][key]).exists()
        for key in artifact_keys
    }
    if not all(artifact_absence.values()):
        present = [path for path, absent in artifact_absence.items() if not absent]
        raise FileExistsError(f"首次冻结前已经存在候选收益产物：{present}")

    input_entries: dict[str, dict[str, Any]] = {}
    for contract_name, contract in config["data_contracts"].items():
        relative = contract["file"]
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(f"冻结输入缺失：{relative}")
        actual_hash = sha256_file(path)
        expected_hash = contract.get("required_sha256")
        if expected_hash is not None and actual_hash != expected_hash:
            raise ValueError(f"冻结输入哈希不匹配：{relative}")
        input_entries[relative] = {
            "contract_name": contract_name,
            "sha256": actual_hash,
            "size_bytes": path.stat().st_size,
        }

    tracked_files = {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
        for path in required_sources
    }
    candidate_snapshot = config["candidate_family"]["candidates"]
    manifest = {
        "study_id": config["protocol"]["study_id"],
        "version": config["protocol"]["version"],
        "state": "FROZEN_BEFORE_FIRST_CANDIDATE_PERFORMANCE_EVALUATION",
        "created_at": datetime.now().astimezone().isoformat(),
        "historical_evidence_label": config["protocol"]["historical_evidence_label"],
        "tracked_files": tracked_files,
        "historical_inputs": input_entries,
        "candidate_count": len(candidate_snapshot),
        "candidate_snapshot": candidate_snapshot,
        "objective_snapshot": config["objective"],
        "cost_snapshot": config["costs"],
        "evaluation_snapshot": config["evaluation"],
        "selection_snapshot": config["selection"],
        "pre_evaluation_artifact_absence": artifact_absence,
        "candidate_features_or_returns_read_by_freeze_script": False,
        "parameter_rescue_after_freeze_allowed": False,
        "paper_or_shadow_authorized": False,
        "live_trading_authorized": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "状态": manifest["state"],
                "候选数": manifest["candidate_count"],
                "冻结清单": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
                "冻结清单SHA256": sha256_file(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结510300日内二元状态筛选V1")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    freeze(args.config.resolve())


if __name__ == "__main__":
    main()
