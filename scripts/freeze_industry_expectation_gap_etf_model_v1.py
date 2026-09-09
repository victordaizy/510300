"""校验并冻结行业预期差—ETF 估值研究模型 V1。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from research.industry_expectation_gap_etf_model import (  # noqa: E402
    ResearchInputError,
    build_industry_map,
    build_research_result,
    evaluate_etf_snapshot,
    evaluate_market_state,
    select_sector_snapshot,
    validate_contract,
    validate_source_registry,
)


CONTRACT_RELATIVE_PATH = "config/industry_expectation_gap_etf_model_v1.yaml"
BASE_FROZEN_FILES = [
    "docs/INDUSTRY_EXPECTATION_GAP_ETF_MODEL_V1_SPEC.md",
    CONTRACT_RELATIVE_PATH,
    "data/reference/industry_expectation_gap_v1_sources.yaml",
    "paper/industry_expectation_gap_v1/2026-08-18_industry_forecast.json",
    "paper/industry_expectation_gap_v1/2026-08-18_market_state.json",
    "paper/industry_expectation_gap_v1/2026-08-18_etf_snapshot.json",
    "research/industry_expectation_gap_etf_model.py",
    "scripts/run_industry_expectation_gap_etf_model_v1.py",
    "scripts/freeze_industry_expectation_gap_etf_model_v1.py",
    "tests/test_industry_expectation_gap_etf_model.py",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ResearchInputError(f"YAML 顶层必须是对象：{path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ResearchInputError(f"JSON 顶层必须是对象：{path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _workspace_path(relative_path: str) -> Path:
    path = (WORKSPACE_ROOT / relative_path).resolve()
    try:
        path.relative_to(WORKSPACE_ROOT.resolve())
    except ValueError as exc:
        raise ResearchInputError(f"路径越出工作区：{relative_path}") from exc
    return path


def _git_text(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=WORKSPACE_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def main() -> int:
    contract = _load_yaml(_workspace_path(CONTRACT_RELATIVE_PATH))
    rules, safety = validate_contract(contract)
    inputs = {name: _workspace_path(value) for name, value in contract["inputs"].items()}
    outputs = {name: _workspace_path(value) for name, value in contract["outputs"].items()}
    sources = validate_source_registry(
        _load_yaml(inputs["source_registry"]),
        WORKSPACE_ROOT,
        contract["information_cutoff"],
    )

    panel = pd.read_parquet(inputs["sector_panel"])
    snapshot, sector_metadata = select_sector_snapshot(panel, contract["as_of_date"], rules)
    industry_frame, industry_summary = build_industry_map(
        _load_json(inputs["industry_forecast_ledger"]),
        snapshot,
        sector_metadata,
        sources,
        rules,
        contract["as_of_date"],
        contract["information_cutoff"],
    )
    market_summary = evaluate_market_state(
        _load_json(inputs["market_state_ledger"]),
        sources,
        contract["as_of_date"],
        contract["information_cutoff"],
    )
    etf_summary = evaluate_etf_snapshot(
        _load_json(inputs["etf_snapshot"]),
        sources,
        contract["as_of_date"],
        contract["information_cutoff"],
        rules,
    )
    preview = build_research_result(
        contract,
        industry_frame,
        industry_summary,
        market_summary,
        etf_summary,
        safety,
    )

    frozen_paths = list(BASE_FROZEN_FILES)
    frozen_paths.extend(str(value).replace("\\", "/") for value in contract["inputs"].values())
    for source in sources.values():
        if "path" in source:
            frozen_paths.append(str(source["path"]).replace("\\", "/"))
    frozen_paths = sorted(set(frozen_paths))

    frozen_files = []
    for relative_path in frozen_paths:
        path = _workspace_path(relative_path)
        if not path.is_file():
            raise ResearchInputError(f"待冻结文件不存在：{relative_path}")
        frozen_files.append(
            {
                "path": relative_path,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )

    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if now.date().isoformat() != contract["as_of_date"]:
        raise ResearchInputError(
            f"系统日期 {now.date().isoformat()} 与冻结日 {contract['as_of_date']} 不一致"
        )
    manifest = {
        "manifest_version": "INDUSTRY_EXPECTATION_GAP_ETF_MODEL_V1_MANIFEST",
        "status": "FROZEN_BEFORE_FORWARD_OUTCOME",
        "frozen_at": now.isoformat(timespec="seconds"),
        "as_of_date": contract["as_of_date"],
        "information_cutoff": contract["information_cutoff"],
        "first_forward_observation_date": contract["first_forward_observation_date"],
        "frozen_files": frozen_files,
        "validation": {
            "industry_count": len(industry_frame),
            "sector_weight_coverage": sector_metadata["weight_coverage"],
            "expectation_gap_observed_index_weight": industry_summary[
                "expectation_gap_observed_index_weight"
            ],
            "industry_aggregation_state": industry_summary["state"],
            "market_gate_state": market_summary["gate_state"],
            "etf_observation_state": etf_summary["observation_state"],
            "etf_predictive_state": etf_summary["predictive_state"],
            "preview_final_state": preview["final_state"],
        },
        "prohibitions": {
            "future_return_read": False,
            "historical_model_fitting": False,
            "failed_family_reuse": False,
            "proxy_identity_inference": False,
            "position_mapping": False,
            "order_generation": False,
            "broker_connection": False,
            "live_trading": False,
        },
        "git": {
            "commit": _git_text("rev-parse", "HEAD"),
            "branch": _git_text("branch", "--show-current"),
            "dirty_before_freeze": bool(_git_text("status", "--porcelain")),
        },
    }
    outputs["manifest"].parent.mkdir(parents=True, exist_ok=True)
    outputs["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"冻结完成：{outputs['manifest']}")
    print(f"冻结文件数：{len(frozen_files)}")
    print(f"行业聚合预览：{industry_summary['state']}")
    print(f"最终状态预览：{preview['final_state']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

