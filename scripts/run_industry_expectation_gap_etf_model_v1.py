"""运行行业预期差—ETF 估值研究模型 V1。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

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
    render_markdown,
    select_sector_snapshot,
    validate_contract,
    validate_source_registry,
)


CONTRACT_PATH = WORKSPACE_ROOT / "config" / "industry_expectation_gap_etf_model_v1.yaml"


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


def _verify_frozen_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise ResearchInputError(f"冻结清单不存在，请先运行冻结脚本：{manifest_path}")
    manifest = _load_json(manifest_path)
    if manifest.get("status") != "FROZEN_BEFORE_FORWARD_OUTCOME":
        raise ResearchInputError("冻结清单状态非法")
    for item in manifest.get("frozen_files", []):
        path = _workspace_path(str(item["path"]))
        if not path.is_file():
            raise ResearchInputError(f"冻结文件已缺失：{item['path']}")
        actual = _sha256(path)
        if actual != item["sha256"]:
            raise ResearchInputError(f"冻结文件哈希不一致：{item['path']}")
    return manifest


def main() -> int:
    contract = _load_yaml(CONTRACT_PATH)
    rules, safety = validate_contract(contract)
    input_paths = {name: _workspace_path(value) for name, value in contract["inputs"].items()}
    output_paths = {name: _workspace_path(value) for name, value in contract["outputs"].items()}

    manifest = _verify_frozen_manifest(output_paths["manifest"])
    source_registry = _load_yaml(input_paths["source_registry"])
    sources = validate_source_registry(
        source_registry,
        WORKSPACE_ROOT,
        contract["information_cutoff"],
    )

    sector_panel = pd.read_parquet(input_paths["sector_panel"])
    sector_snapshot, sector_metadata = select_sector_snapshot(
        sector_panel,
        contract["as_of_date"],
        rules,
    )
    industry_frame, industry_summary = build_industry_map(
        _load_json(input_paths["industry_forecast_ledger"]),
        sector_snapshot,
        sector_metadata,
        sources,
        rules,
        contract["as_of_date"],
        contract["information_cutoff"],
    )
    market_summary = evaluate_market_state(
        _load_json(input_paths["market_state_ledger"]),
        sources,
        contract["as_of_date"],
        contract["information_cutoff"],
    )
    etf_summary = evaluate_etf_snapshot(
        _load_json(input_paths["etf_snapshot"]),
        sources,
        contract["as_of_date"],
        contract["information_cutoff"],
        rules,
    )
    result = build_research_result(
        contract,
        industry_frame,
        industry_summary,
        market_summary,
        etf_summary,
        safety,
    )
    result["provenance"] = {
        "manifest_path": output_paths["manifest"].relative_to(WORKSPACE_ROOT).as_posix(),
        "manifest_sha256": _sha256(output_paths["manifest"]),
        "frozen_at": manifest["frozen_at"],
        "frozen_file_count": len(manifest["frozen_files"]),
        "future_return_read": False,
        "model_fitting_performed": False,
    }

    output_paths["json"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["markdown"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["json"].write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_paths["markdown"].write_text(render_markdown(result), encoding="utf-8")

    print(f"研究状态：{result['final_state']}")
    print(f"行业聚合：{industry_summary['state']}")
    print(f"市场门控：{market_summary['gate_state']}")
    print(f"ETF前瞻成熟度：{etf_summary['predictive_state']}")
    print(f"JSON输出：{output_paths['json']}")
    print(f"Markdown输出：{output_paths['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

