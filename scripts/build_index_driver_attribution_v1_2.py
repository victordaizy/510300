"""按 V1.2 冻结清单构建点时指数驱动归因。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.index_driver_attribution import AttributionRules  # noqa: E402
from research.index_driver_attribution_v1_2 import (  # noqa: E402
    build_index_driver_attribution_v1_2,
)
from scripts.build_index_driver_attribution_v1 import (  # noqa: E402
    _atomic_parquet_write,
    _atomic_text_write,
    _build_report,
    _render_markdown,
    sha256,
)


CONTRACT_FILE = ROOT / "config" / "index_driver_attribution_v1_2.yaml"
MANIFEST_FILE = ROOT / "config" / "index_driver_attribution_v1_2_manifest.json"


def _verify_manifest() -> dict:
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError("归因 V1.2 尚未冻结；先运行 V1.2 冻结脚本")
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    changed: list[str] = []
    for section in ("frozen_files", "parent_frozen_files", "input_files"):
        for relative, expected_hash in manifest[section].items():
            path = ROOT / relative
            if not path.exists() or sha256(path) != expected_hash:
                changed.append(relative)
    if changed:
        raise RuntimeError(f"归因 V1.2 冻结文件指纹变化：{changed}")
    return manifest


def main() -> int:
    manifest = _verify_manifest()
    contract = yaml.safe_load(CONTRACT_FILE.read_text(encoding="utf-8"))
    rules = AttributionRules.from_contract(contract)
    input_frames = {
        name: pd.read_parquet(ROOT / settings["path"])
        for name, settings in contract["inputs"].items()
    }
    result = build_index_driver_attribution_v1_2(
        input_frames["weights"],
        input_frames["constituent_daily"],
        input_frames["industry_intervals"],
        rules,
    )
    outputs = contract["outputs"]
    cross_path = ROOT / outputs["cross_section_path"]
    industry_path = ROOT / outputs["industry_path"]
    daily_path = ROOT / outputs["daily_summary_path"]
    json_path = ROOT / outputs["status_json_path"]
    markdown_path = ROOT / outputs["status_markdown_path"]
    _atomic_parquet_write(result.cross_section, cross_path)
    _atomic_parquet_write(result.industry, industry_path)
    _atomic_parquet_write(result.daily, daily_path)
    output_paths = {
        outputs["cross_section_path"]: cross_path,
        outputs["industry_path"]: industry_path,
        outputs["daily_summary_path"]: daily_path,
    }
    report = _build_report(result, contract, manifest, output_paths)
    report["manifest_sha256"] = sha256(MANIFEST_FILE)
    report["parent_manifest_sha256"] = manifest["parent_manifest_sha256"]
    report["implementation_revision"] = (
        "LATEST_EFFECTIVE_CITIC_L1_TRUNCATES_SUPERSEDED_INTERVAL"
    )
    _atomic_text_write(
        json.dumps(report, ensure_ascii=False, indent=2), json_path
    )
    markdown = _render_markdown(report).replace(
        "# 510300 点时指数驱动归因 V1 数据质量报告",
        "# 510300 点时指数驱动归因 V1.2 数据质量报告",
        1,
    )
    _atomic_text_write(markdown, markdown_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
