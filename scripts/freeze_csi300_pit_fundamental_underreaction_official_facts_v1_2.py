from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from scripts import (  # noqa: E402
    freeze_csi300_pit_fundamental_underreaction_official_facts_v1_1 as _base,
)


CONFIG_PATH = ROOT / "config/csi300_pit_fundamental_underreaction_official_facts_v1_2.yaml"
TIMEZONE = ZoneInfo("Asia/Shanghai")
FROZEN_STATUS = (
    "FROZEN_CROSS_PAGE_TABLE_REVISION_AFTER_V1_1_PILOT_"
    "BEFORE_V1_2_BULK_AND_ANY_FUTURE_RETURN_READ"
)


def frozen_paths(config: dict[str, Any]) -> list[Path]:
    return [
        CONFIG_PATH,
        _base.project_path(config["artifacts"]["protocol_document"]),
        ROOT / "research/csi300_pit_fundamental_underreaction_official_facts_v1_2.py",
        ROOT / "scripts/collect_csi300_pit_fundamental_underreaction_official_facts_v1_2.py",
        ROOT / "scripts/freeze_csi300_pit_fundamental_underreaction_official_facts_v1_2.py",
        ROOT / "tests/test_csi300_pit_fundamental_underreaction_official_facts_v1_2.py",
    ]


def validate_existing(path: Path) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _base.verify_manifest(
        path,
        expected_status=FROZEN_STATUS,
        expected_content_sha256=str(manifest.get("content_sha256") or ""),
    )
    print(f"V1.2官方财务事实冻结清单已存在且匹配：{_base.relative_path(path)}")


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest_path = _base.project_path(config["artifacts"]["protocol_manifest"])
    if manifest_path.exists():
        validate_existing(manifest_path)
        return 0

    checkpoint_root = _base.project_path(config["artifacts"]["checkpoint_root"])
    if checkpoint_root.exists() and any(checkpoint_root.rglob("*.json.gz")):
        raise RuntimeError("已发现V1.2官方PDF采集检查点，拒绝事后冻结")

    parent = _base.validate_parent_manifest(config)
    superseded = _base.validate_superseded_v1(config)
    inputs = _base.validate_inputs(config)
    paths = frozen_paths(config)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少待冻结文件：{missing}")
    files = [_base.file_entry(path) for path in paths]
    parent_bundle = {
        **parent,
        "superseded_official_fact_v1_1": superseded,
        "representative_revision_gate": config["representative_revision_gate"],
    }
    manifest: dict[str, Any] = {
        "manifest_id": "CSI300_PIT_FUNDAMENTAL_UNDERREACTION_OFFICIAL_FACTS_V1_2_MANIFEST",
        "status": FROZEN_STATUS,
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "protocol_id": config["protocol"]["protocol_id"],
        "version": config["protocol"]["version"],
        "parent_freeze": parent_bundle,
        "inputs": inputs,
        "files": files,
        "new_checkpoint_root": _base.relative_path(checkpoint_root),
        "bulk_network_run_started": False,
        "market_price_read": False,
        "future_return_read": False,
        "future_label_created": False,
        "portfolio_return_calculated": False,
        "trading_authorization": False,
    }
    manifest["content_sha256"] = _base.canonical_sha256(
        {
            "status": manifest["status"],
            "protocol_id": manifest["protocol_id"],
            "version": manifest["version"],
            "parent_freeze": manifest["parent_freeze"],
            "inputs": manifest["inputs"],
            "files": manifest["files"],
        }
    )
    _base.atomic_write_json(manifest_path, manifest)
    for path in [*paths, manifest_path]:
        _base.read_only(path)
    print(f"V1.2官方财务事实协议已冻结：{_base.relative_path(manifest_path)}")
    print(f"V1.2官方财务事实内容SHA-256：{manifest['content_sha256']}")
    print("V1.1检查点未修改；冻结过程未读取市场价格或未来收益。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
