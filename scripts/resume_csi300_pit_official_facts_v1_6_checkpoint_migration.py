from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from scripts import (
    migrate_csi300_pit_official_facts_v1_5_complete_checkpoints_to_v1_6 as _migration,
)


def _verify_frozen_inputs_compatible(config: dict[str, Any]) -> None:
    manifest_rule = config["supersedes"]["protocol_manifest"]
    manifest_path = _migration._collector_base.project_path(manifest_rule["path"])
    _migration._collector_base.verify_manifest(
        manifest_path,
        expected_status=manifest_rule["expected_status"],
        expected_content_sha256=manifest_rule["expected_content_sha256"],
    )
    actual_manifest_file_sha256 = _migration._sha256_file(manifest_path)
    if actual_manifest_file_sha256 != manifest_rule["expected_file_sha256"]:
        raise RuntimeError(
            "V1.5冻结清单文件哈希不一致："
            f"{actual_manifest_file_sha256}|{manifest_rule['expected_file_sha256']}"
        )

    self_manifest_path = _migration._collector_base.project_path(
        config["artifacts"]["protocol_manifest"]
    )
    _migration._collector_base.verify_manifest(
        self_manifest_path,
        expected_status=_migration.FROZEN_STATUS,
        expected_content_sha256=None,
        verify_own_content_payload=True,
    )
    receipt_rule = config["supersedes"]["full_collection_receipt"]
    receipt_path = _migration._collector_base.project_path(receipt_rule["path"])
    if _migration._sha256_file(receipt_path) != receipt_rule["expected_sha256"]:
        raise RuntimeError("V1.5全量收据哈希不一致")


def main() -> int:
    _migration._verify_frozen_inputs = _verify_frozen_inputs_compatible
    return _migration.main()


if __name__ == "__main__":
    raise SystemExit(main())
