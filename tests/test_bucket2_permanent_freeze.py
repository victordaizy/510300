"""桶2永久冻结证据包的文件级完整性测试。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = (
    ROOT
    / "reports"
    / "audit"
    / "A_SHARE_BUCKET2_TIME_HOLDOUT_LOWVOL_V1_FREEZE_20260819"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_bucket2_frozen_evidence_hashes_are_unchanged() -> None:
    manifest = json.loads(
        (PACKAGE / "freeze_evidence_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["project_id"] == "A_SHARE_BUCKET2_TIME_HOLDOUT_LOWVOL_V1"
    assert manifest["governance_event"] == "NO_FURTHER_BUCKET2_ANALYSIS"
    for item in manifest["evidence"]:
        path = ROOT / item["path"]
        assert path.is_file(), item["path"]
        assert path.stat().st_size == item["bytes"], item["path"]
        assert _sha256(path) == item["sha256"], item["path"]


def test_bucket2_event_points_to_exact_frozen_manifest_only() -> None:
    event = json.loads(
        (PACKAGE / "NO_FURTHER_BUCKET2_ANALYSIS.json").read_text(encoding="utf-8")
    )
    manifest_path = ROOT / event["evidence_manifest"]["path"]

    assert event["project_id"] == "A_SHARE_BUCKET2_TIME_HOLDOUT_LOWVOL_V1"
    assert event["event_type"] == "NO_FURTHER_BUCKET2_ANALYSIS"
    assert _sha256(manifest_path) == event["evidence_manifest"]["sha256"]
    assert "bucket34" not in event["evidence_manifest"]["path"].lower()
    assert "stratified" not in event["evidence_manifest"]["path"].lower()

