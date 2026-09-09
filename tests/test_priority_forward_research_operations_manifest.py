from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config" / "priority_forward_research_operations_v1_4_manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_monitoring_manifest_frozen_files_are_exact() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert (
        manifest["status"]
        == "FROZEN_SINGLE_ENTRYPOINT_CODEX_AUTOMATION"
    )
    assert len(manifest["frozen_files"]) >= 32
    previous = manifest["supersedes"]
    assert previous["previous_manifest_preserved"] is True
    assert previous["path"].endswith(
        "priority_forward_research_operations_v1_3_manifest.json"
    )
    for item in manifest["frozen_files"]:
        path = ROOT / item["path"]
        assert path.stat().st_size == item["bytes"]
        assert _sha256(path) == item["sha256"]
    for item in manifest["external_automation_evidence"]:
        path = Path(item["path"])
        assert path.stat().st_size == item["bytes"]
        assert _sha256(path) == item["sha256"]
        assert item["status"] == "ACTIVE"
        text = path.read_text(encoding="utf-8")
        if item["id"] == "510300-pcf-iopv":
            assert "run_priority_forward_codex_automation.py --phase morning" in text
            assert "09:25至09:35" in text
            assert "不得运行 generate_510300_small_account_paper_signal.py" in text
            assert "不得生成预测、仓位或订单" in text
            assert "不得连接券商" in text
        elif item["id"] == "510300":
            assert "run_priority_forward_codex_automation.py --phase close" in text
            assert "不得运行 generate_510300_small_account_paper_signal.py" in text
            assert "不得运行V3策略" in text
            assert "不得启动正交低波复制" in text
            assert "不得生成仓位或订单" in text


def test_runtime_evidence_and_report_snapshot_are_exact() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for item in manifest["runtime_evidence"].values():
        path = ROOT / item["path"]
        assert path.stat().st_size == item["bytes"]
        assert _sha256(path) == item["sha256"]
    snapshot_item = manifest["runtime_evidence"]["priority_status_json_snapshot"]
    report = json.loads((ROOT / snapshot_item["path"]).read_text(encoding="utf-8"))
    assert report["threshold_events"]["total_event_count"] == 0
    assert report["directions"]["orthogonal_low_vol_replication"][
        "eligible_to_start"
    ] is False
    assert report["scheduler"]["registered_tasks_verified"] is False
    assert report["scheduler"]["manual_runners_verified"] is True
    for field in (
        "position_mapping_enabled",
        "order_generation_enabled",
        "broker_connection_enabled",
        "live_trading_enabled",
    ):
        assert report["safety"][field] is False
    assert report["safety"]["research_only"] is True


def test_manifest_never_authorizes_trading_or_low_vol_start() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    authorization = manifest["authorization"]
    assert authorization["research_only"] is True
    assert authorization["automatic_trading_authorized"] is False
    assert authorization["position_mapping_enabled"] is False
    assert authorization["order_generation_enabled"] is False
    assert authorization["broker_connection_enabled"] is False
    assert authorization["live_trading_enabled"] is False
    assert manifest["current_state"]["orthogonal_low_vol_started"] is False
    assert manifest["deployment"]["codex_morning_heartbeat_active"] is True
    assert manifest["deployment"]["codex_close_heartbeat_active"] is True
    assert manifest["deployment"]["single_tested_entrypoint"] is True
    assert manifest["deployment"]["real_close_deduplication_receipt_verified"] is True
    assert manifest["deployment"]["windows_task_scheduler_verified"] is False
