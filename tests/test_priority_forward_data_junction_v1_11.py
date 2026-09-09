"""目录迁移回归测试；全部模拟保存写入隔离目录，不访问行情来源。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from market_data.etf_primary_market import ProviderSnapshot
from scripts import free_source_storage_v1_2 as legacy_storage
from scripts import free_source_storage_v1_3 as storage
from scripts import run_priority_forward_codex_automation_v1_11 as entry
from scripts import run_priority_forward_codex_automation_v1_5 as delegate
from scripts.priority_forward_data_paths_v1 import ROOT, WorkspacePaths


def junction(link: Path, target: Path) -> None:
    quote = lambda value: "'" + str(value).replace("'", "''") + "'"
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
         f"New-Item -ItemType Junction -Path {quote(link)} -Target {quote(target)} -ErrorAction Stop | Out-Null"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def migrated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> WorkspacePaths:
    root = tmp_path / "项目"
    target = tmp_path / "数据实际目录"
    root.mkdir()
    target.mkdir()
    junction(root / "data", target)
    paths = WorkspacePaths(root, target)
    monkeypatch.setattr(storage, "PATHS", paths)
    monkeypatch.setattr(storage, "PROJECT_ROOT", root)
    return paths


def test_old_failure_and_new_logical_relative_path(migrated: WorkspacePaths) -> None:
    path = migrated.root / "data" / "原始行情.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        legacy_storage._relative(path, migrated.root)
    assert migrated.checked(path) == path
    assert migrated.relative(path) == "data/原始行情.json"
    assert path.resolve() == migrated.approved_data_root / "原始行情.json"


def test_raw_bytes_unchanged_and_content_addressed_deduplication(migrated: WorkspacePaths, tmp_path: Path) -> None:
    snapshot = ProviderSnapshot(record={"source": "隔离模拟数据", "iopv": 1.25}, raw_payload={"模拟": True})
    observed = datetime(2026, 9, 7, 9, 31, tzinfo=ZoneInfo("Asia/Shanghai"))
    raw_root = migrated.root / "data" / "raw"
    first = storage.write_content_addressed_raw(snapshot, raw_root, "iopv", observed, root=migrated.root)
    second = storage.write_content_addressed_raw(snapshot, raw_root, "iopv", observed, root=migrated.root)
    legacy_root = tmp_path / "旧格式对照"
    legacy_root.mkdir()
    baseline = legacy_storage.write_content_addressed_raw(snapshot, legacy_root / "data/raw", "iopv", observed, root=legacy_root)
    assert first["sha256"] == baseline["sha256"]
    assert first["path"] == baseline["path"]
    assert second["existing_identical"] is True
    assert len(list(raw_root.rglob("*.json"))) == 1
    assert hashlib.sha256((migrated.root / first["path"]).read_bytes()).hexdigest() == first["sha256"]


def test_parquet_atomic_save_keeps_latest_duplicate(migrated: WorkspacePaths) -> None:
    target = migrated.root / "data" / "parsed" / "observations.parquet"
    storage.persist_record_atomic({"时点": "09:31", "净值": 1.0}, target, "时点")
    result = storage.persist_record_atomic({"时点": "09:31", "净值": 1.1}, target, "时点")
    frame = pd.read_parquet(target)
    assert len(frame) == 1
    assert frame.iloc[0]["净值"] == 1.1
    assert result["path"] == "data/parsed/observations.parquet"
    assert result["atomic_replace"] is True
    assert not list(target.parent.glob("*.tmp"))


def test_wrong_junction_or_nested_redirect_is_rejected_before_write(migrated: WorkspacePaths, tmp_path: Path) -> None:
    different = tmp_path / "其他目录"
    different.mkdir()
    wrong_policy = WorkspacePaths(migrated.root, different)
    with pytest.raises(ValueError, match="实际目标"):
        wrong_policy.checked("data/a.json")
    junction(migrated.root / "data" / "unexpected", different)
    with pytest.raises(ValueError, match="额外重定向"):
        storage.persist_record_atomic({"id": 1}, migrated.root / "data/unexpected/a.parquet", "id")
    assert not list(different.iterdir())


def test_external_and_parent_paths_remain_rejected(migrated: WorkspacePaths) -> None:
    for value in (migrated.approved_data_root / "a.json", "data/../outside.json", "../other.json"):
        with pytest.raises(ValueError):
            migrated.checked(value)
    assert migrated.relative("reports/status.json") == "reports/status.json"


@pytest.mark.parametrize("when,decision,exit_code", [
    ("2026-09-06T09:25:00+08:00", "NOT_DUE_NON_WEEKDAY", 0),
    ("2026-09-07T09:20:00+08:00", "WAITING_FOR_SCHEDULE", 2),
    ("2026-09-07T09:25:00+08:00", "RUN_NOW", 0),
    ("2026-09-07T09:35:00+08:00", "RUN_NOW", 0),
    ("2026-09-07T09:36:00+08:00", "MISSED_START_WINDOW", 2),
])
def test_dry_run_retains_original_window_without_writes(when: str, decision: str, exit_code: int, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("模拟检查不得启动子进程、保存回执或申请同日占位")
    for name in ("_run_powershell", "_run_renderer", "_write_once_json", "_atomic_json", "claim_task_attempt"):
        monkeypatch.setattr(delegate, name, forbidden)
    config = delegate.load_config(entry.CONFIG)
    payload, code = delegate.run_phase(config, "morning", datetime.fromisoformat(when), dry_run=True)
    assert code == exit_code
    assert payload["task_results"][0]["decision"] == decision
    assert payload["receipt_written"] is False
    assert payload["task_results"][0]["claim_acquired"] is None
    assert config["outputs"]["codex_task_claim_directory"].endswith("_v1_10")


def test_current_predecessor_49_files_are_unchanged() -> None:
    assert len(entry.verified_predecessor()["files"]) == 49


def test_collector_help_accepts_module_and_file_without_fetching() -> None:
    for command in (
        ["-m", "scripts.collect_510300_primary_market_v1_4", "--help"],
        [str(ROOT / "scripts/collect_510300_primary_market_v1_4.py"), "--help"],
    ):
        result = subprocess.run([sys.executable, *command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=False)
        assert result.returncode == 0, result.stderr
        assert "--watch" in result.stdout


def test_status_separates_legacy_days_from_strict_days() -> None:
    status = entry.collection_status(datetime(2026, 9, 6, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai")))
    assert status["status"] == "NO_SAME_DAY_TASK_RECEIPT"
    assert status["legacy_complete_quality_days_separate"] == 2
    assert status["strict_route_quality_days_in_readiness"] == 0
    assert status["position_impact"] == 0
