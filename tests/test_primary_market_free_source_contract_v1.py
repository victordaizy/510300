from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml

from market_data.etf_primary_market import ProviderSnapshot
from scripts.collect_510300_primary_market import (
    FREE_SOURCE_RECEIPT_FIELDS,
    _write_raw,
    _write_source_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
TIMEZONE = ZoneInfo("Asia/Shanghai")
RUNNER = ROOT / "scripts" / "run_510300_primary_market_forward_v1_1.ps1"
TASK_WRAPPER = (
    ROOT / "scripts" / "run_510300_primary_market_collection_task_v1_1.ps1"
)


def _snapshot() -> ProviderSnapshot:
    return ProviderSnapshot(
        record={
            "fund_code": "510300",
            "trading_day": "2026-08-26",
            "retrieved_at": "2026-08-26T09:30:01+08:00",
            "source": "sse.test",
        },
        raw_payload={"result": [{"TRADE_CODE": "510300"}]},
    )


def test_success_receipt_contains_complete_zero_paid_contract(tmp_path: Path) -> None:
    observed_at = datetime(2026, 8, 26, 9, 30, 1, tzinfo=TIMEZONE)
    raw = _write_raw(_snapshot(), tmp_path / "raw", "pcf", observed_at, root=tmp_path)
    receipt_path = _write_source_receipt(
        receipt_root=tmp_path / "receipts",
        source_id="SSE_PCF_COMMON_QUERY",
        source_url_or_endpoint="https://query.sse.com.cn/commonQuery.do",
        acquired_at=observed_at,
        market_date="2026-08-26",
        raw_metadata=raw,
        parser_version="SsePcfProvider@1",
        schema_version="FREE_SOURCE_ACQUISITION_RECEIPT_V1",
        quality_status="PASS_RAW_CAPTURED_AND_PARSED",
        fallback_source="",
        failure_reason=None,
        root=tmp_path,
    )
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert FREE_SOURCE_RECEIPT_FIELDS <= payload.keys()
    assert payload["raw_response_hash"] == raw["sha256"]
    assert payload["raw_response_path"] == raw["path"]
    assert payload["access_cost_cny"] == 0
    assert payload["paid_data_used"] is False
    assert payload["failure_reason"] is None


def test_failure_receipt_preserves_missing_raw_as_null(tmp_path: Path) -> None:
    observed_at = datetime(2026, 8, 26, 9, 30, 1, tzinfo=TIMEZONE)
    receipt_path = _write_source_receipt(
        receipt_root=tmp_path / "receipts",
        source_id="SSE_YUNHQ_IOPV_SNAPSHOT",
        source_url_or_endpoint="https://yunhq.sse.com.cn:32042/v1/sh1/snap/510300",
        acquired_at=observed_at,
        market_date="2026-08-26",
        raw_metadata=None,
        parser_version="SseIopvSnapshotProvider@1",
        schema_version="FREE_SOURCE_ACQUISITION_RECEIPT_V1",
        quality_status="FAILED_SOURCE_ACCESS_OR_PARSE",
        fallback_source="SECOND_FREE_ETF_QUOTE_PENDING",
        failure_reason="ConnectionError: 测试连接失败",
        root=tmp_path,
    )
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert FREE_SOURCE_RECEIPT_FIELDS <= payload.keys()
    assert payload["raw_response_hash"] is None
    assert payload["raw_response_path"] is None
    assert payload["failure_reason"] == "ConnectionError: 测试连接失败"


def test_raw_response_is_immutable_for_same_capture_identity(tmp_path: Path) -> None:
    observed_at = datetime(2026, 8, 26, 9, 30, 1, tzinfo=TIMEZONE)
    _write_raw(_snapshot(), tmp_path / "raw", "pcf", observed_at, root=tmp_path)
    with pytest.raises(FileExistsError):
        _write_raw(_snapshot(), tmp_path / "raw", "pcf", observed_at, root=tmp_path)


def test_primary_market_config_declares_free_source_receipts_and_fallbacks() -> None:
    config = yaml.safe_load(
        (ROOT / "config" / "primary_market_forward.yaml").read_text(encoding="utf-8")
    )
    assert config["outputs"]["source_receipt_directory"].startswith(
        "reports/data_quality/"
    )
    contract = config["source_contract"]
    assert contract["schema_version"] == "FREE_SOURCE_ACQUISITION_RECEIPT_V1"
    assert contract["sources"]["pcf"]["access_cost_cny"] == 0
    assert contract["sources"]["iopv"]["access_cost_cny"] == 0
    assert (
        contract["sources"]["iopv"]["fallback_source"]
        == "SECOND_FREE_ETF_QUOTE_PENDING"
    )


def _run_powershell(script: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    return subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            *arguments,
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def test_runner_preserves_external_free_source_exit_code(tmp_path: Path) -> None:
    failing_collector = tmp_path / "failing_collector.py"
    failing_collector.write_text(
        "import sys\nprint('模拟免费来源失败', file=sys.stderr)\nsys.exit(3)\n",
        encoding="utf-8",
    )
    result = _run_powershell(
        RUNNER,
        "-SkipPaperSignal",
        "-PythonExecutable",
        sys.executable,
        "-CollectorScript",
        str(failing_collector),
    )
    assert result.returncode == 3, result.stderr
    assert "退出码：3" in result.stderr


def test_task_wrapper_classifies_external_failure_and_writes_receipt(
    tmp_path: Path,
) -> None:
    fake_runner = tmp_path / "fake_runner.ps1"
    fake_runner.write_text(
        "[Console]::Error.WriteLine('模拟免费来源失败')\nexit 3\n",
        encoding="utf-8-sig",
    )
    result = _run_powershell(
        TASK_WRAPPER,
        "-TaskName",
        "测试-PCF-IOPV-免费来源失败",
        "-RunnerPath",
        str(fake_runner),
        "-RuntimeRoot",
        str(tmp_path),
    )
    assert result.returncode == 3, result.stderr
    current_status = (
        tmp_path
        / "reports"
        / "data_quality"
        / "510300_primary_market_task_status.json"
    )
    payload = json.loads(current_status.read_text(encoding="utf-8"))
    assert payload["run_status"] == "FAILED"
    assert payload["collection_status"] == "EXTERNAL_FREE_SOURCE_FAILED"
    assert payload["task_exit_code"] == 3
    receipts = list(
        (tmp_path / "reports" / "data_quality" / "primary_market_task_runs").glob(
            "*.json"
        )
    )
    assert len(receipts) == 1
