from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/Scripts/python.exe"
FORWARD_RUNNER = ROOT / "scripts/run_510300_primary_market_forward_v1_2_1.ps1"
TASK_RUNNER = ROOT / "scripts/run_510300_primary_market_collection_task_v1_2_1.ps1"
INSTALLER = ROOT / "scripts/install_priority_forward_morning_task_v1_8.ps1"


def run_powershell(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            *arguments,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def test_v1_2_1_python_modules_import_from_project_root() -> None:
    completed = subprocess.run(
        [
            str(PYTHON),
            "-c",
            (
                "import scripts.collect_510300_primary_market_v1_2; "
                "import scripts.analyze_510300_primary_market_readiness_v1_2"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_v1_2_1_forward_runner_uses_module_entrypoints_by_default() -> None:
    text = FORWARD_RUNNER.read_text(encoding="utf-8-sig")

    assert '@("-m", "scripts.collect_510300_primary_market_v1_2")' in text
    assert (
        '@("-m", "scripts.analyze_510300_primary_market_readiness_v1_2")'
        in text
    )
    assert "Push-Location -LiteralPath $projectRoot" in text
    assert "Pop-Location" in text


def test_v1_2_1_forward_runner_preserves_success_and_external_failure_codes(
    tmp_path: Path,
) -> None:
    collector = tmp_path / "collector.py"
    analyzer = tmp_path / "analyzer.py"
    collector.write_text("raise SystemExit(0)\n", encoding="utf-8")
    analyzer.write_text("raise SystemExit(0)\n", encoding="utf-8")
    base_arguments = [
        "-File",
        str(FORWARD_RUNNER),
        "-PythonExecutable",
        str(PYTHON),
        "-CollectorScript",
        str(collector),
        "-AnalyzerScript",
        str(analyzer),
        "-MaxSamples",
        "1",
    ]

    passed = run_powershell(base_arguments)
    collector.write_text("raise SystemExit(3)\n", encoding="utf-8")
    external_failure = run_powershell(base_arguments)

    assert passed.returncode == 0, passed.stderr or passed.stdout
    assert external_failure.returncode == 3


def test_v1_2_1_task_receipt_binds_runtime_patch(tmp_path: Path) -> None:
    failed_runner = tmp_path / "failed_runner.ps1"
    failed_runner.write_text("exit 1\n", encoding="utf-8-sig")

    completed = run_powershell(
        [
            "-File",
            str(TASK_RUNNER),
            "-RunnerPath",
            str(failed_runner),
            "-RuntimeRoot",
            str(tmp_path),
        ]
    )
    receipts = sorted(
        (tmp_path / "reports/data_quality/primary_market_task_runs_v1_2_1").glob(
            "*.json"
        )
    )

    assert completed.returncode == 1
    assert len(receipts) == 1
    payload = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.2.1"
    assert payload["run_status"] == "FAILED"
    assert payload["collection_status"] == "PROGRAM_FAILED"
    assert payload["task_exit_code"] == 1
    assert payload["runtime_patch_id"] == (
        "PCF_IOPV_PYTHON_MODULE_ENTRYPOINT_V1_2_1"
    )
    assert payload["python_module_entrypoint"] is True
    assert payload["runner_file"] == str(failed_runner)


def test_v1_2_1_powershell_files_have_bom_and_parse_in_windows_powershell() -> None:
    for path in (FORWARD_RUNNER, TASK_RUNNER, INSTALLER):
        assert path.read_bytes().startswith(b"\xef\xbb\xbf")
        command = (
            "$tokens = $null; $errors = $null; "
            "[System.Management.Automation.Language.Parser]::ParseFile("
            f"'{path}', [ref]$tokens, [ref]$errors) | Out-Null; "
            "if ($errors.Count -gt 0) { "
            "$errors | ForEach-Object { [Console]::Error.WriteLine($_.Message) }; "
            "exit 1 }"
        )
        completed = run_powershell(["-Command", command])
        assert completed.returncode == 0, completed.stderr
