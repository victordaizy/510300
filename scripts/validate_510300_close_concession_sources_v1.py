"""记录针对性测试、只读校时状态和保存输出的离线复算；不执行采集或交易。"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/research/510300_close_concession_source_feasibility_v1"
CN = timezone(timedelta(hours=8))


def run(command):
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=60, check=False)
    text = (completed.stdout + completed.stderr).decode("utf-8")
    return {"command": command, "exit_code": completed.returncode, "output": text}


def main():
    parser = argparse.ArgumentParser(description="510300来源语义验证，所有操作为只读来源或本轮报告写入")
    parser.add_argument("--record-local", action="store_true", help="在Windows记录当前校时状态与测试日志")
    args = parser.parse_args()
    if not args.record_local:
        result = run([sys.executable, "-X", "utf8", "scripts/analyze_510300_close_concession_sources_v1.py", "--verify-saved"])
        print(result["output"])
        raise SystemExit(result["exit_code"])
    output_dir = REPORT / "validation"
    output_dir.mkdir(exist_ok=True)
    started = datetime.now(CN).isoformat()
    clock = subprocess.run(["w32tm", "/query", "/status"], cwd=ROOT, capture_output=True, timeout=30, check=False)
    clock_raw = clock.stdout + clock.stderr
    (output_dir / "w32tm_status.raw").write_bytes(clock_raw)
    try:
        clock_text = clock_raw.decode("utf-8")
        clock_encoding = "utf-8"
    except UnicodeDecodeError:
        clock_text = clock_raw.decode("gb18030")
        clock_encoding = "gb18030"
    clock_receipt = {"started_at_local_clock": started, "received_at_local_clock": datetime.now(CN).isoformat(),
                     "command": ["w32tm", "/query", "/status"], "exit_code": clock.returncode,
                     "raw_sha256": hashlib.sha256(clock_raw).hexdigest(), "decoding": clock_encoding,
                     "output": clock_text, "clock_offset_proven": False,
                     "state": "UNSYNCHRONIZED_FREE_RUNNING_CLOCK" if "Free-running System Clock" in clock_text else "NOT_ADMITTED_FROM_STATUS_ONLY",
                     "clock_settings_changed": False}
    (output_dir / "clock_receipt.json").write_text(json.dumps(clock_receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    commands = [
        [sys.executable, "-X", "utf8", "-m", "unittest", "tests.test_510300_close_concession_sources_v1", "-v"],
        [sys.executable, "-X", "utf8", "scripts/analyze_510300_close_concession_sources_v1.py", "--verify-saved"],
    ]
    results = [run(command) for command in commands]
    receipt = {"recorded_at_local_clock": datetime.now(CN).isoformat(), "python_version": sys.version,
               "status": "PASS_TARGETED_TESTS_AND_SAVED_RECOMPUTATION" if all(r["exit_code"] == 0 for r in results) else "VALIDATION_FAILED",
               "runs": results, "network_downloads_during_tests_and_recomputation": 0,
               "new_accounts": 0, "future_return_reads": 0, "security_audit": False}
    (output_dir / "verification.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"validation": receipt["status"], "clock": clock_receipt["state"]}, ensure_ascii=False, indent=2))
    if not all(r["exit_code"] == 0 for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
