"""冻结 VAL01_NORM_EY_5Y 财务扩展实现及当前阻塞证据。"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.val01_norm_ey_5y_financial_extension import (
    load_config,
    sha256_file,
    verify_frozen_inputs,
)


IMPLEMENTATION_FILES = (
    "config/val01_norm_ey_5y_financial_extension_v1.yaml",
    "docs/VAL01_NORM_EY_5Y_FINANCIAL_EXTENSION_V1_PROTOCOL.md",
    "research/val01_norm_ey_5y_financial_extension.py",
    "research/val01_norm_ey_5y_post_acquisition_audit.py",
    "scripts/download_val01_norm_ey_5y_financial_extension.py",
    "scripts/audit_val01_norm_ey_5y_post_acquisition.py",
    "scripts/freeze_val01_norm_ey_5y_financial_extension_protocol.py",
    "tests/test_val01_norm_ey_5y_financial_extension.py",
    "tests/test_val01_norm_ey_5y_post_acquisition_audit.py",
)

EVIDENCE_FILES = (
    "reports/data_quality/000300_normalized_earnings_financial_extension_v1.json",
    "reports/data_quality/000300_normalized_earnings_financial_extension_v1.md",
)


def _canonical_hash(payload: dict) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_tests(arguments: list[str]) -> dict[str, object]:
    command = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        "-m",
        "pytest",
        *arguments,
        "-q",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"协议冻结前测试失败：\n{result.stdout}\n{result.stderr}"
        )
    return {
        "command": command,
        "return_code": 0,
        "stdout": result.stdout.strip(),
    }


def main() -> int:
    config = load_config()
    missing = [
        path for path in (*IMPLEMENTATION_FILES, *EVIDENCE_FILES)
        if not (ROOT / path).exists()
    ]
    if missing:
        raise FileNotFoundError(f"协议冻结文件缺失：{missing}")
    blocked_report = json.loads(
        (ROOT / config["artifacts"]["acquisition_report_json"]).read_text(
            encoding="utf-8"
        )
    )
    if blocked_report["status"] != "BLOCKED_SECURE_TRANSPORT_TLS":
        raise RuntimeError("当前证据不是预期的安全传输 TLS 阻塞状态")
    frozen_inputs = verify_frozen_inputs(config)
    if frozen_inputs["status"] != "PASS":
        raise RuntimeError("原冻结输入哈希不一致")
    targeted = _run_tests(
        [
            "tests/test_val01_norm_ey_5y_financial_extension.py",
            "tests/test_val01_norm_ey_5y_post_acquisition_audit.py",
        ]
    )
    full_suite = _run_tests([])
    payload = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "status": "BLOCKED_DATA_ACQUISITION_PROTOCOL_IMPLEMENTATION_FROZEN",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "data_acquisition_status": blocked_report["status"],
        "data_outputs_created": False,
        "implementation_files": {
            path: sha256_file(ROOT / path) for path in IMPLEMENTATION_FILES
        },
        "evidence_files": {
            path: sha256_file(ROOT / path) for path in EVIDENCE_FILES
        },
        "frozen_input_audit": frozen_inputs,
        "test_results": {
            "targeted": targeted,
            "full_suite": full_suite,
        },
        "governance": blocked_report["governance"],
        "meaning": "只冻结协议实现和TLS阻塞证据；不表示财务扩展数据、正常化盈利覆盖或模型预测闸门已经通过。",
    }
    payload["manifest_content_sha256"] = _canonical_hash(payload)
    manifest = ROOT / config["artifacts"]["protocol_manifest"]
    manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "data_acquisition_status": payload["data_acquisition_status"],
                "manifest": manifest.relative_to(ROOT).as_posix(),
                "manifest_content_sha256": payload["manifest_content_sha256"],
                "test_results": payload["test_results"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
