"""校验 V1.7 冻结清单后调用原子占位编排器。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "config/priority_forward_research_operations_v1_7_manifest.json"
DEFAULT_CONFIG = ROOT / "config/priority_forward_supervisor_v1_3.yaml"
DELEGATE = ROOT / "scripts/run_priority_forward_codex_automation_v1_5.py"
ACCEPTED_MANIFEST_STATUSES = {
    "FROZEN_ZERO_PAID_FOUNDATION_V1_7_ACTIVE",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_workspace_path(value: Path) -> Path:
    path = value if value.is_absolute() else ROOT / value
    resolved = path.resolve()
    resolved.relative_to(ROOT.resolve())
    return resolved


def verify_manifest(manifest_path: Path, expected_manifest_sha256: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256):
        raise RuntimeError("期望清单 SHA-256 必须是 64 位小写十六进制")
    manifest_path = resolve_workspace_path(manifest_path)
    if manifest_path != DEFAULT_MANIFEST.resolve():
        raise RuntimeError(f"V1.7 入口只接受固定清单：{DEFAULT_MANIFEST}")
    if not manifest_path.is_file():
        raise RuntimeError(f"V1.7 冻结清单不存在：{manifest_path}")
    actual_manifest_sha256 = sha256_file(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise RuntimeError(
            "V1.7 清单文件哈希不匹配："
            f"预期={expected_manifest_sha256}，实际={actual_manifest_sha256}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("V1.7 清单顶层必须是对象")
    if manifest.get("status") not in ACCEPTED_MANIFEST_STATUSES:
        raise RuntimeError(f"V1.7 清单状态不可执行：{manifest.get('status')}")
    if manifest.get("data_purchase_budget_cny") != 0:
        raise RuntimeError("V1.7 清单的数据采购预算不是 0 CNY")
    if manifest.get("maximum_active_research_streams") != 2:
        raise RuntimeError("V1.7 清单的活动研究流上限不是 2")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("V1.7 清单没有受控文件")

    current_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise RuntimeError("V1.7 清单文件记录必须是对象")
        relative_path = str(item.get("path") or "")
        if not relative_path or relative_path in seen_paths:
            raise RuntimeError(f"V1.7 清单文件路径为空或重复：{relative_path!r}")
        seen_paths.add(relative_path)
        path = resolve_workspace_path(Path(relative_path))
        actual_sha256 = sha256_file(path) if path.is_file() else None
        actual_bytes = path.stat().st_size if path.is_file() else None
        record = {
            "path": relative_path,
            "sha256": actual_sha256,
            "bytes": actual_bytes,
        }
        current_records.append(record)
        if actual_sha256 != item.get("sha256") or actual_bytes != item.get("bytes"):
            failures.append(
                {
                    "path": relative_path,
                    "expected_sha256": item.get("sha256"),
                    "actual_sha256": actual_sha256,
                    "expected_bytes": item.get("bytes"),
                    "actual_bytes": actual_bytes,
                }
            )

    current_records.sort(key=lambda item: str(item["path"]))
    expected_records = sorted(files, key=lambda item: str(item["path"]))
    content_sha256 = hashlib.sha256(
        json.dumps(current_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if content_sha256 != manifest.get("content_sha256"):
        failures.append(
            {
                "path": "<manifest-content>",
                "expected_sha256": manifest.get("content_sha256"),
                "actual_sha256": content_sha256,
            }
        )
    if current_records != expected_records:
        failures.append({"path": "<record-order-or-shape>", "status": "MISMATCH"})

    required_runtime_paths = {
        Path(__file__).resolve().relative_to(ROOT.resolve()).as_posix(): "ENTRYPOINT_NOT_TRACKED",
        DELEGATE.resolve().relative_to(ROOT.resolve()).as_posix(): "DELEGATE_NOT_TRACKED",
        DEFAULT_CONFIG.resolve().relative_to(ROOT.resolve()).as_posix(): "SUPERVISOR_CONFIG_NOT_TRACKED",
    }
    for required_path, failure_status in required_runtime_paths.items():
        if required_path not in seen_paths:
            failures.append({"path": required_path, "status": failure_status})

    result = {
        "status": (
            "PASS_V1_7_FROZEN_ENTRYPOINT_VERIFIED"
            if not failures
            else "FAILED_V1_7_FROZEN_ENTRYPOINT_VERIFICATION"
        ),
        "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": actual_manifest_sha256,
        "content_sha256": content_sha256,
        "tracked_file_count": len(current_records),
        "failure_count": len(failures),
        "failures": failures,
        "data_purchase_budget_cny": 0,
        "maximum_active_research_streams": 2,
        "deduplication_evidence": "IMMUTABLE_TASK_RECEIPTS_ONLY",
    }
    if failures:
        raise RuntimeError(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 V1.7 冻结入口")
    parser.add_argument("--phase", choices=("morning", "close"), default="morning")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--now")
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()

    config_path = resolve_workspace_path(arguments.config)
    if config_path != DEFAULT_CONFIG.resolve():
        raise RuntimeError(f"V1.7 入口只接受固定编排配置：{DEFAULT_CONFIG}")
    verification = verify_manifest(arguments.manifest, arguments.expected_manifest_sha256)
    if arguments.verify_only:
        print(json.dumps(verification, ensure_ascii=False, indent=2))
        return 0

    command = [
        sys.executable,
        str(DELEGATE),
        "--phase",
        arguments.phase,
        "--config",
        str(config_path),
    ]
    if arguments.dry_run:
        command.append("--dry-run")
    if arguments.now:
        if not arguments.dry_run:
            raise RuntimeError("--now 只能与 --dry-run 同时使用")
        command.extend(["--now", arguments.now])
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    print(
        "V1.7 冻结清单校验通过，启动不可变回执去重编排器；"
        f"阶段={arguments.phase}，受控文件={verification['tracked_file_count']}。",
        file=sys.stderr,
    )
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
