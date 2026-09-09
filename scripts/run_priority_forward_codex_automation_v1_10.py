"""校验 V1.10 冻结清单后调用原子占位编排器。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts import run_priority_forward_codex_automation_v1_9 as base


ROOT = base.ROOT
DEFAULT_MANIFEST = ROOT / "config/priority_forward_research_operations_v1_10_manifest.json"
DEFAULT_CONFIG = ROOT / "config/priority_forward_supervisor_v1_6.yaml"
EXPECTED_RUNTIME_PATCH_ID = (
    "PCF_IOPV_STRICT_TLS_ROUTE_V1_3_AND_STATUS_MODULE_ENTRYPOINT_V1_8_1"
)
ACCEPTED_MANIFEST_STATUSES = {
    "FROZEN_ZERO_PAID_FOUNDATION_V1_10_STRICT_TLS_ROUTE_AND_STATUS_MODULE_ACTIVE_NEXT_ELIGIBLE_WINDOW",
}

def verify_manifest(manifest_path: Path, expected_manifest_sha256: str) -> dict[str, Any]:
    original = (
        base.DEFAULT_MANIFEST,
        base.DEFAULT_CONFIG,
        base.ACCEPTED_MANIFEST_STATUSES,
        base.EXPECTED_RUNTIME_PATCH_ID,
    )
    base.DEFAULT_MANIFEST = DEFAULT_MANIFEST
    base.DEFAULT_CONFIG = DEFAULT_CONFIG
    base.ACCEPTED_MANIFEST_STATUSES = ACCEPTED_MANIFEST_STATUSES
    base.EXPECTED_RUNTIME_PATCH_ID = EXPECTED_RUNTIME_PATCH_ID
    try:
        result = base.verify_manifest(manifest_path, expected_manifest_sha256)
    finally:
        (
            base.DEFAULT_MANIFEST,
            base.DEFAULT_CONFIG,
            base.ACCEPTED_MANIFEST_STATUSES,
            base.EXPECTED_RUNTIME_PATCH_ID,
        ) = original
    result["status"] = "PASS_V1_10_FROZEN_ENTRYPOINT_VERIFIED"
    result["authority_status_entrypoint"] = (
        "python -m scripts.render_priority_forward_authoritative_status_v1_8_1"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 V1.10 冻结入口")
    parser.add_argument("--phase", choices=("morning", "close"), default="morning")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--now")
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()

    config_path = base.resolve_workspace_path(arguments.config)
    if config_path != DEFAULT_CONFIG.resolve():
        raise RuntimeError(f"V1.10 入口只接受固定编排配置：{DEFAULT_CONFIG}")
    verification = verify_manifest(
        arguments.manifest,
        arguments.expected_manifest_sha256,
    )
    if arguments.verify_only:
        print(json.dumps(verification, ensure_ascii=False, indent=2))
        return 0

    command = [
        sys.executable,
        str(base.DELEGATE),
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
        "V1.10 冻结清单校验通过，启动不可变回执去重编排器；"
        f"阶段={arguments.phase}，受控文件={verification['tracked_file_count']}。",
        file=sys.stderr,
    )
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
