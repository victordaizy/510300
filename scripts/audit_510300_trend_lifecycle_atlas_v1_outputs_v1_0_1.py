"""修正V1输出审计终端序列化，不改变任何冻结研究逻辑。"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import audit_510300_trend_lifecycle_atlas_v1_outputs as base_audit


OUTPUT_JSON = (
    ROOT
    / "reports"
    / "audit"
    / "510300_trend_lifecycle_atlas_v1_output_audit_v1_0_1_correction.json"
)
OUTPUT_MARKDOWN = (
    ROOT
    / "reports"
    / "audit"
    / "510300_TREND_LIFECYCLE_ATLAS_V1_OUTPUT_AUDIT_V1_0_1_CORRECTION.md"
)


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_native(item) for item in value]
    if isinstance(value, tuple):
        return [_native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    payload = _native(base_audit.audit())
    payload["correction"] = {
        "version": "1.0.1",
        "generated_at_asia_shanghai": datetime.now(
            ZoneInfo("Asia/Shanghai")
        ).isoformat(),
        "original_research_logic_modified": False,
        "original_frozen_audit_script_modified": False,
        "original_entrypoint_status": (
            "CHECKS_AND_ARTIFACT_COMPLETED_TERMINAL_PRINT_NUMPY_BOOL_SERIALIZATION_FAILED"
        ),
        "correction_scope": "RECURSIVE_NATIVE_JSON_TYPE_NORMALIZATION_AND_EXIT_STATUS_ONLY",
    }
    _atomic_text(
        OUTPUT_JSON,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    markdown = "\n".join(
        [
            "# 510300 趋势生命周期图谱 V1 输出审计 V1.0.1 修正",
            "",
            f"- 状态：`{payload['status']}`",
            "- 原V1的18项聚焦检查及审计产物均已完成；仅终端打印一个NumPy布尔值时退出失败。",
            "- 本修正只把审计载荷递归转换为原生JSON类型并返回正确退出码。",
            "- 冻结研究逻辑、协议、标签、阈值、变量、输入、结果和原审计脚本均未修改。",
            "",
        ]
    )
    _atomic_text(OUTPUT_MARKDOWN, markdown)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
