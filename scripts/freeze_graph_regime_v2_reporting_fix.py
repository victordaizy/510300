"""冻结V2报告层勘误；计算核心、参数与输入保持不变。"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_MANIFEST = (
    ROOT / "config" / "graph_regime_martin_turtle_v2_implementation_manifest.json"
)
FIX_MANIFEST = (
    ROOT
    / "config"
    / "graph_regime_martin_turtle_v2_implementation_manifest_reporting_fix.json"
)
PRE_FIX_REPORT = ROOT / "reports" / "backtest" / "graph_regime_martin_turtle_v2.json"
IMPLEMENTATION_FILES = (
    "research/graph_regime_martin_turtle_v2.py",
    "scripts/run_graph_regime_martin_turtle_v2.py",
    "scripts/freeze_graph_regime_v2_reporting_fix.py",
    "tests/test_graph_regime_martin_turtle_v2.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _semantic_report_sha256(report_path: Path) -> str:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    semantic = {
        "buy_hold": report["buy_hold"],
        "h00300": report["h00300"],
        "variants": report["variants"],
        "registered_pass_count": report["registered_pass_count"],
        "divergence_events": report["divergence_events"],
    }
    canonical = json.dumps(
        semantic, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _stable(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if key != "frozen_at"}


def main() -> int:
    if not ORIGINAL_MANIFEST.exists() or not PRE_FIX_REPORT.exists():
        raise FileNotFoundError("原实现冻结清单或修复前JSON报告缺失")
    original = json.loads(ORIGINAL_MANIFEST.read_text(encoding="utf-8"))
    core_path = "research/graph_regime_martin_turtle_v2.py"
    if sha256(ROOT / core_path) != original["implementation_files"][core_path]:
        raise RuntimeError("计算核心发生变化，不能归类为报告层勘误")
    for relative_path, expected_hash in original["input_files"].items():
        if sha256(ROOT / relative_path) != expected_hash:
            raise RuntimeError(f"输入快照发生变化：{relative_path}")
    missing = [path for path in IMPLEMENTATION_FILES if not (ROOT / path).exists()]
    if missing:
        raise FileNotFoundError(f"报告层勘误冻结文件缺失：{missing}")
    payload = {
        "project_id": original["project_id"],
        "version": "2.0.1-reporting-fix",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_commit": _git_value("rev-parse", "HEAD"),
        "worktree_dirty_at_freeze": bool(_git_value("status", "--porcelain")),
        "state": "DISCOVERY_ONLY",
        "freeze_stage": "REPORTING_FIX_FROZEN_HISTORICAL_RERUN_ALLOWED",
        "change_scope": "REPORT_RENDERING_ONLY",
        "change_reason": "无交易轨道的Sharpe为null，旧Markdown格式化器不接受null",
        "supersedes_implementation_manifest_sha256": sha256(ORIGINAL_MANIFEST),
        "pre_fix_report_json_sha256": sha256(PRE_FIX_REPORT),
        "pre_fix_semantic_report_sha256": _semantic_report_sha256(PRE_FIX_REPORT),
        "computation_core_unchanged": True,
        "parameters_unchanged": True,
        "inputs_unchanged": True,
        "return_calculation_allowed": True,
        "historical_evidence_label": original["historical_evidence_label"],
        "true_forward_start": None,
        "implementation_files": {
            path: sha256(ROOT / path) for path in IMPLEMENTATION_FILES
        },
        "input_files": original["input_files"],
        "candidate_ids": original["candidate_ids"],
        "external_frozen_baseline": original["external_frozen_baseline"],
        "governance": original["governance"],
    }
    if FIX_MANIFEST.exists():
        existing = json.loads(FIX_MANIFEST.read_text(encoding="utf-8"))
        if _stable(existing) != _stable(payload):
            raise RuntimeError("报告层勘误指纹已变化，禁止覆盖")
        print("V2报告层勘误冻结清单已存在，指纹一致。")
        return 0
    FIX_MANIFEST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
