"""冻结DAILY_01分组支持异常的只计数报告实现。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.daily_01_overnight_absorption_v1 import sha256_file


EVALUATION_MANIFEST = ROOT / "config" / "daily_01_overnight_absorption_v1_evaluation_manifest.json"
MANIFEST = ROOT / "config" / "daily_01_overnight_absorption_v1_failure_reporting_manifest.json"
FROZEN_FILES = [
    "docs/510300_DAILY_01_OVERNIGHT_ABSORPTION_V1_FAILURE_REPORTING_ADDENDUM.md",
    "research/daily_01_overnight_absorption_v1_failure_reporting.py",
    "scripts/report_daily_01_overnight_absorption_v1_failure.py",
    "scripts/freeze_daily_01_overnight_absorption_v1_failure_reporting.py",
    "tests/test_daily_01_overnight_absorption_v1_failure_reporting.py",
]


def _git_output(*arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    evaluation = json.loads(EVALUATION_MANIFEST.read_text(encoding="utf-8"))
    if evaluation.get("historical_run_completed") is not False:
        raise RuntimeError("预测实现清单已经关闭，不能冻结失败报告")
    missing = [relative for relative in FROZEN_FILES if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError(f"冻结文件缺失：{missing}")
    payload = {
        "project_id": "DAILY_01_OVERNIGHT_ABSORPTION_V1_FAILURE_REPORTING",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_commit": _git_output("rev-parse", "HEAD"),
        "worktree_dirty_at_freeze": bool(_git_output("status", "--porcelain")),
        "scope": "POST_EXCEPTION_GROUP_SUPPORT_COUNTS_ONLY",
        "observed_exception": "ValueError: 区块自助缺少有利组或不利组",
        "threshold_changes_allowed": False,
        "return_differences_allowed": False,
        "bootstrap_allowed": False,
        "hac_allowed": False,
        "strategy_backtest_allowed": False,
        "evaluation_manifest_sha256_before_reporting": sha256_file(EVALUATION_MANIFEST),
        "frozen_files": {relative: sha256_file(ROOT / relative) for relative in FROZEN_FILES},
        "failure_report_completed": False,
    }
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(MANIFEST)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
