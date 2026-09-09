"""在任何许可历史文件进入项目前冻结上交所期权快照导入器。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "sse_option_historical_snapshot_import_v1.yaml"
MANIFEST = ROOT / "config" / "sse_option_historical_snapshot_import_v1_manifest.json"
FROZEN_FILES = (
    "config/sse_option_historical_snapshot_import_v1.yaml",
    "docs/SSE_OPTION_HISTORICAL_SNAPSHOT_IMPORT_V1_SPEC.md",
    "reports/data_quality/SSE_OPTION_HISTORICAL_SNAPSHOT_IMPORT_V1.md",
    "data/reference/sse_historical_data_interface_v1_1_5_20260327.pdf",
    "config/510300_option_surface_signal_v1_manifest.json",
    "research/sse_option_historical_snapshot_import_v1.py",
    "scripts/freeze_sse_option_historical_snapshot_import_v1.py",
    "tests/test_sse_option_historical_snapshot_import_v1.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_and_assert(*, require_pristine_output: bool = False) -> dict:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["protocol"]["project_id"] != "SSE_OPTION_HISTORICAL_SNAPSHOT_IMPORT_V1":
        raise ValueError("上交所历史快照导入协议ID漂移")
    if config["protocol"]["strategy_or_return_test_authorized"]:
        raise ValueError("导入协议不得授权策略或收益检验")
    if config["governance"]["purchase_authorized"]:
        raise ValueError("用户未授权购买官方历史数据")
    if config["governance"]["unlicensed_download"] != "FORBIDDEN":
        raise ValueError("未许可下载必须严格禁止")
    reference = ROOT / config["official_source"]["interface_local_file"]
    if sha256(reference) != str(config["official_source"]["interface_sha256"]).lower():
        raise ValueError("官方接口说明书SHA-256不匹配")
    if require_pristine_output:
        output_root = ROOT / config["paths"]["output_root"]
        if output_root.exists() and any(output_root.iterdir()):
            raise RuntimeError("冻结前已经存在规范化历史快照")
    return config


def verify_protocol() -> tuple[dict, dict]:
    config = _load_and_assert()
    if not MANIFEST.exists():
        raise RuntimeError("上交所历史快照导入协议尚未冻结")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if set(manifest.get("frozen_files", {})) != set(FROZEN_FILES):
        raise RuntimeError("上交所历史快照导入冻结文件集合漂移")
    drift = [
        relative
        for relative, expected in manifest["frozen_files"].items()
        if not (ROOT / relative).exists() or sha256(ROOT / relative) != expected
    ]
    if drift:
        raise RuntimeError(f"上交所历史快照导入文件哈希漂移：{drift}")
    return config, manifest


def _run_tests() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_sse_option_historical_snapshot_import_v1.py", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stdout + "\n" + completed.stderr)[-5000:]
        raise RuntimeError(f"上交所历史快照导入冻结前测试失败：\n{detail}")


def main() -> int:
    config = _load_and_assert(require_pristine_output=not MANIFEST.exists())
    missing = [relative for relative in FROZEN_FILES if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError(f"上交所历史快照导入冻结文件缺失：{missing}")
    if MANIFEST.exists():
        verify_protocol()
        print("上交所股票期权历史快照导入V1已冻结，全部哈希一致。")
        return 0
    licensed_root = ROOT / config["paths"]["licensed_input_root"]
    if licensed_root.exists() and any(licensed_root.rglob("Snapshot.csv")):
        raise RuntimeError("许可历史快照已经进入项目，禁止事后冻结导入规则")
    _run_tests()
    payload = {
        "project_id": config["protocol"]["project_id"],
        "state": config["protocol"]["state"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "licensed_data_present_at_freeze": False,
        "published_option_history_price_cny_per_year": int(
            config["official_source"]["published_option_history_price_cny_per_year"]
        ),
        "frozen_files": {relative: sha256(ROOT / relative) for relative in FROZEN_FILES},
        "governance": config["governance"],
    }
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(MANIFEST)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
