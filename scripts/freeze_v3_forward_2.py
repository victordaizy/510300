"""生成一次性V3_FORWARD_2冻结指纹；存在且不同则拒绝覆盖。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config" / "v3_forward_2_manifest.json"
FROZEN_FILES = (
    "config/v3_forward_2.yaml",
    "config/valuation_v2_expected_return.yaml",
    "config/valuation_v3_state_aware_return.yaml",
    "research/valuation_v2_expected_return.py",
    "research/valuation_v3_state_aware_return.py",
    "research/v3_forward_validation.py",
    "research/v3_forward_2_validation.py",
    "scripts/refresh_v3_forward_2_inputs.py",
    "scripts/run_v3_forward_2_daily.py",
    "data/raw/reference/a_share_sw_industry_static.parquet",
    "data/features/000300_valuation_v3_state_aware_return.parquet",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    missing = [relative for relative in FROZEN_FILES if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError(f"冻结文件缺失：{missing}")
    payload = {
        "model_version": "V3_FORWARD_2",
        "parent_version": "V3_FORWARD_1",
        "true_oos_start": "2026-08-14",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "purpose": "零Token前瞻验证；公式参数不变，仅冻结免费数据供应链和字段语义",
        "frozen_files": {
            relative: sha256(ROOT / relative) for relative in FROZEN_FILES
        },
    }
    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
        left = {key: value for key, value in existing.items() if key != "created_at"}
        right = {key: value for key, value in payload.items() if key != "created_at"}
        if left != right:
            raise RuntimeError("V3_FORWARD_2冻结指纹不同；禁止覆盖，请创建新版本")
        print("V3_FORWARD_2冻结清单已存在，指纹一致。")
        return 0
    MANIFEST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已创建冻结清单：{MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
