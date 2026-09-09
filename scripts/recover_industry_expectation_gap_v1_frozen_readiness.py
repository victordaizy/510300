"""从 2026-08-18 前向采集日志恢复冻结时的就绪报告原件。"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.industry_expectation_gap_frozen_input_recovery import (
    FrozenInputRecoveryError,
    file_sha256,
    reconstruct_json_bytes_from_log,
)


SOURCE_LOG = ROOT / "output" / "primary_market_forward_logs" / "20260818.stdout.log"
SOURCE_LOG_SHA256 = "b8630a14b5b5749787e63228515d0ad674c458556441fda428e333a38412edc4"
SOURCE_MARKER = "2026-08-18T15:01:00.111590+08:00"
OUTPUT = (
    ROOT
    / "paper"
    / "industry_expectation_gap_v1"
    / "frozen_inputs"
    / "2026-08-18_primary_market_readiness.json"
)
EXPECTED_SHA256 = "1d4432ce581cf63f449315ef4932230fccc7e15a09affcac9e2662121fa5d2d8"
EXPECTED_BYTES = 967


def _atomic_write_once(path: Path, content: bytes) -> str:
    if path.exists():
        if path.read_bytes() != content:
            raise FrozenInputRecoveryError(f"恢复目标已存在但内容不同：{path}")
        return "EXISTING_IDENTICAL"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return "CREATED"


def main() -> int:
    if not SOURCE_LOG.is_file():
        raise FrozenInputRecoveryError(f"恢复来源日志不存在：{SOURCE_LOG}")
    if file_sha256(SOURCE_LOG) != SOURCE_LOG_SHA256:
        raise FrozenInputRecoveryError("恢复来源日志哈希不一致")
    recovered = reconstruct_json_bytes_from_log(
        SOURCE_LOG,
        marker=SOURCE_MARKER,
        newline="CRLF",
    )
    actual_hash = hashlib.sha256(recovered).hexdigest()
    if actual_hash != EXPECTED_SHA256 or len(recovered) != EXPECTED_BYTES:
        raise FrozenInputRecoveryError(
            f"恢复结果不匹配原冻结清单：sha256={actual_hash}, bytes={len(recovered)}"
        )
    state = _atomic_write_once(OUTPUT, recovered)
    print(f"冻结就绪报告恢复状态：{state}")
    print(f"输出：{OUTPUT}")
    print(f"SHA-256：{actual_hash}")
    print(f"字节数：{len(recovered)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
