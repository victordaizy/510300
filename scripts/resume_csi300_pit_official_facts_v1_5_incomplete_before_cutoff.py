from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from scripts import (  # noqa: E402
    collect_csi300_pit_fundamental_underreaction_official_facts_v1 as _base,
)
from scripts import (  # noqa: E402
    collect_csi300_pit_fundamental_underreaction_official_facts_v1_5 as _v1_5,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "仅续跑指定 UTC 截止时间之前仍未更新的 V1.5 不完整官方 PDF 断点"
        )
    )
    parser.add_argument(
        "--cutoff-utc",
        required=True,
        help="补采启动时间，ISO 8601 UTC 格式；仅选择文件时间早于该时点的断点",
    )
    parser.add_argument("--workers", type=int, default=6, help="并发解析数")
    return parser.parse_args()


def parse_aware_timestamp(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("--cutoff-utc 必须包含 UTC 时区，例如 2026-08-31T22:12:56Z")
    return parsed.timestamp()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("并发数必须至少为 1")

    cutoff_timestamp = parse_aware_timestamp(args.cutoff_utc)
    original_select_work_rows = _base.select_work_rows

    def select_rows_before_cutoff(
        queue: Any,
        config: dict[str, Any],
        *,
        reparse_incomplete: bool,
        limit: int | None,
    ) -> tuple[list[dict[str, Any]], int]:
        del limit
        pending_rows, _ = original_select_work_rows(
            queue,
            config,
            reparse_incomplete=True,
            limit=None,
        )
        selected_rows: list[dict[str, Any]] = []
        for row in pending_rows:
            path = _base.checkpoint_path(config, row)
            if not path.exists() or path.stat().st_mtime < cutoff_timestamp:
                selected_rows.append(row)

        reusable_count = len(queue) - len(selected_rows)
        print(
            "按断点时间续跑："
            f"全部不完整候选 {len(pending_rows):,}；"
            f"截止时间前未更新 {len(selected_rows):,}",
            flush=True,
        )
        return selected_rows, reusable_count

    _v1_5._patch_base_collector()
    _base.select_work_rows = select_rows_before_cutoff
    _base.parse_args = lambda: argparse.Namespace(
        workers=args.workers,
        limit=None,
        reparse_incomplete=True,
    )
    return _v1_5.main()


if __name__ == "__main__":
    raise SystemExit(main())
