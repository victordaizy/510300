from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from research.csi300_pit_fundamental_underreaction_official_facts_v1_7 import (  # noqa: E402
    PARSER_VERSION,
    REQUIRED_METRICS,
    extract_official_pdf_facts,
)
from scripts import (  # noqa: E402
    collect_csi300_pit_fundamental_underreaction_official_facts_v1 as _base,
)


CONFIG_PATH = ROOT / "config/csi300_pit_fundamental_underreaction_official_facts_v1_7.yaml"
FROZEN_STATUS = (
    "FROZEN_ADJACENT_NOTE_EXTENDED_PROFIT_QUARTERLY_IMAGE_IDENTITY_"
    "AFTER_V1_6_FULL_REPARSE_BEFORE_V1_7_GAP_REPARSE_AND_ANY_FUTURE_RETURN_READ"
)
_BASE_PROCESS_DOCUMENT = _base.process_document


def _patch_base_collector() -> None:
    _base.CONFIG_PATH = CONFIG_PATH
    _base.FROZEN_STATUS = FROZEN_STATUS
    _base.PARSER_VERSION = PARSER_VERSION
    _base.REQUIRED_METRICS = REQUIRED_METRICS
    _base.extract_official_pdf_facts = extract_official_pdf_facts
    _base.process_document = process_document


def _validate_complete_checkpoint_migration() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    receipt_path = _base.project_path(
        config["artifacts"]["checkpoint_migration_receipt"]
    )
    if not receipt_path.exists():
        raise FileNotFoundError(f"V1.6完整检查点迁移收据不存在：{receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    supersedes = config["supersedes"]
    checks = {
        "status": receipt.get("status")
        == "PASS_V1_6_COMPLETE_CHECKPOINTS_MIGRATED_TO_V1_7",
        "parser_version": receipt.get("target_parser_version") == PARSER_VERSION,
        "source_complete_count": int(receipt.get("source_complete_count", -1))
        == int(supersedes["expected_complete_checkpoint_count"]),
        "migrated_complete_count": int(receipt.get("migrated_complete_count", -1))
        == int(supersedes["expected_complete_checkpoint_count"]),
        "source_incomplete_count": int(receipt.get("source_incomplete_count", -1))
        == int(supersedes["expected_incomplete_checkpoint_count"]),
        "migrated_incomplete_count": int(receipt.get("migrated_incomplete_count", -1))
        == 0,
        "market_price_read": receipt.get("market_price_read") is False,
        "future_return_read": receipt.get("future_return_read") is False,
        "return_evaluation": receipt.get("return_evaluation") == "NOT_ALLOWED",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"V1.6至V1.7检查点迁移收据不满足运行门：{failed}")


def process_document(
    row: dict[str, Any],
    config: dict[str, Any],
    *,
    reparse_incomplete: bool,
) -> dict[str, Any]:
    _patch_base_collector()
    return _BASE_PROCESS_DOCUMENT(
        row,
        config,
        reparse_incomplete=reparse_incomplete,
    )


def main() -> int:
    _patch_base_collector()
    _validate_complete_checkpoint_migration()
    return _base.main()


_patch_base_collector()


if __name__ == "__main__":
    raise SystemExit(main())
