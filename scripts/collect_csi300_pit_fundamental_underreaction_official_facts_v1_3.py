from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from research.csi300_pit_fundamental_underreaction_official_facts_v1_3 import (  # noqa: E402
    PARSER_VERSION,
    REQUIRED_METRICS,
    extract_official_pdf_facts,
)
from scripts import (  # noqa: E402
    collect_csi300_pit_fundamental_underreaction_official_facts_v1 as _base,
)


CONFIG_PATH = ROOT / "config/csi300_pit_fundamental_underreaction_official_facts_v1_3.yaml"
FROZEN_STATUS = (
    "FROZEN_FORMAT_AND_BOUNDED_IMAGE_STATEMENT_REVISION_AFTER_V1_2_PILOT_"
    "BEFORE_V1_3_BULK_AND_ANY_FUTURE_RETURN_READ"
)

_BASE_PROCESS_DOCUMENT = _base.process_document


def _patch_base_collector() -> None:
    _base.CONFIG_PATH = CONFIG_PATH
    _base.FROZEN_STATUS = FROZEN_STATUS
    _base.PARSER_VERSION = PARSER_VERSION
    _base.REQUIRED_METRICS = REQUIRED_METRICS
    _base.extract_official_pdf_facts = extract_official_pdf_facts
    _base.process_document = process_document


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
    return _base.main()


_patch_base_collector()


if __name__ == "__main__":
    raise SystemExit(main())
