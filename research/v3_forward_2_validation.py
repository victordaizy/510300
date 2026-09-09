"""在不修改V3_FORWARD_1的前提下，将同一验证引擎绑定到V3_FORWARD_2。"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import research.v3_forward_validation as base


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "v3_forward_2.yaml"
MANIFEST_FILE = ROOT / "config" / "v3_forward_2_manifest.json"
FORWARD_DIR = ROOT / "data" / "forward" / "v3_forward_2"
SIGNAL_LOG = FORWARD_DIR / "valuation_forward_signal_log.parquet"
OUTCOME_LOG = FORWARD_DIR / "valuation_forward_outcome_log.parquet"
SHADOW_LOG = FORWARD_DIR / "valuation_shadow_position_log.parquet"
STATUS_FILE = ROOT / "paper" / "v3_forward_2_latest_status.json"
REPORT_DIR = ROOT / "reports" / "forward" / "v3_forward_2"
WEIGHTS_FILE = ROOT / "data" / "raw" / "forward" / "v3_forward_2_weights.parquet"
FINANCIALS_FILE = ROOT / "data" / "raw" / "forward" / "v3_forward_2_financials.parquet"
FORWARD_CONSTITUENT_CLOSE_FILE = (
    ROOT / "data" / "raw" / "forward" / "v3_forward_2_constituent_close.parquet"
)
BASE_VALIDATE_MANIFEST = base.validate_frozen_manifest


def _validate_v2_manifest() -> dict[str, Any]:
    return BASE_VALIDATE_MANIFEST(MANIFEST_FILE)

OVERRIDES = {
    "CONFIG_FILE": CONFIG_FILE,
    "MANIFEST_FILE": MANIFEST_FILE,
    "FORWARD_DIR": FORWARD_DIR,
    "SIGNAL_LOG": SIGNAL_LOG,
    "OUTCOME_LOG": OUTCOME_LOG,
    "SHADOW_LOG": SHADOW_LOG,
    "STATUS_FILE": STATUS_FILE,
    "REPORT_DIR": REPORT_DIR,
    "WEIGHTS_FILE": WEIGHTS_FILE,
    "FINANCIALS_FILE": FINANCIALS_FILE,
    "FORWARD_CONSTITUENT_CLOSE_FILE": FORWARD_CONSTITUENT_CLOSE_FILE,
    "validate_frozen_manifest": _validate_v2_manifest,
}


@contextmanager
def activated_engine() -> Iterator[None]:
    original = {name: getattr(base, name) for name in OVERRIDES}
    try:
        for name, value in OVERRIDES.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(base, name, value)


def run_forward_cycle(*args: Any, **kwargs: Any) -> dict[str, Any]:
    with activated_engine():
        return base.run_forward_cycle(*args, **kwargs)


def validate_frozen_manifest() -> dict[str, Any]:
    with activated_engine():
        return base.validate_frozen_manifest()
