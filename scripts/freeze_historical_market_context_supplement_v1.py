"""在读取历史相似行情后续结果前冻结情境补充协议。"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from research.historical_market_context_supplement import (  # noqa: E402
    HistoricalContextError,
    build_current_market_context,
    build_industry_price_context,
    file_sha256,
    prepare_market_features,
    validate_context_config,
    verify_manifest,
)


CONFIG_RELATIVE = "config/historical_market_context_supplement_v1.yaml"
PROTOCOL_FILES = [
    "docs/HISTORICAL_MARKET_CONTEXT_SUPPLEMENT_V1_SPEC.md",
    CONFIG_RELATIVE,
    "research/historical_market_context_supplement.py",
    "scripts/run_historical_market_context_supplement_v1.py",
    "scripts/freeze_historical_market_context_supplement_v1.py",
    "tests/test_historical_market_context_supplement.py",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise HistoricalContextError(f"YAML顶层必须是对象：{path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise HistoricalContextError(f"JSON顶层必须是对象：{path}")
    return value


def _path(relative: str) -> Path:
    root = WORKSPACE_ROOT.resolve()
    result = (root / relative).resolve()
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise HistoricalContextError(f"路径越出工作区：{relative}") from exc
    return result


def _git_text(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=WORKSPACE_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _frozen_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(WORKSPACE_ROOT).as_posix(),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def main() -> int:
    config = _load_yaml(_path(CONFIG_RELATIVE))
    rules, _, _ = validate_context_config(config)
    frozen = config["frozen_prediction"]
    verify_manifest(WORKSPACE_ROOT, frozen["manifest"], frozen["manifest_sha256"])
    frozen_result = _load_json(_path(frozen["result"]))

    market_path = _path(config["inputs"]["market_total_return_daily"])
    industry_path = _path(config["inputs"]["industry_return_daily"])
    market = pd.read_parquet(market_path)
    industry = pd.read_parquet(industry_path)
    features = prepare_market_features(market, config["as_of_date"], rules)
    market_context = build_current_market_context(
        features, rules, config["feature_blocks"]
    )

    calendar_paths = sorted(WORKSPACE_ROOT.glob(config["inputs"]["trading_calendar_glob"]))
    metadata_paths = sorted(
        WORKSPACE_ROOT.glob(config["inputs"]["trading_calendar_metadata_glob"])
    )
    if not calendar_paths or not metadata_paths:
        raise HistoricalContextError("冻结时缺少交易日历或元数据")
    calendar_dates = []
    for path in calendar_paths:
        frame = pd.read_csv(path)
        calendar_dates.append(pd.to_datetime(frame["trade_date"], errors="coerce"))
    calendar = pd.DatetimeIndex(sorted(pd.concat(calendar_dates).unique()))
    industry_context = build_industry_price_context(
        industry,
        frozen_result["industry_rows"],
        config["as_of_date"],
        calendar,
        rules,
    )

    protocol_paths = [_path(relative) for relative in PROTOCOL_FILES]
    input_paths = [
        market_path,
        industry_path,
        _path(frozen["manifest"]),
        _path(frozen["result"]),
        *calendar_paths,
        *metadata_paths,
    ]
    all_paths = sorted(set(protocol_paths + input_paths), key=lambda path: str(path).lower())
    for path in all_paths:
        if not path.is_file():
            raise HistoricalContextError(f"待冻结文件不存在：{path}")

    complete_features = [
        feature for values in config["feature_blocks"].values() for feature in values
    ]
    current_position = len(features) - 1
    maximum_candidate_position = current_position - rules.analogue_current_embargo_trading_days
    candidate_count = int(
        features.iloc[: maximum_candidate_position + 1]
        .dropna(subset=complete_features)
        .shape[0]
    )
    if candidate_count < rules.minimum_analogue_count:
        raise HistoricalContextError("冻结前完整历史候选不足")
    manifest = {
        "manifest_version": "HISTORICAL_MARKET_CONTEXT_SUPPLEMENT_V1_MANIFEST",
        "status": "FROZEN_BEFORE_ANALOGUE_OUTCOME_READ",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "as_of_date": config["as_of_date"],
        "frozen_files": [_frozen_record(path) for path in all_paths],
        "pre_outcome_validation": {
            "market_feature_date": market_context["as_of_date"],
            "market_feature_complete": True,
            "candidate_count_before_distance_and_spacing": candidate_count,
            "industry_data_as_of": industry_context["data_as_of_date"],
            "industry_stale_trading_days": industry_context["stale_trading_days"],
            "analogue_outcomes_read": False,
        },
        "original_prediction_manifest": {
            "path": frozen["manifest"],
            "sha256": file_sha256(_path(frozen["manifest"])),
        },
        "prohibitions": {
            "funding_liquidity_backfill": False,
            "equity_flow_backfill": False,
            "national_team_identity_inference": False,
            "original_prediction_change": False,
            "position_mapping": False,
            "order_generation": False,
            "broker_connection": False,
            "live_trading": False,
        },
        "git": {
            "commit": _git_text("rev-parse", "HEAD"),
            "branch": _git_text("branch", "--show-current"),
            "dirty_before_freeze": bool(_git_text("status", "--porcelain")),
        },
    }
    manifest_path = _path(config["outputs"]["manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"历史情境协议已冻结：{manifest_path}")
    print(f"冻结前历史候选数：{candidate_count}")
    print(f"相似行情后续结果读取：False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
