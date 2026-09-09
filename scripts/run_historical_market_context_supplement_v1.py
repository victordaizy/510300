"""运行历史行情情境补充层 V1。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from research.historical_market_context_supplement import (  # noqa: E402
    HistoricalContextError,
    build_current_market_context,
    build_historical_context_report,
    build_industry_price_context,
    file_sha256,
    prepare_market_features,
    render_historical_context_markdown,
    select_historical_analogues,
    validate_context_config,
    verify_manifest,
)


CONFIG_PATH = WORKSPACE_ROOT / "config" / "historical_market_context_supplement_v1.yaml"


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


def _relative(path: Path) -> str:
    return path.resolve().relative_to(WORKSPACE_ROOT.resolve()).as_posix()


def _load_calendar(config: Mapping[str, Any]) -> tuple[pd.DatetimeIndex, list[Path], list[Path]]:
    calendar_paths = sorted(WORKSPACE_ROOT.glob(config["inputs"]["trading_calendar_glob"]))
    metadata_paths = sorted(
        WORKSPACE_ROOT.glob(config["inputs"]["trading_calendar_metadata_glob"])
    )
    if not calendar_paths:
        raise HistoricalContextError("没有交易日历")
    metadata_by_stem = {
        path.name.replace(".metadata.json", ""): path for path in metadata_paths
    }
    date_series = []
    for path in calendar_paths:
        metadata_path = metadata_by_stem.get(path.stem)
        if metadata_path is None:
            raise HistoricalContextError(f"交易日历缺少元数据：{path}")
        metadata = _load_json(metadata_path)
        if metadata.get("status") != "PASS":
            raise HistoricalContextError(f"交易日历元数据未通过：{metadata_path}")
        if metadata.get("secondary_cross_check", {}).get("date_set_matches") is not True:
            raise HistoricalContextError(f"交易日历交叉核验未通过：{metadata_path}")
        if metadata.get("sha256") != file_sha256(path):
            raise HistoricalContextError(f"交易日历哈希与元数据不一致：{path}")
        frame = pd.read_csv(path)
        if "trade_date" not in frame.columns:
            raise HistoricalContextError(f"交易日历缺少trade_date：{path}")
        date_series.append(pd.to_datetime(frame["trade_date"], errors="coerce"))
    dates = pd.concat(date_series, ignore_index=True)
    if dates.isna().any() or dates.duplicated().any():
        raise HistoricalContextError("合并交易日历包含非法或重复日期")
    return pd.DatetimeIndex(sorted(dates.unique())), calendar_paths, metadata_paths


def _source(path: Path, frame: pd.DataFrame | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": _relative(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }
    if frame is not None:
        result["rows"] = len(frame)
        result["columns"] = list(frame.columns)
        if "date" in frame.columns:
            dates = pd.to_datetime(frame["date"], errors="coerce")
            result["date_min"] = dates.min().date().isoformat()
            result["date_max"] = dates.max().date().isoformat()
    return result


def _write_immutable(path: Path, content: str) -> str:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise HistoricalContextError(f"冻结补充结果已存在且不同，禁止覆盖：{path}")
        return "EXISTING_IDENTICAL"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return "CREATED"


def main() -> int:
    config = _load_yaml(CONFIG_PATH)
    rules, prohibitions, safety = validate_context_config(config)
    verify_manifest(WORKSPACE_ROOT, config["outputs"]["manifest"])
    frozen = config["frozen_prediction"]
    verify_manifest(WORKSPACE_ROOT, frozen["manifest"], frozen["manifest_sha256"])
    frozen_result = _load_json(_path(frozen["result"]))

    market_path = _path(config["inputs"]["market_total_return_daily"])
    industry_path = _path(config["inputs"]["industry_return_daily"])
    market = pd.read_parquet(market_path)
    industry = pd.read_parquet(industry_path)
    calendar, calendar_paths, metadata_paths = _load_calendar(config)

    features = prepare_market_features(market, config["as_of_date"], rules)
    market_context = build_current_market_context(
        features, rules, config["feature_blocks"]
    )
    analogues, analogue_summary = select_historical_analogues(
        features, config["feature_blocks"], rules
    )
    industry_context = build_industry_price_context(
        industry,
        frozen_result["industry_rows"],
        config["as_of_date"],
        calendar,
        rules,
    )
    provenance = {
        "market_total_return_daily": _source(market_path, market),
        "industry_return_daily": _source(industry_path, industry),
        "trading_calendars": [_source(path) for path in calendar_paths],
        "trading_calendar_metadata": [_source(path) for path in metadata_paths],
        "original_prediction_manifest": {
            "path": frozen["manifest"],
            "sha256": file_sha256(_path(frozen["manifest"])),
        },
        "historical_outcomes_read_after_protocol_freeze": True,
    }
    report = build_historical_context_report(
        config,
        rules,
        prohibitions,
        safety,
        market_context,
        analogues,
        analogue_summary,
        industry_context,
        frozen_result,
        provenance,
    )
    json_path = _path(config["outputs"]["json"])
    markdown_path = _path(config["outputs"]["markdown"])
    json_state = _write_immutable(
        json_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    markdown_state = _write_immutable(
        markdown_path, render_historical_context_markdown(report)
    )
    print(f"历史情境状态：{report['status']}")
    print(f"价格风险情境：{market_context['price_risk_state']}")
    print(f"相似历史样本：{analogue_summary['selected_count']}")
    print(f"行业情境：{industry_context['state']}")
    print(f"JSON：{json_path}（{json_state}）")
    print(f"Markdown：{markdown_path}（{markdown_state}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
