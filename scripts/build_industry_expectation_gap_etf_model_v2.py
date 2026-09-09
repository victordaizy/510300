"""构建行业预期差—ETF模型V2当前板块情境报告。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.index_driver_attribution import AttributionRules  # noqa: E402
from research.index_driver_attribution_v1_2 import (  # noqa: E402
    truncate_superseded_l1_intervals,
)
from research.index_driver_attribution_v1_3 import (  # noqa: E402
    build_index_driver_attribution_v1_3,
    restrict_industry_intervals_to_research_window,
)
from research.industry_expectation_gap_etf_model_v2 import (  # noqa: E402
    build_bucket_snapshot,
    build_citic_crosscheck,
    build_economic_bucket_daily,
    build_industry_flow_context,
    build_industry_snapshot,
    build_theme_snapshot,
    classify_bucket_forward_priorities,
    render_markdown,
    summarize_etf_flow,
    validate_official_cics_snapshot,
    validate_taxonomy,
)
from scripts.acquire_industry_expectation_gap_etf_model_v2 import (  # noqa: E402
    CITIC_L1_CODE_TO_NAME,
)


CONFIG_FILE = ROOT / "config" / "industry_expectation_gap_etf_model_v2.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _combine(existing: pd.DataFrame, extension: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return (
        pd.concat([existing, extension], ignore_index=True)
        .drop_duplicates(keys, keep="last")
        .sort_values(keys)
        .reset_index(drop=True)
    )


def _active_industry_mapping(intervals: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, str]:
    relevant = restrict_industry_intervals_to_research_window(intervals, as_of, as_of)
    resolved = truncate_superseded_l1_intervals(relevant)
    resolved["in_date"] = pd.to_datetime(resolved["in_date"], errors="coerce")
    resolved["out_date"] = pd.to_datetime(resolved["out_date"], errors="coerce")
    active = resolved.loc[
        resolved["in_date"].le(as_of)
        & (resolved["out_date"].isna() | resolved["out_date"].gt(as_of)),
        ["con_code", "industry_l1"],
    ].copy()
    if active["con_code"].duplicated().any():
        raise ValueError("当前中信行业映射仍存在重复有效区间")
    return dict(zip(active["con_code"].astype(str), active["industry_l1"].astype(str), strict=True))


def _attach_bucket_flow(
    buckets: list[dict[str, Any]], industry_flow: dict[str, dict[str, float | None]]
) -> None:
    for bucket in buckets:
        for window in (1, 5):
            net = sum(
                float(industry_flow.get(name, {}).get(f"flow_net_cny_{window}d") or 0.0)
                for name in bucket["industries"]
            )
            amount = sum(
                float(industry_flow.get(name, {}).get(f"flow_amount_cny_{window}d") or 0.0)
                for name in bucket["industries"]
            )
            bucket[f"flow_net_cny_{window}d"] = net
            bucket[f"flow_amount_cny_{window}d"] = amount
            bucket[f"flow_intensity_{window}d"] = net / amount if amount > 0 else None


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    as_of = pd.Timestamp(config["as_of_date"])
    acquisition_status_path = ROOT / config["acquired_inputs"]["acquisition_status"]
    acquisition_status = json.loads(acquisition_status_path.read_text(encoding="utf-8"))
    if acquisition_status.get("status") != "PASS":
        raise RuntimeError("V2数据补齐未通过，禁止构建报告")

    parent_result_path = ROOT / config["parent_frozen_result"]
    parent_context_path = ROOT / config["parent_historical_context"]
    parent_result = json.loads(parent_result_path.read_text(encoding="utf-8"))
    parent_context = json.loads(parent_context_path.read_text(encoding="utf-8"))
    if parent_result.get("final_state") != "NO_VIEW":
        raise RuntimeError("V1冻结结论已变化，V2不能静默继承")

    historical_daily = pd.read_parquet(ROOT / config["inputs"]["historical_constituent_daily"])
    extension = pd.read_parquet(ROOT / config["acquired_inputs"]["constituent_extension"])
    constituent_daily = _combine(historical_daily, extension, ["date", "con_code"])
    recent_start = pd.Timestamp(historical_daily["date"].max())
    recent_daily = constituent_daily.loc[
        pd.to_datetime(constituent_daily["date"]).ge(recent_start)
    ].copy()
    weights = pd.read_parquet(ROOT / config["inputs"]["historical_weights"])
    intervals = pd.read_parquet(ROOT / config["inputs"]["citic_industry_intervals"])
    rules = AttributionRules(
        start_date=recent_start,
        end_date=as_of,
        member_count_minimum=300,
        member_count_maximum=300,
        weight_sum_minimum=0.98,
        weight_sum_maximum=1.02,
        maximum_weight_age_days=int(config["rules"]["maximum_weight_age_calendar_days"]),
        minimum_price_coverage_weight=0.99,
        minimum_return_coverage_weight=float(
            config["rules"]["minimum_industry_return_coverage_weight"]
        ),
        minimum_industry_coverage_weight=0.95,
        maximum_identity_error=1.0e-12,
    )
    recent_attribution = build_index_driver_attribution_v1_3(
        weights, recent_daily, intervals, rules
    )
    extension_audit = recent_attribution.daily.loc[
        pd.to_datetime(recent_attribution.daily["date"]).gt(recent_start)
    ].copy()
    if extension_audit.empty or not extension_audit["valid_for_attribution"].fillna(False).all():
        failures = extension_audit.loc[
            ~extension_audit["valid_for_attribution"].fillna(False),
            ["date", "failure_category"],
        ]
        raise RuntimeError(f"V2最近行业归因未通过：{failures.to_dict(orient='records')}")

    historical_industry = pd.read_parquet(
        ROOT / config["inputs"]["historical_industry_attribution"]
    )
    historical_industry["date"] = pd.to_datetime(historical_industry["date"])
    new_industry = recent_attribution.industry.loc[
        recent_attribution.industry["date"].gt(historical_industry["date"].max())
    ].copy()
    industry_daily = _combine(
        historical_industry, new_industry, ["date", "industry_l1"]
    )
    industry_output = ROOT / config["outputs"]["industry_attribution"]
    atomic_parquet(industry_daily, industry_output)

    taxonomy = config["taxonomy"]
    forecast_rows = parent_result["industry_rows"]
    forecast_industries = [str(row["industry_l1"]) for row in forecast_rows]
    if len(forecast_industries) != int(config["rules"]["required_citic_industry_count"]):
        raise ValueError("冻结行业台账数量不是25")
    validate_taxonomy(forecast_industries, taxonomy)
    bucket_daily = build_economic_bucket_daily(industry_daily, taxonomy)
    bucket_output = ROOT / config["outputs"]["economic_bucket_daily"]
    atomic_parquet(bucket_daily, bucket_output)

    windows = [int(value) for value in config["rules"]["industry_return_windows"]]
    industry_snapshot = build_industry_snapshot(forecast_rows, industry_daily, windows)
    latest_returns = industry_daily.loc[
        pd.to_datetime(industry_daily["date"]).eq(as_of),
        ["industry_l1", "industry_return_1d"],
    ].rename(columns={"industry_return_1d": "return_1d"})
    industry_snapshot = industry_snapshot.merge(
        latest_returns, on="industry_l1", how="left", validate="one_to_one"
    )
    buckets = build_bucket_snapshot(industry_snapshot, bucket_daily, taxonomy, windows)

    moneyflow_historical = pd.read_parquet(
        ROOT / config["inputs"]["historical_component_moneyflow"]
    )
    moneyflow_extension = pd.read_parquet(
        ROOT / config["acquired_inputs"]["component_moneyflow_extension"]
    )
    moneyflow = _combine(moneyflow_historical, moneyflow_extension, ["date", "ts_code"])
    mapping = _active_industry_mapping(intervals, as_of)
    industry_flow = build_industry_flow_context(
        moneyflow,
        constituent_daily[["date", "con_code", "amount"]],
        mapping,
        as_of,
        [int(value) for value in config["rules"]["flow_windows"]],
    )
    for column in (
        "flow_intensity_1d",
        "flow_intensity_5d",
        "flow_net_cny_1d",
        "flow_net_cny_5d",
    ):
        industry_snapshot[column] = industry_snapshot["industry_l1"].map(
            lambda name, column=column: industry_flow.get(str(name), {}).get(column)
        )
    _attach_bucket_flow(buckets, industry_flow)
    themes = build_theme_snapshot(industry_snapshot, taxonomy, industry_flow)
    priorities = classify_bucket_forward_priorities(
        buckets, config["rules"]["qualitative_forward_rules"]
    )

    cics = pd.read_parquet(ROOT / config["acquired_inputs"]["cics_snapshot"])
    cics_l1 = validate_official_cics_snapshot(
        cics, float(config["rules"]["cics_weight_sum_tolerance_percent"])
    )
    cics_by_english = cics_l1.set_index("industry_name_en")["weight_pct"]
    information_technology = float(cics_by_english["Information Technology"])
    communication_services = float(cics_by_english["Communication Services"])

    citic_index = pd.read_parquet(
        ROOT / config["acquired_inputs"]["citic_index_crosscheck"]
    )
    crosscheck = build_citic_crosscheck(
        industry_snapshot, citic_index, CITIC_L1_CODE_TO_NAME, as_of
    )

    fund_share = _combine(
        pd.read_parquet(ROOT / config["inputs"]["historical_fund_share"]),
        pd.read_parquet(ROOT / config["acquired_inputs"]["fund_share_extension"]),
        ["date"],
    )
    margin = _combine(
        pd.read_parquet(ROOT / config["inputs"]["historical_margin"]),
        pd.read_parquet(ROOT / config["acquired_inputs"]["margin_extension"]),
        ["date"],
    )
    etf_daily = pd.read_parquet(ROOT / config["inputs"]["etf_daily"])
    etf_flow = summarize_etf_flow(fund_share, margin, etf_daily, as_of)

    core_tech_weight = next(
        row["index_weight"] for row in buckets if row["bucket_id"] == "CORE_TECHNOLOGY"
    )
    digital_weight = next(
        row["index_weight"] for row in buckets if row["bucket_id"] == "DIGITAL_COMMUNICATION"
    )
    fields_still_unobserved = [
        "CURRENT_NATIONAL_TEAM_HOLDINGS",
        "CURRENT_NATIONAL_TEAM_ACTIVITY",
        "POINT_IN_TIME_CONSENSUS_VINTAGE",
        "THEME_LEVEL_REVENUE_EXPOSURE",
        "MARGIN_DETAIL_2026_08_18_PROVIDER_NOT_YET_AVAILABLE",
    ]
    report = {
        "model_version": config["version"],
        "status": "CURRENT_SECTOR_CONTEXT_READY_ORIGINAL_NO_VIEW_UNCHANGED",
        "as_of_date": config["as_of_date"],
        "information_cutoff": config["information_cutoff"],
        "original_prediction_state": parent_result["final_state"],
        "original_prediction_changed": False,
        "headline": (
            f"官方CICS信息技术加通信服务占沪深300的"
            f"{(information_technology + communication_services):.2f}%；"
            f"中信25行业研究口径下，核心科技加数字内容与通信覆盖"
            f"{(core_tech_weight + digital_weight):.2%}。"
            "当前价格、订单分类资金流和ETF份额已补到最近可得日，但这些事实不自动把V1的NO_VIEW升级成方向预测。"
        ),
        "taxonomy_contract": {
            "official_broad_sector": "CSI_CICS_L1_MUTUALLY_EXCLUSIVE_100_PERCENT",
            "economic_bucket": "MUTUALLY_EXCLUSIVE_BUCKETS_COVER_CURRENT_25_AND_HISTORICAL_EXIT_INDUSTRIES_ONCE",
            "theme": "10_OVERLAPPING_CHAINS_NEVER_CROSS_SUM",
        },
        "official_cics_l1": cics_l1.to_dict(orient="records"),
        "official_technology_crosscheck": {
            "information_technology_weight_pct": information_technology,
            "communication_services_weight_pct": communication_services,
            "combined_weight_pct": information_technology + communication_services,
            "snapshot_date": cics_l1["date"].max().date().isoformat(),
        },
        "economic_buckets": buckets,
        "qualitative_forward_priorities": priorities,
        "themes": themes,
        "industry_rows": industry_snapshot.to_dict(orient="records"),
        "citic_return_crosscheck": crosscheck,
        "etf_flow_and_liquidity": etf_flow,
        "market_price_context": parent_context["market_price_context"],
        "historical_analogue_summary": parent_context["historical_analogue_summary"],
        "data_freshness": {
            "constituent_total_return": config["as_of_date"],
            "industry_attribution": config["as_of_date"],
            "component_moneyflow": config["as_of_date"],
            "fund_share": etf_flow["fund_share_as_of_date"],
            "margin_detail": etf_flow["margin_as_of_date"],
            "cics_official_snapshot": cics_l1["date"].max().date().isoformat(),
            "weight_snapshot": parent_result["industry_aggregation"]["sector_snapshot"]["snapshot_date"],
        },
        "fields_supplemented": [
            "CURRENT_CONSTITUENT_TOTAL_RETURN_THROUGH_2026_08_18",
            "CURRENT_INDUSTRY_PRICE_CONTEXT_THROUGH_2026_08_18",
            "CSI_CICS_L1_OFFICIAL_BROAD_SECTOR_WEIGHTS",
            "MUTUALLY_EXCLUSIVE_ECONOMIC_BUCKETS",
            "OVERLAPPING_THEME_CHAINS_WITHOUT_CROSS_SUM",
            "COMPONENT_ACTIVE_ORDER_CLASSIFICATION_FLOW",
            "510300_FUND_SHARE_ACTIVITY",
            "510300_MARGIN_DETAIL_THROUGH_PROVIDER_LATEST_DATE",
            "CITIC_L1_MARKET_INDEX_DIRECTION_CROSSCHECK",
        ],
        "fields_still_unobserved": fields_still_unobserved,
        "safety": config["safety"],
        "provenance": {
            "config": config["version"],
            "acquisition_status": acquisition_status,
            "parent_result_sha256": sha256(parent_result_path),
            "parent_context_sha256": sha256(parent_context_path),
            "industry_output_sha256": sha256(industry_output),
            "bucket_output_sha256": sha256(bucket_output),
        },
        "interpretation": (
            "V2解决了板块分割和当前数据缺口，但没有凭空制造未来信息。"
            "定性判断仍来自冻结的行业证据台账；价格和资金只用于判断是否已被计价及当前拥挤程度。"
            "国家队身份、点时一致预期和主题收入暴露仍缺可靠证据，因此保持研究观察，不做仓位映射。"
        ),
    }
    report = _safe(report)
    json_path = ROOT / config["outputs"]["json"]
    markdown_path = ROOT / config["outputs"]["markdown"]
    atomic_text(json.dumps(report, ensure_ascii=False, indent=2), json_path)
    atomic_text(render_markdown(report), markdown_path)
    print(
        json.dumps(
            {
                "status": report["status"],
                "as_of_date": report["as_of_date"],
                "official_technology_weight_pct": report["official_technology_crosscheck"]["combined_weight_pct"],
                "research_broad_technology_weight": core_tech_weight + digital_weight,
                "citic_crosscheck_same_sign_ratio": crosscheck["same_sign_ratio"],
                "output_json": json_path.relative_to(ROOT).as_posix(),
                "output_markdown": markdown_path.relative_to(ROOT).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
