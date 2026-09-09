"""信息传播研究的分钟时间戳硬审计。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from research.information_propagation_alpha import select_causal_futures_contract


def _normalize(data: pd.DataFrame, label: str) -> pd.DataFrame:
    if "trade_time" not in data.columns:
        raise ValueError(f"{label}缺少 trade_time")
    output = data.copy()
    output["trade_time"] = pd.to_datetime(output["trade_time"], errors="coerce")
    if output["trade_time"].isna().any():
        raise ValueError(f"{label}存在无法解析的时间戳")
    if output["trade_time"].dt.tz is not None:
        output["trade_time"] = (
            output["trade_time"].dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
        )
    output["session_date"] = output["trade_time"].dt.normalize()
    return output


def _anchors(timestamps: pd.Series) -> dict[str, str | int | None]:
    values = pd.Series(pd.to_datetime(timestamps).drop_duplicates().sort_values())
    if values.empty:
        return {"bar_count": 0, "first": None, "morning_last": None, "afternoon_first": None, "last": None}
    clock = values.dt.time
    morning = values.loc[clock <= pd.Timestamp("11:30:00").time()]
    afternoon = values.loc[clock >= pd.Timestamp("13:00:00").time()]
    return {
        "bar_count": int(len(values)),
        "first": values.iloc[0].isoformat(),
        "morning_last": None if morning.empty else morning.iloc[-1].isoformat(),
        "afternoon_first": None if afternoon.empty else afternoon.iloc[0].isoformat(),
        "last": values.iloc[-1].isoformat(),
    }


def audit_timestamp_alignment(
    etf: pd.DataFrame,
    index: pd.DataFrame,
    futures: pd.DataFrame | None,
    components: pd.DataFrame,
    sample_trading_days: int = 10,
    deterministic_seed: int = 20260814,
    required_anchor_times: tuple[str, ...] = ("09:30:00", "11:30:00", "13:01:00", "15:00:00"),
    expected_etf_component_bars_per_day: int = 241,
    manual_semantics_status: str = "PENDING",
) -> dict[str, Any]:
    """验证同一标签能否安全用于因果连接；IF 允许存在 ETF 交易时段外的额外分钟。"""

    sources: dict[str, pd.DataFrame] = {
        "etf_510300": _normalize(etf, "510300"),
        "index_000300": _normalize(index, "000300"),
        "components_top50_union": _normalize(components, "Top50成分股"),
    }
    if futures is not None:
        sources["if_selected"] = _normalize(select_causal_futures_contract(futures), "IF")
    if "con_code" not in sources["components_top50_union"].columns:
        raise ValueError("Top50成分股缺少 con_code")
    available_days = {
        name: set(frame["session_date"].unique()) for name, frame in sources.items()
    }
    common_days = sorted(set.intersection(*available_days.values()))
    if len(common_days) < sample_trading_days:
        return {
            "status": "FAIL_INSUFFICIENT_COMMON_DAYS",
            "common_trading_day_count": len(common_days),
            "required_sample_trading_days": sample_trading_days,
        }
    rng = np.random.default_rng(deterministic_seed)
    sampled_days = sorted(rng.choice(common_days, size=sample_trading_days, replace=False))
    failures: list[dict[str, Any]] = []
    day_reports: list[dict[str, Any]] = []
    required_clocks = {pd.Timestamp(value).time() for value in required_anchor_times}

    for day in sampled_days:
        day = pd.Timestamp(day)
        etf_times = sources["etf_510300"].loc[
            sources["etf_510300"]["session_date"] == day, "trade_time"
        ].drop_duplicates()
        baseline = set(etf_times)
        report: dict[str, Any] = {"session_date": day.date().isoformat(), "sources": {}}
        market_source_names = ["etf_510300", "index_000300"]
        if "if_selected" in sources:
            market_source_names.append("if_selected")
        for name in market_source_names:
            frame = sources[name]
            times = frame.loc[frame["session_date"] == day, "trade_time"].drop_duplicates()
            report["sources"][name] = _anchors(times)
            missing_from_etf_timeline = sorted(baseline.difference(set(times)))
            report["sources"][name]["missing_etf_timeline_count"] = len(missing_from_etf_timeline)
            if missing_from_etf_timeline:
                failures.append(
                    {
                        "session_date": day.date().isoformat(),
                        "source": name,
                        "reason": "MISSING_ETF_TIMELINE_TIMESTAMPS",
                        "count": len(missing_from_etf_timeline),
                    }
                )
        etf_clocks = set(etf_times.dt.time)
        missing_anchors = sorted(str(value) for value in required_clocks.difference(etf_clocks))
        if len(etf_times) != expected_etf_component_bars_per_day or missing_anchors:
            failures.append(
                {
                    "session_date": day.date().isoformat(),
                    "source": "etf_510300",
                    "reason": "ETF_BAR_COUNT_OR_ANCHOR_MISMATCH",
                    "bar_count": len(etf_times),
                    "missing_anchors": missing_anchors,
                }
            )

        component_day = sources["components_top50_union"].loc[
            sources["components_top50_union"]["session_date"] == day
        ]
        component_failures = 0
        component_counts: list[int] = []
        for symbol, symbol_data in component_day.groupby("con_code"):
            symbol_times = set(symbol_data["trade_time"].drop_duplicates())
            component_counts.append(len(symbol_times))
            if symbol_times != baseline:
                component_failures += 1
                failures.append(
                    {
                        "session_date": day.date().isoformat(),
                        "source": str(symbol),
                        "reason": "COMPONENT_TIMESTAMP_SET_DIFFERS_FROM_ETF",
                        "missing_count": len(baseline.difference(symbol_times)),
                        "extra_count": len(symbol_times.difference(baseline)),
                    }
                )
        report["sources"]["components_top50_union"] = {
            "symbol_count": int(component_day["con_code"].nunique()),
            "symbols_with_timestamp_mismatch": component_failures,
            "minimum_bar_count": min(component_counts) if component_counts else 0,
            "maximum_bar_count": max(component_counts) if component_counts else 0,
        }
        day_reports.append(report)

    payload: dict[str, Any] = {
        "target_asset": "510300.SH",
        "index_role": "仅作价格发现对照，不是研究收益或交易标的",
        "sampled_trading_days": [pd.Timestamp(day).date().isoformat() for day in sampled_days],
        "manual_semantics_status": manual_semantics_status,
        "alignment_failure_count": len(failures),
        "failures": failures,
        "days": day_reports,
    }
    if failures:
        payload["status"] = "FAIL_TIMESTAMP_ALIGNMENT"
    elif manual_semantics_status != "CONFIRMED":
        payload["status"] = "BLOCKED_MANUAL_TIMESTAMP_SEMANTICS"
    else:
        payload["status"] = "PASS"
    return payload
