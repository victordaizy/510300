"""筛选跨ETF官方份额变化驱动的510300二元满仓/空仓固定规则。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from research.intraday_binary_livermore_screen_v1 import (
    CostModel,
    build_benchmark_returns,
    simulate_candidate,
    summarize_period,
    trading_date_with_offset,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "config" / "510300_cross_etf_forced_flow_binary_screen_v1.yaml"
)
CANDIDATE_CONTRACT_PATH = (
    ROOT / "config" / "510300_cross_etf_forced_flow_binary_screen_v1_candidates.yaml"
)
CANDIDATE_CONTRACT_SHA256 = (
    "879800a413eac26c137a56d2cb4ea94c9a544a7a1538c7af09ed36550ff6a742"
)
STUDY_ID = "510300_CROSS_ETF_FORCED_FLOW_BINARY_SCREEN_V1"
EXPECTED_CANDIDATES = [
    "BROAD_FLOW1W_BOTTOM10_CASH_WEEK",
    "BROAD_FLOW4W_BOTTOM10_CASH_WEEK",
    "BROAD_FLOW1W_DUALTAIL_HYSTERESIS",
    "HS300_FLOW1W_DUALTAIL_HYSTERESIS",
    "BROAD_HS300_DUAL_CONFIRM_20_80_HYSTERESIS",
]


def sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    """将 NumPy、Pandas 和日期标量转换成 JSON 值。"""

    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"无法JSON序列化：{type(value).__name__}")


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    """原子写入 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    """原子写入 Parquet。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def load_candidate_contract() -> dict[str, Any]:
    """读取并核对在官方份额历史采集前固定的候选合同。"""

    if sha256_file(CANDIDATE_CONTRACT_PATH) != CANDIDATE_CONTRACT_SHA256:
        raise ValueError("官方份额历史采集前固定的候选合同发生漂移")
    contract = yaml.safe_load(CANDIDATE_CONTRACT_PATH.read_text(encoding="utf-8"))
    protocol = contract["protocol"]
    if protocol["study_id"] != STUDY_ID:
        raise ValueError("候选合同研究编号不匹配")
    if (
        protocol["state"]
        != "CANDIDATE_FAMILY_FIXED_BEFORE_SSE_WEEKLY_HISTORY_COLLECTION"
    ):
        raise ValueError("候选合同状态无效")
    if not bool(protocol["no_parameter_rescue"]):
        raise ValueError("候选合同必须禁止参数营救")
    candidate_ids = [item["candidate_id"] for item in contract["candidates"]]
    if candidate_ids != EXPECTED_CANDIDATES:
        raise ValueError("固定候选或顺序发生漂移")
    if list(contract["scope"]["allowed_target_states"]) != [0, 1]:
        raise ValueError("目标状态必须严格为0和1")
    if list(contract["scope"]["allowed_holdings"]) != ["510300.SH", "CASH_CNY"]:
        raise ValueError("持仓范围必须严格为510300和现金")
    if float(contract["objective"]["minimum_annualized_excess"]) != 0.20:
        raise ValueError("年化净超额硬门必须为20%")
    if float(contract["objective"]["minimum_rolling_excess_median"]) != 0.20:
        raise ValueError("滚动净超额硬门必须为20%")
    return contract


def load_config(path: Path = CONFIG_PATH) -> tuple[dict[str, Any], dict[str, Any]]:
    """读取冻结前主协议并核对候选合同绑定。"""

    contract = load_candidate_contract()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    protocol = config["protocol"]
    if protocol["study_id"] != STUDY_ID:
        raise ValueError("主协议研究编号不匹配")
    if protocol["state"] != "PREFREEZE_IMPLEMENTATION_COMPLETE":
        raise ValueError("主协议尚未达到冻结前实现完成状态")
    if protocol["candidate_contract_sha256"] != CANDIDATE_CONTRACT_SHA256:
        raise ValueError("主协议候选合同哈希不匹配")
    if not bool(protocol["no_parameter_rescue"]):
        raise ValueError("主协议必须禁止参数营救")
    return config, contract


def verify_freeze_manifest(config: dict[str, Any]) -> dict[str, Any]:
    """首跑前核对协议、实现、测试、依赖和输入哈希。"""

    manifest_path = ROOT / config["artifacts"]["freeze_manifest"]
    if not manifest_path.exists():
        raise FileNotFoundError("冻结清单不存在")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "FROZEN_BEFORE_FIRST_SCREEN_RUN":
        raise ValueError("冻结清单状态无效")
    if sha256_file(CONFIG_PATH) != manifest["protocol_sha256"]:
        raise ValueError("冻结后主协议漂移")
    if sha256_file(CANDIDATE_CONTRACT_PATH) != manifest["candidate_contract_sha256"]:
        raise ValueError("冻结后候选合同漂移")
    for group in ["source_artifacts", "input_artifacts"]:
        for relative, expected in manifest[group].items():
            path = ROOT / relative
            if not path.exists() or sha256_file(path) != expected:
                raise ValueError(f"冻结工件漂移：{relative}")
    return manifest


def rolling_last_percentile(
    values: pd.Series,
    *,
    window: int,
    minimum_observations: int,
) -> pd.Series:
    """计算当前值在仅含当前和过去有限观测中的滚动经验分位。"""

    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    output = np.full(len(numeric), np.nan, dtype=float)
    for index, current in enumerate(numeric):
        if not np.isfinite(current):
            continue
        start = max(0, index - window + 1)
        history = numeric[start : index + 1]
        history = history[np.isfinite(history)]
        if len(history) >= minimum_observations:
            output[index] = float(np.mean(history <= current))
    return pd.Series(output, index=values.index, dtype=float)


def _pool_weekly_flow(
    merged: pd.DataFrame,
    symbols: list[str],
    prefix: str,
) -> pd.DataFrame:
    """按前周规模加权汇总指定ETF池的周度创造赎回代理。"""

    pool = merged.loc[merged["ts_code"].isin(symbols)].copy()
    counts = pool.groupby("date")["ts_code"].nunique()
    if not counts.eq(len(symbols)).all():
        raise ValueError(f"{prefix}在至少一个周度日期缺少基金")
    grouped = pool.groupby("date", sort=True).agg(
        **{
            f"{prefix}_weekly_flow_cny": (
                "weekly_flow_cny",
                lambda values: values.sum(min_count=1),
            ),
            f"{prefix}_prior_aum_cny": (
                "prior_aum_cny",
                lambda values: values.sum(min_count=len(symbols)),
            ),
            f"{prefix}_positive_flow_fund_count": (
                "weekly_flow_cny",
                lambda values: int(values.gt(0.0).sum()),
            ),
        }
    )
    grouped[f"{prefix}_flow_ratio_1w"] = (
        grouped[f"{prefix}_weekly_flow_cny"]
        / grouped[f"{prefix}_prior_aum_cny"]
    )
    return grouped.reset_index()


def build_weekly_features(
    shares: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    broad_symbols: list[str],
    hs300_symbols: list[str],
    rolling_window: int,
    minimum_observations: int,
) -> pd.DataFrame:
    """构造固定大ETF池和沪深300子池的周度份额流量特征。"""

    required_shares = {"date", "ts_code", "fund_shares"}
    required_prices = {"date", "con_code", "raw_close"}
    if missing := required_shares - set(shares.columns):
        raise ValueError(f"官方份额缺少字段：{sorted(missing)}")
    if missing := required_prices - set(prices.columns):
        raise ValueError(f"ETF价格缺少字段：{sorted(missing)}")
    share_frame = shares[["date", "ts_code", "fund_shares"]].copy()
    price_frame = prices[["date", "con_code", "raw_close"]].rename(
        columns={"con_code": "ts_code"}
    )
    for frame in [share_frame, price_frame]:
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    share_frame["fund_shares"] = pd.to_numeric(
        share_frame["fund_shares"], errors="raise"
    )
    price_frame["raw_close"] = pd.to_numeric(price_frame["raw_close"], errors="raise")
    if share_frame[["date", "ts_code"]].duplicated().any():
        raise ValueError("官方周度份额存在重复证券日期")
    price_frame = price_frame.loc[
        price_frame["ts_code"].isin(broad_symbols)
        & price_frame["date"].isin(share_frame["date"])
    ]
    if price_frame[["date", "ts_code"]].duplicated().any():
        raise ValueError("周度ETF价格存在重复证券日期")
    merged = share_frame.merge(
        price_frame,
        on=["date", "ts_code"],
        how="left",
        validate="one_to_one",
    ).sort_values(["ts_code", "date"], kind="mergesort")
    if merged["raw_close"].isna().any():
        missing = merged.loc[merged["raw_close"].isna(), ["date", "ts_code"]]
        raise ValueError(f"官方份额缺少同日收盘价：{missing.head().to_dict('records')}")
    if (merged[["fund_shares", "raw_close"]] <= 0.0).any().any():
        raise ValueError("官方份额或ETF收盘价存在非正值")
    merged["previous_fund_shares"] = merged.groupby("ts_code")[
        "fund_shares"
    ].shift(1)
    merged["weekly_flow_cny"] = (
        merged["fund_shares"] - merged["previous_fund_shares"]
    ) * merged["raw_close"]
    merged["prior_aum_cny"] = merged["previous_fund_shares"] * merged["raw_close"]

    broad = _pool_weekly_flow(merged, broad_symbols, "broad")
    hs300 = _pool_weekly_flow(merged, hs300_symbols, "hs300")
    features = broad.merge(hs300, on="date", how="inner", validate="one_to_one")
    features.sort_values("date", kind="mergesort", inplace=True)
    features.reset_index(drop=True, inplace=True)
    features["broad_flow_ratio_4w"] = features["broad_flow_ratio_1w"].rolling(
        4, min_periods=4
    ).sum()
    features["broad_flow_rank_1w"] = rolling_last_percentile(
        features["broad_flow_ratio_1w"],
        window=rolling_window,
        minimum_observations=minimum_observations,
    )
    features["broad_flow_rank_4w"] = rolling_last_percentile(
        features["broad_flow_ratio_4w"],
        window=rolling_window,
        minimum_observations=minimum_observations,
    )
    features["hs300_flow_rank_1w"] = rolling_last_percentile(
        features["hs300_flow_ratio_1w"],
        window=rolling_window,
        minimum_observations=minimum_observations,
    )
    features["historical_publication_timestamp_verified"] = False
    features["execution_earliest"] = "T_PLUS_2_OPEN"
    return features


def _dual_tail_state(
    lower_rank: pd.Series,
    upper_rank: pd.Series | None = None,
    *,
    lower_threshold: float,
    upper_threshold: float,
) -> np.ndarray:
    """从满仓开始，按固定双尾阈值形成无参数更新的二元状态机。"""

    lower = pd.to_numeric(lower_rank, errors="coerce").to_numpy(dtype=float)
    upper = lower if upper_rank is None else pd.to_numeric(
        upper_rank, errors="coerce"
    ).to_numpy(dtype=float)
    states = np.ones(len(lower), dtype=np.int8)
    current = 1
    for index, (low_value, high_value) in enumerate(zip(lower, upper, strict=True)):
        if np.isfinite(low_value) and np.isfinite(high_value):
            if low_value <= lower_threshold and high_value <= lower_threshold:
                current = 0
            elif low_value >= upper_threshold and high_value >= upper_threshold:
                current = 1
        states[index] = current
    return states


def build_candidate_weekly_states(features: pd.DataFrame) -> pd.DataFrame:
    """严格按预冻结的五条规则生成周度0/1状态。"""

    required = {
        "date",
        "broad_flow_rank_1w",
        "broad_flow_rank_4w",
        "hs300_flow_rank_1w",
    }
    if missing := required - set(features.columns):
        raise ValueError(f"周度特征缺少字段：{sorted(missing)}")
    states = features[["date"]].copy()
    broad_1w = pd.to_numeric(features["broad_flow_rank_1w"], errors="coerce")
    broad_4w = pd.to_numeric(features["broad_flow_rank_4w"], errors="coerce")
    hs300_1w = pd.to_numeric(features["hs300_flow_rank_1w"], errors="coerce")
    states["BROAD_FLOW1W_BOTTOM10_CASH_WEEK"] = np.where(
        broad_1w.notna() & broad_1w.le(0.10), 0, 1
    ).astype(np.int8)
    states["BROAD_FLOW4W_BOTTOM10_CASH_WEEK"] = np.where(
        broad_4w.notna() & broad_4w.le(0.10), 0, 1
    ).astype(np.int8)
    states["BROAD_FLOW1W_DUALTAIL_HYSTERESIS"] = _dual_tail_state(
        broad_1w,
        lower_threshold=0.10,
        upper_threshold=0.90,
    )
    states["HS300_FLOW1W_DUALTAIL_HYSTERESIS"] = _dual_tail_state(
        hs300_1w,
        lower_threshold=0.10,
        upper_threshold=0.90,
    )
    states["BROAD_HS300_DUAL_CONFIRM_20_80_HYSTERESIS"] = _dual_tail_state(
        broad_1w,
        hs300_1w,
        lower_threshold=0.20,
        upper_threshold=0.80,
    )
    for candidate_id in EXPECTED_CANDIDATES:
        states[candidate_id] = states[candidate_id].astype(np.int8)
        if not set(states[candidate_id].unique()).issubset({0, 1}):
            raise AssertionError(f"{candidate_id}不是严格二元状态")
    return states


def align_weekly_states_to_daily_targets(
    market: pd.DataFrame,
    weekly_states: pd.DataFrame,
) -> pd.DataFrame:
    """周末份额在下一交易日收盘才视为可见，随后第二日开盘执行。"""

    dates = pd.DatetimeIndex(
        pd.to_datetime(market["trade_date"], errors="raise").dt.normalize()
    )
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("执行交易日必须唯一且升序")
    events: list[dict[str, Any]] = []
    for row in weekly_states.itertuples(index=False):
        feature_date = pd.Timestamp(row.date).normalize()
        availability_index = int(dates.searchsorted(feature_date, side="right"))
        if availability_index >= len(dates):
            continue
        availability_date = pd.Timestamp(dates[availability_index])
        event = {
            "availability_date": availability_date,
            "source_feature_date": feature_date,
        }
        for candidate_id in EXPECTED_CANDIDATES:
            event[candidate_id] = int(getattr(row, candidate_id))
        events.append(event)
    event_frame = pd.DataFrame(events)
    if event_frame.empty or event_frame["availability_date"].duplicated().any():
        raise ValueError("周度状态没有形成唯一的T+1收盘可用事件")
    daily = pd.DataFrame({"trade_date": dates})
    daily = daily.merge(
        event_frame,
        left_on="trade_date",
        right_on="availability_date",
        how="left",
        validate="one_to_one",
    )
    daily["weekly_feature_became_available"] = daily["source_feature_date"].notna()
    daily["latest_source_feature_date"] = daily["source_feature_date"].ffill()
    for candidate_id in EXPECTED_CANDIDATES:
        daily[candidate_id] = daily[candidate_id].ffill().fillna(1).astype(np.int8)
    daily.drop(columns="availability_date", inplace=True)
    availability_rows = daily.loc[daily["weekly_feature_became_available"]]
    if not (
        availability_rows["source_feature_date"] < availability_rows["trade_date"]
    ).all():
        raise AssertionError("历史份额特征未等待到下一交易日收盘")
    return daily


def required_columns(frame: pd.DataFrame, required: list[str], label: str) -> None:
    """核对输入字段。"""

    if missing := set(required) - set(frame.columns):
        raise ValueError(f"{label}缺少字段：{sorted(missing)}")


def load_inputs(
    config: dict[str, Any], contract: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """按冻结哈希读取官方份额、价格、执行行情、基准和分红。"""

    contracts = config["data_contracts"]
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name, data_contract in contracts.items():
        path = ROOT / data_contract["file"]
        if not path.exists():
            raise FileNotFoundError(f"固定输入不存在：{data_contract['file']}")
        actual = sha256_file(path)
        if actual != data_contract["required_sha256"]:
            raise ValueError(f"固定输入哈希漂移：{data_contract['file']}")
        paths[name] = path
        hashes[data_contract["file"]] = actual

    shares = pd.read_parquet(paths["official_weekly_shares"])
    share_report = json.loads(paths["official_share_report"].read_text(encoding="utf-8"))
    prices = pd.read_parquet(paths["etf_price_panel"])
    market = pd.read_parquet(paths["execution_daily"])
    benchmark = pd.read_parquet(paths["benchmark_total_return"])
    dividends = pd.read_csv(paths["cash_distributions"])
    required_columns(
        shares,
        contracts["official_weekly_shares"]["required_columns"],
        "上交所官方周度份额",
    )
    required_columns(
        prices,
        contracts["etf_price_panel"]["required_columns"],
        "ETF价格面板",
    )
    required_columns(
        market,
        contracts["execution_daily"]["required_columns"],
        "510300执行日线",
    )
    required_columns(
        benchmark,
        contracts["benchmark_total_return"]["required_columns"],
        "H00300全收益基准",
    )
    required_columns(
        dividends,
        contracts["cash_distributions"]["required_columns"],
        "510300现金分红",
    )
    shares["date"] = pd.to_datetime(shares["date"], errors="raise").dt.normalize()
    expected_symbols = sorted(contract["universe_freeze"]["broad_pool_expected_symbols"])
    if sorted(shares["ts_code"].astype(str).unique()) != expected_symbols:
        raise ValueError("官方份额ETF池与预冻结合同不一致")
    if len(shares) != int(contracts["official_weekly_shares"]["required_rows"]):
        raise ValueError("官方周度份额固定行数不匹配")
    if shares["date"].nunique() != int(
        contracts["official_weekly_shares"]["required_snapshot_count"]
    ):
        raise ValueError("官方周度份额固定快照数不匹配")
    if str(shares["date"].min().date()) != contracts["official_weekly_shares"][
        "required_first_date"
    ]:
        raise ValueError("官方周度份额首日不匹配")
    if str(shares["date"].max().date()) != contracts["official_weekly_shares"][
        "required_last_date"
    ]:
        raise ValueError("官方周度份额末日不匹配")
    if share_report["status"] != "PASS_COMPLETE_SSE_WEEKLY_SHARE_COLLECTION_NO_FILL":
        raise ValueError("官方周度份额采集报告状态无效")
    if share_report["output"]["sha256"] != hashes[
        contracts["official_weekly_shares"]["file"]
    ]:
        raise ValueError("官方周度份额与采集报告未绑定")
    if share_report["collection"]["fill_used"]:
        raise ValueError("官方周度份额不允许填补")

    market["date"] = pd.to_datetime(market["date"], errors="raise").dt.normalize()
    benchmark["date"] = pd.to_datetime(
        benchmark["date"], errors="raise"
    ).dt.normalize()
    if set(market["symbol"].dropna().astype(str).unique()) != {"510300.SH"}:
        raise ValueError("510300执行资产代码不匹配")
    if set(benchmark["symbol"].dropna().astype(str).unique()) != {"H00300"}:
        raise ValueError("H00300基准代码不匹配")
    market = market.rename(columns={"date": "trade_date"}).sort_values(
        "trade_date", kind="mergesort"
    )
    market.drop_duplicates("trade_date", keep="last", inplace=True)
    market["bar_end"] = market["trade_date"] + pd.Timedelta(hours=15)
    benchmark.sort_values("date", kind="mergesort", inplace=True)
    benchmark.drop_duplicates("date", keep="last", inplace=True)
    window_start = pd.Timestamp("2020-12-01")
    window_end = max(pd.Timestamp(item["end_date"]) for item in contract["evaluation"]["periods"])
    market = market.loc[
        market["trade_date"].between(window_start, window_end)
    ].reset_index(drop=True)
    benchmark = benchmark.loc[
        benchmark["date"].between(window_start, window_end)
    ].reset_index(drop=True)
    missing_benchmark_dates = sorted(set(market["trade_date"]) - set(benchmark["date"]))
    if missing_benchmark_dates:
        raise ValueError(f"H00300缺少510300执行日：{missing_benchmark_dates[:5]}")
    if (market[["open", "close"]] <= 0.0).any().any():
        raise ValueError("510300执行价格非法")
    if (benchmark["close"] <= 0.0).any():
        raise ValueError("H00300基准价格非法")
    for column in ["record_date", "ex_date", "payment_date"]:
        dividends[column] = pd.to_datetime(
            dividends[column], errors="raise"
        ).dt.normalize()
    dividends["cash_dividend_per_share"] = pd.to_numeric(
        dividends["cash_dividend_per_share"], errors="raise"
    )
    if len(dividends) != int(contracts["cash_distributions"]["required_event_count"]):
        raise ValueError("510300分红事件数不匹配")

    tushare_overlap = shares.loc[shares["ts_code"].eq("510300.SH"), ["date", "fund_shares"]]
    comparison_path = ROOT / config["optional_audits"]["510300_tushare_share_overlap"]
    if comparison_path.exists():
        comparison = pd.read_parquet(comparison_path)[["date", "fund_shares"]]
        comparison["date"] = pd.to_datetime(
            comparison["date"], errors="raise"
        ).dt.normalize()
        overlap = tushare_overlap.merge(
            comparison,
            on="date",
            suffixes=("_sse", "_tushare"),
            validate="one_to_one",
        )
        overlap_difference = (
            overlap["fund_shares_sse"] - overlap["fund_shares_tushare"]
        ).abs()
        overlap_audit = {
            "rows": int(len(overlap)),
            "exact_rows": int(overlap_difference.eq(0.0).sum()),
            "maximum_absolute_share_difference": float(overlap_difference.max()),
        }
    else:
        overlap_audit = {"rows": 0, "status": "OPTIONAL_INPUT_ABSENT"}
    audit = {
        "hashes": hashes,
        "official_share_rows": int(len(shares)),
        "official_snapshot_count": int(shares["date"].nunique()),
        "official_first_date": shares["date"].min().date().isoformat(),
        "official_last_date": shares["date"].max().date().isoformat(),
        "official_missing_snapshot_count": int(
            share_report["collection"]["missing_snapshot_count"]
        ),
        "official_missing_symbol_date_count": int(
            share_report["collection"]["missing_symbol_date_count"]
        ),
        "official_fill_used": bool(share_report["collection"]["fill_used"]),
        "historical_publication_timestamp_verified": False,
        "conservative_execution_time": "T_PLUS_2_OPEN",
        "510300_sse_tushare_overlap_audit": overlap_audit,
        "market_rows": int(len(market)),
        "benchmark_rows": int(len(benchmark)),
        "dividend_event_count": int(len(dividends)),
    }
    return {
        "shares": shares,
        "share_report": share_report,
        "prices": prices,
        "market": market,
        "benchmark": benchmark,
        "dividends": dividends,
    }, audit


def cost_model(
    config: dict[str, Any], contract: dict[str, Any], scenario: str
) -> CostModel:
    """按主协议构造基础或压力成本。"""

    costs = config["costs"]
    slippage = (
        costs["base_slippage_bps_per_leg"]
        if scenario == "BASE"
        else costs["stress_slippage_bps_per_leg"]
    )
    return CostModel(
        scenario=scenario,
        commission_rate=float(costs["commission_rate_per_leg"]),
        minimum_commission=float(costs["minimum_commission_cny_per_leg"]),
        slippage_bps=float(slippage),
        cash_annual_rate=float(costs["cash_annual_rate"]),
        trading_days_per_year=int(contract["objective"]["annualization_trading_days"]),
        lot_size=int(costs["lot_size_shares"]),
    )


def evaluate(
    config: dict[str, Any],
    contract: dict[str, Any],
    market: pd.DataFrame,
    benchmark: pd.DataFrame,
    dividends: pd.DataFrame,
    daily_targets: pd.DataFrame,
) -> pd.DataFrame:
    """运行五候选、两成本、三时期和五个起点。"""

    benchmark_returns = build_benchmark_returns(benchmark)
    market_dates = [pd.Timestamp(value) for value in market["trade_date"].tolist()]
    rows: list[dict[str, Any]] = []
    for candidate in contract["candidates"]:
        candidate_id = candidate["candidate_id"]
        states = daily_targets[candidate_id].to_numpy(dtype=np.int8)
        for scenario in contract["evaluation"]["cost_scenarios"]:
            costs = cost_model(config, contract, scenario)
            for period in contract["evaluation"]["periods"]:
                for offset in contract["evaluation"][
                    "start_date_perturbations_trading_days"
                ]:
                    start_date = trading_date_with_offset(
                        market_dates, pd.Timestamp(period["start_date"]), int(offset)
                    )
                    end_date = pd.Timestamp(period["end_date"])
                    ledger, trades, diagnostics = simulate_candidate(
                        market,
                        dividends,
                        states,
                        start_date=start_date,
                        end_date=end_date,
                        costs=costs,
                        initial_capital=float(config["initial_capital_cny"]),
                    )
                    summary = summarize_period(
                        ledger,
                        trades,
                        benchmark_returns,
                        period_id=period["id"],
                        start_date=start_date,
                        end_date=end_date,
                        objective=contract["objective"],
                    )
                    rows.append(
                        {
                            "candidate_id": candidate_id,
                            "scenario": scenario,
                            "start_offset": int(offset),
                            **summary,
                            "t_plus_one_violations": int(
                                diagnostics["t_plus_one_violations"]
                            ),
                        }
                    )
    result = pd.DataFrame(rows)
    expected_rows = (
        len(EXPECTED_CANDIDATES)
        * len(contract["evaluation"]["cost_scenarios"])
        * len(contract["evaluation"]["periods"])
        * len(contract["evaluation"]["start_date_perturbations_trading_days"])
    )
    if len(result) != expected_rows:
        raise AssertionError(f"绩效行数应为{expected_rows}，实际为{len(result)}")
    return result


def summarize_candidates(
    contract: dict[str, Any],
    results: pd.DataFrame,
    daily_targets: pd.DataFrame,
) -> list[dict[str, Any]]:
    """按所有压力时期与起点的最差值汇总候选。"""

    combined_id = contract["evaluation"]["periods"][-1]["id"]
    combined_start = pd.Timestamp(contract["evaluation"]["periods"][-1]["start_date"])
    combined_end = pd.Timestamp(contract["evaluation"]["periods"][-1]["end_date"])
    in_combined = daily_targets["trade_date"].between(combined_start, combined_end)
    summaries: list[dict[str, Any]] = []
    for candidate in contract["candidates"]:
        candidate_id = candidate["candidate_id"]
        rows = results.loc[results["candidate_id"].eq(candidate_id)]
        stress = rows.loc[rows["scenario"].eq("STRESS")]
        stress_combined = stress.loc[
            stress["period_id"].eq(combined_id) & stress["start_offset"].eq(0)
        ].iloc[0]
        base_combined = rows.loc[
            rows["scenario"].eq("BASE")
            & rows["period_id"].eq(combined_id)
            & rows["start_offset"].eq(0)
        ].iloc[0]
        hard_pass = bool(
            (stress["annualized_excess"] >= 0.20).all()
            & (stress["rolling_242d_excess_median"] >= 0.20).all()
        )
        positive_shadow = bool(
            (stress["annualized_excess"] > 0.0).all()
            & (stress["rolling_242d_excess_median"] > 0.0).all()
        )
        state_series = daily_targets.loc[in_combined, candidate_id]
        summaries.append(
            {
                "candidate_id": candidate_id,
                "rule": candidate["rule"],
                "hard_pass": hard_pass,
                "positive_shadow": positive_shadow,
                "stress_min_annualized_excess": float(
                    stress["annualized_excess"].min()
                ),
                "stress_min_rolling_242d_excess_median": float(
                    stress["rolling_242d_excess_median"].min()
                ),
                "stress_combined_annualized_excess_offset0": float(
                    stress_combined["annualized_excess"]
                ),
                "stress_combined_rolling_median_offset0": float(
                    stress_combined["rolling_242d_excess_median"]
                ),
                "base_combined_annualized_excess_offset0": float(
                    base_combined["annualized_excess"]
                ),
                "base_combined_rolling_median_offset0": float(
                    base_combined["rolling_242d_excess_median"]
                ),
                "cash_state_share_combined_window": float(state_series.eq(0).mean()),
                "state_change_count_combined_window": int(
                    max(state_series.diff().ne(0).sum() - 1, 0)
                ),
                "maximum_t_plus_one_violations": int(
                    rows["t_plus_one_violations"].max()
                ),
            }
        )
    return sorted(
        summaries,
        key=lambda item: (
            item["hard_pass"],
            item["stress_min_annualized_excess"],
            item["stress_min_rolling_242d_excess_median"],
        ),
        reverse=True,
    )


def percent(value: float | None) -> str:
    """格式化百分比。"""

    return "NA" if value is None or not np.isfinite(value) else f"{value:.2%}"


def render_markdown(report: dict[str, Any]) -> str:
    """生成简洁的人读研究报告。"""

    best = report["best_candidate"]
    return f"""# 510300跨ETF被迫资金流二元筛选V1

## 结论

- 状态：`{report['status']}`
- 观察池：10只2020年末前固定的上交所大型A股ETF；交易标的仍只允许510300或现金。
- 官方份额：{report['input_audit']['official_snapshot_count']}个周度快照，缺失快照{report['input_audit']['official_missing_snapshot_count']}，未填补。
- 保守时钟：周度份额日期T，直到T+1收盘才视为可见，最早T+2开盘切换。
- 证据级别：`HISTORICALLY_CONTAMINATED_DISCOVERY_ONLY`。

## 固定家族最佳项

- 规则：`{best['candidate_id']}`
- 压力最差年化净超额：{percent(best['stress_min_annualized_excess'])}
- 压力最差滚动242日净超额中位数：{percent(best['stress_min_rolling_242d_excess_median'])}
- 压力综合期年化净超额：{percent(best['stress_combined_annualized_excess_offset0'])}
- 压力综合期滚动242日净超额中位数：{percent(best['stress_combined_rolling_median_offset0'])}
- 综合期空仓占比：{percent(best['cash_state_share_combined_window'])}
- 双20全门：{'通过' if best['hard_pass'] else '未通过'}

## 决策

{report['decision']}

指数换样只产生篮子内部相对调仓，未被包装成整个沪深300的方向信号。份额变化代表已实现申赎代理，也不等于授权参与人必然在公开市场同步买卖全部篮子。
"""


def run(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """执行冻结后的唯一一次固定家族筛选。"""

    config, contract = load_config(config_path)
    manifest = verify_freeze_manifest(config)
    inputs, input_audit = load_inputs(config, contract)
    features = build_weekly_features(
        inputs["shares"],
        inputs["prices"],
        broad_symbols=contract["universe_freeze"]["broad_pool_expected_symbols"],
        hs300_symbols=contract["universe_freeze"]["hs300_pool_symbols"],
        rolling_window=int(contract["features"]["rolling_percentile_window_weeks"]),
        minimum_observations=int(
            contract["features"]["rolling_percentile_minimum_weeks"]
        ),
    )
    weekly_states = build_candidate_weekly_states(features)
    daily_targets = align_weekly_states_to_daily_targets(
        inputs["market"], weekly_states
    )
    results = evaluate(
        config,
        contract,
        inputs["market"],
        inputs["benchmark"],
        inputs["dividends"],
        daily_targets,
    )
    summaries = summarize_candidates(contract, results, daily_targets)
    passing = [item["candidate_id"] for item in summaries if item["hard_pass"]]
    positive_shadow = [
        item["candidate_id"] for item in summaries if item["positive_shadow"]
    ]
    status = (
        "PASSED_FIXED_CROSS_ETF_FORCED_FLOW_BINARY_FAMILY_REQUIRES_PROSPECTIVE_VALIDATION"
        if passing
        else "REJECTED_FIXED_CROSS_ETF_FORCED_FLOW_BINARY_FAMILY_NO_RESCUE"
    )
    decision = (
        "至少一条固定规则通过所有压力成本时期、起点扰动和双20门；只能冻结进入前瞻验证，不能直接映射持仓。"
        if passing
        else "五条预先固定规则均未通过双20全门；禁止改阈值、改周数、改ETF池、改执行延迟或组合条件营救。"
    )
    first_rank_dates = {}
    for column in [
        "broad_flow_rank_1w",
        "broad_flow_rank_4w",
        "hs300_flow_rank_1w",
    ]:
        finite = features.loc[features[column].notna(), "date"]
        first_rank_dates[column] = (
            None if finite.empty else finite.iloc[0].date().isoformat()
        )

    weekly_features_path = ROOT / config["artifacts"]["weekly_features"]
    daily_states_path = ROOT / config["artifacts"]["daily_states"]
    metrics_path = ROOT / config["artifacts"]["metrics"]
    report_json_path = ROOT / config["artifacts"]["report_json"]
    report_markdown_path = ROOT / config["artifacts"]["report_markdown"]
    daily_output = daily_targets.copy()
    daily_output["execution_effective_date"] = daily_output["trade_date"].shift(-1)
    atomic_parquet(features, weekly_features_path)
    atomic_parquet(daily_output, daily_states_path)
    atomic_parquet(results, metrics_path)

    report = {
        "status": status,
        "study_id": STUDY_ID,
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_sha256": sha256_file(CONFIG_PATH),
        "freeze_manifest_sha256": sha256_file(
            ROOT / config["artifacts"]["freeze_manifest"]
        ),
        "candidate_contract_sha256": CANDIDATE_CONTRACT_SHA256,
        "candidate_prefreeze_receipt_sha256": config["protocol"][
            "candidate_prefreeze_receipt_sha256"
        ],
        "scope": {
            "signal_funds_not_limited_to_510300": True,
            "execution_asset": "510300.SH",
            "allowed_states": [0, 1],
            "benchmark": "H00300 total return",
            "signal_clock": "SSE_WEEKLY_SHARE_DATE_T_TREATED_AVAILABLE_T_PLUS_1_CLOSE",
            "execution_time": "T_PLUS_2_OPEN",
            "evidence_label": "HISTORICALLY_CONTAMINATED_DISCOVERY_ONLY",
        },
        "input_audit": input_audit,
        "feature_audit": {
            "weekly_rows": int(len(features)),
            "first_date": features["date"].min().date().isoformat(),
            "last_date": features["date"].max().date().isoformat(),
            "first_rank_dates": first_rank_dates,
            "future_510300_return_columns_read": 0,
            "historical_publication_timestamp_verified": False,
            "conservative_execution_time": "T_PLUS_2_OPEN",
        },
        "candidate_count": len(EXPECTED_CANDIDATES),
        "result_rows": int(len(results)),
        "passing_candidates": passing,
        "positive_shadow_candidates": positive_shadow,
        "best_candidate": summaries[0],
        "candidate_summaries": summaries,
        "maximum_t_plus_one_violations": int(
            results["t_plus_one_violations"].max()
        ),
        "decision": decision,
        "artifacts": {
            weekly_features_path.relative_to(ROOT).as_posix(): sha256_file(
                weekly_features_path
            ),
            daily_states_path.relative_to(ROOT).as_posix(): sha256_file(
                daily_states_path
            ),
            metrics_path.relative_to(ROOT).as_posix(): sha256_file(metrics_path),
        },
        "manifest_frozen_at": manifest["frozen_at"],
        "boundaries": contract["boundaries"],
    }
    atomic_json(report, report_json_path)
    report_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_markdown = report_markdown_path.with_suffix(
        report_markdown_path.suffix + ".tmp"
    )
    temporary_markdown.write_text(render_markdown(report), encoding="utf-8")
    temporary_markdown.replace(report_markdown_path)
    return report


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    report = run(args.config)
    best = report["best_candidate"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "best_candidate": best["candidate_id"],
                "stress_min_annualized_excess": best[
                    "stress_min_annualized_excess"
                ],
                "stress_min_rolling_242d_excess_median": best[
                    "stress_min_rolling_242d_excess_median"
                ],
                "passing_candidates": report["passing_candidates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
