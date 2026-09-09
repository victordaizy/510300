"""筛选中金所IF真实逐合约期限结构映射的510300/现金固定二元规则。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from research.etf_share_premium_level_binary_screen_v1 import (
    atomic_json,
    atomic_parquet,
    percent,
    required_columns,
    rolling_last_percentile,
    sha256_file,
)
from research.intraday_binary_livermore_screen_v1 import (
    CostModel,
    build_benchmark_returns,
    simulate_candidate,
    summarize_period,
    trading_date_with_offset,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "510300_if_true_term_structure_binary_screen_v1_0_1.yaml"
CANDIDATE_CONTRACT_PATH = (
    ROOT
    / "config"
    / "510300_if_true_term_structure_binary_screen_v1_0_1_candidates.yaml"
)
CANDIDATE_CONTRACT_SHA256 = (
    "0d406748722608296935ec7b398e9e4a1fab153b6d411dd83fb70bbe654fa4c1"
)
STUDY_ID = "510300_IF_TRUE_TERM_STRUCTURE_BINARY_SCREEN_V1_0_1"
EXPECTED_CANDIDATES = [
    "IF_NEAR_BASIS_BOTTOM10_CASH",
    "IF_NEAR_BASIS_TOP10_CASH",
    "IF_NEAR_NEXT_CURVE_BOTTOM10_CASH",
    "IF_NEAR_NEXT_CURVE_TOP10_CASH",
    "IF_NEAR_FAR_CURVE_BOTTOM10_CASH",
    "IF_NEAR_FAR_CURVE_TOP10_CASH",
    "IF_OI_WEIGHTED_BASIS_BOTTOM10_CASH",
    "IF_OI_WEIGHTED_BASIS_TOP10_CASH",
]
FEATURE_COLUMNS = [
    "annualized_near_basis",
    "annualized_near_next_curve",
    "annualized_near_far_curve",
    "annualized_oi_weighted_basis",
]
RANK_COLUMNS = [
    "rank_annualized_near_basis",
    "rank_annualized_near_next_curve",
    "rank_annualized_near_far_curve",
    "rank_annualized_oi_weighted_basis",
]


def load_candidate_contract() -> dict[str, Any]:
    """读取并核对正式全历史获取前固定的V1.0.1候选合同。"""

    if sha256_file(CANDIDATE_CONTRACT_PATH) != CANDIDATE_CONTRACT_SHA256:
        raise ValueError("正式全历史获取前固定的V1.0.1候选合同发生漂移")
    contract = yaml.safe_load(CANDIDATE_CONTRACT_PATH.read_text(encoding="utf-8"))
    protocol = contract["protocol"]
    if protocol["study_id"] != STUDY_ID:
        raise ValueError("候选合同研究编号不匹配")
    if (
        protocol["state"]
        != "CANDIDATE_FAMILY_UNCHANGED_AFTER_SOURCE_SCHEMA_PROBE_BEFORE_FULL_ACQUISITION"
    ):
        raise ValueError("候选合同状态无效")
    if not bool(protocol["no_parameter_rescue"]):
        raise ValueError("候选合同必须禁止参数营救")
    candidate_ids = [item["candidate_id"] for item in contract["candidates"]]
    if candidate_ids != EXPECTED_CANDIDATES:
        raise ValueError("固定八规则或顺序发生漂移")
    if list(contract["scope"]["allowed_target_states"]) != [0, 1]:
        raise ValueError("目标状态必须严格为0和1")
    if float(contract["objective"]["minimum_annualized_excess"]) != 0.20:
        raise ValueError("年化净超额硬门必须为20%")
    if float(contract["objective"]["minimum_rolling_excess_median"]) != 0.20:
        raise ValueError("滚动净超额硬门必须为20%")
    return contract


def load_config(path: Path = CONFIG_PATH) -> tuple[dict[str, Any], dict[str, Any]]:
    """读取并核对完整冻结前协议。"""

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
    if list(contract["scope"]["allowed_holdings"]) != ["510300.SH", "CASH_CNY"]:
        raise ValueError("持仓范围必须严格为510300和现金")
    return config, contract


def verify_freeze_manifest(config: dict[str, Any]) -> dict[str, Any]:
    """运行前核对冻结协议、源码、测试、依赖与输入。"""

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


def build_term_features(
    contract_daily: pd.DataFrame,
    expiry_table: pd.DataFrame,
    spot: pd.DataFrame,
    *,
    minimum_dte: int,
    minimum_contracts: int,
    rolling_window: int,
    minimum_observations: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """从逐合约官方收盘构造真实近月、次月、远月曲线及历史分位。"""

    required_columns(
        contract_daily,
        ["symbol", "date", "close", "open_interest", "source_url"],
        "中金所IF逐合约日线",
    )
    required_columns(
        expiry_table,
        ["symbol", "expiry_date", "expiry_source", "active_at_ceiling"],
        "IF合约到期表",
    )
    required_columns(spot, ["date", "close", "symbol"], "沪深300现货指数")
    daily = contract_daily.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="raise").dt.normalize()
    daily["close"] = pd.to_numeric(daily["close"], errors="raise")
    daily["open_interest"] = pd.to_numeric(
        daily["open_interest"], errors="raise"
    )
    daily["symbol"] = daily["symbol"].astype(str)
    daily.sort_values(["date", "symbol"], kind="mergesort", inplace=True)
    if daily.duplicated(["date", "symbol"]).any():
        raise ValueError("IF逐合约输入日期代码重复")
    if not daily["symbol"].str.fullmatch(r"IF\d{4}").all():
        raise ValueError("IF逐合约输入混入非IF代码")

    expiry = expiry_table.copy()
    expiry["symbol"] = expiry["symbol"].astype(str)
    expiry["expiry_date"] = pd.to_datetime(
        expiry["expiry_date"], errors="raise"
    ).dt.normalize()
    if expiry["symbol"].duplicated().any():
        raise ValueError("IF合约到期表代码重复")
    if set(daily["symbol"].unique()) != set(expiry["symbol"].unique()):
        raise ValueError("逐合约日线与到期表代码集合不一致")
    active = expiry.loc[expiry["active_at_ceiling"].astype(bool)]
    if not active["expiry_source"].eq(
        "CFFEX_TRADING_PARAMETER_END_TRADING_DAY_AT_CEILING"
    ).all():
        raise ValueError("样本末日存续合约未使用官方END_TRADING_DAY")

    spot_frame = spot[["date", "close", "symbol"]].copy()
    spot_frame["date"] = pd.to_datetime(
        spot_frame["date"], errors="raise"
    ).dt.normalize()
    spot_frame["close"] = pd.to_numeric(spot_frame["close"], errors="raise")
    if set(spot_frame["symbol"].dropna().astype(str).unique()) != {"000300.SH"}:
        raise ValueError("现货指数代码不是000300.SH")
    spot_frame.sort_values("date", kind="mergesort", inplace=True)
    if spot_frame["date"].duplicated().any():
        raise ValueError("现货指数日期重复")
    spot_frame.rename(columns={"close": "spot_close"}, inplace=True)

    joined = daily.merge(
        expiry[["symbol", "expiry_date", "expiry_source", "active_at_ceiling"]],
        on="symbol",
        how="left",
        validate="many_to_one",
    ).merge(
        spot_frame[["date", "spot_close"]],
        on="date",
        how="inner",
        validate="many_to_one",
    )
    joined["calendar_days_to_expiry"] = (
        joined["expiry_date"] - joined["date"]
    ).dt.days
    eligible = joined.loc[
        joined["close"].gt(0.0)
        & joined["spot_close"].gt(0.0)
        & joined["open_interest"].ge(0.0)
        & joined["calendar_days_to_expiry"].ge(minimum_dte)
    ].copy()
    eligible["annualized_contract_basis"] = (
        np.log(eligible["close"] / eligible["spot_close"])
        * 365.0
        / eligible["calendar_days_to_expiry"]
    )

    rows: list[dict[str, Any]] = []
    insufficient_dates: list[str] = []
    for date, group in eligible.groupby("date", sort=True):
        group = group.sort_values(["expiry_date", "symbol"], kind="mergesort")
        if len(group) < minimum_contracts:
            insufficient_dates.append(pd.Timestamp(date).date().isoformat())
            continue
        if group["expiry_date"].duplicated().any():
            raise ValueError(f"{pd.Timestamp(date).date()}出现重复IF到期日")
        near = group.iloc[0]
        next_contract = group.iloc[1]
        far = group.iloc[-1]
        near_next_span = int(
            (pd.Timestamp(next_contract["expiry_date"]) - pd.Timestamp(near["expiry_date"])).days
        )
        near_far_span = int(
            (pd.Timestamp(far["expiry_date"]) - pd.Timestamp(near["expiry_date"])).days
        )
        if near_next_span <= 0 or near_far_span <= 0:
            raise ValueError(f"{pd.Timestamp(date).date()}期限跨度非正")
        total_open_interest = float(group["open_interest"].sum())
        oi_weighted = (
            np.nan
            if total_open_interest <= 0.0
            else float(
                np.average(
                    group["annualized_contract_basis"],
                    weights=group["open_interest"],
                )
            )
        )
        rows.append(
            {
                "date": pd.Timestamp(date),
                "spot_close": float(near["spot_close"]),
                "eligible_contract_count": int(len(group)),
                "near_symbol": str(near["symbol"]),
                "next_symbol": str(next_contract["symbol"]),
                "far_symbol": str(far["symbol"]),
                "near_expiry_date": pd.Timestamp(near["expiry_date"]),
                "next_expiry_date": pd.Timestamp(next_contract["expiry_date"]),
                "far_expiry_date": pd.Timestamp(far["expiry_date"]),
                "near_dte": int(near["calendar_days_to_expiry"]),
                "next_dte": int(next_contract["calendar_days_to_expiry"]),
                "far_dte": int(far["calendar_days_to_expiry"]),
                "near_close": float(near["close"]),
                "next_close": float(next_contract["close"]),
                "far_close": float(far["close"]),
                "total_open_interest": total_open_interest,
                "annualized_near_basis": float(near["annualized_contract_basis"]),
                "annualized_near_next_curve": float(
                    np.log(float(next_contract["close"]) / float(near["close"]))
                    * 365.0
                    / near_next_span
                ),
                "annualized_near_far_curve": float(
                    np.log(float(far["close"]) / float(near["close"]))
                    * 365.0
                    / near_far_span
                ),
                "annualized_oi_weighted_basis": oi_weighted,
                "feature_asof": pd.Timestamp(date) + pd.Timedelta(hours=15, minutes=15),
            }
        )
    features = pd.DataFrame(rows).sort_values("date", kind="mergesort")
    features.reset_index(drop=True, inplace=True)
    if features.empty or features["date"].duplicated().any():
        raise ValueError("期限结构日特征为空或日期重复")
    if features[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("期限结构核心变量存在缺失")
    for feature in FEATURE_COLUMNS:
        features[f"rank_{feature}"] = rolling_last_percentile(
            features[feature],
            window=rolling_window,
            minimum_observations=minimum_observations,
        )
    audit = {
        "raw_contract_rows": int(len(daily)),
        "raw_contract_dates": int(daily["date"].nunique()),
        "spot_rows": int(len(spot_frame)),
        "joined_contract_rows": int(len(joined)),
        "eligible_contract_rows": int(len(eligible)),
        "feature_rows": int(len(features)),
        "feature_first_date": features["date"].min().date().isoformat(),
        "feature_last_date": features["date"].max().date().isoformat(),
        "minimum_eligible_contract_count": int(
            features["eligible_contract_count"].min()
        ),
        "maximum_eligible_contract_count": int(
            features["eligible_contract_count"].max()
        ),
        "insufficient_contract_dates_excluded": insufficient_dates,
        "active_contract_metadata_rows": int(len(active)),
        "future_price_or_return_columns_read": 0,
    }
    return eligible.reset_index(drop=True), features, audit


def build_candidate_states(features: pd.DataFrame) -> pd.DataFrame:
    """按冻结上下尾定义生成八列严格0/1状态。"""

    frame = features[
        ["date", *FEATURE_COLUMNS, *RANK_COLUMNS, "near_symbol", "next_symbol", "far_symbol"]
    ].copy()
    pairs = [
        ("IF_NEAR_BASIS_BOTTOM10_CASH", "rank_annualized_near_basis", "BOTTOM"),
        ("IF_NEAR_BASIS_TOP10_CASH", "rank_annualized_near_basis", "TOP"),
        (
            "IF_NEAR_NEXT_CURVE_BOTTOM10_CASH",
            "rank_annualized_near_next_curve",
            "BOTTOM",
        ),
        (
            "IF_NEAR_NEXT_CURVE_TOP10_CASH",
            "rank_annualized_near_next_curve",
            "TOP",
        ),
        (
            "IF_NEAR_FAR_CURVE_BOTTOM10_CASH",
            "rank_annualized_near_far_curve",
            "BOTTOM",
        ),
        (
            "IF_NEAR_FAR_CURVE_TOP10_CASH",
            "rank_annualized_near_far_curve",
            "TOP",
        ),
        (
            "IF_OI_WEIGHTED_BASIS_BOTTOM10_CASH",
            "rank_annualized_oi_weighted_basis",
            "BOTTOM",
        ),
        (
            "IF_OI_WEIGHTED_BASIS_TOP10_CASH",
            "rank_annualized_oi_weighted_basis",
            "TOP",
        ),
    ]
    for candidate_id, rank_column, tail in pairs:
        trigger = (
            frame[rank_column].notna() & frame[rank_column].le(0.10)
            if tail == "BOTTOM"
            else frame[rank_column].notna() & frame[rank_column].ge(0.90)
        )
        frame[candidate_id] = np.where(trigger, 0, 1).astype(np.int8)
        if not set(frame[candidate_id].unique()).issubset({0, 1}):
            raise AssertionError(f"{candidate_id}不是严格二元状态")
    return frame


def align_states_to_market(market: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    """对齐510300交易日；期限结构缺失日明确回到满仓。"""

    aligned = market[["trade_date"]].rename(columns={"trade_date": "date"}).merge(
        states,
        on="date",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    aligned["term_feature_observed"] = aligned["_merge"].eq("both")
    aligned.drop(columns="_merge", inplace=True)
    for candidate_id in EXPECTED_CANDIDATES:
        aligned[candidate_id] = aligned[candidate_id].fillna(1).astype(np.int8)
        if not set(aligned[candidate_id].unique()).issubset({0, 1}):
            raise AssertionError(f"{candidate_id}对齐后不是严格二元状态")
    return aligned


def load_inputs(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """按固定哈希读取官方IF、现货、执行资产、全收益基准和分红。"""

    contracts = config["data_contracts"]
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name, item in contracts.items():
        path = ROOT / item["file"]
        if not path.exists():
            raise FileNotFoundError(f"固定工件不存在：{item['file']}")
        actual = sha256_file(path)
        if actual != item["required_sha256"]:
            raise ValueError(f"固定工件哈希漂移：{item['file']}")
        paths[name] = path
        hashes[item["file"]] = actual

    contract_daily = pd.read_parquet(paths["if_contract_daily"])
    expiry = pd.read_parquet(paths["if_contract_expiry"])
    acquisition_audit = json.loads(
        paths["if_acquisition_audit"].read_text(encoding="utf-8")
    )
    spot = pd.read_parquet(paths["spot_index"])
    market = pd.read_parquet(paths["execution_daily"])
    benchmark = pd.read_parquet(paths["benchmark_total_return"])
    dividends = pd.read_csv(paths["cash_distributions"])
    required_columns(
        contract_daily,
        contracts["if_contract_daily"]["required_columns"],
        "中金所IF逐合约日线",
    )
    required_columns(
        expiry,
        contracts["if_contract_expiry"]["required_columns"],
        "IF合约到期表",
    )
    required_columns(spot, contracts["spot_index"]["required_columns"], "沪深300现货")
    required_columns(
        market, contracts["execution_daily"]["required_columns"], "510300日线"
    )
    required_columns(
        benchmark,
        contracts["benchmark_total_return"]["required_columns"],
        "H00300全收益基准",
    )
    required_columns(
        dividends,
        contracts["cash_distributions"]["required_columns"],
        "510300分红",
    )
    if len(contract_daily) != int(contracts["if_contract_daily"]["required_rows"]):
        raise ValueError("IF逐合约固定行数不匹配")
    if len(expiry) != int(contracts["if_contract_expiry"]["required_rows"]):
        raise ValueError("IF到期表固定行数不匹配")
    if acquisition_audit["status"] != "SUCCESS_CFFEX_IF_CONTRACT_HISTORY_ACQUIRED":
        raise ValueError("IF官方输入获取审计状态无效")
    if bool(acquisition_audit["performance_returns_read"]):
        raise ValueError("IF输入获取阶段读取了绩效收益")
    daily_relative = contracts["if_contract_daily"]["file"]
    expiry_relative = contracts["if_contract_expiry"]["file"]
    if acquisition_audit["outputs"][daily_relative] != hashes[daily_relative]:
        raise ValueError("IF逐合约输入与获取审计记录不一致")
    if acquisition_audit["outputs"][expiry_relative] != hashes[expiry_relative]:
        raise ValueError("IF到期表与获取审计记录不一致")

    for frame, symbol, label in [
        (spot, "000300.SH", "沪深300现货"),
        (market, "510300.SH", "510300执行资产"),
        (benchmark, "H00300", "H00300全收益基准"),
    ]:
        if set(frame["symbol"].dropna().astype(str).unique()) != {symbol}:
            raise ValueError(f"{label}代码不匹配")
    market = market.rename(columns={"date": "trade_date"}).copy()
    market["trade_date"] = pd.to_datetime(
        market["trade_date"], errors="raise"
    ).dt.normalize()
    market["bar_end"] = market["trade_date"] + pd.Timedelta(hours=15)
    market.sort_values("trade_date", kind="mergesort", inplace=True)
    market.drop_duplicates("trade_date", keep="last", inplace=True)
    benchmark["date"] = pd.to_datetime(
        benchmark["date"], errors="raise"
    ).dt.normalize()
    benchmark.sort_values("date", kind="mergesort", inplace=True)
    benchmark.drop_duplicates("date", keep="last", inplace=True)
    spot["date"] = pd.to_datetime(spot["date"], errors="raise").dt.normalize()
    spot.sort_values("date", kind="mergesort", inplace=True)
    spot.drop_duplicates("date", keep="last", inplace=True)
    window_start = pd.Timestamp("2012-05-28")
    window_end = pd.Timestamp("2026-08-12")
    market = market.loc[
        market["trade_date"].between(window_start, window_end, inclusive="both")
    ].reset_index(drop=True)
    benchmark = benchmark.loc[
        benchmark["date"].between(window_start, window_end, inclusive="both")
    ].reset_index(drop=True)
    spot = spot.loc[
        spot["date"].between(window_start, window_end, inclusive="both")
    ].reset_index(drop=True)
    if sorted(set(market["trade_date"]) - set(benchmark["date"])):
        raise ValueError("H00300缺少510300执行交易日")
    if sorted(set(market["trade_date"]) - set(spot["date"])):
        raise ValueError("沪深300现货缺少510300执行交易日")
    if (market[["open", "close"]] <= 0.0).any().any():
        raise ValueError("510300执行价格非法")
    if (benchmark["close"] <= 0.0).any() or (spot["close"] <= 0.0).any():
        raise ValueError("指数价格非法")
    for column in ["record_date", "ex_date", "payment_date"]:
        dividends[column] = pd.to_datetime(
            dividends[column], errors="raise"
        ).dt.normalize()
    dividends["cash_dividend_per_share"] = pd.to_numeric(
        dividends["cash_dividend_per_share"], errors="raise"
    )
    if len(dividends) != int(contracts["cash_distributions"]["required_event_count"]):
        raise ValueError("分红事件数不匹配")
    audit = {
        "hashes": hashes,
        "if_contract_rows": int(len(contract_daily)),
        "if_contract_dates": int(
            pd.to_datetime(contract_daily["date"]).dt.normalize().nunique()
        ),
        "if_expiry_rows": int(len(expiry)),
        "official_acquisition_status": acquisition_audit["status"],
        "spot_rows": int(len(spot)),
        "market_rows": int(len(market)),
        "benchmark_rows": int(len(benchmark)),
        "dividend_event_count": int(len(dividends)),
    }
    return {
        "contract_daily": contract_daily,
        "expiry": expiry,
        "spot": spot,
        "market": market,
        "benchmark": benchmark,
        "dividends": dividends,
    }, audit


def cost_model(config: dict[str, Any], contract: dict[str, Any], scenario: str) -> CostModel:
    """按协议构造基础或压力成本。"""

    costs = config["costs"]
    if scenario == "BASE":
        slippage = costs["base_slippage_bps_per_leg"]
    elif scenario == "STRESS":
        slippage = costs["stress_slippage_bps_per_leg"]
    else:
        raise ValueError(f"未知成本情景：{scenario}")
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
    aligned_states: pd.DataFrame,
) -> pd.DataFrame:
    """运行八候选、两成本、五时期和五个起点。"""

    benchmark_returns = build_benchmark_returns(benchmark)
    market_dates = [pd.Timestamp(value) for value in market["trade_date"].tolist()]
    rows: list[dict[str, Any]] = []
    for candidate in contract["candidates"]:
        candidate_id = candidate["candidate_id"]
        states = aligned_states[candidate_id].to_numpy(dtype=np.int8)
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
    aligned_states: pd.DataFrame,
) -> list[dict[str, Any]]:
    """按所有压力时期与起点的最差值汇总候选。"""

    summaries: list[dict[str, Any]] = []
    combined_id = "COMBINED_2013_2026"
    for candidate in contract["candidates"]:
        candidate_id = candidate["candidate_id"]
        candidate_rows = results.loc[results["candidate_id"].eq(candidate_id)]
        stress = candidate_rows.loc[candidate_rows["scenario"].eq("STRESS")]
        stress_combined = stress.loc[
            stress["period_id"].eq(combined_id) & stress["start_offset"].eq(0)
        ].iloc[0]
        base_combined = candidate_rows.loc[
            candidate_rows["scenario"].eq("BASE")
            & candidate_rows["period_id"].eq(combined_id)
            & candidate_rows["start_offset"].eq(0)
        ].iloc[0]
        hard_pass = bool(
            (stress["annualized_excess"] >= 0.20).all()
            & (stress["rolling_242d_excess_median"] >= 0.20).all()
        )
        positive_shadow = bool(
            (stress["annualized_excess"] > 0.0).all()
            & (stress["rolling_242d_excess_median"] > 0.0).all()
        )
        state_series = aligned_states[candidate_id]
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
                "cash_state_share_full_input_window": float(state_series.eq(0).mean()),
                "state_change_count_full_input_window": int(
                    state_series.diff().ne(0).sum() - 1
                ),
                "maximum_t_plus_one_violations": int(
                    candidate_rows["t_plus_one_violations"].max()
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


def render_markdown(report: dict[str, Any]) -> str:
    """生成简洁的人读报告。"""

    best = report["best_candidate"]
    return f"""# 510300 IF真实期限结构二元筛选 V1.0.1

状态：`{report['status']}`

信息只来自中金所IF逐合约收盘和000300现货收盘，T+1开盘执行510300；目标状态严格为满仓或空仓。

## 最佳固定规则

- 规则：`{best['candidate_id']}`
- 压力最差年化净超额：{percent(best['stress_min_annualized_excess'])}
- 压力最差滚动242日超额中位数：{percent(best['stress_min_rolling_242d_excess_median'])}
- 压力综合期年化净超额：{percent(best['stress_combined_annualized_excess_offset0'])}
- 压力综合期滚动242日超额中位数：{percent(best['stress_combined_rolling_median_offset0'])}
- 双20全门：{'通过' if best['hard_pass'] else '未通过'}

决策：{report['decision']}
"""


def run(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """执行冻结后的唯一一次固定家族筛选。"""

    if config_path.resolve() != CONFIG_PATH.resolve():
        raise ValueError("只允许固定协议入口")
    config, contract = load_config(config_path)
    manifest = verify_freeze_manifest(config)
    report_path = ROOT / config["artifacts"]["report_json"]
    if report_path.exists():
        raise FileExistsError("固定报告已存在，禁止重复运行")
    inputs, input_audit = load_inputs(config)
    selection = contract["contract_selection"]
    feature_contract = contract["features"]
    eligible, features, feature_audit = build_term_features(
        inputs["contract_daily"],
        inputs["expiry"],
        inputs["spot"],
        minimum_dte=int(selection["minimum_calendar_days_to_expiry_inclusive"]),
        minimum_contracts=int(selection["minimum_eligible_contracts_per_day"]),
        rolling_window=int(feature_contract["rolling_percentile_window_trading_rows"]),
        minimum_observations=int(
            feature_contract["rolling_percentile_minimum_rows"]
        ),
    )
    states = build_candidate_states(features)
    aligned_states = align_states_to_market(inputs["market"], states)
    results = evaluate(
        config,
        contract,
        inputs["market"],
        inputs["benchmark"],
        inputs["dividends"],
        aligned_states,
    )
    summaries = summarize_candidates(contract, results, aligned_states)
    passing = [item["candidate_id"] for item in summaries if item["hard_pass"]]
    positive = [
        item["candidate_id"] for item in summaries if item["positive_shadow"]
    ]
    status = (
        "DISCOVERY_CANDIDATE_FOUND_REQUIRES_246D_PROSPECTIVE_FREEZE"
        if passing
        else "REJECTED_FIXED_IF_TRUE_TERM_STRUCTURE_BINARY_FAMILY_NO_RESCUE"
    )
    decision = (
        "固定家族存在双20通过者；只允许冻结新的246交易日前瞻验证，不得直接交易。"
        if passing
        else "八个预先固定的真实期限结构上下尾规则均未过双20门槛；禁止改方向、分位、到期筛选或期限选择营救。"
    )
    feature_path = ROOT / config["artifacts"]["daily_features"]
    state_path = ROOT / config["artifacts"]["daily_states"]
    metrics_path = ROOT / config["artifacts"]["metrics"]
    atomic_parquet(features, feature_path)
    atomic_parquet(aligned_states, state_path)
    atomic_parquet(results, metrics_path)
    first_rank_dates = {
        column: features.loc[features[column].notna(), "date"].min().date().isoformat()
        for column in RANK_COLUMNS
    }
    report = {
        "status": status,
        "study_id": STUDY_ID,
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_sha256": manifest["protocol_sha256"],
        "freeze_manifest_sha256": sha256_file(
            ROOT / config["artifacts"]["freeze_manifest"]
        ),
        "candidate_contract_sha256": CANDIDATE_CONTRACT_SHA256,
        "scope": {
            "execution_asset": "510300.SH",
            "information_asset": "CFFEX IF all listed contracts",
            "allowed_states": [0, 1],
            "benchmark": "H00300 total return",
            "signal_time": "T_CFFEX_CLOSE",
            "execution_time": "T_PLUS_1_510300_OPEN",
            "evidence_label": contract["protocol"]["evidence_label"],
        },
        "input_audit": input_audit,
        "feature_audit": {
            **feature_audit,
            "first_rank_dates": first_rank_dates,
            "market_days_without_term_feature": int(
                (~aligned_states["term_feature_observed"]).sum()
            ),
            "eligible_contract_frame_rows": int(len(eligible)),
        },
        "candidate_count": len(EXPECTED_CANDIDATES),
        "result_rows": int(len(results)),
        "passing_candidates": passing,
        "positive_shadow_candidates": positive,
        "best_candidate": summaries[0],
        "candidate_summaries": summaries,
        "maximum_t_plus_one_violations": int(
            results["t_plus_one_violations"].max()
        ),
        "artifacts": {
            feature_path.relative_to(ROOT).as_posix(): sha256_file(feature_path),
            state_path.relative_to(ROOT).as_posix(): sha256_file(state_path),
            metrics_path.relative_to(ROOT).as_posix(): sha256_file(metrics_path),
        },
        "decision": decision,
        "boundaries": config["boundaries"],
    }
    atomic_json(report, report_path)
    markdown_path = ROOT / config["artifacts"]["report_markdown"]
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "passing_candidates": passing,
                "positive_shadow_candidates": positive,
                "best_candidate": summaries[0],
                "maximum_t_plus_one_violations": report[
                    "maximum_t_plus_one_violations"
                ],
                "report_sha256": sha256_file(report_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return report


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
