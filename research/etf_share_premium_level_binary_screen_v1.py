"""筛选510300基金份额变化与收盘折溢价水平映射的固定二元规则。"""

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
    ROOT / "config" / "510300_etf_share_premium_level_binary_screen_v1.yaml"
)
CANDIDATE_CONTRACT_PATH = (
    ROOT / "config" / "510300_etf_share_premium_level_binary_screen_v1_candidates.yaml"
)
CANDIDATE_CONTRACT_SHA256 = (
    "33b62f3917287cbf9dbc194bc7027cac3b5c29c9e409955ed557be8d0f91c658"
)
STUDY_ID = "510300_ETF_SHARE_PREMIUM_LEVEL_BINARY_SCREEN_V1"
EXPECTED_CANDIDATES = [
    "ETF_SHARE1D_BOTTOM10_CASH",
    "ETF_SHARE5D_BOTTOM10_CASH",
    "ETF_SHARE20D_BOTTOM10_CASH",
    "ETF_PREMIUM_BOTTOM10_CASH",
    "ETF_PREMIUM_TOP10_CASH",
    "ETF_ABS_PREMIUM_TOP10_CASH",
    "ETF_SHARE5_BOTTOM20_AND_PREMIUM_BOTTOM20_CASH",
    "ETF_SHARE5_TOP20_AND_PREMIUM_TOP20_CASH",
]


def sha256_file(path: Path) -> str:
    """流式计算文件SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    """将NumPy和Pandas标量安全转换为JSON值。"""

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
    """原子写入JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    """原子写入Parquet。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def load_candidate_contract() -> dict[str, Any]:
    """读取并核对在全历史合并前固定的八规则合同。"""

    if sha256_file(CANDIDATE_CONTRACT_PATH) != CANDIDATE_CONTRACT_SHA256:
        raise ValueError("全历史合并前固定的候选合同发生漂移")
    contract = yaml.safe_load(CANDIDATE_CONTRACT_PATH.read_text(encoding="utf-8"))
    protocol = contract["protocol"]
    if protocol["study_id"] != STUDY_ID:
        raise ValueError("候选合同研究编号不匹配")
    if protocol["state"] != "CANDIDATE_FAMILY_FIXED_BEFORE_FULL_HISTORY_MERGE":
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


def rolling_last_percentile(
    values: pd.Series,
    *,
    window: int,
    minimum_observations: int,
) -> pd.Series:
    """计算当前值在只含当前和过去有限观测的滚动经验分位。"""

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


def build_features(
    raw: pd.DataFrame,
    *,
    rolling_window: int,
    minimum_observations: int,
) -> pd.DataFrame:
    """仅用当日及此前数据构造份额变化和折溢价水平分位。"""

    required = {
        "date",
        "fund_shares",
        "close_premium_to_nav",
        "feature_asof",
        "execution_earliest",
    }
    if missing := required - set(raw.columns):
        raise ValueError(f"ETF份额与折溢价输入缺少字段：{sorted(missing)}")
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["feature_asof"] = pd.to_datetime(frame["feature_asof"], errors="raise")
    frame["fund_shares"] = pd.to_numeric(frame["fund_shares"], errors="raise")
    frame["close_premium_to_nav"] = pd.to_numeric(
        frame["close_premium_to_nav"], errors="raise"
    )
    frame.sort_values("date", kind="mergesort", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    if frame["date"].duplicated().any():
        raise ValueError("ETF联合特征日期重复")
    if frame[["fund_shares", "close_premium_to_nav"]].isna().any().any():
        raise ValueError("ETF联合特征存在缺失")
    if frame["fund_shares"].le(0.0).any():
        raise ValueError("基金份额必须为正")
    if not frame["execution_earliest"].eq("NEXT_TRADING_DAY_OPEN").all():
        raise ValueError("联合输入允许了当日执行")
    if not (
        frame["feature_asof"].dt.normalize().eq(frame["date"])
        & frame["feature_asof"].dt.hour.eq(15)
    ).all():
        raise ValueError("联合输入特征时钟不是T日收盘")

    frame["log_fund_shares"] = np.log(frame["fund_shares"])
    for horizon in [1, 5, 20]:
        frame[f"share_change_{horizon}d"] = frame["log_fund_shares"].diff(horizon)
        frame[f"share_rank_{horizon}d"] = rolling_last_percentile(
            frame[f"share_change_{horizon}d"],
            window=rolling_window,
            minimum_observations=minimum_observations,
        )
    frame["absolute_premium_level"] = frame["close_premium_to_nav"].abs()
    frame["premium_rank"] = rolling_last_percentile(
        frame["close_premium_to_nav"],
        window=rolling_window,
        minimum_observations=minimum_observations,
    )
    frame["absolute_premium_rank"] = rolling_last_percentile(
        frame["absolute_premium_level"],
        window=rolling_window,
        minimum_observations=minimum_observations,
    )
    return frame


def build_candidate_states(features: pd.DataFrame) -> pd.DataFrame:
    """按冻结定义生成八列严格0/1状态，特征不足时保持满仓。"""

    keep = [
        "date",
        "share_change_1d",
        "share_change_5d",
        "share_change_20d",
        "close_premium_to_nav",
        "absolute_premium_level",
        "share_rank_1d",
        "share_rank_5d",
        "share_rank_20d",
        "premium_rank",
        "absolute_premium_rank",
    ]
    frame = features[keep].copy()
    frame["ETF_SHARE1D_BOTTOM10_CASH"] = np.where(
        frame["share_rank_1d"].notna() & frame["share_rank_1d"].le(0.10), 0, 1
    )
    frame["ETF_SHARE5D_BOTTOM10_CASH"] = np.where(
        frame["share_rank_5d"].notna() & frame["share_rank_5d"].le(0.10), 0, 1
    )
    frame["ETF_SHARE20D_BOTTOM10_CASH"] = np.where(
        frame["share_rank_20d"].notna() & frame["share_rank_20d"].le(0.10), 0, 1
    )
    frame["ETF_PREMIUM_BOTTOM10_CASH"] = np.where(
        frame["premium_rank"].notna() & frame["premium_rank"].le(0.10), 0, 1
    )
    frame["ETF_PREMIUM_TOP10_CASH"] = np.where(
        frame["premium_rank"].notna() & frame["premium_rank"].ge(0.90), 0, 1
    )
    frame["ETF_ABS_PREMIUM_TOP10_CASH"] = np.where(
        frame["absolute_premium_rank"].notna()
        & frame["absolute_premium_rank"].ge(0.90),
        0,
        1,
    )
    frame["ETF_SHARE5_BOTTOM20_AND_PREMIUM_BOTTOM20_CASH"] = np.where(
        frame["share_rank_5d"].notna()
        & frame["premium_rank"].notna()
        & frame["share_rank_5d"].le(0.20)
        & frame["premium_rank"].le(0.20),
        0,
        1,
    )
    frame["ETF_SHARE5_TOP20_AND_PREMIUM_TOP20_CASH"] = np.where(
        frame["share_rank_5d"].notna()
        & frame["premium_rank"].notna()
        & frame["share_rank_5d"].ge(0.80)
        & frame["premium_rank"].ge(0.80),
        0,
        1,
    )
    for candidate_id in EXPECTED_CANDIDATES:
        frame[candidate_id] = frame[candidate_id].astype(np.int8)
        if not set(frame[candidate_id].unique()).issubset({0, 1}):
            raise AssertionError(f"{candidate_id}不是严格二元状态")
    return frame


def align_states_to_market(
    market: pd.DataFrame,
    states: pd.DataFrame,
) -> pd.DataFrame:
    """对齐交易日；没有有限联合特征的T日状态明确设为满仓。"""

    aligned = market[["trade_date"]].rename(columns={"trade_date": "date"}).merge(
        states,
        on="date",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    aligned["feature_observed"] = aligned["_merge"].eq("both")
    aligned.drop(columns="_merge", inplace=True)
    for candidate_id in EXPECTED_CANDIDATES:
        aligned[candidate_id] = aligned[candidate_id].fillna(1).astype(np.int8)
        if not set(aligned[candidate_id].unique()).issubset({0, 1}):
            raise AssertionError(f"{candidate_id}对齐后不是严格二元状态")
    return aligned


def required_columns(frame: pd.DataFrame, required: list[str], label: str) -> None:
    """核对数据字段。"""

    if missing := set(required) - set(frame.columns):
        raise ValueError(f"{label}缺少字段：{sorted(missing)}")


def load_inputs(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """按固定哈希读取联合特征、执行行情、基准、分红和合并审计。"""

    contracts = config["data_contracts"]
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name, contract in contracts.items():
        path = ROOT / contract["file"]
        if not path.exists():
            raise FileNotFoundError(f"固定工件不存在：{contract['file']}")
        actual = sha256_file(path)
        if actual != contract["required_sha256"]:
            raise ValueError(f"固定工件哈希漂移：{contract['file']}")
        paths[name] = path
        hashes[contract["file"]] = actual

    feature_input = pd.read_parquet(paths["feature_input"])
    merge_audit = json.loads(paths["merge_audit"].read_text(encoding="utf-8"))
    market = pd.read_parquet(paths["execution_daily"])
    benchmark = pd.read_parquet(paths["benchmark_total_return"])
    dividends = pd.read_csv(paths["cash_distributions"])
    required_columns(
        feature_input,
        contracts["feature_input"]["required_columns"],
        "ETF份额折溢价联合输入",
    )
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
    if len(feature_input) != int(contracts["feature_input"]["required_rows"]):
        raise ValueError("ETF联合输入固定行数不匹配")
    feature_input["date"] = pd.to_datetime(
        feature_input["date"], errors="raise"
    ).dt.normalize()
    if str(feature_input["date"].min().date()) != contracts["feature_input"][
        "required_first_date"
    ]:
        raise ValueError("ETF联合输入首日不匹配")
    if str(feature_input["date"].max().date()) != contracts["feature_input"][
        "required_last_date"
    ]:
        raise ValueError("ETF联合输入末日不匹配")
    if merge_audit["status"] != "SUCCESS_FIXED_CANDIDATE_HISTORY_MERGE_NO_FILL":
        raise ValueError("ETF联合输入合并审计状态无效")
    if merge_audit["output_sha256"] != hashes[contracts["feature_input"]["file"]]:
        raise ValueError("ETF联合输入与合并审计记录不一致")
    if set(market["symbol"].dropna().astype(str).unique()) != {
        contracts["execution_daily"]["required_symbol"]
    }:
        raise ValueError("510300代码不匹配")
    if set(benchmark["symbol"].dropna().astype(str).unique()) != {
        contracts["benchmark_total_return"]["required_symbol"]
    }:
        raise ValueError("H00300代码不匹配")

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
    evaluation = load_candidate_contract()["evaluation"]
    window_start = pd.Timestamp("2016-12-01")
    window_end = max(pd.Timestamp(period["end_date"]) for period in evaluation["periods"])
    market = market.loc[
        market["trade_date"].between(window_start, window_end, inclusive="both")
    ].reset_index(drop=True)
    benchmark = benchmark.loc[
        benchmark["date"].between(window_start, window_end, inclusive="both")
    ].reset_index(drop=True)
    if sorted(set(market["trade_date"]) - set(benchmark["date"])):
        raise ValueError("H00300缺少510300执行交易日")
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
        raise ValueError("分红事件数不匹配")
    audit = {
        "hashes": hashes,
        "feature_rows": int(len(feature_input)),
        "feature_first_date": feature_input["date"].min().date().isoformat(),
        "feature_last_date": feature_input["date"].max().date().isoformat(),
        "merge_audit_status": merge_audit["status"],
        "merge_excluded_dates_without_fill": merge_audit[
            "share_only_dates_excluded_without_fill"
        ],
        "market_rows": int(len(market)),
        "benchmark_rows": int(len(benchmark)),
        "dividend_event_count": int(len(dividends)),
    }
    return {
        "feature_input": feature_input,
        "merge_audit": merge_audit,
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
    """运行八候选、两成本、四时期和五个起点。"""

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
    combined_id = "COMBINED_2017_2026"
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


def percent(value: float | None) -> str:
    """格式化百分比。"""

    return "不可计算" if value is None else f"{value:.2%}"


def render_markdown(report: dict[str, Any]) -> str:
    """生成简洁的人读报告。"""

    best = report["best_candidate"]
    return f"""# 510300 ETF份额与折溢价水平二元筛选 V1

状态：`{report['status']}`

信号只读取T日收盘后可知的基金份额与单位净值，T+1开盘执行；目标状态严格为满仓或空仓。

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
    features = build_features(
        inputs["feature_input"],
        rolling_window=int(
            contract["features"]["rolling_percentile_window_trading_rows"]
        ),
        minimum_observations=int(
            contract["features"]["rolling_percentile_minimum_rows"]
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
        else "REJECTED_FIXED_ETF_SHARE_PREMIUM_LEVEL_BINARY_FAMILY_NO_RESCUE"
    )
    decision = (
        "固定家族存在双20通过者；只允许冻结新的246交易日前瞻验证，不得直接交易。"
        if passing
        else "八个预先固定规则均未过双20门槛；禁止改分位、改方向、改窗口或组合条件营救。"
    )
    state_path = ROOT / config["artifacts"]["daily_states"]
    metrics_path = ROOT / config["artifacts"]["metrics"]
    atomic_parquet(aligned_states, state_path)
    atomic_parquet(results, metrics_path)
    first_rank_dates = {
        column: features.loc[features[column].notna(), "date"].min().date().isoformat()
        for column in [
            "share_rank_1d",
            "share_rank_5d",
            "share_rank_20d",
            "premium_rank",
            "absolute_premium_rank",
        ]
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
            "allowed_states": [0, 1],
            "benchmark": "H00300 total return",
            "signal_time": "T_CLOSE_AFTER_NAV_AND_SHARE_RECORD_AVAILABLE",
            "execution_time": "T_PLUS_1_OPEN",
            "evidence_label": contract["protocol"]["evidence_label"],
        },
        "input_audit": input_audit,
        "feature_audit": {
            "feature_rows": int(len(features)),
            "first_rank_dates": first_rank_dates,
            "market_days_without_joint_feature": int(
                (~aligned_states["feature_observed"]).sum()
            ),
            "future_510300_return_columns_read": 0,
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
