"""筛选美国时段中国ETF新增定价映射的510300/现金开盘前二元规则。"""

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
    _dividend_maps,
    _maximum_affordable_quantity,
    build_benchmark_returns,
    summarize_period,
    trading_date_with_offset,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "510300_us_china_overnight_binary_screen_v1.yaml"
CANDIDATE_CONTRACT_PATH = (
    ROOT / "config" / "510300_us_china_overnight_binary_screen_v1_candidates.yaml"
)
CANDIDATE_CONTRACT_SHA256 = (
    "836f662330d489007bbd35c700629643d5f3e5f5777dc9777a138ac587dff160"
)
STUDY_ID = "510300_US_CHINA_OVERNIGHT_BINARY_SCREEN_V1"
CHINA_ETFS = ["ASHR", "FXI", "MCHI", "KWEB"]
US_SYMBOLS = [*CHINA_ETFS, "SPY", "^VIX"]
EXPECTED_CANDIDATES = [
    "USCN_ASHR_NEG_CONTINUATION_CASH",
    "USCN_BASKET_NEG_CONTINUATION_CASH",
    "USCN_BASKET_BOTTOM10_CONTINUATION_CASH",
    "USCN_SPECIFIC_BOTTOM10_CASH",
    "USCN_ALL4_NEG_BREADTH_CASH",
    "USCN_BASKET_BOTTOM20_VIX_TOP80_CASH",
    "USCN_ASHR_TOP10_OVERREACTION_CASH",
    "USCN_BASKET_TOP10_OVERREACTION_CASH",
]


def sha256_file(path: Path) -> str:
    """流式计算文件SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    """将NumPy和Pandas值转换为JSON可序列化类型。"""

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
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default)
        + "\n",
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
    """读取在数据下载前已经固定的候选合同。"""

    actual_hash = sha256_file(CANDIDATE_CONTRACT_PATH)
    if actual_hash != CANDIDATE_CONTRACT_SHA256:
        raise ValueError("候选合同在数据获取后发生漂移")
    contract = yaml.safe_load(CANDIDATE_CONTRACT_PATH.read_text(encoding="utf-8"))
    protocol = contract["protocol"]
    if protocol["study_id"] != STUDY_ID:
        raise ValueError("候选合同研究编号不匹配")
    if protocol["state"] != "CANDIDATE_FAMILY_FIXED_BEFORE_DATA_ACQUISITION":
        raise ValueError("候选合同状态无效")
    if not bool(protocol["no_parameter_rescue"]):
        raise ValueError("候选合同必须禁止参数营救")
    candidate_ids = [item["candidate_id"] for item in contract["candidates"]]
    if candidate_ids != EXPECTED_CANDIDATES:
        raise ValueError("候选合同的八条规则或顺序发生漂移")
    if list(contract["scope"]["allowed_target_states"]) != [0, 1]:
        raise ValueError("候选合同目标状态必须严格为0和1")
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
    if float(contract["objective"]["minimum_annualized_excess"]) != 0.20:
        raise ValueError("年化净超额硬门必须为20%")
    if float(contract["objective"]["minimum_rolling_excess_median"]) != 0.20:
        raise ValueError("滚动净超额硬门必须为20%")
    if int(contract["protocol"]["candidate_family_size"]) != 8:
        raise ValueError("候选家族数量必须固定为8")
    return config, contract


def verify_freeze_manifest(config: dict[str, Any]) -> dict[str, Any]:
    """运行前核对协议、源码、测试、依赖和输入均未漂移。"""

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
    """用当前及过去有限观测计算滚动经验分位。"""

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


def _prepare_us_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """核对一个美国标的的日期、身份与调整收盘。"""

    required = {"date", "adj_close", "symbol", "source"}
    if missing := required - set(frame.columns):
        raise ValueError(f"{symbol}输入缺少字段：{sorted(missing)}")
    prepared = frame.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="raise").dt.normalize()
    prepared["adj_close"] = pd.to_numeric(prepared["adj_close"], errors="raise")
    prepared.sort_values("date", kind="mergesort", inplace=True)
    if prepared["date"].duplicated().any():
        raise ValueError(f"{symbol}日期重复")
    if set(prepared["symbol"].astype(str).unique()) != {symbol}:
        raise ValueError(f"{symbol}文件身份不匹配")
    if (prepared["adj_close"] <= 0.0).any() or not np.isfinite(
        prepared["adj_close"].to_numpy(float)
    ).all():
        raise ValueError(f"{symbol}调整收盘非法")
    return prepared.reset_index(drop=True)


def strict_preopen_interval_returns(
    china_dates: pd.Series,
    us_frame: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """计算前一中国交易日到当日开盘前的严格美国时段区间收益。"""

    prepared = _prepare_us_frame(us_frame, symbol)
    dates = pd.to_datetime(china_dates, errors="raise").dt.normalize().reset_index(
        drop=True
    )
    us_dates = prepared["date"].to_numpy(dtype="datetime64[ns]")
    us_close = prepared["adj_close"].to_numpy(dtype=float)
    interval_return = np.full(len(dates), np.nan, dtype=float)
    baseline_dates = np.full(
        len(dates), np.datetime64("NaT", "ns"), dtype="datetime64[ns]"
    )
    latest_dates = np.full(
        len(dates), np.datetime64("NaT", "ns"), dtype="datetime64[ns]"
    )
    session_counts = np.zeros(len(dates), dtype=np.int16)
    for index in range(1, len(dates)):
        previous_china_date = np.datetime64(dates.iloc[index - 1], "ns")
        current_china_date = np.datetime64(dates.iloc[index], "ns")
        baseline_index = int(np.searchsorted(us_dates, previous_china_date, side="left") - 1)
        latest_index = int(np.searchsorted(us_dates, current_china_date, side="left") - 1)
        if baseline_index < 0 or latest_index <= baseline_index:
            continue
        interval_return[index] = float(us_close[latest_index] / us_close[baseline_index] - 1.0)
        baseline_dates[index] = us_dates[baseline_index]
        latest_dates[index] = us_dates[latest_index]
        session_counts[index] = np.int16(latest_index - baseline_index)
    return pd.DataFrame(
        {
            "date": dates,
            f"{symbol}_interval_return": interval_return,
            f"{symbol}_baseline_us_date": pd.to_datetime(baseline_dates),
            f"{symbol}_latest_us_date": pd.to_datetime(latest_dates),
            f"{symbol}_new_session_count": session_counts,
        }
    )


def build_preopen_features(
    china_dates: pd.Series,
    us_frames: dict[str, pd.DataFrame],
    *,
    rolling_window: int,
    minimum_observations: int,
) -> pd.DataFrame:
    """构造只含中国开盘前已完成美国时段信息的固定特征。"""

    if set(us_frames) != set(US_SYMBOLS):
        raise ValueError("美国输入标的集合不完整")
    features = pd.DataFrame(
        {"date": pd.to_datetime(china_dates, errors="raise").dt.normalize()}
    ).reset_index(drop=True)
    for symbol in US_SYMBOLS:
        interval = strict_preopen_interval_returns(features["date"], us_frames[symbol], symbol)
        features = features.merge(interval, on="date", how="left", validate="one_to_one")

    return_columns = [f"{symbol}_interval_return" for symbol in US_SYMBOLS]
    features["signal_complete"] = features[return_columns].notna().all(axis=1)
    china_columns = [f"{symbol}_interval_return" for symbol in CHINA_ETFS]
    features["china_basket_return"] = features[china_columns].median(axis=1)
    features["china_specific_return"] = (
        features["china_basket_return"] - features["SPY_interval_return"]
    )
    features["all4_negative"] = features[china_columns].lt(0.0).all(axis=1)
    features.loc[~features["signal_complete"], "china_basket_return"] = np.nan
    features.loc[~features["signal_complete"], "china_specific_return"] = np.nan
    features.loc[~features["signal_complete"], "all4_negative"] = False
    features["rank_ashr"] = rolling_last_percentile(
        features["ASHR_interval_return"],
        window=rolling_window,
        minimum_observations=minimum_observations,
    )
    features["rank_china_basket"] = rolling_last_percentile(
        features["china_basket_return"],
        window=rolling_window,
        minimum_observations=minimum_observations,
    )
    features["rank_china_specific"] = rolling_last_percentile(
        features["china_specific_return"],
        window=rolling_window,
        minimum_observations=minimum_observations,
    )
    features["rank_vix_return"] = rolling_last_percentile(
        features["^VIX_interval_return"],
        window=rolling_window,
        minimum_observations=minimum_observations,
    )
    latest_columns = [f"{symbol}_latest_us_date" for symbol in US_SYMBOLS]
    features["signal_asof_us_date"] = features[latest_columns].max(axis=1)
    features["same_calendar_day_us_bar_used"] = False
    for symbol in US_SYMBOLS:
        invalid = features[f"{symbol}_latest_us_date"].ge(features["date"])
        if bool(invalid.fillna(False).any()):
            raise AssertionError(f"{symbol}错误使用中国开盘后的美国日期")
    return features


def build_candidate_states(features: pd.DataFrame) -> pd.DataFrame:
    """生成八列严格0/1的固定开盘前状态。"""

    required = {
        "date",
        "signal_complete",
        "ASHR_interval_return",
        "china_basket_return",
        "china_specific_return",
        "all4_negative",
        "rank_ashr",
        "rank_china_basket",
        "rank_china_specific",
        "rank_vix_return",
        "signal_asof_us_date",
    }
    if missing := required - set(features.columns):
        raise ValueError(f"开盘前特征缺少字段：{sorted(missing)}")
    complete = features["signal_complete"].astype(bool)
    output = features[["date", "signal_asof_us_date", "signal_complete"]].copy()

    def state_when(condition: pd.Series) -> pd.Series:
        return pd.Series(np.where(complete & condition.fillna(False), 0, 1), index=features.index)

    output["USCN_ASHR_NEG_CONTINUATION_CASH"] = state_when(
        features["ASHR_interval_return"].lt(0.0)
    )
    output["USCN_BASKET_NEG_CONTINUATION_CASH"] = state_when(
        features["china_basket_return"].lt(0.0)
    )
    output["USCN_BASKET_BOTTOM10_CONTINUATION_CASH"] = state_when(
        features["rank_china_basket"].notna()
        & features["rank_china_basket"].le(0.10)
    )
    output["USCN_SPECIFIC_BOTTOM10_CASH"] = state_when(
        features["rank_china_specific"].notna()
        & features["rank_china_specific"].le(0.10)
    )
    output["USCN_ALL4_NEG_BREADTH_CASH"] = state_when(
        features["all4_negative"].astype(bool)
    )
    output["USCN_BASKET_BOTTOM20_VIX_TOP80_CASH"] = state_when(
        features["rank_china_basket"].notna()
        & features["rank_china_basket"].le(0.20)
        & features["rank_vix_return"].notna()
        & features["rank_vix_return"].ge(0.80)
    )
    output["USCN_ASHR_TOP10_OVERREACTION_CASH"] = state_when(
        features["rank_ashr"].notna() & features["rank_ashr"].ge(0.90)
    )
    output["USCN_BASKET_TOP10_OVERREACTION_CASH"] = state_when(
        features["rank_china_basket"].notna()
        & features["rank_china_basket"].ge(0.90)
    )
    for candidate_id in EXPECTED_CANDIDATES:
        output[candidate_id] = output[candidate_id].astype(np.int8)
        if not set(output[candidate_id].unique()).issubset({0, 1}):
            raise AssertionError(f"{candidate_id}不是严格二元状态")
    return output


def required_columns(frame: pd.DataFrame, required: list[str], label: str) -> None:
    """核对输入字段。"""

    if missing := set(required) - set(frame.columns):
        raise ValueError(f"{label}缺少字段：{sorted(missing)}")


def load_inputs(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """按固定哈希读取六个美国输入及510300执行、基准和分红。"""

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

    us_frames: dict[str, pd.DataFrame] = {}
    for key, symbol in config["us_symbol_contract_keys"].items():
        frame = pd.read_parquet(paths[key])
        required_columns(frame, contracts[key]["required_columns"], symbol)
        if len(frame) != int(contracts[key]["required_rows"]):
            raise ValueError(f"{symbol}固定行数不匹配")
        prepared = _prepare_us_frame(frame, symbol)
        if prepared["date"].min().date().isoformat() != contracts[key]["required_first_date"]:
            raise ValueError(f"{symbol}固定首日不匹配")
        if prepared["date"].max().date().isoformat() != contracts[key]["required_last_date"]:
            raise ValueError(f"{symbol}固定末日不匹配")
        us_frames[symbol] = prepared

    market = pd.read_parquet(paths["execution_daily"])
    benchmark = pd.read_parquet(paths["benchmark_total_return"])
    dividends = pd.read_csv(paths["cash_distributions"])
    required_columns(market, contracts["execution_daily"]["required_columns"], "510300日线")
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
    if set(market["symbol"].dropna().astype(str).unique()) != {
        contracts["execution_daily"]["required_symbol"]
    }:
        raise ValueError("510300代码不匹配")
    if set(benchmark["symbol"].dropna().astype(str).unique()) != {
        contracts["benchmark_total_return"]["required_symbol"]
    }:
        raise ValueError("H00300代码不匹配")
    market = market.rename(columns={"date": "trade_date"}).copy()
    market["trade_date"] = pd.to_datetime(market["trade_date"], errors="raise").dt.normalize()
    market.sort_values("trade_date", kind="mergesort", inplace=True)
    market.drop_duplicates("trade_date", keep="last", inplace=True)
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="raise").dt.normalize()
    benchmark.sort_values("date", kind="mergesort", inplace=True)
    benchmark.drop_duplicates("date", keep="last", inplace=True)
    window_start = pd.Timestamp(config["evaluation_window"]["feature_start_date"])
    window_end = pd.Timestamp(config["evaluation_window"]["end_date"])
    market = market.loc[
        market["trade_date"].between(window_start, window_end, inclusive="both")
    ].reset_index(drop=True)
    benchmark = benchmark.loc[
        benchmark["date"].between(window_start, window_end, inclusive="both")
    ].reset_index(drop=True)
    if sorted(set(market["trade_date"]) - set(benchmark["date"])):
        raise ValueError("H00300缺少510300执行交易日")
    if (market[["open", "close"]] <= 0.0).any().any() or (benchmark["close"] <= 0.0).any():
        raise ValueError("执行或基准价格非法")
    for column in ["record_date", "ex_date", "payment_date"]:
        dividends[column] = pd.to_datetime(dividends[column], errors="raise").dt.normalize()
    dividends["cash_dividend_per_share"] = pd.to_numeric(
        dividends["cash_dividend_per_share"], errors="raise"
    )
    if len(dividends) != int(contracts["cash_distributions"]["required_event_count"]):
        raise ValueError("分红事件数不匹配")
    audit = {
        "hashes": hashes,
        "us_symbols": {
            symbol: {
                "rows": int(len(frame)),
                "first_date": frame["date"].min().date().isoformat(),
                "last_date": frame["date"].max().date().isoformat(),
            }
            for symbol, frame in us_frames.items()
        },
        "market_rows": int(len(market)),
        "market_first_date": market["trade_date"].min().date().isoformat(),
        "market_last_date": market["trade_date"].max().date().isoformat(),
        "benchmark_rows": int(len(benchmark)),
        "dividend_event_count": int(len(dividends)),
    }
    return {
        "us_frames": us_frames,
        "market": market,
        "benchmark": benchmark,
        "dividends": dividends,
    }, audit


def cost_model(config: dict[str, Any], scenario: str) -> CostModel:
    """构造基础或压力成本。"""

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
        trading_days_per_year=int(config["annualization_trading_days"]),
        lot_size=int(costs["lot_size_shares"]),
    )


def simulate_preopen_candidate(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    target_states: np.ndarray,
    signal_asof_dates: pd.Series,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    costs: CostModel,
    initial_capital: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """按D日开盘前状态在D日开盘成交，执行整手、分红和T+1。"""

    if len(market) != len(target_states) or len(market) != len(signal_asof_dates):
        raise ValueError("市场、状态和信号时点长度不一致")
    if not set(np.unique(target_states)).issubset({0, 1}):
        raise ValueError("目标状态只能为0或1")
    in_window = market["trade_date"].between(start_date, end_date)
    indices = np.flatnonzero(in_window.to_numpy())
    if len(indices) == 0:
        raise ValueError("模拟区间没有市场行情")
    first_index = int(indices[0])
    last_index = int(indices[-1])
    if first_index == 0:
        raise ValueError("模拟起点必须有前一交易日作为账本锚点")
    anchor_date = pd.Timestamp(market.iloc[first_index - 1]["trade_date"])

    cash = float(initial_capital)
    shares = 0.0
    receivable = 0.0
    last_buy_date: pd.Timestamp | None = None
    payment_schedule: dict[pd.Timestamp, float] = {}
    ex_map = _dividend_maps(dividends)
    ledger_rows: list[dict[str, Any]] = [
        {
            "date": anchor_date,
            "equity": float(initial_capital),
            "shares": 0.0,
            "cash": float(initial_capital),
            "dividend_receivable": 0.0,
            "actual_state": 0,
            "desired_state_preopen": int(target_states[first_index]),
            "daily_commission": 0.0,
            "daily_slippage_cost": 0.0,
            "blocked_t1_exit_count": 0,
        }
    ]
    trade_rows: list[dict[str, Any]] = []
    total_blocked = 0
    total_reinvestment_buys = 0
    same_or_future_signal_date_violations = 0

    for index in range(first_index, last_index + 1):
        row = market.iloc[index]
        date = pd.Timestamp(row["trade_date"])
        cash *= 1.0 + costs.cash_annual_rate / costs.trading_days_per_year
        daily_commission = 0.0
        daily_slippage = 0.0
        daily_blocked = 0
        shares_at_open = shares
        for event in ex_map.get(date, []):
            entitlement = shares_at_open * float(event["cash_dividend_per_share"])
            if entitlement > 0.0:
                receivable += entitlement
                payment_date = pd.Timestamp(event["payment_date"])
                payment_schedule[payment_date] = (
                    payment_schedule.get(payment_date, 0.0) + entitlement
                )

        desired_state = int(target_states[index])
        signal_asof = pd.Timestamp(signal_asof_dates.iloc[index])
        if pd.notna(signal_asof) and signal_asof >= date:
            same_or_future_signal_date_violations += 1
        raw_open = float(row["open"])
        execution_time = date + pd.Timedelta(hours=9, minutes=30)
        if desired_state == 0 and shares > 0.0:
            if last_buy_date is not None and last_buy_date == date:
                daily_blocked += 1
                total_blocked += 1
            else:
                quantity = shares
                execution_price = raw_open * (1.0 - costs.slippage_bps / 10_000.0)
                notional = quantity * execution_price
                commission = costs.commission(notional)
                slippage_cost = quantity * (raw_open - execution_price)
                cash += notional - commission
                shares = 0.0
                daily_commission += commission
                daily_slippage += slippage_cost
                trade_rows.append(
                    {
                        "date": date,
                        "signal_asof_us_date": signal_asof,
                        "execution_time": execution_time,
                        "side": "SELL",
                        "reason": "PREOPEN_STATE_EXIT_TO_CASH",
                        "raw_open": raw_open,
                        "execution_price": execution_price,
                        "quantity": quantity,
                        "execution_notional": notional,
                        "commission": commission,
                        "slippage_cost": slippage_cost,
                        "scenario": costs.scenario,
                    }
                )
        elif desired_state == 1:
            quantity, execution_price, commission = _maximum_affordable_quantity(
                cash, raw_open, costs
            )
            if quantity > 0.0:
                notional = quantity * execution_price
                slippage_cost = quantity * (execution_price - raw_open)
                reason = (
                    "PREOPEN_STATE_ENTRY_FULL"
                    if shares == 0.0
                    else "FULL_STATE_CASH_REINVESTMENT"
                )
                if shares > 0.0:
                    total_reinvestment_buys += 1
                cash -= notional + commission
                shares += quantity
                last_buy_date = date
                daily_commission += commission
                daily_slippage += slippage_cost
                trade_rows.append(
                    {
                        "date": date,
                        "signal_asof_us_date": signal_asof,
                        "execution_time": execution_time,
                        "side": "BUY",
                        "reason": reason,
                        "raw_open": raw_open,
                        "execution_price": execution_price,
                        "quantity": quantity,
                        "execution_notional": notional,
                        "commission": commission,
                        "slippage_cost": slippage_cost,
                        "scenario": costs.scenario,
                    }
                )

        payable_dates = [payment for payment in payment_schedule if payment <= date]
        payment_today = float(
            sum(payment_schedule.pop(payment) for payment in payable_dates)
        )
        if payment_today > 0.0:
            cash += payment_today
            receivable -= payment_today
            if abs(receivable) < 1e-8:
                receivable = 0.0
        close = float(row["close"])
        equity = cash + shares * close + receivable
        ledger_rows.append(
            {
                "date": date,
                "equity": equity,
                "shares": shares,
                "cash": cash,
                "dividend_receivable": receivable,
                "actual_state": int(shares > 0.0),
                "desired_state_preopen": desired_state,
                "daily_commission": daily_commission,
                "daily_slippage_cost": daily_slippage,
                "blocked_t1_exit_count": daily_blocked,
            }
        )

    ledger = pd.DataFrame(ledger_rows)
    ledger["date"] = pd.to_datetime(ledger["date"]).dt.normalize()
    ledger["daily_return"] = ledger["equity"].pct_change(fill_method=None).fillna(0.0)
    ledger["equity_peak"] = ledger["equity"].cummax()
    ledger["drawdown"] = ledger["equity"] / ledger["equity_peak"] - 1.0
    trades = pd.DataFrame(trade_rows)
    diagnostics = {
        "t_plus_one_blocked_exit_attempts": int(total_blocked),
        "t_plus_one_violations": 0,
        "same_or_future_signal_date_violations": int(
            same_or_future_signal_date_violations
        ),
        "full_state_reinvestment_buy_legs": int(total_reinvestment_buys),
        "anchor_date": anchor_date.date().isoformat(),
        "actual_start_date": pd.Timestamp(ledger["date"].iloc[1]).date().isoformat(),
        "actual_end_date": pd.Timestamp(ledger["date"].iloc[-1]).date().isoformat(),
    }
    return ledger, trades, diagnostics


def evaluate(
    config: dict[str, Any],
    contract: dict[str, Any],
    market: pd.DataFrame,
    benchmark: pd.DataFrame,
    dividends: pd.DataFrame,
    states: pd.DataFrame,
) -> pd.DataFrame:
    """运行八候选、两成本、四时期和五个起点。"""

    benchmark_returns = build_benchmark_returns(benchmark)
    market_dates = [pd.Timestamp(value) for value in market["trade_date"].tolist()]
    objective = contract["objective"]
    rows: list[dict[str, Any]] = []
    for candidate in contract["candidates"]:
        candidate_id = candidate["candidate_id"]
        target_states = states[candidate_id].to_numpy(dtype=np.int8)
        for scenario in contract["evaluation"]["cost_scenarios"]:
            costs = cost_model(config, scenario)
            for period in contract["evaluation"]["periods"]:
                for offset in contract["evaluation"][
                    "start_date_perturbations_trading_days"
                ]:
                    start_date = trading_date_with_offset(
                        market_dates, pd.Timestamp(period["start_date"]), int(offset)
                    )
                    ledger, trades, diagnostics = simulate_preopen_candidate(
                        market,
                        dividends,
                        target_states,
                        states["signal_asof_us_date"],
                        start_date=start_date,
                        end_date=pd.Timestamp(period["end_date"]),
                        costs=costs,
                        initial_capital=float(config["initial_capital_cny"]),
                    )
                    summary = summarize_period(
                        ledger,
                        trades,
                        benchmark_returns,
                        period_id=period["id"],
                        start_date=start_date,
                        end_date=pd.Timestamp(period["end_date"]),
                        objective=objective,
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
                            "same_or_future_signal_date_violations": int(
                                diagnostics["same_or_future_signal_date_violations"]
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_candidates(
    contract: dict[str, Any],
    results: pd.DataFrame,
    states: pd.DataFrame,
) -> list[dict[str, Any]]:
    """汇总每个固定规则的最差门槛和综合期表现。"""

    summaries: list[dict[str, Any]] = []
    combined_id = "COMBINED_2017_2026"
    for candidate in contract["candidates"]:
        candidate_id = candidate["candidate_id"]
        candidate_rows = results.loc[results["candidate_id"].eq(candidate_id)]
        stress = candidate_rows.loc[candidate_rows["scenario"].eq("STRESS")]
        base = candidate_rows.loc[candidate_rows["scenario"].eq("BASE")]
        stress_combined = stress.loc[
            stress["period_id"].eq(combined_id) & stress["start_offset"].eq(0)
        ].iloc[0]
        base_combined = base.loc[
            base["period_id"].eq(combined_id) & base["start_offset"].eq(0)
        ].iloc[0]
        hard_pass = bool(stress["both_20pct_gates"].all())
        positive_shadow = bool(
            (stress["annualized_excess"] > 0.0).all()
            & (stress["rolling_242d_excess_median"] > 0.0).all()
        )
        state_series = states[candidate_id]
        summaries.append(
            {
                "candidate_id": candidate_id,
                "rule": candidate["rule"],
                "mechanism": candidate["mechanism"],
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
                    max(0, state_series.diff().ne(0).sum() - 1)
                ),
                "maximum_t_plus_one_violations": int(
                    candidate_rows["t_plus_one_violations"].max()
                ),
                "maximum_same_or_future_signal_date_violations": int(
                    candidate_rows["same_or_future_signal_date_violations"].max()
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
    return f"""# 510300 美国中国ETF隔夜信息二元筛选 V1

状态：`{report['status']}`

只使用D日09:15前已经结束的美国时段；美国同日尚未发生的日线严格禁用，D日开盘执行，持仓严格为510300满仓或现金空仓。

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
    features = build_preopen_features(
        inputs["market"]["trade_date"],
        inputs["us_frames"],
        rolling_window=int(
            contract["features"]["rolling_percentile_window_china_days"]
        ),
        minimum_observations=int(
            contract["features"]["rolling_percentile_minimum_observations"]
        ),
    )
    states = build_candidate_states(features)
    if not states["date"].equals(inputs["market"]["trade_date"]):
        raise AssertionError("开盘前状态与510300交易日错位")
    results = evaluate(
        config,
        contract,
        inputs["market"],
        inputs["benchmark"],
        inputs["dividends"],
        states,
    )
    summaries = summarize_candidates(contract, results, states)
    passing = [item["candidate_id"] for item in summaries if item["hard_pass"]]
    positive = [
        item["candidate_id"] for item in summaries if item["positive_shadow"]
    ]
    status = (
        "DISCOVERY_CANDIDATE_FOUND_REQUIRES_246D_PROSPECTIVE_FREEZE"
        if passing
        else "REJECTED_FIXED_US_CHINA_OVERNIGHT_BINARY_FAMILY_NO_RESCUE"
    )
    decision = (
        "固定家族存在双20通过者；只允许冻结新的246交易日前瞻验证，不得直接交易。"
        if passing
        else "八个预先固定规则均未过双20门槛；禁止改阈值、改符号或改信息时钟营救。"
    )
    feature_path = ROOT / config["artifacts"]["features"]
    state_path = ROOT / config["artifacts"]["daily_states"]
    metrics_path = ROOT / config["artifacts"]["metrics"]
    atomic_parquet(features, feature_path)
    atomic_parquet(states, state_path)
    atomic_parquet(results, metrics_path)
    report = {
        "status": status,
        "study_id": STUDY_ID,
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_sha256": manifest["protocol_sha256"],
        "candidate_contract_sha256": CANDIDATE_CONTRACT_SHA256,
        "freeze_manifest_sha256": sha256_file(
            ROOT / config["artifacts"]["freeze_manifest"]
        ),
        "scope": {
            "execution_asset": "510300.SH",
            "allowed_states": [0, 1],
            "benchmark": "H00300 total return",
            "signal_cutoff": "D 09:15 Asia/Shanghai",
            "execution_time": "D open call auction",
            "evidence_label": contract["protocol"]["evidence_label"],
        },
        "input_audit": input_audit,
        "feature_audit": {
            "feature_rows": int(len(features)),
            "complete_signal_rows": int(features["signal_complete"].sum()),
            "incomplete_signal_rows_default_full": int(
                (~features["signal_complete"]).sum()
            ),
            "first_complete_signal_date": features.loc[
                features["signal_complete"], "date"
            ].min().date().isoformat(),
            "last_complete_signal_date": features.loc[
                features["signal_complete"], "date"
            ].max().date().isoformat(),
            "same_calendar_day_us_bar_used_count": int(
                features["same_calendar_day_us_bar_used"].sum()
            ),
            "maximum_us_sessions_aggregated": int(
                features[[f"{symbol}_new_session_count" for symbol in US_SYMBOLS]]
                .max(axis=1)
                .max()
            ),
            "future_return_columns_read": 0,
        },
        "candidate_count": len(EXPECTED_CANDIDATES),
        "result_rows": int(len(results)),
        "passing_candidates": passing,
        "positive_shadow_candidates": positive,
        "best_candidate": summaries[0],
        "candidate_summaries": summaries,
        "maximum_t_plus_one_violations": int(results["t_plus_one_violations"].max()),
        "maximum_same_or_future_signal_date_violations": int(
            results["same_or_future_signal_date_violations"].max()
        ),
        "artifacts": {
            feature_path.relative_to(ROOT).as_posix(): sha256_file(feature_path),
            state_path.relative_to(ROOT).as_posix(): sha256_file(state_path),
            metrics_path.relative_to(ROOT).as_posix(): sha256_file(metrics_path),
        },
        "decision": decision,
        "boundaries": contract["boundaries"],
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
                "maximum_same_or_future_signal_date_violations": report[
                    "maximum_same_or_future_signal_date_violations"
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
