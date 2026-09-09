"""510300上涨/下跌/震荡三态趋势路由V1.0.1合同修正版。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.engine import (  # noqa: E402
    BacktestCosts,
    run_long_cash_backtest,
    summarize_backtest,
)


CONFIG_PATH = ROOT / "config" / "510300_three_state_trend_router_v1_0_1.yaml"
MANIFEST_PATH = ROOT / "config" / "510300_three_state_trend_router_v1_0_1_manifest.json"

STATE_BULL = "BULL_TREND"
STATE_BEAR = "BEAR_TREND"
STATE_RANGE = "RANGE_OR_UNCONFIRMED"
ALL_STATES = (STATE_BULL, STATE_BEAR, STATE_RANGE)


class ContractError(RuntimeError):
    """冻结协议、数据合同或时钟不满足要求。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_path(value: str) -> Path:
    return ROOT / PurePosixPath(value)


def _nested(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    value: Any = mapping
    for key in keys:
        value = value[key]
    return value


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    expected = {
        ("protocol", "project_id"): "510300_THREE_STATE_TREND_ROUTER_V1_0_1",
        ("protocol", "one_shot"): True,
        ("scope", "execution_asset"): "510300.SH",
        ("scope", "signal_asset"): "000300.SH_PRICE_INDEX",
        ("dates", "evaluation_start"): "2015-01-05",
        ("dates", "evaluation_end"): "2026-08-14",
        ("directional_movement_system", "period_trading_days"): 14,
        ("directional_movement_system", "trend_entry_adx"): 25.0,
        ("directional_movement_system", "trend_exit_adx"): 20.0,
        ("states", STATE_BULL, "target_position"): 1.0,
        ("states", STATE_BEAR, "target_position"): 0.0,
        ("states", STATE_RANGE, "target_position"): 0.0,
        ("range_policy", "historical_policy"): "NO_TRADE",
        ("range_policy", "range_t_backtest_allowed"): False,
        ("portfolio", "initial_capital_cny"): 20000.0,
        ("portfolio", "lot_size_shares"): 100,
        ("portfolio", "base_costs", "commission_rate_per_leg"): 0.0003,
        ("portfolio", "base_costs", "minimum_commission_cny_per_leg"): 5.0,
        ("portfolio", "base_costs", "slippage_bps_per_leg"): 5.0,
        ("portfolio", "stress_costs", "slippage_bps_per_leg"): 10.0,
        ("portfolio_gates", "base_net_sharpe_minimum"): 1.2,
        ("portfolio_gates", "stress_net_sharpe_minimum"): 1.2,
        ("selection_bias", "prior_manifest_count"): 346,
        ("selection_bias", "non_manifest_prefreeze_failed_attempt_count"): 1,
        ("selection_bias", "expected_total_trial_count_including_current"): 348,
    }
    for keys, wanted in expected.items():
        observed = _nested(config, keys)
        if observed != wanted:
            raise ContractError(
                f"冻结字段{'.'.join(keys)}异常：期望{wanted!r}，得到{observed!r}"
            )
    if config["scope"]["allowed_holdings"] != ["510300.SH", "CASH_CNY"]:
        raise ContractError("持仓集合必须严格等于510300.SH与人民币现金")
    forbidden_scope = (
        "leverage_allowed",
        "short_selling_allowed",
        "derivatives_execution_allowed",
        "live_trading_authorized",
    )
    if any(bool(config["scope"][key]) for key in forbidden_scope):
        raise ContractError("作用域不得授权杠杆、卖空、衍生品或实盘")
    rescue_fields = (
        "parameter_rescue_after_result",
        "threshold_rescue_after_result",
        "window_rescue_after_result",
        "state_label_rescue_after_result",
        "range_policy_rescue_after_result",
        "direction_reversal_after_result",
        "combination_with_rejected_candidates_after_result",
    )
    if any(config["protocol"][key] != "forbidden" for key in rescue_fields):
        raise ContractError("结果后救援开关没有全部冻结为forbidden")
    governance_switches = (
        "paper_signal_allowed",
        "shadow_signal_allowed",
        "position_mapping_enabled",
        "order_generation",
        "broker_connection",
        "position_change",
        "live_trading_authorized",
    )
    if any(bool(config["governance"][key]) for key in governance_switches):
        raise ContractError("研究协议不得产生信号、仓位、订单、券商连接或实盘")
    system = config["directional_movement_system"]
    if float(system["trend_exit_adx"]) >= float(system["trend_entry_adx"]):
        raise ContractError("ADX退出阈值必须严格低于进入阈值")
    if config["mechanism_evaluation"]["forward_horizons_trading_days"] != [5, 20, 60]:
        raise ContractError("机制评价期限必须固定为5、20、60日")
    if config["mechanism_evaluation"]["primary_horizon_trading_days"] != 20:
        raise ContractError("主要状态期限必须固定为20日")


def validate_manifest(config: dict[str, Any]) -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise ContractError("冻结清单不存在，禁止读取候选结果")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("project_id") != config["protocol"]["project_id"]:
        raise ContractError("冻结清单项目编号不匹配")
    if manifest.get("state") != "FROZEN_BEFORE_FIRST_CANDIDATE_EVALUATION":
        raise ContractError("冻结清单状态不允许候选评价")
    if manifest.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise ContractError("冻结后配置文件发生漂移")
    mismatches: dict[str, dict[str, str]] = {}
    for section in ("tracked_files", "input_files"):
        for relative, expected in manifest.get(section, {}).items():
            path = _project_path(relative)
            actual = sha256_file(path) if path.exists() else "MISSING"
            if actual != expected:
                mismatches[relative] = {"expected": expected, "actual": actual}
    if mismatches:
        raise ContractError(f"冻结文件发生漂移：{mismatches}")
    if manifest.get("candidate_2015_plus_outcomes_read_before_freeze") is not False:
        raise ContractError("冻结前候选结果可见性证明失败")
    if manifest.get("candidate_portfolio_returns_read_before_freeze") is not False:
        raise ContractError("冻结前组合收益可见性证明失败")
    selection = manifest.get("selection_bias_control", {})
    if int(selection.get("prior_manifest_count", -1)) != int(
        config["selection_bias"]["prior_manifest_count"]
    ):
        raise ContractError("冻结前清单计数不符合协议")
    if int(selection.get("total_trial_count_including_current", -1)) != int(
        config["selection_bias"]["expected_total_trial_count_including_current"]
    ):
        raise ContractError("累计试验计数不符合协议")
    return manifest


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_market(frame: pd.DataFrame, required: list[str], name: str) -> pd.DataFrame:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ContractError(f"{name}缺少字段：{missing}")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result.sort_values("date", inplace=True)
    result.reset_index(drop=True, inplace=True)
    if result["date"].duplicated().any():
        raise ContractError(f"{name}存在重复日期")
    return result


def load_inputs(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    specifications = config["inputs"]
    for name, specification in specifications.items():
        path = _project_path(specification["path"])
        if not path.exists():
            raise ContractError(f"缺少输入{name}：{path}")
        expected_hash = specification.get("sha256")
        if expected_hash and sha256_file(path) != expected_hash:
            raise ContractError(f"输入{name}哈希不匹配")

    audit = _load_json(_project_path(specifications["input_audit"]["path"]))
    if audit.get("status") != specifications["input_audit"]["required_status"]:
        raise ContractError("预冻结输入审计状态不通过")
    if audit.get("candidate_2015_plus_outcomes_read_or_computed") is not False:
        raise ContractError("预冻结输入审计读取了候选结果")
    if audit.get("candidate_portfolio_returns_read_or_computed") is not False:
        raise ContractError("预冻结输入审计读取了组合收益")
    if audit.get("range_t_data_gate", {}).get("range_t_return_evaluation") != "NOT_ALLOWED":
        raise ContractError("震荡做T数据门没有保持NOT_ALLOWED")

    etf_spec = specifications["etf_daily"]
    index_spec = specifications["index_daily"]
    tri_spec = specifications["benchmark_total_return"]
    dividend_spec = specifications["dividends"]
    etf = _normalize_market(
        pd.read_parquet(_project_path(etf_spec["path"])),
        etf_spec["required_columns"],
        "510300日线",
    )
    index = _normalize_market(
        pd.read_parquet(_project_path(index_spec["path"])),
        index_spec["required_columns"],
        "000300日线",
    )
    tri = _normalize_market(
        pd.read_parquet(_project_path(tri_spec["path"])),
        tri_spec["required_columns"],
        "H00300全收益指数",
    )
    dividends = pd.read_csv(_project_path(dividend_spec["path"]))
    missing_dividends = sorted(
        set(dividend_spec["required_columns"]).difference(dividends.columns)
    )
    if missing_dividends:
        raise ContractError(f"分红输入缺少字段：{missing_dividends}")
    for field in ("record_date", "ex_date", "payment_date"):
        dividends[field] = pd.to_datetime(dividends[field], errors="raise").dt.normalize()
    dividends["cash_dividend_per_share"] = pd.to_numeric(
        dividends["cash_dividend_per_share"], errors="raise"
    )
    dividends.sort_values("ex_date", inplace=True)
    dividends.reset_index(drop=True, inplace=True)

    for name, frame, specification in (
        ("510300日线", etf, etf_spec),
        ("000300日线", index, index_spec),
        ("H00300全收益指数", tri, tri_spec),
    ):
        if len(frame) != int(specification["required_rows"]):
            raise ContractError(f"{name}行数不符合冻结合同")
        if str(frame["date"].min().date()) != specification["required_first_date"]:
            raise ContractError(f"{name}起始日期不符合冻结合同")
        if str(frame["date"].max().date()) != specification["required_last_date"]:
            raise ContractError(f"{name}截止日期不符合冻结合同")
        symbols = sorted(frame["symbol"].dropna().astype(str).unique().tolist())
        if symbols != [specification["required_symbol"]]:
            raise ContractError(f"{name}证券代码不符合冻结合同：{symbols}")

    for frame, name in ((etf, "510300日线"), (index, "000300日线")):
        numeric = frame[["open", "high", "low", "close"]].apply(
            pd.to_numeric, errors="coerce"
        )
        if numeric.isna().any().any() or (numeric <= 0).any().any():
            raise ContractError(f"{name}存在空值、零值或负值")
        invalid = (
            (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1))
            | (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1))
        )
        if invalid.any():
            raise ContractError(f"{name}违反OHLC约束")
        frame[["open", "high", "low", "close"]] = numeric
    tri["close"] = pd.to_numeric(tri["close"], errors="coerce")
    if tri["close"].isna().any() or (tri["close"] <= 0).any():
        raise ContractError("H00300存在无效收盘值")
    if len(dividends) != int(dividend_spec["required_rows"]):
        raise ContractError("分红事件数量不符合冻结合同")
    if str(dividends["ex_date"].min().date()) != dividend_spec["required_first_ex_date"]:
        raise ContractError("首个除息日不符合冻结合同")
    if str(dividends["ex_date"].max().date()) != dividend_spec["required_last_ex_date"]:
        raise ContractError("最后除息日不符合冻结合同")
    if (dividends["cash_dividend_per_share"] <= 0).any():
        raise ContractError("分红金额必须为正数")

    etf_dates = set(etf["date"])
    index_dates = set(index["date"])
    tri_dates = set(tri["date"])
    if etf_dates != index_dates:
        raise ContractError("510300与000300交易日不一致")
    allowed_extra = {pd.Timestamp(value) for value in tri_spec["allowed_extra_dates"]}
    if etf_dates.difference(tri_dates):
        raise ContractError("H00300缺少执行资产交易日")
    if tri_dates.difference(etf_dates) != allowed_extra:
        raise ContractError("H00300额外日期不符合冻结合同")
    start = pd.Timestamp(config["dates"]["evaluation_start"])
    end = pd.Timestamp(config["dates"]["evaluation_end"])
    etf_eval = etf.loc[etf["date"].between(start, end), "date"].reset_index(drop=True)
    index_eval = index.loc[index["date"].between(start, end), "date"].reset_index(drop=True)
    expected_rows = int(config["data_contract"]["expected_evaluation_rows"])
    if len(etf_eval) != expected_rows or len(index_eval) != expected_rows:
        raise ContractError("评价期交易日数不符合冻结合同")
    if not etf_eval.equals(index_eval):
        raise ContractError("评价期信号与执行资产交易日不一致")
    return etf, index, tri, dividends, audit


def wilder_rma(values: np.ndarray | pd.Series, period: int) -> np.ndarray:
    """严格按首个完整窗口均值初始化的Wilder递归平滑。"""

    array = np.asarray(values, dtype=float)
    if period <= 0:
        raise ValueError("Wilder周期必须为正整数")
    output = np.full(len(array), np.nan, dtype=float)
    finite_positions = np.flatnonzero(np.isfinite(array))
    if len(finite_positions) < period:
        return output
    first = int(finite_positions[0])
    initial_end = first + period - 1
    if initial_end >= len(array):
        return output
    initial_window = array[first : initial_end + 1]
    if not np.isfinite(initial_window).all():
        raise ValueError("Wilder初始窗口必须连续且有限")
    output[initial_end] = float(initial_window.mean())
    for index in range(initial_end + 1, len(array)):
        current = array[index]
        previous = output[index - 1]
        if not math.isfinite(current) or not math.isfinite(previous):
            output[index] = np.nan
        else:
            output[index] = (previous * (period - 1) + current) / period
    return output


def build_directional_movement_panel(
    index_daily: pd.DataFrame, period: int
) -> pd.DataFrame:
    frame = index_daily[["date", "open", "high", "low", "close"]].copy()
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    previous_close = np.roll(close, 1)
    previous_close[0] = np.nan
    up_move = np.diff(high, prepend=np.nan)
    down_move = -np.diff(low, prepend=np.nan)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    true_range = np.nanmax(
        np.column_stack(
            (
                high - low,
                np.abs(high - previous_close),
                np.abs(low - previous_close),
            )
        ),
        axis=1,
    )
    smoothed_tr = wilder_rma(true_range, period)
    smoothed_plus_dm = wilder_rma(plus_dm, period)
    smoothed_minus_dm = wilder_rma(minus_dm, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * smoothed_plus_dm / smoothed_tr
        minus_di = 100.0 * smoothed_minus_dm / smoothed_tr
        denominator = plus_di + minus_di
        dx = 100.0 * np.abs(plus_di - minus_di) / denominator
    plus_di[~np.isfinite(plus_di)] = np.nan
    minus_di[~np.isfinite(minus_di)] = np.nan
    dx[~np.isfinite(dx)] = np.nan
    adx = wilder_rma(dx, period)
    frame["true_range"] = true_range
    frame["smoothed_true_range"] = smoothed_tr
    frame["plus_di"] = plus_di
    frame["minus_di"] = minus_di
    frame["dx"] = dx
    frame["adx"] = adx
    return frame


def classify_three_states(
    panel: pd.DataFrame, entry_adx: float, exit_adx: float
) -> pd.DataFrame:
    if exit_adx >= entry_adx:
        raise ValueError("退出阈值必须低于进入阈值")
    result = panel.copy()
    current_state = STATE_RANGE
    states: list[str] = []
    directions: list[str] = []
    transition_reasons: list[str] = []
    previous_state: str | None = None
    for row in result.itertuples(index=False):
        adx = float(row.adx) if pd.notna(row.adx) else math.nan
        plus_di = float(row.plus_di) if pd.notna(row.plus_di) else math.nan
        minus_di = float(row.minus_di) if pd.notna(row.minus_di) else math.nan
        if not all(math.isfinite(value) for value in (adx, plus_di, minus_di)):
            direction = "UNAVAILABLE"
            current_state = STATE_RANGE
            reason = "INDICATOR_UNAVAILABLE"
        elif plus_di == minus_di:
            direction = "TIE"
            current_state = STATE_RANGE
            reason = "DIRECTION_TIE"
        else:
            direction = "UP" if plus_di > minus_di else "DOWN"
            if current_state == STATE_RANGE:
                if adx >= entry_adx and direction == "UP":
                    current_state = STATE_BULL
                    reason = "ADX_ENTRY_UP"
                elif adx >= entry_adx and direction == "DOWN":
                    current_state = STATE_BEAR
                    reason = "ADX_ENTRY_DOWN"
                else:
                    reason = "RANGE_NOT_CONFIRMED"
            elif current_state == STATE_BULL:
                if adx < exit_adx:
                    current_state = STATE_RANGE
                    reason = "BULL_EXIT_ADX"
                elif adx >= entry_adx and direction == "DOWN":
                    current_state = STATE_BEAR
                    reason = "BULL_TO_BEAR_STRONG_DIRECTION"
                else:
                    reason = "BULL_RETAIN_HYSTERESIS"
            elif current_state == STATE_BEAR:
                if adx < exit_adx:
                    current_state = STATE_RANGE
                    reason = "BEAR_EXIT_ADX"
                elif adx >= entry_adx and direction == "UP":
                    current_state = STATE_BULL
                    reason = "BEAR_TO_BULL_STRONG_DIRECTION"
                else:
                    reason = "BEAR_RETAIN_HYSTERESIS"
            else:  # pragma: no cover - 状态只可能来自固定枚举
                raise ValueError(f"未知状态：{current_state}")
        states.append(current_state)
        directions.append(direction)
        changed = previous_state is None or previous_state != current_state
        transition_reasons.append(reason if changed else f"RETAIN::{reason}")
        previous_state = current_state
    result["direction"] = directions
    result["state"] = states
    result["transition_reason"] = transition_reasons
    result["target_position"] = np.where(result["state"] == STATE_BULL, 1.0, 0.0)
    result["state_changed"] = result["state"].ne(result["state"].shift()).astype(bool)
    result["episode_id"] = result["state_changed"].cumsum().astype(int)
    result["episode_row"] = result.groupby("episode_id").cumcount().add(1).astype(int)
    return result


def build_state_panel(index_daily: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    system = config["directional_movement_system"]
    directional = build_directional_movement_panel(
        index_daily, int(system["period_trading_days"])
    )
    return classify_three_states(
        directional,
        float(system["trend_entry_adx"]),
        float(system["trend_exit_adx"]),
    )


def _state_statistics(frame: pd.DataFrame, state: str, column: str) -> dict[str, Any]:
    values = frame.loc[frame["state"] == state, column].dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return {
            "observations": 0,
            "mean": None,
            "median": None,
            "q20": None,
            "positive_rate": None,
        }
    return {
        "observations": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q20": float(np.quantile(values, 0.20)),
        "positive_rate": float(np.mean(values > 0)),
    }


def _ordered_means(period: dict[str, Any]) -> bool:
    values = [period[state]["mean"] for state in (STATE_BULL, STATE_RANGE, STATE_BEAR)]
    return all(value is not None and math.isfinite(float(value)) for value in values) and (
        float(values[0]) > float(values[1]) > float(values[2])
    )


def _ordered_positive_rates(period: dict[str, Any]) -> bool:
    values = [
        period[state]["positive_rate"]
        for state in (STATE_BULL, STATE_RANGE, STATE_BEAR)
    ]
    return all(value is not None and math.isfinite(float(value)) for value in values) and (
        float(values[0]) > float(values[1]) > float(values[2])
    )


def evaluate_state_mechanism(
    state_panel: pd.DataFrame,
    total_return_index: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    start = pd.Timestamp(config["dates"]["evaluation_start"])
    end = pd.Timestamp(config["dates"]["evaluation_end"])
    outcomes = total_return_index[["date", "close"]].rename(
        columns={"close": "total_return_close"}
    )
    merged = state_panel.merge(outcomes, on="date", how="left", validate="one_to_one")
    if merged["total_return_close"].isna().any():
        raise ContractError("状态面板存在缺失的全收益指数收盘值")
    horizons = config["mechanism_evaluation"]["forward_horizons_trading_days"]
    for horizon in horizons:
        merged[f"forward_return_{horizon}d"] = (
            merged["total_return_close"].shift(-int(horizon))
            / merged["total_return_close"]
            - 1.0
        )
    evaluation = merged.loc[merged["date"].between(start, end)].copy()
    primary = int(config["mechanism_evaluation"]["primary_horizon_trading_days"])
    primary_column = f"forward_return_{primary}d"
    period_definitions = {
        "FULL": (start, end),
        "EARLY": tuple(pd.Timestamp(value) for value in config["dates"]["early_period"]),
        "LATE": tuple(pd.Timestamp(value) for value in config["dates"]["late_period"]),
    }
    periods: dict[str, Any] = {}
    for period_name, (period_start, period_end) in period_definitions.items():
        period_frame = evaluation.loc[
            evaluation["date"].between(period_start, period_end)
        ]
        periods[period_name] = {
            state: _state_statistics(period_frame, state, primary_column)
            for state in ALL_STATES
        }
    by_horizon: dict[str, Any] = {}
    for horizon in horizons:
        column = f"forward_return_{horizon}d"
        by_horizon[str(horizon)] = {
            state: _state_statistics(evaluation, state, column) for state in ALL_STATES
        }

    evaluation_states = state_panel.loc[state_panel["date"].between(start, end)].copy()
    state_days = {
        state: int((evaluation_states["state"] == state).sum()) for state in ALL_STATES
    }
    episodes = (
        evaluation_states.groupby("episode_id", as_index=False)
        .agg(
            state=("state", "first"),
            start_date=("date", "min"),
            end_date=("date", "max"),
            rows=("date", "size"),
        )
    )
    episode_counts = {
        state: int((episodes["state"] == state).sum()) for state in ALL_STATES
    }
    median_episode_rows = {
        state: (
            float(episodes.loc[episodes["state"] == state, "rows"].median())
            if episode_counts[state] > 0
            else None
        )
        for state in ALL_STATES
    }
    minimum_days = config["mechanism_evaluation"]["minimum_state_days"]
    minimum_episodes = config["mechanism_evaluation"]["minimum_state_episodes"]
    gates = {
        "minimum_state_days": all(
            state_days[state] >= int(minimum_days[state]) for state in ALL_STATES
        ),
        "minimum_bull_and_bear_episodes": (
            episode_counts[STATE_BULL] >= int(minimum_episodes[STATE_BULL])
            and episode_counts[STATE_BEAR] >= int(minimum_episodes[STATE_BEAR])
        ),
        "full_sample_mean_order": _ordered_means(periods["FULL"]),
        "full_sample_positive_rate_order": _ordered_positive_rates(periods["FULL"]),
        "early_mean_order": _ordered_means(periods["EARLY"]),
        "late_mean_order": _ordered_means(periods["LATE"]),
    }
    passed = all(gates.values())
    transition_frame = evaluation.loc[
        evaluation["state_changed"] & evaluation["state"].isin([STATE_BULL, STATE_BEAR])
    ]
    transition_metrics = {
        state: _state_statistics(transition_frame, state, primary_column)
        for state in (STATE_BULL, STATE_BEAR)
    }
    report = {
        "passed": bool(passed),
        "primary_horizon_trading_days": primary,
        "state_days": state_days,
        "episode_counts": episode_counts,
        "median_episode_rows": median_episode_rows,
        "periods": periods,
        "by_horizon": by_horizon,
        "transition_entries": transition_metrics,
        "gates": gates,
    }
    return report, episodes


def build_targets(state_panel: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    start = pd.Timestamp(config["dates"]["evaluation_start"])
    end = pd.Timestamp(config["dates"]["evaluation_end"])
    targets = state_panel.loc[
        state_panel["date"].between(start, end), ["date", "state", "target_position"]
    ].copy()
    targets["risk_off_override"] = targets["state"].ne(STATE_BULL)
    targets["trade_allowed"] = True
    targets["signal_reason"] = "三态趋势路由::" + targets["state"].astype(str)
    return targets


def _costs(config: dict[str, Any], name: str) -> BacktestCosts:
    values = config["portfolio"][name]
    return BacktestCosts(
        commission_rate=float(values["commission_rate_per_leg"]),
        minimum_commission_cny=float(values["minimum_commission_cny_per_leg"]),
        stamp_duty_rate=float(values["stamp_duty_rate"]),
        slippage_bps=float(values["slippage_bps_per_leg"]),
        lot_size=int(config["portfolio"]["lot_size_shares"]),
        cash_annual_rate=float(config["portfolio"]["cash_annual_rate"]),
    )


def _augment_summary(summary: dict[str, Any], ledger: pd.DataFrame) -> dict[str, Any]:
    result = dict(summary)
    result["total_slippage_cost_cny"] = float(ledger["daily_slippage_cost_cny"].sum())
    result["total_execution_cost_cny"] = float(
        ledger["daily_total_execution_cost_cny"].sum()
    )
    return result


def _period_return_metrics(
    ledger: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> dict[str, Any]:
    frame = ledger.loc[ledger["date"].between(start, end)].copy()
    if frame.empty:
        return {
            "observations": 0,
            "total_return": None,
            "annualized_mean_return": None,
            "annualized_volatility": None,
            "sharpe_zero_cash_rate": None,
            "max_drawdown": None,
        }
    returns = frame["daily_return"].to_numpy(dtype=float)
    total_return = float(np.prod(1.0 + returns) - 1.0)
    annualized_mean = float(np.mean(returns) * 242.0)
    annualized_volatility = (
        float(np.std(returns, ddof=1) * math.sqrt(242.0)) if len(returns) > 1 else 0.0
    )
    sharpe = (
        annualized_mean / annualized_volatility
        if annualized_volatility > 0
        else None
    )
    wealth = np.cumprod(1.0 + returns)
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    return {
        "observations": int(len(frame)),
        "total_return": total_return,
        "annualized_mean_return": annualized_mean,
        "annualized_volatility": annualized_volatility,
        "sharpe_zero_cash_rate": float(sharpe) if sharpe is not None else None,
        "max_drawdown": float(np.min(drawdown)),
    }


def evaluate_portfolio(
    etf_daily: pd.DataFrame,
    dividends: pd.DataFrame,
    targets: pd.DataFrame,
    mechanism_passed: bool,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    start = config["dates"]["evaluation_start"]
    end = config["dates"]["evaluation_end"]
    initial_cash = float(config["portfolio"]["initial_capital_cny"])
    base_costs = _costs(config, "base_costs")
    stress_costs = _costs(config, "stress_costs")
    base_ledger, base_trades = run_long_cash_backtest(
        etf_daily,
        dividends,
        targets,
        initial_cash,
        base_costs,
        start,
        end,
    )
    stress_ledger, stress_trades = run_long_cash_backtest(
        etf_daily,
        dividends,
        targets,
        initial_cash,
        stress_costs,
        start,
        end,
    )
    buy_hold_targets = targets[["date"]].copy()
    buy_hold_targets["target_position"] = 1.0
    buy_hold_targets["risk_off_override"] = False
    buy_hold_targets["trade_allowed"] = True
    buy_hold_targets["signal_reason"] = "买入持有基准"
    buy_hold_base_ledger, buy_hold_base_trades = run_long_cash_backtest(
        etf_daily,
        dividends,
        buy_hold_targets,
        initial_cash,
        base_costs,
        start,
        end,
    )
    buy_hold_stress_ledger, buy_hold_stress_trades = run_long_cash_backtest(
        etf_daily,
        dividends,
        buy_hold_targets,
        initial_cash,
        stress_costs,
        start,
        end,
    )
    base_summary = _augment_summary(
        summarize_backtest(base_ledger, base_trades, initial_cash), base_ledger
    )
    stress_summary = _augment_summary(
        summarize_backtest(stress_ledger, stress_trades, initial_cash), stress_ledger
    )
    buy_hold_base_summary = _augment_summary(
        summarize_backtest(buy_hold_base_ledger, buy_hold_base_trades, initial_cash),
        buy_hold_base_ledger,
    )
    buy_hold_stress_summary = _augment_summary(
        summarize_backtest(
            buy_hold_stress_ledger, buy_hold_stress_trades, initial_cash
        ),
        buy_hold_stress_ledger,
    )
    periods = {
        "EARLY": tuple(pd.Timestamp(value) for value in config["dates"]["early_period"]),
        "LATE": tuple(pd.Timestamp(value) for value in config["dates"]["late_period"]),
    }
    subperiods = {
        name: {
            "base": _period_return_metrics(base_ledger, period_start, period_end),
            "stress": _period_return_metrics(stress_ledger, period_start, period_end),
            "buy_hold_base": _period_return_metrics(
                buy_hold_base_ledger, period_start, period_end
            ),
        }
        for name, (period_start, period_end) in periods.items()
    }
    base_sharpe = base_summary["sharpe_zero_cash_rate"]
    stress_sharpe = stress_summary["sharpe_zero_cash_rate"]
    buy_hold_drawdown = abs(float(buy_hold_base_summary["max_drawdown"]))
    drawdown_ratio = (
        abs(float(base_summary["max_drawdown"])) / buy_hold_drawdown
        if buy_hold_drawdown > 0
        else None
    )
    thresholds = config["portfolio_gates"]
    gates = {
        "mechanism_gate_passed": bool(mechanism_passed),
        "base_net_sharpe_minimum": (
            base_sharpe is not None
            and float(base_sharpe) >= float(thresholds["base_net_sharpe_minimum"])
        ),
        "stress_net_sharpe_minimum": (
            stress_sharpe is not None
            and float(stress_sharpe) >= float(thresholds["stress_net_sharpe_minimum"])
        ),
        "stress_total_return_positive": float(stress_summary["total_return"]) > 0,
        "maximum_drawdown_ratio_vs_buy_hold": (
            drawdown_ratio is not None
            and drawdown_ratio
            <= float(thresholds["maximum_drawdown_ratio_vs_buy_hold_maximum"])
        ),
        "early_period_sharpe_positive": (
            subperiods["EARLY"]["base"]["sharpe_zero_cash_rate"] is not None
            and float(subperiods["EARLY"]["base"]["sharpe_zero_cash_rate"]) > 0
        ),
        "late_period_sharpe_positive": (
            subperiods["LATE"]["base"]["sharpe_zero_cash_rate"] is not None
            and float(subperiods["LATE"]["base"]["sharpe_zero_cash_rate"]) > 0
        ),
    }
    passed = all(gates.values())
    report = {
        "evaluated": True,
        "passed": bool(passed),
        "base": base_summary,
        "stress": stress_summary,
        "buy_hold_base": buy_hold_base_summary,
        "buy_hold_stress": buy_hold_stress_summary,
        "subperiods": subperiods,
        "maximum_drawdown_ratio_vs_buy_hold": drawdown_ratio,
        "gates": gates,
    }
    frames = {
        "base_ledger": base_ledger,
        "base_trades": base_trades,
        "stress_ledger": stress_ledger,
        "stress_trades": stress_trades,
    }
    return report, frames


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def render_markdown(report: dict[str, Any]) -> str:
    mechanism = report["mechanism_evaluation"]
    portfolio = report["portfolio_evaluation"]
    base = portfolio["base"]
    stress = portfolio["stress"]
    states = mechanism["state_days"]
    episodes = mechanism["episode_counts"]
    full = mechanism["periods"]["FULL"]
    lines = [
        "# 510300 三态趋势路由 V1.0.1 结果",
        "",
        f"- 状态：`{report['status']}`",
        f"- 评价区间：{report['evaluation_window']['start']}—{report['evaluation_window']['end']}",
        "- 状态映射：上涨趋势100%，下跌趋势0%，震荡/未确认0%。",
        "- 震荡做T：`NOT_ALLOWED_DATA_AND_T_PLUS_ONE_GATE_FAILED`。",
        "",
        "## 状态识别",
        "",
        "| 状态 | 交易日 | 独立区间 | 未来20日均值 | 正收益率 |",
        "|---|---:|---:|---:|---:|",
    ]
    for state in (STATE_BULL, STATE_RANGE, STATE_BEAR):
        mean = full[state]["mean"]
        positive = full[state]["positive_rate"]
        lines.append(
            f"| `{state}` | {states[state]} | {episodes[state]} | "
            f"{mean:.2%} | {positive:.2%} |"
        )
    lines.extend(
        [
            "",
            f"- 状态机制门：`{'PASS' if mechanism['passed'] else 'FAIL'}`。",
            "",
            "## 组合结果",
            "",
            "| 成本口径 | 净夏普 | 年化收益 | 最大回撤 | 总收益 | 交易次数 |",
            "|---|---:|---:|---:|---:|---:|",
            f"| 基准成本 | {base['sharpe_zero_cash_rate']:.3f} | {base['cagr']:.2%} | "
            f"{base['max_drawdown']:.2%} | {base['total_return']:.2%} | {base['trade_count']} |",
            f"| 压力成本 | {stress['sharpe_zero_cash_rate']:.3f} | {stress['cagr']:.2%} | "
            f"{stress['max_drawdown']:.2%} | {stress['total_return']:.2%} | {stress['trade_count']} |",
            "",
            f"- 买入持有基准成本夏普：{portfolio['buy_hold_base']['sharpe_zero_cash_rate']:.3f}。",
            f"- 最大回撤比例（策略/买入持有）：{portfolio['maximum_drawdown_ratio_vs_buy_hold']:.3f}。",
            f"- 历史净夏普1.2目标：`{'PASS' if report['adjudication']['historical_target_achieved'] else 'FAIL'}`。",
            "- 独立前向目标：`NOT_ACHIEVED`。",
            "",
            "## 边界",
            "",
            "结果出现后不得调整ADX周期、25/20阈值、状态映射、震荡动作或组合其他失败因子。",
            "本结果不生成Paper、Shadow、仓位、订单、券商连接或实盘授权。",
            "",
        ]
    )
    return "\n".join(lines)


def run_study(write: bool = True) -> dict[str, Any]:
    config = load_config()
    manifest = validate_manifest(config)
    etf, index, total_return_index, dividends, input_audit = load_inputs(config)
    state_panel = build_state_panel(index, config)
    mechanism, episodes = evaluate_state_mechanism(
        state_panel, total_return_index, config
    )
    targets = build_targets(state_panel, config)
    portfolio, portfolio_frames = evaluate_portfolio(
        etf, dividends, targets, bool(mechanism["passed"]), config
    )
    historical_pass = bool(portfolio["passed"])
    status = (
        "PASS_HISTORICAL_THREE_STATE_TREND_ROUTER_FORWARD_REQUIRED"
        if historical_pass
        else "REJECTED_FROZEN_THREE_STATE_TREND_ROUTER_TARGET_OR_STATE_GATE_FAILED_NO_RESCUE"
    )
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "project_id": config["protocol"]["project_id"],
        "status": status,
        "evidence_class": config["protocol"]["evidence_class"],
        "scope": {
            "execution_asset": config["scope"]["execution_asset"],
            "signal_asset": config["scope"]["signal_asset"],
            "allowed_holdings": config["scope"]["allowed_holdings"],
            "long_only": True,
            "leverage_allowed": False,
        },
        "evaluation_window": {
            "start": config["dates"]["evaluation_start"],
            "end": config["dates"]["evaluation_end"],
            "trading_days": int(config["data_contract"]["expected_evaluation_rows"]),
        },
        "freeze": {
            "manifest_path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "frozen_at_asia_shanghai": manifest["frozen_at_asia_shanghai"],
            "candidate_2015_plus_outcomes_read_before_freeze": False,
            "candidate_portfolio_returns_read_before_freeze": False,
            "selection_bias_control": manifest["selection_bias_control"],
        },
        "state_definition": {
            "indicator": "WILDER_DIRECTIONAL_MOVEMENT_SYSTEM",
            "period_trading_days": config["directional_movement_system"][
                "period_trading_days"
            ],
            "trend_entry_adx": config["directional_movement_system"][
                "trend_entry_adx"
            ],
            "trend_exit_adx": config["directional_movement_system"][
                "trend_exit_adx"
            ],
            "position_mapping": {
                state: config["states"][state]["target_position"] for state in ALL_STATES
            },
        },
        "range_policy": {
            "historical_policy": "NO_TRADE",
            "range_t_return_evaluation": "NOT_ALLOWED",
            "data_gate_status": input_audit["range_t_data_gate"]["status"],
            "stock_etf_t_plus_one": True,
            "observed_intraday_start": input_audit["range_t_data_gate"][
                "observed_intraday"
            ]["first_date"],
            "required_intraday_start": input_audit["range_t_data_gate"][
                "required_first_date"
            ],
        },
        "mechanism_evaluation": mechanism,
        "portfolio_evaluation": portfolio,
        "adjudication": {
            "return_evaluation": "COMPLETED_FIXED_THREE_STATE_MAPPING",
            "base_net_sharpe": portfolio["base"]["sharpe_zero_cash_rate"],
            "stress_net_sharpe": portfolio["stress"]["sharpe_zero_cash_rate"],
            "target_net_sharpe": config["objective"]["target_net_sharpe"],
            "historical_target_achieved": historical_pass,
            "verified_forward_target_achieved": False,
            "goal_achieved": False,
            "reason": (
                "历史状态门与组合硬门全部通过，但仍需独立前向验证"
                if historical_pass
                else "状态机制门或净组合硬门至少一项失败，冻结协议禁止结果后救援"
            ),
        },
        "boundaries": {
            "parameter_rescue_after_result": "FORBIDDEN",
            "threshold_or_window_rescue_after_result": "FORBIDDEN",
            "state_mapping_or_range_policy_rescue_after_result": "FORBIDDEN",
            "combination_with_rejected_candidates": "FORBIDDEN",
            "paper_signal_allowed": False,
            "shadow_signal_allowed": False,
            "position_mapping_enabled": False,
            "order_generation": False,
            "broker_connection": False,
            "position_change": False,
            "live_trading_authorized": False,
        },
        "artifacts": {},
    }
    if write:
        paths = {key: _project_path(value) for key, value in config["paths"].items()}
        for key in (
            "state_panel",
            "targets",
            "base_ledger",
            "base_trades",
            "stress_ledger",
            "stress_trades",
            "result_json",
            "result_markdown",
        ):
            paths[key].parent.mkdir(parents=True, exist_ok=True)
        state_output = state_panel.loc[
            state_panel["date"].between(
                pd.Timestamp(config["dates"]["evaluation_start"]),
                pd.Timestamp(config["dates"]["evaluation_end"]),
            )
        ].copy()
        state_output.to_parquet(paths["state_panel"], index=False)
        targets.to_parquet(paths["targets"], index=False)
        portfolio_frames["base_ledger"].to_parquet(paths["base_ledger"], index=False)
        portfolio_frames["base_trades"].to_parquet(paths["base_trades"], index=False)
        portfolio_frames["stress_ledger"].to_parquet(paths["stress_ledger"], index=False)
        portfolio_frames["stress_trades"].to_parquet(paths["stress_trades"], index=False)
        artifact_keys = (
            "state_panel",
            "targets",
            "base_ledger",
            "base_trades",
            "stress_ledger",
            "stress_trades",
        )
        report["artifacts"] = {
            key: _artifact_record(paths[key]) for key in artifact_keys
        }
        report["artifacts"]["episodes_not_persisted"] = {
            "rows": int(len(episodes)),
            "reason": "区间汇总已进入结果JSON，不增加重复文件",
        }
        paths["result_json"].write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        paths["result_markdown"].write_text(render_markdown(report), encoding="utf-8")
    return report
